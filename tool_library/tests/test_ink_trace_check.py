"""Variations for ink_trace_check: a contact stroke inks its region, an air stroke does not, a back-and-forth pad wipe
clears a written mark while an untouched mark stays, a yawed surface under a prefix, and misuse is refused."""
import os, sys, tempfile
import numpy as np
from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.verifiers.ink_trace_check import ink_trace_check as T
import mujoco


def world(yaw=0.0, pre=""):
    return f"""<mujoco><asset><texture name="{pre}ink" type="2d" builtin="flat" width="400" height="300" rgb1="0.98 0.98 0.97"/>
 <material name="m" texture="{pre}ink"/></asset><worldbody>
 <body name="{pre}board" euler="0 0 {yaw}"><geom name="{pre}panel" type="box" size="0.2 0.15 0.002" pos="0 0 0.002" material="m"/></body>
 <body name="pen"><freejoint/><geom type="sphere" size="0.003" mass="0.01"/><site name="pen_tip" pos="0 0 0"/></body>
 <body name="eraser"><freejoint/><geom name="eraser_pad" type="box" size="0.03 0.012 0.004" mass="0.03"/></body>
</worldbody></mujoco>"""


def traj(m, pen_path, pad_path):
    q = np.zeros((len(pen_path), m.nq))
    for i, (p, e) in enumerate(zip(pen_path, pad_path)):
        q[i, 0:3] = p; q[i, 3] = 1.0; q[i, 7:10] = e; q[i, 10] = 1.0
    return q


def run():
    fails = 0
    m = mujoco.MjModel.from_xml_path(write(world()))
    top = 0.004
    line = [(x, 0.05, top - 0.0005) for x in np.linspace(-0.1, 0.1, 40)]
    park = [(0.0, 0.3, 0.1)] * 40
    # 1. a contact stroke inks its region and leaves the rest clean
    r = T(m, traj(m, line, park), "panel", "ink", nibs=["pen_tip"],
          regions=[{"name": "stroke", "rect": [-0.08, 0.08, 0.045, 0.055], "expect": "inked"},
                   {"name": "below", "rect": [-0.08, 0.08, -0.06, -0.04], "expect": "clean"}])
    fails += report("contact stroke inks", r["ok"] and r["regions"]["stroke"]["ink"] > 0.2 and r["nibs"]["pen_tip"]["contact_frames"] == 40, r)
    # 2. the same stroke 4 mm in the air draws nothing: the inked expectation fails with an actionable issue
    air = [(x, y, z + 0.004) for x, y, z in line]
    r = T(m, traj(m, air, park), "panel", "ink", nibs=["pen_tip"], regions=[{"name": "stroke", "rect": [-0.08, 0.08, 0.045, 0.055], "expect": "inked"}])
    fails += report("air stroke does not ink", (not r["ok"]) and r["regions"]["stroke"]["ink"] == 0.0 and "did not touch" in r["issues"][0], r)
    # 3. write two marks, then wipe one back and forth: it ends clean, the other stays inked, judged at two frames
    a = [(x, 0.05, top - 0.0005) for x in np.linspace(-0.12, -0.06, 20)]
    b = [(x, 0.05, top - 0.0005) for x in np.linspace(0.06, 0.12, 20)]
    wipe = [(0.09 + 0.03 * np.sin(k / 3.0), 0.05, top + 0.004 - 0.0002) for k in range(120)]
    pen = a + [(0.0, 0.0, 0.05)] + b + [(0.0, 0.3, 0.1)] * 120
    pad = [(0.0, 0.3, 0.1)] * 41 + [(0.0, 0.3, 0.1)] * 0 + wipe
    pad = pad[:len(pen)] + [(0.0, 0.3, 0.1)] * (len(pen) - len(pad))
    regs = [{"name": "b_written", "rect": [0.065, 0.115, 0.045, 0.055], "expect": "inked", "at_frame": 40},
            {"name": "b_wiped", "rect": [0.065, 0.115, 0.045, 0.055], "expect": "clean"},
            {"name": "a_kept", "rect": [-0.115, -0.065, 0.045, 0.055], "expect": "inked"}]
    r = T(m, traj(m, pen, pad), "panel", "ink", nibs=["pen_tip"], pads=["eraser_pad"], regions=regs)
    fails += report("wipe clears only what it covers", r["ok"] and r["pads"]["eraser_pad"]["contact_frames"] > 50, r)
    # 4. a lift-and-move is not joined: the gap between two dabs stays clean
    dabs = [(-0.1, 0.0, top - 0.0005), (0.1, 0.0, top - 0.0005)]
    r = T(m, traj(m, dabs, park[:2]), "panel", "ink", nibs=["pen_tip"], regions=[{"name": "gap", "rect": [-0.05, 0.05, -0.005, 0.005], "expect": "clean"}])
    fails += report("jump not joined", r["ok"], r)
    # 5. a yawed board attached under a prefix: suffix names resolve and world rects follow the yaw
    m2 = mujoco.MjModel.from_xml_path(write(world(yaw=30, pre="b_")))
    c, s = np.cos(np.radians(30)), np.sin(np.radians(30))
    line2 = [(c * x, s * x, top - 0.0005) for x in np.linspace(-0.05, 0.05, 30)]
    r = T(m2, traj(m2, line2, park[:30]), "panel", "ink", regions=[{"name": "diag", "rect": [-0.03, 0.03, -0.017, 0.017], "expect": "inked", "min_ink": 0.01}])
    fails += report("yawed prefixed surface", r["ok"] and r["surface"]["geom"] == "b_panel", r)
    # 6. misuse is refused
    try:
        T(m, traj(m, line, park), "panel", "ink", colour="red")
        fails += report("unknown parameter refused", False, {})
    except ValueError:
        fails += report("unknown parameter refused", True, {})
    try:
        T(m, np.zeros((3, m.nq + 1)), "panel", "ink")
        fails += report("wrong qpos width refused", False, {})
    except ValueError:
        fails += report("wrong qpos width refused", True, {})
    return fails


_tmp = tempfile.mkdtemp()


def write(xml):
    p = os.path.join(_tmp, f"w{len(os.listdir(_tmp))}.xml"); open(p, "w").write(xml); return p


def report(name, ok, r):
    print(("[PASS] " if ok else "[FAIL] ") + name + ("" if ok else f" :: {str(r)[:600]}"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
