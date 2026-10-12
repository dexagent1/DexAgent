"""rotation_turns_gate: a knob turned 400 deg (wraps counted) vs 300 deg, a free body turned 2.25 turns about its axis, a
back-and-forth twist that cancels, the direction, a thread that runs out vs stops short, a turn relative to a turning base,
misuse."""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

from tool_library.verifiers.rotation_turns_gate import rotation_turns_gate  # noqa: E402

FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f" {detail}"))
    FAILS += 0 if ok else 1


XML = """<mujoco><worldbody>
<body name="bottle" pos="0 0 0"><geom type="cylinder" size="0.03 0.05" pos="0 0 0.05"/>
  <body name="carrier" pos="0 0 0.1"><joint name="yaw" type="hinge" axis="0 0 1"/><joint name="lift" type="slide" axis="0 0 1" range="-0.002 0.02"/>
    <geom type="cylinder" size="0.02 0.01"/></body></body>
<body name="box" pos="0.3 0 0.03"><freejoint/><geom type="box" size="0.05 0.03 0.03" mass="0.1"/></body>
<body name="tray" pos="-0.3 0 0.01"><freejoint/><geom type="box" size="0.08 0.08 0.01" mass="0.1"/></body>
</worldbody></mujoco>"""


def quat(R):
    w = R.as_quat()
    return [w[3], w[0], w[1], w[2]]


def main():
    tmp = Path(tempfile.mkdtemp())
    scene = tmp / "scene.xml"
    scene.write_text(XML)
    m = mujoco.MjModel.from_xml_path(str(scene))
    q0 = m.qpos0.copy()
    ya, la = int(m.jnt_qposadr[m.joint("yaw").id]), int(m.jnt_qposadr[m.joint("lift").id])
    ba, ta = (int(m.jnt_qposadr[m.body(b).jntadr[0]]) for b in ("box", "tray"))

    def save(name, yaw=None, lift=None, box_yaw=None, tray_yaw=None, n=90):
        Q = np.repeat(q0[None], n, 0)
        if yaw is not None:
            Q[:, ya] = np.radians(yaw)
        if lift is not None:
            Q[:, la] = lift
        for a, ang in ((ba, box_yaw), (ta, tray_yaw)):
            if ang is not None:
                for t in range(n):
                    Q[t, a + 3:a + 7] = quat(Rot.from_euler("z", ang[t], degrees=True))
        p = tmp / f"{name}.npz"
        np.savez(p, qpos=Q, fps=30.0)
        return str(p)

    k400, k300 = save("k400", yaw=np.linspace(0, 400, 90)), save("k300", yaw=np.linspace(0, 300, 90))
    r1 = rotation_turns_gate(str(scene), k400, 360, joint="yaw")
    r2 = rotation_turns_gate(str(scene), k300, 360, joint="yaw")
    check("a knob turned 400 deg passes a full turn; 300 deg fails", r1["ok"] and (not r2["ok"]) and abs(r2["total_deg"] - 300) < 0.5, f"{r1} {r2}")
    spin = save("spin", box_yaw=np.linspace(0, 810, 90))
    r3 = rotation_turns_gate(str(scene), spin, 720, body="box", axis=[0, 0, 1])
    twist = save("twist", box_yaw=np.r_[np.linspace(0, 170, 45), np.linspace(170, 0, 45)])
    r4 = rotation_turns_gate(str(scene), twist, 90, body="box")
    check("a body turned 2.25 turns integrates past 180 deg; a twist out and back cancels",
          r3["ok"] and abs(r3["total_deg"] - 810) < 1 and (not r4["ok"]) and r4["peak_deg"] > 160, f"{r3} {r4}")
    ccw = save("ccw", box_yaw=np.linspace(0, -100, 90))
    r5 = rotation_turns_gate(str(scene), ccw, 90, body="box", direction=1)
    r6 = rotation_turns_gate(str(scene), ccw, 90, body="box", direction=-1)
    check("the direction is enforced when given", (not r5["ok"]) and r6["ok"], f"{r5} {r6}")
    out = save("out", yaw=np.linspace(0, 1440, 90), lift=np.linspace(0, 0.0192, 90))
    short = save("short", yaw=np.linspace(0, 1440, 90), lift=np.linspace(0, 0.012, 90))
    r7 = rotation_turns_gate(str(scene), out, 720, joint="yaw", release_joint="lift", release_travel_m=0.0192)
    r8 = rotation_turns_gate(str(scene), short, 720, joint="yaw", release_joint="lift", release_travel_m=0.0192)
    check("a thread that ran out passes; four turns that stop short of the release fail", r7["ok"] and (not r8["ok"]) and not r8["released"],
          f"{r7} {r8}")
    both = save("both", box_yaw=np.linspace(0, 120, 90), tray_yaw=np.linspace(0, 100, 90))
    r9 = rotation_turns_gate(str(scene), both, 90, body="box", relative_to="tray")
    check("relative to a base that turned with it, a 120 deg turn is only 20 deg", (not r9["ok"]) and abs(r9["total_deg"] - 20) < 1, str(r9))
    raised = []
    for kw in ({}, {"joint": "yaw", "body": "box"}, {"joint": "lift"}, {"joint": "yaw", "direction": 2},
               {"joint": "yaw", "release_joint": "lift"}):
        try:
            rotation_turns_gate(str(scene), k400, 90, **kw)
        except ValueError as e:
            raised.append(str(e).split(":")[0])
    check("no part, two parts, a slide joint, a bad direction and a release joint without travel raise",
          raised == ["joint / body", "joint / body", "joint", "direction", "release_joint / release_travel_m"], str(raised))
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
