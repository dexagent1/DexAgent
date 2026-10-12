"""tilt_pour_gate: a bottle tipped over a mug for a second (pour), the same tilt beside the mug, a tilt that stays too
shallow, a pour too short, a kept tilt that returns upright before the end, a table region as the target, misuse."""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

from tool_library.verifiers.tilt_pour_gate import tilt_pour_gate  # noqa: E402

FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f" {detail}"))
    FAILS += 0 if ok else 1


# a 20 cm bottle (origin at its base) and a mug 15 cm to its right
XML = """<mujoco><worldbody>
<body name="table" pos="0 0 -0.02"><geom type="box" size="0.4 0.4 0.02"/></body>
<body name="bottle" pos="0 0 0"><freejoint/><geom type="cylinder" size="0.03 0.1" pos="0 0 0.1" mass="0.3"/></body>
<body name="mug" pos="0.15 0 0"><freejoint/><geom type="cylinder" size="0.04 0.045" pos="0 0 0.045" mass="0.2"/></body>
</worldbody></mujoco>"""


def main():
    tmp = Path(tempfile.mkdtemp())
    scene = tmp / "scene.xml"
    scene.write_text(XML)
    m = mujoco.MjModel.from_xml_path(str(scene))
    q0 = m.qpos0.copy()
    ba = int(m.jnt_qposadr[m.body("bottle").jntadr[0]])

    def save(name, tilts, base=(0.0, 0.0, 0.12), n=None):
        """The bottle lifted to `base` and tipped about y by tilts[t] deg (the mouth swings toward +x)."""
        Q = np.repeat(q0[None], len(tilts), 0)
        for t, a in enumerate(tilts):
            R = Rot.from_euler("y", a, degrees=True)
            Q[t, ba:ba + 3] = base
            w = R.as_quat()
            Q[t, ba + 3:ba + 7] = [w[3], w[0], w[1], w[2]]
        p = tmp / f"{name}.npz"
        np.savez(p, qpos=Q, fps=30.0)
        return str(p)

    pour = np.r_[np.linspace(0, 110, 30), np.full(30, 110), np.linspace(110, 0, 30)]
    good = save("good", pour)
    r1 = tilt_pour_gate(str(scene), good, "bottle", target="mug", min_tilt_deg=90)
    beside = save("beside", pour, base=(0.0, 0.25, 0.12))
    r2 = tilt_pour_gate(str(scene), beside, "bottle", target="mug", min_tilt_deg=90)
    check("tipped 110 deg with the mouth over the mug for a second pours; beside the mug it never pours",
          r1["ok"] and r1["longest_pour_s"] >= 0.5 and (not r2["ok"]) and "never over" in r2["issues"][0], f"{r1} {r2}")
    shallow = save("shallow", pour * 0.5)
    r3 = tilt_pour_gate(str(scene), shallow, "bottle", target="mug", min_tilt_deg=90)
    quick = save("quick", np.r_[np.linspace(0, 110, 30), np.linspace(110, 0, 30)])
    r4 = tilt_pour_gate(str(scene), quick, "bottle", target="mug", min_tilt_deg=90, min_time_s=0.5)
    check("a 55 deg tilt is too shallow; a tip that comes straight back is too short",
          (not r3["ok"]) and "at most" in r3["issues"][0] and (not r4["ok"]) and "only" in r4["issues"][0], f"{r3} {r4}")
    horizontal = save("horizontal", np.r_[np.linspace(0, 90, 30), np.full(20, 90)])
    back = save("back", np.r_[np.linspace(0, 90, 30), np.full(10, 90), np.linspace(90, 0, 10)])
    r5 = tilt_pour_gate(str(scene), horizontal, "bottle", min_tilt_deg=70, min_time_s=0.2, at_end=True)
    r6 = tilt_pour_gate(str(scene), back, "bottle", min_tilt_deg=70, min_time_s=0.2, at_end=True)
    check("a tilt kept to the end passes; one returned upright before the end fails", r5["ok"] and not r6["ok"], f"{r5} {r6}")
    r7 = tilt_pour_gate(str(scene), good, "bottle", region={"box": [[0.1, -0.05], [0.25, 0.05]]}, min_tilt_deg=90)
    check("a table region works as the target", r7["ok"], str(r7))
    raised = []
    for kw in ({"target": "mug", "region": {"box": [[0, 0], [1, 1]]}}, {"min_tilt_deg": 0}, {"target": "cup"}):
        try:
            tilt_pour_gate(str(scene), good, "bottle", **kw)
        except ValueError as e:
            raised.append(str(e).split(":")[0])
    check("a target and a region together, a zero tilt and an unknown target raise", raised == ["target / region", "min_tilt_deg", "target"], str(raised))
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
