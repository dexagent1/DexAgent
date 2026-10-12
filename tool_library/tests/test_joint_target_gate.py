"""joint_target_gate: a drawer slid open far enough and not far enough, a lid closed and held, a latch that springs back,
an explicit target and a fraction of the range, misuse."""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.verifiers.joint_target_gate import joint_target_gate  # noqa: E402

FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f" {detail}"))
    FAILS += 0 if ok else 1


XML = """<mujoco><worldbody>
<body name="carcass" pos="0 0 0.05"><geom type="box" size="0.1 0.1 0.05"/>
  <body name="drawer" pos="0 0 0"><joint name="slide" type="slide" axis="0 -1 0" range="0 0.2"/><geom type="box" size="0.08 0.08 0.02"/></body>
</body>
<body name="box" pos="0.3 0 0.03"><geom type="box" size="0.05 0.05 0.03"/>
  <body name="lid" pos="0 0.05 0.03"><joint name="lid_hinge" type="hinge" axis="1 0 0" range="0 3.0"/><geom type="box" size="0.05 0.05 0.003" pos="0 -0.05 0"/></body>
</body>
<body name="knob" pos="-0.3 0 0.02"><joint name="free_hinge" type="hinge" axis="0 0 1"/><geom type="cylinder" size="0.01 0.01"/></body>
</worldbody></mujoco>"""


def main():
    tmp = Path(tempfile.mkdtemp())
    scene = tmp / "scene.xml"
    scene.write_text(XML)
    m = mujoco.MjModel.from_xml_path(str(scene))
    sa, la = int(m.jnt_qposadr[m.joint("slide").id]), int(m.jnt_qposadr[m.joint("lid_hinge").id])

    def save(name, slide=None, lid=None, n=40):
        Q = np.zeros((n, m.nq))
        if slide is not None:
            Q[:, sa] = slide
        if lid is not None:
            Q[:, la] = lid
        p = tmp / f"{name}.npz"
        np.savez(p, qpos=Q, fps=30.0)
        return str(p)

    far = save("far", slide=np.linspace(0, 0.15, 40))
    near = save("near", slide=np.linspace(0, 0.06, 40))
    r1 = joint_target_gate(str(scene), far, "slide", state="open")
    r2 = joint_target_gate(str(scene), near, "slide", state="open")
    check("a drawer slid out 15 of 20 cm is open; 6 cm is not (closest reported)",
          r1["ok"] and (not r2["ok"]) and abs(r2["closest"] - 0.06) < 1e-6, f"{r1} {r2}")
    lid = np.r_[np.linspace(1.6, 0.0, 30), np.zeros(10)]
    closed = save("closed", lid=lid)
    r3 = joint_target_gate(str(scene), closed, "lid_hinge", state="closed")
    spring = save("spring", lid=np.r_[np.linspace(1.6, 0.0, 30), np.linspace(0.0, 0.5, 10)])
    r4 = joint_target_gate(str(scene), spring, "lid_hinge", state="closed")
    r5 = joint_target_gate(str(scene), spring, "lid_hinge", state="closed", hold=False)
    check("a lid closed and kept passes; one that reopens fails naming the frame, unless hold is off",
          r3["ok"] and (not r4["ok"]) and r4["lost_frame"] is not None and r5["ok"], f"{r3} {r4} {r5}")
    r6 = joint_target_gate(str(scene), far, "slide", target=0.15, tol=0.005)
    r7 = joint_target_gate(str(scene), far, "slide", fraction=0.9, tol=0.005)
    check("an explicit target is met; 90% of the range (0.18 m) is not", r6["ok"] and not r7["ok"], f"{r6} {r7}")
    raised = []
    for kw in ({"state": "ajar"}, {"state": "open", "target": 0.1}, {}, {"joint": "nope", "state": "open"},
               {"joint": "free_hinge", "state": "open"}):
        try:
            joint_target_gate(str(scene), far, kw.pop("joint", "slide"), **kw)
        except ValueError as e:
            raised.append(str(e).split(":")[0])
    check("a bad state, two targets, no target, an unknown joint and an unlimited joint with a state raise",
          raised == ["state", "target / fraction / state", "target / fraction / state", "joint", "state"], str(raised))
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
