"""press_displacement_gate: a pump pressed through its stroke twice vs once, a press too shallow, a button that springs
back (peak vs end), a brick pressed onto its base measured relative to the base while both move, misuse."""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.verifiers.press_displacement_gate import press_displacement_gate  # noqa: E402

FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f" {detail}"))
    FAILS += 0 if ok else 1


XML = """<mujoco><worldbody>
<body name="table" pos="0 0 -0.02"><geom type="box" size="0.4 0.4 0.02"/></body>
<body name="spray" pos="0 0 0.06"><freejoint/><geom type="cylinder" size="0.025 0.06" mass="0.2"/>
  <body name="head" pos="0 0 0.07"><joint name="pump" type="slide" axis="0 0 -1" range="0 0.012"/><geom type="box" size="0.01 0.01 0.01" mass="0.01"/></body>
</body>
<body name="base" pos="0.2 0 0.02"><freejoint/><geom type="box" size="0.02 0.02 0.02" mass="0.05"/></body>
<body name="brick" pos="0.2 0 0.062"><freejoint/><geom type="box" size="0.02 0.02 0.02" mass="0.02"/></body>
</worldbody></mujoco>"""


def main():
    tmp = Path(tempfile.mkdtemp())
    scene = tmp / "scene.xml"
    scene.write_text(XML)
    m = mujoco.MjModel.from_xml_path(str(scene))
    q0 = m.qpos0.copy()
    pa = int(m.jnt_qposadr[m.joint("pump").id])
    ba, ka = (int(m.jnt_qposadr[m.body(b).jntadr[0]]) for b in ("base", "brick"))

    def stroke(depth, n=20):
        return np.r_[np.linspace(0, depth, n // 2), np.linspace(depth, 0, n - n // 2)]

    def save(name, pump=None, base_dx=None, brick_dz=None):
        n = len(pump) if pump is not None else len(base_dx)
        Q = np.repeat(q0[None], n, 0)
        if pump is not None:
            Q[:, pa] = pump
        if base_dx is not None:
            Q[:, ba] += base_dx
            Q[:, ka] += base_dx
            Q[:, ka + 2] += brick_dz
        p = tmp / f"{name}.npz"
        np.savez(p, qpos=Q, fps=30.0)
        return str(p)

    twice = save("twice", np.r_[stroke(0.011), np.zeros(5), stroke(0.011)])
    once = save("once", np.r_[stroke(0.011), np.zeros(25)])
    r1 = press_displacement_gate(str(scene), twice, 0.0096, joint="pump", min_strokes=2)
    r2 = press_displacement_gate(str(scene), once, 0.0096, joint="pump", min_strokes=2)
    check("two full pump strokes pass 'several times'; a single stroke fails with the count",
          r1["ok"] and r1["strokes"] == 2 and (not r2["ok"]) and r2["strokes"] == 1, f"{r1} {r2}")
    shallow = save("shallow", np.r_[stroke(0.004), stroke(0.004)])
    r3 = press_displacement_gate(str(scene), shallow, 0.0096, joint="pump")
    check("a press of 4 mm where the stroke is 12 mm fails", (not r3["ok"]) and abs(r3["displacement_m"] - 0.004) < 1e-4, str(r3))
    r4 = press_displacement_gate(str(scene), once, 0.0096, joint="pump", peak=True)
    r5 = press_displacement_gate(str(scene), once, 0.0096, joint="pump", peak=False)
    check("a button that springs back passes on its peak and fails when it must stay pressed", r4["ok"] and not r5["ok"], f"{r4} {r5}")
    n = 30
    dx = np.linspace(0, 0.05, n)
    seat = save("seat", base_dx=dx, brick_dz=-np.linspace(0, 0.002, n))
    float_ = save("float", base_dx=dx, brick_dz=-np.linspace(0, 0.0004, n))
    r6 = press_displacement_gate(str(scene), seat, 0.0015, body="brick", relative_to="base", peak=False)
    r7 = press_displacement_gate(str(scene), float_, 0.0015, body="brick", relative_to="base", peak=False)
    check("a brick seated 2 mm onto a base that slid 5 cm passes; 0.4 mm fails", r6["ok"] and not r7["ok"], f"{r6} {r7}")
    raised = []
    for kw in ({}, {"joint": "pump", "body": "brick"}, {"joint": "pump", "min_strokes": 0}, {"body": "brick", "direction": [0, 1]}):
        try:
            press_displacement_gate(str(scene), twice, 0.01, **kw)
        except ValueError as e:
            raised.append(str(e).split(":")[0])
    check("no part, two parts, zero strokes and a short direction raise",
          raised == ["joint / body", "joint / body", "min_strokes", "direction"], str(raised))
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
