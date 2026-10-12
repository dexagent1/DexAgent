"""lift_clearance_gate: lifted 4 cm clear, lifted only 5 mm, raised while still touching its support, pulled sideways off a
vertical support, already off its support when the window starts, misuse."""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.verifiers.lift_clearance_gate import lift_clearance_gate  # noqa: E402

FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f" {detail}"))
    FAILS += 0 if ok else 1


XML = """<mujoco><worldbody>
<body name="table" pos="0 0 -0.02"><geom type="box" size="0.4 0.4 0.02"/></body>
<body name="cube" pos="0 0 0.015"><freejoint/><geom type="box" size="0.015 0.015 0.015" mass="0.02"/></body>
<body name="wall" pos="0.2 0 0.05"><geom type="box" size="0.005 0.05 0.05"/></body>
<body name="cap" pos="0.22 0 0.08"><freejoint/><geom type="cylinder" size="0.012 0.015" mass="0.01" euler="0 90 0"/></body>
</worldbody></mujoco>"""


def main():
    tmp = Path(tempfile.mkdtemp())
    scene = tmp / "scene.xml"
    scene.write_text(XML)
    m = mujoco.MjModel.from_xml_path(str(scene))
    q0 = m.qpos0.copy()
    ca, pa = (int(m.jnt_qposadr[m.body(b).jntadr[0]]) for b in ("cube", "cap"))
    n = 30

    def save(name, dz=None, cap_dx=None):
        Q = np.repeat(q0[None], n, 0)
        if dz is not None:
            Q[:, ca + 2] = 0.015 + dz
        if cap_dx is not None:
            Q[:, pa] = 0.22 + cap_dx
        p = tmp / f"{name}.npz"
        np.savez(p, qpos=Q, fps=30.0)
        return str(p)

    r1 = lift_clearance_gate(str(scene), save("up", dz=np.linspace(0, 0.04, n)), "cube")
    r2 = lift_clearance_gate(str(scene), save("low", dz=np.linspace(0, 0.005, n)), "cube")
    check("lifted 4 cm passes; lifted 5 mm fails (10 mm needed)", r1["ok"] and (not r2["ok"]) and abs(r2["rise_m"] - 0.005) < 1e-4,
          f"{r1} {r2}")
    r3 = lift_clearance_gate(str(scene), save("touch", dz=np.linspace(0, 0.0006, n)), "cube", min_height_m=0.0005)
    check("raised 0.6 mm, past a 0.5 mm height but still within contact of the table, fails", (not r3["ok"]) and r3["touches_support"], str(r3))
    # a cap on a wall (lying against it, a bottle on its side): pulled 3 cm away sideways, no vertical rise
    side = save("side", cap_dx=np.linspace(-0.0029, 0.03, n))
    r4 = lift_clearance_gate(str(scene), side, "cap", support="wall")
    r5 = lift_clearance_gate(str(scene), save("side_short", cap_dx=np.linspace(-0.0029, 0.004, n)), "cap", support="wall")
    check("a cap pulled 3 cm sideways off its support is clear; 7 mm is not", r4["ok"] and (not r5["ok"]), f"{r4} {r5}")
    # the window starts after the cube already left the table: measured from the last frame it rested
    Q = np.load(save("late", dz=np.linspace(0, 0.04, n)))["qpos"]
    np.savez(tmp / "late.npz", qpos=Q, fps=30.0)
    r6 = lift_clearance_gate(str(scene), str(tmp / "late.npz"), "cube", frames=[20, 29])
    check("a window starting mid-lift is measured from the last frame the cube rested on the table",
          r6["ok"] and r6["reference_frame"] < 20 and r6["rise_m"] > 0.03, str(r6))
    raised = []
    for kw in ({"at": "peak"}, {"min_height_m": -1}, {"support": "nope"}):
        try:
            lift_clearance_gate(str(scene), str(tmp / "late.npz"), "cube", **kw)
        except ValueError as e:
            raised.append(str(e).split(":")[0])
    check("a bad 'at', a negative height and an unknown support raise", raised == ["at", "min_height_m", "support"], str(raised))
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
