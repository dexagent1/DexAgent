"""release_settle_gate: set down and let go, dropped from 8 cm, sliding away after the release, a brick seating itself
on release, a part rocking in place, released too late to judge, upright / upside down, a named support, misuse."""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

from tool_library.verifiers.release_settle_gate import release_settle_gate  # noqa: E402

FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f" {detail}"))
    FAILS += 0 if ok else 1


XML = """<mujoco><worldbody>
<body name="table" pos="0 0 -0.02"><geom type="box" size="0.4 0.4 0.02"/></body>
<body name="plate" pos="0.2 0 0.005"><freejoint/><geom type="box" size="0.05 0.05 0.005" mass="0.05"/></body>
<body name="cube" pos="0 0 0.015"><freejoint/><geom type="box" size="0.015 0.015 0.015" mass="0.02"/></body>
<body name="robot_left_pad_a" pos="-0.0198 0 0.015"><freejoint/><geom type="box" size="0.005 0.01 0.01" mass="0.01"/></body>
<body name="robot_left_pad_b" pos="0.0198 0 0.015"><freejoint/><geom type="box" size="0.005 0.01 0.01" mass="0.01"/></body>
</worldbody></mujoco>"""


def quat(e):
    w = Rot.from_euler("xyz", e, degrees=True).as_quat()
    return [w[3], w[0], w[1], w[2]]


def main():
    tmp = Path(tempfile.mkdtemp())
    scene = tmp / "scene.xml"
    scene.write_text(XML)
    m = mujoco.MjModel.from_xml_path(str(scene))
    q0 = m.qpos0.copy()
    A = {b: int(m.jnt_qposadr[m.body(b).jntadr[0]]) for b in ("cube", "robot_left_pad_a", "robot_left_pad_b")}
    n = 60

    def save(name, cube_xyz, cube_euler=None, open_at=20, open_gap=0.04):
        Q = np.repeat(q0[None], n, 0)
        for t in range(n):
            c = cube_xyz[t]
            Q[t, A["cube"]:A["cube"] + 3] = c
            if cube_euler is not None:
                Q[t, A["cube"] + 3:A["cube"] + 7] = quat(cube_euler[t])
            g = 0.0198 if t < open_at else open_gap
            hold_c = cube_xyz[min(t, open_at - 1)]
            Q[t, A["robot_left_pad_a"]:A["robot_left_pad_a"] + 3] = [hold_c[0] - g, hold_c[1], hold_c[2]]
            Q[t, A["robot_left_pad_b"]:A["robot_left_pad_b"] + 3] = [hold_c[0] + g, hold_c[1], hold_c[2]]
        p = tmp / f"{name}.npz"
        np.savez(p, qpos=Q, fps=30.0)
        return str(p)

    rest = np.tile([0.0, 0.0, 0.015], (n, 1))
    r1 = release_settle_gate(str(scene), save("rest", rest), "cube", on="table", orientation="upright")
    check("set down and let go: stays, supported by the table, upright", r1["ok"] and r1["release_frame"] == 19, str(r1))
    fall = rest.copy()
    fall[:20, 2] = 0.095
    fall[20:28, 2] = np.linspace(0.095, 0.015, 8)
    r2 = release_settle_gate(str(scene), save("drop", fall), "cube")
    check("let go 8 cm above the table: it falls", (not r2["ok"]) and "falls" in r2["issues"][0], str(r2))
    slide = rest.copy()
    slide[20:, 0] = np.linspace(0.0, 0.06, n - 20)
    r3 = release_settle_gate(str(scene), save("slide", slide), "cube")
    check("sliding 6 cm away after the release fails", not r3["ok"], str(r3))
    seat = rest.copy()
    seat[:20, 2] = 0.022
    seat[20:24, 2] = np.linspace(0.022, 0.015, 4)
    eul = np.zeros((n, 3))
    eul[:20, 1] = 12.0
    eul[20:24, 1] = np.linspace(12, 0, 4)
    r4 = release_settle_gate(str(scene), save("seat", seat, eul), "cube")
    rock = rest.copy()
    eul2 = np.zeros((n, 3))
    eul2[20:, 0] = 4.0 * np.sin(np.arange(n - 20) * 0.6)
    r5 = release_settle_gate(str(scene), save("rock", rock, eul2), "cube")
    check("a cube seating itself 7 mm / 12 deg right after the release, and one rocking 4 deg in place, both pass",
          r4["ok"] and r5["ok"], f"{r4} {r5}")
    late = save("late", rest, open_at=n - 2)
    r6 = release_settle_gate(str(scene), late, "cube")
    check("let go two frames before the end: too little time to show it stays", (not r6["ok"]) and "rest at least" in r6["issues"][0], str(r6))
    flip = save("flip", rest, np.tile([180.0, 0, 0], (n, 1)))
    r7 = release_settle_gate(str(scene), flip, "cube", orientation="upside_down", up_local=[0, 0, 1])
    r8 = release_settle_gate(str(scene), flip, "cube", orientation="upright", up_local=[0, 0, 1])
    r9 = release_settle_gate(str(scene), save("rest2", rest), "cube", on="plate")
    check("upside down judged against a body axis; resting on the table is not resting on the plate",
          r7["ok"] and (not r8["ok"]) and not r9["ok"], f"{r7} {r8} {r9}")
    raised = []
    for kw in ({"orientation": "sideways"}, {"settle_s": 0}, {"on": "nowhere"}, {"release_frame": 999}):
        try:
            release_settle_gate(str(scene), save("rest3", rest), "cube", **kw)
        except ValueError as e:
            raised.append(str(e).split(":")[0])
    check("a bad orientation, settle time, support and release frame raise", raised == ["orientation", "settle_s", "on", "release_frame"], str(raised))
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
