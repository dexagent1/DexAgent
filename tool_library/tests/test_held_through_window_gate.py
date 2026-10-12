"""held_through_window_gate: a cube pinched and carried, a pinch that opens mid-carry (drop), one pad pushing (no
opposition), the other hand holding it, a steadying hold that keeps or loses the object, misuse."""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.verifiers.held_through_window_gate import held_through_window_gate  # noqa: E402

FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f" {detail}"))
    FAILS += 0 if ok else 1


# two fingertip pads per hand (free bodies driven kinematically by the trajectory) and a 3 cm cube
XML = """<mujoco><worldbody>
<body name="table" pos="0 0 -0.02"><geom type="box" size="0.4 0.4 0.02"/></body>
<body name="cube" pos="0 0 0.015"><freejoint/><geom type="box" size="0.015 0.015 0.015" mass="0.02"/></body>
<body name="robot_left_pad_a" pos="-0.0198 0 0.015"><freejoint/><geom type="box" size="0.005 0.01 0.01" mass="0.01"/></body>
<body name="robot_left_pad_b" pos="0.0198 0 0.015"><freejoint/><geom type="box" size="0.005 0.01 0.01" mass="0.01"/></body>
<body name="robot_right_pad_a" pos="0.3 0.2 0.05"><freejoint/><geom type="box" size="0.005 0.01 0.01" mass="0.01"/></body>
<body name="robot_right_pad_b" pos="0.3 0.25 0.05"><freejoint/><geom type="box" size="0.005 0.01 0.01" mass="0.01"/></body>
</worldbody></mujoco>"""


def main():
    tmp = Path(tempfile.mkdtemp())
    scene = tmp / "scene.xml"
    scene.write_text(XML)
    m = mujoco.MjModel.from_xml_path(str(scene))
    q0 = m.qpos0.copy()
    adr = {b: int(m.jnt_qposadr[m.body(b).jntadr[0]]) for b in ("cube", "robot_left_pad_a", "robot_left_pad_b", "robot_right_pad_a", "robot_right_pad_b")}
    n = 40
    lift = np.r_[np.zeros(5), np.linspace(0, 0.06, 15), np.full(20, 0.06)]

    def save(name, left_gap=None, cube_dz=None, right=False, one_pad=False):
        Q = np.repeat(q0[None], n, 0)
        dz = lift if cube_dz is None else cube_dz
        for t in range(n):
            Q[t, adr["cube"] + 2] = 0.015 + dz[t]
            g = 0.0198 if left_gap is None else left_gap[t]
            side = "right" if right else "left"
            Q[t, adr[f"robot_{side}_pad_a"]:adr[f"robot_{side}_pad_a"] + 3] = [-g, 0, 0.015 + lift[t]]
            Q[t, adr[f"robot_{side}_pad_b"]:adr[f"robot_{side}_pad_b"] + 3] = [0.3 if one_pad else g, 0 if not one_pad else 0.3, 0.015 + lift[t]]
        p = tmp / f"{name}.npz"
        np.savez(p, qpos=Q, fps=30.0)
        return str(p)

    carry = save("carry")
    r1 = held_through_window_gate(str(scene), carry, "cube", side="left", frames=[20, 39])
    check("a cube pinched by two opposing pads and carried is held through the window", r1["ok"] and r1["held_fraction"] == 1.0, str(r1))
    gap = np.full(n, 0.0198)
    gap[28:] = 0.03
    dz = lift.copy()
    dz[28:] = np.maximum(0.0, 0.06 - 0.01 * np.arange(1, 13))
    drop = save("drop", left_gap=gap, cube_dz=dz)
    r2 = held_through_window_gate(str(scene), drop, "cube", side="left", frames=[20, 39])
    check("a pinch that opens mid-carry fails, naming where the hold was lost",
          (not r2["ok"]) and r2["longest_unheld_from"] == 28, str(r2))
    push = save("push", one_pad=True)
    r3 = held_through_window_gate(str(scene), push, "cube", side="left", frames=[20, 39])
    check("a single pad pushing the cube is not a hold (no opposing contacts)", (not r3["ok"]) and not r3["end_opposing"], str(r3))
    other = save("other", right=True)
    r4 = held_through_window_gate(str(scene), other, "cube", side="left", frames=[20, 39])
    r5 = held_through_window_gate(str(scene), other, "cube", side=None, frames=[20, 39])
    r6 = held_through_window_gate(str(scene), other, "cube", hand_bodies=["robot_right_pad"], frames=[20, 39])
    check("held by the right pads: the left hand fails, either hand and explicit hand bodies pass",
          (not r4["ok"]) and r5["ok"] and r6["ok"], f"{r4} {r5} {r6}")
    # a steadying hold: one pad pressed on the cube resting on the table, the cube stays (or slides 3 cm)
    Q = np.repeat(q0[None], n, 0)
    for t in range(n):
        Q[t, adr["robot_left_pad_a"]:adr["robot_left_pad_a"] + 3] = [-0.0198, 0, 0.015]
        Q[t, adr["robot_left_pad_b"]:adr["robot_left_pad_b"] + 3] = [0.3, 0.3, 0.05]
    np.savez(tmp / "steady.npz", qpos=Q, fps=30.0)
    Q2 = Q.copy()
    Q2[:, adr["cube"]] = np.linspace(0, 0.03, n)
    Q2[:, adr["robot_left_pad_a"]] = -0.0198 + np.linspace(0, 0.03, n)
    np.savez(tmp / "slides.npz", qpos=Q2, fps=30.0)
    r7 = held_through_window_gate(str(scene), str(tmp / "steady.npz"), "cube", side="left", mode="fixed")
    r8 = held_through_window_gate(str(scene), str(tmp / "slides.npz"), "cube", side="left", mode="fixed")
    check("a steadying hold passes when the object stays and fails when it slides 3 cm",
          r7["ok"] and (not r8["ok"]) and r8["moved_m"] > 0.02, f"{r7} {r8}")
    raised = []
    for kw in ({"mode": "grip"}, {"side": "middle"}, {"hand_bodies": ["nobody_"]}, {"min_frac": 0}):
        try:
            held_through_window_gate(str(scene), carry, "cube", **kw)
        except ValueError as e:
            raised.append(str(e).split(":")[0])
    check("a bad mode, side, hand body and fraction raise", raised == ["mode", "side", "hand_bodies", "min_frac"], str(raised))
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
