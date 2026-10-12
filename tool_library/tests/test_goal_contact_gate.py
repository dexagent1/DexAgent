"""goal_contact_gate: touching at the end, a push that stops short, contact throughout, apart, table contact, misuse."""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.verifiers.goal_contact_gate import goal_contact_gate  # noqa: E402

FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f" {detail}"))
    FAILS += 0 if ok else 1


XML = """<mujoco><worldbody>
<body name="table" pos="0 0 -0.02"><geom type="box" size="0.4 0.4 0.02"/></body>
<body name="yellow" pos="0 0 0.02"><freejoint/><geom type="box" size="0.02 0.02 0.02" mass="0.02"/></body>
<body name="lavender" pos="0.1 0 0.02"><freejoint/><geom type="box" size="0.02 0.02 0.02" mass="0.02"/></body>
</worldbody></mujoco>"""


def main():
    tmp = Path(tempfile.mkdtemp())
    scene = tmp / "scene.xml"
    scene.write_text(XML)
    m = mujoco.MjModel.from_xml_path(str(scene))
    q0 = m.qpos0.copy()
    lav = m.jnt_qposadr[m.joint(m.body("lavender").jntadr[0]).id]

    def traj(xs, name):
        Q = np.repeat(q0[None], len(xs), 0)
        Q[:, lav] = xs
        p = tmp / f"{name}.npz"
        np.savez(p, qpos=Q, fps=30.0, labels=np.array(["right: push"] * len(xs)))
        return str(p)

    # 1. a push that ends with the side faces touching passes
    good = traj(np.linspace(0.1, 0.04, 20), "good")
    r = goal_contact_gate(str(scene), good, "lavender", "yellow")
    check("a push that ends with the faces touching passes", r["ok"] and r["touching_at_end"], str(r))
    # 2. the lego S4 miss: the brick moved 4.7 cm but stopped 13 mm short
    short = traj(np.linspace(0.1, 0.053, 20), "short")
    r = goal_contact_gate(str(scene), short, "lavender", "yellow")
    check("a push that stops 13 mm short fails with the gap", (not r["ok"]) and abs(r["end_gap_m"] - 0.013) < 0.002
          and "13.0 mm" in r["issues"][0], str(r))
    # 3. contact throughout: touching, then away for a third of the window, then back
    xs = np.r_[np.full(8, 0.04), np.full(7, 0.07), np.full(5, 0.04)]
    back = traj(xs, "back")
    r_end = goal_contact_gate(str(scene), back, "lavender", "yellow")
    r_thr = goal_contact_gate(str(scene), back, "lavender", "yellow", throughout=True, stride=1)
    check("contact lost mid-window passes at the end but fails throughout, naming the frame",
          r_end["ok"] and (not r_thr["ok"]) and r_thr["first_lost_frame"] == 8, f"{r_end} {r_thr}")
    # 4. apart: separated passes, still touching fails
    ra = goal_contact_gate(str(scene), short, "lavender", "yellow", apart=True, apart_m=0.005)
    rb = goal_contact_gate(str(scene), good, "lavender", "yellow", apart=True)
    check("apart passes when separated and fails when touching", ra["ok"] and not rb["ok"], f"{ra} {rb}")
    # 5. the support surface by name: a resting brick touches the table throughout
    rt = goal_contact_gate(str(scene), good, "yellow", "table", throughout=True)
    lifted = traj(np.linspace(0.1, 0.04, 20), "lifted")
    Q = np.load(lifted)["qpos"]
    Q[:, m.jnt_qposadr[m.joint(m.body("yellow").jntadr[0]).id] + 2] = 0.05
    np.savez(lifted, qpos=Q, fps=30.0)
    rl = goal_contact_gate(str(scene), lifted, "yellow", "table")
    check("'table' names the support: resting passes, lifted fails", rt["ok"] and not rl["ok"], f"{rt} {rl}")
    # 6. misuse raises naming the parameter
    raised = []
    for kw in ({"body_b": "nope"}, {"frames": [5, 2]}, {"tol_m": 0}):
        try:
            goal_contact_gate(str(scene), good, kw.pop("body_a", "lavender"), kw.pop("body_b", "yellow"), **kw)
        except ValueError as e:
            raised.append(str(e).split(":")[0])
    check("an unknown body, an empty window and a zero tolerance raise ValueError", raised == ["body_b", "frames", "tol_m"], str(raised))
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
