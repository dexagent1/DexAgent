"""rope_severed_gate: a join switched off mid-window with the halves pulled apart, a join never cut, a join cut whose halves
stay together, a cut before the window, a trajectory without eq_active, automatic join discovery, misuse."""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.verifiers.rope_severed_gate import rope_severed_gate  # noqa: E402

FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f" {detail}"))
    FAILS += 0 if ok else 1


# two rope halves joined end to end by a CONNECT equality (construct_cuttable_rope's cut join)
XML = """<mujoco><worldbody>
<body name="table" pos="0 0 -0.02"><geom type="box" size="0.4 0.4 0.02"/></body>
<body name="ropeA" pos="-0.05 0 0.005"><freejoint/><geom type="capsule" size="0.004 0.05" euler="0 90 0" mass="0.005"/></body>
<body name="ropeB" pos="0.05 0 0.005"><freejoint/><geom type="capsule" size="0.004 0.05" euler="0 90 0" mass="0.005"/></body>
</worldbody>
<equality><connect name="cut_join" body1="ropeA" body2="ropeB" anchor="0.05 0 0"/></equality></mujoco>"""


def main():
    tmp = Path(tempfile.mkdtemp())
    scene = tmp / "scene.xml"
    scene.write_text(XML)
    m = mujoco.MjModel.from_xml_path(str(scene))
    q0 = m.qpos0.copy()
    bb = int(m.jnt_qposadr[m.body("ropeB").jntadr[0]])
    n = 30

    def save(name, cut_at=None, apart_m=0.0, record=True):
        Q = np.repeat(q0[None], n, 0)
        ea = np.ones((n, 1), bool)
        if cut_at is not None:
            ea[cut_at:] = False
            Q[cut_at:, bb] += np.linspace(0, apart_m, n - cut_at)
        f = tmp / f"{name}.npz"
        extra = {"eq_active": ea, "eq_names": np.array(["cut_join"])} if record else {}
        np.savez(f, qpos=Q, fps=30.0, subgoal=np.r_[np.ones(10, int), 2 * np.ones(20, int)], **extra)
        return str(f)

    good = save("good", cut_at=15, apart_m=0.02)
    r1 = rope_severed_gate(str(scene), good, joins="cut_join", subgoal=2)
    r1b = rope_severed_gate(str(scene), good, subgoal=2)
    check("a join switched off at frame 15 with the halves 2 cm apart passes (named and found automatically)",
          r1["ok"] and r1["joins"]["cut_join"]["cut_frame"] == 15 and r1b["ok"] and "cut_join" in r1b["joins"], f"{r1} {r1b}")
    whole = save("whole")
    r2 = rope_severed_gate(str(scene), whole, joins="cut_join")
    r2b = rope_severed_gate(str(scene), whole)
    check("a join never cut fails, named or discovered", (not r2["ok"]) and "still whole" in r2["issues"][0] and not r2b["ok"], f"{r2} {r2b}")
    stuck = save("stuck", cut_at=15, apart_m=0.0)
    r3 = rope_severed_gate(str(scene), stuck, joins="cut_join")
    check("a join switched off whose halves stay together fails", (not r3["ok"]) and "did not separate" in r3["issues"][0], str(r3))
    r4 = rope_severed_gate(str(scene), good, joins="cut_join", frames=[20, 29])
    check("a cut before the window does not count for it", (not r4["ok"]) and "already cut" in r4["issues"][0], str(r4))
    r5 = rope_severed_gate(str(scene), save("norec", cut_at=15, apart_m=0.02, record=False))
    check("a trajectory without eq_active fails with the instruction to record it", (not r5["ok"]) and "eq_active" in r5["issues"][0], str(r5))
    raised = []
    for kw in ({"joins": "nope"}, {"min_gap_m": -1}, {"subgoal": 7}):
        try:
            rope_severed_gate(str(scene), good, **kw)
        except ValueError as e:
            raised.append(str(e).split(":")[0])
    check("an unknown join, a negative gap and a missing subgoal raise", raised == ["joins", "min_gap_m", "subgoal"], str(raised))
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
