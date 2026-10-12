"""A short move must arrive: the at_target tolerance is at most half the subgoal's planned displacement.

Measured on an EgoDex unbraid run: a 23 mm slide of a ribbon strand moved it 4.8 mm and subgoal_outcome_check still
reported at_target ok with a 19 mm error (the fixed 30 mm tolerance was larger than the move). Long moves keep the
30 mm cap; a move of a few millimetres is judged to 3 mm.

    python tool_library/tests/test_outcome_short_move.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.common import outcome as O  # noqa: E402

FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="strip" pos="0 0 0.0006"><freejoint/>'
                                   '<geom type="box" size="0.006 0.008 0.0006"/></body></worldbody></mujoco>')
T = 40


def run(start, end, target, axes="xy", **over):
    """The strip slid from start to end over the subgoal, judged against target."""
    Q = np.zeros((T, m.nq))
    Q[:, 3] = 1.0
    Q[:, :3] = np.asarray(start, float)[None] + np.linspace(0, 1, T)[:, None] * (np.asarray(end, float) - np.asarray(start, float))[None]
    return O.evaluate(m, Q, 30.0, list(range(T)), dict({"object": "strip", "target_pos": list(target), "target_axes": axes}, **over))


z = 0.0006
# the run's slide: 23.4 mm planned, 4.8 mm moved, 19 mm left
r = run([0.1609 + 0.0234, 0.2284, z], [0.1609 + 0.0234 - 0.0048, 0.2284, z], [0.1609, 0.2284, z])
check("a 23 mm slide that moved the strip 4.8 mm is not at its target", not r["checks"]["at_target"]
      and abs(r["measured"]["target_tol_m"] - 0.0117) < 1e-4 and abs(r["measured"]["planned_move_m"] - 0.0234) < 1e-4,
      f"error {1000 * r['measured']['target_err_m']:.1f} mm, tolerance {1000 * r['measured']['target_tol_m']:.1f} mm")
r = run([0.1609 + 0.0234, 0.2284, z], [0.1609 + 0.002, 0.2284, z], [0.1609, 0.2284, z])
check("the same slide arriving within 2 mm is at its target", r["checks"]["at_target"], f"error {1000 * r['measured']['target_err_m']:.1f} mm")
# a long move keeps the absolute cap: 113 mm planned, 25 mm off still passes the 30 mm tolerance, 35 mm off does not
r = run([0.25, 0.17, z], [0.345 - 0.025, 0.235, z], [0.345, 0.235, z])
check("a 115 mm slide 25 mm short passes (the 30 mm cap rules long moves)", r["checks"]["at_target"] and r["measured"]["target_tol_m"] == 0.03,
      f"tolerance {1000 * r['measured']['target_tol_m']:.1f} mm")
r = run([0.25, 0.17, z], [0.345 - 0.035, 0.235, z], [0.345, 0.235, z])
check("a 115 mm slide 35 mm short fails", not r["checks"]["at_target"])
# a set-down whose xy target is right under the part (the carry before it did the travel): the planned move is the
# whole drop, so a cup that lands 20 mm off after a 15 cm descent passes as before
r = run([0.2, 0.3, 0.15], [0.2 + 0.02, 0.3, 0.0006], [0.2 + 0.004, 0.3, 0.0006])
check("a 15 cm set-down landing 16 mm off in xy keeps the 30 mm cap (the drop is part of the planned move)", r["checks"]["at_target"],
      f"planned {1000 * r['measured']['planned_move_m']:.0f} mm, tolerance {1000 * r['measured']['target_tol_m']:.1f} mm")
# a move of a few millimetres is judged to the 3 mm floor
r = run([0.2, 0.3, z], [0.2, 0.3, z], [0.2 + 0.004, 0.3, z])
check("a 4 mm move that did not happen fails (judged to the 3 mm floor)", not r["checks"]["at_target"]
      and r["measured"]["target_tol_m"] == 0.003, f"error {1000 * r['measured']['target_err_m']:.1f} mm")
# an expectation's own pos_tol_m (insert: 15 mm) still caps it
r = run([0.2, 0.3, z], [0.2 + 0.03, 0.3, z], [0.2 + 0.05, 0.3, z], pos_tol_m=0.015)
check("an insert's 15 mm cap still applies to a 50 mm move 20 mm short", not r["checks"]["at_target"] and r["measured"]["target_tol_m"] == 0.015)

print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
sys.exit(1 if FAILS else 0)
