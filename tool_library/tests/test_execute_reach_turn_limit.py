"""execute_subgoal_plan turns a stress-tested pose for reach on its own only by a little (MAX_REACH_TURN_DEG, or the entry's
max_reach_turn_deg): a larger turn is another grasp, whose stress tests were never run, so it is reported, not tried.

    python tool_library/tests/test_execute_reach_turn_limit.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import tool_library.skills.execute_subgoal_plan as X  # noqa: E402
import tool_library.skills.optimize_pose_for_reach as OPR  # noqa: E402

FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def stub(turn):
    def fake(scene, side, pose, **kw):
        return {"pose": dict(pose, turned=True), "rotation_deg": {"primary": turn, "tilt": 0.0}}
    return fake


sg = {"motion": "grasp_lift", "side": "right", "pose": {"pos": [0, 0.5, 0.1], "euler": [3.14, 0, 0]}, "lift_m": 0.08}
real = OPR.optimize_pose_for_reach
try:
    OPR.optimize_pose_for_reach = stub(10.0)
    v = X._reach_variants("scene.xml", sg, auto_turn=False)
    check("a 10 deg turn is tried", len(v) == 1 and not v[0][0].get("_skip") and v[0][0]["pose"].get("turned"), str([n for _, n in v]))
    OPR.optimize_pose_for_reach = stub(40.0)
    v = X._reach_variants("scene.xml", sg, auto_turn=False)
    check("a 40 deg turn is reported, not tried", len(v) == 1 and "max_reach_turn_deg" in v[0][0].get("_skip", ""), v[0][0].get("_skip", ""))
    v = X._reach_variants("scene.xml", dict(sg, max_reach_turn_deg=45), auto_turn=False)
    check("an entry's max_reach_turn_deg allows it", len(v) == 1 and not v[0][0].get("_skip"))
    both = {"motion": "both", "hands": {"left": dict(sg, side="left"), "right": dict(sg)}}
    v = X._both_reach_variants("scene.xml", both, {"left", "right"}, False, True)
    check("two-hand variants carry a hand's skip", v and all(x[0].get("_skip") for x in v), str(len(v)))
finally:
    OPR.optimize_pose_for_reach = real

print("ALL PASS" if not FAILS else f"{FAILS} FAILED")
sys.exit(1 if FAILS else 0)
