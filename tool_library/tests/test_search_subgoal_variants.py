"""search_subgoal_variants without physics: knobs proposed from failures, the sampled space, plan edits, the ranking,
the upstream knobs, and the prefix copy a trial resumes from."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root: runs as a script too
from tool_library.skills import search_subgoal_variants as S

PLAN = {"subgoals": [
    {"index": 1, "motion": "grasp_lift", "side": "left", "object": "cup", "pose": "poses/g.json", "t_close": 0.5},
    {"index": 2, "motion": "carry", "side": "left", "object": "cup", "target_pos": [0.0, 0.4, 0.14], "duration_s": 2.0},
    {"index": 3, "motion": "place", "side": "left", "object": "cup", "into": "bowl", "offset": [0.0, 0.0, 0.004]}]}
MISS = {"index": 3, "failed": ["at_target", "release_settle"],
        "measured": {"target_err_m": 0.02, "release_settle": {"moved_since_release_m": 0.01}}}


class SearchLogicTests(unittest.TestCase):
    def test_place_miss_proposes_place_knobs_only(self):
        k = S.auto_knobs(PLAN["subgoals"][2], MISS)
        self.assertIn("offset", k)
        self.assertIn("set_down_s", k)
        self.assertNotIn("t_close", k)                    # a grasp knob does not apply to a place

    def test_two_hand_checks_map_to_their_hand(self):
        both = {"index": 2, "motion": "both", "hands": {"left": {"motion": "grasp_lift"}, "right": {"motion": "hold"}}}
        k = S.auto_knobs(both, {"failed": ["left:rise", "right:no_slip"]})
        self.assertTrue(any(x.startswith("left.") for x in k) and any(x.startswith("right.") for x in k))

    def test_space_values_respect_ranges_and_deltas(self):
        knobs = {2: {"offset": {"delta_low": [-0.01, -0.01, 0.0], "delta_high": [0.01, 0.01, 0.0]}, "set_down_s": [0.6, 2.5],
                     "arm_path": {"choices": [None, "joint"]}},
                 1: {"duration_s": {"scale": [1.0, 2.0]}}}
        sp = S.Space(knobs, PLAN)
        for point in sp.lhs(8, np.random.default_rng(0)) + sp.refit(sp.lhs(2, np.random.default_rng(1)), 4, np.random.default_rng(2)):
            v = sp.values(point)
            self.assertTrue(0.6 <= v["2@set_down_s"] <= 2.5)
            self.assertTrue(all(abs(a - b) <= 0.01 + 1e-9 for a, b in zip(v["2@offset"][:2], [0.0, 0.0])))
            self.assertEqual(v["2@offset"][2], 0.004)       # a zero-width component keeps the entry's value
            self.assertTrue(2.0 <= v["1@duration_s"] <= 4.0)
            self.assertIn(v["2@arm_path"], (None, "joint"))

    def test_lhs_covers_each_dimension_once_per_stratum(self):
        sp = S.Space({0: {"t_close": [0.0, 1.0]}}, PLAN)
        xs = sorted(sp.values(p)["0@t_close"] for p in sp.lhs(5, np.random.default_rng(3)))
        self.assertEqual([int(x * 5) for x in xs], [0, 1, 2, 3, 4])

    def test_ranking_prefers_progress_then_goals_then_fewer_failures_then_shortfall(self):
        far = {"subgoals_passed": 2, "first_failure": {"failed": ["at_target"], "measured": {"target_err_m": 0.03}}}
        near = {"subgoals_passed": 2, "first_failure": {"failed": ["at_target"], "measured": {"target_err_m": 0.01}}}
        two = {"subgoals_passed": 2, "first_failure": {"failed": ["at_target", "at_rest"], "measured": {"target_err_m": 0.001}}}
        deeper = {"subgoals_passed": 3, "first_failure": {"failed": ["at_target", "at_rest"]}}
        done_goals = {"subgoals_passed": 3, "goals_ok": 2, "first_failure": None}
        crashed = {"subgoals_passed": 0, "error": "timeout"}
        order = sorted([far, near, two, deeper, done_goals, crashed], key=S.score_of, reverse=True)
        self.assertEqual(order, [done_goals, deeper, near, far, two, crashed])

    def test_upstream_knobs_reach_the_carry_and_the_grasp_of_the_same_object(self):
        up = S.upstream_knobs(PLAN, 2, MISS)
        self.assertEqual(sorted(up), [0, 1, 2])
        self.assertIn("target_pos", up[1])
        self.assertIn("t_close", up[0])

    def test_prefix_copy_keeps_only_passing_segments_before_the_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp, "exec"), Path(tmp, "trial")
            rows = [{"index": 1, "tag": "subgoal_01", "ok": True}, {"index": 2, "tag": "subgoal_02", "ok": True},
                    {"index": 3, "tag": "subgoal_03", "ok": False}]
            for r in rows:
                (src / r["tag"]).mkdir(parents=True)
                (src / r["tag"] / "rollout.npz").write_bytes(b"x")
            (src / "outcomes.json").write_text(json.dumps(rows))
            (src / "trajectory.npz").write_bytes(b"old result")
            S._copy_prefix(src, dst, 2)
            self.assertEqual(sorted(p.name for p in dst.iterdir()), ["outcomes.json", "subgoal_01"])
            self.assertEqual([r["tag"] for r in json.loads((dst / "outcomes.json").read_text())], ["subgoal_01"])

    def test_knob_in_a_plan_entry_and_a_hand(self):
        e = {"motion": "both", "hands": {"left": {"t_open": 0.5}}}
        S._set(e, "left.t_open", 1.2)
        S._set(e, "right.t_close", 0.9)
        self.assertEqual(S._get(e, "left.t_open"), 1.2)
        self.assertEqual(e["hands"]["right"]["t_close"], 0.9)


if __name__ == "__main__":
    unittest.main()
