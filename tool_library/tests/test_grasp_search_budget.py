"""A proposal budget must not bypass physical checks or hide its searched scope."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tool_library.tests import grasp_robust_fixtures as F
from tool_library.skills.handle_pinch_pose import handle_pinch_pose
from tool_library.skills.optimize_grasp_pose import optimize_grasp_pose


class GraspSearchBudgetTest(unittest.TestCase):
    def test_valid_supplied_start_gets_full_checks_before_fallback(self):
        with patch("tool_library.skills.optimize_grasp_pose.family_jobs", side_effect=AssertionError("unneeded fallback")):
            result = optimize_grasp_pose(F.scene(), family="rigid", starts=F.pose(F.BOX_PINCH3),
                                         workers=1, max_evals=16)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["n_evals"], 1)
        self.assertEqual(result["supplied_starts_tested_first"], 1)
        self.assertEqual(result["start_generation"], [])
        self.assertEqual(set(result["pose"]["stress_tests"]), {"grasp_lift_test", "grasp_hold_test"})
        self.assertTrue(all(t["ok"] for t in result["pose"]["stress_tests"].values()))

    def test_supplied_start_does_not_bypass_collision_or_write_output(self):
        pose = F.pose(F.BOX_PINCH3)
        pose["pos"][2] -= .2
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "pose.json"
            result = optimize_grasp_pose(F.scene(), family="rigid", starts=pose, workers=1,
                                         max_evals=1, out_json=str(out))
            self.assertFalse(result["ok"], result)
            self.assertEqual(result["n_evals"], 1)
            self.assertEqual(result["n_screened_out"], 1)
            self.assertFalse(out.exists())

    def test_bar_seed_budget_reports_partial_search(self):
        result = handle_pinch_pose(F.scene("cyl", "left"), "cyl_cyl_body",
                                   fingers=["thumb", "index"], time_budget_s=1e-6, iters=2)
        self.assertEqual(result["ik_search"]["seeds_tried"], 1)
        self.assertGreater(result["ik_search"]["seeds_total"], 1)
        self.assertEqual(result["ik_search"]["stopped"], "time_budget")


if __name__ == "__main__":
    unittest.main()
