"""A dry cable motion must never become the state of a later grasp screen."""
import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tool_library.tests import test_grasp_state_provenance as fixture_module
from tool_library.skills.handle_pinch_pose import handle_pinch_pose

EX = importlib.import_module("tool_library.skills.execute_subgoal_plan")


class DeformableDryRunStateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture_module.GraspStateProvenanceTest.setUpClass()
        cls.fixture = fixture_module.GraspStateProvenanceTest
        cls.pose = handle_pinch_pose(cls.fixture.hs, bar="cable_segment6", s=.5,
                                    fingers=["thumb", "index", "middle"])

    @classmethod
    def tearDownClass(cls):
        fixture_module.GraspStateProvenanceTest.tearDownClass()

    def test_detects_undeclared_chain_but_not_robot(self):
        model = self.fixture.model
        ids = EX._deformable_body_ids(model)
        self.assertTrue({model.body(f"cable_link{i}").id for i in range(7)} <= ids)
        self.assertFalse(any(model.body(b).name.startswith("robot_") for b in ids))

    @staticmethod
    def coupling(model, qpos, fps, **kwargs):
        # Isolate state propagation from grasp quality/IK. The real planner,
        # chain interpolation, resume and physical-prefix loading still run.
        return {"qpos": qpos, "labels": kwargs["labels"], "report": {"infeasible": []}, "arm": {}}

    def _grasp(self, index):
        return {"index": index, "motion": "grasp_lift", "side": "left", "object": "cable_link6",
                "pose": self.pose, "lift_m": .04, "duration_s": .2}

    def _dry(self, plan, out, resume=False):
        with patch.object(EX.AH, "couple", side_effect=self.coupling), \
             patch.object(EX.AH.ArmProbe, "check", return_value=[]), \
             patch.object(EX, "_hand_gap_failure", return_value=None), \
             patch.object(EX, "plan_subgoal_keyframes", wraps=EX.plan_subgoal_keyframes) as planner:
            result = EX.execute_subgoal_plan(str(self.fixture.scene), plan, str(out),
                                            dry_run=True, resume=resume, plan_arm_paths=False,
                                            stop_on_fail=False)
        return result, planner.call_count

    def test_next_grasp_never_screened_from_rigidly_moved_cable(self):
        out = self.fixture.ws / "dry"
        result, calls = self._dry({"subgoals": [self._grasp(1), self._grasp(2)]}, out)
        self.assertFalse(result["ok"], result)
        self.assertEqual(calls, 1, result)
        first, second = result["subgoals"]
        self.assertTrue(first["ok"], first)
        self.assertFalse(first["terminal_state_valid"])
        self.assertEqual(second["failed"], ["state_requires_physics"])
        self.assertFalse((out / "subgoal_02/keyframes.json").exists())
        self.assertFalse((out / "trajectory.npz").exists())

    def test_real_prefix_is_reused_before_new_dry_contact(self):
        out = self.fixture.ws / "resume"
        home = {"index": 1, "motion": "wait", "side": "left", "duration_s": .2}
        result = EX.execute_subgoal_plan(str(self.fixture.scene), {"subgoals": [home]}, str(out),
                                        plan_arm_paths=False)
        self.assertTrue(result["ok"], result)
        recorded = out / "subgoal_01/rollout_runtime.npz"
        mtime = recorded.stat().st_mtime_ns
        result, calls = self._dry({"subgoals": [home, self._grasp(2), self._grasp(3)]}, out, resume=True)
        self.assertEqual(recorded.stat().st_mtime_ns, mtime)
        self.assertEqual(calls, 1, result)
        self.assertTrue(result["subgoals"][0]["ok"])
        self.assertNotIn("dry_run", result["subgoals"][0])
        self.assertEqual(result["subgoals"][2]["failed"], ["state_requires_physics"])


if __name__ == "__main__":
    unittest.main()
