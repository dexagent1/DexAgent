"""Failed/dry reruns cannot expose a stale success or publish a physical prefix."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tool_library.skills.execute_subgoal_plan import execute_subgoal_plan


class ExecutorOutputLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.scene = str(ROOT / "setup/scene.xml")
        self.first = {"index": 1, "motion": "wait", "side": "left", "duration_s": .2}

    def tearDown(self):
        self.tmp.cleanup()

    def initial_success(self):
        result = execute_subgoal_plan(self.scene, {"subgoals": [self.first]}, str(self.out), plan_arm_paths=False)
        self.assertTrue(result["ok"], result)
        return (self.out / "trajectory.npz").read_bytes()

    def assert_archived(self, old):
        self.assertFalse((self.out / "trajectory.npz").exists())
        files = list(self.out.glob("trajectory.previous_unvalidated.*.npz"))
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].read_bytes(), old)

    def test_failure_before_first_rollout_archives_previous_trajectory(self):
        old = self.initial_success()
        result = execute_subgoal_plan(self.scene, {"subgoals": [dict(self.first, motion="invalid_motion")]},
                                      str(self.out), plan_arm_paths=False)
        self.assertFalse(result["ok"], result)
        self.assertNotIn("trajectory", result["files"])
        self.assert_archived(old)

    def test_successful_dry_suffix_cannot_publish_physical_prefix(self):
        old = self.initial_success()
        prefix = self.out / "subgoal_01/rollout_runtime.npz"
        mtime = prefix.stat().st_mtime_ns
        result = execute_subgoal_plan(self.scene, {"subgoals": [self.first, dict(self.first, index=2)]},
                                      str(self.out), resume=True, dry_run=True, plan_arm_paths=False)
        self.assertTrue(result["ok"], result)
        self.assertTrue(result["dry_run"])
        self.assertEqual(prefix.stat().st_mtime_ns, mtime)
        self.assertNotIn("trajectory", result["files"])
        self.assertNotIn("trajectory_sha256", result)
        self.assert_archived(old)


if __name__ == "__main__":
    unittest.main()
