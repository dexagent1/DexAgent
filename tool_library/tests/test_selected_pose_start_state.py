"""Re-executing a grasp must replace its earlier starting-state proof."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root: runs as a script too
from tool_library.common.evidence import file_digest
from tool_library.skills.execute_subgoal_plan import _pose_execution_evidence, _pose_candidates, _screen_entry_poses
from tool_library.common.subgoal import SubgoalPlanner


class SelectedPoseStartStateTests(unittest.TestCase):
    def test_single_pose_screen_tracks_the_same_moved_object_as_planning(self):
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="item"><freejoint/>'
                                             '<geom size=".02" mass=".1"/></body></worldbody></mujoco>')
        pose = {"pos": [0., 0., .1], "euler": [0., 0., 0.], "finger_targets": {},
                "grip_fingers": ["index"],
                "anchor": {"body": "item", "pos": [0., 0., 0.], "quat_wxyz": [1., 0., 0., 0.]}}
        data = mujoco.MjData(model)
        data.qpos[:3] = [.12, -.2, .04]
        planner = SimpleNamespace(model=model, pl=SimpleNamespace(data=data), state={})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pose.json"
            path.write_text(json.dumps(pose))
            for candidate in (pose, str(path)):
                sg = {"pose": candidate, "motion": "pin", "side": "left"}
                candidates = _pose_candidates(sg)
                self.assertEqual(len(candidates), 1)
                expected = SubgoalPlanner._pose(planner, sg)
                with patch('tool_library.skills.execute_subgoal_plan.AH.screen_poses', return_value=[]) as screen:
                    _screen_entry_poses(model, "left", candidates, sg, data.qpos)
                    actual = screen.call_args.args[2][0]
                    np.testing.assert_allclose(actual["pos"], expected["pos"])
                    np.testing.assert_allclose(actual["pos"], [.12, -.2, .14])
                    self.assertEqual(screen.call_args.kwargs["approach_m"], [.05])

    def test_changed_state_updates_and_both_hands_share_the_same_proof(self):
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="item"><freejoint/>'
                                             '<geom size=".02" mass=".1"/></body></worldbody></mujoco>')
        pose = {"pos": [0., 0., .1], "euler": [0., 0., 0.], "finger_targets": {}}
        runtime = SimpleNamespace(scene_sha256="test-runtime")
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            def record(side, q):
                return _pose_execution_evidence(model, q, dict(side=side, pose=pose), None, folder, runtime, "as planned")
            start = model.qpos0.copy()
            first = record("left", start)
            moved = start.copy(); moved[:3] += [.1, .2, .3]
            second = record("left", moved)
            self.assertNotEqual(first["start_state_proposal_sha256"], second["start_state_proposal_sha256"])
            with np.load(second["start_state_proposal"]) as z:
                np.testing.assert_array_equal(z["qpos"][0], moved)
            path = Path(second["start_state_proposal"])
            stamp = path.stat().st_mtime_ns
            third = record("right", moved)
            self.assertEqual(third["start_state_proposal_sha256"], file_digest(path))
            self.assertEqual(second["start_state_proposal_sha256"], third["start_state_proposal_sha256"])
            self.assertEqual(stamp, path.stat().st_mtime_ns)


if __name__ == "__main__":
    unittest.main()
