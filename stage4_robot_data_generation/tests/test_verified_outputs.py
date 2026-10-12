"""Release-boundary tests: failed or merely planned episodes never become videos."""
import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # the repository: tool_library
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # stage4_robot_data_generation: stage4
from stage4 import export, physics, randomize


class VerifiedOutputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="stage4_verified_")
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "sample"
        self.ep = SimpleNamespace(labels=["hold", "release"], fps=30.0, joint_names=lambda: ["joint"],
                                  model=SimpleNamespace(nq=1, nv=1, nu=1, neq=0))
        self.proposal = np.array([[0.0], [1.0]])
        self.actual = np.array([[0.0], [0.8]])
        self.good = {"ok": True, "physics": {"ok": True, "failed": []}}
        self.execution = {"robot_schema": "dexagent.rigid_robot.v1", "qpos": self.actual,
                          "qvel": np.zeros((2, 1)), "ctrl": np.zeros((2, 1)), "reference": self.proposal,
                          "eq_active": np.zeros((2, 0))}
        self.execution.update(initial_qpos=self.actual[0], initial_qvel=np.zeros(1), initial_state=np.zeros(1),
                              final_state=np.zeros(1), state_spec=1, time_s=np.array([0., 1.]),
                              substep_qpos=self.actual[1:], substep_qvel=np.zeros((1, 1)),
                              substep_ctrl=np.zeros((1, 1)), substep_time_s=np.array([1.]),
                              scene_sha256="unit-fixture", actuator_names=np.array(["motor"]))
        scene = Path(self.tmp.name) / "scene.xml"
        scene.write_text("<mujoco/>")
        self.execution.update(_scene_path=str(scene), scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest())

    def test_failed_or_missing_replay_writes_nothing(self):
        for report, replay in [({"ok": False, "physics": {"ok": True}}, self.actual),
                               ({"ok": True, "physics": {"ok": False}}, self.actual),
                               (self.good, None)]:
            with self.assertRaises(ValueError):
                randomize.save_sample(self.ep, self.out, self.proposal, {}, {}, report, 0, replay=replay)
            self.assertFalse(self.out.exists())

    def test_saved_trajectory_is_replay_and_export_rejects_changes(self):
        randomize.save_sample(self.ep, self.out, self.proposal, {}, {}, self.good, 0, replay=self.actual,
                              execution=self.execution)
        with np.load(self.out / "trajectory.npz") as z:
            np.testing.assert_array_equal(z["qpos"], self.actual)
        info = json.loads((self.out / "sample.json").read_text())
        export.validate_sample(self.out, info)
        with np.load(self.out / "trajectory.npz") as z:
            np.testing.assert_array_equal(z["ctrl"], self.execution["ctrl"])
            self.assertIn("qvel", z)
        np.savez(self.out / "trajectory.npz", qpos=self.proposal)
        with self.assertRaisesRegex(ValueError, "changed after verification"):
            export.validate_sample(self.out, info)

    def test_old_artifact_without_physics_evidence_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no passing physics"):
            export.validate_sample(self.out, {"checks": {"ok": True}})

    def test_scene_overrides_cannot_change_after_validation(self):
        randomize.save_sample(self.ep, self.out, self.proposal, {}, {}, self.good, 0, replay=self.actual,
                              execution=self.execution)
        info = json.loads((self.out / "sample.json").read_text())
        info["body_overrides"] = {"cabinet": {"pos": [1, 2, 3]}}
        with self.assertRaisesRegex(ValueError, "overrides changed"):
            export.validate_sample(self.out, info)

    def test_replay_without_runtime_controls_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "control and state"):
            randomize.save_sample(self.ep, self.out, self.proposal, {}, {}, self.good, 0, replay=self.actual)
        self.assertFalse(self.out.exists())

    def test_bad_original_is_never_ignored_on_augmentation(self):
        ph = physics.Replay.__new__(physics.Replay)
        ph.baseline = {"ok": False, "failed": ["subgoal 1: in_hand"], "not_judged": []}
        result = ph._judged({"checks": {"subgoal 1: in_hand": True}})
        self.assertFalse(result["ok"])
        self.assertIn("original replay: subgoal 1: in_hand", result["failed"])

    def test_randomizer_stops_before_drawing_when_original_fails(self):
        cfg = {"checks": {"physics_replay": True}, "randomize": {"objects": {}}}
        with patch.object(physics, "Replay") as replay:
            replay.return_value.original.return_value = {"ok": False, "failed": ["simulation stable"]}
            with self.assertRaisesRegex(ValueError, "original Stage 3 episode"):
                randomize.Randomizer(self.ep, cfg)

    def test_physics_disabled_cannot_publish(self):
        with self.assertRaisesRegex(ValueError, "requires physics replay"):
            randomize.run({"checks": {"physics_replay": False}})


if __name__ == "__main__":
    unittest.main()
