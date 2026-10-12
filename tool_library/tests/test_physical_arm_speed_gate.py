"""Final acceptance observes full physical speed evidence, not just video frames."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root: runs as a script too
from tool_library.common import episode as E, rigid_robot as RR
from tool_library.verifiers.arm_reach_check import arm_reach_check


class PhysicalArmSpeedGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.ws = Path(cls.tmp.name)
        proposal, _ = E.build_execution_model(RR.SETUP / "scene.xml")
        cls.runtime = RR.compile_rigid(proposal)
        m = cls.runtime.model
        cls.part = E.robot_parts(m)["right"]
        Q = np.tile(m.key_qpos[0], (16, 1))
        trial = RR.servo_rollout(m, Q, 30.)
        cls.arrays = RR.episode_arrays(m, trial, Q)
        cls.arrays.update(labels=np.asarray(["both: wait"] * len(Q)), fps=30.)
        cls.scene = cls.ws / "scene.xml"
        cls.scene.write_text(cls.runtime.xml)

    def verdict(self, edits=None, remove=()):
        arrays = dict(self.arrays)
        arrays.update(edits or {})
        for key in remove:
            arrays.pop(key)
        path = self.ws / "recording.npz"
        np.savez_compressed(path, **arrays)
        return arm_reach_check(str(self.scene), str(path))

    def test_real_stationary_motor_recording_passes(self):
        result = self.verdict()
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["sides"]["right"]["physical_speed"]["speed_source"], "substep_qvel")

    def test_between_frame_velocity_spike_is_rejected(self):
        # Deliberately corrupt only a substep; rendered qpos stay identical.
        velocity = self.arrays["substep_qvel"].copy()
        velocity[1, self.part["arm_dadr"][0]] = np.radians(210.)
        result = self.verdict({"substep_qvel": velocity})
        self.assertFalse(result["ok"])
        self.assertLess(result["sides"]["right"]["frame_peak_joint_speed_deg_s"], 180.)
        self.assertEqual(result["sides"]["right"]["peak_joint_speed_deg_s"], 210.)
        self.assertIn("arm_speed", " ".join(result["issues"]))

    def test_initial_overspeed_is_also_rejected(self):
        velocity = self.arrays["initial_qvel"].copy()
        velocity[self.part["arm_dadr"][0]] = np.radians(200.)
        result = self.verdict({"initial_qvel": velocity})
        self.assertFalse(result["ok"])
        self.assertEqual(result["sides"]["right"]["physical_speed"]["speed_time_s"], 0.)

    def test_missing_nonfinite_and_invalid_time_evidence_fail(self):
        self.assertFalse(self.verdict(remove=("substep_qvel",))["ok"])
        bad = self.arrays["substep_qvel"].copy()
        bad[0, 0] = np.nan
        self.assertFalse(self.verdict({"substep_qvel": bad})["ok"])
        times = self.arrays["substep_time_s"].copy()
        times[1] = times[0]
        self.assertFalse(self.verdict({"substep_time_s": times})["ok"])


if __name__ == "__main__":
    unittest.main()
