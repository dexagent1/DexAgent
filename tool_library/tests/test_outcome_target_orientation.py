"""A requested carry/place orientation is measured, not replaced by upright."""
import sys
import unittest
from pathlib import Path
import mujoco
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root: runs as a script too
from tool_library.common.outcome import evaluate


class TargetOrientationTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="part"><freejoint/><geom type="box" size=".02 .03 .04"/></body></worldbody></mujoco>')
        self.q = np.tile(self.model.qpos0, (3, 1))
        self.q[:, 3:7] = [0., 1., 0., 0.]

    def test_inverted_and_quaternion_sign_equivalent_targets_pass(self):
        for target in ([0., 1., 0., 0.], [0., -1., 0., 0.]):
            r = evaluate(self.model, self.q, 30., [0, 1, 2], dict(object="part", target_quat_wxyz=target))
            self.assertTrue(r["ok"], r)
            self.assertEqual(r["measured"]["target_rotation_error_deg"], 0.)

    def test_wrong_orientation_and_conflicting_upright_reject(self):
        for kwargs in (dict(target_quat_wxyz=[1., 0., 0., 0.]),
                       dict(target_quat_wxyz=[0., 1., 0., 0.], upright=True)):
            r = evaluate(self.model, self.q, 30., [0, 1, 2], dict(object="part", **kwargs))
            self.assertFalse(r["ok"], r)

    def test_nonunit_and_nonfinite_targets_reject(self):
        for target in ([0., 0., 0., 0.], [2., 0., 0., 0.], [float("nan"), 1., 0., 0.], [1., 0.]):
            with self.subTest(target=target), self.assertRaises(ValueError):
                evaluate(self.model, self.q, 30., [0, 1, 2], dict(object="part", target_quat_wxyz=target))


if __name__ == "__main__":
    unittest.main()
