"""Slip that recovers between the old three samples must still fail locally."""
import sys
import unittest
from types import SimpleNamespace
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root: runs as a script too
from tool_library.common.outcome import evaluate


class TransientSlipTests(unittest.TestCase):
    def test_slip_and_recovery_is_measured_on_every_frame(self):
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="wrist"/>'
            '<body name="part"><freejoint/><geom type="sphere" size=".01"/></body></worldbody></mujoco>')
        q = np.tile(model.qpos0, (5, 1))
        parts = {'right': {'rig': SimpleNamespace(), 'ee_body': model.body('wrist').id}}
        expect = dict(hand='right', object='part', no_slip=True)
        self.assertTrue(evaluate(model, q, 30., list(range(5)), expect, parts=parts)['ok'])
        q[1, 0] = .035
        result = evaluate(model, q, 30., list(range(5)), expect, parts=parts)
        self.assertFalse(result['ok'])
        self.assertIn('no_slip', result['failed'])
        self.assertEqual(result['measured']['slip_peak_frame'], 1)
        self.assertAlmostEqual(result['measured']['slip_in_hand_m'], .035)
        # Motion before grasp is deliberately outside the carrying window.
        expect['slip_from_rel'] = 2
        self.assertTrue(evaluate(model, q, 30., list(range(5)), expect, parts=parts)['ok'])


if __name__ == '__main__':
    unittest.main()
