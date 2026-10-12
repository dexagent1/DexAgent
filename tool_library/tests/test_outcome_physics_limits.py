"""Segment feedback must reject final-audit failures without virtual-wrist errors."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root: runs as a script too
from tool_library.common import rigid_robot as RR, outcome as O


class OutcomePhysicsLimitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = RR.compile_rigid(RR.SETUP / "scene.xml").model
        cls.q = np.tile(cls.model.key_qpos[0], (2, 1))

    def evaluate(self, hand_pen=0., object_pen=0., tracking=0., **expect):
        diag = {"hand_pen_m": [0., hand_pen], "object_pen_m": [0., object_pen],
                "track_pos_m": [0., tracking], "track_rot_deg": [0., 0.]}
        return O.evaluate(self.model, self.q, 30., [0, 1], {"hand": "right", **expect}, diagnostics=diag)

    def test_penetration_cap_cannot_be_weakened_by_plan(self):
        self.assertTrue(self.evaluate(hand_pen=.0029, object_pen=.0029)["ok"])
        for key, args in (("hand_penetration", {"hand_pen": .0031, "max_hand_pen_m": .1}),
                          ("object_penetration", {"object_pen": .0031, "max_obj_pen_m": .1})):
            result = self.evaluate(**args)
            self.assertFalse(result["ok"], result)
            self.assertIn(key, result["failed"])
        self.assertFalse(self.evaluate(hand_pen=.002, max_hand_pen_m=.001)["ok"])

    def test_rigid_tracking_failure_has_a_verdict_without_virtual_wrist(self):
        result = self.evaluate(tracking=.05)
        self.assertFalse(result["ok"], result)
        self.assertIn("servo_tracking", result["failed"])
        self.assertNotIn("wrist_ry_max_deg", result["measured"])


if __name__ == "__main__":
    unittest.main()
