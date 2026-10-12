"""A grasp that lifts by passing fingers through each other must still fail."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tool_library.tests import grasp_robust_fixtures as F
from tool_library.verifiers.grasp_approach_open_test import grasp_approach_open_test
from tool_library.verifiers.grasp_hold_test import grasp_hold_test
from tool_library.verifiers.grasp_lift_test import grasp_lift_test
from tool_library.verifiers.grasp_shake_test import grasp_shake_test

VERIFIERS = (grasp_approach_open_test, grasp_lift_test, grasp_hold_test, grasp_shake_test)


class StressPenetrationTest(unittest.TestCase):
    def test_safe_approach_lift_hold_and_shake_on_three_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            heavy = F.box_variant(Path(directory), name="heavy", mass=.18)
            cylinder_xml = Path(directory) / "cylinder.xml"
            cylinder_xml.write_text('<mujoco><worldbody><body name="cyl"><freejoint/>'
                                    '<geom name="cyl_body" type="cylinder" size="0.026 0.045" '
                                    'pos="0 0 0.045" mass="0.12"/>'
                                    '</body></worldbody></mujoco>')
            cylinder = F.scene("cyl", "left", objects=[
                {"name": "cyl", "xml": str(cylinder_xml), "pos": [0, 0, 0]}])
            cases = [(F.scene(), F.pose(F.BOX_PINCH3)),
                     (cylinder, F.pose("cyl_small_left_generated")),
                     (heavy, F.pose(F.BOX_PINCH3))]
            for i, (scene, pose) in enumerate(cases):
                for verifier in VERIFIERS:
                    with self.subTest(case=i, verifier=verifier.__name__):
                        result = verifier(scene, pose)
                        self.assertTrue(result["ok"], result)
                        self.assertTrue(result["penetration"]["ok"])
                        self.assertGreater(result["penetration"]["steps"], 100)

    def test_crossed_fingers_fail_even_when_object_lifts_and_holds(self):
        scene, pose = F.scene(), F.pose("box_right_crossed_invalid")
        for verifier in VERIFIERS:
            with self.subTest(verifier=verifier.__name__):
                result = verifier(scene, pose)
                self.assertFalse(result["ok"], result)
                self.assertEqual(result["first_fail"], "penetration")
                self.assertGreater(result["penetration"]["max_contact_mm"], 10)
                self.assertEqual(result["penetration"]["first_bad"]["phase"], "settle")
                if verifier is grasp_lift_test:
                    self.assertTrue(all(v["achieved_mm"] > .5 * v["commanded_mm"]
                                        for v in result["lifts"]))
                if verifier is grasp_hold_test:
                    hold = next(p for p in result["phases"] if p["name"] == "hold")
                    self.assertTrue(hold["ok"], result)


if __name__ == "__main__":
    unittest.main()
