"""Search must adjust a valid contact grasp without disturbing its geometry."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tool_library.tests import grasp_robust_fixtures as F
from tool_library.skills.optimize_grasp_pose import optimize_grasp_pose
from tool_library.verifiers.grasp_hold_test import grasp_hold_test
from tool_library.verifiers.grasp_lift_test import grasp_lift_test


class ControlSearchTest(unittest.TestCase):
    def test_refine_long_box_grip(self):
        with tempfile.TemporaryDirectory() as directory:
            objects = []
            for name, x in (("barb", .09), ("bara", -.07)):
                xml = Path(directory) / f"{name}.xml"
                xml.write_text(f'<mujoco><worldbody><body name="{name}" pos="0 0 .04"><freejoint/>'
                               f'<geom name="{name}_g" type="box" size=".05 .015 .04" '
                               'mass=".08" friction="1 .01 .001"/></body></worldbody></mujoco>')
                objects.append({"name": name, "xml": str(xml), "pos": [x, .46, 0]})
            scene = F.scene("bara", "left", objects=objects)
            pose = F.pose("bar_left_wrap")
            before = grasp_hold_test(scene, pose)
            self.assertFalse(before["ok"], before)
            self.assertEqual(before["first_fail"], "lift_0.08")
            result = optimize_grasp_pose(scene, family="seeds", starts=pose, workers=2,
                                         max_evals=32, iterations=2)
            self.assertTrue(result["ok"], result)
            self.assertLessEqual(result["n_evals"], 32)
            for verifier in (grasp_lift_test, grasp_hold_test):
                checked = verifier(scene, result["pose"])
                self.assertTrue(checked["ok"], checked)
                self.assertTrue(checked["penetration"]["ok"])


if __name__ == "__main__":
    unittest.main()
