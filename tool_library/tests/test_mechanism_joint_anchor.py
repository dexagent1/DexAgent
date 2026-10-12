"""Physical turn tests must be invariant to a body's arbitrary local origin.

The two caps have identical collision geometry, mass, inertia and joint axis
in world coordinates. Only the moving body's frame is translated; an arc
about its origin instead of the joint anchor loses the offset cap's grasp.
"""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from tool_library.common import hand as H
from tool_library.skills.pose_from_points import pose_from_points
from tool_library.verifiers.grasp_turn_test import grasp_turn_test
from tool_library.verifiers.grasp_turn_test_keep_state import grasp_turn_test_keep_state

ROOT = Path(__file__).resolve().parents[2]


class MechanismJointAnchorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="mechanism_anchor_")
        cls.scenes = []
        for offset in (0., .07):
            path = Path(cls.tmp.name) / f"cap_{offset}.xml"
            path.write_text(f'''<mujoco><worldbody>
              <body name="base"><geom type="cylinder" size=".03 .03" pos="0 0 .03" mass=".3"/>
                <geom type="cylinder" size=".011 .02" pos="0 0 .08"/>
                <body name="cap" pos="{offset} 0 .1">
                  <joint name="hinge" type="hinge" pos="{-offset} 0 0" axis="0 0 1" damping=".01" frictionloss=".002"/>
                  <geom name="cap_geom" type="cylinder" size=".018 .012" pos="{-offset} 0 .012" mass=".01" friction="1 .01 .001"/>
                </body>
              </body></worldbody></mujoco>''')
            cls.scenes.append(dict(hand_xml=str(ROOT / "setup/robot/sharpa_right.xml"), side="right", floor=True,
                grasp_body="fixture_cap", joint_init={"fixture_hinge": .6},
                objects=[dict(name="fixture", xml=str(path), pos=[0., .45, 0.])]))
        result = pose_from_points(cls.scenes[0],
            {"thumb": [.018, .45, .112], "index": [-.018, .45, .112]},
            {"thumb": [1., 0., 0.], "index": [-1., 0., 0.]},
            approach_dirs=[[0., 0., 1.], [0., -.5, .85]], iters=240)
        cls.pose = result.get("pose", result)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_identical_mechanisms_turn_identically_with_an_offset_body_origin(self):
        states = [H.open_scene(sc) for sc in self.scenes]
        for m, d, _, _ in states:
            mujoco.mj_forward(m, d)
        m0, d0, _, _ = states[0]
        m1, d1, _, _ = states[1]
        np.testing.assert_allclose(d0.geom_xpos[m0.geom("fixture_cap_geom").id],
                                   d1.geom_xpos[m1.geom("fixture_cap_geom").id], atol=1e-9)
        np.testing.assert_allclose(d0.xanchor[m0.joint("fixture_hinge").id],
                                   d1.xanchor[m1.joint("fixture_hinge").id], atol=1e-9)
        self.assertGreater(np.linalg.norm(d1.xanchor[m1.joint("fixture_hinge").id] -
                                          d1.xpos[m1.body("fixture_cap").id]), .06)
        for test in (grasp_turn_test, grasp_turn_test_keep_state):
            with self.subTest(verifier=test.__name__):
                reports = [test(sc, copy.deepcopy(self.pose), "fixture_hinge", sweep_deg=40., seconds=2.)
                           for sc in self.scenes]
                for report in reports:
                    self.assertTrue(report["ok"], report)
                    self.assertGreaterEqual(len(report["measured"]["pads_end"]), 2)
                    self.assertTrue(report["penetration"]["ok"])
                    self.assertLessEqual(report["penetration"]["max_contact_mm"], 3.)
                    self.assertGreater(report["penetration"]["steps"], 100)
                self.assertAlmostEqual(reports[0]["measured"]["turned_deg"],
                                       reports[1]["measured"]["turned_deg"], delta=.2)

    def test_zero_motion_without_contact_is_not_a_blocked_joint_diagnosis(self):
        pose = copy.deepcopy(self.pose)
        pose["pos"][2] += .2
        for test in (grasp_turn_test, grasp_turn_test_keep_state):
            with self.subTest(verifier=test.__name__):
                report = test(self.scenes[0], pose, "fixture_hinge", sweep_deg=10., seconds=.1)
                self.assertFalse(report["ok"])
                self.assertEqual(report["failure_kind"], "no_contact")
                self.assertEqual(report["measured"]["pads_start"], [])
                self.assertAlmostEqual(report["measured"]["turned_deg"], 0., delta=.1)

    def test_opposite_joint_motion_is_signed_and_rejected(self):
        # Apply an opposing physical load to a part the hand misses. This
        # separates measured motion direction from the commanded wrist arc;
        # the former absolute-displacement metric counted it as progress.
        pose = copy.deepcopy(self.pose)
        pose["pos"][2] += .2
        native = H.ramp
        def opposing_load(model, data, rig, *args, **kwargs):
            if kwargs.get("label") == "sweep":
                data.qfrc_applied[model.joint("fixture_hinge").dofadr[0]] = -.02
            return native(model, data, rig, *args, **kwargs)
        for test in (grasp_turn_test, grasp_turn_test_keep_state):
            with self.subTest(verifier=test.__name__), patch.object(H, "ramp", side_effect=opposing_load):
                report = test(self.scenes[0], pose, "fixture_hinge", sweep_deg=10., seconds=.3)
                self.assertFalse(report["ok"])
                self.assertTrue(report["penetration"]["ok"], report)
                self.assertEqual(report["failure_kind"], "wrong_direction", report)
                self.assertLess(report["measured"]["fraction"], 0.)
                self.assertLess(report["measured"]["signed_joint_delta"], -1.)

    def test_deep_finger_overlap_fails_even_with_a_named_contact_pose(self):
        from tool_library.tests import grasp_robust_fixtures as F
        pose = F.pose("box_right_crossed_invalid")
        pose["pos"][2] += .4
        pose["finger_targets_open"] = dict(pose["finger_targets"])
        for test in (grasp_turn_test, grasp_turn_test_keep_state):
            with self.subTest(verifier=test.__name__):
                report = test(self.scenes[0], pose, "fixture_hinge", sweep_deg=10., seconds=.1)
                self.assertFalse(report["ok"])
                self.assertEqual(report["failure_kind"], "penetration", report)
                self.assertGreater(report["penetration"]["max_contact_mm"], 10.)
                self.assertEqual(report["penetration"]["first_bad"]["phase"], "initial approach")

    def test_valid_clockwise_cap_turn_still_passes(self):
        for test in (grasp_turn_test, grasp_turn_test_keep_state):
            with self.subTest(verifier=test.__name__):
                report = test(self.scenes[0], self.pose, "fixture_hinge", direction=-1, sweep_deg=40., seconds=2.)
                self.assertTrue(report["ok"], report)
                self.assertGreaterEqual(report["measured"]["fraction"], .8)
                self.assertLess(report["measured"]["signed_joint_delta"], -30.)


if __name__ == "__main__":
    unittest.main()
