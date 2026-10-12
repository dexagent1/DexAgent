"""A pin descends to genuine first contact, never through a stale press pose."""
import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
from tool_library.common import bimanual as B, hand as H, plan as P
from tool_library.skills.hand_scene_from_recon import hand_scene_from_recon
from tool_library.skills.pin_pose import pin_pose
from tool_library.skills.plan_subgoal_keyframes import plan_subgoal_keyframes
from tool_library.skills.stage_on_bench import stage_on_bench


class PinContactGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.ws = Path(cls.tmp.name)
        asset = cls.ws / "block.xml"
        asset.write_text('<mujoco><worldbody><body name="block" pos="0 0 .025"><freejoint/>'
                         '<geom name="solid" type="box" size=".025 .02 .025" mass=".12"/>'
                         '</body></worldbody></mujoco>')
        cls.scene = cls.ws / "scene.xml"
        stage_on_bench(str(ROOT / "setup/scene.xml"), [{"name": "block", "xml": str(asset), "x": .12, "y": .4}],
                       str(cls.scene), rig={"origin": [0, 0, 0], "R": np.eye(3).tolist()})
        manifest = cls.ws / "scene_recon.json"
        manifest.write_text(json.dumps({"scene_xml": "scene.xml", "objects": [
            {"id": "block", "asset_xml": "block.xml", "root_body": "block_block", "type": "rigid"}]}))
        cls.hs = hand_scene_from_recon(str(manifest), "right", "block")
        result = pin_pose(cls.hs, families=["tip"], n_points=1, n_dirs=4)
        assert result["ok"], result
        cls.pose = result["pose"]
        cls.model, cls.initial, cls.rig, cls.ids = H.open_scene(cls.hs)
        cls.parts = sorted(cls.ids["grasp_gids"])
        cls.entry = {"index": 1, "motion": "pin", "mode": "press", "side": "right", "object": "block_block",
                     "pose": cls.pose, "direction": [0, 0, -1], "force_n": 12}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_tool_and_planner_use_first_contact_then_positive_press(self):
        pose = copy.deepcopy(self.pose)
        pose["pos"][2] -= .002
        planned = plan_subgoal_keyframes(str(self.scene), dict(self.entry, pose=pose), auto_via=False)
        press = planned["expect"]["pin_press"]
        self.assertLess(press["contact_offset_m"], -.001)
        self.assertAlmostEqual(press["travel_m"], 12 / B.WRIST_KP_REF + B.PIN_EXTRA_M)
        self.assertTrue(press["geometry"]["ok"])
        model = mujoco.MjModel.from_xml_path(str(self.scene))
        rig = P.scene_rigs(model)["right"]
        data = mujoco.MjData(model)
        descent = next(k for k in planned["keyframes"] if "descend on" in k["label"])
        data.qpos[:] = descent["qpos"]
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(P.wrist_world(model, data, rig)[0], press["geometry"]["contact_pos"], atol=1e-8)

    def test_large_negative_contact_offset_refused_before_keyframes(self):
        pose = copy.deepcopy(self.pose)
        pose["pos"][2] -= .04
        with self.assertRaisesRegex(ValueError, "press_contact_offset_too_large"):
            plan_subgoal_keyframes(str(self.scene), dict(self.entry, pose=pose), auto_via=False)

    def test_incidental_finger_is_not_a_successful_pin(self):
        pose = dict(self.pose, grip_fingers=["thumb"])
        report = B.pin_press_geometry(self.model, self.initial, self.rig, pose, self.parts)
        self.assertFalse(report["ok"])
        self.assertEqual(report["reason"], "press_wrong_first_contact")

    def test_geometry_safety_limits_cannot_be_relaxed(self):
        for limits in ({"max_overlap_m": .004}, {"max_contact_offset_m": .02}):
            with self.assertRaisesRegex(ValueError, "limits"):
                B.pin_press_geometry(self.model, self.initial, self.rig, self.pose, self.parts, **limits)
        with self.assertRaisesRegex(ValueError, "cannot be relaxed"):
            pin_pose(self.hs, max_overlap_m=.004)

    def test_separated_nonconverged_contact_iterate_is_refused(self):
        # The normal is perpendicular to the physical gap: repeated signed
        # distance steps cannot find a top contact along this direction.
        pose = copy.deepcopy(self.pose)
        pose["pos"][2] += .008
        report = B.pin_press_geometry(self.model, self.initial, self.rig, pose, self.parts, direction=[1, 0, 0])
        self.assertFalse(report["ok"], report)
        self.assertIn(report["reason"], ("press_misses_part", "press_contact_offset_too_large", "press_contact_not_converged"))

    def test_clear_endpoints_do_not_hide_collision_during_approach(self):
        asset = self.ws / "obstacle.xml"
        asset.write_text('<mujoco><worldbody><body name="obstacle" pos="10 10 10">'
                         '<geom name="probe" type="sphere" size=".004"/></body></worldbody></mujoco>')
        hs = copy.deepcopy(self.hs)
        hs["objects"].append({"name": "obstacle", "xml": str(asset)})
        model, initial, rig, ids = H.open_scene(hs)
        p, R = P.pose_from_dict(self.pose)
        obstacle = model.geom("obstacle_probe").id
        data = mujoco.MjData(model)
        data.qpos[:] = initial.qpos
        P.set_wrist(model, data, rig, p + [.025, 0, 0], R)
        rig.set_fingers(data, self.pose["finger_targets"])
        mujoco.mj_forward(model, data)
        middle = data.geom_xpos[model.geom("hand_collision_hand_right_thumb_1").id].copy()
        model.body_pos[int(model.geom_bodyid[obstacle])] = middle
        for dx in (0., .05):
            P.set_wrist(model, data, rig, p + [dx, 0, 0], R)
            mujoco.mj_forward(model, data)
            gap = min(H.geom_distance(model, data, g, obstacle, .1) for g in rig.collision_gids)
            self.assertGreater(gap, .003)
        report = B.pin_press_geometry(model, initial, rig, dict(self.pose, approach=[1, 0, 0]),
                                      sorted(ids["grasp_gids"]))
        self.assertFalse(report["ok"])
        self.assertEqual(report["reason"], "press_approach_collision")
        self.assertGreater(report["samples"], 1)
        self.assertGreater(report["max_overlap_m"], .003)


if __name__ == "__main__":
    unittest.main()
