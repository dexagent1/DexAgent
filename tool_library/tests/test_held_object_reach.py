"""Grasp screening must account for the carried object's translation and rotation."""
import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root: runs as a script too
from tool_library.common import armhand as AH, plan as P
from tool_library.verifiers.arm_workspace_map import named_orientation


class HeldObjectReachTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[2] / "setup/scene.xml"
        spec = mujoco.MjSpec.from_file(str(path))
        obj = spec.worldbody.add_body(name="reach_payload", pos=[.23, .55, .15])
        obj.add_freejoint(name="reach_payload_free")
        obj.add_geom(name="reach_payload_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
                     size=[.01, .01, .01], mass=.05)
        fixed = obj.add_body(name="attached_part", pos=[0., 0., .02])
        fixed.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[.005, 0., 0.], mass=.005)
        for key in spec.keys:
            key.qpos = list(key.qpos) + [.23, .55, .15, 1., 0., 0., 0.]
        cls.model = spec.compile()
        cls.rig = P.scene_rigs(cls.model)["right"]
        cls.rotation = named_orientation("right", "palm_down")
        cls.pose = dict(pos=[.2, .55, .15], euler=Rotation.from_matrix(cls.rotation).as_euler("XYZ").tolist(),
                        finger_targets=cls.rig.finger_targets(.3), approach=[0., 0., 1.])
        data = mujoco.MjData(cls.model)
        mujoco.mj_resetDataKeyframe(cls.model, data, 0)
        P.set_wrist(cls.model, data, cls.rig, cls.pose["pos"], cls.rotation)
        cls.q = data.qpos.copy()

    def test_object_and_hand_follow_same_transform(self):
        q = self.q.copy()
        target = dict(object="reach_payload", target_pos=[.25, .58, .21], target_euler_deg=[30., 20., 70.], steps=4)
        states = AH._held_waypoint_states(self.model, self.rig, q, [target])
        self.assertEqual(len(states), 4)
        data = mujoco.MjData(self.model)
        for _, state in states:
            data.qpos[:] = state
            mujoco.mj_kinematics(self.model, data)
            b = self.model.body("reach_payload").id
            obj_pos, obj_rot = data.xpos[b], data.xmat[b].reshape(3, 3)
            hand_pos, hand_rot = P.wrist_world(self.model, data, self.rig)
            np.testing.assert_allclose(obj_rot.T @ (hand_pos - obj_pos), [-.03, 0., 0.], atol=1e-10)
            np.testing.assert_allclose(obj_rot.T @ hand_rot, self.rotation, atol=1e-10)
        np.testing.assert_allclose(obj_pos, target["target_pos"], atol=1e-10)
        np.testing.assert_allclose(obj_rot, Rotation.from_euler("XYZ", target["target_euler_deg"], degrees=True).as_matrix(), atol=1e-10)
        np.testing.assert_array_equal(q, self.q)

    def test_following_target_preserves_orientation(self):
        waypoints = [dict(object="reach_payload", target_pos=[.23, .55, .20], target_euler_deg=[0., 0., 45.], steps=2),
                     dict(object="reach_payload", target_pos=[.24, .55, .20], steps=2)]
        states = AH._held_waypoint_states(self.model, self.rig, self.q, waypoints)
        j = self.model.joint("reach_payload_free").qposadr[0]
        np.testing.assert_allclose(states[-1][1][j + 3:j + 7], states[1][1][j + 3:j + 7], atol=1e-10)

    def test_reachable_and_unreachable_destinations(self):
        good = [dict(object="reach_payload", target_pos=[.24, .55, .19], target_euler_deg=[0., 0., 10.], steps=3)]
        positive = AH.screen_poses(self.model, "right", [self.pose], self.q, waypoints=good)[0]
        self.assertTrue(positive["ok"], positive)
        bad = [dict(object="reach_payload", target_pos=[.24, 2.5, .19], target_euler_deg=[0., 0., 10.], steps=2)]
        negative = AH.screen_poses(self.model, "right", [self.pose], self.q, waypoints=bad)[0]
        self.assertFalse(negative["ok"], negative)
        self.assertTrue(negative["failed_at"].startswith("waypoint"), negative)

    def test_legacy_translation_does_not_change_object(self):
        _, q = AH._held_waypoint_states(self.model, self.rig, self.q, [[0., 0., .1]])[0]
        j = self.model.joint("reach_payload_free").qposadr[0]
        np.testing.assert_array_equal(q[j:j + 7], self.q[j:j + 7])
        data = mujoco.MjData(self.model); data.qpos[:] = q
        mujoco.mj_kinematics(self.model, data)
        np.testing.assert_allclose(P.wrist_world(self.model, data, self.rig)[0], [.2, .55, .25], atol=1e-10)

    def test_bad_or_articulated_targets_reject(self):
        for target in [dict(object="attached_part", target_pos=[0, 0, 0]),
                       dict(object="missing", target_pos=[0, 0, 0]),
                       dict(object="reach_payload", target_pos=[0, float("nan"), 0]),
                       dict(object="reach_payload", target_pos=[0, 0, 0], steps=1.5),
                       dict(object="reach_payload", target_pos=[0, 0, 0], typo=0)]:
            with self.subTest(target=target), self.assertRaises(ValueError):
                AH._held_waypoint_states(self.model, self.rig, self.q, [target])


if __name__ == "__main__":
    unittest.main()
