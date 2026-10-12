"""Injected recorded-state slip tests, not claims of physically executed motion.

Real rigid robot execution is covered separately by test_rigid_pnp_integration;
ordinary physical hold positives are covered by test_grasp_stress_penetration.
"""
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root: runs as a script too
from tool_library.common import hand as H, held_contact as HC, outcome as O


def quat(degrees):
    q = Rotation.from_euler("z", degrees, degrees=True).as_quat()
    return q[[3, 0, 1, 2]]


class HeldContactTests(unittest.TestCase):
    @staticmethod
    def rigid():
        m = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body name="wrist" pos="0 0 .3"><freejoint/><geom type="sphere" size=".01"/></body>
          <body name="part" pos=".05 0 .3"><freejoint/><geom type="box" size=".01 .02 .03"/></body>
        </worldbody></mujoco>''')
        return m, np.tile(m.qpos0, (5, 1))

    @staticmethod
    def chain(joint="ball", contact_root=False):
        root_x, child_x = (0., .2) if contact_root else (-.2, .2)
        m = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
          <body name="wrist" pos="0 0 .019"><geom type="sphere" size=".01"/></body>
          <body name="part" pos="{root_x} 0 0"><freejoint/>
            <geom type="sphere" size=".01"/>
            <body name="link" pos="{child_x} 0 0"><joint name="bend" type="{joint}" axis="0 0 1"/>
              <body name="touch"><geom type="sphere" size=".01"/></body>
            </body>
          </body></worldbody></mujoco>''')
        return m, np.tile(m.qpos0, (5, 1))

    def evaluate(self, m, q, **extra):
        wrist = m.body("wrist").id
        parts = {"right": {"rig": SimpleNamespace(body_ids={wrist}), "ee_body": wrist}}
        return O.evaluate(m, q, 30., range(len(q)),
                          dict(hand="right", object="part", no_slip=True, **extra), parts=parts)

    def test_rigid_rotation_and_recovery_fails_without_translation(self):
        m, q = self.rigid()
        q[1, 10:14] = quat(90.)
        result = self.evaluate(m, q)
        self.assertFalse(result["ok"], result)
        self.assertEqual(result["failed"], ["no_slip"])
        self.assertAlmostEqual(result["measured"]["slip_in_hand_m"], 0.)
        self.assertAlmostEqual(result["measured"]["slip_in_hand_deg"], 90.)
        self.assertEqual(result["measured"]["slip_rotation_peak_frame"], 1)
        # Preserve the existing phase window: pre-grasp changes are excluded.
        self.assertTrue(self.evaluate(m, q, slip_from_rel=2)["ok"])

    def test_translation_and_recovery_fails(self):
        m, q = self.rigid()
        q[1, 7] += .035
        result = self.evaluate(m, q)
        self.assertFalse(result["ok"])
        self.assertAlmostEqual(result["measured"]["slip_in_hand_m"], .035)

    def test_object_rotating_rigidly_with_wrist_passes(self):
        m, q = self.rigid()
        for t, angle in enumerate([0., 45., 90., 135., 180.]):
            R = Rotation.from_euler("z", angle, degrees=True).as_matrix()
            q[t, :3] += [t * .01, 0., t * .02]
            q[t, 3:7] = quat(angle)
            q[t, 7:10] = q[t, :3] + R @ np.array([.05, 0., 0.])
            q[t, 10:14] = quat(angle)
        result = self.evaluate(m, q)
        self.assertTrue(result["ok"], result)
        self.assertLess(result["measured"]["slip_in_hand_deg"], .001)

    def test_ball_link_and_its_fixed_contact_descendant_may_bend(self):
        m, q = self.chain()
        q[1, 7:11] = quat(90.)
        result = self.evaluate(m, q)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["measured"]["slip_held_body"], "touch")
        self.assertFalse(result["measured"]["slip_turn_judged"])
        self.assertAlmostEqual(result["measured"]["slip_in_hand_deg"], 90.)
        self.assertIsNotNone(result["measured"]["slip_contact_point_local"])

    def test_chain_root_can_swing_while_the_contacted_link_stays_held(self):
        m, q = self.chain()
        for t, angle in enumerate([0., 30., 60., 90., 120.]):
            R = Rotation.from_euler("z", angle, degrees=True).as_matrix()
            q[t, :3] = -R @ np.array([.2, 0., 0.])
            q[t, 3:7] = quat(angle)
            q[t, 7:11] = quat(-angle)
        result = self.evaluate(m, q)
        self.assertTrue(result["ok"], result)
        self.assertGreater(np.linalg.norm(q[-1, :3] - q[0, :3]), .2)
        self.assertLess(result["measured"]["slip_in_hand_m"], .001)

    def test_hinge_link_rotation_is_judged(self):
        m, q = self.chain(joint="hinge")
        q[1, 7] = np.pi / 2
        result = self.evaluate(m, q)
        self.assertFalse(result["ok"], result)
        self.assertTrue(result["measured"]["slip_turn_judged"])

    def test_free_rigid_root_with_ball_appendage_is_not_exempt(self):
        m, q = self.chain(contact_root=True)
        q[1, 3:7] = quat(90.)
        result = self.evaluate(m, q)
        self.assertFalse(result["ok"], result)
        self.assertEqual(result["measured"]["slip_held_body"], "part")
        self.assertTrue(result["measured"]["slip_turn_judged"])

    def test_hold_samples_every_substep_and_preserves_the_recorder(self):
        # Inject recorded states through the hold callback to verify sampling.
        # No physics success is claimed for this adversarial state sequence.
        m, q = self.rigid()
        q[1, 7] += .035
        q[2, 10:14] = quat(90.)
        d = mujoco.MjData(m); d.qpos[:] = q[0]; mujoco.mj_forward(m, d)
        wrist = m.body("wrist").id
        rig = SimpleNamespace(model=m, wrist_act={}, root_body=wrist, body_ids={wrist})
        seen = []
        def playback(model, data, rig, pos, euler, fingers, seconds, record, label):
            for state in q[1:]:
                data.qpos[:] = state
                mujoco.mj_forward(model, data)
                record(model, data, label)
        with patch.object(H, "step_hold", side_effect=playback):
            result = H.hold_with_slip(m, d, rig, [m.body("part").id], [0, 0, 0], [0, 0, 0], {}, .8,
                                      lambda m, d, label: seen.append(label))[0]
        self.assertEqual(seen, ["hold"] * 4)
        self.assertAlmostEqual(result["drift_m"], .035)
        self.assertAlmostEqual(result["turn_deg"], 90.)
        self.assertTrue(result["turn_judged"])
        np.testing.assert_allclose(d.qpos, q[0])  # endpoints coincide despite the detected slip

    def test_multi_object_hold_measures_each_body_on_the_same_samples(self):
        m = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="wrist"/>
          <body name="first" pos=".05 0 .3"><freejoint/><geom type="sphere" size=".01"/></body>
          <body name="second" pos="-.05 0 .3"><freejoint/><geom type="sphere" size=".01"/></body>
        </worldbody></mujoco>''')
        d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        q = np.tile(d.qpos, (3, 1)); q[1, 7] += .04
        rig = SimpleNamespace(model=m, wrist_act={}, root_body=m.body("wrist").id,
                              body_ids={m.body("wrist").id})
        def playback(model, data, rig, pos, euler, fingers, seconds, record, label):
            for state in q[1:]:
                data.qpos[:] = state
                mujoco.mj_forward(model, data)
                record(model, data, label)
        with patch.object(H, "step_hold", side_effect=playback):
            results = H.hold_with_slip(m, d, rig, [m.body("first").id, m.body("second").id],
                                       [0, 0, 0], [0, 0, 0], {}, .8)
        self.assertAlmostEqual(results[0]["drift_m"], 0.)
        self.assertAlmostEqual(results[1]["drift_m"], .04)


if __name__ == "__main__":
    unittest.main()
