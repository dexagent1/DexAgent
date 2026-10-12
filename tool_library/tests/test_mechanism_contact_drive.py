"""Independent synthetic contact physics, without a sample scene or executor plan."""
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root: runs as a script too
from tool_library.common.mechanism_contact import contact_drive, friction_upper_bound
from tool_library.skills.reachable_contact_search import reachable_contact_search


def fixture(kind="hinge", rotation=(0., 0., 0.), face=1, friction=.8):
    R = Rotation.from_euler("xyz", rotation)
    quat = np.roll(R.as_quat(), 1)
    p = np.array([.15, 0., .006 * face])
    n = np.array([0., 0., float(face)])
    contact = R.apply(p) + [0., 0., .5]
    normal = R.apply(n)
    sphere = contact + normal * .0095
    axis = "0 1 0" if kind == "hinge" else "0 0 1" if kind == "slide" else "1 0 0"
    driver = (f'<joint name="drive" type="slide" axis="{" ".join(map(str, R.apply([1., 0., 0.])))}"/>'
              if kind == "tangent" else "")
    actuator = '<actuator><velocity joint="drive" kv="100"/></actuator>' if driver else ""
    xml = f'''<mujoco><option timestep="0.001" gravity="0 0 0" cone="elliptic" integrator="implicitfast"/>
      <default><geom friction="{friction} 0 0" condim="{3 if friction else 1}" solref=".003 1"/></default>
      <worldbody><body pos="0 0 .5" quat="{' '.join(map(str, quat))}">
        <body name="panel"><joint name="joint" type="{'hinge' if kind == 'hinge' else 'slide'}" axis="{axis}" damping=".03"/>
        <geom name="panel_geom" type="box" size=".2 .04 .006" pos=".15 0 0" mass=".2"/>
        </body></body>
        <body name="pusher" {'mocap="true"' if not driver else ''} pos="{' '.join(map(str, sphere))}">
        {driver}<geom name="test_collision_hand_right_pad" type="sphere" size=".01" mass=".02"/>
        </body></worldbody>{actuator}</mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, contact, normal, R


class MechanismContactDriveTests(unittest.TestCase):
    def test_real_normal_contact_drives_signed_hinge_and_slide_in_rotated_frames(self):
        for kind in ("hinge", "slide"):
            for rotation in ((0., 0., 0.), (.7, -.4, 1.2)):
                for face in (1, -1):
                    with self.subTest(kind=kind, rotation=rotation, face=face):
                        m, d, p, n, _ = fixture(kind, rotation, face, friction=0.)
                        # For this independent panel, top pressure gives +y
                        # hinge torque and -z slide force.
                        desired = 1 if kind == "hinge" else -1
                        report = contact_drive(m, d, "joint", desired * .3, [p], [n])
                        self.assertEqual(report["status"], "normal_drives" if face == 1 else "cannot_drive")
                        self.assertGreater(d.ncon, 0)
                        initial_effort = float(d.qfrc_constraint[0])
                        self.assertGreater(initial_effort * report["best_normal_effort_per_N"] * desired, 0.)
                        for _ in range(100):
                            mujoco.mj_step(m, d)
                        self.assertGreater(float(d.qpos[0]) * desired * face, 1e-4)

    def test_tangential_friction_is_possible_authority_not_normal_push(self):
        for friction in (0., .8):
            m, d, p, n, R = fixture("tangent", (.4, .3, -.8), friction=friction)
            report = contact_drive(m, d, "joint", .1, [p], [n], [friction, 0., 0.])
            self.assertEqual(report["status"], "friction_required" if friction else "cannot_drive")
            d.ctrl[0] = .1
            for step in range(200):
                mujoco.mj_step(m, d)
            self.assertAlmostEqual(d.time, .2)
            self.assertFalse(any(w.number for w in d.warning))
            if friction:
                self.assertGreater(d.qpos[0], .005)
            else:
                self.assertLess(abs(d.qpos[0]), 1e-8)

    def test_opposed_contacts_can_drive_both_directions_and_axis_contact_cannot(self):
        m, d, p, n, _ = fixture()
        for target in (-.3, .3):
            report = contact_drive(m, d, "joint", target, [p, p - [0., 0., .012]], [n, -n])
            self.assertEqual(report["status"], "normal_drives")
        report = contact_drive(m, d, "joint", .3, [[0., 0., .5]], [[0., 0., 1.]])
        self.assertEqual(report["status"], "cannot_drive")

    def test_angular_friction_and_explicit_pair_overrides_are_included(self):
        # Point at the hinge: translational friction cannot help, but a
        # condim=6 rolling couple may. Never filter it from force-only logic.
        m = mujoco.MjModel.from_xml_string('''<mujoco><default><geom contype="0" conaffinity="0"/></default><worldbody><body name="panel">
          <joint name="joint" axis="0 1 0"/><geom name="panel_geom" type="box" size=".1 .1 .01"/>
          </body><geom name="test_collision_hand_right_pad" type="sphere" size=".01" pos="0 0 1"/>
          </worldbody><contact><pair geom1="panel_geom" geom2="test_collision_hand_right_pad"
          condim="6" friction=".7 .9 .02 .03 .04"/></contact></mujoco>''')
        d = mujoco.MjData(m)
        bounds = friction_upper_bound(m, "right", {m.body("panel").id})
        np.testing.assert_allclose(bounds, [1., .02, .04])
        report = contact_drive(m, d, "joint", .3, [[0., 0., 0.]], [[0., 0., 1.]], bounds)
        self.assertEqual(report["status"], "friction_required")
        self.assertGreater(report["best_cone_effort_upper_per_N"], 0.)

    def test_search_prefers_drive_over_raw_lever_and_preserves_fallback(self):
        m, _, _, _, _ = fixture()
        candidates = [dict(point=[.3, 0., .494], normal=[0., 0., -1.], finger_dir=[1., 0., 0.]),
                      dict(point=[.15, 0., .506], normal=[0., 0., 1.], finger_dir=[1., 0., 0.])]
        def optimize(scene, side, pose, **kw):
            return dict(pose=pose, reach=dict(ok=True, worst_pos_err_mm=0., worst_rot_err_deg=0.),
                        rotation_deg={}, budget_reached=False, n_evaluated=1)
        with patch("tool_library.common.episode.build_execution_model", return_value=(m, None)), \
             patch("tool_library.skills.flat_hand_pose.flat_hand_pose", return_value={}), \
             patch("tool_library.skills.optimize_pose_for_reach.optimize_pose_for_reach", side_effect=optimize) as opt:
            r = reachable_contact_search("synthetic.xml", "right", {}, candidates=candidates,
                sweep=dict(joint="joint", to=.3), ref_point=[0., 0., .5], ref_axis=[0., 1., 0.], time_budget_s=None)
            self.assertEqual(r["n_reachable"], 2)
            self.assertEqual(r["best"]["index"], 1)
            self.assertEqual(r["candidates"][-1]["contact_drive"]["status"], "cannot_drive")
            before = opt.call_count
            r = reachable_contact_search("synthetic.xml", "right", {}, candidates=candidates,
                sweep=dict(joint="joint", to=.3), require_drive=True, time_budget_s=None)
            self.assertEqual(opt.call_count - before, 2)
            self.assertEqual(r["n_drive_rejected"], 1)
            self.assertTrue(r["candidates"][-1]["reachable"])
            before = opt.call_count
            r = reachable_contact_search("synthetic.xml", "right", {}, candidates=candidates,
                fingers=["thumb", "index"], sweep=dict(joint="joint", to=.3), require_drive=True, time_budget_s=None)
            self.assertEqual(opt.call_count - before, 2)
            self.assertEqual(r["n_drive_rejected"], 0)


if __name__ == "__main__":
    unittest.main()
