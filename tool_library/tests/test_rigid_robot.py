"""Physical robot regressions: structural attachment and recorded motor replay.

Run with ``python -m unittest tool_library.tests.test_rigid_robot``. No renderer,
API, prerecorded successful rollout, or task-specific grasp pose is required.
"""
import sys
import unittest
from pathlib import Path
import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root: runs as a script too
from tool_library.common import episode as E, rigid_robot as RR, hand as H


class RigidRobotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proposal, _ = E.build_execution_model(RR.SETUP / "scene.xml")
        cls.robot = RR.compile_rigid(cls.proposal)
        cls.parts = E.robot_parts(cls.proposal)

    def reference(self, frames=31, angle=.08):
        q = np.tile(self.proposal.key_qpos[0], (frames, 1))
        u = np.linspace(0., 1., frames)
        u = u ** 3 * (10 - 15 * u + 6 * u * u)
        q[:, self.parts["right"]["arm_qadr"][0]] += angle * u
        return self.robot.to_proposal_qpos(self.robot.to_runtime_qpos(q))

    def test_fixed_tree_and_named_coordinate_roundtrip(self):
        m = self.robot.model
        self.assertTrue(RR.is_rigid(m))
        self.assertEqual(m.nq, self.proposal.nq - 12)
        q = self.reference()
        qr = self.robot.to_runtime_qpos(q)
        recovered = self.robot.to_runtime_qpos(self.robot.to_proposal_qpos(qr))
        np.testing.assert_array_equal(recovered, qr)
        for side in ("left", "right"):
            hand = m.body(f"robot_{side}_{side}_base_yaw").id
            self.assertEqual(int(m.body_parentid[hand]), m.body(f"robot_{side}_arm_gripper").id)
            self.assertEqual(int(m.body_jntnum[hand]), 0)
        self.assertFalse(np.any(m.body_gravcomp))

    def test_inconsistent_floating_command_is_rejected(self):
        q = self.reference(3, 0.)
        q[1, self.parts["right"]["rig"].wrist_qpos["x"]] += .02
        with self.assertRaisesRegex(ValueError, "uncoupled proposal"):
            E.servo_rollout(self.proposal, q, 30.)

    def test_actual_motor_states_replay_without_controller(self):
        ref = self.reference()
        constants = {key: getattr(self.robot.model, key).copy() for key in
                     ("actuator_gainprm", "actuator_biasprm", "actuator_forcerange", "dof_armature")}
        out = E.servo_rollout(self.proposal, ref, 30.)
        for key, value in constants.items():
            np.testing.assert_array_equal(value, getattr(self.robot.model, key), err_msg=f"runtime mutated serialized {key}")
        self.assertIsNone(out["unstable"])
        self.assertLess(float(out["track_pos_m"].max()), .001)
        self.assertLess(float(out["hand_pen_m"].max()), .001)
        m = mujoco.MjModel.from_xml_string(self.robot.xml)
        d = mujoco.MjData(m)
        mujoco.mj_setState(m, d, out["initial_state"], out["state_spec"])
        mujoco.mj_forward(m, d)
        peak = 0.
        for t, ctrl in enumerate(out["substep_ctrl"]):
            d.ctrl[:] = ctrl
            d.eq_active[:] = out["substep_eq_active_before"][t]
            m.eq_data[:] = out["substep_eq_data"][t]
            mujoco.mj_step(m, d); mujoco.mj_forward(m, d)
            peak = max(peak, float(np.max(np.abs(d.qpos - out["substep_qpos"][t]))))
        self.assertLess(peak, 1e-10)
        for a in range(m.nu):
            if m.actuator(a).name.endswith("_motor"):
                self.assertLessEqual(float(np.abs(out["substep_ctrl"][:, a]).max()), float(m.actuator_ctrlrange[a, 1]))
        np.testing.assert_allclose(np.diff(out["substep_time_s"]), m.opt.timestep, atol=1e-12)

    def test_full_state_survives_a_segment_boundary(self):
        q = self.reference(33)
        first = E.servo_rollout(self.proposal, q[:17], 30.)
        second = E.servo_rollout(self.proposal, q[16:], 30., qpos_start=first["qpos"][-1],
                                 qvel_start=first["qvel"][-1], state_start=first["final_state"])
        np.testing.assert_array_equal(first["runtime_qpos"][-1], second["initial_qpos"])
        np.testing.assert_array_equal(first["qvel"][-1], second["initial_qvel"])
        np.testing.assert_array_equal(first["final_state"], second["initial_state"])
        self.assertIsNone(second["unstable"])

    def test_visible_finger_self_contact_blocks_an_impossible_full_curl(self):
        m, d, rig, _ = H.build_hand_scene([], RR.SETUP / "robot/sharpa_right.xml",
                                         side="right", floor=False, settle_s=0.)
        start = d.qpos.copy()
        target = rig.finger_targets(.98)
        rig.set_fingers(d, target); H.kinematic_forward(m, d)
        self.assertGreater(H.hand_self_overlap(m, d, rig), .008)
        d.qpos[:] = start; d.qvel[:] = 0.
        rig.finger_ctrl(d.ctrl, target)
        mujoco.mj_forward(m, d)
        peak = 0.
        for _ in range(700):
            mujoco.mj_step(m, d); mujoco.mj_forward(m, d)
            peak = max(peak, H.hand_self_overlap(m, d, rig))
        self.assertLess(peak, .003)
        self.assertGreater(max(abs(d.qpos[rig.finger_qpos[k]] - v) for k, v in target.items()), .05)

    def test_thread_release_is_derived_from_travel(self):
        m = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="cap">
          <joint name="lift" type="slide" axis="0 0 1"/><geom size=".01" mass=".02"/>
          </body></worldbody><equality><joint name="thread" joint1="lift"/></equality></mujoco>''')
        d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        engine = RR.MechanismReplay(m, d, [{"engaged": "thread", "lift_joint": "lift", "travel_m": .05, "frac": .96}])
        d.qpos[0] = .047; engine.post_step(d)
        self.assertEqual(int(d.eq_active[0]), 1)
        d.qpos[0] = .0481; engine.post_step(d)
        self.assertEqual(int(d.eq_active[0]), 0)
        self.assertEqual(engine.snapshot()["inactive"], ["thread"])
        engine.close()

    def test_small_rigid_chain_moves_by_bounded_motor_effort(self):
        # A two-joint toy arm with a fixed hand and one finger. A discontinuous
        # reference cannot teleport this arm; torque integration must take time.
        xml = """<mujoco><option timestep="0.002" gravity="0 0 0" integrator="implicitfast"/>
          <worldbody><body name="robot_right_arm_link1"><joint name="robot_right_arm_dof_joint1" axis="0 0 1"/>
            <geom type="capsule" size="0.01" fromto="0 0 0 0.2 0 0" mass="0.2" contype="0" conaffinity="0"/>
            <body name="robot_right_arm_gripper" pos="0.2 0 0"><joint name="robot_right_arm_dof_joint2" axis="0 1 0"/>
              <geom type="sphere" size="0.02" mass="0.1" contype="0" conaffinity="0"/>
              <body name="robot_right_right_base_yaw" pos="0.04 0 0"><body name="robot_right_right_index_tip_body">
                <joint name="robot_right_right_index_MCP_FE" axis="0 1 0" range="0 1"/>
                <geom type="sphere" size="0.01" mass="0.02" contype="0" conaffinity="0"/>
                <site name="robot_right_right_index_tip" pos="0.02 0 0"/>
              </body></body></body></body></worldbody>
          <actuator><motor name="robot_right_arm_dof_joint1_motor" joint="robot_right_arm_dof_joint1" ctrlrange="-0.1 0.1"/>
            <motor name="robot_right_arm_dof_joint2_motor" joint="robot_right_arm_dof_joint2" ctrlrange="-0.1 0.1"/>
            <position name="robot_right_right_index_MCP_FE_position" joint="robot_right_right_index_MCP_FE" kp="30" forcerange="-0.8 0.8"/>
          </actuator></mujoco>"""
        m = mujoco.MjModel.from_xml_string(xml)
        q = np.zeros((4, m.nq)); q[1:, 0] = .5
        out = RR.servo_rollout(m, q, 30.)
        self.assertIsNone(out["unstable"])
        self.assertGreater(float(out["qpos"][-1, 0]), 0.)
        self.assertLess(float(out["qpos"][-1, 0]), .1)
        self.assertLessEqual(float(np.abs(out["substep_ctrl"][:, :2]).max()), .1)


if __name__ == "__main__":
    unittest.main()
