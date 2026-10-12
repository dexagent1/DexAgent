"""All robot groups stay measurable after virtual wrist coordinates are removed."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root: runs as a script too
from tool_library.common import episode as E, episode_audit as EA, rigid_robot as RR
from tool_library.verifiers.robot_joint_jump_audit import robot_joint_jump_audit
from tool_library.verifiers.arm_branch_flip_audit import arm_branch_flip_audit
from tool_library.verifiers.joint_acceleration_audit import joint_acceleration_audit
from tool_library.verifiers.hand_joint_rate_audit import hand_joint_rate_audit


class RigidEpisodeAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        proposal, _ = E.build_execution_model(str(Path(__file__).resolve().parents[2] / "setup/scene.xml"))
        cls.model = RR.compile_rigid(proposal).model
        cls.trajectory = dict(qpos=np.tile(cls.model.key_qpos[0], (61, 1)), fps=30.,
                              labels=np.array(["right: hold"] * 61))

    def test_groups_cover_all_physical_joints_without_virtual_wrist(self):
        ep = EA.Episode(self.model, self.trajectory)
        self.assertEqual(set(ep.hands), {"left", "right"})
        for hand in ep.hands.values():
            self.assertEqual(hand["wrist_qadr"], [])
            self.assertEqual(len(hand["arm_qadr"]), 6)
            self.assertGreater(len(hand["finger_qadr"]), 10)
            self.assertEqual(hand["root"], hand["base"])
        covered = {q for _, q, _, _ in EA.robot_scalar_joints(ep)}
        self.assertEqual(covered, set(range(self.model.nq)))

    def test_stationary_robot_passes_motion_audits(self):
        for audit in (robot_joint_jump_audit, arm_branch_flip_audit, joint_acceleration_audit, hand_joint_rate_audit):
            with self.subTest(audit=audit.__name__):
                report = audit(self.model, self.trajectory)
                self.assertTrue(report["ok"], report)

    def test_real_arm_and_finger_glitches_still_reject(self):
        ep = EA.Episode(self.model, self.trajectory)
        for group in ("arm_qadr", "finger_qadr"):
            with self.subTest(group=group):
                q = self.trajectory["qpos"].copy()
                q[30, ep.hands["right"][group][0]] += .5
                report = robot_joint_jump_audit(self.model, dict(self.trajectory, qpos=q))
                self.assertFalse(report["ok"], report)
                self.assertGreater(report["n_events"], 0)

    def test_contact_queries_preserve_actual_runtime_model(self):
        ep = EA.Episode(self.model, self.trajectory)
        contact = ep.contact_model()
        for field in ("geom_contype", "geom_conaffinity", "pair_geom1", "pair_geom2", "pair_friction"):
            np.testing.assert_array_equal(getattr(contact, field), getattr(self.model, field))
        self.assertTrue(RR.is_rigid(contact))


if __name__ == "__main__":
    unittest.main()
