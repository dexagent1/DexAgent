"""Multiple unnamed MJCF joints must never overwrite another object's state."""
import sys
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root: runs as a script too
from tool_library.common import episode as E, outcome as O, rigid_robot as RR


class AnonymousJointStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base, _ = E.build_execution_model(RR.SETUP / "scene.xml")
        spec = RR._SPECS[base].copy()
        for name, x in (("block_a", -.08), ("block_b", .09)):
            body = spec.worldbody.add_body(name=name, pos=[x, .4, .04])
            body.add_freejoint()
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[.02] * 3, mass=.1)
            for key in spec.keys:
                key.qpos = np.r_[key.qpos, [x, .4, .04, 1., 0., 0., 0.]]
                key.qvel = np.r_[key.qvel, np.zeros(6)]
        cls.proposal = spec.compile()
        RR.register_model(cls.proposal, spec)
        cls.robot = RR.compile_rigid(cls.proposal)

    def test_both_objects_and_velocities_survive_conversion(self):
        m = self.robot.model
        q = np.tile(m.key_qpos[0], (2, 1))
        q[1, -14:-11] += [.01, .02, .06]
        q[1, -7:-4] += [-.03, .01, .08]
        proposal = self.robot.to_proposal_qpos(q)
        np.testing.assert_array_equal(self.robot.to_runtime_qpos(proposal), q)
        for name, rise in (("block_a", .06), ("block_b", .08)):
            result = O.evaluate(self.proposal, proposal, 30., [0, 1], {"object": name, "rise_m": .05})
            self.assertTrue(result["ok"], result)
            self.assertAlmostEqual(result["measured"]["rise_m"], rise)
        velocity = np.arange(self.proposal.nv, dtype=float)
        converted = RR.map_joints(velocity, self.proposal, m, velocity=True)
        np.testing.assert_array_equal(converted[-12:], velocity[-12:])

    def test_recorded_state_preserves_each_anonymous_free_joint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            q = self.robot.model.key_qpos[0].copy()
            q[-14:-11] += [.02, .03, .06]
            q[-7:-4] += [-.04, .02, .07]
            names = E.joint_names(self.robot.model)
            self.assertEqual(len(names), len(set(names)))
            np.savez(root / "trajectory.npz", qpos=q[None], joint_names=names,
                     scene_sha256=self.robot.scene_sha256)
            (root / "scene_physics.xml").write_text(self.robot.xml)
            state = E.state_qpos(self.proposal, root / "trajectory.npz")
            np.testing.assert_allclose(state, self.robot.to_proposal_qpos(q), atol=1e-8)

    def test_reordered_bodies_and_multiple_unnamed_joints(self):
        def model(order):
            bodies = {
                "a": '<body name="a"><joint type="slide" axis="1 0 0"/>'
                     '<joint type="hinge" axis="0 0 1"/><geom size=".02" mass=".1"/></body>',
                "b": '<body name="b"><body><joint type="ball"/>'
                     '<geom size=".02" mass=".1"/></body></body>',
            }
            return mujoco.MjModel.from_xml_string('<mujoco><worldbody>' + ''.join(bodies[k] for k in order)
                                                 + '</worldbody></mujoco>')
        a, b = model("ab"), model("ba")
        q = np.array([.12, -.31, 1., 0., 0., 0.])
        np.testing.assert_array_equal(RR.map_joints(q, a, b), [1., 0., 0., 0., .12, -.31])
        np.testing.assert_array_equal(RR.map_joints(RR.map_joints(q, a, b), b, a), q)


if __name__ == "__main__":
    unittest.main()
