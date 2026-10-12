"""Planning tools consume the actual rigid episode state, without resetting objects."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from tool_library.common import episode as E, rigid_robot as RR


class RecordedPlanningStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base, _ = E.build_execution_model(str(ROOT / 'setup/scene.xml'))
        spec = RR._SPECS[base].copy()
        body = spec.worldbody.add_body(name='test_block', pos=[0., .3, .03])
        body.add_freejoint(name='test_block_free')
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[.02]*3, mass=.1)
        for key in spec.keys:
            key.qpos = np.r_[key.qpos, [0., .3, .03, 1., 0., 0., 0.]]
            key.qvel = np.r_[key.qvel, np.zeros(6)]
        cls.proposal = spec.compile()
        RR.register_model(cls.proposal, spec)
        cls.runtime = RR.compile_rigid(cls.proposal)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'trajectory.npz'
        self.q = self.runtime.model.key_qpos[0].copy()
        self.q[-7:-4] = [.12, .41, .18]
        self.q[0] += .12
        self.names = E.joint_names(self.runtime.model)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, q=None, names=None):
        np.savez(self.path, qpos=np.array([self.q if q is None else q]),
                 joint_names=np.array(self.names if names is None else names),
                 scene_sha256=self.runtime.scene_sha256)
        (self.path.parent / 'scene_physics.xml').write_text(self.runtime.xml)

    def test_rigid_recording_expands_only_virtual_wrist_coordinates(self):
        self.write()
        actual = E.state_qpos(self.proposal, self.path)
        expected = self.runtime.to_proposal_qpos(self.q)
        np.testing.assert_allclose(actual, expected, atol=1e-8)
        np.testing.assert_array_equal(RR.map_joints(actual, self.proposal, self.runtime.model), self.q)
        a, b = mujoco.MjData(self.proposal), mujoco.MjData(self.runtime.model)
        a.qpos[:] = actual; b.qpos[:] = self.q
        mujoco.mj_kinematics(self.proposal, a); mujoco.mj_kinematics(self.runtime.model, b)
        for side in ('left', 'right'):
            name = f'robot_{side}_{side}_base_yaw'
            np.testing.assert_allclose(a.body(name).xpos, b.body(name).xpos, atol=1e-8)
            np.testing.assert_allclose(a.body(name).xmat, b.body(name).xmat, atol=1e-8)

    def test_column_order_follows_identity(self):
        self.write(self.q[::-1], self.names[::-1])
        np.testing.assert_array_equal(E.state_qpos(self.runtime.model, self.path), self.q)

    def test_bound_runtime_scene_is_used_and_missing_binding_is_rejected(self):
        np.savez(self.path, qpos=self.q[None], joint_names=self.names,
                 scene_sha256=self.runtime.scene_sha256)
        with self.assertRaisesRegex(ValueError, 'matching recorded scene'):
            E.state_qpos(self.proposal, self.path)
        (self.path.parent / 'scene_physics.xml').write_text(self.runtime.xml)
        np.testing.assert_allclose(E.state_qpos(self.proposal, self.path),
                                   self.runtime.to_proposal_qpos(self.q), atol=1e-8)

    def test_missing_object_coordinates_cannot_revert_to_initial_state(self):
        self.write(self.q[:-7], self.names[:-7])
        with self.assertRaisesRegex(ValueError, 'missing'):
            E.state_qpos(self.proposal, self.path)

    def test_duplicate_names_and_unnamed_wrong_dimension_fail(self):
        self.write(names=[self.names[1]] + self.names[1:])
        with self.assertRaises(ValueError):
            E.state_qpos(self.proposal, self.path)
        np.savez(self.path, qpos=self.q[None])
        with self.assertRaisesRegex(ValueError, 'named recording'):
            E.state_qpos(self.proposal, self.path)


if __name__ == '__main__':
    unittest.main()
