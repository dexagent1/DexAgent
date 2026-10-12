"""Rotating an asymmetric part changes its set-down height and support footprint."""
import sys
import unittest
from types import SimpleNamespace
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root: runs as a script too
from tool_library.common import episode as E
from tool_library.common.subgoal import SubgoalPlanner


class PlacementTargetExtentTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body name="support"><geom type="box" size=".2 .2 .025" pos="0 0 .025"/></body>
          <body name="part" pos=".4 .3 .2" euler=".3 .2 .1"><freejoint/>
            <geom type="box" size=".01 .02 .04" pos=".01 0 .04" mass=".1"/>
          </body></worldbody></mujoco>''')
        # Geometry queries need the real MuJoCo object/support state, but no
        # robot or actuator planning. Keep this fixture independent of arms.
        self.planner = SubgoalPlanner.__new__(SubgoalPlanner)
        self.planner.model = self.model
        self.planner.pl = SimpleNamespace(data=mujoco.MjData(self.model))
        mujoco.mj_forward(self.model, self.planner.pl.data)

    def test_target_orientation_controls_height_and_does_not_change_start(self):
        p = self.planner
        before = p.pl.data.qpos.copy()
        for mode in ('into', 'on', 'over'):
            sg = {mode: 'support', 'target_euler_deg': [0, 90, 0]}
            target, _ = p._target_from(sg, 'part')
            # Body-local x in [0,.02] becomes world z in [-.02,0].
            clearance = {'into': .003, 'on': .002, 'over': .08}[mode]
            self.assertAlmostEqual(target[2], .05 + .02 + clearance, places=7)
            np.testing.assert_array_equal(p.pl.data.qpos, before)
            R = Rotation.from_euler('XYZ', [0, 90, 0], degrees=True).as_matrix()
            rect = p._footprint_at('part', [0, 0], R)
            np.testing.assert_allclose(rect, [0, .08, -.02, .02], atol=1e-10)

    def test_predicted_extent_matches_actual_fk_for_multiple_orientations(self):
        p = self.planner
        data = mujoco.MjData(self.model)
        body = self.model.body('part').id
        for angles in ([0, 0, 0], [180, 0, 0], [37, -23, 61]):
            R = Rotation.from_euler('XYZ', angles, degrees=True)
            q = R.as_quat()
            data.qpos[:] = self.model.qpos0
            data.qpos[3:7] = np.r_[q[3], q[:3]]
            mujoco.mj_forward(self.model, data)
            for axis in np.eye(3):
                predicted = p._relative_extent('part', axis, R.as_matrix())
                measured = np.asarray(E.bodies_extent(self.model, data, {body}, u=axis)) - data.xpos[body] @ axis
                np.testing.assert_allclose(predicted, measured, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
