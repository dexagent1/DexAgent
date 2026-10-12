"""Geometry perturbations must re-seat pads even with an unchanged approach."""
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np

from tool_library.common import hand as H
from tool_library.skills.optimize_grasp_pose import DEFAULT_BOUNDS, _opened, _theta, perturb
from tool_library.tests import grasp_robust_fixtures as F


class RecloseGeometryTest(unittest.TestCase):
    def setUp(self):
        self.scene, self.pose = F.scene(), F.pose(F.BOX_PINCH3)
        self.model, self.initial, self.rig, self.ids = _opened(self.scene)

    def gaps(self, pose):
        d = mujoco.MjData(self.model)
        d.qpos[:] = self.initial.qpos
        H.apply_pose(self.model, d, self.rig, pose)
        return np.array([
            min(float(H.geom_distance(self.model, d, g, o, .1))
                for g in self.rig.tip_geoms(f) for o in self.ids["grasp_gids"])
            for f in self.pose["grip_fingers"]])

    def test_shift_reseats_without_changing_approach_parameter(self):
        for distance in (.005, .01):
            with self.subTest(distance=distance):
                theta = _theta(np.zeros(len(DEFAULT_BOUNDS)), self.pose, DEFAULT_BOUNDS)
                theta["off_pinch_m"] = distance
                shifted = perturb(self.scene, self.pose, theta)
                stale = dict(shifted, finger_targets=self.pose["finger_targets"])
                self.assertGreater(np.max(np.abs(self.gaps(stale))), .004)
                self.assertLess(np.max(np.abs(self.gaps(shifted))), .002)

    def test_approach_only_preserves_contact_geometry(self):
        self.pose = dict(self.pose, approach=[1., 0., 0.])
        theta = _theta(np.zeros(len(DEFAULT_BOUNDS)), self.pose, DEFAULT_BOUNDS)
        theta["approach_up"] = .5
        changed = perturb(self.scene, self.pose, theta)
        self.assertEqual(changed["finger_targets"], self.pose["finger_targets"])
        np.testing.assert_allclose(self.gaps(changed), self.gaps(self.pose), atol=1e-12)
        self.assertFalse(np.allclose(changed["approach"], self.pose["approach"]))


if __name__ == "__main__":
    unittest.main()
