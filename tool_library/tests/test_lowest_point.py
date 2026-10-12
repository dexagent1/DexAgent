"""Support diagnostics measure actual surfaces, including compiled mesh transforms."""
import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root: runs as a script too
from tool_library.common import episode as E
from tool_library.common import outcome


class LowestPointTest(unittest.TestCase):
    def test_tilted_frustum_mesh_matches_source_vertices(self):
        angles = np.arange(16) * (2 * np.pi / 16)
        vertices = np.vstack([
            np.c_[radius * np.cos(angles), radius * np.sin(angles), np.full(16, z)]
            for radius, z in ((.022, 0.), (.014, .060))
        ])
        body_pos = np.array([.1, -.2, .12])
        geom_pos = np.array([.01, .02, .005])

        def nums(x):
            return " ".join(str(float(v)) for v in np.asarray(x).ravel())

        def quat(rot):
            q = rot.as_quat()
            return nums(np.r_[q[3], q[:3]])

        for body_angles, geom_angles in [([0., 0., 0.], [0., 0., 0.]),
                                         ([23., 11., -7.], [17., -13., 31.])]:
            with self.subTest(body_angles=body_angles, geom_angles=geom_angles):
                body_rot = Rotation.from_euler("xyz", body_angles, degrees=True)
                geom_rot = Rotation.from_euler("xyz", geom_angles, degrees=True)
                model = mujoco.MjModel.from_xml_string(f"""
                  <mujoco><asset><mesh name="frustum" vertex="{nums(vertices)}"/></asset>
                  <worldbody><body name="object" pos="{nums(body_pos)}" quat="{quat(body_rot)}">
                  <freejoint/><geom type="mesh" mesh="frustum" pos="{nums(geom_pos)}"
                  quat="{quat(geom_rot)}" mass=".1"/>
                  </body></worldbody></mujoco>""")
                data = mujoco.MjData(model)
                mujoco.mj_forward(model, data)
                expected = body_rot.apply(geom_rot.apply(vertices) + geom_pos) + body_pos
                self.assertAlmostEqual(E.lowest_point(model, data, {model.body("object").id}),
                                       float(expected[:, 2].min()), places=7)

    def test_wide_floating_box_is_not_supported_by_its_bounding_sphere(self):
        model = mujoco.MjModel.from_xml_string("""
          <mujoco><worldbody><geom type="plane" size="1 1 .1"/>
          <body name="object" pos="0 0 .13"><freejoint/>
          <geom type="box" size=".10 .10 .005" mass=".1"/>
          </body></worldbody></mujoco>""")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        q = np.tile(data.qpos, (2, 1))
        report = outcome.evaluate(model, q, 30, [0, 1], {"object": "object", "supported": True})
        self.assertFalse(report["ok"])
        self.assertIn("supported", report["failed"])
        self.assertEqual(report["measured"]["support_contacts"], 0)
        self.assertAlmostEqual(report["measured"]["lowest_point_z"], .125)


if __name__ == "__main__":
    unittest.main()
