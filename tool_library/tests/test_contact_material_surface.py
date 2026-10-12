"""A container cavity is empty space; material walls remain penetration obstacles."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tool_library.common.material_surface import MaterialSurface, _ellipsoid_closest
from tool_library.verifiers.hand_object_contact_consistency_check import (
    hand_object_contact_consistency_check, _surface_points, _inside)


class ContactMaterialTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='material_contact_')
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.walls = [([0, 0, .01], [.10, .10, .01]),
                      ([.09, 0, .08], [.01, .10, .06]),
                      ([-.09, 0, .08], [.01, .10, .06]),
                      ([0, .09, .08], [.08, .01, .06]),
                      ([0, -.09, .08], [.08, .01, .06])]

    def scene(self, mesh=False, open_mesh=False, collision_mesh=False):
        asset = ''
        if mesh:
            # One closed, consistently wound open-top material shell; the cavity
            # itself is connected to the exterior and is not filled.
            vertices = []
            for half, z in ((.10, 0), (.10, .14), (.08, .14), (.08, .02)):
                vertices.extend([[-half, -half, z], [half, -half, z], [half, half, z], [-half, half, z]])
            quads = [[0, 3, 2, 1], [12, 13, 14, 15]]
            for i in range(4):
                j = (i+1) % 4
                quads.extend([[i, j, 4+j, 4+i], [4+i, 4+j, 8+j, 8+i], [8+i, 8+j, 12+j, 12+i]])
            faces = [[q[0], q[1], q[2]] for q in quads] + [[q[0], q[2], q[3]] for q in quads]
            shell = trimesh.Trimesh(vertices, faces, process=False)
            shell.fix_normals()
            if open_mesh:
                shell.update_faces(np.arange(len(shell.faces)-1))
            shell.export(self.path/'container.obj')
            asset = f'<asset><mesh name="shell" file="{self.path / "container.obj"}"/></asset>'
            geoms = f'<geom name="shell" type="mesh" mesh="shell" contype="{int(collision_mesh)}" conaffinity="{int(collision_mesh)}"/>'
        else:
            geoms = ''.join(f'<geom name="wall{i}" type="box" pos="{" ".join(map(str,p))}" size="{" ".join(map(str,s))}"/>'
                            for i, (p, s) in enumerate(self.walls))
        path = self.path/'scene.xml'
        path.write_text(f'<mujoco>{asset}<worldbody><body name="container">{geoms}</body></worldbody></mujoco>')
        return path

    def check(self, points, **kwargs):
        path = self.scene(**kwargs)
        semantic = {'objects': [{'id': 'container'}], 'subgoals': [{'index': 1, 'motion': 'hold',
                    'acting_hand': 'right', 'object_refs': ['container'], 'time_window_s': [.1, .4]}]}
        recon = {'objects': [{'id': 'container', 'root_body': 'container'}]}
        (self.path/'sem.json').write_text(json.dumps(semantic))
        (self.path/'recon.json').write_text(json.dumps(recon))
        points = np.asarray(points, float)
        np.savez(self.path/'hands.npz', names=np.array(['rightIndexFingerTip', 'rightMiddleFingerTip'])[:len(points)],
                 rig_xyz=np.tile(points[None], (30, 1, 1)))
        return hand_object_contact_consistency_check(path, self.path/'hands.npz', self.path/'sem.json', self.path/'recon.json', fps=30)

    def test_old_hull_incorrectly_fills_empty_cavity(self):
        m = mujoco.MjModel.from_xml_path(str(self.scene())); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        cavity = np.array([0, 0, .08]); surface = _surface_points(m, d, 'container')
        self.assertTrue(_inside(surface, cavity))
        self.assertGreater(cKDTree(surface).query(cavity)[0], .05)
        material = MaterialSurface(m, {m.body('container').id})
        distance, _ = material.distances(d, [cavity])
        self.assertAlmostEqual(distance[0], .06, places=6)

    def test_touching_inner_wall_with_other_tip_in_cavity_passes(self):
        report = self.check([[.079, 0, .08], [0, 0, .08]])
        self.assertTrue(report['ok'], report)
        self.assertEqual(report['objects']['container']['inside_depth_m'], 0)
        self.assertAlmostEqual(report['objects']['container']['gap_m'], .001, places=4)

    def test_empty_cavity_does_not_falsely_count_as_contact(self):
        report = self.check([[0, 0, .08]])
        self.assertFalse(report['ok'], report)
        self.assertAlmostEqual(report['objects']['container']['gap_m'], .06, places=4)
        self.assertEqual(report['objects']['container']['inside_depth_m'], 0)

    def test_true_material_penetration_remains_failure(self):
        self.walls[1] = ([.11, 0, .08], [.04, .10, .06])
        report = self.check([[.11, 0, .08]])
        self.assertFalse(report['ok'], report)
        self.assertAlmostEqual(report['objects']['container']['inside_depth_m'], .04, places=4)

    def test_visual_mesh_cavity_is_preserved(self):
        report = self.check([[.079, 0, .08], [0, 0, .08]], mesh=True)
        self.assertTrue(report['ok'], report)
        self.assertEqual(report['objects']['container']['inside_depth_m'], 0)

    def test_collision_mesh_convex_hull_still_blocks_cavity(self):
        report = self.check([[0, 0, .08]], mesh=True, collision_mesh=True)
        self.assertFalse(report['ok'], report)
        self.assertGreater(report['objects']['container']['inside_depth_m'], .05)

    def test_hidden_explicit_collision_pair_material_is_included(self):
        path = self.scene(mesh=True)
        xml = path.read_text().replace('name="shell" type="mesh"', 'name="shell" type="mesh" group="3" rgba="0 0 0 0"')
        xml = xml.replace('</worldbody>', '<geom name="peer" type="sphere" pos="2 0 0" size=".01"/></worldbody>')
        xml = xml.replace('</mujoco>', '<contact><pair geom1="shell" geom2="peer"/></contact></mujoco>')
        m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        material = MaterialSurface(m, {m.body('container').id})
        distance, _ = material.distances(d, [[0, 0, .08]])
        self.assertLess(distance[0], -.05)

    def test_open_visual_surface_cannot_certify_interior(self):
        report = self.check([[.079, 0, .08]], mesh=True, open_mesh=True)
        self.assertFalse(report['ok'], report)
        self.assertIn('shell', report['objects']['container']['uncovered_material_geoms'])

    def test_capsule_rounded_tip_is_actual_material(self):
        m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="b"><geom type="capsule" size=".03 .1"/></body></worldbody></mujoco>')
        d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        distance, surface = MaterialSurface(m, {1}).distances(d, [[0, 0, .131], [0, 0, .1]])
        np.testing.assert_allclose(distance, [.001, -.03], atol=1e-9)
        np.testing.assert_allclose(surface[0], [0, 0, .13], atol=1e-9)

    def test_ellipsoid_closest_boundary_handles_centre_and_minor_axis(self):
        size = np.array([.08, .04, .02])
        for p in (np.zeros(3), np.array([.03, 0, 0]), np.array([.10, .05, .03])):
            closest = _ellipsoid_closest(size, p)
            self.assertAlmostEqual(np.linalg.norm(closest/size), 1, places=7)
        np.testing.assert_allclose(_ellipsoid_closest(size, np.zeros(3)), [0, 0, .02])


if __name__ == '__main__':
    unittest.main()
