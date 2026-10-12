"""Synthetic geometry and actual MuJoCo FK tests; no reference task solution."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root: runs as a script too
from tool_library.common import knot as K
from tool_library.skills.inspect_rope_state import inspect_rope_state
from tool_library.skills.diagnose_stage3_failure import diagnose_stage3_failure


def loop_with_tail(endpoint=(0., 0., .12), n=32):
    angles = np.linspace(.08, 2 * np.pi - .08, n)
    circle = np.column_stack([.05 * np.cos(angles), .05 * np.sin(angles), np.full(n, .1)])
    return np.vstack([[.09, .02, .1], circle, [.06, -.02, .14], endpoint]), [1, n]


class RopeInspectionTests(unittest.TestCase):
    def test_spatial_aperture_and_outside_endpoint(self):
        P, bounds = loop_with_tail((.12, 0., .12))
        out = inspect_rope_state(points=P, loop_bounds=bounds, radius_m=.003, topology=False)
        loop = out["loops"][0]
        self.assertTrue(out["diagnostic_only"])
        self.assertNotIn("ok", out)
        self.assertTrue(loop["valid_aperture"])
        self.assertAlmostEqual(loop["aperture_clear_radius_m"], .047, delta=.001)
        np.testing.assert_allclose(loop["aperture_center"], [0., 0., .1], atol=.001)
        self.assertFalse(loop["endpoints"]["last"]["projection_inside"])
        json.dumps(out, allow_nan=False)

    def test_material_loop_passage_is_not_merely_plane_crossing(self):
        P, bounds = loop_with_tail()
        inside = np.repeat(P[None], 3, axis=0)
        inside[:, -1, 2] = [.12, .1, .08]  # Includes an exactly on-plane sample.
        out = K.endpoint_passages(inside, bounds, radius=.003)
        self.assertEqual(len(out["passage_candidates"]), 1)
        self.assertEqual(out["passage_candidates"][0]["frames"], [0, 2])
        self.assertGreater(out["passage_candidates"][0]["boundary_clearance_m"], .04)
        outside = inside.copy()
        outside[:, -1, 0] = .12
        out = K.endpoint_passages(outside, bounds, radius=.003)
        self.assertEqual(out["passage_candidates"], [])
        self.assertEqual(out["rejected_plane_crossings"][0]["reason"], "outside aperture")

    def test_rotated_moving_loop_tracks_material_not_world_xy(self):
        P, bounds = loop_with_tail()
        H = np.repeat(P[None], 2, axis=0)
        H[1, 1:bounds[1] + 1, 2] += .04  # Loop moves over a stationary endpoint.
        from scipy.spatial.transform import Rotation
        R = Rotation.from_euler("xyz", [83, 41, 24], degrees=True).as_matrix()
        out = K.endpoint_passages(H @ R.T + [.2, -.3, .05], bounds, radius=.003)
        self.assertEqual(len(out["passage_candidates"]), 1)

    def test_invalid_loops_and_end_caps_are_not_passages(self):
        P, bounds = loop_with_tail()
        H = np.repeat(P[None], 2, axis=0)
        H[1, -1, 2] = .08
        out = K.endpoint_passages(H, bounds, radius=.003, endpoint_radius=.06)
        self.assertEqual(out["passage_candidates"], [])
        self.assertEqual(out["rejected_plane_crossings"][0]["reason"], "insufficient boundary clearance")
        # A half-circle's invented closing chord is not a closed material loop.
        invalid = K.loop_geometry(P, [1, 17], radius=.003)
        self.assertFalse(invalid["valid_aperture"])
        self.assertTrue(any("too far apart" in x for x in invalid["issues"]))

    def test_nonplanar_and_self_intersecting_boundaries_are_flagged(self):
        P, bounds = loop_with_tail()
        P[1:bounds[1] + 1, 2] += .04 * np.sin(np.linspace(0, 4 * np.pi, bounds[1]))
        self.assertFalse(K.loop_geometry(P, bounds, .003)["valid_aperture"])
        bow = np.array([[0., 0., .1], [1., 1., .1], [0., 1., .1], [1., 0., .1], [.001, 0., .1]]) * .1
        bad = K.loop_geometry(bow, [0, 4], .0001)
        self.assertFalse(bad["valid_aperture"])
        self.assertIn("projected boundary self-intersects", bad["issues"])

    def test_straight_has_no_aperture_and_inputs_fail_closed(self):
        line = np.column_stack([np.linspace(0, .5, 12), np.zeros(12), np.full(12, .1)])
        self.assertEqual(inspect_rope_state(points=line, topology=False)["loops"], [])
        with self.assertRaises(ValueError):
            inspect_rope_state(points=line, scene="unread.xml")
        with self.assertRaises(ValueError):
            inspect_rope_state(points=line, history_start=0)
        with self.assertRaises(ValueError):
            inspect_rope_state(points=line, radius_m=float("nan"))
        with self.assertRaises(TypeError):
            inspect_rope_state(points=line, expected_knot="trefoil")
        with self.assertRaises(ValueError):
            K.endpoint_passages([line, line], [0, 11])

    def test_actual_scene_state_and_body_coordinates(self):
        import mujoco
        from tool_library.skills.construct_rope import construct_rope
        with tempfile.TemporaryDirectory() as tmp:
            asset = construct_rope(tmp, name="inspect_cord", segments=12, radius=.004, grips=False)
            model = mujoco.MjModel.from_xml_path(asset["xml"])
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            # Two independently computed FK frames, with a rigid translation.
            q0 = data.qpos.copy()
            q1 = q0.copy()
            q1[1] += .17
            path = Path(tmp) / "trajectory.npz"
            np.savez(path, qpos=[q0, q1], fps=24.)
            out = inspect_rope_state(scene=asset["xml"], trajectory=path, frame=-1, topology=False)
            self.assertEqual(out["source"]["frame"], 1)
            self.assertEqual(out["material_coordinates"][0]["body"], "inspect_cordB_first")
            self.assertEqual(out["material_coordinates"][-2]["body"], "inspect_cordB_last")
            self.assertAlmostEqual(out["radius_m"], .004)
            self.assertAlmostEqual(out["endpoints"]["first"][1], .17, places=6)
            self.assertAlmostEqual(out["length_m"], .6, places=5)
            self.assertFalse(out["source"]["physics_verified"])

    def test_topology_failure_prompts_geometry_not_upstream_guess(self):
        r = diagnose_stage3_failure(verdict={"ok": False, "issues": [
            "rope_knot_check: closures disagree: rope is ambiguous, expected trefoil"]},
            check_hand_scenes=False, include_workspace_history=False)
        self.assertEqual(r["classes"][0]["class"], "deformable_topology")
        self.assertIsNone(r["upstream"])
        self.assertEqual(r["repairs"][0]["tool"], "inspect_rope_state")
        self.assertFalse(any(rp["tool"] == "report_blocked" for rp in r["repairs"]))
        r = diagnose_stage3_failure(outcomes=[{"index": 2, "motion": "carry", "side": "left", "object": "cord",
                                   "ok": False, "failed": ["knot"], "measured": {}}],
                                   check_hand_scenes=False, include_workspace_history=False)
        self.assertEqual(r["classes"][0]["class"], "deformable_topology")


if __name__ == "__main__":
    unittest.main()
