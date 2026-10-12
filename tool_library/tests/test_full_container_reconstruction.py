"""Reconstruction rest evidence must distinguish partial from full containment."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
from tool_library.common import task_gates as G
from tool_library.common.final_goal_language import measure_custom
from tool_library.verifiers.rest_inside_container_test import rest_inside_container_test, _full_containment
from tool_library.verifiers.scene_recon_manifest_check import scene_recon_manifest_check


class FullContainerReconstructionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def scene(self, half=(.012, .012, .012), interior=(.05, .04, .05), yaw=0.):
        x, y, height = interior
        ox, oy, oz = half
        path = self.ws / f"scene_{len(list(self.ws.glob('scene*xml')))}.xml"
        walls = [(x + .002, 0, .002, y + .004), (-x - .002, 0, .002, y + .004),
                 (0, y + .002, x, .002), (0, -y - .002, x, .002)]
        geoms = ''.join(f'<geom type="box" pos="{px} {py} {height/2}" size="{sx} {sy} {height/2}" mass=".05"/>'
                        for px, py, sx, sy in walls)
        path.write_text(f'<mujoco><option timestep=".002"/><worldbody><geom type="plane" size="1 1 .1"/>'
                        f'<body name="receptacle" pos=".2 .1 0" euler="0 0 {yaw}"><freejoint/>'
                        f'<geom type="box" pos="0 0 .002" size="{x+.004} {y+.004} .002" mass=".2"/>{geoms}</body>'
                        f'<body name="payload" pos="-.2 0 {oz}" euler="0 0 {yaw}"><freejoint/>'
                        f'<geom type="box" size="{ox} {oy} {oz}" mass=".02"/></body></worldbody></mujoco>')
        return path

    def contract(self, scene, full=True, report=None):
        sem = {"task_success": "The payload is " + ("fully " if full else "") + "inside the receptacle.",
               "objects": [{"id": n, "name": n, "type": "rigid", "properties": []} for n in ("payload", "receptacle")]}
        (self.ws / "sem.json").write_text(json.dumps(sem))
        (self.ws / "mask.png").write_bytes(b"manifest form fixture")
        (self.ws / "camera.json").write_text(json.dumps({k: 1 for k in ("fx", "fy", "cx", "cy", "pos", "quat_wxyz", "width", "height")}))
        entries = [{"id": n, "type": "rigid", "root_body": n, "bodies": [n], "asset_xml": str(scene),
                    "skill": "construct_cubes", "skill_origin": "library", "pose": {"x": 0, "y": 0, "yaw_deg": 0},
                    "mask": "mask.png", "iou": 1., "verification_plan": {"settle_stability_check": "rest"},
                    "properties_covered": {"shape": "parameter"}, "verifiers": {"settle_stability_check": {"ok": True}}}
                   for n in ("payload", "receptacle")]
        if report is not None:
            entries[0]["verifiers"]["rest_inside_container_test"] = report
        man = {"scene_semantic": "sem.json", "scene_xml": str(scene), "camera_json": "camera.json",
               "frame0_compare_png": "mask.png", "simulator": "mujoco", "objects": entries}
        path = self.ws / "scene_recon.json"
        path.write_text(json.dumps(man))
        return sem, man, path

    def test_small_block_fully_fits(self):
        report = rest_inside_container_test(self.scene(), "payload", "receptacle", require_full_containment=True)
        self.assertTrue(report["ok"], report)
        self.assertTrue(report["inside"])
        self.assertTrue(report["fully_contained"])
        self.assertEqual(report["full_containment"]["tolerance_m"], .002)

    def test_tall_object_can_rest_partially_but_cannot_claim_full_containment(self):
        scene = self.scene(half=(.012, .012, .045))
        partial = rest_inside_container_test(scene, "payload", "receptacle")
        self.assertTrue(partial["ok"], partial)
        self.assertTrue(partial["inside"])
        self.assertFalse(partial["fully_contained"])
        self.assertGreater(partial["full_containment"]["protrusion_m"][2], .04)
        full = rest_inside_container_test(scene, "payload", "receptacle", require_full_containment=True)
        self.assertFalse(full["ok"])
        self.assertTrue(any("not fully contained" in issue for issue in full["issues"]))

    def test_full_containment_has_fixed_two_mm_tolerance(self):
        for protrusion, expected in ((.0015, True), (.0025, False)):
            with self.subTest(protrusion=protrusion):
                scene = self.scene(half=(.012, .012, (.05 - .004 + protrusion)/2))
                result = rest_inside_container_test(scene, "payload", "receptacle", require_full_containment=True)
                self.assertEqual(result["fully_contained"], expected, result)
                self.assertEqual(result["ok"], expected, result)

    def test_rotated_rectangular_container_uses_its_own_axes(self):
        result = rest_inside_container_test(self.scene(half=(.045, .012, .012), interior=(.05, .02, .05), yaw=45),
                                             "payload", "receptacle", require_full_containment=True)
        self.assertTrue(result["ok"], result)
        size = np.diff(np.asarray(result["full_containment"]["object_bounds"]), axis=0)[0]
        np.testing.assert_allclose(size, [.09, .024, .024], atol=.001)

    def test_measurement_matches_final_goal_gate(self):
        scene = self.scene(yaw=37)
        sem, man, _ = self.contract(scene)
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        obj, vessel = model.body("payload").id, model.body("receptacle").id
        adr = int(model.jnt_qposadr[int(model.body_jntadr[obj])])
        data.qpos[adr:adr+3] = [.2, .1, .016]
        mujoco.mj_forward(model, data)
        full = _full_containment(model, data, obj, vessel)
        gc = G.GoalContext(G.Episode(model, qpos=data.qpos), sem, man)
        final = measure_custom(gc, {"class": "final_full_containment", "A": "payload", "B": "receptacle"}, 0)
        self.assertEqual(full["ok"], final["ok"])
        offset = np.asarray(full["axes_world"]).T @ np.asarray(full["origin_world"])
        for field in ("object_bounds", "container_bounds"):
            np.testing.assert_allclose(np.asarray(full[field]) + offset, final["measured"][field], atol=1e-12)

    def test_manifest_independently_rejects_forged_or_partial_full_claim(self):
        scene = self.scene(half=(.012, .012, .045))
        report = rest_inside_container_test(scene, "payload", "receptacle")
        report["fully_contained"] = True
        report["full_containment"]["ok"] = True
        _, _, path = self.contract(scene, report=report)
        result = scene_recon_manifest_check(path)
        self.assertFalse(result["ok"], result)
        self.assertFalse(result["full_containment_trials"]["ok"])
        self.assertTrue(any("partial inside/rest report" in issue for issue in result["issues"]))
        self.assertTrue(result["full_containment_trials"]["trials"][0]["independently_replayed"])

    def test_manifest_does_not_require_full_for_ordinary_inside(self):
        scene = self.scene(half=(.012, .012, .045))
        report = rest_inside_container_test(scene, "payload", "receptacle")
        _, _, path = self.contract(scene, full=False, report=report)
        result = scene_recon_manifest_check(path)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["full_containment_trials"]["trials"], [])

    def test_proposed_orientation_is_replayed_without_editing_scene(self):
        scene = self.scene(half=(.012, .012, .035), interior=(.06, .04, .05))
        original = scene.read_bytes()
        upright = rest_inside_container_test(scene, "payload", "receptacle", require_full_containment=True)
        self.assertFalse(upright["ok"])
        quat = [np.sqrt(.5), 0., np.sqrt(.5), 0.]
        proposed = rest_inside_container_test(scene, "payload", "receptacle", require_full_containment=True, placed_quat_wxyz=quat)
        self.assertTrue(proposed["ok"], proposed)
        _, _, path = self.contract(scene, report=proposed)
        result = scene_recon_manifest_check(path)
        self.assertTrue(result["ok"], result)
        np.testing.assert_allclose(result["full_containment_trials"]["trials"][0]["reports"][0]["trial_args"]["placed_quat_wxyz"], quat)
        self.assertEqual(scene.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
