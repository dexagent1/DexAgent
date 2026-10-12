"""A model-written plan cannot turn a slipped grasp into an accepted one."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root: runs as a script too
from tool_library.common import held_contact as HC, outcome as O
from tool_library.common.subgoal import SubgoalPlanner
from tool_library.skills.stage_on_bench import stage_on_bench


class RequiredGraspQualityTests(unittest.TestCase):
    def test_required_single_and_both_hand_checks_cannot_be_removed_or_redirected(self):
        generated = dict(hand="right", object="part", no_slip=True)
        for change in ({"no_slip": False}, {"slip_from_rel": 99}, {"slip_from": 99},
                       {"object": "other"}, {"hand": "left"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                HC.merge_motion_expectations(generated, change)
            with self.subTest(both=change), self.assertRaises(ValueError):
                HC.merge_motion_expectations({"hands": {"right": generated}}, {"hands": {"right": change}})
        both = HC.merge_motion_expectations({"hands": {"right": generated}}, {"hands": {}})
        self.assertTrue(both["hands"]["right"]["no_slip"])
        self.assertNotIn("_motion_no_slip", generated)

    def test_secondary_object_retention_cannot_be_waived(self):
        retained = dict(also_in_hand=[dict(body="first", hand="right", fingers=["thumb", "index"])])
        for value in ([], None, [dict(body="other", hand="right")]):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "already held"):
                HC.merge_motion_expectations(retained, dict(also_in_hand=value))
        self.assertEqual(HC.merge_motion_expectations(retained, {})["also_in_hand"], retained["also_in_hand"])

    def test_secondary_object_slip_and_rotation_between_old_samples_reject(self):
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody>'
            '<body name="wrist" pos="0 0 .019"><geom type="sphere" size=".01"/></body>'
            '<body name="part"><freejoint/><geom type="sphere" size=".01"/></body></worldbody></mujoco>')
        base = np.tile(model.qpos0, (5, 1))
        wrist = model.body("wrist").id
        parts = {"right": {"rig": SimpleNamespace(body_ids={wrist}), "ee_body": wrist}}
        expectation = dict(hand="right", slip_tol_m=1., also_in_hand=[dict(body="part", hand="right")])
        self.assertTrue(O.evaluate(model, base, 30., range(5), expectation, parts=parts)["ok"])
        for rotation in (False, True):
            q = base.copy()
            if rotation:
                q[1, 3:7] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
            else:
                q[1, 0] += .035
            result = O.evaluate(model, q, 30., range(5), expectation, parts=parts)
            self.assertFalse(result["ok"], result)
            self.assertIn("also_in_hand:part", result["failed"])
            metric = "rotation_peak_frame" if rotation else "translation_peak_frame"
            self.assertEqual(result["measured"]["also_in_hand"]["part"][metric], 1)

    def test_retiming_rederives_required_window_and_keeps_custom_sliding_semantics(self):
        expectation = HC.merge_motion_expectations(dict(no_slip=True), {"slip_tol_m": .01})
        O.resolve_label_frames(expectation, ["right: approach part", "right: lift part"])
        self.assertEqual(expectation["slip_from_rel"], 1)
        O.resolve_label_frames(expectation, ["right: approach part"] * 3 + ["right: lift part"])
        self.assertEqual(expectation["slip_from_rel"], 3)
        self.assertEqual(expectation["slip_tol_m"], .01)
        slide = HC.merge_motion_expectations(dict(target_pos=[0, 0, 0]), {"no_slip": False})
        self.assertFalse(slide["no_slip"])
        self.assertNotIn("_motion_no_slip", slide)

    def test_tolerance_cannot_hide_transient_slip(self):
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="wrist"/>'
            '<body name="part"><freejoint/><geom type="sphere" size=".01"/></body></worldbody></mujoco>')
        q = np.tile(model.qpos0, (5, 1))
        parts = {"right": {"rig": SimpleNamespace(), "ee_body": model.body("wrist").id}}
        expectation = dict(hand="right", object="part", no_slip=True, slip_tol_m=1.)
        self.assertTrue(O.evaluate(model, q, 30., range(5), expectation, parts=parts)["ok"])
        q[1, 0] = .035
        result = O.evaluate(model, q, 30., range(5), expectation, parts=parts)
        self.assertFalse(result["ok"])
        self.assertIn("no_slip", result["failed"])

    def test_real_planner_rejects_waiver_and_preserves_valid_grasp(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            root = Path(__file__).resolve().parents[2]
            asset = ws / "box.xml"
            asset.write_text('<mujoco><worldbody><body name="box"><freejoint/>'
                             '<geom type="box" size=".024 .019 .028" pos="0 0 .028" mass=".1"/>'
                             '</body></worldbody></mujoco>')
            stage_on_bench(str(root / "setup/scene.xml"),
                           assets=[dict(name="box", xml=str(asset), x=.12, y=.38, yaw_deg=20.)],
                           out_xml=str(ws / "scene.xml"), rig={"origin": [0, 0, 0], "R": np.eye(3).tolist()})
            model = mujoco.MjModel.from_xml_path(str(ws / "scene.xml"))
            pose = json.loads((root / "tool_library/tests/fixtures/grasp_robust/box_bench_right_generated.json").read_text())
            sg = dict(index=1, motion="grasp_lift", side="right", object="box_box", pose=pose)
            q = model.key_qpos[0].copy()
            good = SubgoalPlanner(model, q, fps=30.).run(sg)
            self.assertTrue(good["expect"]["_motion_no_slip"])
            with self.assertRaisesRegex(ValueError, "requires no_slip"):
                SubgoalPlanner(model, q, fps=30.).run({**sg, "expect": {"no_slip": False}})


if __name__ == "__main__":
    unittest.main()
