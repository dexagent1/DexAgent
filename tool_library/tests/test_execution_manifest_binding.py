"""Adversarial edits of generator metadata around a fixed executor receipt.

These fixtures isolate the manifest gate, not whole-robot physical acceptance.
Scene/trajectory/report bytes are fixed while only manifest claims are changed.
"""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root: runs as a script too
from tool_library.common.evidence import file_digest
from tool_library.verifiers.trajectory_manifest_check import trajectory_manifest_check, _bind_execution_identity


class ExecutionManifestBindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmp.name)
        scene = self.ws / "scene.xml"
        scene.write_text('<mujoco><worldbody><body name="object"><freejoint/><geom size=".01" mass=".1"/></body>'
                         '<body name="other" pos=".1 0 0"><freejoint/><geom size=".01" mass=".1"/></body></worldbody></mujoco>')
        model = mujoco.MjModel.from_xml_path(str(scene))
        np.savez(self.ws / "trajectory.npz", qpos=np.repeat(model.qpos0[None], 3, axis=0), fps=30.,
                 labels=np.array(["left: carry"] * 3))
        (self.ws / "sim_video.mp4").write_bytes(b"manifest existence fixture, not a rendered delivery")
        (self.ws / "stage4_task").mkdir()
        (self.ws / "stage4_task/stage4.json").write_text('{}')
        (self.ws / "recon.json").write_text('{"scene_xml":"scene.xml"}')
        (self.ws / "semantic.json").write_text('{"subgoals":[{"index":1,"motion":"carry","object_refs":["object"]}]}')
        self.executed = {"index": 1, "stage1_indices": [1], "step": None, "motion": "carry", "side": "left",
                         "object": "object", "frames": [0, 2], "ok": True}
        self.manifest = {"scene_semantic": "semantic.json", "scene_recon": "recon.json", "scene_executed": "scene.xml",
                         "subgoals": [{**self.executed, "outcome": {"ok": True}}],
                         "episode": {"trajectory": "trajectory.npz", "sim_video": "sim_video.mp4", "execution_report": "execution_report.json"},
                         "stage4": {"task_dir": "stage4_task"}, "checks": {name: {"ok": True} for name in
                         ("episode_physics_check", "robot_self_collision_check", "arm_reach_check", "stage4_handoff_check")}}

    def tearDown(self):
        self.tmp.cleanup()

    def check(self, manifest=None, execution_entries=None):
        execution = {"ok": True, "dry_run": False, "subgoals": execution_entries or [self.executed],
                     "scene_sha256": file_digest(self.ws / "scene.xml"), "trajectory_sha256": file_digest(self.ws / "trajectory.npz")}
        (self.ws / "execution_report.json").write_text(json.dumps(execution))
        (self.ws / "trajectory_manifest.json").write_text(json.dumps(manifest or self.manifest))
        return trajectory_manifest_check(str(self.ws / "trajectory_manifest.json"),
                                         trusted_execution_reports=[file_digest(self.ws / "execution_report.json")])

    def test_matching_non_grasp_metadata_passes_manifest_gate(self):
        report = self.check()
        self.assertTrue(report["ok"], report)

    def test_grasp_cannot_be_relabelled_carry_to_erase_stress_requirement(self):
        self.executed["motion"] = "grasp_lift"
        report = self.check()
        self.assertFalse(report["ok"], "trusted grasp report was accepted as a carry without any pose or stress evidence")
        self.assertIn("motion", " ".join(report["issues"]))

    def test_hand_object_step_and_stage1_scope_cannot_change(self):
        for key, value in (("side", "right"), ("object", "other"), ("step", 2), ("stage1_indices", [1, 2])):
            with self.subTest(key=key):
                changed = copy.deepcopy(self.manifest)
                changed["subgoals"][0][key] = value
                report = self.check(changed)
                self.assertFalse(report["ok"], f"changed {key} passed")

    def test_manifest_cannot_omit_or_duplicate_trusted_execution_entries(self):
        second = {**self.executed, "index": 2, "stage1_indices": [2], "step": 1, "frames": [2, 2]}
        self.assertFalse(self.check(execution_entries=[self.executed, second])["ok"])
        changed = copy.deepcopy(self.manifest)
        changed["subgoals"].append(copy.deepcopy(changed["subgoals"][0]))
        self.assertFalse(self.check(changed)["ok"])

    def test_false_executor_entry_cannot_hide_under_top_level_success(self):
        self.executed["ok"] = False
        self.assertFalse(self.check()["ok"])

    def test_truthy_non_boolean_outcome_does_not_pass(self):
        for value in ("false", 1, [True]):
            with self.subTest(value=value):
                changed = copy.deepcopy(self.manifest)
                changed["subgoals"][0]["outcome"]["ok"] = value
                self.assertFalse(self.check(changed)["ok"])

    def test_float_bool_or_different_windows_are_rejected(self):
        for frames in ([0., 2.], [False, 2], [1, 2], [0, 1]):
            with self.subTest(frames=frames):
                changed = copy.deepcopy(self.manifest)
                changed["subgoals"][0]["frames"] = frames
                self.assertFalse(self.check(changed)["ok"])

    def test_index_identity_is_exact_and_distinct_from_stage1_scope(self):
        for index in (True, 1., 2):
            with self.subTest(index=index):
                changed = copy.deepcopy(self.manifest)
                changed["subgoals"][0]["index"] = index
                self.assertFalse(self.check(changed)["ok"])
        # A split Stage 1 action may have a different executor entry number.
        self.executed["index"] = 7
        self.manifest["subgoals"][0]["index"] = 7
        self.assertTrue(self.check()["ok"])

    def test_release_flag_cannot_change_or_use_boolean_coercion(self):
        for release in (True, 0, "false"):
            with self.subTest(release=release):
                changed = copy.deepcopy(self.manifest)
                changed["subgoals"][0]["release"] = release
                self.assertFalse(self.check(changed)["ok"])

    def bimanual(self):
        actions = {side: {"motion": "carry", "object": obj, "release": False}
                   for side, obj in (("left", "object"), ("right", "other"))}
        self.executed.update(motion="both", side="both", object={s: a["object"] for s, a in actions.items()}, hands=actions)
        self.manifest["subgoals"][0] = {**copy.deepcopy(self.executed), "outcome": {"ok": True}}

    def test_bimanual_action_classes_and_release_are_bound_per_hand(self):
        self.bimanual()
        self.assertTrue(self.check()["ok"])
        for key, value in (("motion", "push"), ("object", "object"), ("release", True), ("side", "left")):
            with self.subTest(key=key):
                changed = copy.deepcopy(self.manifest)
                changed["subgoals"][0]["hands"]["right"][key] = value
                self.assertFalse(self.check(changed)["ok"])

    def test_legacy_both_report_without_hand_actions_is_unverified(self):
        self.bimanual()
        self.executed.pop("hands")
        report = self.check()
        self.assertFalse(report["ok"])
        self.assertIn("rerun this simultaneous entry", " ".join(report["issues"]))

    def test_together_motion_cannot_be_overridden_to_hide_per_hand_stress(self):
        self.bimanual()
        self.executed["motion"] = "lift_together"
        self.manifest["subgoals"][0]["motion"] = "lift_together"
        # Both nested entries still claim carry, which does not demand a
        # fresh stress test. The actual top-level lift must demand it.
        report = self.check()
        self.assertFalse(report["ok"])
        self.assertIn("stress", " ".join(report["issues"]))

    def test_real_bimanual_execution_persists_bound_hand_actions(self):
        from tool_library.skills.execute_subgoal_plan import execute_subgoal_plan
        entry = {"index": 1, "motion": "both", "hands": {
            side: {"motion": "wait", "duration_s": .2} for side in ("left", "right")}}
        output = self.ws / "physical_bimanual"
        scene = Path(__file__).resolve().parents[2] / "setup/scene.xml"
        result = execute_subgoal_plan(str(scene), {"subgoals": [entry]}, str(output), plan_arm_paths=False)
        self.assertTrue(result["ok"], result)
        self.assertFalse(result["dry_run"])
        saved = json.loads((output / "execution_report.json").read_text())
        executed = saved["subgoals"][0]
        self.assertEqual(executed["hands"], {
            side: {"motion": "wait", "object": None, "release": False} for side in ("left", "right")})
        self.assertIs(executed["release"], False)
        self.assertEqual(saved["trajectory_sha256"], file_digest(output / "trajectory.npz"))
        issues = []
        _bind_execution_identity({**entry, "frames": executed["frames"]}, executed, "physical both", issues)
        self.assertEqual(issues, [])

    def test_actual_selected_pose_cannot_disappear_from_carry_manifest(self):
        self.executed["selected_pose_evidence"] = {"path": "used.json", "sha256": "0" * 64}
        report = self.check()
        self.assertFalse(report["ok"])
        self.assertIn("pose binding", " ".join(report["issues"]))


if __name__ == "__main__":
    unittest.main()
