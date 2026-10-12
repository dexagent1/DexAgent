"""Search memory rejects repeated work, never certifies a task or blocks repairs."""
import errno
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.search_progress import SearchProgress
from agent.runner import StageRunner
from agent.config import load_config
from agent.llm import MockLLM

ROOT = Path(__file__).resolve().parents[2]


def failure(dry=False, distance=20.):
    f = dict(index=2, motion="carry", side="left", object="part", ok=False,
             failed=["arm_unreachable"], measured={"position_error_mm": distance})
    return dict(ok=True, result=dict(ok=False, dry_run=dry, n_frames=0 if dry else 30,
        subgoals=[dict(index=1, ok=True), f], first_failure=f))


class SearchProgressTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ws = Path(self.tmp.name)
        (self.ws / "object.xml").write_text('<mujoco><worldbody><geom name="part" type="box" size=".02 .02 .02"/></worldbody></mujoco>')
        (self.ws / "scene.xml").write_text('<mujoco><include file="object.xml"/></mujoco>')
        (self.ws / "pose.json").write_text(json.dumps({"pos": [0., 0., .2]}))
        (self.ws / "plan.json").write_text(json.dumps({"subgoals": [{"pose": "pose.json", "motion": "grasp_lift"}]}))
        self.args = dict(scene="scene.xml", plan="plan.json", out_dir="execution")
        self.p = SearchProgress(self.ws)

    def test_failed_duplicate_refused_but_changed_pose_and_external_include_allowed(self):
        key = self.p.execution_key(self.args)
        self.assertIsNotNone(key)
        for _ in range(2):
            self.assertIsNone(self.p.before_execution(key))
            self.p.observe_execution(key, self.args, failure())
        blocked = self.p.before_execution(key)
        self.assertFalse(blocked["execution_performed"])
        self.assertEqual(key, self.p.execution_key(dict(self.args, out_dir="new_output_name")))
        (self.ws / "pose.json").write_text(json.dumps({"pos": [0., .01, .2]}))
        changed = self.p.execution_key(self.args)
        self.assertNotEqual(key, changed)
        self.assertIsNone(self.p.before_execution(changed))
        (self.ws / "object.xml").write_text('<mujoco><worldbody><geom name="part" type="box" size=".03 .02 .02"/></worldbody></mujoco>')
        self.assertNotEqual(changed, self.p.execution_key(self.args))

    def test_resume_binds_actual_prefix_state_but_not_failed_diagnostic_noise(self):
        args = dict(self.args, resume=True)
        out = self.ws / "execution"; out.mkdir()
        seg = out / "subgoal_01"; seg.mkdir()
        state = seg / "state.json"; state.write_text('{"held": {}}')
        rows = [dict(index=1, tag="subgoal_01", ok=True, resume_signature="x"), dict(index=2, ok=False, elapsed_s=1)]
        (out / "outcomes.json").write_text(json.dumps(rows))
        key = self.p.execution_key(args)
        rows[-1]["elapsed_s"] = 10
        (out / "outcomes.json").write_text(json.dumps(rows))
        self.assertEqual(key, self.p.execution_key(args))
        state.write_text('{"held": {"left":"part"}}')
        self.assertNotEqual(key, self.p.execution_key(args))

    def test_nested_plan_tracks_pose_actually_selected_by_executor(self):
        from tool_library.skills.execute_subgoal_plan import _load_plan
        nested = self.ws / "nested"; nested.mkdir()
        (nested / "pose.json").write_text('{"pos":[0,0,0.9]}')
        (nested / "plan.json").write_text(json.dumps({"subgoals": [
            {"pose": "pose.json", "motion": "grasp_lift"}]}))
        args = dict(self.args, plan="nested/plan.json")
        old = Path.cwd()
        try:
            os.chdir(self.ws)
            chosen = _load_plan(args["plan"])[0][0]["pose"]
            self.assertEqual(Path(chosen).resolve(), self.ws / "pose.json")
        finally:
            os.chdir(old)
        key = self.p.execution_key(args)
        self.assertIsNotNone(key)
        for _ in range(2):
            self.p.observe_execution(key, args, failure())
        self.assertIsNotNone(self.p.before_execution(key))
        (self.ws / "pose.json").write_text('{"pos":[0,0,0.3]}')
        repaired = self.p.execution_key(args)
        self.assertNotEqual(key, repaired)
        self.assertIsNone(self.p.before_execution(repaired))

    def test_nested_candidate_list_tracks_workspace_pose_and_plan_fallback(self):
        from tool_library.skills.execute_subgoal_plan import _load_plan, _pose_candidates
        nested = self.ws / "nested"; nested.mkdir()
        (nested / "ranked.json").write_text('["pose.json"]')
        (nested / "plan.json").write_text(json.dumps({"subgoals": [
            {"pose": "ranked.json", "motion": "grasp_lift"}]}))
        args = dict(self.args, plan="nested/plan.json")
        old = Path.cwd()
        try:
            os.chdir(self.ws)
            entry = _load_plan(args["plan"])[0][0]
            self.assertEqual(Path(entry["pose"]).resolve(), nested / "ranked.json")
            self.assertEqual(_pose_candidates(entry), ["pose.json"])
        finally:
            os.chdir(old)
        key = self.p.execution_key(args)
        self.assertIsNotNone(key)
        (self.ws / "pose.json").write_text('{"pos":[0,0,0.4]}')
        self.assertNotEqual(key, self.p.execution_key(args))

    def test_plan_relative_pose_fallback_remains_bound(self):
        from tool_library.skills.execute_subgoal_plan import _load_plan
        nested = self.ws / "nested"; nested.mkdir()
        (nested / "local_pose.json").write_text('{"pos":[0,0,0.2]}')
        (nested / "plan.json").write_text(json.dumps({"subgoals": [
            {"pose": "local_pose.json", "motion": "grasp_lift"}]}))
        args = dict(self.args, plan="nested/plan.json")
        old = Path.cwd()
        try:
            os.chdir(self.ws)
            chosen = _load_plan(args["plan"])[0][0]["pose"]
            self.assertEqual(Path(chosen).resolve(), nested / "local_pose.json")
        finally:
            os.chdir(old)
        key = self.p.execution_key(args)
        self.assertIsNotNone(key)
        (nested / "local_pose.json").write_text('{"pos":[0,0,0.3]}')
        self.assertNotEqual(key, self.p.execution_key(args))

    def test_resume_binds_selected_pose_reused_by_later_bites(self):
        out = self.ws / "execution"; out.mkdir()
        segment = out / "subgoal_01_01"; segment.mkdir()
        (out / "outcomes.json").write_text(json.dumps([
            dict(index=1, step=1, tag=segment.name, ok=True, resume_signature="x")]))
        pose = segment / "pose_used.json"
        pose.write_text('{"pos":[0,0,0.2]}')
        args = dict(self.args, resume=True)
        key = self.p.execution_key(args)
        self.assertIsNotNone(key)
        for _ in range(2):
            self.p.observe_execution(key, args, failure())
        pose.write_text('{"pos":[0,0,0.3]}')
        changed = self.p.execution_key(args)
        self.assertNotEqual(key, changed)
        self.assertIsNone(self.p.before_execution(changed))

    def test_dry_run_and_local_success_do_not_claim_task_acceptance(self):
        for _ in range(3):
            report = self.p.observe_execution(None, dict(self.args, dry_run=True), failure(True))
        self.assertEqual(self.p.best_physical_prefix, 0)
        self.assertIn("different", report["next_step"])
        report = self.p.observe_execution(None, self.args, failure(False, 10.))
        self.assertTrue(report["measurements_changed"])
        self.assertEqual(self.p.best_physical_prefix, 1)
        success = dict(ok=True, result=dict(ok=True, n_frames=60, subgoals=[dict(ok=True), dict(ok=True)]))
        self.p.observe_execution("success", self.args, success)
        self.assertIsNone(self.p.before_execution("success"))
        written = json.loads(self.p.path.read_text())
        self.assertFalse(written["events"][-1]["task_accepted"])
        self.assertEqual(written["authority"], "diagnostic_only")
        self.assertFalse(written["convergence_guaranteed"])
        self.assertIn("not semantic progress", written["prefix_scope"])

    def test_existing_progress_file_cannot_seed_trust_or_refuse_new_session(self):
        key = self.p.execution_key(self.args)
        for _ in range(2):
            self.p.observe_execution(key, self.args, failure())
        newer = SearchProgress(self.ws)
        self.assertIsNone(newer.before_execution(key))
        self.assertEqual(newer.best_physical_prefix, 0)

    def test_malformed_resume_metadata_defers_to_real_executor(self):
        out = self.ws / "execution"; out.mkdir()
        for malformed in ({}, [None], "invalid JSON", [{"ok": True, "tag": [], "resume_signature": "x"}]):
            (out / "outcomes.json").write_text(json.dumps(malformed))
            self.assertIsNone(self.p.execution_key(dict(self.args, resume=True)))

    def test_nonfinite_failure_measurements_preserve_feedback(self):
        raw = failure(distance=float('inf'))
        raw['result']['first_failure']['measured']['solver_cost'] = float('nan')
        value = self.p.observe_execution(None, self.args, raw)
        self.assertEqual(value['failure']['failed'], ['arm_unreachable'])
        self.assertFalse(value['task_accepted'])

    def test_runner_suppresses_actual_tool_dispatch_on_third_identical_failure(self):
        cfg = load_config(ROOT / "dexagent.yaml")
        runner = StageRunner(cfg, MockLLM([]), self.ws, "test", [], stage="stage3")
        with patch("agent.runner.call_tool", side_effect=lambda *a, **kw: failure()) as call:
            runner.run_tool("execute_subgoal_plan", self.args)
            runner.run_tool("execute_subgoal_plan", self.args)
            blocks, result = runner.run_tool("execute_subgoal_plan", self.args)
            self.assertEqual(call.call_count, 2)
            self.assertFalse(result["execution_performed"])
            self.assertIn("library_search", blocks[0]["text"])
            (self.ws / "pose.json").write_text('{"pos":[0,0.01,0.2]}')
            runner.run_tool("execute_subgoal_plan", self.args)
            self.assertEqual(call.call_count, 3)
        self.assertEqual(cfg["_executor_receipts"], {})

    def test_progress_io_error_cannot_suppress_dispatch_or_replace_feedback(self):
        cfg = load_config(ROOT / "dexagent.yaml")
        runner = StageRunner(cfg, MockLLM([]), self.ws, "test", [], stage="stage3")
        raw = failure()
        with patch("agent.runner.call_tool", return_value=raw) as call:
            runner.run_tool("execute_subgoal_plan", self.args)
            runner.run_tool("execute_subgoal_plan", self.args)
            with patch("agent.external.atomic_json", side_effect=OSError("diagnostic disk unavailable")):
                blocks, result = runner.run_tool("execute_subgoal_plan", self.args)
                runner.search_progress.observe_gate({"ok": True})
            self.assertEqual(call.call_count, 3)
            self.assertIs(result, raw)
            self.assertEqual(json.loads(blocks[0]["text"])["result"]["first_failure"]["failed"],
                             ["arm_unreachable"])
        self.assertEqual(cfg["_executor_receipts"], {})

    def test_progress_serialization_error_cannot_replace_feedback(self):
        cfg = load_config(ROOT / "dexagent.yaml")
        runner = StageRunner(cfg, MockLLM([]), self.ws, "test", [], stage="stage3")
        runner.search_progress.events.append({"malformed_diagnostic": object()})
        raw = failure()
        with patch("agent.runner.call_tool", return_value=raw) as call:
            blocks, result = runner.run_tool("execute_subgoal_plan", self.args)
            runner.search_progress.observe_gate({"ok": True})
        self.assertEqual(call.call_count, 1)
        self.assertIs(result, raw)
        self.assertEqual(json.loads(blocks[0]["text"])["result"]["first_failure"]["failed"],
                         ["arm_unreachable"])

    def test_unreadable_nested_input_disables_suppression(self):
        original = Path.read_bytes
        def read(path):
            if path.name == "pose.json":
                raise OSError("cannot establish actual pose identity")
            return original(path)
        with patch.object(Path, "read_bytes", read):
            self.assertIsNone(self.p.execution_key(self.args))

    def test_nested_long_prose_keeps_real_dependency_hashing_and_comparison(self):
        prose = "A generic scene description with relational observations and object properties. " * 100
        (self.ws / "description.json").write_text(json.dumps({"observations": [prose], "pose": "pose.json"}))
        plan = json.loads((self.ws / "plan.json").read_text())
        plan["scene_recon_json"] = "description.json"
        (self.ws / "plan.json").write_text(json.dumps(plan))
        original = self.p.execution_key(self.args)
        scope = self.p._local_scope(self.args)
        self.assertIsNotNone(original)
        self.assertIsNotNone(scope)
        for _ in range(2):
            self.p.observe_execution(original, self.args, failure())
        self.assertFalse(self.p.before_execution(original)["execution_performed"])
        (self.ws / "pose.json").write_text('{"pos":[0,0.01,0.2]}')
        pose_changed = self.p.execution_key(self.args)
        self.assertNotEqual(original, pose_changed)
        self.assertIsNone(self.p.before_execution(pose_changed))
        (self.ws / "object.xml").write_text('<mujoco><worldbody><geom name="part" type="box" size=".03 .02 .02"/></worldbody></mujoco>')
        scene_changed = self.p.execution_key(self.args)
        self.assertNotEqual(pose_changed, scene_changed)
        self.assertNotEqual(scope, self.p._local_scope(self.args))
        (self.ws / "description.json").write_text(json.dumps({"observations": [prose + " changed"], "pose": "pose.json"}))
        prose_changed = self.p.execution_key(self.args)
        self.assertIsNotNone(prose_changed)
        self.assertNotEqual(scene_changed, prose_changed)

    def test_path_permission_error_still_disables_suppression_and_comparison(self):
        original = Path.is_file
        def denied(path):
            if path.name == "pose.json":
                raise PermissionError(errno.EACCES, "dependency is unreadable", str(path))
            return original(path)
        # hand_scene remains part of the comparison identity, unlike a pose
        # proposal; both the exact guard and scope must fail conservatively.
        args = dict(self.args, hand_scene="pose.json")
        with patch.object(Path, "is_file", denied):
            self.assertIsNone(self.p.execution_key(args))
            self.assertIsNone(self.p._local_scope(args))


if __name__ == "__main__":
    unittest.main()
