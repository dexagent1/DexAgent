"""Measured search guidance is conservative, advisory, and never release authority."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.config import load_config
from agent.llm import MockLLM
from agent.runner import StageRunner
from agent.search_progress import SearchProgress

ROOT = Path(__file__).resolve().parents[2]


def execution(distance=20., prefix=1, failed=None, measured=None, success=False):
    failure = {"index": prefix + 1, "motion": "carry", "side": "left", "object": "part", "ok": False,
               "failed": failed or ["arm_unreachable"], "measured": measured or {"position_error_mm": distance}}
    rows = [{"index": i + 1, "tag": "subgoal_%02d" % (i + 1), "ok": True} for i in range(prefix)]
    return {"ok": True, "result": {"ok": success, "n_frames": 10,
            "subgoals": rows if success else rows + [failure], "first_failure": None if success else failure}}


def gate(passed=(), scope="fixed", unsupported=False):
    def goal(index, ok):
        return {"index": index, "goal": "original goal " + str(index), "ok": ok,
                "required_classes": ["place"], "predicates": [] if unsupported else [
                    {"class": "place", "A": "part", "B": "support", "ok": ok, "measured": {"error_m": .01}}]}
    report = {"subgoals": [goal(i, str(i) in passed) for i in (1, 2)],
              "task_success": goal("final_task", "final_task" in passed)}
    suite = {"ok": False, "comparison_scope": scope, "task_goal_report": report,
             "candidate_binding": {"trajectory": str(sorted(passed))},
             "diagnostic_report": {"path": "gate.json", "sha256": "diagnostic", "authority": "diagnostic_only"}}
    return {"ok": False, "issues": ["independent original goal failed"], "reports": {"stage3_gate_suite": suite}}


class SearchComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ws = Path(self.tmp.name)
        (self.ws / "scene.xml").write_text('<mujoco><worldbody><geom type="box" size=".02 .02 .02"/></worldbody></mujoco>')
        (self.ws / "pose.json").write_text('{"pos":[0,0,.2]}'.replace('.2', '0.2'))
        self.plan = {"subgoals": [{"motion": "grasp_lift", "object": "part", "pose": "pose.json"},
                                  {"motion": "carry", "object": "part", "target": [0., .1, .2]}]}
        self.write_plan()
        self.args = {"scene": "scene.xml", "plan": "plan.json", "out_dir": "execution"}
        self.p = SearchProgress(self.ws)

    def write_plan(self):
        (self.ws / "plan.json").write_text(json.dumps(self.plan))

    def observe(self, raw=None, args=None):
        args = args or self.args
        return self.p.observe_execution(self.p.execution_key(args), args, raw or execution())

    def artifacts(self, raw, contents=b"original physical recording"):
        out = self.ws / "execution"
        out.mkdir(exist_ok=True)
        rows = []
        for row in raw["result"]["subgoals"]:
            if not row["ok"]:
                break
            record = dict(row, resume_signature="bound-inputs", checks={"rise": True})
            rows.append(record)
            folder = out / row["tag"]
            folder.mkdir(exist_ok=True)
            for name in ("rollout.npz", "rollout_runtime.npz"):
                (folder / name).write_bytes(contents)
            for name, value in (("outcome.json", record), ("state.json", {"held": "part"}), ("phases.json", [])):
                (folder / name).write_text(json.dumps(value))
        (out / "outcomes.json").write_text(json.dumps(rows))
        (out / "scene_physics.xml").write_text((self.ws / "scene.xml").read_text())

    def test_measured_improvement_regression_stagnation_and_changed_args(self):
        first = self.observe()
        self.assertEqual(first["comparison"]["status"], "baseline")
        (self.ws / "pose.json").write_text('{"pos":[0,0.01,0.2]}')
        better = self.observe(execution(10.))
        self.assertEqual(better["comparison"]["status"], "improved_measurements")
        for distance in (15., 20.):
            worse = self.observe(execution(distance))
            self.assertEqual(worse["comparison"]["status"], "regressed_measurements")
            self.assertEqual(worse["comparison"]["best"]["metrics"], {"position_error_mm": 10.})
        self.assertEqual(worse["comparison"]["attempts_since_supported_improvement"], 2)
        self.assertIn("Exploratory worse", worse["comparison"]["next_step"])
        (self.ws / "pose.json").write_text('{"pos":[0,0.02,0.2]}')
        same = self.observe(execution(10.))
        self.assertEqual(same["comparison"]["status"], "unchanged")
        self.assertFalse(same["comparison"]["semantic_progress"])

    def test_actual_reach_schema_compares_per_side_without_fabricating_physics(self):
        def reach(mm=20., deg=10., jumps=2, side="left"):
            result = execution(prefix=0, measured={"infeasible": [
                {"side": side, "frames": [0, 10], "label": "reach", "worst_mm": mm, "worst_deg": deg},
                {"side": side, "frames": [10, 20], "label": "close", "worst_mm": mm / 2, "worst_deg": deg / 2}],
                "branch_jumps": {side: [{}] * jumps}})
            result["result"].pop("n_frames")
            return result
        baseline = self.observe(reach())["comparison"]
        better = self.observe(reach(mm=10., deg=5., jumps=1))
        self.assertEqual(better["comparison"]["status"], "improved_measurements")
        self.assertEqual(better["comparison"]["evidence_kind"], "kinematic_reach_screen")
        self.assertFalse(better["physical_recording"])
        self.assertFalse(better["task_accepted"])
        self.assertIsNone(better["comparison"]["retained_prefix"])
        self.assertEqual(better["comparison"]["best"]["metrics"]["reach:left:branch_jump_count"], 1)
        mixed = self.observe(reach(mm=5., deg=15., jumps=1))["comparison"]
        self.assertEqual(mixed["status"], "mixed")
        changed_side = self.observe(reach(side="right"))["comparison"]
        self.assertEqual(changed_side["status"], "unsupported_measurements")
        missing = reach()
        del missing["result"]["first_failure"]["measured"]["infeasible"][0]["worst_deg"]
        self.assertEqual(self.observe(missing)["comparison"]["status"], "incomparable")
        nonfinite = reach(mm=float("inf"))
        self.assertEqual(self.observe(nonfinite)["comparison"]["status"], "incomparable")
        physical = reach()
        physical["result"]["n_frames"] = 20
        self.assertNotEqual(baseline["scope"], self.observe(physical)["comparison"]["scope"])

    def test_subdividing_plan_and_changing_targets_or_physics_is_not_progress(self):
        self.observe()
        self.plan["subgoals"].insert(0, {"motion": "wait"})
        self.write_plan()
        divided = self.observe(execution(1., prefix=2))["comparison"]
        self.assertEqual(divided["status"], "baseline")
        self.assertIn("no improvement", divided["reason"])
        self.plan["subgoals"][-1]["target"] = [0., .01, .2]
        self.write_plan()
        changed = self.observe(execution(0., prefix=3))["comparison"]
        self.assertNotEqual(divided["scope"], changed["scope"])
        self.assertEqual(changed["status"], "baseline")
        (self.ws / "scene.xml").write_text('<mujoco><worldbody><geom type="box" size=".01 .01 .01"/></worldbody></mujoco>')
        physics = self.observe()["comparison"]
        self.assertNotEqual(changed["scope"], physics["scope"])
        self.assertEqual(physics["status"], "baseline")

    def test_mixed_or_unsupported_and_nonfinite_measurements_never_improve(self):
        original = execution(failed=["servo_tracking"], measured={"max_track_pos_mm": 20., "max_track_rot_deg": 10.})
        self.artifacts(original)
        retained = self.observe(original)["comparison"]["retained_prefix"]
        mixed = self.observe(execution(failed=["servo_tracking"], measured={"max_track_pos_mm": 10., "max_track_rot_deg": 20.}))
        self.assertEqual(mixed["comparison"]["status"], "mixed")
        self.assertEqual(mixed["comparison"]["best"]["metrics"]["max_track_pos_mm"], 20.)
        self.assertEqual(mixed["comparison"]["retained_prefix"], retained)
        for _ in range(2):
            unsupported = self.observe(execution(failed=["unknown_check"], measured={"score": 9000}))
        self.assertEqual(unsupported["comparison"]["status"], "different_failure")
        self.assertEqual(unsupported["comparison"]["retained_prefix"], retained)
        p = SearchProgress(self.ws)
        for _ in range(2):
            report = p.observe_execution(None, self.args, execution(float('inf')))
        self.assertEqual(report["comparison"]["status"], "unsupported_measurements")
        self.assertFalse(report["task_accepted"])

    def test_physical_prefix_archive_survives_regression_and_is_not_authority(self):
        good = execution(10.)
        self.artifacts(good)
        archived = self.observe(good)["comparison"]["retained_prefix"]
        path = Path(archived["path"])
        self.artifacts(execution(30.), contents=b"regressed attempt")
        worse = self.observe(execution(30.))["comparison"]
        self.assertEqual(worse["retained_prefix"]["path"], str(path))
        self.assertEqual((path / "subgoal_01" / "rollout_runtime.npz").read_bytes(), b"original physical recording")
        self.assertEqual(json.loads((path / "outcomes.json").read_text())[0]["resume_signature"], "bound-inputs")
        self.assertEqual(json.loads((path / "plan_original.json").read_text()), self.plan)
        self.assertEqual(archived["authority"], "unverified_diagnostic_proposals")
        restarted = SearchProgress(self.ws)
        self.assertEqual(restarted.local_scopes, {})
        self.assertIsNone(restarted.before_execution(self.p.execution_key(self.args)))

    def test_missing_recording_never_creates_fake_prefix_and_dry_run_does_not_compare(self):
        report = self.observe()["comparison"]
        self.assertIsNone(report["retained_prefix"])
        self.assertIn("unavailable", report["retention_attempt"])
        dry = self.observe(args=dict(self.args, dry_run=True))["comparison"]
        self.assertEqual(dry["status"], "incomparable")
        self.assertFalse((self.ws / ".search_prefixes").exists())

    def test_scope_preserves_checks_but_accepts_pose_proposal_changes(self):
        first = self.observe()["comparison"]["scope"]
        self.plan["subgoals"][0]["pose"] = {"pos": [0., 0., .3]}
        self.write_plan()
        self.assertEqual(first, self.observe()["comparison"]["scope"])
        self.plan["subgoals"][0]["expect"] = {"rise_m": .001}
        self.write_plan()
        self.assertNotEqual(first, self.observe()["comparison"]["scope"])

    def test_changed_trusted_goal_contract_cannot_claim_comparable_progress(self):
        from types import SimpleNamespace
        contract = SimpleNamespace(hashes={"semantic": "original"}, models={}, check=lambda: {"ok": True})
        self.p.bind_contract(contract)
        self.observe()
        contract.check = lambda: {"ok": False}
        result = self.observe(execution(0.))["comparison"]
        self.assertEqual(result["status"], "incomparable")

    def test_repeated_micro_subdivision_cannot_improve_original_goals(self):
        self.p.observe_gate(gate(["1"]))
        self.observe()
        for n in (4, 8, 16):
            self.plan["subgoals"] = [{"motion": "wait", "stage1_indices": [1]} for _ in range(n)]
            self.write_plan()
            result = self.observe(execution(prefix=n, success=True))
            self.assertEqual(result["comparison"]["status"], "baseline")
            self.assertFalse(result["comparison"]["semantic_progress"])
            verdict = self.p.observe_gate(gate(["1"]))
            self.assertEqual(verdict["semantic_comparison"]["status"], "unchanged")
            self.assertFalse(verdict["task_accepted"])

    def test_independent_goals_compare_sets_not_counts_and_final_goal_matters(self):
        self.p.observe_gate(gate(["1"]))
        mixed = self.p.observe_gate(gate(["2"]))["semantic_comparison"]
        self.assertEqual(mixed["status"], "mixed_goals")
        self.assertEqual(mixed["best"]["passed_goals"], ["1"])
        improved = self.p.observe_gate(gate(["1", "2"]))["semantic_comparison"]
        self.assertEqual(improved["status"], "improved_goals")
        self.assertEqual(improved["latest"]["failed_goals"], ["final_task"])
        changed = self.p.observe_gate(gate(["1", "2", "final_task"], scope="changed_physics"))["semantic_comparison"]
        self.assertEqual(changed["status"], "baseline")
        failed = self.p.observe_gate(gate(["1"]))
        self.assertEqual(failed["semantic_comparison"]["status"], "regressed_goals")
        self.assertFalse(failed["task_accepted"])

    def test_local_success_does_not_hide_final_failure_or_unsupported_goals(self):
        self.observe(execution(success=True))
        event = self.p.observe_gate(gate(unsupported=True))
        self.assertTrue(event["latest_local_execution_ok"])
        self.assertFalse(event["task_accepted"])
        self.assertEqual(event["semantic_comparison"]["status"], "unsupported_goal_measurements")
        self.assertIn("final task gates failed", event["next_step"])
        latest = self.observe()
        self.assertEqual(latest["latest_final_gate"]["issues"], ["independent original goal failed"])

    def test_raw_tool_error_is_visible_without_poisoning_physical_comparison(self):
        runner = StageRunner(load_config(ROOT / "dexagent.yaml"), MockLLM([]), self.ws, "test", [], stage="stage3")
        raw = {"ok": False, "error": "dependency unavailable"}
        with patch("agent.runner.call_tool", return_value=raw) as dispatch:
            for _ in range(3):
                blocks, result = runner.run_tool("execute_subgoal_plan", self.args)
        self.assertIs(result, raw)
        self.assertEqual(dispatch.call_count, 3)
        self.assertIn("dependency unavailable", blocks[0]["text"])
        self.assertIn("not physical infeasibility", blocks[-1]["text"])
        self.assertEqual(runner.search_progress.local_scopes, {})
        self.assertEqual(runner.cfg["_executor_receipts"], {})

    def test_failed_finish_sends_current_gate_measurements_and_comparison_to_model(self):
        class Recorder(MockLLM):
            def complete(self, system, messages, tools):
                self.messages = copy.deepcopy(messages)
                return super().complete(system, messages, tools)
        llm = Recorder([[{"type": "tool_use", "id": "f1", "name": "finish", "input": {}}],
                        [{"type": "tool_use", "id": "f2", "name": "finish", "input": {}}]])
        runner = StageRunner(load_config(ROOT / "dexagent.yaml"), llm, self.ws, "test", [], stage="stage3", max_rounds=2)
        verdict = gate(["1"])
        verdict["reports"]["stage3_gate_suite"]["failed_checks"] = {
            "stage1_goal_gate": {"ok": False, "measured": {"target_error_m": .42}}}
        with patch.object(runner, "_verify", return_value=verdict):
            out = runner.run("test generic search feedback")
        model_feedback = json.dumps(llm.messages[-1])
        self.assertIn("target_error_m", model_feedback)
        self.assertIn("0.42", model_feedback)
        self.assertIn("semantic_comparison", model_feedback)
        self.assertIn("final_task", model_feedback)
        self.assertFalse(out["finished"])
        self.assertEqual(runner.cfg["_executor_receipts"], {})


if __name__ == "__main__":
    unittest.main()
