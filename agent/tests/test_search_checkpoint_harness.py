"""The harness makes direct searches resumable without granting cached authority."""
import copy
import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault("DEXAGENT_QUIET", "1")
from agent.llm import MockLLM
from agent.runner import StageRunner

M = importlib.import_module("tool_library.skills.optimize_grasp_pose")


class SearchCheckpointHarnessTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ws = self.root / "runs" / "example" / "stage3" / ".candidates" / "candidate"
        self.lib = self.root / "framework" / "tool_library"
        (self.lib / "skills").mkdir(parents=True)
        (self.lib / "skills" / "optimize_grasp_pose.py").write_text("# orchestration fixture\n")
        (self.lib / "index.json").write_text(json.dumps({"tools": [M.TOOL]}))
        self.cfg = {"inputs": {"library": str(self.lib)}, "agent": {"max_rounds": 5},
                    "run": {"name": "example", "output_dir": str(self.root / "runs")}}
        self.args = {"hand_scene": {"objects": []}, "family": "seeds",
                     "starts": {"pos": [0, 0, .1], "euler": [0, 0, 0]}, "workers": 1,
                     "max_evals": 10, "iterations": 3, "population": 4, "tests": ["grasp_lift_test"]}

    def runner(self, stage="stage3", **kw):
        return StageRunner(self.cfg, kw.pop("llm", MockLLM([])), self.ws, "test", [], stage=stage, **kw)

    def test_default_paths_unique_explicit_parameters_and_config_defaults_preserved(self):
        self.cfg["compute"] = {"workers": 3, "launcher": ["wrapper", "--cpu"]}
        runner = self.runner()
        args = dict(self.args, max_overlap_m=.004, lift_heights_m=[.11, .17], test_args={"grasp_lift_test": {"shake": False}})
        original = copy.deepcopy(args)
        with patch("agent.runner.call_tool", return_value={"ok": True, "result": {"ok": False}}) as call:
            blocks, first = runner.run_tool("optimize_grasp_pose", args)
            _, second = runner.run_tool("optimize_grasp_pose", args)
        self.assertEqual(args, original)
        effective = first["effective_args"]
        for key, value in args.items():
            self.assertEqual(effective[key], value)
        self.assertEqual(effective["workers"], 1)
        self.assertEqual(effective["launcher"], ["wrapper", "--cpu"])
        self.assertEqual(Path(effective["checkpoint_json"]).parent, self.ws / ".search_checkpoints")
        self.assertEqual(effective["stop_file"], str(self.ws / ".stop_search"))
        self.assertNotEqual(effective["checkpoint_json"], second["effective_args"]["checkpoint_json"])
        self.assertEqual(call.call_args_list[0].args[3], effective)
        self.assertIn(effective["checkpoint_json"], blocks[-1]["text"])
        rows = [json.loads(row) for row in runner.trace.read_text().splitlines()]
        self.assertEqual(rows[0]["event"], "tool_dispatch")
        self.assertEqual(rows[0]["effective_args"], effective)
        self.assertEqual(rows[0]["args"], original)

    def test_explicit_paths_nulls_and_missing_resume_path_are_preserved(self):
        runner = self.runner()
        variants = [dict(checkpoint_json="saved.json", stop_file="stop", resume=True, checkpoint_batch_size=1),
                    dict(checkpoint_json=None, stop_file=None), dict(resume=True)]
        with patch("agent.runner.call_tool", return_value={"ok": False, "error": "normal tool validation"}) as call:
            for variant in variants:
                _, raw = runner.run_tool("optimize_grasp_pose", dict(self.args, **variant))
                actual = call.call_args.args[3]
                for key, value in variant.items():
                    self.assertEqual(actual[key], value)
                if variant == {"resume": True}:
                    self.assertNotIn("checkpoint_json", actual)
                    self.assertNotIn("stop_file", actual)
                self.assertFalse(raw["ok"])

    def test_legacy_schema_other_stages_optout_and_run_python_are_unchanged(self):
        for variation in ("legacy", "stage2", "optout"):
            with self.subTest(variation=variation):
                self.cfg["agent"]["cooperative_search_checkpoints"] = variation != "optout"
                runner = self.runner(stage="stage2" if variation == "stage2" else "stage3")
                if variation == "legacy":
                    runner.index = copy.deepcopy(runner.index)
                    runner.index["tools"][0]["inputs"] = [i for i in M.TOOL["inputs"] if i["name"] not in
                        ("checkpoint_json", "resume", "stop_file", "checkpoint_batch_size")]
                with patch("agent.runner.call_tool", return_value={"ok": True, "result": {}}) as call:
                    _, raw = runner.run_tool("optimize_grasp_pose", self.args)
                self.assertEqual(call.call_args.args[3], self.args)
                self.assertNotIn("search_checkpoint", raw)
        runner = self.runner()
        with patch.object(runner.dev, "run_python", return_value={"ok": True, "stdout": ""}) as run:
            runner.run_tool("run_python", {"code": "# explicit imports remain caller controlled"})
        run.assert_called_once_with(code="# explicit imports remain caller controlled")

    def test_real_dispatch_stops_without_next_turn_verify_finalize_or_receipt(self):
        calls = [[{"type": "tool_use", "name": "optimize_grasp_pose", "input": self.args},
                  {"type": "tool_use", "name": "finish", "input": {}}]]
        llm = MockLLM(calls)
        finalize = Mock(side_effect=AssertionError("stop cannot finalize"))
        runner = self.runner(llm=llm, finalize=finalize)
        marker = self.ws / ".stop_search"
        def evaluate(job):
            marker.touch()
            return {"ok": False, "score": -1, "index": job["index"], "start": job["start_index"],
                    "theta": job["theta"], "stage": "tests", "reason": "test:failed"}
        with patch("agent.library.call_tool_inprocess", side_effect=lambda lib, tool, args: M.optimize_grasp_pose(**args)), \
                patch.object(M, "_checkpoint_binding", return_value={"test": "binding"}), \
                patch.object(M, "evaluate", side_effect=evaluate), \
                patch.object(runner, "_verify", side_effect=AssertionError("stop cannot verify")) as verify, \
                patch.object(llm, "complete", wraps=llm.complete) as complete:
            out = runner.run("generic orchestration test")
        self.assertFalse(out["finished"])
        self.assertTrue(out["stopped"])
        self.assertEqual(out["stop_reason"], "cooperative_search_checkpoint")
        self.assertEqual(out["rounds"], 1)
        self.assertEqual(complete.call_count, 1)
        verify.assert_not_called()
        finalize.assert_not_called()
        self.assertEqual(self.cfg["_executor_receipts"], {})
        checkpoint = out["search_checkpoint"]
        self.assertTrue(Path(checkpoint["checkpoint_json"]).is_file())
        self.assertEqual(checkpoint["n_evals"], 1)
        rows = [json.loads(row) for row in runner.trace.read_text().splitlines()]
        tool = next(row for row in rows if row["event"] == "tool")
        self.assertEqual(tool["effective_args"], checkpoint["effective_args"])
        self.assertEqual(tool["args"], self.args)
        self.assertFalse(tool["result"]["result"]["ok"])
        self.assertEqual(tool["result"]["result"]["passing"], [])
        failure = json.loads((self.ws / "failure_report.json").read_text())
        self.assertTrue(failure["stopped"])
        self.assertNotIn("diagnosis", failure)

    def test_explicit_resume_missing_path_gets_real_validation_error(self):
        runner = self.runner()
        with patch("agent.library.call_tool_inprocess", side_effect=lambda lib, tool, args: M.optimize_grasp_pose(**args)):
            _, result = runner.run_tool("optimize_grasp_pose", dict(self.args, resume=True))
        self.assertFalse(result["ok"])
        self.assertIn("require checkpoint_json", result["error"])
        self.assertFalse((self.ws / ".search_checkpoints").exists())

    def test_fresh_harness_does_not_restore_checkpoint_or_contract_authority(self):
        upstream = self.root / "source.txt"
        upstream.write_text("trusted upstream")
        self.cfg["_verification_inputs"] = [str(upstream)]
        self.cfg["_executor_receipts"] = {"forged": "stale report"}
        self.ws.mkdir(parents=True)
        (self.ws / "input_contract.json").write_text('{"sha256": {}}')
        journal = self.ws / ".search_checkpoints" / "old.json"
        journal.parent.mkdir()
        journal.write_text('{"status": "complete", "ok": true, "executor_receipts": {"fake": true}}')
        runner = self.runner()
        self.assertTrue(runner.contract.check()["ok"])
        self.assertEqual(self.cfg["_executor_receipts"], {})
        self.assertIn(str(upstream), json.loads((self.ws / "input_contract.json").read_text())["sha256"])
        upstream.write_text("changed upstream")
        self.assertFalse(runner.contract.check()["ok"])


if __name__ == "__main__":
    unittest.main()
