"""Missing Stage 3 tool calls are protocol feedback, never implicit acceptance."""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault("DEXAGENT_QUIET", "1")
from agent.llm import MockLLM
from agent.runner import StageRunner


def text_reply():
    return [{"type": "text", "text": "I do not see tools."}]


def call(name, args=None):
    return [{"type": "tool_use", "name": name, "input": args or {}}]


class RecordingLLM(MockLLM):
    def __init__(self, turns):
        super().__init__(turns)
        self.messages = []
        self.closed = []

    def complete(self, system, messages, tools):
        self.messages.append(copy.deepcopy(messages))
        return super().complete(system, messages, tools)

    def close(self, result):
        self.closed.append(copy.deepcopy(result))


class NoToolProgressTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.lib = self.root / "tool_library"
        self.lib.mkdir()
        (self.lib / "index.json").write_text('{"tools":[]}')
        self.ws = self.root / "candidate"
        self.cfg = {"inputs": {"library": str(self.lib)}, "agent": {"max_rounds": 12}}

    def runner(self, turns, stage="stage3", **kwargs):
        self.llm = RecordingLLM(turns)
        return StageRunner(self.cfg, self.llm, self.ws, "generic protocol test", [], stage=stage, **kwargs)

    def assert_no_orphan_results(self, messages):
        emitted = set()
        for message in messages:
            for block in message["content"]:
                if block["type"] == "tool_use":
                    emitted.add(block["id"])
                elif block["type"] == "tool_result":
                    self.assertIn(block["tool_use_id"], emitted)

    def test_one_text_reply_recovers_with_real_call_and_explicit_finish(self):
        runner = self.runner([text_reply(), call("library_search", {"query": "generic skill"}), call("finish")])
        with patch.object(runner, "_verify", return_value={"ok": True, "issues": [], "reports": {}}) as verify, \
                patch.object(runner.dev, "library_search", return_value={"tools": []}) as search:
            result = runner.run("run generic task")
        self.assertTrue(result["finished"])
        self.assertNotIn("stopped", result)
        self.assertEqual(result["rounds"], 3)
        verify.assert_called_once()
        search.assert_called_once_with(query="generic skill")
        second_turn = self.llm.messages[1]
        self.assertEqual(second_turn[-1]["content"][0]["type"], "text")
        self.assertIn("Tools remain available", second_turn[-1]["content"][0]["text"])
        self.assert_no_orphan_results(self.llm.messages[-1])
        self.assertEqual(len(self.llm.closed), 1)

    def test_repeated_text_stops_normally_without_verification_or_acceptance(self):
        finalize = Mock(side_effect=AssertionError("text cannot publish"))
        runner = self.runner([text_reply()] * 6, finalize=finalize)
        (self.ws / "proposal.txt").write_text("unfinished useful candidate")
        with patch.object(runner, "_verify", side_effect=AssertionError("text cannot verify")) as verify:
            result = runner.run("run generic task")
        self.assertFalse(result["finished"])
        self.assertTrue(result["stopped"])
        self.assertEqual(result["stop_reason"], "backend_no_tool_progress")
        self.assertEqual(result["rounds"], 3)
        self.assertIsNone(result["verdict"])
        self.assertEqual(len(self.llm.messages), 3)
        verify.assert_not_called()
        finalize.assert_not_called()
        self.assertEqual(self.cfg["_executor_receipts"], {})
        self.assertEqual(self.llm.closed, [result])
        checkpoint = json.loads(Path(result["backend_checkpoint"]["path"]).read_text())
        self.assertIn("finish", checkpoint["available_tool_names"])
        self.assertEqual(checkpoint["consecutive_no_tool_turns"], 3)
        self.assertIn("unverified", checkpoint["authority"])
        self.assertEqual((self.ws / "proposal.txt").read_text(), "unfinished useful candidate")
        failure = json.loads((self.ws / "failure_report.json").read_text())
        self.assertEqual(failure["stop_reason"], "backend_no_tool_progress")
        self.assertNotIn("diagnosis", failure)
        self.assertIn("feasibility is unassessed", failure["reason"])
        self.assert_no_orphan_results(self.llm.messages[-1])
        events = [json.loads(line) for line in runner.trace.read_text().splitlines()]
        self.assertEqual(sum(row["event"] == "backend_no_tool_action" for row in events), 3)
        self.assertFalse(any(row["event"] == "verify" for row in events))

    def test_actual_call_resets_consecutive_no_action_count(self):
        runner = self.runner([text_reply(), text_reply(), call("library_search", {"query": "skill"}),
                              text_reply(), text_reply(), call("finish")])
        with patch.object(runner.dev, "library_search", return_value={"tools": []}), \
                patch.object(runner, "_verify", return_value={"ok": True, "issues": [], "reports": {}}):
            result = runner.run("run generic task")
        self.assertTrue(result["finished"])
        self.assertEqual(result["rounds"], 6)
        self.assertFalse((self.ws / "backend_checkpoint.json").exists())
        self.assert_no_orphan_results(self.llm.messages[-1])

    def test_existing_failed_verdict_does_not_trigger_more_gates_or_stale_diagnosis(self):
        runner = self.runner([call("finish"), text_reply(), text_reply(), text_reply()],
                             verdict_hook=lambda *_: "previous physical diagnosis")
        with patch.object(runner, "_verify", return_value={"ok": False, "issues": ["old gate failure"], "reports": {}}) as verify:
            result = runner.run("run generic task")
        verify.assert_called_once()
        self.assertEqual(result["stop_reason"], "backend_no_tool_progress")
        failure = json.loads((self.ws / "failure_report.json").read_text())
        self.assertEqual(failure["last_verdict"]["issues"], ["old gate failure"])
        self.assertNotIn("diagnosis", failure)
        self.assert_no_orphan_results(self.llm.messages[-1])

    def test_other_stages_keep_existing_implicit_finish_behavior(self):
        runner = self.runner([text_reply()], stage="stage1")
        with patch.object(runner, "_verify", return_value={"ok": True, "issues": [], "reports": {}}) as verify:
            result = runner.run("run generic task")
        verify.assert_called_once()
        self.assertTrue(result["finished"])
        self.assertNotIn("stopped", result)


if __name__ == "__main__":
    unittest.main()
