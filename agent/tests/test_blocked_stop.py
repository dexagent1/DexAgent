"""report_blocked in the harness: without an upstream stage it ends the stage at once as blocked (after a failed
finish; Stage 3 grants no deferrals), an administrative checkpoint stops without verifying, an upstream report names
the stage to re-open, and a block before any finish is refused.

    python -m unittest agent.tests.test_blocked_stop
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.runner import StageRunner
from agent.tests.test_runner_no_tool_progress import RecordingLLM, call

FAILED = {"ok": False, "issues": ["fresh execution receipt required"], "reports": {}}
PASSED = {"ok": True, "issues": [], "reports": {}}


class BlockedStopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        lib = self.root / "tool_library"
        lib.mkdir()
        (lib / "index.json").write_text('{"tools":[]}')
        self.ws = self.root / "candidate"
        self.cfg = {"inputs": {"library": str(lib)}, "agent": {"max_rounds": 30}}

    def runner(self, turns, **kw):
        self.llm = RecordingLLM(turns)
        return StageRunner(self.cfg, self.llm, self.ws, "generic test", [], stage=kw.pop("stage", "stage3"), **kw)

    def events(self, runner):
        return [json.loads(line) for line in runner.trace.read_text().splitlines()]

    def test_stage3_block_without_upstream_ends_the_stage_at_once(self):
        runner = self.runner([call("finish"), call("report_blocked", {"reason": "still unsolved"}), call("finish")])
        with patch.object(runner, "_verify", return_value=FAILED) as verify:
            out = runner.run("generic task")
        verify.assert_called_once()
        self.assertEqual(out["rounds"], 2)
        self.assertFalse(out["finished"])
        self.assertTrue(out["failure"].startswith("blocked:"))
        self.assertFalse([v for v in self.events(runner) if v["event"] == "unsolved_search_deferred"])

    def test_operator_marker_stops_immediately_without_verifying_explicit_or_legacy(self):
        for marker in ({"kind": "administrative_checkpoint"},
                       {"requested_utc": "now", "reason": "administrative milestone checkpoint", "instruction": "finish current work and stop"}):
            with self.subTest(marker=marker):
                transport = self.root / "transport"
                transport.mkdir(exist_ok=True)
                self.cfg["agent"]["transport_dir"] = str(transport)
                (transport / "operator_checkpoint.json").write_text(json.dumps(marker))
                runner = self.runner([call("report_blocked", {"reason": "operator checkpoint"})])
                with patch.object(runner, "_verify", side_effect=AssertionError("administrative stop must not verify")):
                    out = runner.run("generic task")
                self.assertEqual(out["stop_reason"], "operator_checkpoint")
                self.assertEqual(out["rounds"], 1)
                self.assertIsNone(out["verdict"])
                self.assertFalse(out["finished"])
                self.assertEqual(self.llm.closed, [out])

    def test_upstream_evidence_and_other_stages_are_unchanged(self):
        for stage, args in (("stage3", {"upstream": "stage2", "evidence": "measured input mismatch", "reason": "upstream repair"}),
                            ("stage2", {"reason": "input unavailable"})):
            with self.subTest(stage=stage):
                runner = self.runner([call("finish"), call("report_blocked", args)], stage=stage)
                with patch.object(runner, "_verify", return_value=FAILED):
                    out = runner.run("generic task")
                self.assertEqual(out["rounds"], 2)
                self.assertNotIn("stopped", out)
                self.assertTrue(out["failure"].startswith("blocked:"))
                if stage == "stage3":
                    self.assertEqual(out["upstream"]["stage"], "stage2")

    def test_normal_block_still_requires_failed_finish(self):
        runner = self.runner([call("report_blocked"), call("finish")], max_rounds=2)
        with patch.object(runner, "_verify", return_value=PASSED):
            out = runner.run("generic task")
        self.assertTrue(out["finished"])
        self.assertIn("call finish first", json.dumps(self.llm.messages[1]))


if __name__ == "__main__":
    unittest.main()
