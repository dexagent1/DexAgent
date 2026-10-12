"""A tool's result hook can append feedback and end a stalled search; a stage's wall-clock budget ends a run."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.runner import StageRunner
from agent.tests.test_runner_no_tool_progress import RecordingLLM, call

EXECUTED = {"ok": True, "result": {"ok": False, "n_frames": 10}}


class HookAndWallTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        lib = root / "tool_library"
        lib.mkdir()
        (lib / "index.json").write_text('{"tools":[]}')
        self.ws = root / "candidate"
        self.cfg = {"inputs": {"library": str(lib)}, "agent": {"max_rounds": 30}}

    def test_hook_text_reaches_the_model_and_a_stop_ends_the_search(self):
        seen = []

        def hook(ws, args, raw):
            seen.append(args)
            n = len(seen)
            return {"text": f"[harness] execution #{n}: no improvement", "stop": "search_stalled" if n >= 3 else None}

        llm = RecordingLLM([call("execute_subgoal_plan", {"plan": {"n": k}}) for k in range(6)])
        runner = StageRunner(self.cfg, llm, self.ws, "generic test", [], stage="stage3",
                             result_hooks={"execute_subgoal_plan": hook})
        with patch("agent.runner.call_tool", return_value=EXECUTED):
            out = runner.run("generic task")
        self.assertEqual(len(seen), 3)                          # the third execution's hook ended the search
        self.assertEqual(out["stop_reason"], "search_stalled")
        self.assertFalse(out["finished"])
        self.assertIn("execution #1: no improvement", json.dumps(llm.messages[1][-1]))

    def test_wall_budget_gives_grace_rounds_then_stops(self):
        cfg = dict(self.cfg, stage3={"max_wall_minutes": 1e-6})
        llm = RecordingLLM([call("library_search", {"query": "x"}) for _ in range(10)])
        runner = StageRunner(cfg, llm, self.ws, "generic test", [], stage="stage3")
        with patch.object(runner.dev, "library_search", return_value={"tools": []}):
            out = runner.run("generic task")
        self.assertEqual(out["stop_reason"], "wall_time_budget")
        self.assertLessEqual(out["rounds"], 6)
        self.assertIn("wall-clock budget", json.dumps(llm.messages))


if __name__ == "__main__":
    unittest.main()
