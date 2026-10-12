"""run_dexagent between stages: every stage's report is printed, and when stages remain the user is asked whether to
continue; no stops the run there (with the command that continues it), --yes runs through, no terminal means stop, and
a cross-stage repair is confirmed before the earlier stage is re-opened.

    python -m unittest agent.tests.test_pipeline_confirm
"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("DEXAGENT_QUIET", "1")

import run_dexagent as D  # noqa: E402


class PipelineConfirmTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg_path = Path(self.tmp.name) / "c.yaml"
        self.cfg_path.write_text(f"run:\n  name: p\n  output_dir: {self.tmp.name}\n")
        self.ran = []

    def fake_stage(self, results=None):
        results = results or {}

        def run(stage, cfg_path, extra, cfg):
            self.ran.append(stage)
            r = {"stage": stage, "finished": True, "workspace": str(Path(self.tmp.name) / "p" / stage), "rounds": 3, "tool_calls": 7,
                 "tools_used": {"execute_subgoal_plan": 2}, "new_tools": [{"name": "my_check", "type": "verifier"}]}
            r.update(results.get(stage, {}))
            return r
        return run

    def run_main(self, args, answers=(), tty=True, results=None):
        it = iter(answers)
        with patch.object(D, "run_agent_stage", self.fake_stage(results)), \
             patch.object(D, "run_stage4", lambda cfg: {"stage": "stage4", "finished": True, "workspace": ""}), \
             patch("sys.stdin.isatty", return_value=tty), patch("builtins.input", lambda q: next(it)):
            rc = D.main([str(self.cfg_path), *args])
        summary = json.loads((Path(self.tmp.name) / "p" / "run_summary.json").read_text())
        return rc, summary

    def test_no_stops_after_the_stage(self):
        rc, summary = self.run_main(["--to", "stage3"], answers=["y", "n"])
        self.assertEqual((rc, self.ran), (0, ["stage1", "stage2"]))
        self.assertEqual(summary["stopped_by_user_after"], "stage2")
        self.assertTrue(summary["stages"][-1]["stopped_by_user"])

    def test_yes_runs_every_stage_and_no_terminal_stops(self):
        rc, _ = self.run_main(["--to", "stage3", "--yes"])
        self.assertEqual(self.ran, ["stage1", "stage2", "stage3"])
        self.ran.clear()
        rc, summary = self.run_main(["--to", "stage3"], tty=False)       # nobody to ask: stop there, exit 2 (not finished)
        self.assertEqual((rc, self.ran, summary["stopped_by_user_after"]), (2, ["stage1"], "stage1"))

    def test_repair_is_confirmed(self):
        blocked = {"stage3": {"finished": False, "upstream": {"stage": "stage2", "evidence": "nothing to grip"}, "failure": "blocked"}}
        rc, summary = self.run_main(["--from", "stage3", "--to", "stage3"], answers=["n"], results=blocked)
        self.assertEqual((rc, self.ran), (1, ["stage3"]))
        self.assertIn("user stopped", summary["repair_loops"][0]["refused"])


if __name__ == "__main__":
    unittest.main()
