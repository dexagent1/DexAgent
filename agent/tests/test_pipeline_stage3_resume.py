"""Pipeline recovery forwards the attempt while preserving current-stage checks."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("DEXAGENT_QUIET", "1")
import run_dexagent as D
from stage3_trajectory_generation import run_stage3 as S3


class PipelineResumeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / "runs" / "trial"
        self.attempt = self.out / "stage3" / ".candidates" / "old"
        self.attempt.mkdir(parents=True)
        self.cfg = {"run": {"name": "trial", "output_dir": str(self.root / "runs")},
                    "inputs": {"video": str(self.root / "video.mp4")}, "repair": {"max_loops_per_stage": 1},
                    "examples": {"reuse": False, "save_assets": False}}

    def test_forwarded_only_to_stage3_including_after_stage2_repair(self):
        seen = []
        def stage(name, config_path, extra, cfg):
            seen.append((name, list(extra)))
            workspace = self.out / name
            workspace.mkdir(exist_ok=True)
            if [name for name, _ in seen].count("stage3") == 1 and name == "stage3":
                return {"stage": name, "workspace": str(workspace), "finished": False,
                        "failure": "generic changed reconstruction", "upstream": {"stage": "stage2", "evidence": "measured mismatch"}}
            return {"stage": name, "workspace": str(workspace), "finished": True}
        with patch.object(D, "load_config", return_value=self.cfg), patch.object(D, "run_agent_stage", side_effect=stage), \
                patch("agent.reuse.save_run", return_value=None), patch.object(D, "summary_table"):
            result = D.main(["config.yaml", "--from", "stage2", "--to", "stage3", "--resume-stage3-attempt", str(self.attempt),
                             "--max-rounds", "7", "--llm", "mock:turns.json", "--video", "replacement.mp4", "--yes"])
        self.assertEqual(result, 0)
        self.assertEqual([name for name, _ in seen], ["stage2", "stage3", "stage2", "stage3"])
        for name, args in seen:
            self.assertEqual(args[args.index("--max-rounds") + 1], "7")
            self.assertEqual(args[args.index("--llm") + 1], "mock:turns.json")
            self.assertEqual(args[args.index("--video") + 1], "replacement.mp4")
            if name == "stage3":
                self.assertEqual(args[args.index("--resume-attempt") + 1], str(self.attempt))
            else:
                self.assertNotIn("--resume-attempt", args)
        summary = json.loads((self.out / "run_summary.json").read_text())
        self.assertEqual(len(summary["repair_loops"]), 1)
        self.assertTrue(summary["repair_loops"][0]["resolved"])

    def test_no_resume_flag_keeps_fresh_pipeline_behavior(self):
        def stage(name, config_path, extra, cfg):
            self.assertNotIn("--resume-attempt", extra)
            return {"stage": name, "workspace": str(self.out / name), "finished": True}
        with patch.object(D, "load_config", return_value=self.cfg), patch.object(D, "run_agent_stage", side_effect=stage), \
                patch("agent.reuse.save_run", return_value=None), patch.object(D, "summary_table"):
            result = D.main(["config.yaml", "--from", "stage3", "--to", "stage3", "--yes"])
        self.assertEqual(result, 0)

    def test_cooperative_stop_ends_pipeline_without_upstream_repair(self):
        stopped = {"stage": "stage3", "workspace": str(self.out / "stage3"), "finished": False,
                   "stopped": True, "stop_reason": "cooperative_search_checkpoint",
                   "search_checkpoint": {"checkpoint_json": "saved.json"}, "failure": "administrative stop"}
        with patch.object(D, "load_config", return_value=self.cfg), patch.object(D, "run_agent_stage", return_value=stopped) as stage, \
                patch.object(D, "run_stage4") as stage4, patch.object(D, "summary_table"):
            result = D.main(["config.yaml", "--from", "stage3", "--resume-stage3-attempt", str(self.attempt), "--yes"])
        self.assertEqual(result, 1)
        self.assertEqual(stage.call_count, 1)
        stage4.assert_not_called()
        summary = json.loads((self.out / "run_summary.json").read_text())
        self.assertEqual(summary["repair_loops"], [])
        self.assertTrue(summary["stages"][0]["stopped"])
        self.assertEqual(summary["stages"][0]["search_checkpoint"], stopped["search_checkpoint"])

    def test_stage_adapter_keeps_cooperative_metadata(self):
        def main(args):
            self.assertEqual(args, ["config.yaml", "--resume-attempt", str(self.attempt)])
            (self.out / "stage3" / "run.json").write_text(json.dumps({"finished": False, "stopped": True,
                "stop_reason": "cooperative_search_checkpoint", "search_checkpoint": {"checkpoint_json": "saved.json"}}))
            return 1
        with patch.object(S3, "main", side_effect=main):
            out = D.run_agent_stage("stage3", "config.yaml", ["--resume-attempt", str(self.attempt)], self.cfg)
        self.assertFalse(out["finished"])
        self.assertTrue(out["stopped"])
        self.assertEqual(out["search_checkpoint"]["checkpoint_json"], "saved.json")

    def test_actual_stage_driver_still_rejects_source_identity_on_resume(self):
        (self.root / "video.mp4").write_bytes(b"pinned source")
        for stage, filename in (("stage1", "scene_semantic.json"), ("stage2", "scene_recon.json")):
            directory = self.out / stage
            directory.mkdir(exist_ok=True)
            (directory / filename).write_text("{}")
        self.cfg["_executor_receipts"] = {"old": "forged accepted receipt"}
        (self.attempt / "input_contract.json").write_text('{"sha256": {}}')
        (self.attempt / "run.json").write_text('{"finished":false,"verdict":{"ok":true}}')
        with patch.object(S3, "load_config", return_value=self.cfg), \
                patch("agent.input_identity.video_verifier", return_value=lambda ws: {"ok": False, "issues": ["source identity mismatch"]}) as identity, \
                patch("agent.runner.StageRunner", side_effect=AssertionError("failed identity must stop before model")) as runner:
            out = D.run_agent_stage("stage3", "config.yaml", ["--resume-attempt", str(self.attempt)], self.cfg)
        self.assertFalse(out["finished"])
        self.assertEqual(identity.call_count, 1)
        runner.assert_not_called()
        self.assertEqual(self.cfg["_executor_receipts"], {})
        self.assertFalse(json.loads((self.out / "stage3" / "run.json").read_text())["finished"])


if __name__ == "__main__":
    unittest.main()
