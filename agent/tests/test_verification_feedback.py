"""Final-gate failure evidence survives large passing prefixes and retries."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.verification_feedback import verification_feedback
from stage3_trajectory_generation.run_stage3 import make_verifiers
from tool_library.common.evidence import file_digest


class VerificationFeedbackTests(unittest.TestCase):
    def report(self):
        failure = {"ok": False, "goal": "part is at destination", "predicate": "at_target",
                   "measured": {"distance_m": .14, "threshold_m": .02, "frame": 901, "body": "part"}}
        goals = {"ok": False, "subgoals": [{"ok": True, "measured": [1.] * 2000}] * 200,
                 "task_success": {"ok": False, "predicates": [failure]}}
        return {"ok": False, "issues": ["final task_success failed"],
                "reports": {"verbose_pass": {"ok": True, "values": [1.] * 20000}, "stage1_goal_gate": goals}}

    def test_failed_final_goal_is_visible_after_large_successful_prefix(self):
        suite = self.report()
        original = copy.deepcopy(suite)
        text = verification_feedback({"ok": False, "issues": suite["issues"], "reports": suite})
        # Also exercise the driver's shape, which has already separated failed checks.
        text = verification_feedback({"ok": False, "issues": suite["issues"], "reports": {
            "stage3_gate_suite": {"failed_checks": suite["reports"],
                                  "diagnostic_report": {"path": "/tmp/raw.json", "sha256": "abc"}}}})
        self.assertLessEqual(len(text), 12000)
        out = json.loads(text)
        goal = out["failed_checks"]["stage1_goal_gate"]["task_success"]
        self.assertEqual(goal["predicates"][0]["measured"]["distance_m"], .14)
        self.assertEqual(goal["predicates"][0]["measured"]["frame"], 901)
        self.assertEqual(out["diagnostic_report"]["path"], "/tmp/raw.json")
        self.assertEqual(suite, original)

    def test_huge_failures_keep_valid_json_and_report_locator(self):
        reports = {f"audit_{i}": {"ok": False, "issues": ["reason " * 500] * 500,
                                "measured": {str(n): [3.] * 1000 for n in range(30)}} for i in range(35)}
        text = verification_feedback({"ok": False, "reports": {"stage3_gate_suite": {
            "failed_checks": reports, "diagnostic_report": {"path": "/tmp/full.json"}}}}, limit=4000)
        self.assertLessEqual(len(text), 4000)
        out = json.loads(text)
        self.assertEqual(len(out["failed_check_names"]), 35)
        self.assertFalse(out["ok"])
        self.assertEqual(out["diagnostic_report"]["path"], "/tmp/full.json")

    def test_manifest_failure_survives_simultaneous_task_failure(self):
        raw = {"ok": False, "reports": {
            "trajectory_manifest_check": {"ok": False, "issues": ["selected pose differs from executed pose"]},
            "stage3_gate_suite": {"ok": False, "failed_checks": {
                "stage1_goal_gate": {"ok": False, "task_success": {"ok": False, "issues": ["wrong destination"]}}}}}}
        out = json.loads(verification_feedback(raw))
        self.assertEqual(set(out["failed_check_names"]), {"trajectory_manifest_check", "stage1_goal_gate"})
        self.assertIn("selected pose", out["failed_checks"]["trajectory_manifest_check"]["issues"][0])

    def test_each_invocation_saves_full_report_and_binds_candidate_separately(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            lib = ws / "library"; lib.mkdir()
            (lib / "index.json").write_text('{"tools":[]}')
            sem, recon = ws / "sem.json", ws / "recon.json"
            sem.write_text('{}'); recon.write_text('{}')
            (ws / "scene_physics.xml").write_text('<mujoco/>')
            (ws / "input_contract.json").write_text('{}')
            (ws / "trajectory.npz").write_bytes(b"candidate one")
            (ws / "trajectory_manifest.json").write_text('{}')
            cfg = {"inputs": {}, "stage3": {}}
            raw = self.report()
            with patch("agent.library.call_tool", return_value={"ok": True, "result": raw}):
                gate = make_verifiers(cfg, sem, recon, lib)[1]
                first = gate(ws)
                (ws / "trajectory.npz").write_bytes(b"candidate two")
                second = gate(ws)
                sem.write_text('{"new_goal":true}')
                third = gate(ws)
            self.assertFalse(first["ok"])
            self.assertEqual(first["task_goal_report"], raw["reports"]["stage1_goal_gate"])
            path = Path(first["diagnostic_report"]["path"])
            self.assertEqual(json.loads(path.read_text())["report"], raw)
            self.assertEqual(file_digest(path), first["diagnostic_report"]["sha256"])
            self.assertNotEqual(path, Path(second["diagnostic_report"]["path"]))
            self.assertEqual(first["comparison_scope"], second["comparison_scope"])
            self.assertNotEqual(first["candidate_binding"], second["candidate_binding"])
            self.assertNotEqual(second["comparison_scope"], third["comparison_scope"])


if __name__ == "__main__":
    unittest.main()
