"""Progress snapshots observe, and cannot supply, final acceptance evidence."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root: runs as a script too
from tool_library.verifiers.stage3_gate_suite import stage3_gate_suite


class GateProgressTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ws = Path(self.tmp.name)
        (self.ws / "scene.xml").touch()
        (self.ws / "trajectory.npz").touch()
        (self.ws / "recon.json").write_text(json.dumps({"scene_xml": "scene.xml"}))
        (self.ws / "trajectory_manifest.json").write_text(json.dumps({
            "scene_executed": "scene.xml", "subgoals": []}))

    def run_suite(self, callback, **kwargs):
        with patch("tool_library.verifiers.stage3_gate_suite._call", side_effect=callback):
            return stage3_gate_suite(str(self.ws), "semantic.json", str(self.ws / "recon.json"), **kwargs)

    def test_observes_each_running_check_without_skipping_any(self):
        calls = []
        def check(name, kwargs):
            progress = json.loads((self.ws / "gate_progress.json").read_text())
            self.assertEqual(progress["active_check"], name)
            self.assertFalse(progress["complete"])
            self.assertEqual(list(progress["checks"]), calls)
            calls.append(name)
            return {"ok": True, "issues": []}
        result = self.run_suite(check)
        progress = json.loads((self.ws / "gate_progress.json").read_text())
        self.assertTrue(result["ok"])
        self.assertTrue(progress["complete"])
        self.assertTrue(progress["diagnostic_only"])
        self.assertIsNone(progress["active_check"])
        self.assertEqual(list(progress["checks"]), list(result["reports"]))
        self.assertIn("stage4_physics_replay_gate", calls)

    def test_stable_grasp_audit_covers_lift_without_agent_expectations(self):
        seen = {}
        def check(name, kwargs):
            if name == "held_object_slip_audit":
                seen.update(kwargs)
            return {"ok": True}
        result = self.run_suite(check)
        self.assertTrue(result["ok"])
        self.assertTrue({"lift", "carry", "transport", "pour", "tilt", "set down"}.issubset(seen["words"]))

    def test_stale_success_cannot_override_crashing_check(self):
        (self.ws / "gate_progress.json").write_text(json.dumps({"ok": True, "complete": True}))
        def check(name, kwargs):
            if name == "physics_recording_check":
                raise ValueError("invalid recording")
            return {"ok": True}
        result = self.run_suite(check)
        progress = json.loads((self.ws / "gate_progress.json").read_text())
        self.assertFalse(result["ok"])
        self.assertFalse(progress["checks"]["physics_recording_check"]["ok"])
        self.assertIn("invalid recording", " ".join(progress["issues"]))
        self.assertTrue(result["deferred"])

    def test_unwritable_progress_does_not_change_verdict(self):
        original = Path.write_text
        def write(path, *args, **kwargs):
            if path.name.startswith(".gate_progress.json."):
                raise PermissionError("read-only workspace")
            return original(path, *args, **kwargs)
        with patch.object(Path, "write_text", write):
            result = self.run_suite(lambda *args: {"ok": True})
        self.assertTrue(result["ok"])
        self.assertIn("stage4_physics_replay_gate", result["reports"])


if __name__ == "__main__":
    unittest.main()
