"""Execution repair feedback preserves current failure evidence and gate authority."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.config import load_config
from agent.llm import MockLLM
from agent.runner import StageRunner, _executor_feedback
from tool_library.skills.diagnose_stage3_failure import diagnose_stage3_failure

ROOT = Path(__file__).resolve().parents[2]


def failure_record():
    failure = {"index": 101, "motion": "carry", "ok": False, "side": "right", "object": "part",
               "failed": ["arm_unreachable"], "measured": {"position_error_mm": 17.3}}
    return {"ok": True, "result": {"ok": False, "first_failure": failure,
            "n_subgoals": 120, "files": {"execution_report": "/tmp/execution_report.json"},
            "subgoals": [{"index": i, "ok": True, "measured": {"reach": [1.0] * 2000}}
                         for i in range(100)] + [failure]}}


def test_long_prefix_keeps_failure_and_raw_measurements():
    record = failure_record()
    original = copy.deepcopy(record)
    text = _executor_feedback(record, 10000)
    shown = json.loads(text)
    assert len(text) <= 10000
    assert shown["result"]["first_failure"]["measured"]["position_error_mm"] == 17.3
    assert shown["result"]["first_failure"]["failed"] == ["arm_unreachable"]
    assert shown["result"]["files"]["execution_report"]
    assert record == original


def test_large_failed_report_remains_valid_json():
    record = failure_record()
    record["result"]["first_failure"]["measured"]["all_frames"] = ["long diagnostic " * 500] * 500
    shown = _executor_feedback(record, 10000)
    assert len(shown) <= 10000
    parsed = json.loads(shown)
    assert parsed["feedback_truncated"]
    assert parsed["result"]["first_failure"]["failed"] == ["arm_unreachable"]


def test_nested_measurements_cannot_remove_failure_fields():
    wide = {str(i): ["long detail " * 100 for _ in range(30)] for i in range(30)}
    # The actual executor's field order puts failed/error after four identity
    # fields. Truncating that dict to four items erased every repair cue.
    failure = dict(index=20, motion="carry", side="right", object="part",
                   failed=["no_slip"], measured=wide,
                   error="The held part moved relative to the hand.", expect=wide)
    record = {"ok": True, "result": dict(ok=False, first_failure=failure,
        files={"execution_report": "/tmp/report.json"}, n_subgoals=21,
        subgoals=[dict(index=i, ok=True) for i in range(20)] + [dict(failure, ok=False)])}
    original = copy.deepcopy(record)
    text = _executor_feedback(record, 10000)
    shown = json.loads(text)
    assert len(text) <= 10000
    assert shown["feedback_truncated"]
    for key in ("index", "motion", "side", "object", "failed", "error"):
        assert shown["result"]["first_failure"][key] == failure[key]
    assert record == original


def test_failed_execution_gets_diagnosis_without_finish_or_receipt(tmp_path):
    cfg = load_config(ROOT / "dexagent.yaml")
    calls = []
    def hook(ws, verdict):
        calls.append(verdict)
        return "Diagnosis: arm_unreachable; inspect the complete motion."
    runner = StageRunner(cfg, MockLLM([]), tmp_path, "test", [], stage="stage3", verdict_hook=hook)
    record = failure_record()
    with patch("agent.runner.call_tool", return_value=record):
        blocks, raw = runner.run_tool("execute_subgoal_plan", {"scene": "test.xml", "plan": []})
    assert raw is record
    assert calls[0]["executor_result"] is record["result"]
    assert any("arm_unreachable" in b["text"] and "Diagnosis:" in b["text"] for b in blocks)
    assert cfg["_executor_receipts"] == {}
    assert json.loads(blocks[0]["text"])["result"]["ok"] is False


def test_current_diagnosis_excludes_stale_gate_reports(tmp_path):
    (tmp_path / "gate_suite_report.json").write_text(json.dumps({"ok": False,
        "issues": ["stage2 must be revised: misplaced geometry"]}))
    outcomes = [failure_record()["result"]["first_failure"]]
    result = diagnose_stage3_failure(workspace=tmp_path, verdict={"ok": False, "issues": []},
        outcomes=outcomes, include_workspace_history=False, check_hand_scenes=False)
    classes = {c["class"] for c in result["classes"]}
    assert "unreachable" in classes
    assert "upstream_suspect" not in classes
    assert result["n_failed_outcomes"] == 1


class ExecutorFeedbackTests(unittest.TestCase):
    def test_prefix(self):
        test_long_prefix_keeps_failure_and_raw_measurements()

    def test_large_failure(self):
        test_large_failed_report_remains_valid_json()

    def test_nested_failure(self):
        test_nested_measurements_cannot_remove_failure_fields()

    def test_automatic_diagnosis(self):
        with tempfile.TemporaryDirectory() as tmp:
            test_failed_execution_gets_diagnosis_without_finish_or_receipt(Path(tmp))

    def test_current_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            test_current_diagnosis_excludes_stale_gate_reports(Path(tmp))


if __name__ == "__main__":
    unittest.main()
