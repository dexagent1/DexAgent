"""Bounded diagnostic feedback; never an input to acceptance.

Keep failing goals and measured evidence ahead of passing, verbose audits. The
complete per-invocation report remains on disk, identified by content hash.
"""
from __future__ import annotations

import json
import math


def _failed(value):
    if isinstance(value, dict):
        return value.get("ok") is False or bool(value.get("failed") or value.get("issues") or value.get("error"))
    return False


def compact_evidence(value, *, depth=6, items=12, text_limit=900):
    """Produce JSON-safe evidence with explicit omissions and failure priority."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_value": str(value)}
    if isinstance(value, str):
        return value if len(value) <= text_limit else value[:text_limit] + " [truncated; see diagnostic_report]"
    if not isinstance(value, (dict, list, tuple)):
        return value
    if depth <= 0:
        return {"omitted_nested_items": len(value), "read": "diagnostic_report"}
    recur = lambda v: compact_evidence(v, depth=depth - 1, items=items, text_limit=text_limit)
    if isinstance(value, (list, tuple)):
        # Preserve all scalar indices/identities when small; put failed reports
        # first so a long successful prefix cannot bury the current failure.
        ordered = sorted(enumerate(value), key=lambda p: (not _failed(p[1]), p[0]))
        out = [recur(v) for _, v in ordered[:items]]
        if len(value) > items:
            out.append({"omitted_items": len(value) - items, "read": "diagnostic_report"})
        return out
    essential = ("ok", "task_success", "failed", "issues", "error", "errors", "index", "step", "predicate",
                 "goal", "type", "motion", "side", "object", "frames", "measured", "threshold",
                 "limit", "expected", "actual", "unresolved", "checks")
    rank = {key: i for i, key in enumerate(essential)}
    ordered = sorted(value, key=lambda k: (0 if _failed(value[k]) else 1,
                                          rank.get(k, len(rank)),
                                          isinstance(value[k], (dict, list)), str(k)))
    out = {k: recur(value[k]) for k in ordered[:items]}
    if len(value) > items:
        out["omitted_keys"] = [str(k) for k in ordered[items:]]
    return out


def verification_feedback(verdict, limit=12000):
    """Return valid JSON, always exposing failed check names and report paths.

    This function only controls what the model sees. `StageRunner._verify`
    decides acceptance from original verifier results before formatting.
    """
    if limit < 1000:
        raise ValueError("verification feedback needs at least 1000 characters")
    reports = verdict.get("reports") or {}
    suite = reports.get("stage3_gate_suite") or {}
    detailed = dict(suite.get("failed_checks") or {})
    detailed.update({
        name: report for name, report in reports.items()
        if name != "stage3_gate_suite" and isinstance(report, dict) and report.get("ok") is not True})
    if not detailed and suite.get("ok") is False:
        detailed["stage3_gate_suite"] = {k: v for k, v in suite.items() if k != "task_goal_report"}
    names = sorted(detailed, key=lambda name: ("goal" not in name and "task" not in name, name))
    base = {"ok": verdict.get("ok"), "authority": "diagnostic_only; original verifier verdict controls acceptance",
            "failed_check_names": names, "diagnostic_report": suite.get("diagnostic_report"),
            "comparison_scope": suite.get("comparison_scope"),
            "candidate_binding": suite.get("candidate_binding"),
            "next_step": "Read failed evidence; repair the candidate; rerun affected skills and finish. Read diagnostic_report for omitted detail."}
    for items, depth, text_limit in ((12, 7, 800), (8, 6, 500), (5, 5, 300), (3, 4, 200), (2, 3, 100)):
        out = {**base, "issues": compact_evidence(verdict.get("issues", []), items=items, text_limit=text_limit),
               "failed_checks": {name: compact_evidence(detailed[name], depth=depth, items=items, text_limit=text_limit)
                                 for name in names}}
        encoded = json.dumps(out, ensure_ascii=False, default=str, allow_nan=False)
        if len(encoded) <= limit:
            return encoded
    # Preserve each failure's identity even if every report is huge. Fill the
    # remaining budget in priority order, leaving an explicit omission record.
    out = {**base, "feedback_truncated": True, "failed_checks": {}}
    for name in names:
        report = compact_evidence(detailed[name], depth=4, items=4, text_limit=180)
        trial = {**out, "failed_checks": {**out["failed_checks"], name: report}}
        if len(json.dumps(trial, ensure_ascii=False, default=str)) <= limit:
            out = trial
    encoded = json.dumps(out, ensure_ascii=False, default=str, allow_nan=False)
    if len(encoded) > limit:
        # Extreme number/length of check names: full list still exists on disk.
        out = {"ok": verdict.get("ok"), "feedback_truncated": True,
               "diagnostic_report": suite.get("diagnostic_report"),
               "error": "Diagnostic exceeds feedback budget; read full report before repair.",
               "failed_check_count": len(names)}
        encoded = json.dumps(out, default=str)
    return encoded
