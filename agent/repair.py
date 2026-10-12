"""Cross-stage repair. A later stage that finds an earlier stage's result wrong ends with
report_blocked(reason, upstream='stage1' | 'stage2', evidence); run_dexagent.py then re-opens that stage with the reason in
its first message, re-runs the stages after it, and continues. The notes below carry the reason between the runs:

    <workspace of the re-opened stage>/repair_request.json    why it is re-opened, by which stage, with what evidence
    <workspace of each stage after it>/upstream_revised.json  which earlier stage was revised, and why

A stage reads its note when it builds its first message and marks it used (the note is renamed with its loop number), so a
later run of the same stage does not see it again. A stage with a note never reuses an example: the input it would reuse
is the one found wrong."""
from __future__ import annotations

import json
import time
from pathlib import Path

ORDER = ["stage1", "stage2", "stage3", "stage4"]
REQUEST = "repair_request.json"
REVISED = "upstream_revised.json"
NAMES = {"stage1": "Stage 1 (semantic understanding)", "stage2": "Stage 2 (scene reconstruction)",
         "stage3": "Stage 3 (trajectory generation)", "stage4": "Stage 4"}


def valid_upstream(stage: str, upstream: str | None) -> str | None:
    """None when `upstream` may be re-opened by `stage` (an earlier agent stage), else why not."""
    if upstream is None:
        return None
    if upstream not in ("stage1", "stage2"):
        return f"upstream must be 'stage1' or 'stage2', not {upstream!r}"
    if stage in ORDER and ORDER.index(upstream) >= ORDER.index(stage):
        return f"{upstream} is not an earlier stage than {stage}: only an earlier stage can be re-opened"
    return None


def write_request(ws, loop: int, requested_by: str, upstream: str, reason: str, evidence: str) -> Path:
    p = Path(ws) / REQUEST
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"loop": int(loop), "requested_by": requested_by, "stage": upstream, "reason": reason,
                             "evidence": evidence, "at": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
    return p


def write_revised(ws, loop: int, upstream: str, requested_by: str, reason: str) -> Path:
    p = Path(ws) / REVISED
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"loop": int(loop), "revised": upstream, "requested_by": requested_by, "reason": reason}, indent=1))
    return p


def archive_failure(ws, loop: int) -> str | None:
    """The blocked stage's failure_report.json kept as failure_report_loop<k>.json before the stage runs again."""
    p = Path(ws) / "failure_report.json"
    if not p.exists():
        return None
    q = p.with_name(f"failure_report_loop{int(loop)}.json")
    p.replace(q)
    return str(q)


def has_note(ws) -> bool:
    return (Path(ws) / REQUEST).exists() or (Path(ws) / REVISED).exists()


def take_note(ws) -> str:
    """The text a stage appends to its first message (empty when no repair concerns it); the notes are marked used."""
    ws = Path(ws)
    parts = []
    rq = ws / REQUEST
    if rq.exists():
        d = json.loads(rq.read_text())
        parts.append(
            f"REPAIR REQUEST (repair loop {d.get('loop')}): {NAMES.get(d.get('requested_by'), d.get('requested_by'))} found this stage's "
            f"deliverable wrong and re-opened this stage.\nReason: {d.get('reason')}\nEvidence: {d.get('evidence')}\n"
            "Your previous deliverable and workspace are still here. Check the evidence against the video, revise what it shows is "
            "wrong (and only that), verify, and call finish. This stage is re-opened once: if the request is mistaken, say why with "
            "report_blocked (without upstream) instead of changing what is right.")
        rq.replace(ws / f"repair_request_loop{d.get('loop')}.used.json")
    rv = ws / REVISED
    if rv.exists():
        d = json.loads(rv.read_text())
        parts.append(
            f"UPSTREAM REVISED (repair loop {d.get('loop')}): {NAMES.get(d.get('revised'), d.get('revised'))} was re-opened at the "
            f"request of {NAMES.get(d.get('requested_by'), d.get('requested_by'))} and its deliverable changed.\nReason: {d.get('reason')}\n"
            "Your workspace still holds your earlier work from before the revision: re-check everything that depends on the revised "
            "stage before reusing it. An earlier stage is re-opened at most once per run.")
        rv.replace(ws / f"upstream_revised_loop{d.get('loop')}.used.json")
    return ("\n\n" + "\n\n".join(parts)) if parts else ""
