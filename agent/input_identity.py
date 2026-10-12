"""Bind stage inputs to the exact human video; metadata alone is not identity evidence."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def video_identity(video):
    """Read the encoded video's digest and timing without modifying it."""
    import cv2
    path = Path(video).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    cap = cv2.VideoCapture(str(path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        readable, _ = cap.read()
    finally:
        cap.release()
    if not readable or frames <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"video cannot be decoded: {path}")
    return {"path": str(path), "sha256": digest.hexdigest(), "frames": frames, "fps": fps,
            "duration_s": frames / fps, "width": width, "height": height}


def check_video_identity(video, semantics, *, expected_identity=None, require_identity=True):
    """Compare the actual source with immutable provenance and the semantics' timing.

    `expected_identity` is an independently recorded input identity, usually the
    Stage 1 input manifest. Legacy annotations can instead reference an existing
    original video; equal bytes prove that the provided clip is the same source.
    A matching duration or filename alone never proves identity.
    """
    semantic_path = None
    if not isinstance(semantics, dict):
        semantic_path = Path(semantics).resolve()
        semantics = json.loads(semantic_path.read_text())
    actual = video_identity(video)
    declared = semantics.get("video") or {}
    expected = expected_identity or semantics.get("input_identity") or declared
    issues, warnings = [], []
    digest = expected.get("sha256")
    evidence = "recorded_sha256" if digest else None
    if not digest and declared.get("path"):
        source = Path(declared["path"])
        if not source.is_absolute() and semantic_path is not None:
            source = semantic_path.parent / source
        if source.is_file():
            source_identity = video_identity(source)
            digest = source_identity["sha256"]
            evidence = "existing_original_video"
        else:
            warnings.append(f"declared source video is unavailable: {source}")
    if digest and str(digest) != actual["sha256"]:
        issues.append("source video SHA-256 differs from Stage 1 input provenance")
    elif not digest and require_identity:
        issues.append("Stage 1 has no source digest or available original video; input identity is unverified")
    for record in (declared, expected):
        for key in ("frames", "width", "height"):
            if record.get(key) is not None and int(record[key]) != actual[key]:
                issues.append(f"video {key} mismatch: annotated {record[key]}, actual {actual[key]}")
        for key, tolerance in (("fps", 0.01), ("duration_s", max(0.002, 1 / actual["fps"]))):
            if record.get(key) is not None and (not math.isfinite(float(record[key]))
                    or abs(float(record[key]) - actual[key]) > tolerance):
                issues.append(f"video {key} mismatch: annotated {record[key]}, actual {actual[key]:.6f}")
    for goal in semantics.get("subgoals", []):
        window = goal.get("time_window_s")
        if window and (float(window[0]) < 0 or float(window[-1]) > actual["duration_s"] + 1 / actual["fps"]):
            issues.append(f"subgoal {goal.get('index')} time window {window} exceeds the actual video")
    return {"ok": not issues, "issues": list(dict.fromkeys(issues)), "warnings": warnings,
            "identity_verified": bool(digest) and str(digest) == actual["sha256"],
            "identity_evidence": evidence, "actual": actual}


def video_verifier(video, semantic, *, expected_identity=None):
    """A final gate for the exact input captured by the stage driver."""
    def source_video_identity_check(workspace):
        from agent.config import semantics_file
        path = semantic if semantic is not None else semantics_file(workspace)
        try:
            return check_video_identity(video, path, expected_identity=expected_identity)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return {"ok": False, "issues": [f"cannot establish source video identity: {exc}"]}
    return source_video_identity_check


def input_failure(workspace, report):
    """Return actionable upstream failure through the ordinary repair mechanism."""
    from agent.external import atomic_json
    workspace = Path(workspace)
    out = {"finished": False, "failure": "Source video and Stage 1 provenance are inconsistent",
           "upstream": {"stage": "stage1", "evidence": report},
           "verdict": {"ok": False, "reports": {"source_video_identity_check": report}}}
    atomic_json(workspace / "run.json", out)
    atomic_json(workspace / "failure_report.json", out)
    return out
