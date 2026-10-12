"""Uncovered frames check: does the episode account for the whole demonstration?

    from tool_library.verifiers.uncovered_frames_check import check_uncovered_frames
    check_uncovered_frames("trajectory.npz", video_frames=1230, video_fps=30)

The trajectory of a replicated task is locked to the video's timeline: it
must be as long as the video, every frame must carry a phase label, and no
phase may be a sliver. A short trajectory means the end of the task was never
planned; an unlabelled stretch means nobody decided what the robot does
there; a one-frame phase is a label typo that breaks phase-based anchoring.
"""
from __future__ import annotations

TOOL = {
    "name": "uncovered_frames_check",
    "title": "Uncovered frames check",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "Trajectory length matches the video, every frame has a phase label, no phase is shorter than a few frames.",
    "description": (
        "Compares the trajectory's frame count and frame rate with the demonstration video's "
        "(given directly or read from the video file; a video_path that cannot be opened or reports no frames "
        "fails), fails when the trajectory covers less "
        "than cover_frac (default 0.98) of the video's duration or runs more than 10% longer, "
        "lists frames with empty labels, and lists phases shorter than min_phase_frames "
        "(default 3). Optionally also requires a list of expected phase patterns to each "
        "match at least one frame, which is how a subgoal plan is checked against the "
        "trajectory that claims to implement it."
    ),
    "inputs": [
        {"name": "trajectory", "type": "str | (Q, labels, fps)", "description": "trajectory.npz with qpos, labels and fps."},
        {"name": "video_frames, video_fps", "type": "int, float | None", "description": "The demonstration's length; or give video_path."},
        {"name": "video_path", "type": "str | None", "description": "Read frame count and fps from this file (needs opencv)."},
        {"name": "expected_phases", "type": "list[str] | None", "description": "fnmatch patterns that must each match a frame."},
        {"name": "cover_frac", "type": "float", "description": "Minimum covered fraction of the video duration. Default 0.98."},
        {"name": "min_phase_frames", "type": "int", "description": "Shortest legal phase run. Default 3."},
        {"name": "label_sep", "type": "str, regex", "description": "Labels may carry telemetry after this separator; only the part before it is the phase. Default three or more spaces (' | ' separates the two hands of a two-hand step and is kept)."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; durations, covered fraction, video_error (video_path unreadable), unlabeled frame ranges, short phases, missing expected phases, phase table."},
    ],
    "requirements": ["numpy", "opencv-python (optional)"],
    "tags": ["timeline", "coverage", "phases", "video-locked"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope", "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting", "Biology Experiment"],
    "status": "tested",
    "entrypoint": "uncovered_frames_check",
}

import json
import re
import sys


from tool_library.common import trajectory as TJ


def _ranges(idx):
    out = []
    for i in idx:
        if out and i == out[-1][1] + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out


def check_uncovered_frames(trajectory, video_frames=None, video_fps=None, video_path=None,
                           expected_phases=None, cover_frac=0.98, min_phase_frames=3,
                           label_sep=r"\s{3,}"):
    Q, labels, fps = TJ.as_trajectory(trajectory)
    phases = [re.split(label_sep, lab, maxsplit=1)[0].strip() for lab in labels]
    video_error = None
    if video_path is not None:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            video_error = f"cannot open the video {video_path}"
        else:
            video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = float(cap.get(cv2.CAP_PROP_FPS)) or fps
            if video_frames <= 0:
                video_error = f"the video {video_path} reports {video_frames} frames"
        cap.release()
    T = len(Q)
    rep = {"frames": T, "fps": fps, "duration_s": round(T / fps, 3)}
    ok = True
    if video_error:                                   # the length cannot be judged: a failure, not a silent skip
        rep["video_error"] = video_error
        ok = False
    if video_frames:
        vfps = video_fps or fps
        covered = (T / fps) / (video_frames / vfps)
        rep.update(video_frames=int(video_frames), video_fps=vfps, covered_frac=round(float(covered), 4),
                   duration_ok=bool(cover_frac <= covered <= 1.10))
        ok &= rep["duration_ok"]
    empty = [t for t, lab in enumerate(labels) if not lab.strip()]
    rep["unlabeled_ranges"] = _ranges(empty)
    rep["unlabeled_frames"] = len(empty)
    ok &= not empty
    runs = TJ.phase_runs(phases)
    rep["phases"] = [{"label": lab, "frames": [a, b], "n": b - a + 1} for lab, a, b in runs]
    rep["short_phases"] = [p for p in rep["phases"] if p["n"] < min_phase_frames and p["label"].strip()]
    ok &= not rep["short_phases"]
    if expected_phases:
        rep["missing_phases"] = [p for p in expected_phases if not TJ.frames_matching(phases, p)]
        ok &= not rep["missing_phases"]
    rep["ok"] = bool(ok)
    return rep


if __name__ == "__main__":
    rep = check_uncovered_frames(sys.argv[1], video_path=sys.argv[2] if len(sys.argv) > 2 else None)
    print(json.dumps(rep, indent=1, default=str)); sys.exit(0 if rep["ok"] else 1)


uncovered_frames_check = check_uncovered_frames   # the library calls every tool by its own name
