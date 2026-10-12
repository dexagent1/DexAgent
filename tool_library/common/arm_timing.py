"""Bounded timing retries for a reachable path whose physical arm overshoots.

Retiming changes commands only. A new simulation must start from the original
complete integration/mechanism state, and must pass every ordinary outcome gate.
"""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np

from tool_library.common import armplan as AP
from tool_library.common import episode as E
from tool_library.common import rigid_robot as RR

MAX_RETRIES = 2


def runtime_speed_peaks(model, diagnostics, fps):
    """Peak physical joint speeds, including motion between output frames."""
    velocities = diagnostics.get("substep_qvel")
    times = np.asarray(diagnostics.get("substep_time_s", []), float)
    if (times.ndim != 1 or not np.isfinite(times).all()
            or np.any(np.diff(np.r_[0., times]) <= 0)
            or not np.isfinite(fps) or fps <= 0):
        raise ValueError("physical arm timestamps must be finite and increase from zero; fps must be positive")
    source = "substep_qvel"
    initial = np.asarray(diagnostics["initial_qvel"], float)
    if velocities is None or not len(velocities):
        positions = diagnostics.get("substep_qpos")
        source = "substep_qpos/time_s"
        if positions is not None and len(positions):
            Q = np.vstack([diagnostics["initial_qpos"], positions])
            dt = np.diff(np.r_[0., times])
            if np.any(dt <= 0):
                raise ValueError("physical arm timestamps must increase")
            # Arm joints are scalar hinge joints; only their columns are used.
            velocities = np.zeros((len(dt), model.nv))
            for part in E.robot_parts(model).values():
                velocities[:, part["arm_dadr"]] = np.diff(Q[:, part["arm_qadr"]], axis=0) / dt[:, None]
        elif len(diagnostics.get("qpos", [])) > 1:
            raise ValueError("physical arm speed requires integration-step evidence")
        else:
            velocities = np.empty((0, model.nv))
    velocities = np.asarray(velocities, float)
    if velocities.shape != (len(times), model.nv) or initial.shape != (model.nv,):
        raise ValueError("physical arm velocity evidence does not match runtime DOFs")
    if not np.isfinite(velocities).all() or not np.isfinite(initial).all():
        raise ValueError("non-finite physical arm velocity")
    all_v = np.vstack([initial, velocities])
    all_t = np.r_[0., times]
    out = {}
    for side, part in E.robot_parts(model).items():
        speed = np.max(np.degrees(np.abs(all_v[:, part["arm_dadr"]])), axis=1)
        sample = int(np.argmax(speed))
        out[side] = {"peak_speed_deg_s": float(speed[sample]), "speed_source": source,
                     "speed_sample": sample - 1, "speed_time_s": float(all_t[sample]),
                     "speed_frame": int(np.ceil(all_t[sample] * fps))}
    return out


def retry_factor(evaluation, coupled, speed_limit_deg_s, retry_count, attempts_left):
    """Only an otherwise passing, reachable, continuous path merits a retry."""
    report = coupled.get("report", {}) if coupled else {}
    if (set(evaluation.get("failed", ())) != {"arm_speed"}
            or retry_count >= MAX_RETRIES or attempts_left <= 0
            or not report.get("ok") or report.get("infeasible")
            or any(v.get("branch_jumps") for v in report.values() if isinstance(v, dict))):
        return None
    peaks = evaluation.get("measured", {}).get("arm_speed", {})
    speed = max(peaks.values(), default=0.)
    limit = float(speed_limit_deg_s)
    if not np.isfinite(speed) or not np.isfinite(limit) or limit <= 0 or speed <= limit:
        return None
    # Integer subdivision retains every original path vertex, including corners.
    return min(3, max(2, int(np.ceil(1.05 * speed / limit))))


def slow_reference(model, coupled, fps, factor, max_frames=3000):
    """Subdivide each existing joint-space interval without changing its path.

    Both arms and fingers share one time map. Rebuild virtual wrist coordinates
    by FK; interpolating those Euler coordinates would break the fixed mount.
    """
    if factor != int(factor) or factor < 2:
        raise ValueError("timing subdivision must be an integer of at least two")
    factor = int(factor)
    report = coupled["report"]
    if (not report.get("ok") or report.get("infeasible")
            or any(v.get("branch_jumps") for v in report.values() if isinstance(v, dict))):
        raise ValueError("timing cannot repair an unreachable or discontinuous arm path")
    old = np.asarray(coupled["qpos"], float)
    count = (len(old) - 1) * factor + 1
    if len(old) < 2 or count > max_frames:
        raise ValueError(f"timing retry needs {count} frames, limit {max_frames}")
    src = np.arange(count, dtype=float) / factor
    runtime = RR.compile_rigid(model)
    physical = AP.resample(runtime.to_runtime_qpos(old), src, AP._quat_cols(runtime.model))
    Q = runtime.to_proposal_qpos(physical, seed=old[0])
    out = copy.deepcopy(coupled)
    out["qpos"] = Q
    out["labels"] = [coupled["labels"][int(i)] for i in np.floor(src)]
    out["src_index"] = np.interp(src, np.arange(len(old)), coupled.get("src_index", np.arange(len(old))))
    parts = E.robot_parts(model)
    out["arm"] = {side: Q[:, parts[side]["arm_qadr"]].copy() for side in coupled["arm"]}
    out["report"]["retimed_s"] = round(float(report.get("retimed_s", 0.)) + (count - len(old)) / fps, 3)
    for side, arm in out["arm"].items():
        out["report"][side]["peak_speed_deg_s"] = float(np.degrees(np.abs(np.diff(arm, axis=0))).max() * fps)
    return out


def scale_expectation_frames(expect, factor):
    """Move explicitly numbered local checks with the commands they describe.

    Label-based checks remain unresolved until the new labels are evaluated.
    Call on a pristine expectation, before resolve_label_frames mutates it.
    """
    out = copy.deepcopy(expect)
    for key in ("pads_at", "pads_at_rel", "slip_from", "slip_from_rel"):
        if key in out:
            out[key] = int(out[key]) * factor
    if isinstance(out.get("seated"), dict) and out["seated"].get("frames_rel"):
        out["seated"]["frames_rel"] = [int(i) * factor for i in out["seated"]["frames_rel"]]
    if out.get("hands"):
        out["hands"] = {side: scale_expectation_frames(value, factor) for side, value in out["hands"].items()}
    return out


def archive_failed_trial(directory, evaluation, coupled, factor, attempt):
    """Keep complete bad rollout evidence before its usual output is replaced."""
    directory = Path(directory)
    archive = directory / "timing_attempts" / f"attempt_{attempt:02d}"
    # A resumed run must not overwrite an earlier failed trial's evidence.
    suffix = 1
    while archive.exists():
        archive = directory / "timing_attempts" / f"attempt_{attempt:02d}_{suffix:02d}"
        suffix += 1
    archive.mkdir(parents=True)
    for name in ("rollout.npz", "rollout_runtime.npz", "keyframes.json"):
        source = directory / name
        if source.is_file():
            shutil.copy2(source, archive / name)
    record = {"evaluation": evaluation, "arm_plan": coupled["report"],
              "next_time_factor": factor, "artifact_role": "failed_timing_trial"}
    (archive / "outcome.json").write_text(json.dumps(record, indent=1, default=float))
    return str(archive)
