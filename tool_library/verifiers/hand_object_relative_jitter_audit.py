"""Hand-object relative jitter audit: a held object chattering in the grip.

    from tool_library.verifiers.hand_object_relative_jitter_audit import hand_object_relative_jitter_audit
    hand_object_relative_jitter_audit("scene.xml", "trajectory.npz")

Soft contacts that make and break every few milliseconds, two fingers fighting over a part, a
servo hunting against its target: the held object then vibrates relative to the hand at a
frequency no real grasp shows. It may not drift (held_object_slip_audit passes) and the hand
itself may move smoothly; only the relative motion's high-frequency part reveals it.
"""
from __future__ import annotations

TOOL = {
    "name": "hand_object_relative_jitter_audit",
    "title": "Hand-object relative jitter audit",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "High-frequency jitter of a held object relative to the hand during hold phases: the residual of the "
               "relative pose after a short smoothing filter, peak and RMS.",
    "description": (
        "held_object_slip_audit measures low-frequency drift; this audit measures the high-frequency part. For every "
        "run of phases in which a hand holds an object (labels with grip, lift, carry, hold, set down, lower, place, "
        "pour, insert, turn, twist; object named in the label or the one touching the hand on most of the run), the "
        "object's position and orientation in the hand's wrist frame are high-passed by subtracting a Savitzky-Golay "
        "fit (quadratic, smooth_frames, default 9 at 30 fps, 0.3 s), which keeps slow slip and seating and removes "
        "nothing a real grasp does. The residual's peak and RMS must stay under peak_mm / rms_mm (defaults 3 mm, 1 "
        "mm; the paper's executed episodes peak at 1.1 mm) and peak_deg / rms_deg (defaults 4 deg, 1.5 deg). Reports "
        "the worst window per hand and object."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the episode was executed in."},
        {"name": "trajectory", "type": "str", "description": "trajectory.npz (qpos [T, nq], fps, labels)."},
        {"name": "frames", "type": "[first, last] | None", "description": "Inclusive frame window. Default all."},
        {"name": "peak_mm, rms_mm", "type": "float", "description": "Allowed positional jitter. Defaults 3, 1."},
        {"name": "peak_deg, rms_deg", "type": "float", "description": "Allowed rotational jitter. Defaults 4, 1.5."},
        {"name": "smooth_frames", "type": "int", "description": "Savitzky-Golay window at 30 fps (odd). Default 9."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; worst {side, object, frames, labels, peak_mm, rms_mm, peak_deg, rms_deg, peak_frame}; windows."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "audit", "grasp", "jitter", "chatter", "contact"],
    "status": "tested",
    "entrypoint": "hand_object_relative_jitter_audit",
}

import numpy as np
from scipy.signal import savgol_filter

from tool_library.common import episode_audit as EA


def _rotvecs(R):
    """Relative orientations as rotation vectors about the window's mean orientation (small angles)."""
    from scipy.spatial.transform import Rotation as Rot
    r = Rot.from_matrix(R)
    mean = r.mean()
    return (mean.inv() * r).as_rotvec()


def hand_object_relative_jitter_audit(scene, trajectory, frames=None, peak_mm=3.0, rms_mm=1.0, peak_deg=4.0, rms_deg=1.5,
                                      smooth_frames=9):
    EA.check_positive(peak_mm=peak_mm, rms_mm=rms_mm, peak_deg=peak_deg, rms_deg=rms_deg, smooth_frames=smooth_frames)
    ep = EA.Episode(scene, trajectory, frames)
    W = max(5, int(round(smooth_frames * ep.fps / 30.0)) | 1)
    wins = EA.held_windows(ep, words=EA.HOLD_WORDS, merge=True, min_frames=W + 2)
    rows, cache = [], {}
    for side, ob, a, b in wins:
        key = (side, ob)
        if key not in cache:
            cache[key] = EA.relative_pose(ep, ep.objects[ob]["body"], ep.hands[side]["root"])
        p, R = cache[key]
        pw = p[a:b + 1]
        rv = _rotvecs(R[a:b + 1])
        hp = pw - savgol_filter(pw, W, 2, axis=0)
        hr = rv - savgol_filter(rv, W, 2, axis=0)
        e = np.linalg.norm(hp, axis=1) * 1000
        er = np.degrees(np.linalg.norm(hr, axis=1))
        k = int(np.argmax(e / peak_mm + er / peak_deg))
        r = {"side": side, "object": ob, "frames": [a, b], "labels": [ep.labels[a], ep.labels[b]],
             "peak_mm": round(float(e.max()), 2), "rms_mm": round(float(np.sqrt(np.mean(e ** 2))), 2),
             "peak_deg": round(float(er.max()), 2), "rms_deg": round(float(np.sqrt(np.mean(er ** 2))), 2), "peak_frame": a + k}
        r["fail"] = bool(r["peak_mm"] > peak_mm or r["rms_mm"] > rms_mm or r["peak_deg"] > peak_deg or r["rms_deg"] > rms_deg)
        rows.append(r)
    bad = [r for r in rows if r["fail"]]
    issues = [f"{r['object']} jitters in the {r['side']} hand during frames {r['frames'][0]}-{r['frames'][1]} ({r['labels'][0]}): "
              f"{r['peak_mm']} mm peak / {r['rms_mm']} mm RMS, {r['peak_deg']} deg peak / {r['rms_deg']} deg RMS around its smooth path "
              f"(worst at frame {r['peak_frame']}): the contacts chatter; settle the grip (fewer fingers fighting, finger targets "
              f"just past the surface, softer pad contact) before moving" for r in bad]
    worst = max(bad, key=lambda r: max(r["peak_mm"] / peak_mm, r["peak_deg"] / peak_deg)) if bad else None
    return EA.finish(issues, worst, windows=rows, n_windows=len(rows), smooth_frames=W,
                     note=None if rows else "no hold phase naming (or touching) a free object", frames=[ep.a, ep.b])


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(hand_object_relative_jitter_audit(sys.argv[1], sys.argv[2]), indent=1))
