"""Joint acceleration audit: sustained arm joint and flange accelerations a real arm cannot deliver.

    from tool_library.verifiers.joint_acceleration_audit import joint_acceleration_audit
    joint_acceleration_audit("scene.xml", "trajectory.npz")

arm_jerk_check already bounds the raw second difference of each arm joint (3 deg/frame^2 at
30 fps, about 47 rad/s^2): a single-frame spike budget. A command can stay under that spike
budget on every frame and still ask the arm for 20-40 rad/s^2 for a third of a second, which
the motors do not deliver, and the payload in the hand feels the flange's acceleration, which
no joint-space check bounds.
"""
from __future__ import annotations

TOOL = {
    "name": "joint_acceleration_audit",
    "title": "Joint acceleration audit",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "Sustained (smoothed over about 0.3 s) arm joint accelerations against a per-joint limit, and the flange's "
               "linear acceleration against a payload limit.",
    "description": (
        "Covers what arm_jerk_check (raw single-frame second difference, 47 rad/s^2 spike budget) misses: "
        "acceleration that is kept, not spiked. Each arm joint trajectory is differentiated twice with a "
        "Savitzky-Golay filter (cubic, window window_s, default 0.3 s), which removes frame noise and keeps what a "
        "controller must actually deliver, and compared with limit_rad_s2 (default 15 rad/s^2 for every joint, or "
        "a per-joint list): a small arm's motors accelerate a joint at about that rate; the paper's executed episodes "
        "stay under 10. The flange (the arm tip the hand is bolted to) gets the same smoothing in Cartesian space "
        "against flange_limit_m_s2 (default 6 m/s^2): what a held object feels. Arms come from the bench stations "
        "(episode.robot_parts) or, for other robots, joints on bodies named '_arm'. Reports per arm and joint the "
        "peak smoothed acceleration with its frame and phase label."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the episode was executed in."},
        {"name": "trajectory", "type": "str", "description": "trajectory.npz (qpos [T, nq], fps, labels)."},
        {"name": "frames", "type": "[first, last] | None", "description": "Inclusive frame window. Default all."},
        {"name": "limit_rad_s2", "type": "float | list[float]", "description": "Per-joint (or one) smoothed acceleration limit. Default 15."},
        {"name": "flange_limit_m_s2", "type": "float | None", "description": "Smoothed flange linear acceleration limit; None skips. Default 6."},
        {"name": "window_s", "type": "float", "description": "Savitzky-Golay window. Default 0.3 s."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; worst {side, joint, frame, label, value, limit}; arms {side: {joint: peak rad/s^2, frame}, flange_peak_m_s2}."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "audit", "acceleration", "arm", "deployability", "smoothness"],
    "status": "tested",
    "entrypoint": "joint_acceleration_audit",
}

import numpy as np
from scipy.signal import savgol_filter

from tool_library.common import episode_audit as EA


def joint_acceleration_audit(scene, trajectory, frames=None, limit_rad_s2=15.0, flange_limit_m_s2=6.0, window_s=0.3):
    EA.check_positive(window_s=window_s)
    if flange_limit_m_s2 is not None:
        EA.check_positive(flange_limit_m_s2=flange_limit_m_s2)
    ep = EA.Episode(scene, trajectory, frames)
    m = ep.model
    arms = {s: h for s, h in ep.hands.items() if h["arm_qadr"]}
    if not arms:
        return EA.finish([], None, arms={}, note="no arm joints in the scene")
    W = max(5, int(round(window_s * ep.fps)) | 1)
    if ep.T < W + 2:
        return EA.finish([], None, arms={}, note=f"episode shorter than the {W}-frame smoothing window")
    events, out = [], {}
    for side, h in arms.items():
        lim = np.atleast_1d(np.asarray(limit_rad_s2, float))
        if len(lim) == 1:
            lim = np.full(len(h["arm_qadr"]), float(lim[0]))
        elif len(lim) != len(h["arm_qadr"]):
            raise ValueError(f"limit_rad_s2: give one value or {len(h['arm_qadr'])} (one per arm joint)")
        if (lim <= 0).any():
            raise ValueError("limit_rad_s2 must be positive")
        A = ep.Q[:, h["arm_qadr"]]
        acc = np.abs(savgol_filter(A, W, 3, deriv=2, axis=0)) * ep.fps ** 2
        rec = {}
        for i, j in enumerate(h["arm_joints"]):
            seg = acc[ep.a:ep.b + 1, i]
            k = int(np.argmax(seg)); t = ep.a + k
            name = m.joint(j).name
            rec[name] = {"peak_rad_s2": round(float(seg[k]), 2), "frame": t, "limit": float(lim[i])}
            over = np.where(seg > lim[i])[0]
            if len(over):
                events.append({"side": side, "joint": name, "frame": t, "label": ep.labels[t], "value": round(float(seg[k]), 2),
                               "limit": float(lim[i]), "unit": "rad/s^2", "frames_over": int(len(over)),
                               "first_over": int(ep.a + over[0])})
        if flange_limit_m_s2 is not None and h["flange"] >= 0:
            P = ep.body_pos(h["flange"])
            fa = np.linalg.norm(savgol_filter(P, W, 3, deriv=2, axis=0), axis=1)[ep.a:ep.b + 1] * ep.fps ** 2
            k = int(np.argmax(fa)); t = ep.a + k
            rec["flange_peak_m_s2"] = round(float(fa[k]), 3)
            rec["flange_peak_frame"] = t
            over = np.where(fa > flange_limit_m_s2)[0]
            if len(over):
                events.append({"side": side, "joint": "flange", "frame": t, "label": ep.labels[t], "value": round(float(fa[k]), 3),
                               "limit": float(flange_limit_m_s2), "unit": "m/s^2", "frames_over": int(len(over)),
                               "first_over": int(ep.a + over[0])})
        out[side] = rec
    issues = [f"{e['side']} {e['joint']} accelerates at {e['value']} {e['unit']} (smoothed over {W} frames) at frame "
              f"{e['frame']} ({e['label']}), limit {e['limit']} ({e['frames_over']} frames over from frame {e['first_over']}); "
              f"stretch that leg's duration or blend its start and end (a quintic leg of the same length halves the peak)"
              for e in events]
    worst = max(events, key=lambda e: e["value"] / e["limit"]) if events else None
    return EA.finish(issues, worst, arms=out, window_frames=W, frames=[ep.a, ep.b])


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(joint_acceleration_audit(sys.argv[1], sys.argv[2]), indent=1))
