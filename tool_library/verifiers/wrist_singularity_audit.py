"""Wrist singularity audit: frames where an arm runs near a singular configuration with large joint rates.

    from tool_library.verifiers.wrist_singularity_audit import wrist_singularity_audit
    wrist_singularity_audit("scene.xml", "trajectory.npz")

Near a singularity (wrist axes aligned, elbow straight) the arm's Jacobian loses rank: a
small flange motion needs large joint motion, and a real arm controller saturates or wobbles
there. Passing near a singularity slowly is harmless; passing it with large joint rates is not.
"""
from __future__ import annotations

TOOL = {
    "name": "wrist_singularity_audit",
    "title": "Wrist singularity audit",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "Frames where an arm's flange Jacobian is near singular (smallest singular value) while its joints move fast "
               "or amplify the flange motion many times.",
    "description": (
        "Per-frame Jacobian measure, distinct from arm_branch_flip_audit (windows of large joint motion with a still "
        "flange) and from the speed checks (which do not know why a joint is fast). For each arm it computes the "
        "6 x n Jacobian of the flange body over the arm joints, rotation rows scaled by rot_scale_m (default 0.3 m, "
        "so a radian of flange rotation weighs like 30 cm of travel), and its smallest singular value sigma_min "
        "(metres per radian). A frame is near-singular when sigma_min < smin_m (default 0.01 m). It fails when, on "
        "a near-singular frame, the fastest arm joint exceeds rate_deg_s (default 90 deg/s, about the rate the "
        "arm fill caps at) or the joint speed per unit of flange twist exceeds amp_rad_per_m (default 60 rad per "
        "metre: the joints race while the flange crawls; the paper's executed episodes stay under 25). Reports the "
        "worst frame per arm with sigma_min, joint rate, flange speed, phase label, and how many frames run near "
        "singular in total."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the episode was executed in."},
        {"name": "trajectory", "type": "str", "description": "trajectory.npz (qpos [T, nq], fps, labels)."},
        {"name": "frames", "type": "[first, last] | None", "description": "Inclusive frame window. Default all."},
        {"name": "smin_m", "type": "float", "description": "sigma_min below which a frame is near-singular. Default 0.01."},
        {"name": "rate_deg_s", "type": "float", "description": "Arm joint speed that fails on a near-singular frame. Default 90."},
        {"name": "amp_rad_per_m", "type": "float", "description": "Joint speed / flange twist speed that fails on a near-singular frame. Default 60."},
        {"name": "rot_scale_m", "type": "float", "description": "Weight of flange rotation against translation. Default 0.3."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; worst {side, frame, label, sigma_min_m, joint_rate_deg_s, amplification, flange_speed}; arms {side: min sigma, frames near singular, worst frame}."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "audit", "arm", "singularity", "jacobian", "deployability"],
    "status": "tested",
    "entrypoint": "wrist_singularity_audit",
}

import mujoco
import numpy as np

from tool_library.common import episode_audit as EA


def wrist_singularity_audit(scene, trajectory, frames=None, smin_m=0.01, rate_deg_s=90.0, amp_rad_per_m=60.0, rot_scale_m=0.3):
    EA.check_positive(smin_m=smin_m, rate_deg_s=rate_deg_s, amp_rad_per_m=amp_rad_per_m, rot_scale_m=rot_scale_m)
    ep = EA.Episode(scene, trajectory, frames)
    m = ep.model
    arms = {s: h for s, h in ep.hands.items() if h["arm_qadr"] and h["flange"] >= 0}
    if not arms:
        return EA.finish([], None, arms={}, note="no arm with a flange body in the scene")
    d = mujoco.MjData(m)
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    events, out = [], {}
    for side, h in arms.items():
        dadr = [int(m.jnt_dofadr[j]) for j in h["arm_joints"]]
        A = ep.Q[:, h["arm_qadr"]]
        rate = np.degrees(np.abs(np.gradient(A, axis=0)).max(axis=1)) * ep.fps if ep.T > 1 else np.zeros(ep.T)
        P = ep.body_pos(h["flange"]); q = ep.body_quat(h["flange"])
        v = np.linalg.norm(np.gradient(P, axis=0), axis=1) * ep.fps if ep.T > 1 else np.zeros(ep.T)
        w = np.zeros(ep.T)
        if ep.T > 1:
            st = EA.quat_angle(q[:-1], q[1:]) * ep.fps
            w = np.r_[st[0], 0.5 * (st[:-1] + st[1:]), st[-1]]
        twist = v + rot_scale_m * w
        smin = np.full(ep.T, np.inf)
        near = []
        for t in ep.window:
            d.qpos[:] = ep.Q[t]
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            mujoco.mj_jacBody(m, d, jp, jr, h["flange"])
            J = np.vstack([jp[:, dadr], rot_scale_m * jr[:, dadr]])
            smin[t] = float(np.linalg.svd(J, compute_uv=False)[-1])
            if smin[t] < smin_m:
                near.append(t)
                amp = np.radians(rate[t]) / max(twist[t], 1e-3)
                if rate[t] > rate_deg_s or (amp > amp_rad_per_m and rate[t] > 30.0):
                    events.append({"side": side, "frame": t, "label": ep.labels[t], "sigma_min_m": round(smin[t], 5),
                                   "joint_rate_deg_s": round(float(rate[t]), 1), "amplification": round(float(amp), 1),
                                   "flange_speed_m_s": round(float(v[t]), 4), "flange_rate_deg_s": round(float(np.degrees(w[t])), 1)})
        k = min(ep.window, key=lambda t: smin[t])
        out[side] = {"min_sigma_m": round(float(smin[k]), 5), "min_sigma_frame": int(k), "min_sigma_label": ep.labels[k],
                     "frames_near_singular": len(near), "rate_at_min_deg_s": round(float(rate[k]), 1)}
    issues = []
    for side in sorted({e["side"] for e in events}):
        evs = [e for e in events if e["side"] == side]
        e = max(evs, key=lambda x: max(x["joint_rate_deg_s"] / rate_deg_s, x["amplification"] / amp_rad_per_m))
        issues.append(f"{side} arm near a singularity (sigma_min {e['sigma_min_m']} m) at frame {e['frame']} ({e['label']}) with "
                      f"joints at {e['joint_rate_deg_s']} deg/s for a flange moving {1000 * e['flange_speed_m_s']:.0f} mm/s "
                      f"(x{e['amplification']} amplification; {len(evs)} such frames): move the hand target or the approach so "
                      f"the arm stays away from the aligned-axis pose, or slow that stretch")
    worst = max(events, key=lambda x: max(x["joint_rate_deg_s"] / rate_deg_s, x["amplification"] / amp_rad_per_m)) if events else None
    return EA.finish(issues, worst, arms=out, n_events=len(events), frames=[ep.a, ep.b])


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(wrist_singularity_audit(sys.argv[1], sys.argv[2]), indent=1))
