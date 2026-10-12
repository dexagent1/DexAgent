"""Arm branch flip audit: the arm's joints swing a long way while its flange barely moves.

    from tool_library.verifiers.arm_branch_flip_audit import arm_branch_flip_audit
    arm_branch_flip_audit("scene.xml", "trajectory.npz")

A 6-joint arm reaches a flange pose with a few discrete IK solutions (elbow up / down, wrist
flipped). An arm fitted frame by frame can hop from one solution to another: the hand stays
where it is, the arm swings through a large joint motion that serves nothing and that the real
arm performs at speed next to the table and the other station. Done quickly it shows up as a
joint speed or weld error; done over a second it passes every per-frame check.
"""
from __future__ import annotations

TOOL = {
    "name": "arm_branch_flip_audit",
    "title": "Arm branch flip audit",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "Windows in which an arm's joints travel far (an IK elbow or wrist branch switch) while its flange stays "
               "nearly still in position and orientation.",
    "description": (
        "Covers what arm_reach_check (joint speed above 180 deg/s, weld residual) and joint_speed_check (per-frame "
        "budget) miss: a slow branch switch that respects every per-frame speed. For each arm (bench stations via "
        "episode.robot_parts; other robots: joints on bodies whose name contains '_arm') it slides windows of each "
        "length in window_s over the episode and measures the largest joint excursion inside the window (max - min of "
        "any one joint, degrees) against the flange's largest displacement from its pose at the window start "
        "(millimetres and degrees). A window where a joint swings at least min_joint_deg while the flange stays "
        "within max_flange_mm and max_flange_deg is a branch switch (the configuration changed, the end effector did "
        "not). Default 45 degrees against 30 mm and 15 degrees, windows 0.5 s and 2 s. Reports the worst window per "
        "arm with the joint that swung, its frames and phase labels, and the sign changes of each joint inside it "
        "(an elbow or wrist joint crossing zero is the usual signature)."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the episode was executed in."},
        {"name": "trajectory", "type": "str", "description": "trajectory.npz (qpos [T, nq], fps, labels)."},
        {"name": "frames", "type": "[first, last] | None", "description": "Inclusive frame window. Default all."},
        {"name": "window_s", "type": "list[float]", "description": "Window lengths in seconds. Default [0.5, 2.0]."},
        {"name": "min_joint_deg", "type": "float", "description": "Joint excursion that counts as a swing. Default 45."},
        {"name": "max_flange_mm, max_flange_deg", "type": "float", "description": "Flange motion under which the swing served nothing. Defaults 30 mm, 15 deg."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; worst {side, frames, labels, joint, joint_excursion_deg, flange_mm, flange_deg}; arms {side: worst window measurements}."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "audit", "arm", "ik", "branch", "elbow", "wrist"],
    "status": "tested",
    "entrypoint": "arm_branch_flip_audit",
}

import numpy as np

from tool_library.common import episode_audit as EA


def arm_branch_flip_audit(scene, trajectory, frames=None, window_s=(0.5, 2.0), min_joint_deg=45.0, max_flange_mm=30.0,
                          max_flange_deg=15.0):
    EA.check_positive(min_joint_deg=min_joint_deg, max_flange_mm=max_flange_mm, max_flange_deg=max_flange_deg)
    ws = [float(w) for w in (window_s if isinstance(window_s, (list, tuple)) else [window_s])]
    if not ws or min(ws) <= 0:
        raise ValueError("window_s must be one or more positive durations in seconds")
    ep = EA.Episode(scene, trajectory, frames)
    m = ep.model
    arms = {s: h for s, h in ep.hands.items() if h["arm_qadr"] and h["flange"] >= 0}
    if not arms:
        return EA.finish([], None, arms={}, note="no arm with a flange body in the scene")
    out, flips = {}, []
    for side, h in arms.items():
        A = np.degrees(ep.Q[:, h["arm_qadr"]])
        P = ep.body_pos(h["flange"]); q = ep.body_quat(h["flange"])
        names = [m.joint(j).name for j in h["arm_joints"]]
        best = None
        for w_s in ws:
            W = max(2, int(round(w_s * ep.fps)))
            for a in range(ep.a, max(ep.a, ep.b - W) + 1):
                b = min(a + W, ep.b)
                seg = A[a:b + 1]
                exc = seg.max(0) - seg.min(0)
                j = int(np.argmax(exc))
                if exc[j] < 0.5 * min_joint_deg and best is not None:
                    continue
                fd = 1000 * float(np.max(np.linalg.norm(P[a:b + 1] - P[a], axis=1)))
                fr = float(np.degrees(np.max(EA.quat_angle(np.repeat(q[a:a + 1], b - a + 1, 0), q[a:b + 1]))))
                score = exc[j] / (1.0 + fd / max_flange_mm + fr / max_flange_deg)
                rec = {"side": side, "frames": [a, b], "labels": [ep.labels[a], ep.labels[b]], "window_s": w_s,
                       "joint": names[j], "joint_excursion_deg": round(float(exc[j]), 1), "flange_mm": round(fd, 1),
                       "flange_deg": round(fr, 1), "score": round(float(score), 2),
                       "sign_changes": [names[i] for i in range(len(names)) if seg[:, i].min() < 0 < seg[:, i].max()]}
                if best is None or score > best["score"]:
                    best = rec
                if exc[j] >= min_joint_deg and fd <= max_flange_mm and fr <= max_flange_deg:
                    flips.append(rec)
        out[side] = best
    issues = []
    for side in sorted({f["side"] for f in flips}):
        f = max((x for x in flips if x["side"] == side), key=lambda x: x["joint_excursion_deg"])
        issues.append(f"{side} arm, frames {f['frames'][0]}-{f['frames'][1]} ({f['labels'][0]} -> {f['labels'][1]}): "
                      f"{f['joint']} swings {f['joint_excursion_deg']} deg while the flange moves only {f['flange_mm']} mm / "
                      f"{f['flange_deg']} deg (IK branch switch"
                      + (f"; {', '.join(f['sign_changes'])} cross zero" if f["sign_changes"] else "") +
                      "); seed the arm fit of that stretch from the previous frame's configuration (continuous branch) or "
                      "move the hand target so the arm need not switch")
    worst = max(flips, key=lambda x: x["joint_excursion_deg"]) if flips else None
    return EA.finish(issues, worst, arms=out, n_flip_windows=len(flips), frames=[ep.a, ep.b])


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(arm_branch_flip_audit(sys.argv[1], sys.argv[2]), indent=1))
