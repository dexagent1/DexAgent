"""Hand joint rate audit: finger chatter and finger accelerations.

    from tool_library.verifiers.hand_joint_rate_audit import hand_joint_rate_audit
    hand_joint_rate_audit("scene.xml", "trajectory.npz")

joint_speed_check already holds every finger joint to 0.30 rad per frame at 30 fps, so finger
SPEED is covered. What it cannot see is a finger that oscillates (a servo chattering against a
contact, a grip that toggles every frame) at a speed under that budget, or a finger that
reverses at full speed (an acceleration no finger motor delivers).
"""
from __future__ import annotations

TOOL = {
    "name": "hand_joint_rate_audit",
    "title": "Hand joint rate audit",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "Finger joint chatter (rapid back-and-forth reversals) and finger joint accelerations; finger speed itself "
               "is joint_speed_check's.",
    "description": (
        "joint_speed_check covers finger speed (0.30 rad/frame at 30 fps), so this audit covers what it misses on "
        "the same joints. Chatter: for every finger joint it marks the frames where the joint reverses direction "
        "with both the step before and the step after larger than amp_rad (default 0.01 rad, above servo noise), and "
        "counts reversals inside every window of window_s (default 0.5 s); max_reversals or more (default 6, i.e. "
        "six turnarounds in half a second) is a finger vibrating against a contact or a grip command toggling, "
        "which on hardware is audible buzzing and wears the tendons. Acceleration: the raw second difference of "
        "each finger joint against max_acc_rad_s2 (default 150 rad/s^2; the paper's executed episodes peak near 70 "
        "when a contact makes or breaks): a finger that reverses at full speed in one frame. Fingers come from the "
        "bench hands (episode.robot_parts) or, for other robots, the actuated joints of the hand bodies. Reports "
        "per hand the worst joint for each rule with its frames and phase label."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the episode was executed in."},
        {"name": "trajectory", "type": "str", "description": "trajectory.npz (qpos [T, nq], fps, labels)."},
        {"name": "frames", "type": "[first, last] | None", "description": "Inclusive frame window. Default all."},
        {"name": "amp_rad", "type": "float", "description": "Smallest step on both sides of a reversal that counts (at 30 fps). Default 0.01."},
        {"name": "window_s", "type": "float", "description": "Window for counting reversals. Default 0.5."},
        {"name": "max_reversals", "type": "int", "description": "Reversals in one window that make chatter. Default 6."},
        {"name": "max_acc_rad_s2", "type": "float", "description": "Raw second-difference limit. Default 150."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; worst {side, joint, kind, frame, label, value}; hands {side: max reversals per window, peak acceleration, joints}."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "audit", "fingers", "chatter", "acceleration", "smoothness"],
    "status": "tested",
    "entrypoint": "hand_joint_rate_audit",
}

import numpy as np

from tool_library.common import episode_audit as EA


def hand_joint_rate_audit(scene, trajectory, frames=None, amp_rad=0.01, window_s=0.5, max_reversals=6, max_acc_rad_s2=150.0):
    EA.check_positive(amp_rad=amp_rad, window_s=window_s, max_reversals=max_reversals, max_acc_rad_s2=max_acc_rad_s2)
    ep = EA.Episode(scene, trajectory, frames)
    m = ep.model
    hands = {s: h for s, h in ep.hands.items() if h["finger_qadr"]}
    if not hands:
        return EA.finish([], None, hands={}, note="no finger joints in the scene")
    amp = amp_rad * 30.0 / ep.fps
    W = max(3, int(round(window_s * ep.fps)))
    events, out = [], {}
    for side, h in hands.items():
        F = ep.Q[ep.a:ep.b + 1][:, h["finger_qadr"]]
        jn = {int(m.jnt_qposadr[j]): m.joint(j).name for j in h["finger_joints"]}
        names = [jn.get(q, f"qpos{q}") for q in h["finger_qadr"]]
        st = np.diff(F, axis=0)
        best_rev, best_acc = None, None
        for i in range(F.shape[1]):
            s = st[:, i]
            rev = np.zeros(len(s), bool)
            for k in range(1, len(s)):
                if abs(s[k]) > amp and abs(s[k - 1]) > amp and s[k] * s[k - 1] < 0:
                    rev[k] = True
            if len(rev) >= W:
                cnt = np.convolve(rev.astype(int), np.ones(W, int), "valid")
                k = int(np.argmax(cnt)); c = int(cnt[k])
            else:
                k, c = 0, int(rev.sum())
            if best_rev is None or c > best_rev["value"]:
                best_rev = {"joint": names[i], "value": c, "frame": ep.a + k, "label": ep.labels[ep.a + k]}
            if c >= max_reversals:
                events.append({"side": side, "joint": names[i], "kind": "chatter", "value": c, "unit": "reversals",
                               "frame": ep.a + k, "label": ep.labels[ep.a + k], "window": [ep.a + k, ep.a + k + W]})
            if len(F) >= 3:
                acc = np.abs(np.diff(F[:, i], n=2)) * ep.fps ** 2
                k2 = int(np.argmax(acc)); v = float(acc[k2])
                if best_acc is None or v > best_acc["value"]:
                    best_acc = {"joint": names[i], "value": round(v, 1), "frame": ep.a + k2 + 1, "label": ep.labels[ep.a + k2 + 1]}
                if v > max_acc_rad_s2:
                    events.append({"side": side, "joint": names[i], "kind": "acceleration", "value": round(v, 1), "unit": "rad/s^2",
                                   "frame": ep.a + k2 + 1, "label": ep.labels[ep.a + k2 + 1],
                                   "frames_over": int((acc > max_acc_rad_s2).sum())})
        out[side] = {"max_reversals_per_window": best_rev, "peak_acceleration": best_acc}
    issues = []
    for (side, kind) in sorted({(e["side"], e["kind"]) for e in events}):
        evs = [e for e in events if e["side"] == side and e["kind"] == kind]
        w = max(evs, key=lambda e: e["value"])
        if kind == "chatter":
            issues.append(f"{side} hand: {w['joint']} reverses {w['value']} times in {W} frames from frame {w['frame']} "
                          f"({w['label']}) and {len(evs) - 1} more joints chatter: the finger vibrates against its contact or "
                          f"the grip command toggles; hold one finger target through the phase (or lower the servo gain "
                          f"against the contact)")
        else:
            issues.append(f"{side} hand: {w['joint']} accelerates at {w['value']} rad/s^2 at frame {w['frame']} ({w['label']}), "
                          f"limit {max_acc_rad_s2}; ramp that finger command over a few frames instead of switching it")
    worst = max(events, key=lambda e: e["value"] / (max_reversals if e["kind"] == "chatter" else max_acc_rad_s2)) if events else None
    return EA.finish(issues, worst, hands=out, window_frames=W, frames=[ep.a, ep.b])


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(hand_joint_rate_audit(sys.argv[1], sys.argv[2]), indent=1))
