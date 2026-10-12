"""Robot joint jump audit: single-frame discontinuities in any robot joint.

    from tool_library.verifiers.robot_joint_jump_audit import robot_joint_jump_audit
    robot_joint_jump_audit("scene.xml", "trajectory.npz")

joint_speed_check holds each joint group to an absolute per-frame budget and arm_jerk_check
holds the arm's second difference to 3 deg/frame^2. Neither sees a step that is small in
absolute terms but sits alone among still neighbours (a seam where two planned segments were
concatenated, an IK re-solve on one frame), nor a glitch frame on a finger or a floating
wrist, which have no jerk budget at all.
"""
from __future__ import annotations

TOOL = {
    "name": "robot_joint_jump_audit",
    "title": "Robot joint jump audit",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "Single-frame discontinuities in arm, wrist and finger joints: isolated steps much larger than both "
               "neighbouring steps, and glitch frames displaced off their neighbours and put back.",
    "description": (
        "Covers what joint_speed_check (absolute per-frame budget) and arm_jerk_check (arm second difference "
        "only) miss: discontinuities measured against the joint's own neighbouring motion, on every robot joint "
        "including the fingers and a floating hand's virtual wrist. Two rules per joint: an isolated step, a "
        "frame-to-frame change at least `ratio` times larger than the larger of the two neighbouring steps and "
        "above the group's floor (the seam of two concatenated segments, a one-frame re-solve: a velocity that "
        "switches on and off within one frame, which no servo with finite acceleration follows); and a glitch, "
        "a frame that sits off the midpoint of its neighbours by more than the floor and more than glitch_ratio "
        "times the neighbours' half-distance (displaced for one frame, then back). Floors are per group at 30 fps "
        "(arm 1.5 deg, wrist rotation 1.5 deg, wrist translation 3 mm, fingers 6 deg), set well above what a "
        "physics-executed servo produces when a contact makes or breaks, and scale with the frame rate for steps. "
        "Groups come from the bench hands (episode.robot_parts) or, for other robots, from the side word and "
        "'_arm' in the body names. Reports the worst event per group with frame, phase label and joint."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the episode was executed in."},
        {"name": "trajectory", "type": "str", "description": "trajectory.npz (qpos [T, nq], fps, labels)."},
        {"name": "frames", "type": "[first, last] | None", "description": "Inclusive frame window. Default all."},
        {"name": "ratio", "type": "float", "description": "Step / larger neighbouring step above which a step is isolated. Default 10."},
        {"name": "glitch_ratio", "type": "float", "description": "Offset / neighbours' half-distance above which a frame is a glitch. Default 3."},
        {"name": "min_step", "type": "dict | None", "description": "Floors at 30 fps: arm_deg, wrist_deg, wrist_mm, finger_deg. Defaults 1.5, 1.5, 3.0, 6.0."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; worst {frame, label, joint, kind, value}; groups {group: max isolated ratio, max glitch, events}; events (up to 30)."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "audit", "smoothness", "seam", "glitch", "fingers", "arm"],
    "status": "tested",
    "entrypoint": "robot_joint_jump_audit",
}

import numpy as np

from tool_library.common import episode_audit as EA

DEFAULT_MIN_STEP = dict(arm_deg=1.5, wrist_deg=1.5, wrist_mm=3.0, finger_deg=6.0)


def robot_joint_jump_audit(scene, trajectory, frames=None, ratio=10.0, glitch_ratio=3.0, min_step=None):
    EA.check_positive(ratio=ratio, glitch_ratio=glitch_ratio)
    fl = dict(DEFAULT_MIN_STEP)
    if min_step:
        unknown = set(min_step) - set(fl)
        if unknown:
            raise ValueError(f"min_step: unknown keys {sorted(unknown)}; valid {sorted(fl)}")
        fl.update(min_step)
    ep = EA.Episode(scene, trajectory, frames)
    m, Q, sc = ep.model, ep.Q, 30.0 / ep.fps
    joints = EA.robot_scalar_joints(ep)
    if not joints:
        raise ValueError(f"scene: no robot joints (bodies starting with '{ep.robot_prefix}')")
    events, groups = [], {}
    for j, qa, name, group in joints:
        kind = group.rsplit("_", 1)[-1]
        slide = int(m.jnt_type[j]) == EA.SLIDE
        if kind == "fingers":
            floor, unit, k = np.radians(fl["finger_deg"]), "deg", np.degrees(1.0)
        elif kind == "wrist" and slide:
            floor, unit, k = fl["wrist_mm"] / 1000.0, "mm", 1000.0
        elif kind == "wrist":
            floor, unit, k = np.radians(fl["wrist_deg"]), "deg", np.degrees(1.0)
        else:
            floor, unit, k = (fl["wrist_mm"] / 1000.0, "mm", 1000.0) if slide else (np.radians(fl["arm_deg"]), "deg", np.degrees(1.0))
        x = Q[:, qa]
        st = np.abs(np.diff(x))
        g = groups.setdefault(group, {"max_isolated_ratio": 0.0, "max_glitch": 0.0, "events": 0, "unit": unit})
        for t in range(ep.a, min(ep.b, ep.T - 2) + 1):          # step t -> t+1
            s = st[t]
            if s < floor * sc:
                continue
            nb = []
            if t >= 1:
                nb.append(st[t - 1])
            if t + 1 < len(st):
                nb.append(st[t + 1])
            if not nb:
                continue
            r = s / max(max(nb), 1e-9)
            g["max_isolated_ratio"] = max(g["max_isolated_ratio"], round(float(min(r, 1e6)), 1))
            if r >= ratio:
                events.append({"frame": t + 1, "label": ep.labels[t + 1], "joint": name, "group": group, "kind": "isolated_step",
                               "value": EA.r3(s * k, 2), "unit": unit, "ratio": EA.r3(min(r, 1e6), 1),
                               "neighbour_steps": [EA.r3(v * k, 3) for v in nb]})
        for t in range(max(ep.a, 1), min(ep.b, ep.T - 2) + 1):
            dev = abs(x[t] - 0.5 * (x[t - 1] + x[t + 1]))
            half = 0.5 * abs(x[t + 1] - x[t - 1])
            g["max_glitch"] = max(g["max_glitch"], round(float(dev * k), 3))
            if dev >= floor and dev >= glitch_ratio * half and (x[t] - x[t - 1]) * (x[t + 1] - x[t]) < 0:
                events.append({"frame": t, "label": ep.labels[t], "joint": name, "group": group, "kind": "glitch",
                               "value": EA.r3(dev * k, 2), "unit": unit, "neighbours_apart": EA.r3(2 * half * k, 3)})
    for e in events:
        groups[e["group"]]["events"] += 1
    issues = []
    for grp in sorted({e["group"] for e in events}):
        evs = [e for e in events if e["group"] == grp]
        w = max(evs, key=lambda e: e["value"])
        frames_hit = sorted({e["frame"] for e in evs})
        what = ("stepped %s %s in one frame while its neighbouring steps were %s" % (w["value"], w["unit"], w["neighbour_steps"])
                if w["kind"] == "isolated_step" else
                "sits %s %s off its neighbours for one frame" % (w["value"], w["unit"]))
        issues.append(f"{grp}: {w['joint']} at frame {w['frame']} ({w['label']}) {what} ({len(frames_hit)} frame(s): "
                      f"{frames_hit[:8]}); blend the segments at that seam (re-plan the join from the executed state) or "
                      f"re-solve that frame's IK from its neighbour")
    events.sort(key=lambda e: -(e["value"] or 0))
    return EA.finish(issues, events[0] if events else None, groups=groups, events=events[:30], n_events=len(events),
                     frames=[ep.a, ep.b], fps=ep.fps, floors_at_30fps=fl)


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(robot_joint_jump_audit(sys.argv[1], sys.argv[2]), indent=1))
