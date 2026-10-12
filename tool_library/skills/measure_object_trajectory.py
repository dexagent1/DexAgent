"""Measure an object's path through an episode: when it moves, how far, how high.

    from tool_library.skills.measure_object_trajectory import measure_object_trajectory
    m = measure_object_trajectory("scene.xml", "trajectory.npz", body="cup")
    m["motion_window"], m["max_lift_m"], m["start_to_end_m"], m["yaw_change_deg"]

The numbers a plan is written against and a rollout is judged by: the first
and last frame of sustained motion (6 cm of displacement, four frames,
padded by three), the start and end pose, the peak lift, the total travel,
the net yaw, and the speed profile. Works on any free or articulated body in
any MuJoCo scene, from the qpos trajectory alone.
"""
from __future__ import annotations

TOOL = {
    "name": "measure_object_trajectory",
    "title": "Measure object trajectory",
    "type": "skill",
    "category": "Measurements",
    "summary": "Motion window, start and end pose, lift, travel, net yaw and speed of a body along a qpos trajectory.",
    "description": (
        "Replays the trajectory kinematically and reads the body's world pose on every frame. "
        "The motion window is the first to last frame where the body is more than "
        "motion_thresh (6 cm) from its start pose, requiring at least min_frames (4) such "
        "frames and padded by 3 frames, so fit jitter never counts as motion. Start and end "
        "are medians over the first and last 5 frames. Also returns max lift above the start, "
        "straight-line start-to-end distance, path length, net yaw change, peak speed, the "
        "largest per-frame step with its frame and phase label, and the full position track. "
        "Nothing is specific to one object or task."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF scene."},
        {"name": "trajectory", "type": "str | ndarray [T, nq]", "description": "trajectory.npz (qpos, labels, fps) or qpos array."},
        {"name": "body", "type": "str", "description": "Body to measure."},
        {"name": "motion_thresh_m, min_frames, pad", "type": "float, int, int", "description": "Motion window rule. Defaults 0.06, 4, 3."},
    ],
    "outputs": [
        {"name": "measurement", "type": "dict", "description": "moved, motion_window [first, last] (displaced from the start pose), first_move/last_move (any motion at all), start_pos, end_pos, start_to_end_m, path_length_m, max_lift_m with frame and label, yaw_change_deg, peak_speed_mps, max_step with frame/label, positions [T, 3], phases inside the window."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["measurement", "object", "trajectory", "motion window"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope", "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting", "Biology Experiment"],
    "status": "tested",
    "entrypoint": "measure_object_trajectory",
}

import json
import sys

import numpy as np

from tool_library.common import trajectory as TJ


def measure_object_trajectory(scene, trajectory, body, motion_thresh_m=0.06, min_frames=4, pad=3):
    model = TJ.load_model(scene)
    Q, labels, fps = TJ.as_trajectory(trajectory)
    P, R = TJ.pose_track(model, Q, TJ.body_id(model, body))
    n5 = min(5, len(P))
    start, end = np.median(P[:n5], axis=0), np.median(P[-n5:], axis=0)
    window = TJ.motion_window(P, motion_thresh_m, min_frames, pad)
    step = np.linalg.norm(np.diff(P, axis=0), axis=1) if len(P) > 1 else np.zeros(0)
    lift = P[:, 2] - start[2]
    k = int(np.argmax(lift))
    yaw = np.degrees(np.unwrap(np.arctan2(R[:, 1, 0], R[:, 0, 0])))
    i = int(np.argmax(step)) if len(step) else 0
    moving = np.flatnonzero(step > 0.5e-3)           # frames with more than half a millimetre of motion
    out = {"body": body, "n_frames": len(P), "fps": fps, "moved": window is not None,
           "motion_window": list(window) if window else None,
           "first_move": {"frame": int(moving[0]) + 1, "label": labels[min(int(moving[0]) + 1, len(labels) - 1)]} if len(moving) else None,
           "last_move": {"frame": int(moving[-1]) + 1, "label": labels[min(int(moving[-1]) + 1, len(labels) - 1)]} if len(moving) else None,
           "start_pos": [round(float(v), 4) for v in start], "end_pos": [round(float(v), 4) for v in end],
           "start_to_end_m": round(float(np.linalg.norm(end - start)), 4),
           "path_length_m": round(float(step.sum()), 4),
           "max_lift_m": round(float(lift[k]), 4), "max_lift_at": {"frame": k, "label": labels[k]},
           "yaw_change_deg": round(float(yaw[-1] - yaw[0]), 2),
           "peak_speed_mps": round(float(step.max() * fps), 3) if len(step) else 0.0,
           "max_step": {"m": round(float(step[i]), 4), "frame": i + 1, "label": labels[min(i + 1, len(labels) - 1)]} if len(step) else None,
           "positions": P.round(5).tolist()}
    if window:
        out["phases_in_window"] = sorted(set(labels[window[0]:window[1] + 1]), key=labels.index)
    return out


if __name__ == "__main__":
    m = measure_object_trajectory(sys.argv[1], sys.argv[2], sys.argv[3])
    m.pop("positions")
    print(json.dumps(m, indent=1))
