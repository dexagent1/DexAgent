"""Final pose check: did the object end where the demonstration left it?

    from tool_library.verifiers.final_pose_check import check_final_pose
    check_final_pose("scene.xml", "rollout.npz", body="cup", reference="demo.npz")
    check_final_pose("scene.xml", "rollout.npz", body="cup", target_pos=[0.3, 0.1, 0.02])

The task gate of a pick-and-place: the manipulated object's final position
within 10 cm of the demonstration's, and its final yaw within 30 degrees when
orientation matters. The error is measured on the last frame, never averaged
over the trajectory, because a mean tracking error hides an object that was
dropped halfway and lay still.
"""
from __future__ import annotations

TOOL = {
    "name": "final_pose_check",
    "title": "Final pose check",
    "type": "verifier",
    "category": "Task & Rollout Gates",
    "summary": "Final position (and optionally yaw or full rotation) of a body against the demonstration's or a target, judged on the last frame.",
    "description": (
        "Reads the body's pose on the last frame of the trajectory (optionally the median of "
        "the last n frames to ignore a settling jitter) and compares it with the same body's "
        "last pose in a reference trajectory or with an explicit target. Fails when the "
        "position error exceeds pos_tol (0.10 m by default, the paper's task gate) or when "
        "the rotation error exceeds rot_tol_deg, measured as yaw about the up axis for "
        "objects that stand on a table or as the full geodesic angle when full_rotation is "
        "set. Also reports the path-tracking error against the reference as information, "
        "never as the gate."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF scene."},
        {"name": "trajectory", "type": "str | ndarray [T, nq]", "description": "The rollout to judge."},
        {"name": "body", "type": "str", "description": "Body whose end pose is the task outcome."},
        {"name": "reference", "type": "str | ndarray | None", "description": "Demonstration trajectory; its last frame is the target."},
        {"name": "target_pos, target_yaw_deg", "type": "list[3] m, float | None", "description": "Explicit target instead of a reference."},
        {"name": "pos_tol", "type": "float, metres", "description": "Default 0.10."},
        {"name": "rot_tol_deg", "type": "float | None", "description": "Rotation gate; None skips rotation. Default 30."},
        {"name": "full_rotation", "type": "bool", "description": "Judge the full 3D rotation instead of yaw. Default False."},
        {"name": "settle_frames", "type": "int", "description": "Median over the last n frames. Default 1."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; final_pos, target_pos, pos_err_m, rot_err_deg, track_mean_m (vs reference), track_final_m."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["task gate", "pick and place", "rollout"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Battery Insertion", "Computer Installation"],
    "status": "tested",
    "entrypoint": "final_pose_check",
}

import json
import sys

import numpy as np

from tool_library.common import trajectory as TJ


def _yaw(R):
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


def check_final_pose(scene, trajectory, body, reference=None, target_pos=None, target_yaw_deg=None,
                     pos_tol=None, rot_tol_deg=30.0, full_rotation=False, settle_frames=1):
    model = TJ.load_model(scene)
    Q, labels, fps = TJ.as_trajectory(trajectory)
    P, R = TJ.pose_track(model, Q, TJ.body_id(model, body))
    n = max(1, min(int(settle_frames), len(P)))
    p_end, R_end = np.median(P[-n:], axis=0), R[-1]
    tol = TJ.GATES["final_pos_err_m"] if pos_tol is None else pos_tol
    rep = {"body": body, "final_pos": [round(float(v), 4) for v in p_end], "final_yaw_deg": round(_yaw(R_end), 2)}
    R_t = None
    if reference is not None:
        Qr = TJ.as_trajectory(reference)[0]
        Pr, Rr = TJ.pose_track(model, Qr, TJ.body_id(model, body))
        target_pos, R_t = Pr[-1], Rr[-1]
        m = min(len(P), len(Pr))
        rep["track_mean_m"] = round(float(np.linalg.norm(P[:m] - Pr[:m], axis=1).mean()), 4)
        rep["track_final_m"] = round(float(np.linalg.norm(P[-1] - Pr[-1])), 4)
        rep["n_frames"] = [len(P), len(Pr)]
    if target_pos is None:
        raise ValueError("give a reference trajectory or target_pos")
    err = float(np.linalg.norm(p_end - np.asarray(target_pos, float)))
    rep.update(target_pos=[round(float(v), 4) for v in target_pos], pos_err_m=round(err, 4), pos_tol_m=tol)
    ok = err <= tol
    if rot_tol_deg is not None and (R_t is not None or target_yaw_deg is not None):
        if full_rotation and R_t is not None:
            rerr = np.degrees(TJ.angle_between(R_t, R_end))
        else:
            ty = _yaw(R_t) if target_yaw_deg is None else float(target_yaw_deg)
            rerr = abs((_yaw(R_end) - ty + 180.0) % 360.0 - 180.0)
        rep.update(rot_err_deg=round(float(rerr), 2), rot_tol_deg=rot_tol_deg)
        ok &= rerr <= rot_tol_deg
    rep["ok"] = bool(ok)
    return rep


if __name__ == "__main__":
    rep = check_final_pose(sys.argv[1], sys.argv[2], sys.argv[3], reference=sys.argv[4])
    print(json.dumps(rep, indent=1, default=str)); sys.exit(0 if rep["ok"] else 1)


final_pose_check = check_final_pose   # the library calls every tool by its own name
