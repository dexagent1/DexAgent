"""Retarget fingertip motion onto a robot hand: per-finger IK, per frame, smoothed.

    from tool_library.skills.retarget_hand_motion import retarget_hand_motion
    res = retarget_hand_motion("hand_scene.xml", hand_root="robot_right_right_hand_C_MC",
            fingertips={"thumb": "robot_right_right_thumb_DP", "index": "robot_right_right_index_DP", ...},
            targets={"thumb": tips_T_3, "index": tips_T_3, ...},   # in the hand-root frame
            scale=1.0, fps=30)
    res["qpos"], res["joint_names"], res["residual_mm"]

Human fingertip positions (from a hand mesh or keypoints, expressed in the
wrist frame) are the part of a demonstration that transfers to a robot hand
of a different size: scale them by the ratio of hand lengths, then solve
each finger's joints so its tip reaches the target, warm-started from the
previous frame, clamped to the joint limits, and smoothed. Fingertip bodies
can carry a local tip point. Fingers whose chain cannot reach a target
report the residual instead of failing.
"""
from __future__ import annotations

TOOL = {
    "name": "retarget_hand_motion",
    "title": "Retarget hand motion",
    "type": "skill",
    "category": "Trajectory & Execution",
    "summary": "Fingertip targets in the hand frame -> robot finger joint trajectory by per-finger limit-aware IK with warm starts and smoothing.",
    "description": (
        "For each named finger it builds the chain of joints between the hand root body and "
        "the fingertip body, converts the per-frame target (hand-root frame, optionally scaled "
        "about the root and offset) to world using the root's pose, and solves position-only "
        "damped least squares IK with the previous frame as the start and a pull toward the "
        "open posture in the null space. Joint values are clamped to the model's limits, "
        "smoothed with a zero-phase filter and rate-limited to max_rate_rad_s. Returns the "
        "finger joint trajectory in qpos order, the joint names, per-finger residuals in "
        "millimetres and the frames where a finger could not reach. Works with any hand "
        "whose fingers are hinge/slide chains under one root body; the wrist pose itself is "
        "handled by the arm (solve_arm_ik / attach_arms)."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "Scene with the robot hand."},
        {"name": "hand_root", "type": "str", "description": "Root body of the hand; targets are in its frame."},
        {"name": "fingertips", "type": "dict finger -> body name", "description": "Fingertip bodies."},
        {"name": "targets", "type": "dict finger -> ndarray [T, 3]", "description": "Tip positions in the hand-root frame."},
        {"name": "tip_points", "type": "dict finger -> [3] | None", "description": "Tip point in each fingertip body frame; default the body origin."},
        {"name": "scale, offset", "type": "float, [3]", "description": "Applied to targets about the root: p' = scale * p + offset."},
        {"name": "fps, max_rate_rad_s, smooth_window", "type": "float, float, int", "description": "Defaults 30, 6.0, 7."},
        {"name": "qpos_base", "type": "ndarray | None", "description": "Scene state the hand root pose is read from; default qpos0."},
    ],
    "outputs": [
        {"name": "result", "type": "dict", "description": "qpos [T, n_finger_joints], qadr, joint_names, residual_mm {finger: max, mean}, unreachable {finger: frames}, ok."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["retargeting", "hand", "fingers", "ik"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope", "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting", "Biology Experiment"],
    "status": "tested",
    "entrypoint": "retarget_hand_motion",
}

import numpy as np

from tool_library.common import trajectory as TJ


def retarget_hand_motion(scene, hand_root, fingertips, targets, tip_points=None, scale=1.0, offset=(0, 0, 0),
                         fps=30.0, max_rate_rad_s=6.0, smooth_window=7, qpos_base=None, reach_tol_m=0.004):
    import mujoco
    model = TJ.load_model(scene)
    root = TJ.body_id(model, hand_root)
    under = TJ.subtree(model, root)
    data = mujoco.MjData(model)
    if qpos_base is not None:
        data.qpos[:] = qpos_base
    mujoco.mj_kinematics(model, data)
    p_root, R_root = data.xpos[root].copy(), data.xmat[root].reshape(3, 3).copy()
    chains = {}
    for finger, body in fingertips.items():
        tip = TJ.body_id(model, body)
        joints = [j for j in TJ.chain_joints(model, tip) if int(model.jnt_bodyid[j]) in under and int(model.jnt_bodyid[j]) != root]
        if not joints:
            raise ValueError(f"finger '{finger}': no joints between {hand_root} and {body}")
        chains[finger] = TJ.Chain(model, tip, joints)
    T = len(next(iter(targets.values())))
    cols = sorted({int(a) for ch in chains.values() for a in ch.qadr})
    Q = np.zeros((T, len(cols)))
    col_of = {a: i for i, a in enumerate(cols)}
    residual, unreachable = {}, {}
    for finger, ch in chains.items():
        tp = None if not tip_points or finger not in tip_points else np.asarray(tip_points[finger], float)
        tgt = np.asarray(targets[finger], float) * scale + np.asarray(offset, float)
        rest = np.where(ch.limited, np.clip(np.zeros(len(ch.qadr)), ch.lo, ch.hi), 0.0)
        err = np.zeros(T)
        for t in range(T):
            pw = p_root + R_root @ tgt[t]
            e, _ = ch.solve(data, pw, None, point=tp, rest=rest, iters=60, damping=2e-3)
            if e > reach_tol_m:                 # retry from the open posture
                q_prev = data.qpos[ch.qadr].copy()
                data.qpos[ch.qadr] = rest
                e2, _ = ch.solve(data, pw, None, point=tp, rest=rest, iters=60, damping=2e-3)
                if e2 >= e:
                    data.qpos[ch.qadr] = q_prev
                    ch.solve(data, pw, None, point=tp, rest=rest, iters=20, damping=2e-3)
                e = min(e, e2)
            err[t] = e
            for a in ch.qadr:
                Q[t, col_of[int(a)]] = data.qpos[a]
        residual[finger] = {"max_mm": round(1000 * float(err.max()), 2), "mean_mm": round(1000 * float(err.mean()), 2)}
        bad = np.flatnonzero(err > reach_tol_m)
        if len(bad):
            unreachable[finger] = bad.tolist()
    Q = TJ.rate_limit(TJ.savgol(Q, smooth_window, 2), max_rate_rad_s / fps)
    lo = np.array([model.jnt_range[j][0] for j in range(model.njnt) if int(model.jnt_qposadr[j]) in col_of])
    hi = np.array([model.jnt_range[j][1] for j in range(model.njnt) if int(model.jnt_qposadr[j]) in col_of])
    lim = np.array([bool(model.jnt_limited[j]) for j in range(model.njnt) if int(model.jnt_qposadr[j]) in col_of])
    Q = np.where(lim, np.clip(Q, lo, hi), Q)
    names = [model.joint(j).name for j in range(model.njnt) if int(model.jnt_qposadr[j]) in col_of]
    return {"qpos": Q, "qadr": cols, "joint_names": names, "residual_mm": residual, "unreachable": unreachable,
            "ok": not unreachable, "fps": fps}
