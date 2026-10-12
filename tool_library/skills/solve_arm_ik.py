"""Inverse kinematics for any joint chain in a MuJoCo scene: one pose or a whole path.

    from tool_library.skills.solve_arm_ik import solve_arm_ik
    res = solve_arm_ik("scene.xml", body="robot_right_arm_gripper",
                       positions=P_T_3, rotations=R_T_3_3, rest=q_rest)
    res["qpos"], res["pos_err_m"], res["rot_err_deg"]

Damped least squares on the chain's own joints, limit-aware: a joint sitting
on a stop that is asked to go further is frozen for the step, so the others
take over instead of a clamp tearing the solution onto another branch. The
optional rest posture pulls the redundant directions, which is what keeps a
7-DoF arm (or a 6-DoF arm driven position-only) from drifting across a
path. Rotation can be solved fully, ignored (position-only), or weighted.
"""
from __future__ import annotations

TOOL = {
    "name": "solve_arm_ik",
    "title": "Solve arm IK",
    "type": "skill",
    "category": "Trajectory & Execution",
    "summary": "Damped-least-squares IK for the joints above any body, limit-aware with null-space rest pull, for one target or a path with warm starts and retries.",
    "description": (
        "Builds the chain of hinge and slide joints between the world and the target body, "
        "then iterates a damped least squares step on the position error (and the rotation "
        "error as an axis-angle, scaled by rot_weight) with a per-step cap. Joints on a limit "
        "that would move further out are removed from the Jacobian for that step; a hinge "
        "whose range spans a full turn is treated as unlimited. A rest posture is pulled in "
        "the null space. For a path each frame starts from the previous solution and, when a "
        "frame does not converge within the tolerances, retries from the previous frame's "
        "solution and from perturbed postures, keeping the best. A point in the body frame "
        "can be given as the controlled point (an attachment point rather than the body "
        "origin). Returns the joint trajectory and per-frame residuals. Any robot, any chain."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF scene."},
        {"name": "body", "type": "str", "description": "Body the chain ends at."},
        {"name": "positions", "type": "ndarray [T, 3] or [3]", "description": "World targets for the body (or point)."},
        {"name": "rotations", "type": "ndarray [T, 3, 3] or [3, 3] | None", "description": "World target rotations; None for position-only."},
        {"name": "point", "type": "list[3] | None", "description": "Controlled point in the body frame. Default the origin."},
        {"name": "rest", "type": "ndarray [n] or [T, n] | None", "description": "Posture pulled in the null space."},
        {"name": "q0", "type": "ndarray [n] | None", "description": "Starting joint values; default the scene's qpos0."},
        {"name": "joints", "type": "list[str] | None", "description": "Restrict the chain to these joints."},
        {"name": "pos_tol_m, rot_tol_deg, rot_weight, iters, retries", "type": "float, float, float, int, int", "description": "Defaults 0.002, 1.0, 0.2, 80, 10."},
    ],
    "outputs": [
        {"name": "result", "type": "dict", "description": "qpos [T, n] joint values in chain order, joint_names, qadr (qpos columns), pos_err_m [T], rot_err_deg [T], ok (all frames within tolerance), worst frame."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["ik", "kinematics", "arm", "path"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope", "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting", "Biology Experiment"],
    "status": "tested",
    "entrypoint": "solve_arm_ik",
}

import numpy as np

from tool_library.common import trajectory as TJ


def solve_arm_ik(scene, body, positions, rotations=None, point=None, rest=None, q0=None, joints=None,
                 pos_tol_m=0.002, rot_tol_deg=1.0, rot_weight=0.2, iters=80, retries=10, seed=0,
                 qpos_base=None):
    import mujoco
    model = TJ.load_model(scene)
    bid = TJ.body_id(model, body)
    jids = [TJ.joint_id(model, n) for n in joints] if joints else None
    ch = TJ.Chain(model, bid, jids)
    P = np.atleast_2d(np.asarray(positions, float))
    R = None if rotations is None else np.asarray(rotations, float).reshape(-1, 3, 3)
    if R is not None and len(R) == 1 and len(P) > 1:
        R = np.repeat(R, len(P), axis=0)
    T, n = len(P), len(ch.qadr)
    rest_arr = None if rest is None else np.broadcast_to(np.asarray(rest, float), (T, n))
    data = mujoco.MjData(model)
    if qpos_base is not None:
        data.qpos[:] = qpos_base
    if q0 is not None:
        data.qpos[ch.qadr] = q0
    rng = np.random.default_rng(seed)
    Q = np.zeros((T, n)); ep = np.zeros(T); er = np.zeros(T)
    tol_r = np.radians(rot_tol_deg)
    for t in range(T):
        Rt = None if R is None else R[t]
        rt = None if rest_arr is None else rest_arr[t]
        start = data.qpos[ch.qadr].copy()
        best = (ch.solve(data, P[t], Rt, point=point, rest=rt, iters=iters, rot_weight=rot_weight),
                data.qpos[ch.qadr].copy())
        seeds = ([Q[t - 1]] if t > 0 else []) + [start + rng.normal(0.0, 0.25, n) for _ in range(retries)]
        for s in seeds:
            if best[0][0] <= pos_tol_m and best[0][1] <= tol_r:
                break
            data.qpos[ch.qadr] = np.where(ch.limited, np.clip(s, ch.lo, ch.hi), s)
            err = ch.solve(data, P[t], Rt, point=point, iters=iters, rot_weight=rot_weight)
            if err[0] + 0.1 * err[1] < best[0][0] + 0.1 * best[0][1]:
                best = (err, data.qpos[ch.qadr].copy())
        data.qpos[ch.qadr] = best[1]
        Q[t], ep[t], er[t] = best[1], best[0][0], np.degrees(best[0][1])
    k = int(np.argmax(ep / pos_tol_m + er / rot_tol_deg))
    return {"qpos": Q, "joint_names": [model.joint(j).name for j in ch.joints], "qadr": ch.qadr.tolist(),
            "pos_err_m": ep, "rot_err_deg": er,
            "ok": bool(ep.max() <= pos_tol_m and er.max() <= rot_tol_deg),
            "worst": {"frame": k, "pos_err_m": round(float(ep[k]), 5), "rot_err_deg": round(float(er[k]), 3)}}
