"""Optimize a trajectory inside time windows while every other frame stays locked to the video.

    from tool_library.skills.optimize_frame_locked import optimize_frame_locked
    best = optimize_frame_locked("scene.xml", "trajectory.npz",
        end_effector="robot_right_right_base_yaw", arm_tip="robot_right_arm_gripper",
        edits=[{"frames": ["*: descend", "*: grip"], "params": {"dz": (-0.03, 0.03), "dyaw_deg": (-15, 15)}}],
        evaluate=lambda Q: {"gates": {...}, "terms": {...}}, iterations=6, population=12)
    best["qpos"], best["params"]

A replicated episode is locked to the demonstration's timeline: the phase
boundaries are where the video has them and may not move. What can move is
the pose of the hand inside a phase: a few centimetres lower at the grasp,
a few degrees of yaw at the insertion. This skill parameterizes such edits
as rigid offsets of the end-effector pose over a frame window, applied with
a smooth bump so the frames at the window's edges are untouched, re-solves
the hand's chain (and the arm bolted to it) by IK on the edited frames, and
searches the offsets by the cross-entropy method against the caller's
gates and reward.
"""
from __future__ import annotations

TOOL = {
    "name": "optimize_frame_locked",
    "title": "Optimize frame-locked",
    "type": "skill",
    "category": "Trajectory & Execution",
    "summary": "CEM over rigid end-effector offsets inside phase windows, applied with a smooth bump and re-solved by IK, with every frame outside the windows left exactly as recorded.",
    "description": (
        "Each edit names a frame window (phase patterns or a frame range) and the offsets to "
        "search: dx, dy, dz (metres, world) and dyaw_deg (about the world up axis through the "
        "hand), each with bounds. A candidate applies offset x bump(t) across the window "
        "(bump rises and falls with a smoothstep over ramp frames, so joint speed and the "
        "frames outside are untouched), solves the end-effector chain on each edited frame "
        "and then the arm chain bolted to it (position-only when the weld carries no torque), "
        "and hands the full qpos to evaluate(Q) -> {gates, terms}. Candidates rank by failed "
        "gates then reward (optimize_reward_cem). Carried objects can be listed so they ride "
        "the hand's displacement over the window. Returns the best trajectory, its offsets "
        "and evaluation. The timeline, the phase labels and the untouched frames are "
        "preserved by construction."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF scene."},
        {"name": "trajectory", "type": "str | ndarray [T, nq]", "description": "The verified-or-nearly trajectory."},
        {"name": "end_effector", "type": "str", "description": "Hand root body (its chain is solved first)."},
        {"name": "arm_tip", "type": "str | None", "description": "Body the arm chain ends at; default found through the weld to the end effector."},
        {"name": "edits", "type": "list[{frames, params, ramp?}]", "description": "Windows and bounded offsets to search."},
        {"name": "evaluate", "type": "callable(Q) -> {gates, terms}", "description": "Verifies a candidate trajectory."},
        {"name": "carried", "type": "dict body -> [frames] | None", "description": "Free bodies that ride the hand on those frames."},
        {"name": "iterations, population, elite, seed", "type": "int", "description": "CEM controls. Defaults 6, 12, 3, 0."},
        {"name": "weights", "type": "dict | None", "description": "Reward weights per term."},
    ],
    "outputs": [
        {"name": "best", "type": "dict", "description": "qpos [T, nq], params, gates, gates_failed, terms, reward, ik_worst (pos m, rot deg), n_evals, history."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["optimization", "video-locked", "ik", "stage 3"],
    "used_in": ["Place Toy in Drawer", "Giftbox Packing", "Battery Insertion", "Computer Installation", "Scissor Cutting"],
    "status": "tested",
    "entrypoint": "optimize_frame_locked",
}

import numpy as np

from tool_library.common import trajectory as TJ
from tool_library.skills.optimize_reward_cem import optimize_reward_cem


def _window(labels, spec):
    if isinstance(spec, (list, tuple)) and len(spec) == 2 and all(isinstance(v, (int, np.integer)) for v in spec):
        return list(range(int(spec[0]), int(spec[1]) + 1))
    return TJ.frames_matching(labels, spec)


def _bump(n, ramp):
    ramp = int(min(ramp, max(1, n // 2)))
    w = np.ones(n)
    r = TJ.smoothstep(np.arange(1, ramp + 1) / (ramp + 1))
    w[:ramp], w[-ramp:] = r, r[::-1]
    return w


def apply_offsets(model, Q0, labels, end_effector, arm_tip, edits, values, carried=None, rest_pull=True):
    """Q with the offsets applied; returns (Q, worst ik error)."""
    import mujoco
    sides = TJ.discover_sides(model, {"ee": end_effector})["ee"]
    ee, fol = sides["ee"], sides["follower"]
    if arm_tip is not None and (fol is None or model.body(fol["chain"].bid).name != arm_tip):
        fol = {"chain": TJ.Chain(model, TJ.body_id(model, arm_tip)), "point": np.zeros(3), "pos_only": False}
    Q = Q0.copy()
    data = mujoco.MjData(model)
    P0, R0 = TJ.pose_track(model, Q0, ee.bid)
    tipP = tipR = None
    if fol:
        tipP, tipR = TJ.pose_track(model, Q0, fol["chain"].bid)
        tipP = tipP + np.einsum("tij,j->ti", tipR, fol["point"])
    worst = (0.0, 0.0)
    disp = {}                                   # frame -> (dR, dp) applied to the hand
    for k, ed in enumerate(edits):
        frames = _window(labels, ed["frames"])
        if not frames:
            raise ValueError(f"edit {k}: window {ed['frames']} matches no frame")
        w = _bump(len(frames), ed.get("ramp", 8))
        v = values[k]
        for wi, t in zip(w, frames):
            dR = TJ.rot_z(np.radians(v.get("dyaw_deg", 0.0)) * wi)
            dp = wi * np.array([v.get("dx", 0.0), v.get("dy", 0.0), v.get("dz", 0.0)])
            pR, pp = disp.get(t, (np.eye(3), np.zeros(3)))
            disp[t] = (dR @ pR, dp + pp)
    for t, (dR, dp) in sorted(disp.items()):
        # rotate about the hand's own position, then translate
        p_new = P0[t] + dp
        R_new = dR @ R0[t]
        data.qpos[:] = Q[t]
        if t - 1 in disp:
            data.qpos[ee.qadr] = Q[t - 1, ee.qadr]
        e = ee.solve(data, p_new, R_new, rest=Q0[t, ee.qadr] if rest_pull else None)
        Q[t, ee.qadr] = data.qpos[ee.qadr]
        if fol:
            tp = P0[t] + dp + dR @ (tipP[t] - P0[t])
            if t - 1 in disp:
                data.qpos[fol["chain"].qadr] = Q[t - 1, fol["chain"].qadr]
            e2 = fol["chain"].solve(data, tp, None if fol["pos_only"] else dR @ tipR[t], point=fol["point"],
                                    rest=Q0[t, fol["chain"].qadr])
            Q[t, fol["chain"].qadr] = data.qpos[fol["chain"].qadr]
            e = (max(e[0], e2[0]), max(e[1], e2[1]))
        worst = (max(worst[0], e[0]), max(worst[1], e[1]))
        for body, fr in (carried or {}).items():
            if t in set(fr):
                adr = TJ.free_joint_qadr(model, body)
                p = Q0[t, adr:adr + 3]; Rb = TJ.quat2mat(Q0[t, adr + 3:adr + 7])
                Q[t, adr:adr + 3] = P0[t] + dp + dR @ (p - P0[t])
                q2 = TJ.mat2quat(dR @ Rb)
                Q[t, adr + 3:adr + 7] = q2 if np.dot(q2, Q0[t, adr + 3:adr + 7]) >= 0 else -q2
    return Q, worst


def optimize_frame_locked(scene, trajectory, end_effector, edits, evaluate, arm_tip=None, carried=None,
                          iterations=6, population=12, elite=3, seed=0, weights=None, log=None):
    model = TJ.load_model(scene)
    Q0, labels, fps = TJ.as_trajectory(trajectory)
    bounds = {}
    for k, ed in enumerate(edits):
        for name, (lo, hi) in ed["params"].items():
            bounds[f"{k}:{name}"] = (lo, hi)

    def unpack(params):
        vals = [dict() for _ in edits]
        for key, v in params.items():
            k, name = key.split(":", 1)
            vals[int(k)][name] = v
        return vals

    cache = {}

    def ev(params):
        Q, worst = apply_offsets(model, Q0, labels, end_effector, arm_tip, edits, unpack(params), carried)
        out = dict(evaluate(Q))
        out.setdefault("gates", {})
        out["gates"] = {**out["gates"], "ik_reach": worst[0] <= 0.003 and worst[1] <= np.radians(2.0)}
        out["ik_worst"] = {"pos_m": round(float(worst[0]), 5), "rot_deg": round(float(np.degrees(worst[1])), 3)}
        cache[tuple(sorted(params.items()))] = Q
        return out

    best = optimize_reward_cem(ev, bounds, weights=weights, iterations=iterations, population=population,
                               elite=elite, seed=seed, log=log)
    best["qpos"] = cache[tuple(sorted(best["params"].items()))]
    best["ik_worst"] = best["evaluation"]["ik_worst"]
    best["labels"], best["fps"] = labels, fps
    return best
