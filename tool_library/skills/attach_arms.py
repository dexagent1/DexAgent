"""Place an arm's base so it can follow a hand's whole path, then bolt the hand to it.

    from tool_library.skills.attach_arms import attach_arms
    res = attach_arms("scene.xml", "trajectory.npz",
                      arm_root="robot_right_arm_root", arm_tip="robot_right_arm_gripper",
                      hand_body="robot_right_right_base_yaw",
                      candidates={"radius_frac": [0.45, 0.6, 0.75], "n_yaw": 16},
                      out_scene="scene_mounted.xml")
    res["base_pos"], res["base_yaw_deg"], res["residual"], res["qpos"]

A hand trajectory retargeted from video floats in the world. The arm that
carries it has to stand somewhere it can reach every frame without passing
through the table or folding onto a stop. This searches the base position
and yaw on the floor (or at a given height), scoring the full IK fit of the
arm to the hand's flange path over subsampled frames, prefers mounts with
joint-limit reserve and an elbow-up branch, fits all frames from the winner,
and writes the scene with the base moved and a weld between arm tip and hand.
"""
from __future__ import annotations

TOOL = {
    "name": "attach_arms",
    "title": "Attach arms",
    "type": "skill",
    "category": "Trajectory & Execution",
    "summary": "Search the base pose (x, y, yaw) of an arm so its tip tracks a hand's recorded path, fit all frames, and write the mounted scene with the tip welded to the hand.",
    "description": (
        "Reads the hand body's world pose track from the trajectory (or takes an explicit "
        "pose path), estimates the arm's reach from its kinematics, and scores candidate "
        "bases on rings around the path's centroid (radius fractions of the reach, n_yaw "
        "headings each facing the centroid plus facing_offset_deg for assets whose zero yaw "
        "does not point along the first link). Each candidate is scored by the arm's IK "
        "residual on subsampled frames (position in mm plus 3 x tilt degrees), a penalty for "
        "joints within 5 degrees of a stop, and a penalty for the tip falling below the table. "
        "The best few are refined with Nelder-Mead, the winner is fitted on every frame "
        "(trust-region warm starts, branch resets blended), and the scene is rewritten with "
        "the root body moved and a weld (torquescale 0 when position_only) between the tip "
        "and the hand. An optional hand offset gives the flange pose in the hand frame."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF scene containing the arm and the hand."},
        {"name": "trajectory", "type": "str | ndarray | None", "description": "qpos trajectory the hand path is read from."},
        {"name": "arm_root, arm_tip, hand_body", "type": "str", "description": "World-fixed root body of the arm, its flange body, the hand body it should carry."},
        {"name": "hand_offset", "type": "(R [3,3], t [3]) | None", "description": "Flange pose in the hand frame; default identity."},
        {"name": "poses", "type": "(P [T,3], R [T,3,3]) | None", "description": "Explicit flange path instead of the trajectory."},
        {"name": "base_z, candidates, facing_offset_deg, position_only", "type": "float | None, dict, float, bool", "description": "Mount height (default the root's), ring search, yaw convention, weld type."},
        {"name": "table_top_z", "type": "float | None", "description": "Penalize tips below this."},
        {"name": "out_scene", "type": "str | None", "description": "Write the mounted scene here."},
    ],
    "outputs": [
        {"name": "result", "type": "dict", "description": "base_pos, base_yaw_deg, residual {pos_mm max/p95, tilt_deg max}, resets, qpos [T, n] arm joints, joint_names, candidates tried, out_scene."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["mount", "arm", "ik", "reachability", "rig"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope", "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting", "Biology Experiment"],
    "status": "tested",
    "entrypoint": "attach_arms",
}

import numpy as np

from tool_library.common import trajectory as TJ


def _reach(model, root_bid, tip_bid):
    import mujoco
    data = mujoco.MjData(model)
    ch = TJ.Chain(model, tip_bid)
    best = 0.0
    for q in [np.zeros(len(ch.qadr)), 0.5 * (ch.lo + ch.hi)] + [np.clip(np.zeros(len(ch.qadr)) + d, ch.lo, ch.hi) for d in (-0.3, 0.3)]:
        data.qpos[ch.qadr] = np.where(ch.limited, np.clip(q, ch.lo, ch.hi), q)
        mujoco.mj_kinematics(model, data)
        best = max(best, float(np.linalg.norm(data.xpos[tip_bid] - data.xpos[root_bid])))
    # stretched: sum of link offsets along the chain is the true maximum
    total, b = 0.0, tip_bid
    while b != root_bid and b != 0:
        total += float(np.linalg.norm(model.body_pos[b])); b = model.body_parentid[b]
    return max(best, total)


def _score(model, ch, root_bid, base_pos, yaw, P, R, table_top_z, cfg, limit_margin=np.radians(5)):
    import mujoco
    model.body_pos[root_bid] = base_pos
    model.body_quat[root_bid] = TJ.mat2quat(TJ.rot_z(yaw))
    data = mujoco.MjData(model)
    rng = np.random.default_rng(0)
    cost, q = 0.0, np.zeros(len(ch.qadr))
    w_rot = 0.0 if cfg["rot_w"] == 0.0 else 1.0
    for t in range(len(P)):
        best = TJ._arm_multistart(model, data, ch, P[t], R[t], [q], rng, cfg, n_random=3 if t == 0 else 1, iters=80)
        q = best[1]
        pe, ti, ro = TJ._arm_residual(model, data, ch, q, P[t], R[t])
        cost += pe + w_rot * (3.0 * ti + 0.3 * ro)
        near = ((q - ch.lo) < limit_margin) | ((ch.hi - q) < limit_margin)
        cost += 20.0 * near.sum()
        if table_top_z is not None:
            mujoco.mj_kinematics(model, data)
            z_min = min(float(data.xpos[b][2]) for b in TJ.subtree(model, root_bid) if b != root_bid)
            cost += 1000.0 * max(0.0, table_top_z - z_min)
    return cost / len(P)


def attach_arms(scene, trajectory=None, arm_root=None, arm_tip=None, hand_body=None, hand_offset=None, poses=None,
                base_z=None, candidates=None, facing_offset_deg=0.0, position_only=True, table_top_z=None,
                out_scene=None, n_score_frames=24, refine_top=3):
    import mujoco
    from scipy.optimize import minimize
    spec = mujoco.MjSpec.from_file(str(scene))
    model = spec.compile()
    root_bid, tip_bid = TJ.body_id(model, arm_root), TJ.body_id(model, arm_tip)
    if model.body_parentid[root_bid] != 0:
        raise ValueError(f"'{arm_root}' must be a direct child of the world so its pose can be set")
    ch = TJ.Chain(model, tip_bid)
    if poses is None:
        Q, labels, fps = TJ.as_trajectory(trajectory)
        P, R = TJ.pose_track(model, Q, TJ.body_id(model, hand_body))
        if hand_offset is not None:
            Ro, to = np.asarray(hand_offset[0], float), np.asarray(hand_offset[1], float)
            P = P + np.einsum("tij,j->ti", R, to); R = np.einsum("tij,jk->tik", R, Ro)
    else:
        P, R = np.asarray(poses[0], float), np.asarray(poses[1], float)
    T = len(P)
    sub = np.unique(np.linspace(0, T - 1, min(n_score_frames, T)).astype(int))
    Ps, Rs = P[sub], R[sub]
    reach = _reach(model, root_bid, tip_bid)
    z0 = float(model.body_pos[root_bid][2]) if base_z is None else float(base_z)
    cen = P.mean(axis=0)
    cand = candidates or {}
    fr = cand.get("radius_frac", [0.45, 0.6, 0.75]); n_yaw = int(cand.get("n_yaw", 16))
    off = np.radians(facing_offset_deg)
    # a position-only weld leaves the tip's orientation free: fit position alone
    cfg = {**TJ.ARM_FIT, **({"rot_w": 0.0, "reset_tilt_deg": 1e9} if position_only else {})}
    tried = []
    for f in fr:
        for k in range(n_yaw):
            ang = 2 * np.pi * k / n_yaw
            pos = np.array([cen[0] + f * reach * np.cos(ang), cen[1] + f * reach * np.sin(ang), z0])
            yaw = float(np.arctan2(cen[1] - pos[1], cen[0] - pos[0]) + off)
            c = _score(model, ch, root_bid, pos, yaw, Ps, Rs, table_top_z, cfg)
            tried.append((c, pos, yaw))
    tried.sort(key=lambda x: x[0])
    best = tried[0]
    for c0, pos, yaw in tried[:refine_top]:
        x0 = np.array([pos[0], pos[1], yaw])
        r = minimize(lambda x: _score(model, ch, root_bid, np.array([x[0], x[1], z0]), x[2], Ps, Rs, table_top_z, cfg),
                     x0, method="Nelder-Mead", options=dict(maxiter=120, xatol=1e-3, fatol=1e-2))
        if r.fun < best[0]:
            best = (float(r.fun), np.array([r.x[0], r.x[1], z0]), float(r.x[2]))
    cost, pos, yaw = best
    model.body_pos[root_bid] = pos
    model.body_quat[root_bid] = TJ.mat2quat(TJ.rot_z(yaw))
    Qa, rp, rt, rr, resets = TJ.fit_arm_to_poses(model, ch, P, R, cfg=cfg)
    res = {"base_pos": [round(float(v), 4) for v in pos], "base_yaw_deg": round(float(np.degrees(yaw)), 2),
           "score": round(cost, 3), "reach_m": round(reach, 4),
           "residual": {"pos_mm_max": round(float(rp.max()), 2), "pos_mm_p95": round(float(np.percentile(rp, 95)), 2),
                        "tilt_deg_max": round(float(rt.max()), 2), "roll_deg_max": round(float(rr.max()), 2)},
           "resets": resets, "qpos": Qa, "qadr": ch.qadr.tolist(), "joint_names": [model.joint(j).name for j in ch.joints],
           "position_only": bool(position_only), "candidates_tried": len(tried),
           "ok": bool(rp.max() < 30.0 and (position_only or rt.max() < 20.0))}
    if out_scene:
        body = spec.body(arm_root)
        body.pos, body.quat = pos, TJ.mat2quat(TJ.rot_z(yaw))
        if hand_body:
            eq = spec.add_equality()
            eq.type = mujoco.mjtEq.mjEQ_WELD
            eq.name1, eq.name2 = arm_tip, hand_body
            eq.objtype = mujoco.mjtObj.mjOBJ_BODY
            d = np.zeros(11); d[6] = 1.0; d[10] = 0.0 if position_only else 1.0
            eq.data = d
        spec.compile()
        with open(out_scene, "w") as f:
            f.write(spec.to_xml())
        res["out_scene"] = str(out_scene)
    return res
