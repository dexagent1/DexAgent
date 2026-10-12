"""Re-fit a hand trajectory onto a real arm: joint commands a controller can run.

    from tool_library.skills.refit_to_real_rig import refit_to_real_rig
    out = refit_to_real_rig("scene.xml", flange_body="robot_right_arm_gripper",
                            positions=P, rotations=R, fps=30,
                            table_top_z=0.0, floor_margin_m=0.01, out_path="right_arm.npz")
    out["qpos"], out["residual"], out["resets"]

A simulated episode gives the flange a pose on every frame. This turns that
path into a joint trajectory the real arm tracks: frame-by-frame IK warm
started inside a trust region (no branch jumps), a multi-start branch reset
when the warm chain is left far off pose (blended over the preceding frames
so the command stays continuous), zero-phase smoothing, a joint-rate clamp
at the controller's ceiling, then a forward-kinematics check that the hand
never dips below the table, raising the target and re-solving where it does.
"""
from __future__ import annotations

TOOL = {
    "name": "refit_to_real_rig",
    "title": "Refit to real rig",
    "type": "skill",
    "category": "Trajectory & Execution",
    "summary": "Flange pose path -> smooth, rate-limited, table-safe joint trajectory for a real arm, with branch-reset handling.",
    "description": (
        "Fits the arm chain above flange_body to the given pose path with a damped least "
        "squares IK tuned for real arms (per-axis rotation weights with the roll axis soft, "
        "trust-region warm starts, position primary). A frame whose warm solution is more "
        "than 30 mm or 20 degrees off triggers a multi-start search; a better branch is "
        "adopted and blended over the previous 12 frames. The joint trajectory is then "
        "smoothed with a zero-phase Savitzky-Golay filter, clamped to rate_max_deg_s "
        "(180 deg/s), and checked by forward kinematics against the table: frames where the "
        "flange (or the given hand point) falls below table_top_z + floor_margin are raised "
        "and re-solved. Optionally the path is first transformed by a rigid offset "
        "(sim-to-real base registration) and clamped above the table. Writes an npz with "
        "the joint trajectory, residuals and timestamps when out_path is given."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "Scene containing the arm (the real arm's kinematics and limits)."},
        {"name": "flange_body", "type": "str", "description": "Body whose pose the path describes."},
        {"name": "positions, rotations", "type": "ndarray [T, 3], [T, 3, 3]", "description": "Target pose path."},
        {"name": "fps", "type": "float", "description": "Frame rate of the path."},
        {"name": "table_top_z, floor_margin_m", "type": "float | None, float", "description": "Table safety floor for the flange or hand point. Default margin 0.01."},
        {"name": "hand_point", "type": "list[3] | None", "description": "Lowest point of the hand in the flange frame to keep above the floor."},
        {"name": "offset", "type": "(R [3,3], t [3]) | None", "description": "Rigid transform applied to the path first."},
        {"name": "rate_max_deg_s", "type": "float", "description": "Controller joint-rate ceiling. Default 180."},
        {"name": "joint_ranges_deg", "type": "dict joint -> (lo, hi) | None", "description": "Real stops tighter than the model's."},
        {"name": "seed_q", "type": "ndarray | None", "description": "Starting posture for the first frame."},
        {"name": "out_path", "type": "str | None", "description": "Write qpos, residuals, time stamps."},
    ],
    "outputs": [
        {"name": "result", "type": "dict", "description": "qpos [T, n], joint_names, residual {pos_mm max/p95, tilt_deg max, roll_deg max}, resets (frames), floor_fixes (frames raised), max_rate_deg_s, out_path."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["real robot", "deployment", "ik", "smoothing", "rate limit"],
    "used_in": ["Multi-Cup Grasping", "Giftbox Packing", "Battery Insertion", "Drawing and Wiping", "Open the Bottle", "Scissor Cutting"],
    "status": "tested",
    "entrypoint": "refit_to_real_rig",
}

import numpy as np

from tool_library.common import trajectory as TJ


def refit_to_real_rig(scene, flange_body, positions, rotations, fps=30.0, table_top_z=None, floor_margin_m=0.01,
                      hand_point=None, offset=None, rate_max_deg_s=180.0, joint_ranges_deg=None, seed_q=None,
                      smooth_window=9, out_path=None, max_floor_passes=3):
    import mujoco
    model = TJ.load_model(scene)
    ch = TJ.Chain(model, TJ.body_id(model, flange_body))
    if joint_ranges_deg:
        lo, hi = ch.lo.copy(), ch.hi.copy()
        for k, j in enumerate(ch.joints):
            r = joint_ranges_deg.get(model.joint(j).name)
            if r:
                lo[k], hi[k] = np.radians(r[0]), np.radians(r[1])
        ch.set_limits(lo, hi)
    P = np.asarray(positions, float).copy(); R = np.asarray(rotations, float).copy()
    if offset is not None:
        P = P @ np.asarray(offset[0]).T + np.asarray(offset[1]); R = np.einsum("ij,tjk->tik", np.asarray(offset[0]), R)
    floor = None if table_top_z is None else table_top_z + floor_margin_m
    hp = None if hand_point is None else np.asarray(hand_point, float)
    if floor is not None:                        # clamp the targets themselves first
        low = P[:, 2] + (0.0 if hp is None else np.einsum("tij,j->ti", R, hp)[:, 2])
        P[:, 2] += np.maximum(0.0, floor - low)
    Q, rp, rt, rr, resets = TJ.fit_arm_to_poses(model, ch, P, R, seed=seed_q)
    dq_max = np.radians(rate_max_deg_s) / fps
    Qs = TJ.rate_limit(TJ.savgol(Q, smooth_window, 3), dq_max)
    Qs = np.where(ch.limited, np.clip(Qs, ch.lo, ch.hi), Qs)
    floor_fixes = []
    if floor is not None:
        data = mujoco.MjData(model)
        for _ in range(max_floor_passes):
            bad = []
            for t in range(len(Qs)):
                data.qpos[ch.qadr] = Qs[t]
                mujoco.mj_kinematics(model, data)
                p, Rf = ch.pose(data, hp)
                if p[2] < floor - 1e-4:
                    bad.append((t, floor - p[2]))
            if not bad:
                break
            for t, dz in bad:
                P[t, 2] += dz + 0.002
                data.qpos[ch.qadr] = Qs[t]
                ch.solve(data, P[t], R[t], rest=Qs[t], iters=60, rot_weight=0.35)
                Qs[t] = data.qpos[ch.qadr]
            floor_fixes += [t for t, _ in bad]
            Qs = TJ.rate_limit(Qs, dq_max)
    rate = np.degrees(np.abs(np.diff(Qs, axis=0)).max()) * fps if len(Qs) > 1 else 0.0
    # residuals of the final, smoothed command
    data = mujoco.MjData(model)
    pos_mm, tilt, roll = np.zeros(len(Qs)), np.zeros(len(Qs)), np.zeros(len(Qs))
    for t in range(len(Qs)):
        pos_mm[t], tilt[t], roll[t] = TJ._arm_residual(model, data, ch, Qs[t], P[t], R[t])
    res = {"qpos": Qs, "joint_names": [model.joint(j).name for j in ch.joints], "qadr": ch.qadr.tolist(),
           "residual": {"pos_mm_max": round(float(pos_mm.max()), 2), "pos_mm_p95": round(float(np.percentile(pos_mm, 95)), 2),
                        "tilt_deg_max": round(float(tilt.max()), 2), "roll_deg_max": round(float(roll.max()), 2)},
           "resets": resets, "floor_fixes": sorted(set(floor_fixes)), "max_rate_deg_s": round(float(rate), 1),
           "ok": bool(rate <= rate_max_deg_s + 1e-6 and pos_mm.max() < 30.0)}
    if out_path:
        np.savez_compressed(out_path, qpos=Qs, joint_names=np.array(res["joint_names"]), fps=fps,
                            t=np.arange(len(Qs)) / fps, pos_mm=pos_mm, tilt_deg=tilt, roll_deg=roll)
        res["out_path"] = str(out_path)
    return res
