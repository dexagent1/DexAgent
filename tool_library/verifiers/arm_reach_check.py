"""Is every hand really bolted to its arm, on every frame?"""
from __future__ import annotations

TOOL = {
    "name": "arm_reach_check",
    "title": "Arm reach and weld check",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "For each arm, the residual between its flange and the hand bolted to it on every frame (position and "
               "rotation of the fixed mount), the arm's joint speed and configuration jumps, and how close its joints run "
               "to their limits. A disconnected or twisted wrist fails.",
    "description": (
        "The hand is bolted to the arm's flange: on every frame the hand root must sit exactly where the arm's joints put "
        "the flange, through the fixed mount calibrated in the trusted setup's home state. The delivered trajectory "
        "cannot supply its own calibration. This verifier measures the mount residual (position mm, rotation deg) on every "
        "frame, the peak arm joint speed and any one-frame configuration jump, and the smallest margin of every arm joint "
        "to its range. Physical recordings additionally check actual velocities at every integrator step, including the "
        "initial velocity; brief between-frame overspeed cannot hide in the rendered frame rate. Missing or malformed "
        "physical velocity evidence fails. Proposal-only diagnostics retain the frame-difference check. It fails when "
        "the residual exceeds `pos_tol_m` / `tilt_tol_deg` on any frame (defaults 2 mm / 2 deg: "
        "a visible gap or twist at the wrist), or a joint moves faster than `max_joint_speed_deg_s`, and reports the worst "
        "frame with its label. execute_subgoal_plan (couple_arms) plans the arm with the hand and passes it by "
        "construction; an episode whose arms were fitted afterwards usually does not. With `poses` it screens candidate "
        "wrist poses for the arm of `side` (exact IK, same tolerances) and, at each solved arm, checks the arm's links "
        "(convex hulls) against its own hand, the other station, the table and the objects at the scene's keyframe (the "
        "executor's arm screen): an arm that holds the hand only by running a link 8 mm into its own palm fails here, not "
        "in the executor's dry run. Poses may be dicts, pose json paths, or a json holding a list (generate_* out_json)."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "Bench scene MJCF."},
        {"name": "trajectory", "type": "str | None", "description": "Episode npz with the arm joints filled (omit when checking poses)."},
        {"name": "poses", "type": "list[dict | str] | str | None", "description": "Candidate wrist poses ({pos, euler, finger_targets}) to test for reach instead of an episode: dicts, pose json paths, or one json file holding a pose or a list of them; returns which the arm of `side` can reach without its links running into its own hand or the scene."},
        {"name": "side", "type": "str | None", "description": "'left' or 'right' for the poses mode."},
        {"name": "arm_collision", "type": "bool", "description": "Poses mode: also check the solved arm's links against its own hand, the other station, the table and the objects (default true; a pose with a hit fails and names the link)."},
        {"name": "state", "type": "str | list | None", "description": "Poses mode: the scene state the poses are checked in (where the objects and the other station are): a trajectory.npz (its last frame) or a qpos; default the scene's keyframe."},
        {"name": "pos_tol_m", "type": "float", "description": "Weld position tolerance. Default 0.002."},
        {"name": "tilt_tol_deg", "type": "float", "description": "Weld rotation tolerance. Default 2."},
        {"name": "max_joint_speed_deg_s", "type": "float", "description": "Arm joint speed that fails (a branch flip shows here). Default 180."},
        {"name": "limit_margin_deg", "type": "float", "description": "Warn when a joint runs closer than this to its stop. Default 2."},
        {"name": "baseline", "type": "str | None", "description": "Deprecated: baseline-relative acceptance is rejected; each episode must meet absolute limits."},
        {"name": "setup", "type": "str | None", "description": "Trusted setup directory providing the calibrated mount."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok, failed, per side {mount, pos_err_mm {max, mean, worst_frame, label}, tilt_err_deg {...}, frames_over, peak_joint_speed_deg_s, speed_frames_over, limit_margin_deg, joints_at_limit}. Poses mode: ok, reachable [indices], issues, poses [{index, source, ok, reach_ok, pos_err_mm, rot_err_deg, arm_qpos, arm_hits [{geoms, depth_mm, what}], own_hand_mm}]."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "audit", "arm", "reach", "ik", "weld"],
    "used_in": ["Open the Bottle", "Giftbox Packing", "Place Toy in Drawer", "Battery Insertion", "Place Cup in Bowl", "Computer Installation"],
    "status": "tested",
    "entrypoint": "arm_reach_check",
}

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from tool_library.common import episode as E


def _weld_series(model, parts, side, Q, p_mount, R_mount, limit_margin_deg):
    pt = parts[side]
    data = mujoco.MjData(model)
    pe, te = np.zeros(len(Q)), np.zeros(len(Q))
    lo = np.array([model.jnt_range[j][0] for j in pt["arm_joints"]]); hi = np.array([model.jnt_range[j][1] for j in pt["arm_joints"]])
    margin = np.inf; at_limit = set()
    for t in range(len(Q)):
        data.qpos[:] = Q[t]; mujoco.mj_kinematics(model, data)
        Ry = data.xmat[pt["ee_body"]].reshape(3, 3); py = data.xpos[pt["ee_body"]]
        pg, Rg = data.xpos[pt["gripper"]], data.xmat[pt["gripper"]].reshape(3, 3)
        pe[t] = np.linalg.norm(pg - (py + Ry @ p_mount))
        te[t] = np.degrees(np.linalg.norm(Rot.from_matrix((Ry @ R_mount).T @ Rg).as_rotvec()))
        q = Q[t][pt["arm_qadr"]]
        m = np.minimum(q - lo, hi - q); margin = min(margin, float(m.min()))
        for j, mm in zip(pt["arm_joints"], m):
            if mm < np.radians(limit_margin_deg):
                at_limit.add(model.joint(j).name)
    return pe, te, margin, at_limit


def arm_reach_check(scene, trajectory=None, pos_tol_m=0.002, tilt_tol_deg=2.0, limit_margin_deg=2.0, poses=None, side=None,
                    max_joint_speed_deg_s=180.0, baseline=None, arm_collision=True, state=None, setup=None):
    model = mujoco.MjModel.from_xml_path(str(scene))
    parts = E.robot_parts(model)
    if poses is not None:                       # reachability of candidate wrist poses before anything is executed
        q_state = None
        if state is not None:                   # objects and the other station where the episode has them
            q_state = E.state_qpos(model, state)
            if len(q_state) != model.nq:
                raise ValueError(f"state has {len(q_state)} qpos values, the scene {model.nq}")
        return _poses_reachable(model, parts, side, _load_poses(poses), pos_tol_m, tilt_tol_deg,
                                scene=str(scene) if arm_collision else None, q_state=q_state)
    ep = E.load_episode(trajectory); Q, labels = ep["qpos"], ep["labels"]
    if Q.ndim != 2 or Q.shape[1] != model.nq or len(Q) < 2 or not np.isfinite(Q).all() or len(labels) != len(Q):
        raise ValueError("trajectory must contain at least two finite labelled frames in the scene joint layout")
    fps = float(ep.get("fps", 30.0) or 30.0)
    if not np.isfinite(max_joint_speed_deg_s) or max_joint_speed_deg_s <= 0:
        raise ValueError("max_joint_speed_deg_s must be positive and finite")
    physical_speeds, velocity_issues = {}, []
    with np.load(str(trajectory), allow_pickle=False) as archive:
        if "robot_schema" in archive.files or any(k.startswith("substep_") for k in archive.files):
            required = {"substep_qvel", "substep_time_s", "initial_qvel"}
            missing = required - set(archive.files)
            if missing:
                velocity_issues.append("physical arm speed evidence missing: " + ", ".join(sorted(missing)))
            else:
                from tool_library.common.arm_timing import runtime_speed_peaks
                try:
                    evidence = {k: archive[k] for k in required}
                    if len(evidence["substep_qvel"]) == 0:
                        raise ValueError("physical arm speed evidence has no integrator samples")
                    physical_speeds = runtime_speed_peaks(model, evidence, fps)
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    velocity_issues.append(f"invalid physical arm speed evidence: {exc}")
    Qb = None
    if baseline is not None:                    # a verified example this episode was adapted from: no worse than it
        raise ValueError("baseline-relative arm acceptance is unsupported; every episode must meet the absolute limits")
    out, failed, issues = {}, list(velocity_issues), list(velocity_issues)
    if set(parts) != {"left", "right"}:
        failed.append("missing robot hands")
    from tool_library.common.rigid_robot import trusted_mounts
    from pathlib import Path
    mounts = trusted_mounts(**({"setup_scene": str(Path(setup) / "scene.xml")} if setup else {}))
    for side, pt in parts.items():
        if pt["gripper"] < 0 or not pt["arm_joints"]:
            out[side] = {"error": "no arm"}; failed.append(side); continue
        p_mount, quat_mount = mounts[side]
        R_mount = Rot.from_quat(np.roll(quat_mount, -1)).as_matrix().T
        p_mount = -R_mount @ p_mount
        mount_src = "trusted setup home"
        pe, te, margin, at_limit = _weld_series(model, parts, side, Q, p_mount, R_mount, limit_margin_deg)
        A = Q[:, pt["arm_qadr"]]
        sp = np.degrees(np.abs(np.diff(A, axis=0)).max(axis=1)) * fps if len(A) > 1 else np.zeros(1)
        speed_lim = float(max_joint_speed_deg_s)
        if Qb is None:
            ep_, et_ = pe, te
        else:
            pe0, te0, _m, _l = _weld_series(model, parts, side, Qb, p_mount, R_mount, limit_margin_deg)
            if len(pe0) == len(pe):
                ep_, et_ = pe - pe0, te - te0
            else:
                ep_, et_ = pe - pe0.max(), te - te0.max()
            A0 = Qb[:, pt["arm_qadr"]]
            sp0 = np.degrees(np.abs(np.diff(A0, axis=0)).max(axis=1)) * fps if len(A0) > 1 else np.zeros(1)
            speed_lim = max(speed_lim, 1.5 * float(sp0.max()) + 30.0)
        kp, kt = int(np.argmax(pe)), int(np.argmax(te))
        over = [int(i) for i in np.where((ep_ > pos_tol_m) | (et_ > tilt_tol_deg))[0]]
        fast = [int(i) + 1 for i in np.where(sp > speed_lim)[0]]
        physical = physical_speeds.get(side)
        physical_fast = physical is not None and physical["peak_speed_deg_s"] > speed_lim
        peak = max(float(sp.max()), physical["peak_speed_deg_s"] if physical else 0.)
        if fast or physical_fast:
            when = f" at physical time {physical['speed_time_s']:.6f} s" if physical_fast else " between rendered frames"
            issues.append(f"{side} arm_speed: peak joint speed {peak:.3f} deg/s{when}, limit {speed_lim:.3f} deg/s")
        out[side] = {"mount": mount_src,
                     "pos_err_mm": {"max": round(1000 * float(pe.max()), 2), "mean": round(1000 * float(pe.mean()), 2), "worst_frame": kp, "label": labels[kp]},
                     "tilt_err_deg": {"max": round(float(te.max()), 2), "mean": round(float(te.mean()), 2), "worst_frame": kt, "label": labels[kt]},
                     "frames_over": over[:20], "n_over": len(over), "labels_over": sorted({str(labels[i]) for i in over})[:6],
                     "peak_joint_speed_deg_s": round(peak, 1), "speed_frames_over": fast[:20],
                     "frame_peak_joint_speed_deg_s": round(float(sp.max()), 1),
                     "physical_speed": physical,
                     "limit_margin_deg": round(float(np.degrees(margin)), 2), "joints_at_limit": sorted(at_limit)}
        if Qb is not None:
            out[side]["relative_to_baseline"] = {"pos_excess_mm_max": round(1000 * float(ep_.max()), 2), "rot_excess_deg_max": round(float(et_.max()), 2),
                                                 "speed_limit_deg_s": round(speed_lim, 1)}
        if over or fast or physical_fast or margin < -1e-8:
            failed.append(side)
            if over:
                issues.append(f"{side}: fixed hand mount exceeds position or rotation tolerance")
            if margin < -1e-8:
                issues.append(f"{side}: arm joint exceeds its physical range")
    res = {"ok": not failed, "failed": failed, "issues": issues, "sides": out, "n_frames": int(len(Q))}
    if Qb is not None:
        res["baseline"] = str(baseline)
    return res


def _load_poses(poses):
    """[(source, pose dict)] from a list of pose dicts / pose json paths, or one json path or dict. A json may hold a pose,
    {"pose": ...}, {"poses": [...]}, {"grasps": [...]} or a list of poses (every one is screened)."""
    import json
    from pathlib import Path

    def unwrap(d, src):
        if isinstance(d, list):
            return [x for i, e in enumerate(d) for x in unwrap(e, f"{src}[{i}]")]
        if isinstance(d, dict):
            for k in ("grasps", "poses"):
                if k in d and "pos" not in d and isinstance(d[k], list):
                    return unwrap(d[k], f"{src}.{k}")
            if "pose" in d and "pos" not in d:
                return unwrap(d["pose"], f"{src}.pose")
            return [(src, d)]
        raise ValueError(f"poses: {src} is a {type(d).__name__}, not a pose dict, a list of them or a pose json path")

    def one(e, src):
        if isinstance(e, (str, Path)):
            path = Path(e)
            if not path.exists():
                raise ValueError(f"poses: {src} names {str(e)!r}, which does not exist")
            return unwrap(json.loads(path.read_text()), str(e))
        return unwrap(e, src)

    if isinstance(poses, (str, Path, dict)):
        out = one(poses, "poses")
    elif isinstance(poses, (list, tuple)):
        out = [x for i, e in enumerate(poses) for x in one(e, f"poses[{i}]")]
    else:
        raise ValueError(f"poses must be a list of pose dicts or json paths, or one json path; got {type(poses).__name__}")
    for src, p in out:
        if not (isinstance(p.get("pos"), (list, tuple)) and len(p["pos"]) == 3 and isinstance(p.get("euler"), (list, tuple)) and len(p["euler"]) == 3):
            raise ValueError(f"poses: {src} has no 3-vector 'pos' and 'euler' (the wrist pose every grasp tool writes)")
    if not out:
        raise ValueError("poses is empty")
    return out


def _poses_reachable(model, parts, side, poses, pos_tol_m, tilt_tol_deg, scene=None, q_state=None):
    """Solve the arm for each candidate wrist pose (pos, euler of the hand's wrist frame) from the home posture, and with
    `scene` check the solved arm's links against its own hand and the scene (armhand.ArmProbe, the executor's screen)."""
    from tool_library.common import hand as H
    from tool_library.common import trajectory as TJ
    if side not in parts:
        raise ValueError(f"side must be one of {list(parts)}")
    poses = [p if isinstance(p, tuple) else (f"poses[{i}]", p) for i, p in enumerate(poses)]
    from tool_library.common import armhand as AH
    probe = rig = None
    if scene is not None:
        from tool_library.common import plan as P
        probe = AH.ArmProbe(scene)
        rig = P.scene_rigs(model).get(side)
    q_home = np.asarray(q_state if q_state is not None else (model.key_qpos[0] if model.nkey else model.qpos0), float)
    where = "the given state" if q_state is not None else "the scene's keyframe (pass state= the episode so far if they have moved)"
    pt = parts[side]
    data = mujoco.MjData(model)
    data.qpos[:] = model.key_qpos[0] if model.nkey else model.qpos0; mujoco.mj_kinematics(model, data)
    p_mount, R_mount, _src = AH.mount(model, side, parts=parts)
    ch = TJ.Chain(model, pt["gripper"], pt["arm_joints"])
    home = data.qpos[pt["arm_qadr"]].copy()
    rows = []
    rng = np.random.default_rng(0)
    issues = []
    for i, (src, pose) in enumerate(poses):
        p = np.asarray(pose["pos"], float); R = H.euler_to_mat(np.asarray(pose["euler"], float))
        pt_, Rt = p + R @ p_mount, R @ R_mount
        best = None
        for k in range(12):
            data.qpos[pt["arm_qadr"]] = home if k == 0 else np.clip(home + rng.normal(0, 0.4, len(home)), ch.lo, ch.hi)
            err = ch.solve(data, pt_, Rt, iters=150, rot_weight=0.5)
            if best is None or err[0] + 0.05 * err[1] < best[0] + 0.05 * best[1]:
                best = (err[0], err[1], data.qpos[pt["arm_qadr"]].copy())
            if best[0] <= pos_tol_m and best[1] <= np.radians(tilt_tol_deg):
                break
        reach_ok = best[0] <= pos_tol_m and best[1] <= np.radians(tilt_tol_deg)
        hits, own = [], 0.0
        if probe is not None and rig is not None:
            dq = mujoco.MjData(model); dq.qpos[:] = q_home
            rig.teleport(dq, p, np.asarray(pose["euler"], float))
            rig.set_fingers(dq, {k: float(v) for k, v in (pose.get("finger_targets") or {}).items() if k in rig.finger_qpos})
            dq.qpos[pt["arm_qadr"]] = best[2]
            for h in probe.check(dq.qpos[None].copy(), frames=[0]):
                nm = " ".join(h["geoms"])
                what = ("own hand" if f"_collision_hand_{side}_" in nm or f"robot_{side}_{side}_" in nm else
                        "other hand" if "_collision_hand_" in nm else "other arm" if nm.count("_arm_") >= 2 else "table or object")
                hits.append({"geoms": h["geoms"], "depth_mm": h["depth_mm"], "what": what})
                if what == "own hand":
                    own = max(own, float(h["depth_mm"]))
        ok = reach_ok and not hits
        if not reach_ok:
            issues.append(f"{src}: the {side} arm cannot hold this wrist pose ({1000 * best[0]:.1f} mm / {np.degrees(best[1]):.1f} deg off); "
                          "re-orient the pose (optimize_pose_for_reach) or choose another grasp")
        for h in hits:
            g0, g1 = h["geoms"]
            arm_g, other_g = (g0, g1) if "_arm_" in g0 else (g1, g0)
            ctx = "" if h["what"] == "own hand" else f"; objects and the other station as in {where}"
            issues.append(f"{src}: at the solved arm, {arm_g} runs {h['depth_mm']} mm into {other_g} ({h['what']}{ctx}); "
                          "re-orient the pose about its contact (optimize_pose_for_reach) or approach from another side")
        rows.append({"index": i, "source": src, "ok": bool(ok), "reach_ok": bool(reach_ok), "pos_err_mm": round(1000 * float(best[0]), 1),
                     "rot_err_deg": round(float(np.degrees(best[1])), 1), "arm_qpos": [round(float(x), 4) for x in best[2]],
                     "arm_hits": hits, "own_hand_mm": round(own, 1), "arm_collision_checked": probe is not None})
    return {"ok": all(r["ok"] for r in rows), "reachable": [r["index"] for r in rows if r["ok"]], "issues": issues, "poses": rows}
