"""The grasp of a second object by the free fingers of a hand that already holds one (finger allocation)."""
from __future__ import annotations

TOOL = {
    "name": "second_grasp_pose",
    "title": "Second grasp pose (finger allocation)",
    "type": "skill",
    "category": "Grasp & Poses",
    "summary": "Pose for taking a SECOND object with the free fingers of a hand that already holds one (the first cup between "
               "thumb and index, the second on middle, ring and pinky dipped inside it and spread), searched with the first "
               "object riding the hand, so the two grasps are designed together; the input of a `grasp_more` plan entry.",
    "description": (
        "The two grasps of one hand cannot be chosen apart: the first object rides the hand to the second, so where "
        "the first grasp put it decides whether the free fingers can reach the second at all (a cup pinched between "
        "thumb and index hangs exactly where the other three fingers would dip). Given the first grasp (`first_pose`, "
        "one pose or a list of candidates: then the PAIRS are ranked; or its pad points, and the first grasp is designed in the "
        "second's wrist family) and the second object, the free fingers are "
        "the fingers the first pose's grip_fingers leave (or `free_fingers`). Family 'inside' (an open container: a "
        "cup, a bowl, a box): the free fingers go in together through the mouth, straight, and spread by their "
        "abduction joints against the inner wall (abduction is the grip: curling three fingers inside a cup "
        "retracts the tips and presses one wall, tipping it). The search turns the hand so the free fingers point "
        "into the mouth (straight down, and tilted by `tilts_deg` toward 8 directions) at every `yaw_step_deg` about "
        "the free fingers' axis, and puts their tip centroid `depth_m` below the rim on the mouth's axis. Every "
        "candidate is judged with the first object carried rigidly from its grasp (lifted lift1_m) to the new wrist "
        "pose: the first object must clear the second, the table and every bystander by clear_m and tilt at most "
        "max_first_tilt_deg; the hand (insertion shape: the spreading fingers turned a third of the way inward) must clear the second "
        "object, the table and the bystanders by clear_m, also along the approach (approach_m back along the free fingers' axis, the first object riding "
        "along); sweeping the abduction from the insertion shape to the spread, at least two free fingers must meet "
        "the inner wall, from opposite sides (the spread target lies past the wall, so the servos press it). With bench_scene the surviving poses are "
        "screened for the station's arm down the clearance ranking, in batches, until top_k pass. Ranked by the smallest clearance (larger first), then the first object's "
        "tilt. The pose carries finger_targets (the free fingers spread, the held fingers as the first grasp clamps "
        "them), finger_targets_open (the free fingers together: the insertion shape), grip_fingers = the free fingers, "
        "press_depth_rad 0 (the spread presses; no flexion clamp), the approach, the first grasp it was designed "
        "with (`first_pose_index`) and its numbers. By default, a bounded multi_object_hold_test search verifies the "
        "ranked pairs and modest first-grip clamp variations, preserving actuator/contact limits. Every returned "
        "candidate includes its exact first_pose. Full rigid-arm episode verification is still required."
    ),
    "inputs": [
        {"name": "scene", "type": "dict | str", "description": "Hand scene (hand_scene_from_recon) holding both objects, as they lie before the first grasp."},
        {"name": "first_pose", "type": "dict | str | list", "description": "The first grasp (pose dict or JSON path), or a list of candidates."},
        {"name": "first_object", "type": "str", "description": "Body of the first object (the one the first pose grasps)."},
        {"name": "first_points, first_normals", "type": "dict | None", "description": "With first_pose None: the first grasp's pad points and outward normals ({thumb: [x, y, z], index: ...}); the first grasp is then DESIGNED with the second: pose_from_points with the wrist held in the family the second grasp uses (the free fingers, straight, pointing down or tilted 25 deg), every first_yaw_step_deg (30) about the vertical, kept only when the executor's clamp presses every pad into its surface."},
        {"name": "second_object", "type": "str", "description": "Body of the second object."},
        {"name": "free_fingers", "type": "list | None", "description": "Fingers for the second grasp (default: those the first pose's grip_fingers leave)."},
        {"name": "family", "type": "str", "description": "'inside' (default; the only family so far)."},
        {"name": "depth_m", "type": "float", "description": "How far below the rim the free fingertips go. Default 0.025."},
        {"name": "lift1_m", "type": "float", "description": "The first object's lift before the transfer (it rides at this height). Default 0.04."},
        {"name": "yaw_step_deg", "type": "float", "description": "Turn about the free fingers' axis between candidates. Default 15."},
        {"name": "tilts_deg", "type": "list", "description": "Tilts of the free fingers' axis from straight down. Default [0, 15, 30]."},
        {"name": "max_first_tilt_deg", "type": "float", "description": "How far the first object may tilt from its grasp orientation. Default 20."},
        {"name": "clear_m", "type": "float", "description": "Clearance of the hand and the first object from the second and the scene. Default 0.004."},
        {"name": "approach_m", "type": "float", "description": "Approach distance (checked). Default 0.05."},
        {"name": "bench_scene, side, state", "type": "str | list | None", "description": "Screen each pose for the station's arm (state: the bench state, a trajectory.npz or a qpos)."},
        {"name": "top_k", "type": "int", "description": "Poses returned. Default 5."},
        {"name": "verify_physics", "type": "bool", "description": "Default True: only return pairs passing the default multi-object hold stress test. False returns geometric/reach candidates requiring separate stress validation."},
        {"name": "max_physics_trials", "type": "int", "description": "Maximum combined hold trials across poses and clamp variations. Default 12."},
        {"name": "out_json", "type": "str | None", "description": "Write the best pose here (the ranked list to <stem>_all.json)."},
    ],
    "outputs": [
        {"name": "result", "type": "dict", "description": "{ok, pose (with `first_pose`: the first grasp it was designed with), poses [ranked, each with `second_grasp` {first_pose_index, yaw_deg, tilt_deg, clear_mm, first_tilt_deg, wall_fingers, reach}], first_poses, n_candidates, rejected {reason: count}, free_fingers, mouth}."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "grasp", "finger allocation", "two objects", "cup"],
    "used_in": [],
    "status": "tested",
    "entrypoint": "second_grasp_pose",
}

import json
import copy
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def _unit(v):
    v = np.asarray(v, float)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _align(a, b):
    """Rotation taking unit vector a onto unit vector b."""
    a, b = _unit(a), _unit(b)
    v = np.cross(a, b); c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        ax = _unit(np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0]))
        return Rot.from_rotvec(np.pi * ax).as_matrix()
    return Rot.from_rotvec(_unit(v) * np.arctan2(np.linalg.norm(v), c)).as_matrix()


def _min_dist(model, data, ga, gb, cap=0.05):
    from tool_library.common import hand as H
    best = cap
    for a in ga:
        for b in gb:
            if np.linalg.norm(data.geom_xpos[a] - data.geom_xpos[b]) > model.geom_rbound[a] + model.geom_rbound[b] + cap:
                continue
            best = min(best, H.geom_distance(model, data, a, b, cap))
    return best


def _mouth(model, data, gids, depth):
    """The container's mouth: axis point (xy centre of its geometry), rim height, and the inner radius `depth` below the
    rim (horizontal rays from the axis to the wall, the smallest of 12)."""
    from tool_library.common import hand as H
    ext = H.object_extent(model, data, gids)
    c = np.asarray(ext["centre"], float); rim = float(ext["hi"][2])
    z = rim - float(depth)
    rmin = np.inf
    for a in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        vec = np.array([np.cos(a), np.sin(a), 0.0])
        pnt = np.array([c[0], c[1], z])
        for g in gids:
            t = int(model.geom_type[g])
            dist = float(mujoco.mj_rayMesh(model, data, g, pnt, vec)) if t == int(mujoco.mjtGeom.mjGEOM_MESH) else \
                float(mujoco.mju_rayGeom(data.geom_xpos[g], data.geom_xmat[g], model.geom_size[g], pnt, vec, t))
            if dist >= 0:
                rmin = min(rmin, dist)
    return np.array([c[0], c[1], rim]), (None if not np.isfinite(rmin) else float(rmin))


def _straight(rig, fingers, base):
    """`base` finger targets with `fingers` straight and together (flexion and abduction at zero, within range)."""
    from tool_library.common import hand as H
    out = dict(base)
    for k, (lo, hi) in rig.finger_range.items():
        if H.finger_of_joint(k) in fingers and H.joint_role(k) in ("abduct", "flex", "pinky_arch"):
            out[k] = float(np.clip(0.0, lo, hi))
    return out


def design_first_grasps(scene, model, data0, rig, points, normals, free, yaw_step_deg=30.0, tilts_deg=(0.0, 25.0), max_residual_mm=3.0,
                        max_overlap_mm=5.0, min_press_mm=1.5):
    """First-grasp candidates in the wrist family the second grasp uses: the hand turned so the free fingers, straight,
    point down (straight down and tilted by tilts_deg toward four directions: the way they enter an open container), at
    every yaw_step_deg about the vertical; at each, pose_from_points puts the first grasp's pads on `points` with the wrist
    orientation held (fix_orientation), so the first object rides the hand to wherever the second grasp turns it. A pose
    is kept only when the executor's clamp drives EVERY pad into its surface by at least min_press_mm (hand.pad_shift_world
    along minus the normal): in most fingers-down orientations the index's flexion slides its pad 14 mm along a cup wall
    and 2-4 mm away from it, and the cup is never gripped."""
    from tool_library.common import hand as H
    from tool_library.skills.pose_from_points import pose_from_points
    d = mujoco.MjData(model); d.qpos[:] = data0.qpos
    ins = _straight(rig, free, rig.finger_targets(0.5))
    rig.teleport(d, np.zeros(3), np.zeros(3)); rig.set_fingers(d, ins); mujoco.mj_kinematics(model, d)
    f_ref = _unit(np.mean([d.site_xpos[rig.fingertip_sites[f]] - rig.knuckle_point(d, f) for f in free], axis=0))
    c_loc = np.mean([d.site_xpos[rig.fingertip_sites[f]] for f in points], axis=0)
    cen = np.mean([np.asarray(v, float) for v in points.values()], axis=0)
    R_al = _align(f_ref, [0.0, 0.0, -1.0])
    out = []
    for tilt in tilts_deg:
        dirs = [None] if float(tilt) == 0.0 else [np.array([np.cos(a), np.sin(a), 0.0]) for a in np.radians([0.0, 90.0, 180.0, 270.0])]
        for td in dirs:
            Rt = np.eye(3) if td is None else Rot.from_rotvec(np.radians(float(tilt)) * _unit(np.cross([0.0, 0.0, 1.0], td))).as_matrix()
            for yaw in np.arange(0.0, 360.0, float(yaw_step_deg)):
                R = Rt @ Rot.from_rotvec(np.radians(yaw) * np.array([0.0, 0.0, 1.0])).as_matrix() @ R_al
                seed = (cen - R @ c_loc + np.array([0.0, 0.0, 0.01]), H.mat_to_euler(R))
                try:
                    ps = pose_from_points(scene, points=points, normals=normals, seeds=[(seed[0].tolist(), seed[1].tolist())], fix_orientation=True)
                except Exception:
                    continue
                ps = ps.get("pose", ps)
                if float(ps.get("worst_residual_mm", 99)) > max_residual_mm or float(ps.get("overlap_mm", 99)) > max_overlap_mm:
                    continue
                c = H.executor_clamp(rig, ps)
                sh = H.pad_shift_world(rig, c["closed"], c["press"], ps["euler"])
                into = {f: float(sh[f] @ -_unit(normals[f])) for f in points if f in sh}
                if min(into.values()) < float(min_press_mm) / 1000.0:
                    continue
                ps = dict(ps); ps.update({"family_yaw_deg": float(yaw), "family_tilt_deg": float(tilt), "press_into_mm": {f: round(1000 * v, 1) for f, v in into.items()}})
                out.append(ps)
    return out


def _verify_pairs(scene, poses, objects, limit=12, top_k=5):
    """Breadth-first physical search: all original pairs before clamp variants."""
    from tool_library.verifiers.multi_object_hold_test import multi_object_hold_test
    from tool_library.common import hand as H
    if int(limit) != limit or int(limit) < 1:
        raise ValueError("max_physics_trials must be a positive integer")
    accepted, trials, seen = [], [], set()
    for scale in (1., 1.25, 1.5, 2.):
        for index, source in enumerate(poses):
            if len(trials) >= int(limit) or len(accepted) >= max(1, int(top_k)):
                return accepted, trials
            if any(p["physics_search"]["candidate_index"] == index for p in accepted):
                continue
            pose = copy.deepcopy(source)
            first = pose["first_pose"]
            depth = float(first.get("press_depth_rad", H.PRESS_DEPTH_RAD))
            # Zero deliberately disables flexion (e.g. an abduction grip).
            if scale != 1.:
                if depth == 0. or abs(depth) > .4:
                    continue
                first["press_depth_rad"] = float(np.sign(depth) * min(.4, abs(depth) * scale))
            key = (index, float(first.get("press_depth_rad", H.PRESS_DEPTH_RAD)))
            if key in seen:
                continue
            seen.add(key)
            report = multi_object_hold_test(scene, [first, pose], objects)
            row = {"candidate_index": index, "first_press_depth_rad": key[1],
                   "ok": bool(report["ok"]), "first_fail": report.get("first_fail"), "report": report}
            trials.append(row)
            if report["ok"]:
                pose["physics_search"] = {k: row[k] for k in ("candidate_index", "first_press_depth_rad", "ok")}
                accepted.append(pose)
    return accepted, trials


def second_grasp_pose(scene, first_pose, first_object, second_object, free_fingers=None, family="inside", depth_m=0.025, lift1_m=0.04,
                      yaw_step_deg=15.0, tilts_deg=(0.0, 15.0, 30.0), max_first_tilt_deg=20.0, clear_m=0.004, approach_m=0.05,
                      first_points=None, first_normals=None, first_yaw_step_deg=30.0,
                      bench_scene=None, side=None, state=None, top_k=5, out_json=None, verify_physics=True, max_physics_trials=12):
    from tool_library.common import hand as H
    from tool_library.common.subgoal import load_pose
    if family != "inside":
        raise ValueError(f"family {family!r} is not supported (one of: 'inside')")
    if isinstance(scene, (str, Path)):
        scene = H.load_scene_spec(scene)
    model, data0, rig, ids = H.open_scene(scene)
    side = side or rig.side
    if first_pose is None:
        if not first_points or not first_normals:
            raise ValueError("give first_pose, or first_points + first_normals for the first grasp to be designed with the second")
        free0 = list(free_fingers) if free_fingers else [f for f in H.FINGERS if f not in first_points]
        firsts = design_first_grasps(scene, model, data0, rig, first_points, first_normals, free0, yaw_step_deg=first_yaw_step_deg)
        firsts.sort(key=lambda f: -min(f["press_into_mm"].values()))
        if not firsts:
            return {"ok": False, "pose": None, "poses": [], "first_poses": [], "n_candidates": 0, "rejected": {"first_grasp": 1}, "free_fingers": free0,
                    "error": "no first grasp with the free fingers pointing down puts the pads on first_points and presses them into their surfaces"}
    else:
        firsts = first_pose if isinstance(first_pose, list) else [first_pose]
        firsts = [dict(load_pose(p)) for p in firsts]
    bm, qb = None, None
    if bench_scene is not None:
        from tool_library.common import armhand as AH
        from tool_library.skills.hand_scene_from_recon import resolve_bench_state
        bm = mujoco.MjModel.from_xml_path(str(bench_scene))
        qb = resolve_bench_state(scene, bm, state)
        if first_pose is None:
            # a designed first grasp must be one the arm holds too (with its lift): the pair is only as good as both
            scr = AH.screen_poses(bm, side, firsts, qb, probe=AH.ArmProbe(str(bench_scene)), approach_m=approach_m, waypoints=[[0.0, 0.0, float(lift1_m)]])
            n0 = len(firsts)
            firsts = [f for f, r in zip(firsts, scr) if r.get("ok")]
            if not firsts:
                return {"ok": False, "pose": None, "poses": [], "first_poses": [], "n_candidates": 0, "rejected": {"first_grasp_arm": n0},
                        "free_fingers": None, "error": "the station's arm holds none of the first grasps designed (move the objects into its reach)"}
    b1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, first_object)
    b2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, second_object)
    if b1 < 0 or b2 < 0:
        raise ValueError(f"no body {first_object!r} / {second_object!r} in the hand scene")
    j1 = next((j for j in range(model.njnt) if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE) and int(model.jnt_bodyid[j]) == int(model.body_rootid[b1])), None)
    if j1 is None:
        raise ValueError(f"{first_object!r} is not a free object (it must ride the hand)")
    r1 = int(model.jnt_bodyid[j1])

    def sub(b):
        out, st = {int(b)}, [int(b)]
        while st:
            x = st.pop()
            for c in range(model.nbody):
                if int(model.body_parentid[c]) == x and c not in out:
                    out.add(c); st.append(c)
        return out
    g1 = sorted(g for g in ids["obj_gids"] if int(model.geom_bodyid[g]) in sub(r1))
    g2 = sorted(g for g in ids["obj_gids"] if int(model.geom_bodyid[g]) in sub(int(model.body_rootid[b2])))
    others = sorted(g for g in ids["obj_gids"] if g not in g1 and g not in g2)
    floor = [ids["floor_gid"]] if ids.get("floor_gid", -1) >= 0 else []
    mouth, r_in = _mouth(model, data0, g2, depth_m)
    if r_in is None:
        raise ValueError(f"{second_object!r} has no cavity at {depth_m} m below its rim (family 'inside' needs an open container)")
    target = mouth - np.array([0.0, 0.0, float(depth_m)])
    d = mujoco.MjData(model)
    rejected, judged = {}, []
    hand_g = sorted(rig.collision_gids)
    for fi, fp in enumerate(firsts):
        F1 = set(fp.get("grip_fingers") or [])
        if not F1:
            raise ValueError(f"first pose {fi} names no grip_fingers")
        free = list(free_fingers) if free_fingers else [f for f in H.FINGERS if f not in F1]
        if set(free) & F1:
            raise ValueError(f"free fingers {sorted(set(free) & F1)} hold the first object")
        clamp = H.executor_clamp(rig, fp)
        held_t = dict(clamp["press"])                  # the held fingers as the first grasp clamps them
        # insertion shape: free fingers straight, together; spread: their abduction to the end that moves each tip away
        # from the free fingers' centroid
        straight = _straight(rig, free, held_t)
        d.qpos[:] = data0.qpos
        rig.teleport(d, np.zeros(3), np.zeros(3)); rig.set_fingers(d, straight); mujoco.mj_kinematics(model, d)
        tips0 = {f: d.site_xpos[rig.fingertip_sites[f]].copy() for f in free}
        cen0 = np.mean(list(tips0.values()), axis=0)
        f_ref = _unit(np.mean([tips0[f] - rig.knuckle_point(d, f) for f in free], axis=0))
        # spread: each outer free finger's abduction to the end that moves its tip away from the free fingers' centroid
        # (the middle one of three spreads to neither side: it stays and fills the gap); insertion: the same fingers
        # turned a third of that toward the centroid, so the bunch enters a narrow mouth
        spread, ins = dict(straight), dict(straight)
        for f in free:
            aa = [k for k in rig.finger_range if H.finger_of_joint(k) == f and H.joint_role(k) == "abduct"]
            for k in aa:
                lo, hi = rig.finger_range[k]
                best_v, best_d = 0.0, -np.inf
                for v in (lo, hi):
                    trial = dict(straight); trial[k] = v
                    rig.set_fingers(d, trial); mujoco.mj_kinematics(model, d)
                    dd = float(np.linalg.norm(d.site_xpos[rig.fingertip_sites[f]] - cen0))
                    if dd > best_d:
                        best_d, best_v = dd, v
                if best_d > float(np.linalg.norm(tips0[f] - cen0)) + 0.004:
                    spread[k] = best_v
                    ins[k] = float(np.clip(-best_v / 3.0, min(lo, hi), max(lo, hi)))
        rig.set_fingers(d, ins); mujoco.mj_kinematics(model, d)
        cen0 = np.mean([d.site_xpos[rig.fingertip_sites[f]] for f in free], axis=0)     # the bunch's tips go to the mouth
        # The first lift translates both the wrist and its held object. Their
        # relative transform is fixed at the original grasp; subtracting the
        # lift from only the wrist falsely lowers the carried collision model.
        p1 = np.asarray(fp["pos"], float)
        R1 = H.euler_to_mat(np.asarray(fp["euler"], float))
        T_o = (data0.xpos[r1].copy(), data0.xmat[r1].reshape(3, 3).copy())
        rel_p, rel_R = R1.T @ (T_o[0] - p1), R1.T @ T_o[1]
        R_al = _align(f_ref, [0.0, 0.0, -1.0])
        tilt_dirs = [np.array([np.cos(a), np.sin(a), 0.0]) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
        for tilt in tilts_deg:
            for td in ([None] if float(tilt) == 0.0 else tilt_dirs):
                Rt = np.eye(3) if td is None else Rot.from_rotvec(np.radians(float(tilt)) * _unit(np.cross([0.0, 0.0, 1.0], td))).as_matrix()
                for yaw in np.arange(0.0, 360.0, float(yaw_step_deg)):
                    R = Rt @ Rot.from_rotvec(np.radians(yaw) * np.array([0.0, 0.0, 1.0])).as_matrix() @ R_al
                    eul = H.mat_to_euler(R)
                    fdir = R @ f_ref
                    p = target - R @ cen0
                    # the first object riding the hand from its grasp
                    Ro1 = R @ rel_R
                    tilt1 = float(np.degrees(np.arccos(np.clip((Ro1[:, 2] @ T_o[1][:, 2]), -1.0, 1.0))))
                    info = {"first_pose_index": fi, "yaw_deg": float(yaw), "tilt_deg": float(tilt), "tilt_dir": None if td is None else td.round(3).tolist(),
                            "first_tilt_deg": round(tilt1, 1)}
                    if tilt1 > max_first_tilt_deg:
                        rejected["first_tilt"] = rejected.get("first_tilt", 0) + 1
                        continue

                    def place(pp, fingers):
                        d.qpos[:] = data0.qpos
                        rig.teleport(d, pp, eul); rig.set_fingers(d, fingers)
                        q = Rot.from_matrix(Ro1).as_quat()
                        a = int(model.jnt_qposadr[j1])
                        d.qpos[a:a + 3] = pp + R @ rel_p; d.qpos[a + 3:a + 7] = [q[3], q[0], q[1], q[2]]
                        mujoco.mj_kinematics(model, d)
                    why, clr = None, np.inf
                    for s_ in (0.0, 0.5, 1.0):         # the pose and the way in along the free fingers' axis
                        place(p - fdir * float(approach_m) * s_, ins)
                        # which pair is too close names the rejection (the hand or the first object against the second
                        # object, the table, or a bystander)
                        pairs = {"hand-second": (hand_g, g2), "hand-table": (hand_g, floor), "hand-bystander": (hand_g, others),
                                 "first-second": (g1, g2), "first-table": (g1, floor), "first-bystander": (g1, others)}
                        dd = {k_: (_min_dist(model, d, a_, b_) if a_ and b_ else np.inf) for k_, (a_, b_) in pairs.items()}
                        k_min = min(dd, key=dd.get)
                        clr = min(clr, dd[k_min])
                        if dd[k_min] < clear_m:
                            why = ("clearance:" if s_ == 0.0 else "approach_clearance:") + k_min
                            break
                    info["clear_mm"] = round(1000 * clr, 1)
                    if why is None:
                        # the spread swept from the insertion shape: where each spreading tip meets the wall (a target past
                        # a thin wall puts the tip beyond it, where the distance is positive again)
                        wall, hit = [], {}
                        for f in free:
                            ks = [k for k in rig.finger_range if H.finger_of_joint(k) == f and abs(spread[k] - ins[k]) > 1e-6]
                            tg = sorted(rig.tip_geoms(f))
                            if not ks or not tg:
                                continue
                            for u in np.linspace(0.0, 1.0, 13)[1:]:
                                fg = dict(ins)
                                for k in ks:
                                    fg[k] = ins[k] + u * (spread[k] - ins[k])
                                place(p, fg)
                                if _min_dist(model, d, tg, g2, 0.03) < 0.001:
                                    wall.append(f); hit[f] = d.site_xpos[rig.fingertip_sites[f]].copy() - target
                                    break
                        info["wall_fingers"] = wall
                        if len(wall) < 2:
                            why = "spread_misses_wall"
                        else:
                            v = [hit[f][:2] for f in wall]
                            if max(float(-(_unit(a) @ _unit(b))) for a in v for b in v) < 0.5:
                                why = "spread_one_side"      # the walls pressed must face each other
                    if why:
                        rejected[why] = rejected.get(why, 0) + 1
                        continue
                    pose = {"side": side, "pos": p.tolist(), "euler": eul.tolist(), "finger_targets": spread, "finger_targets_open": ins,
                            "grip_fingers": free, "press_depth_rad": 0.0, "approach": (-fdir).tolist(), "family": "inside",
                            "second_of": first_object, "object": second_object}
                    judged.append({"pose": pose, "info": info, "why": None})
    if bm is not None and judged:
        from tool_library.common import armhand as AH
        judged.sort(key=lambda j: (-j["info"]["clear_mm"], j["info"]["first_tilt_deg"]))
        # screened in batches down the clearance ranking until top_k poses pass the arm (a reachable pair can rank low on
        # clearance: the best-clearing ones are often the ones the arm cannot hold)
        batch, live, n_ok = max(4 * int(top_k), 12), [], 0
        probe = AH.ArmProbe(str(bench_scene))
        for i0 in range(0, min(len(judged), 10 * batch), batch):
            chunk = judged[i0:i0 + batch]
            scr = AH.screen_poses(bm, side, [j["pose"] for j in chunk], qb, probe=probe, approach_m=approach_m)
            for j, r in zip(chunk, scr):
                j["info"]["reach"] = {k: r.get(k) for k in ("ok", "residual_mm", "residual_deg", "reason")}
                if not r.get("ok"):
                    j["why"] = "arm"
                    rejected["arm"] = rejected.get("arm", 0) + 1
                else:
                    n_ok += 1
            live += chunk
            if n_ok >= int(top_k):
                break
        judged = live
    judged.sort(key=lambda j: (j["why"] is not None, -j["info"]["clear_mm"], j["info"]["first_tilt_deg"]))
    poses = []
    for j in judged[:max(1, int(top_k))]:
        ps = dict(j["pose"]); ps["second_grasp"] = dict(j["info"], ok=j["why"] is None)
        ps["first_pose_index"] = j["info"]["first_pose_index"]
        ps["first_pose"] = copy.deepcopy(firsts[ps["first_pose_index"]])
        poses.append(ps)
    physics_trials = []
    if verify_physics:
        poses, physics_trials = _verify_pairs(scene, [p for p in poses if p["second_grasp"]["ok"]],
                                             [first_object, second_object], max_physics_trials, top_k)
    best = poses[0] if poses and poses[0]["second_grasp"]["ok"] else None
    res = {"ok": best is not None, "pose": best, "poses": poses, "first_poses": firsts, "n_candidates": len(judged) + sum(rejected.values()), "rejected": rejected,
           "free_fingers": (best or {}).get("grip_fingers"), "mouth": {"rim_centre": mouth.round(4).tolist(), "inner_radius_m": round(r_in, 4)}}
    res.update(physics_verified=bool(verify_physics and best is not None), physics_trials=physics_trials)
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        if best is not None:
            Path(out_json).write_text(json.dumps(best, indent=1, default=float))
        Path(str(Path(out_json).with_suffix("")) + "_all.json").write_text(json.dumps(poses, indent=1, default=float))
        res["out_json"] = str(out_json)
    return res
