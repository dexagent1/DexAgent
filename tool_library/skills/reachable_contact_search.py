"""Where on a part can this hand make a contact the arm can reach? Sample the surface, pose, check reach, rank."""
from __future__ import annotations

TOOL = {
    "name": "reachable_contact_search",
    "title": "Reachable contact search",
    "type": "skill",
    "category": "Grasp & Poses",
    "summary": "Samples candidate contact points on a part (a face of a box geom, or points you give), builds a flat-hand pose at "
               "each, turns it for reach along the planned motion, and ranks the reachable ones by a criterion such as the "
               "farthest from a hinge: the answer to 'where can the arm act on this part'.",
    "description": (
        "Reach is a property of where the contact is, not only of how the hand is turned. A lid lifted near its hinge "
        "needs a force the base cannot take, farther out the arm cannot reach; the right contact is the farthest point the "
        "arm still reaches. This skill takes the bench scene, the side, the hand scene and either explicit `candidates` "
        "[{point, normal, finger_dir}] or a `face` of a box geom (`geom`, `face` in +x/-x/+y/-y/+z/-z, `n` points along "
        "each edge, `inset_m` from the edges) sampled in world coordinates with the outward normal. For every candidate "
        "it builds `flat_hand_pose` (contact 'pad' or 'back', `pitch_deg`, `fingers`, `thumb`), runs "
        "`optimize_pose_for_reach` with the given `sweep` (a joint motion the hand must follow, rigid or level) and "
        "`waypoints`, and records the worst residual, the overlap and the score: `score` is 'far_from' or 'near' a "
        "`ref_point` (a hinge, the base), or 'residual'. Reachable candidates (within `pos_tol_m`, no overlap beyond "
        "`max_overlap_m`) come first, ranked by the score; the best pose is written to `out_json`. The physics is still to be "
        "proved by executing the subgoal: this finds where a contact can be made, not whether it moves the part. "
        "Automatic finger directions cover several headings in the surface plane around the actual configured station. "
        "A bounded search visits distinct contact locations before extra headings/modes at the same point and reports "
        "location coverage separately from the number of pose variants evaluated. "
        "For a mechanism, sweep.hand_follows='auto' compares rigid and level following independently and returns the chosen "
        "mode in best.hand_follows; use that explicit mode in the plan and verify physical contact throughout execution. "
        "For carried objects, include later translation and orientation targets in waypoints before choosing an initial grasp. "
        "Sampled contacts use the hand scene's recorded bench_state by default; after parts move, rebuild the hand scene "
        "from the episode so far. An explicit inconsistent object state is rejected. co_sides must already contact the "
        "moving part in that state, not be parked at home or holding its stationary base. "
        "Absence of a witness proves neither unreachable geometry nor an upstream reconstruction error."
        " For a joint sweep, contact_drive reports whether pressure at the sampled surface can produce the requested "
        "signed joint effort, including a conservative bound on tangential and angular friction. Reachable contacts "
        "that drive by pressure rank before contacts requiring friction, and those before contacts that cannot drive. "
        "This assumes unilateral contact on that face; it does not establish actual contact, sufficient force, or "
        "passive motion. Opposing-thumb grasps are not rejected from a single face normal. require_drive explicitly "
        "assumes the sampled face is the only contact force source (no incidental thumb, edge or other contacts) and "
        "excludes candidates whose unilateral wrench cone cannot drive, while retaining them in the diagnostic rows."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "Bench scene MJCF."},
        {"name": "side", "type": "str", "description": "'left' or 'right'."},
        {"name": "hand_scene", "type": "dict | str", "description": "Hand scene (hand_scene_from_recon) of the part."},
        {"name": "candidates", "type": "list | None", "description": "[{point, normal, finger_dir}] explicit contacts (world)."},
        {"name": "geom", "type": "str | None", "description": "A box geom of the part to sample a face of (with `face`)."},
        {"name": "face", "type": "str", "description": "'+x' | '-x' | '+y' | '-y' | '+z' | '-z' in the geom's frame. Default '-z' (the underside)."},
        {"name": "n", "type": "int", "description": "Points per edge of the face grid. Default 4."},
        {"name": "inset_m", "type": "float", "description": "Keep this far from the face's edges. Default 0.02."},
        {"name": "finger_dir", "type": "list | None", "description": "Explicit world finger direction. If absent, search heading_samples headings around the actual station-to-contact projection in the face plane."},
        {"name": "heading_samples", "type": "int", "description": "Automatic in-plane finger headings. Default 4 (90-degree spacing, refined by range_deg). Explicit finger_dir on a candidate is respected without extra headings."},
        {"name": "contact", "type": "str", "description": "'pad' or 'back' for flat_hand_pose. Default 'back'."},
        {"name": "pitch_deg", "type": "float", "description": "flat_hand_pose pitch. Default 10."},
        {"name": "fingers", "type": "list | None", "description": "Fingers expected to touch. Default index, middle, ring."},
        {"name": "thumb", "type": "str", "description": "Default 'auto'."},
        {"name": "sweep", "type": "dict | None", "description": "{joint, to, hand_follows: 'rigid'|'level'|'auto', steps, co_sides} with moving-part collision checks. Auto compares both modes and returns the selection explicitly; execution must use the same mode."},
        {"name": "waypoints", "type": "list | None", "description": "World wrist offsets or {object, target_pos, target_euler_deg, steps} held-object targets; screen the whole intended manipulation, including rotation."},
        {"name": "score", "type": "str", "description": "'far_from' | 'near' | 'residual'. Default 'far_from'."},
        {"name": "ref_point", "type": "list | None", "description": "Reference for far_from / near (a point on the hinge axis, the base's centre)."},
        {"name": "ref_axis", "type": "list | None", "description": "With ref_point, the hinge axis direction: the score is then the lever (distance to the axis line)."},
        {"name": "pos_tol_m", "type": "float", "description": "Reach tolerance. Default 0.004."},
        {"name": "max_overlap_m", "type": "float", "description": "Default 0.003."},
        {"name": "range_deg", "type": "float", "description": "Primary turn range for the reach optimizer. Default 45."},
        {"name": "out_json", "type": "str | None", "description": "The best pose."},
        {"name": "state", "type": "str | list | None", "description": "Sample the face where the part is now: a trajectory.npz (its last frame) or a qpos; default hand_scene.bench_state when recorded, otherwise the scene keyframe. Explicit state must agree with the hand scene's object coordinates."},
        {"name": "time_budget_s", "type": "float | None", "description": "Total search budget (300 seconds), shared across sampled contacts. Finishes the current candidate's complete path before returning partial results. None permits an unbounded search. Absence of a witness is inconclusive."},
        {"name": "require_drive", "type": "bool", "description": "Default False. Opt into the assumption that ONLY the recorded grip-finger contacts on the sampled face supply force, excluding incidental thumb/edge contacts: exclude poses whose unilateral wrench cone cannot drive. Reach remains separately reported. Does not reject declared opposing-thumb contacts; execution is still required."},
    ],
    "outputs": [
        {"name": "result", "type": "dict", "description": "candidates [{point, normal, reachable, worst_pos_err_mm, worst_rot_err_deg, overlap_mm, score_m, rotation_deg}], n_reachable, best (index + pose), verdict."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "reach", "contact", "search", "lid", "non-prehensile"],
    "used_in": ["Giftbox Packing", "Place Toy in Drawer", "Battery Insertion", "Meal Box (extra sample)"],
    "status": "tested",
    "entrypoint": "reachable_contact_search",
}

import json
import time
from pathlib import Path

import mujoco
import numpy as np

from tool_library.common import episode as E

FACES = {"+x": (0, 1.0), "-x": (0, -1.0), "+y": (1, 1.0), "-y": (1, -1.0), "+z": (2, 1.0), "-z": (2, -1.0)}


def face_points(scene, geom, face="-z", n=4, inset_m=0.02, state=None):
    """World points and the outward normal on one face of a box geom at the scene's keyframe (or `state`)."""
    model = mujoco.MjModel.from_xml_path(str(scene)); data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    if state is not None:
        data.qpos[:] = E.state_qpos(model, state)
    mujoco.mj_forward(model, data)
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
    if gid < 0:
        raise ValueError(f"no geom {geom!r}")
    if int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_BOX):
        raise ValueError("face sampling needs a box geom; give explicit candidates for other shapes")
    if face not in FACES or isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("face must name a box face and n must be a positive integer")
    if not np.isfinite(inset_m) or inset_m < 0:
        raise ValueError("inset_m must be finite and nonnegative")
    ax, sgn = FACES[face]
    half = model.geom_size[gid].copy(); R = data.geom_xmat[gid].reshape(3, 3); c = data.geom_xpos[gid]
    others = [i for i in range(3) if i != ax]
    pts = []
    # A thin face may have no room for the requested inset. Collapse that
    # dimension to its centre instead of reversing the interval off the face.
    def grid(axis):
        extent = max(0., float(half[axis]) - inset_m)
        return [0.] if n == 1 or extent == 0 else np.linspace(-extent, extent, n)
    for u in grid(others[0]):
        for v in grid(others[1]):
            local = np.zeros(3); local[ax] = sgn * half[ax]; local[others[0]] = u; local[others[1]] = v
            pts.append((c + R @ local).tolist())
    normal = (R[:, ax] * sgn).tolist()
    return pts, normal


def _station_origin(scene, side, state=None):
    model = mujoco.MjModel.from_xml_path(str(scene)); data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    if state is not None:
        data.qpos[:] = E.state_qpos(model, state)
    joints = [j for j in range(model.njnt) if (model.joint(j).name or "").startswith(f"robot_{side}_arm_")]
    if not joints:
        raise ValueError(f"the {side} station has no arm")
    mujoco.mj_forward(model, data)
    return data.xpos[int(model.jnt_bodyid[joints[0]])].copy()


def _headings(point, normal, origin, n):
    normal = np.asarray(normal, float)
    if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-9:
        raise ValueError("contact normal must be a nonzero finite vector")
    normal = normal / np.linalg.norm(normal)
    direction = np.asarray(point, float) - origin
    direction -= normal * (normal @ direction)
    if np.linalg.norm(direction) < 1e-9:
        direction = np.eye(3)[np.argmin(np.abs(normal))]
        direction -= normal * (normal @ direction)
    direction /= np.linalg.norm(direction)
    tangent = np.cross(normal, direction)
    return [(np.cos(a) * direction + np.sin(a) * tangent).tolist() for a in np.linspace(0., 2 * np.pi, n, endpoint=False)]


def reachable_contact_search(scene, side, hand_scene, candidates=None, geom=None, face="-z", n=4, inset_m=0.02, finger_dir=None,
                             contact="back", pitch_deg=10.0, fingers=None, thumb="auto", sweep=None, waypoints=None, score="far_from",
                             ref_point=None, ref_axis=None, pos_tol_m=0.004, max_overlap_m=0.003, range_deg=45.0, out_json=None, state=None,
                             time_budget_s=300.0, heading_samples=4, require_drive=False):
    from tool_library.skills.flat_hand_pose import flat_hand_pose
    from tool_library.skills.optimize_pose_for_reach import optimize_pose_for_reach, _check_co_holders
    from tool_library.skills.hand_scene_from_recon import resolve_bench_state
    started = time.monotonic()
    if time_budget_s is not None and (not np.isfinite(time_budget_s) or time_budget_s <= 0):
        raise ValueError("time_budget_s must be positive and finite, or None")
    if isinstance(heading_samples, bool) or not isinstance(heading_samples, (int, np.integer)) or not 1 <= heading_samples <= 16:
        raise ValueError("heading_samples must be an integer in [1, 16]")
    deadline = float('inf') if time_budget_s is None else started + float(time_budget_s)
    if isinstance(hand_scene, (str, Path)):
        hand_scene = json.loads(Path(hand_scene).read_text())
    model = E.build_execution_model(str(scene))[0]
    resolved = resolve_bench_state(hand_scene, model, state)
    if resolved is not None:
        state = resolved
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    if resolved is not None:
        data.qpos[:] = resolved
    # Invalid participant state must fail once, before spending the search
    # budget repeatedly dragging a parked hand around every proposed contact.
    _check_co_holders(model, data, sweep)
    mujoco.mj_forward(model, data)
    from tool_library.common.mechanism_contact import contact_drive, friction_upper_bound
    drive_friction = None
    if sweep:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(sweep.get("joint", "")))
        if jid < 0:
            raise ValueError("sweep requires an existing mechanism joint")
        drive_friction = friction_upper_bound(model, side, E.subtree_bodies(model, int(model.jnt_bodyid[jid])))
    anchor = None
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(hand_scene.get("grasp_body", "")))
    if body > 0:
        from scipy.spatial.transform import Rotation
        quat = Rotation.from_matrix(data.xmat[body].reshape(3, 3)).as_quat()
        anchor = {"body": model.body(body).name, "pos": data.xpos[body].tolist(),
                  "quat_wxyz": [float(quat[3]), *map(float, quat[:3])]}
    if candidates is None:
        if geom is None:
            raise ValueError("give candidates or a geom + face to sample")
        pts, normal = face_points(scene, geom, face, n, inset_m, state=state)
        candidates = [{"point": p, "normal": normal, "finger_dir": finger_dir} for p in pts]
    modes = ["rigid", "level"] if sweep and sweep.get("hand_follows") == "auto" else [sweep.get("hand_follows", "rigid") if sweep else None]
    base, expanded = None, []
    for k, cnd in enumerate(candidates):
        direction = cnd.get("finger_dir") if cnd.get("finger_dir") is not None else finger_dir
        if direction is None:
            if base is None:
                base = _station_origin(scene, side, state)
            headings = _headings(cnd["point"], cnd["normal"], base, heading_samples)
        else:
            headings = [direction]
        for h, heading in enumerate(headings):
            for mode in modes:
                expanded.append(dict(cnd, finger_dir=heading, contact_index=k, heading_index=h, hand_follows=mode))
    # Spend a bounded budget across locations before returning to another
    # heading/mode at the same location. A complete failed path can consume
    # several seconds; exhausting all variants of the first point otherwise
    # leaves most of the requested surface completely unexplored.
    candidates = sorted(expanded, key=lambda c: (c["heading_index"], modes.index(c["hand_follows"]), c["contact_index"]))
    fingers = tuple(fingers) if fingers else ("index", "middle", "ring")
    rows, best, poses = [], None, {}
    for k, cnd in enumerate(candidates):
        remaining = deadline - time.monotonic()
        if rows and remaining <= 0:
            break
        per_contact = None if time_budget_s is None else max(.001, remaining / (len(candidates) - k))
        row = {"index": k, "contact_index": cnd["contact_index"], "heading_index": cnd["heading_index"], "hand_follows": cnd["hand_follows"],
               "finger_dir": cnd["finger_dir"], "point": [round(float(v), 4) for v in cnd["point"]], "normal": [round(float(v), 3) for v in cnd["normal"]]}
        try:
            pose_key = (cnd["contact_index"], cnd["heading_index"])
            if pose_key not in poses:
                poses[pose_key] = flat_hand_pose(hand_scene, cnd["point"], cnd["normal"], cnd["finger_dir"], fingers=fingers,
                                                contact=contact, thumb=thumb, pitch_deg=pitch_deg)
                if anchor:
                    # These explicit points were sampled in the bench state,
                    # not in the asset's older/settled local pose.
                    poses[pose_key]["anchor"] = anchor
            pose = poses[pose_key]
            selected_sweep = dict(sweep, hand_follows=cnd["hand_follows"]) if sweep else None
            r = optimize_pose_for_reach(scene, side, pose, hand_scene=hand_scene, axis="normal", range_deg=range_deg, step_deg=15.0, sweep=selected_sweep, waypoints=waypoints,
                                        pos_tol_m=pos_tol_m, max_overlap_m=max_overlap_m, state=state, time_budget_s=per_contact)
            reach = r["reach"]
            row.update({"reachable": bool(reach["ok"]), "worst_pos_err_mm": reach["worst_pos_err_mm"], "worst_rot_err_deg": reach["worst_rot_err_deg"],
                        "overlap_mm": reach.get("overlap_mm"), "overlap_with": reach.get("overlap_with"), "rotation_deg": r["rotation_deg"], "pose_overlap_mm": pose.get("overlap_mm"),
                        "contact_gap_mm": reach.get("contact_gap_mm"), "contact_gap_at": reach.get("contact_gap_at"),
                        "surface_point_error_mm": reach.get("surface_point_error_mm"),
                        "budget_reached": r["budget_reached"], "n_evaluated": r["n_evaluated"], "reach_checked": True, "eligible": True})
            if sweep:
                # Use all measured grip contacts after the reach rotation. The
                # sampled surface normal stays fixed in the object frame;
                # rotating the hand does not rotate the supporting surface.
                points = list((r["pose"].get("points") or {}).values()) or [cnd["point"]]
                drive = (contact_drive(model, data, sweep["joint"], float(sweep["to"]),
                                       points, [cnd["normal"]] * len(points), drive_friction)
                         if "thumb" not in fingers else {
                             "status": "not_assessed", "can_drive": None,
                             "reason": "single face normal does not describe an opposing-thumb grasp",
                             "execution_required": True})
                row["contact_drive"] = drive
                if require_drive and drive["can_drive"] is False:
                    row.update(eligible=False, exclusion_reason="recorded unilateral contacts cannot drive the requested joint direction")
            if ref_point is not None:
                dv = np.asarray(cnd["point"], float) - np.asarray(ref_point, float)
                if ref_axis is not None:                 # the lever about a hinge: distance to the axis line, not to a point on it
                    ax = np.asarray(ref_axis, float); ax /= max(np.linalg.norm(ax), 1e-9); dv = dv - ax * (ax @ dv)
                row["dist_to_ref_m"] = round(float(np.linalg.norm(dv)), 4)
            row["_pose"] = r["pose"]
        except Exception as e:                       # one bad candidate is a row, not a crash
            row.update({"reachable": False, "error": f"{type(e).__name__}: {e}"})
        rows.append(row)

    def key(row):
        if not row.get("reachable"):
            return (2, 0, row.get("worst_pos_err_mm", 1e9))
        if not row.get("eligible", True):
            return (1, 0, row.get("worst_pos_err_mm", 1e9))
        drive_rank = {"normal_drives": 0, "no_motion": 0, "friction_required": 1, "not_assessed": 2, "cannot_drive": 3}.get(
            row.get("contact_drive", {}).get("status"), 0)
        if score == "far_from" and "dist_to_ref_m" in row:
            return (0, drive_rank, -row["dist_to_ref_m"])
        if score == "near" and "dist_to_ref_m" in row:
            return (0, drive_rank, row["dist_to_ref_m"])
        return (0, drive_rank, row.get("worst_pos_err_mm", 0))
    rows.sort(key=key)
    n_ok = sum(1 for r in rows if r.get("reachable"))
    best = rows[0] if rows and rows[0].get("reachable") and rows[0].get("eligible", True) else None
    out = {"candidates": [{k: v for k, v in r.items() if k != "_pose"} for r in rows], "n_reachable": n_ok, "n_candidates": len(rows),
           "n_requested": len(candidates), "seconds": round(time.monotonic() - started, 3),
           "n_contact_locations_evaluated": len({r["contact_index"] for r in rows}),
           "n_contact_locations_requested": len({c["contact_index"] for c in candidates}),
           "n_drive_rejected": sum(r.get("eligible") is False for r in rows),
           "n_eligible_reachable": sum(bool(r.get("reachable")) and r.get("eligible", True) for r in rows),
           "contact_drive_scope": "nominal sampled surface; reach and execution are separate checks" if sweep else None,
           "budget_reached": len(rows) < len(candidates) or any(r.get("budget_reached") for r in rows),
           "best": None if best is None else {"index": best["index"], "point": best["point"], "pose": best["_pose"], "hand_follows": best["hand_follows"], "dist_to_ref_m": best.get("dist_to_ref_m"), "contact_drive": best.get("contact_drive")},
           "search_status": "feasible_witness" if best else "inconclusive",
           "verdict": (f"{n_ok} of {len(rows)} contacts reachable; best at {best['point']}" + (f", {best['dist_to_ref_m']:.3f} m from the reference" if best and best.get("dist_to_ref_m") is not None else "")
                       if best else f"No feasible witness among {len(rows)} sampled contacts (closest {min((r.get('worst_pos_err_mm', 1e9) for r in rows), default=None)} mm); finite search does not prove the part unreachable")}
    if out_json and best:
        Path(out_json).write_text(json.dumps(best["_pose"], indent=1, default=float)); out["out_json"] = str(out_json)
    return out
