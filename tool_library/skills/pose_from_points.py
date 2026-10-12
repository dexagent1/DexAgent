TOOL = {
    "name": "pose_from_points",
    "title": "Hand pose from fingertip points",
    "type": "skill",
    "category": "Grasp & Poses",
    "summary": "Solve a dexterous-hand wrist pose and finger joints that put each named fingertip pad on its own labelled surface point, keeping the rest of the hand out of the object.",
    "description": "The pose is solved, not searched. Each finger that is given a surface point and outward normal gets a target one pad thickness outside the surface; damped-least-squares inverse kinematics over the six virtual wrist DoF and every finger joint drives the fingertip sites onto those targets from a multi-start sweep of wrist seeds (back along the approach direction, over yaw and pitch). After a quarter of the iterations, interpenetration of the hand with the objects (the pads excepted: pressing the part is their job) and with the floor (pads included, and the hand's visible meshes against the floor plane, as the Stage 3 penetration audits judge it) is pushed out in the null space of the fingertip task; overlap_mm reports the same. Seeds are ranked by worst residual (to 2 mm), then deepest overlap (to 5 mm), then how squarely the palm faces the object. Fingers without a point are curled into the palm. The recorded approach (pads to wrist; the planner reaches in and backs out along it) is, for a one-sided contact (the normals agree: a press, a push, a cradle), the outward normal when the hand moved 3, 6 and 10 cm out along it stays clear of the scene and the floor (a fingertip pressing a strip comes down from above), else the pads-to-wrist direction in the surface plane (fingers slid under a lid); for an opposed grasp the pads-to-wrist direction. Works with any hand MJCF that follows the virtual-wrist and fingertip-site naming convention described in common/hand.py (Sharpa and Shadow hands included). Kinematic only: contact forces are measured afterwards by the grasp verifiers. The hand gets contact pairs only with the object geoms within its own span of the targets (pair_region 'auto') and the IK evaluates kinematics and collisions without the constraint solver, so a scene of many small parts (three magnetic rings, 75 geoms) costs well under a millisecond per iteration; `time_budget_s` caps the seed sweep and the result reports how many seeds ran.",
    "inputs": [
        {"name": "scene", "type": "dict", "description": "{hand_xml, side, objects:[{name, xml, pos, yaw_deg}], floor} - the last object is the one being grasped"},
        {"name": "points", "type": "dict finger->xyz (m, world)", "description": "surface points for the fingers that must touch, keys from thumb/index/middle/ring/pinky"},
        {"name": "normals", "type": "dict finger->xyz", "description": "outward surface normals at those points"},
        {"name": "standoff_m", "type": "float | None", "description": "fingertip-site standoff from the surface so the pad, not the site, rests on it (default: the pad radius plus 1 mm, read from the hand)"},
        {"name": "approach_dirs", "type": "list of xyz", "description": "directions the wrist seeds start back along (default: mean normal and +z)"},
        {"name": "palm_face_at", "type": "xyz", "description": "point the palm should face when ranking seeds (default: the target centroid)"},
        {"name": "parallel_fingers", "type": "bool", "description": "pin abduction to zero and the thumb to its opposition convention"},
        {"name": "curl_free", "type": "float", "description": "flexion fraction for fingers without a point (0 leaves them open)"},
        {"name": "iters, yaw_step_deg, pitches_deg, seed_back_m", "type": "numbers", "description": "IK budget and seed sweep density"},
        {"name": "seeds", "type": "list | None", "description": "explicit wrist seeds [(pos, euler), ...] instead of the sweep (a family of orientations the arm can reach, from arm_workspace_map)"},
        {"name": "fix_orientation", "type": "bool", "description": "keep each seed's wrist orientation and solve only the wrist position and the fingers (fingers that must stay level under a thin lid). Default false"},
        {"name": "clear_overlap", "type": "bool", "description": "for a one-sided contact, back the hand out along the surface normal until the fingers (pads included) just touch, so a contact made with the back of the fingers (palm down under a lid) is placed on the finger's back; the pose records cleared_m and contact_side. Default true"},
        {"name": "time_budget_s", "type": "float | None", "description": "stop trying further wrist seeds after this many seconds (scene build included) and return the best pose so far with a note; progress {seeds_tried, seeds_total, elapsed_s} is in the result either way. Default None (every seed)"},
        {"name": "pair_region", "type": "str | dict | None", "description": "which object geoms get hand contact pairs: 'auto' (default) those within the hand's own span of the target points (nothing the posed hand can reach is left out; a scene of many small parts no longer pays for every pair), a {center | points, radius_m} region, or None for every object geom"},
        {"name": "out_json", "type": "path", "description": "optional file to write the pose to"},
    ],
    "outputs": [
        {"name": "pose", "type": "dict", "description": "{side, pos, euler (intrinsic XYZ), finger_targets {joint: rad}, grip_fingers, approach (unit, pads to wrist: for a one-sided contact the outward surface normal when the hand can back out along it, else the pads-to-wrist direction in the surface plane), approach_kind (normal | along_surface | pad_to_wrist), points, normals, targets, residual_mm per finger, worst_residual_mm, overlap_mm (deepest overlap: non-pad geometry in the objects, any geometry or visible mesh in the floor), palm_facing, standoff_m, progress {seeds_tried, seeds_total, elapsed_s, stopped}, note (when the time budget cut the seed sweep short)}; the input every grasp verifier takes"},
    ],
    "status": "tested",
    "entrypoint": "pose_from_points",
    "requirements": ["mujoco>=3.1", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["grasp", "inverse-kinematics", "dexterous-hand", "fingertip"],
    "used_in": ["Giftbox Packing", "Knot the Rope", "Battery Insertion", "Drawing and Wiping", "Open the Bottle", "Scissor Cutting"],
}

import json
import numpy as np


def pose_from_points(scene, points, normals, standoff_m=None, approach_dirs=None,
                     palm_face_at=None, parallel_fingers=False, curl_free=0.65,
                     iters=320, yaw_step_deg=30, pitches_deg=(-90, -60, -30, 0, 30, 60, 90),
                     seed_back_m=0.13, seeds=None, fix_orientation=False, clear_overlap=True, out_json=None,
                     time_budget_s=None, pair_region="auto", _built=None):
    import time
    import mujoco
    from tool_library.common import hand as H

    t0 = time.monotonic()
    if not points:
        raise ValueError("points is empty")
    for f in points:
        if f not in normals:
            raise ValueError(f"finger '{f}' has a point but no normal")
    if _built is not None:                       # a caller solving many poses in one scene (generate_flat_pinches)
        model, data0, rig, ids = _built
    else:
        if pair_region == "auto":                # pairs for the object geoms the posed hand can reach, not every one
            pair_region = {"points": [list(map(float, p)) for p in points.values()]}
        elif pair_region is not None and not isinstance(pair_region, dict):
            raise ValueError("pair_region must be 'auto', a {center | points, radius_m} dict or None")
        model, data0, rig, ids = H.open_scene(scene, pair_region=pair_region)
    if standoff_m is None:                       # the pad touches when its site is one pad radius out
        standoff_m = H.pad_radius(model, rig, "index") + 0.001
    targets = {}
    for f, p in points.items():
        n = np.asarray(normals[f], float)
        n = n / (np.linalg.norm(n) + 1e-12)
        targets[f] = np.asarray(p, float) + n * float(standoff_m)
    cen = np.mean([targets[f] for f in targets], axis=0)
    if approach_dirs is None:
        nbar = np.mean([np.asarray(normals[f], float) for f in points], axis=0)
        approach_dirs = [nbar if np.linalg.norm(nbar) > 1e-6 else [0, 0, 1], [0.0, 0.0, 1.0]]
    if seeds is None:
        seeds = H.seed_sweep(cen, approach_dirs, back_m=seed_back_m,
                             yaw_step_deg=yaw_step_deg, pitches_deg=pitches_deg)
    stats = {}
    remaining = None if time_budget_s is None else max(0.0, float(time_budget_s) - (time.monotonic() - t0))
    worst, res, pen, qpos, face = H.solve_fingertip_ik(
        model, data0, rig, ids, targets, seeds,
        face_at=(palm_face_at if palm_face_at is not None else cen),
        parallel=parallel_fingers, iters=iters, fix_orientation=fix_orientation,
        time_budget_s=remaining, stats=stats)
    d = mujoco.MjData(model)
    d.qpos[:] = qpos
    if curl_free:
        curl_free = H.curl_free_fingers_clear(model, d, rig, ids, set(points), fracs=(float(curl_free), 0.85, 0.95, 0.45, 0.25, 0.1))
    mujoco.mj_forward(model, d)
    nbar = np.mean([np.asarray(normals[f], float) / (np.linalg.norm(normals[f]) + 1e-12) for f in points], axis=0)
    one_sided = np.linalg.norm(nbar) > 0.5
    cleared_m = 0.0
    if one_sided and clear_overlap:
        # a one-sided contact may touch with the back of the fingers (fingers under a lid, palm down): the pad
        # sites then sit inside the part. Back the whole hand out along the surface normal until the fingers
        # (pads included) just touch: the contact is the finger's back, and the sites record where they ended.
        nbar_u = nbar / np.linalg.norm(nbar)
        ids_part = dict(ids); ids_part["floor_gid"] = -1            # clear the part; the floor is reported, not dodged
        full = lambda: H.deepest_overlap(model, d, rig, ids_part, keep_tips=())
        if full() > 0.0015:
            base = rig.wrist_pos(d).copy() if hasattr(rig, "wrist_pos") else np.array([d.qpos[rig.wrist_qpos[k]] for k in ("x", "y", "z")])
            lo_t, hi_t = 0.0, 0.03
            for _ in range(10):
                mid = 0.5 * (lo_t + hi_t)
                for i, k in enumerate(("x", "y", "z")):
                    d.qpos[rig.wrist_qpos[k]] = base[i] + nbar_u[i] * mid
                mujoco.mj_forward(model, d)
                if full() > 0.0015:
                    lo_t = mid
                else:
                    hi_t = mid
            for i, k in enumerate(("x", "y", "z")):
                d.qpos[rig.wrist_qpos[k]] = base[i] + nbar_u[i] * hi_t
            mujoco.mj_forward(model, d)
            cleared_m = float(hi_t); qpos = d.qpos.copy()
            res = {f: float(np.linalg.norm(d.site_xpos[rig.fingertip_sites[f]] - targets[f])) for f in res}
            worst = max(res.values())
    # the pads may press the part, never the table; the floor is judged on the hand's visible meshes, as the Stage 3
    # penetration audits judge it (a scissors pinch reported 0-6 mm with the thumb's distal hull 7.5-9.9 mm in the table)
    overlap = H.deepest_overlap(model, d, rig, ids, keep_tips=tuple(points), visual_floor=True)
    pose = H.pose_from_data(model, d, rig, rig.side)
    # squeezed: pads 3 mm past the surface with the wrist fixed
    H.finger_ik(model, d, rig, {f: np.asarray(points[f], float) + np.asarray(normals[f], float) / (np.linalg.norm(normals[f]) + 1e-12) * (float(standoff_m) - 0.003) for f in points})
    squeezed = {k: float(d.qpos[a]) for k, a in rig.finger_qpos.items()}
    d.qpos[:] = qpos
    if curl_free:
        H.curl_free_fingers(rig, d, set(points), float(curl_free))
    mujoco.mj_forward(model, d)
    appr = np.asarray(pose["pos"], float) - cen     # the pads-to-wrist direction
    approach_kind = "pad_to_wrist"
    if one_sided:
        # a one-sided contact (a press, a push, a cradle) comes in along its surface normal when the space out along the
        # normal is free, as flat_hand_pose decides it: a fingertip pressing a strip on the table comes down from above and
        # leaves straight up (projected into the surface plane, it slid in along the table: the planner's retreat axis is
        # this approach). Where the hand cannot back out along the normal (fingers slid under a lid: the normal points into
        # the table) it enters along the surface, the pads-to-wrist direction with its normal part removed
        nbar = nbar / np.linalg.norm(nbar)
        dn = mujoco.MjData(model)
        dn.qpos[:] = d.qpos
        w0 = np.array([dn.qpos[rig.wrist_qpos[k]] for k in ("x", "y", "z")])

        def overlap_out(t):
            for i, k in enumerate(("x", "y", "z")):
                dn.qpos[rig.wrist_qpos[k]] = w0[i] + nbar[i] * t
            mujoco.mj_forward(model, dn)
            return H.deepest_overlap(model, dn, rig, ids, keep_tips=())
        if all(overlap_out(t) <= 0.002 for t in (0.03, 0.06, 0.1)):
            appr, approach_kind = nbar.copy(), "normal"
        else:
            appr, approach_kind = appr - nbar * (nbar @ appr), "along_surface"
    pose.update({
        "grip_fingers": sorted(points), "finger_targets_squeezed": squeezed, "press_depth_rad": 0.12, "curl_free": curl_free,
        "approach": (appr / max(np.linalg.norm(appr), 1e-9)).tolist(),       # the planner reaches in along it (pads to wrist)
        "approach_kind": approach_kind,
        "cleared_m": round(cleared_m, 4), "contact_side": "back" if cleared_m > 0.002 else "pad",
        "normals": {f: [float(v) for v in np.asarray(normals[f], float) / (np.linalg.norm(normals[f]) + 1e-12)] for f in points},
        "points": {f: [float(v) for v in points[f]] for f in points},           # the surface points themselves
        "residual_mm": {f: round(v * 1000, 2) for f, v in res.items()},
        "worst_residual_mm": round(worst * 1000, 2),
        "overlap_mm": round(overlap * 1000, 2),
        "palm_facing": round(float(face), 3),
        "standoff_m": float(standoff_m),
        "targets": {f: [float(v) for v in t] for f, t in targets.items()},
        "progress": {"seeds_tried": stats.get("seeds_tried"), "seeds_total": stats.get("seeds_total"),
                     "elapsed_s": round(time.monotonic() - t0, 2), "stopped": stats.get("stopped")},
    })
    if stats.get("stopped"):
        pose["note"] = (f"time budget of {float(time_budget_s):.0f} s reached after {stats['seeds_tried']} of {stats['seeds_total']} "
                        "wrist seeds: this is the best pose so far; give a larger time_budget_s or fewer seeds "
                        "(yaw_step_deg, pitches_deg, approach_dirs) for a full sweep")
    if out_json:
        with open(out_json, "w") as fh:
            json.dump(pose, fh, indent=1)
    return pose
