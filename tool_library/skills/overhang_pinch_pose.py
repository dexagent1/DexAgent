"""Edge pinch of a flat part where there is room under its edge: fingers under it, thumb on top."""
from __future__ import annotations

TOOL = {
    "name": "overhang_pinch_pose",
    "title": "Overhang pinch pose",
    "type": "skill",
    "category": "Grasp & Poses",
    "summary": "Solve an edge pinch of a flat part (a phone, a card, a plate) where a hand can get under its edge: the grip "
               "fingers under the part where it overhangs the table's edge (or where its end is tilted up), deeper than the "
               "thumb on its top, so the pinch also carries the part's moment, with an approach from beyond the edge.",
    "description": (
        "A part thinner than about two pad radii cannot be pinched from its sides where it lies (thin_grasp_diagnosis); "
        "where it overhangs the support's edge, or where one end is tilted up (tilt_up), fingers fit under it. A pinch "
        "there holds only if it carries the part's moment about the pinch: the thumb on the top `thumb_inset_m` in from "
        "the edge and the grip fingers on the underside deeper, at `finger_inset_m` (default from the moment balance "
        "F_thumb (f - t) = m g (d_com - f) with the thumb pressing at most 5 N, nonprehensile.overhang_need; up to 15 mm "
        "deeper when the part overhangs farther than that needs). The "
        "contact points come from rays onto the part's own geometry. The hand scene's floor is made the bench table as "
        "the finite box it is (nonprehensile.with_table_floor, from the scene's bench_scene or `bench_scene`), so the "
        "fingers below the tabletop beyond its edge are not counted as in the table, and the result carries that scene "
        "(`hand_scene`) for the stress tests. With `overhang_m` the part is first moved along `edge_dir` in the hand "
        "scene until it overhangs that much: the pose for where a slide will leave it (it records the part's pose it "
        "was solved for as `anchor`, and the executor re-anchors it to where the part really ends). For each finger "
        "direction in `finger_yaws_deg` (the finger axis turned about the vertical from straight in under the edge; an "
        "arm often reaches a slanted entry only) the fingertip IK is seeded palm-up with the fingers pointing in (for a raised end also with the hand in front of "
        "the edge and the fingers curled down and in under it: under a tilted part there is room for fingers, not a "
        "palm), and "
        "the approach is written: the pre-grasp straight above the pose, over to above the point where the open hand is "
        "clear of the part, straight down beside it to the pinch height, then in along the fingers (`approach` +z, "
        "`approach_waypoints`). A candidate is "
        "kept when its worst pad residual is under `max_residual_mm`, its overlap with the part and the table (visible "
        "meshes included) under `max_overlap_mm`, every grip finger's pad clears the table's edge (or top, for a raised "
        "edge) by 3 mm, and the open hand touches neither on the way in nor while it closes: the pose carries the "
        "opening it was checked with (`open_frac`, the widest rung of the executor's ladder 55-15 % the executor keeps "
        "(narrower only where the open fingers would be in the table at the pose), whose slide in does not sweep into "
        "the part and whose close does not swing a finger through it; a thumb opened 55 % over a 12 mm box curled into its end face and shoved it 14 mm). "
        "Candidates are ranked by residual and overlap; "
        "give them all to the plan entry (pose_candidates) so the executor picks the one the arm holds."
    ),
    "inputs": [
        {"name": "scene", "type": "dict | str", "description": "Hand scene (hand_scene_from_recon output or its json); the grasped part is its last object / grasp_body."},
        {"name": "edge_dir", "type": "list", "description": "[dx, dy]: horizontal outward direction of the part's edge to pinch (toward the table edge it overhangs, or toward its raised end)."},
        {"name": "overhang_m", "type": "float | None", "description": "Solve for the part slid along edge_dir until it overhangs the table's edge this far (default None: as it lies)."},
        {"name": "raised", "type": "bool", "description": "The edge is raised above the support (a tilted part) instead of overhanging its edge: the fingers' clearance is their height above the table; the part is held where it is in the hand scene (another hand presses it). Default false."},
        {"name": "grip", "type": "str", "description": "'edge' (default: fingers under, thumb on top), 'sides' (a raised end only: thumb and fingers across the part on its two side faces, near the raised end where they stand a pad radius + 1 mm above the table; under a tilted part there is room for fingers, not for a palm; such a pinch resists the part's weight about the line through its pads only by the pads' torsional friction: on 7-11 mm boxes 14-15 cm long tilted 30 mm it held on the bench but failed the stress ladder, lift or turn_roll, so expect few or no candidates for a long part) or 'auto' (edge, else sides)."},
        {"name": "fingers", "type": "list", "description": "Grip fingers under the part. Default ['index', 'middle', 'ring'] (two when the edge is too short for three)."},
        {"name": "s", "type": "float", "description": "Where along the edge (0..1). Default 0.5."},
        {"name": "thumb_inset_m, finger_inset_m", "type": "float | None", "description": "Insets from the edge (defaults: pad radius + 3 mm; from the moment balance)."},
        {"name": "finger_yaws_deg", "type": "list[float]", "description": "Finger directions to try, degrees about the vertical from straight in. Default [0, 25, -25, 50, -50]."},
        {"name": "thumb_shifts_m", "type": "list[float]", "description": "Thumb positions along the edge tried with each direction, from the fingers' centre (a turned hand puts its thumb to one side). Default [0, 0.012, -0.012]."},
        {"name": "arm_screen", "type": "bool", "description": "With a bench scene (`bench_scene` or the hand scene's), keep only the candidates the station's arm holds at the pre-grasp, the approach waypoints, the grasp and the lift (armhand.screen_poses; dropped ones in `arm_dropped`). Default true."},
        {"name": "max_residual_mm, max_overlap_mm", "type": "float", "description": "Acceptance (defaults 3, 2)."},
        {"name": "bench_scene", "type": "str | None", "description": "Bench MJCF the table is read from (default the scene's bench_scene)."},
        {"name": "iters", "type": "int", "description": "IK iterations per seed. Default 200."},
        {"name": "screen_lift", "type": "bool", "description": "Run each candidate through the grasp_stress_test ladder in the hand scene (approach, close, lifts of 6, 10 and 15 cm, 4 cm carries in five directions, 25 deg turns, hold; hand.stress_rollout) and drop the ones that fail (listed in `dropped`). Default true."},
        {"name": "out_json", "type": "str | None", "description": "Write {pose, poses, hand_scene, ...} here."},
    ],
    "outputs": [
        {"name": "result", "type": "dict", "description": "ok, pose (best candidate), poses (ranked; each with pos, euler, finger_targets, open_frac, grip_fingers, points, normals, approach, approach_m, approach_waypoints, anchor, edge_pinch {edge_dir, need_overhang_m, finger_inset_m, thumb_inset_m, raised}, finger_yaw_deg, worst_residual_mm, overlap_mm, finger_clearance_mm), hand_scene (the table-floor scene the poses were solved in: give it to grasp_lift_test / grasp_stress_test), need_overhang_m, overhang_m (of the part in that scene), tried [{finger_yaw_deg, ok, why}]."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["grasp", "pinch", "thin-object", "flat", "overhang", "edge", "phone", "stage3"],
    "status": "tested",
    "entrypoint": "overhang_pinch_pose",
}

import json
from pathlib import Path

import mujoco
import numpy as np

from tool_library.common import hand as H
from tool_library.common import nonprehensile as NP


def overhang_pinch_pose(scene, edge_dir, overhang_m=None, raised=False, fingers=("index", "middle", "ring"), s=0.5, thumb_inset_m=None,
                        finger_inset_m=None, finger_yaws_deg=(0.0, 25.0, -25.0, 50.0, -50.0), thumb_shifts_m=(0.0, 0.012, -0.012),
                        max_residual_mm=3.0, max_overlap_mm=2.0, bench_scene=None, iters=200, arm_screen=True, screen_lift=True, grip="edge",
                        out_json=None):
    from scipy.spatial.transform import Rotation as Rot
    if raised and grip == "auto":
        r = overhang_pinch_pose(scene, edge_dir, raised=True, fingers=fingers, s=s, thumb_inset_m=thumb_inset_m, finger_inset_m=finger_inset_m,
                                finger_yaws_deg=finger_yaws_deg, thumb_shifts_m=thumb_shifts_m, max_residual_mm=max_residual_mm,
                                max_overlap_mm=max_overlap_mm, bench_scene=bench_scene, iters=iters, arm_screen=arm_screen,
                                screen_lift=screen_lift, grip="edge")
        if r["ok"]:
            _write(r, out_json)
            return r
        r2 = overhang_pinch_pose(scene, edge_dir, raised=True, fingers=fingers, max_residual_mm=max_residual_mm, max_overlap_mm=max_overlap_mm,
                                 bench_scene=bench_scene, iters=iters, arm_screen=arm_screen, screen_lift=screen_lift, grip="sides", out_json=out_json)
        r2["edge_grip_why"] = r.get("why")
        return r2
    spec = NP.with_table_floor(scene, bench_scene)
    if not raised and not isinstance(spec.get("floor"), dict):
        raise ValueError("overhang_pinch_pose: the hand scene has no bench_scene to read the table from: give bench_scene (an "
                         "overhang is measured against the table's edge)")
    fingers = [f for f in (fingers or ("index", "middle", "ring")) if f != "thumb"]
    if not fingers:
        raise ValueError("fingers must name at least one finger besides the thumb")
    e2 = NP._horizontal(edge_dir, "edge_dir")[:2]
    if raised and grip == "sides":
        return _raised_side_pinch(spec, e2, fingers, max_residual_mm, max_overlap_mm, bench_scene, iters, arm_screen, screen_lift, out_json)
    model, d0, rig, ids = H.open_scene(spec)
    bid = H.grasp_body(model, spec, ids)
    bodies = NP.E.subtree_bodies(model, bid)
    fl = NP.table_box(model, d0)
    rect = NP._support_rect({"center": spec["floor"]["center"], "half_extents": spec["floor"]["half_extents"],
                             "quat": spec["floor"].get("quat", (1, 0, 0, 0))}) if isinstance(spec.get("floor"), dict) else None
    if overhang_m is not None:
        if rect is None:
            raise ValueError("overhang_m needs the table (bench_scene)")
        cur, _ = NP.overhang_along(NP.footprint(model, d0, bodies), rect, e2)
        spec = NP.shift_object(spec, [e2[0] * (float(overhang_m) - cur), e2[1] * (float(overhang_m) - cur), 0.0])
        model, d0, rig, ids = H.open_scene(spec)
        bid = H.grasp_body(model, spec, ids)
        bodies = NP.E.subtree_bodies(model, bid)
    dims = NP.hand_dims(model, rig)
    fp = NP.footprint(model, d0, bodies)
    mass = float(sum(model.body_mass[b] for b in bodies))
    reach = NP.extent_along(fp, e2)
    com_in = float(reach - (np.asarray(d0.subtree_com[bid][:2]) - fp["centre"]) @ e2)
    need, f_in, t_in = NP.overhang_need(dims, mass, com_in, thumb_inset_m)
    if finger_inset_m is not None:
        f_in = float(finger_inset_m)
        need = f_in + 0.5 * dims["tip_thickness_m"] + NP.EDGE_CLEAR_M
    ov_now = NP.overhang_along(fp, rect, e2)[0] if rect is not None else None
    if raised:
        # a raised (tilted) end: the part is held where it is (another hand presses it), and the fingers go as deep as the
        # gap under it leaves room for a fingertip off the table, at most 15 mm deeper than the moment balance asks
        for o in spec["objects"]:
            if o is spec["objects"][-1]:
                o["hold"] = True
        model, d0, rig, ids = H.open_scene(spec)
        bid = H.grasp_body(model, spec, ids); bodies = NP.E.subtree_bodies(model, bid); fp = NP.footprint(model, d0, bodies)
        gs = NP.hand_geoms(model, bodies)
        top_z = float(H.floor_surface(model, d0, ids["floor_gid"])[0][2]) if ids.get("floor_gid", -1) >= 0 else 0.0
        reach_e = NP.extent_along(fp, e2)
        best = None
        if finger_inset_m is None:
            for cand in np.arange(f_in + 0.015, t_in + dims["tip_thickness_m"] - 1e-6, -0.002):
                xy = fp["centre"] + e2 * (reach_e - cand)
                zb = NP.surface_z(model, d0, gs, xy, fp["z_low"] - 0.05, up=True)
                if zb is not None and zb - top_z >= 2 * dims["pad_radius_m"] + NP.TABLE_CLEAR_M + 0.002:
                    best = float(cand); break
            if best is None:
                out = {"ok": False, "pose": None, "poses": [], "hand_scene": spec, "tried": [],
                       "why": "the raised end leaves no fingertip-thick gap under it a fingertip deeper than the thumb: tilt it further"}
                _write(out, out_json)
                return out
            f_in = best
    if finger_inset_m is None and ov_now is not None and not raised:
        # deeper fingers carry the part's moment with less thumb force: use the overhang there is, up to 15 mm deeper
        room = ov_now - 0.5 * dims["tip_thickness_m"] - NP.EDGE_CLEAR_M - 0.003
        f_in = max(f_in, min(room, f_in + 0.015))
        need = max(need, f_in + 0.5 * dims["tip_thickness_m"] + NP.EDGE_CLEAR_M)
    try:
        pts, nrm, info = NP.edge_pinch_targets(model, d0, bodies, e2, t_in, f_in, fingers, s=s)
    except ValueError:
        if len(fingers) < 3:
            raise
        fingers = fingers[:2]                       # a narrow edge: two fingers under it
        pts, nrm, info = NP.edge_pinch_targets(model, d0, bodies, e2, t_in, f_in, fingers, s=s)
    clear = NP.finger_support_clearance(model, d0, ids.get("floor_gid", -1), pts, fingers, dims)
    out = {"ok": False, "pose": None, "poses": [], "hand_scene": spec, "need_overhang_m": round(need, 4),
           "overhang_m": None if ov_now is None else round(ov_now, 4), "finger_inset_m": round(f_in, 4), "thumb_inset_m": round(t_in, 4),
           "finger_clearance_mm": {f: round(1000 * v, 1) for f, v in clear.items()}, "tried": []}
    if min(clear.values()) < NP.EDGE_CLEAR_M - 1e-4:
        out["why"] = (f"the grip fingers {1000 * f_in:.0f} mm under the part would be {1000 * min(clear.values()):.1f} mm clear of the table "
                      f"(at least {1000 * NP.EDGE_CLEAR_M:.0f} mm): it overhangs {1000 * (ov_now or 0):.0f} mm, the pinch needs "
                      f"{1000 * need:.0f} mm (slide it farther, or give overhang_m)" if not raised else
                      "the raised edge leaves no fingertip-thick gap under it: tilt it further")
        _write(out, out_json)
        return out
    so = H.pad_radius(model, rig, "index") + 0.001
    tg = {f: np.asarray(pts[f], float) + np.asarray(nrm[f], float) * so for f in pts}
    pk = H.grasp_pocket(model, d0, rig)
    anchor = NP.body_anchor(model, d0, bid)
    # the part's footprint as a rectangle (centre, axes, half sizes) for where the fingertips leave it
    prect = (fp["centre"], fp["u"], fp["v"], fp["half_u"], fp["half_v"], fp["z_high"])
    along = np.array([info["along_dir"][0], info["along_dir"][1], 0.0])
    families = []
    for yaw in finger_yaws_deg:
        for shift in thumb_shifts_m:
            families.append((float(yaw), float(shift), list(fingers)))
        if len(fingers) > 2:                         # the same with two fingers (a third often finds no room)
            families.append((float(yaw), 0.0, list(fingers[:2])))
    for yaw, shift, fset in families:
        din = Rot.from_euler("z", float(yaw), degrees=True).as_matrix() @ np.array([-e2[0], -e2[1], 0.0])
        tg_f = {f: tg[f] for f in ["thumb"] + fset}
        tg_f["thumb"] = tg["thumb"] + along * shift          # the thumb where this hand heading puts it naturally
        pts_f = {f: pts[f] for f in tg_f}; pts_f["thumb"] = pts["thumb"] + along * shift
        seeds = NP.edge_pinch_seeds(model, d0, rig, tg_f, -din[:2], front=bool(raised))
        worst, res, pen, q, face = H.solve_fingertip_ik(model, d0, rig, ids, tg_f, seeds, iters=int(iters))
        d = mujoco.MjData(model); d.qpos[:] = q; mujoco.mj_forward(model, d)
        H.curl_free_fingers_clear(model, d, rig, ids, set(tg_f))
        mujoco.mj_forward(model, d)
        ovl = H.deepest_overlap(model, d, rig, ids, keep_tips=tuple(tg_f), visual_floor=True)
        pose = H.pose_from_data(model, d, rig, rig.side)
        Rw = H.euler_to_mat(pose["euler"])
        fh = Rw @ np.asarray(pk["f"], float); fh[2] = 0.0
        fh = NP._unit(fh) if np.linalg.norm(fh) > 1e-6 else -NP._unit([e2[0], e2[1], 0.0])
        # back the fingertips out along the fingers until they are clear of the part, at the pinch height
        back = max(NP.exit_distance(prect, tg_f[f][:2], -fh[:2])[0] for f in fset)
        back = (back if np.isfinite(back) else 0.05) + dims["pad_radius_m"] + 0.008
        up = np.array([0.0, 0.0, 0.08])
        rec = {"finger_yaw_deg": float(yaw), "thumb_shift_m": shift, "fingers": fset, "worst_residual_mm": round(worst * 1000, 2), "overlap_mm": round(ovl * 1000, 2)}
        if worst * 1000 > max_residual_mm or ovl * 1000 > max_overlap_mm:
            rec.update(ok=False, why="residual or overlap over the limit"); out["tried"].append(rec); continue
        # the open hand (as the executor opens it) comes straight down above the waypoint: at the waypoint and on the way
        # down to it nothing of it may touch the part or the table (a thumb still over the part pushed it through the
        # table), so the waypoint backs out along the fingers until the whole column is clear
        closed = {k: float(v) for k, v in pose["finger_targets"].items()}
        dd = mujoco.MjData(model); dd.qpos[:] = d0.qpos

        def _ovl(off, fingers_q):
            dd.qpos[:] = d0.qpos
            rig.teleport(dd, np.asarray(pose["pos"]) + off, pose["euler"]); rig.set_fingers(dd, fingers_q); mujoco.mj_forward(model, dd)
            return H.deepest_overlap(model, dd, rig, ids, keep_tips=(), visual_floor=True)

        # the executor's opening (open_frac on the pose: it opens that far, narrower where the open fingers would be in
        # the table at the pose, down to its last rung), chosen as the widest rung of its ladder the executor would keep
        # (NP.executor_keeps_open) for which (1) the whole column above the waypoint is clear (a thumb still over the
        # part pushed it through the table), (2) the slide in along the fingers does not sweep the hand into the part
        # and (3) closing from it does not swing a finger through the part (a 12 mm box: the thumb opened 55 % curled
        # into its end face and shoved it 14 mm back onto the table, and the pinch let go at the top of the lift);
        # (2) and (3) are measured against the part, no deeper than the closed pose grazes it + 0.5 mm
        wp, wov, frac_ok, why = None, None, None, None
        for frac in NP.OPEN_LADDER:
            op = H.approach_open_targets(rig, closed, frac, ["thumb"] + fset, False)
            if not NP.executor_keeps_open(model, d0, rig, ids, pose, op, frac):
                why = "the open fingers are in the table at the pose"
                continue
            wp = None
            for extra in np.arange(0.0, 0.10, 0.01):
                cand = -fh * (back + extra)
                wov = max(_ovl(cand + [0.0, 0.0, dz], op) for dz in (0.0, 0.015, 0.03, 0.05, 0.08))
                if wov <= 0.001:
                    wp = cand
                    break
            if wp is None:
                why = f"no approach column beside the part is clear for the open hand ({1000 * wov:.1f} mm overlap)"
                continue
            sw = NP.sweep_into_part(model, d0, rig, ids, pose, closed, op, [wp * (1.0 - fr) for fr in (0.25, 0.5, 0.75, 0.9)])
            if sw:
                why = sw
                continue
            frac_ok = frac
            break
        if frac_ok is None:
            rec.update(ok=False, why=why)
            out["tried"].append(rec); continue
        # pre-grasp straight above the pose (clear of the part at any approach distance), over to above the waypoint,
        # down beside the part, then in along the fingers
        pose.update({"grip_fingers": ["thumb"] + fset, "points": {f: v.tolist() for f, v in pts_f.items()},
                     "normals": {f: nrm[f].tolist() for f in pts_f}, "targets": {f: v.tolist() for f, v in tg_f.items()},
                     "approach": [0.0, 0.0, 1.0], "approach_m": 0.09,
                     "approach_waypoints": [(wp + up).round(5).tolist(), wp.round(5).tolist()], "anchor": anchor, "press_depth_rad": 0.12,
                     "open_frac": frac_ok,
                     "edge_pinch": {"edge_dir": e2.round(6).tolist(), "need_overhang_m": round(need, 4), "finger_inset_m": round(f_in, 4),
                                    "thumb_inset_m": round(t_in, 4), "raised": bool(raised)},
                     "finger_yaw_deg": float(yaw), "thumb_shift_m": shift, "worst_residual_mm": rec["worst_residual_mm"], "overlap_mm": rec["overlap_mm"],
                     "residual_mm": {f: round(v * 1000, 2) for f, v in res.items()}, "standoff_m": so,
                     "finger_clearance_mm": out["finger_clearance_mm"]})
        rec.update(ok=True); out["tried"].append(rec)
        out["poses"].append(pose)
    out["poses"].sort(key=lambda p: (round(p["worst_residual_mm"]), p["overlap_mm"]))
    bench = bench_scene or spec.get("bench_scene")
    if arm_screen and bench and Path(str(bench)).exists() and out["poses"]:
        # the station's arm: candidates it cannot hold (at the pre-grasp, the approach waypoints, the grasp, the lift) go
        scr = NP.arm_screen(bench, rig.side, out["poses"])
        for pz, (ok_, mm, why) in zip(out["poses"], scr):
            pz["arm_screen"] = {"ok": ok_, "residual_mm": mm, "reason": why}
        out["arm_dropped"] = [{"finger_yaw_deg": pz["finger_yaw_deg"], **pz["arm_screen"]} for pz in out["poses"] if not pz["arm_screen"]["ok"]]
        out["poses"] = [pz for pz in out["poses"] if pz["arm_screen"]["ok"]]
    if screen_lift and out["poses"]:
        # the approach-close-lift rollout in the hand scene: only candidates that lift are given to the executor
        res = NP.lift_screen(spec, out["poses"])
        for pz, (ok_, ff, mm) in zip(out["poses"], res):
            pz["lift_screen"] = {"ok": ok_, "first_fail": ff, "lift_mm": mm}
        out["dropped"] = [{"index": i, **pz["lift_screen"]} for i, pz in enumerate(out["poses"]) if not pz["lift_screen"]["ok"]]
        out["poses"] = [pz for pz in out["poses"] if pz["lift_screen"]["ok"]]
    if out["poses"]:
        out.update(ok=True, pose=out["poses"][0])
    else:
        out["why"] = "no finger direction gave a pinch within the residual and overlap limits"
    _write(out, out_json)
    return out


def _raised_side_pinch(spec, e2, fingers, max_residual_mm, max_overlap_mm, bench_scene, iters, arm_screen, screen_lift, out_json):
    """grip 'sides' on a raised (tilted) end: the thumb on one side face, the grip fingers on the other, across the part
    near the raised end where the faces' middle stands a pad radius + 1 mm above the table (the deepest such depth up to
    40 mm in), approached from above."""
    spec = dict(spec)
    spec["objects"] = [dict(o) for o in spec["objects"]]
    spec["objects"][-1]["hold"] = True
    model, d0, rig, ids = H.open_scene(spec)
    bid = H.grasp_body(model, spec, ids)
    bodies = NP.E.subtree_bodies(model, bid)
    gs = NP.hand_geoms(model, bodies)
    fp = NP.footprint(model, d0, bodies)
    dims = NP.hand_dims(model, rig)
    top_z = float(H.floor_surface(model, d0, ids["floor_gid"])[0][2]) if ids.get("floor_gid", -1) >= 0 else 0.0
    t = np.array([-e2[1], e2[0]])
    reach = NP.extent_along(fp, e2)
    rel = fp["points"][:, :2] - fp["centre"]
    out = {"ok": False, "pose": None, "poses": [], "hand_scene": spec, "grip": "sides", "tried": []}
    fset = list(fingers)[:2]
    best = None
    for dep in np.arange(0.040, 0.009, -0.003):
        xy0 = fp["centre"] + e2 * (reach - dep)
        lo, hi = NP.footprint_chord(fp, e2, dep)
        zs = []
        for side in (lo + 0.002, hi - 0.002):
            q = xy0 + t * (side - float((xy0 - fp["centre"]) @ t))
            zt = NP.surface_z(model, d0, gs, q, fp["z_high"] + 0.05)
            zb = NP.surface_z(model, d0, gs, q, fp["z_low"] - 0.05, up=True)
            if zt is None or zb is None:
                break
            zs.append(0.5 * (zt + zb))
        if len(zs) == 2 and min(zs) - top_z >= dims["pad_radius_m"] + NP.TABLE_CLEAR_M + 0.001:
            best = (float(dep), lo, hi, zs); break
    if best is None:
        out["why"] = "the raised end's side faces do not stand a pad radius above the table anywhere 1-4 cm in from it: tilt it further"
        return _write(out, out_json)
    dep, lo, hi, zs = best
    base = fp["centre"] + e2 * (reach - dep)
    base = base - t * float((base - fp["centre"]) @ t)
    pts = {"thumb": np.array([*(base + t * lo), zs[0]])}; nrm = {"thumb": np.array([-t[0], -t[1], 0.0])}
    for k, f in enumerate(fset):
        off = (k - 0.5 * (len(fset) - 1)) * 0.019
        pts[f] = np.array([*(base + t * hi + e2 * off), zs[1]]); nrm[f] = np.array([t[0], t[1], 0.0])
    so = H.pad_radius(model, rig, "index") + 0.001
    tg = {f: pts[f] + nrm[f] * so for f in pts}
    grip = ["thumb"] + fset
    cands, tried = NP.solve_sector_poses(spec, (model, d0, rig, ids), pts, nrm, tg, approach_dirs=((0.0, 0.0, 1.0),), sectors=8, iters=iters,
                                         max_residual_mm=max_residual_mm, max_overlap_mm=max_overlap_mm, grip=grip, caps=True, arm=(bench_scene or spec.get("bench_scene"), rig.side) if arm_screen else None)
    out["tried"] = tried
    anchor = NP.body_anchor(model, d0, bid)
    for pz in cands:
        # opened only as far as the descent from above and the close stay out of the part and the table
        frac, why = NP.pick_open_frac(model, d0, rig, ids, pz, grip, path_offsets=[(0.0, 0.0, dz) for dz in (0.015, 0.03, 0.05, 0.08)])
        if frac is None:
            out["tried"].append({"sector_deg": pz.get("sector_deg"), "ok": False, "why": why})
            continue
        pz.update({"grip_fingers": grip, "normals": {f: v.tolist() for f, v in nrm.items()}, "approach": [0.0, 0.0, 1.0], "approach_m": 0.08,
                   "open_frac": frac, "anchor": anchor, "press_depth_rad": 0.15, "edge_pinch": {"edge_dir": e2.round(6).tolist(), "raised": True, "grip": "sides",
                                                                             "inset_m": round(dep, 4)}})
        out["poses"].append(pz)
    bench = bench_scene or spec.get("bench_scene")
    if arm_screen and bench and Path(str(bench)).exists() and out["poses"]:
        scr = NP.arm_screen(bench, rig.side, out["poses"])
        for pz, (ok_, mm, why) in zip(out["poses"], scr):
            pz["arm_screen"] = {"ok": ok_, "residual_mm": mm, "reason": why}
        out["arm_dropped"] = [pz["arm_screen"] for pz in out["poses"] if not pz["arm_screen"]["ok"]]
        out["poses"] = [pz for pz in out["poses"] if pz["arm_screen"]["ok"]]
    if screen_lift and out["poses"]:
        res = NP.lift_screen(spec, out["poses"])
        for pz, (ok_, ff, mm) in zip(out["poses"], res):
            pz["lift_screen"] = {"ok": ok_, "first_fail": ff, "lift_mm": mm}
        out["dropped"] = [pz["lift_screen"] for pz in out["poses"] if not pz["lift_screen"]["ok"]]
        out["poses"] = [pz for pz in out["poses"] if pz["lift_screen"]["ok"]]
    if out["poses"]:
        out.update(ok=True, pose=out["poses"][0])
    else:
        out["why"] = "no side pinch of the raised end within the limits, held by the arm and through the stress rollout"
    return _write(out, out_json)


def _write(out, out_json):
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(out, indent=1, default=float))
        out["out_json"] = str(out_json)
    return out
