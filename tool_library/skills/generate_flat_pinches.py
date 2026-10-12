"""Pinches for a flat object lying on the table: thumb against one or two fingers across its width or length."""
from __future__ import annotations

TOOL = {
    "name": "generate_flat_pinches",
    "title": "Generate flat-object pinches",
    "type": "skill",
    "category": "Grasp & Poses",
    "summary": "Thumb-versus-finger pinches across a flat object's short or long axis at mid-height (a hard drive, a phone, a "
               "plush toy, a lid lying on the table), solved by pose_from_points, screened for fingers in the table and, with "
               "bench_scene, for the station's arm (at the grasp, the pre-grasp and any waypoints), then lift-tested.",
    "description": (
        "generate_rigid_grasps wraps or pinches an object from its sides at fractions of its height and finds nothing on "
        "a part 2-3 cm tall: the finger links reach the table before the pads reach the faces. This skill measures the "
        "object's footprint in its own horizontal axes, and for each axis (short first) and each side the thumb takes, "
        "places the thumb pad and one or two finger pads on the two opposite faces at `height_frac` of the object's height "
        "(never lower than the pad radius plus `table_clear_m` above the table), at the centre of the face and at "
        "`along_fracs` of its length, and solves each with pose_from_points from above-behind seeds (pitched, wrist up). "
        "Every contact point is ray-cast on the part's own collision shapes from outside along the pinch axis (a rounded "
        "part gets pads on its real surface, not on its bounding box), with the normal from two neighbouring rays. "
        "A pose whose hand penetrates the table or the object beyond `max_overlap_mm` is dropped, and so is one whose "
        "pre-opened hand, on its approach line or while closing, goes more than `max_overlap_mm` into anything other than "
        "the part (the walls of a tray or a drawer, a neighbour): rejected.approach counts them. With bench_scene every "
        "pose is screened for the arm like generate_rigid_grasps does (arm_waypoints = the carry's displacement), and the "
        "best `verify_top` are lift-tested with the executor's full clamp (a ladder of clamp depths and pre-openings, then the wrist "
        "seated 2 mm deeper along its approach; a seated pose carries `seat_m`). Every pose carries hold {opposed, "
        "opposition_deg, need_n}: the force each planned contact must push to hold the part at 1.5 g tilted 30 deg with "
        "realistic (2 cm) twisting friction (hand.grip_hold_quality); pinches that hold with margin rank first, so a pinch "
        "line through the centre of mass beats one 30-45 mm off it (a 600 g drive twisted out of those). Returns the ranked poses, the format every grasp "
        "verifier and execute_subgoal_plan take (give the list as the subgoal's pose: the executor falls back down it). The hand scene is "
        "built once with contact pairs only near the part, every candidate is solved in it, and `time_budget_s` bounds the call: when it "
        "runs out the poses found so far come back with a note and the progress counts."
    ),
    "inputs": [
        {"name": "scene", "type": "dict | str", "description": "Hand scene {hand_xml, side, objects, floor, grasp_body} or its JSON (hand_scene_from_recon); the last object (or grasp_body) is grasped."},
        {"name": "fingers", "type": "list[list[str]]", "description": "Finger sets opposing the thumb. Default [['index'], ['middle'], ['index', 'middle']] (thumb against the middle finger near one end lifted a 600 g, 24 mm drive that no index pinch lifted)."},
        {"name": "axes", "type": "list[str]", "description": "'short' and/or 'long': the axis the pinch spans. Default ['short', 'long']."},
        {"name": "height_frac", "type": "float | list", "description": "Contact heights inside the usable band (0 = a pad radius above the table, 1 = a pad radius below the top edge: higher rides over the edge, lower puts the thumb in the table). Default [1.0, 0.6]."},
        {"name": "along_fracs", "type": "list[float]", "description": "Positions along the face, as fractions of its half-length from its centre. Default [0, -0.4, 0.4]."},
        {"name": "table_clear_m", "type": "float", "description": "Pad clearance above the table. Default 0.002."},
        {"name": "max_overlap_mm", "type": "float", "description": "Hand overlap with the object or table allowed in the static pose. Default 2.5."},
        {"name": "bench_scene, arm_waypoints, arm_seed", "type": "str | list", "description": "Screen every pose for the station's arm (see generate_rigid_grasps)."},
        {"name": "verify_top", "type": "int", "description": "Lift-test this many of the best poses (0: none), each with a short ladder of clamp depth / pre-opening until one lifts; the working pair is written into the pose. Default 3."},
        {"name": "open_frac, press_depth_rad", "type": "float", "description": "The pinch's pre-opening and clamp written into every pose (the executor uses them). Defaults 0.1 and 0.12: a flat part is pinched from a nearly closed hand with a light clamp."},
        {"name": "grip", "type": "str", "description": "'down' (fingers pointing down into a pinch across the part), 'pencil' (a part thinner than two pads, a pen or closed scissors: clamped at its equator near one end with the fingers lying along it, the wrist behind and above that end) or 'auto' (pencil candidates first when the part is thinner than two pads plus 6 mm). Default 'auto'. Measured limit: on a 13 mm pen the pencil pinches need the contact 40 deg above the equator to keep the thumb tip's back off the table, and they seat but slip after ~4 mm of lift; a pen this thin needs another strategy (a lever, an edge, a tripod from above)."},
        {"name": "time_budget_s", "type": "float | None", "description": "Wall-clock budget for the whole call: candidates not solved and lift tests not run when it runs out are skipped, and the poses found so far are returned with a note. Default 900; None for no limit."},
        {"name": "pair_margin_m", "type": "float | None", "description": "Hand contact pairs are built only for object geoms within the hand's span plus this margin of the grasped part (a scene of 75 ring walls otherwise pays for every pair at every step). Default 0.05; None pairs every object geom."},
        {"name": "out_json", "type": "str | None", "description": "Write the ranked list here."},
        {"name": "state", "type": "str | list | None", "description": "Current bench state for arm/collision screening: named trajectory.npz or proposal qpos. Defaults to the hand scene's recorded bench_state when available, otherwise the initial keyframe. Explicit object state must match the hand scene."},
    ],
    "outputs": [
        {"name": "result", "type": "dict", "description": "grasps (ranked pose dicts with pinch {axis, thumb_side, along, fingers}, hold {opposed, opposition_deg, need_n}, arm, lift), n_candidates, n_feasible, n_arm_feasible, footprint {axes, half_lengths_m, height_m}, progress {candidates_total, candidates_solved, lift_tests_run, lift_tests_planned, elapsed_s, stopped}, note (when the time budget ran out)."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["grasp", "pinch", "flat-object", "stage3"],
    "used_in": ["Computer Installation", "Place Toy in Drawer", "Scissor Cutting"],
    "status": "tested",
    "entrypoint": "generate_flat_pinches",
}

import json
from pathlib import Path

import numpy as np


def _hold_rank(g):
    """Sort key of a pinch by how hard its contacts must push to hold the part (hand.grip_hold_quality need_n): those
    within hand.HOLD_NEED_OK_N first, then in 2 N bins (a heavy part never gets under the bound, and among its pinches
    the one through the centre of mass still needs the least)."""
    from tool_library.common import hand as H
    n = (g.get("hold") or {}).get("need_n")
    if n is None:
        return (2, 0)
    return (0, 0) if n <= H.HOLD_NEED_OK_N else (1, int(n // 2.0))


def _footprint(model, data, gids, bid):
    """The object's horizontal axes (its body x/y projected on the table), half-lengths along them, centre and top."""
    import mujoco
    R = np.asarray(data.xmat[bid], float).reshape(3, 3)
    ax = []
    for k in range(3):
        v = R[:, k].copy(); v[2] = 0.0
        if np.linalg.norm(v) > 0.3:
            ax.append(v / np.linalg.norm(v))
    if len(ax) < 2:
        ax = [np.array([1.0, 0, 0]), np.array([0, 1.0, 0])]
    a1 = ax[0]; a2 = np.array([-a1[1], a1[0], 0.0])
    pts = []
    for g in gids:
        Rg = np.asarray(data.geom_xmat[g], float).reshape(3, 3); pg = np.asarray(data.geom_xpos[g], float)
        if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH):
            mid = int(model.geom_dataid[g])
            v = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]].astype(float)
            pts.append(v @ Rg.T + pg)
        else:
            c = np.asarray(model.geom_aabb[g, :3], float); h = np.asarray(model.geom_aabb[g, 3:], float)
            corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * h + c
            pts.append(corners @ Rg.T + pg)
    P = np.vstack(pts)
    c = P.mean(axis=0)
    s1, s2 = (P - c) @ a1, (P - c) @ a2
    cen = c + a1 * 0.5 * (s1.max() + s1.min()) + a2 * 0.5 * (s2.max() + s2.min())
    h1, h2 = 0.5 * (s1.max() - s1.min()), 0.5 * (s2.max() - s2.min())
    zlo, zhi = float(P[:, 2].min()), float(P[:, 2].max())
    if h1 > h2:                                   # short axis first
        a1, a2, h1, h2 = a2, a1, h2, h1
    return {"short": a1, "long": a2, "half": {"short": float(h1), "long": float(h2)}, "centre": cen, "zlo": zlo, "zhi": zhi}


def _ray_part(model, data, gids, origin, direction):
    """Distance along `direction` from `origin` to the part's own collision geoms (walls around it ignored), or None."""
    import mujoco
    best = None
    o = np.asarray(origin, float); v = np.asarray(direction, float)
    for g in gids:
        if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH):
            dist = mujoco.mj_rayMesh(model, data, int(g), o, v)
        else:
            dist = mujoco.mju_rayGeom(data.geom_xpos[g], data.geom_xmat[g], model.geom_size[g], o, v, int(model.geom_type[g]))
        if dist >= 0 and (best is None or dist < best):
            best = float(dist)
    return best


def _surface_contact(model, data, gids, point, out_dir, tangent):
    """The real surface point and outward normal of the part near `point`: a ray from outside along -out_dir (and two
    neighbouring rays for the normal). A rounded part's surface is not its bounding box's face."""
    out_dir = np.asarray(out_dir, float) / max(np.linalg.norm(out_dir), 1e-9)
    tangent = np.asarray(tangent, float) / max(np.linalg.norm(tangent), 1e-9)
    up = np.cross(out_dir, tangent); up /= max(np.linalg.norm(up), 1e-9)
    start = np.asarray(point, float) + out_dir * 0.05
    d0 = _ray_part(model, data, gids, start, -out_dir)
    if d0 is None:
        return None, None
    p0 = start - out_dir * d0
    pts = []
    for e in (tangent, up):
        di = _ray_part(model, data, gids, start + e * 0.002, -out_dir)
        if di is None:
            return p0, out_dir
        pts.append(start + e * 0.002 - out_dir * di)
    n = np.cross(pts[0] - p0, pts[1] - p0)
    if np.linalg.norm(n) < 1e-9:
        return p0, out_dir
    n = n / np.linalg.norm(n)
    if n @ out_dir < 0:
        n = -n
    return p0, n


def _sweep_overlap(model, data0, rig, ids, pose, open_frac, gids_part, back=(0.0, 0.03, 0.06, 0.09)):
    """Deepest overlap (m) of the hand with everything but the part and the floor while it arrives as the executor brings
    it: the pre-opened hand along its approach line, then half closed and closed at the pose. A part lying among walls
    (a tray, a drawer) can be pinched at the pose while the opening fingers or the arriving palm go through the walls."""
    import mujoco
    from tool_library.common import hand as H
    closed = {k: float(v) for k, v in (pose.get("finger_targets") or {}).items() if k in rig.finger_range}
    grip = list(pose.get("grip_fingers") or [])
    opened = H.open_targets(rig, closed, float(open_frac), curl=[f for f in H.FINGERS if grip and f not in grip])
    half = {k: 0.5 * (opened.get(k, v) + v) for k, v in closed.items()}
    part = set(int(g) for g in gids_part)
    env = [int(g) for g in ids["obj_gids"] if int(g) not in part]
    if not env:
        return 0.0
    d = mujoco.MjData(model); d.qpos[:] = data0.qpos
    H.apply_pose(model, d, rig, pose, opened)
    axis = H.approach_axis(model, d, rig, pose)
    pos = np.asarray(pose["pos"], float); eul = np.asarray(pose["euler"], float)
    states = [(pos + axis * b, opened) for b in back] + [(pos, half), (pos, closed)]
    worst = 0.0
    for p_, tg in states:
        H.apply_pose(model, d, rig, {"pos": p_.tolist(), "euler": eul.tolist()}, tg)
        P = np.asarray(d.geom_xpos, float); R = np.asarray(model.geom_rbound, float)
        for g in rig.collision_gids:
            for og in env:
                if float(np.linalg.norm(P[g] - P[og])) > float(R[g] + R[og]) + 1e-4:
                    continue
                dist = mujoco.mj_geomDistance(model, d, g, og, 0.0, None)
                worst = max(worst, -float(dist))
    return worst


def generate_flat_pinches(scene, fingers=(("index",), ("middle",), ("index", "middle")), axes=("short", "long"), height_frac=(1.0, 0.6),
                          along_fracs=(0.0, -0.4, 0.4), table_clear_m=0.002, max_overlap_mm=2.5, max_floor_mm=1.5,
                          max_span_m=0.115, bench_scene=None, arm_waypoints=None, arm_seed=None, verify_top=3, out_json=None,
                          iters=200, max_candidates=24, open_frac=0.1, press_depth_rad=0.12, grip="auto",
                          time_budget_s=900.0, pair_margin_m=0.05, state=None):
    import time
    import mujoco
    from tool_library.common import hand as H
    from tool_library.skills.pose_from_points import pose_from_points

    t0 = time.monotonic()
    over_budget = lambda: time_budget_s is not None and time.monotonic() - t0 > float(time_budget_s)
    spec = H.load_scene_spec(scene)
    bench_model, bench_q = None, None
    if bench_scene:
        from tool_library.common.episode import build_execution_model
        from tool_library.skills.hand_scene_from_recon import resolve_bench_state
        bench_model, _ = build_execution_model(str(bench_scene))
        bench_q = resolve_bench_state(spec, bench_model, state)
        if bench_q is None:
            bench_q = np.asarray(bench_model.key_qpos[0] if bench_model.nkey else bench_model.qpos0, float).copy()
    model, data0, rig, ids = H.open_scene(spec)
    bid = H.grasp_body(model, spec, ids)
    if pair_margin_m is not None:
        # one scene for every candidate, with hand pairs only for the geoms near the part: the pinches all touch it, and
        # nothing farther than the hand's span from it can meet the posed hand (rebuilt once, before any solve)
        P = []
        for g in sorted(ids["grasp_gids"]):
            c = np.asarray(data0.geom_xpos[g], float); r = float(model.geom_rbound[g])
            P += [c + r * np.array(v, float) for v in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))]
        if P:
            model, data0, rig, ids = H.open_scene(spec, pair_region={"points": np.asarray(P).tolist(), "margin_m": float(pair_margin_m)})
            bid = H.grasp_body(model, spec, ids)
    built = (model, data0, rig, ids)
    gids_part = sorted(ids["grasp_gids"])
    root = int(model.body_rootid[bid])
    com, mass = data0.subtree_com[root].copy(), float(model.body_subtreemass[root])
    fp = _footprint(model, data0, gids_part, bid)
    floor_z = 0.0
    if ids.get("floor_gid", -1) >= 0:
        floor_z = float(data0.geom_xpos[ids["floor_gid"]][2])
    pad_r = H.pad_radius(model, rig, "index")
    height = fp["zhi"] - fp["zlo"]
    # the pad centre must sit a pad radius below the top edge (or it rides over the edge and slips) and a pad radius
    # above the table: height_frac places the contact inside that band
    z_min = max(floor_z, fp["zlo"]) + pad_r + float(table_clear_m)
    z_max = fp["zhi"] - pad_r - 0.001
    if z_max < z_min:                                  # thinner than a pad: the middle of the face is all there is
        z_min = z_max = 0.5 * (fp["zlo"] + fp["zhi"])
    zs = sorted({round(z_min + float(hf) * (z_max - z_min), 4) for hf in np.atleast_1d(height_frac)}, reverse=True)
    # the robot stands at -y: seeds come from above and from above-behind (the hand pitched forward over the part)
    approach_dirs = [[0.0, 0.0, 1.0], [0.0, -0.6, 0.8]]
    cands, out, skipped = [], [], []
    for axis in axes:
        a = fp[axis]; other = fp["long" if axis == "short" else "short"]
        half, half_o = fp["half"][axis], fp["half"]["long" if axis == "short" else "short"]
        if 2.0 * half > float(max_span_m):
            skipped.append(f"{axis} axis {2000 * half:.0f} mm wider than the hand's pinch span"); continue
        for z, thumb_sign, along in [(z_, ts, al) for z_ in zs for al in along_fracs for ts in (1.0, -1.0)]:
            base = fp["centre"] + other * float(along) * half_o
            p_th = base + a * thumb_sign * half; p_th[2] = z
            p_fi = base - a * thumb_sign * half; p_fi[2] = z
            n_th, n_fi = a * thumb_sign, -a * thumb_sign
            # the part's real surface at those heights (a rounded part is not its bounding box)
            s_th, m_th = _surface_contact(model, data0, gids_part, p_th, n_th, other)
            s_fi, m_fi = _surface_contact(model, data0, gids_part, p_fi, n_fi, other)
            if s_th is None or s_fi is None:
                continue
            p_th, n_th, p_fi, n_fi = s_th, m_th, s_fi, m_fi
            for fs in fingers:
                pts = {"thumb": p_th.tolist()}; nrm = {"thumb": n_th.tolist()}
                spread = [0.0] if len(fs) == 1 else np.linspace(-0.009 * (len(fs) - 1), 0.009 * (len(fs) - 1), len(fs))
                for f, s_ in zip(fs, spread):
                    q = p_fi + other * float(s_)
                    pts[f] = q.tolist(); nrm[f] = n_fi.tolist()
                cands.append({"axis": axis, "thumb_side": "+" if thumb_sign > 0 else "-", "along": float(along), "fingers": list(fs),
                              "points": pts, "normals": nrm, "z": float(z)})
    # a part thinner than two pads (a pen, a marker, a closed scissors) cannot be pinched with the fingers pointing down:
    # their tips reach the table first. The PENCIL grip lays it along the fingers: thumb and index pads clamp it at its
    # equator near one end, the fingers pointing along its length, the wrist behind and above that end.
    pencil = grip == "pencil" or (grip == "auto" and height < 2.0 * pad_r + 0.006)
    if pencil:
        cands = [c for c in cands if grip != "pencil"]
        lng, sht = fp["long"], fp["short"]
        zc = 0.5 * (fp["zlo"] + fp["zhi"])
        rad = 0.5 * height                              # a thin part treated as round across its width
        up = np.array([0.0, 0.0, 1.0])
        # the contact rises just far enough above the equator that the pad (its centre one pad radius plus 1 mm out
        # along the normal) clears the table: ~10 degrees on a 13 mm pen, well inside the friction cone
        sphi0 = float(np.clip((max(floor_z, fp["zlo"]) + pad_r + float(table_clear_m) - zc) / (rad + pad_r + 0.001), 0.0, 0.85))
        # higher contacts lift the thumb tip's back off the table (measured 8.4 / 5.1 / 2.2 mm into it at 10 / 24 / 37 deg
        # on a 13 mm pen) while the friction cone (pads mu 1.0) still holds the part up to ~45 deg
        for sphi in sorted({round(sphi0, 3), round(np.sin(np.radians(25.0)), 3), round(np.sin(np.radians(40.0)), 3)}):
            cphi = float(np.sqrt(1.0 - sphi ** 2))
            for end in (-1.0, 1.0):
                for along in (0.75,):
                    base = fp["centre"] + lng * end * along * fp["half"]["long"]; base[2] = zc
                    for thumb_sign in (1.0, -1.0):
                        n_th = cphi * sht * thumb_sign + sphi * up; n_fi = cphi * (-sht * thumb_sign) + sphi * up
                        p_th = base + rad * n_th; p_fi = base + rad * n_fi
                        s_th, m_th = _surface_contact(model, data0, gids_part, p_th, n_th, lng)
                        s_fi, m_fi = _surface_contact(model, data0, gids_part, p_fi, n_fi, lng)
                        if s_th is not None and s_fi is not None:
                            p_th, n_th, p_fi, n_fi = s_th, m_th, s_fi, m_fi
                        cands.insert(0, {"axis": "short", "thumb_side": "+" if thumb_sign > 0 else "-", "along": float(end * along), "fingers": ["index"],
                                         "points": {"thumb": p_th.tolist(), "index": p_fi.tolist()},
                                         "normals": {"thumb": n_th.tolist(), "index": n_fi.tolist()},
                                         "z": float(zc), "grip": "pencil", "contact_deg": round(float(np.degrees(np.arcsin(sphi))), 1),
                                         # the wrist starts behind the clamped end and above it: fingers along the part
                                         "approach_dirs": [(lng * end * 0.8 + np.array([0.0, 0.0, 0.6])).tolist(), (lng * end * 0.5 + np.array([0.0, 0.0, 0.87])).tolist()]})
    rejected = {"solve": 0, "residual": 0, "overlap": 0, "table": 0, "approach": 0}
    progress = {"candidates_total": min(len(cands), int(max_candidates)), "candidates_solved": 0, "lift_tests_run": 0,
                "lift_tests_planned": 0, "elapsed_s": 0.0, "stopped": None}
    for c in cands[:int(max_candidates)]:
        if over_budget():
            progress["stopped"] = "time_budget"; break
        progress["candidates_solved"] += 1
        try:
            remaining = None if time_budget_s is None else max(1.0, float(time_budget_s) - (time.monotonic() - t0))
            r = pose_from_points(spec, c["points"], c["normals"], approach_dirs=c.get("approach_dirs", approach_dirs), iters=iters,
                                 pitches_deg=(-45, -15, 15, 45), yaw_step_deg=60, clear_overlap=False, time_budget_s=remaining,
                                 _built=built)
        except Exception:                             # an infeasible candidate, not a crash of the generator
            rejected["solve"] += 1
            continue
        pose = r.get("pose", r)
        if pose.get("worst_residual_mm", 99.0) > 4.0:
            rejected["residual"] += 1; continue
        if pose.get("overlap_mm", 99.0) > float(max_overlap_mm):
            rejected["overlap"] += 1; continue
        # no finger link in the table: the static pose against the floor
        d = mujoco.MjData(model); d.qpos[:] = data0.qpos
        H.apply_pose(model, d, rig, pose)
        cl = H.clearance(model, d, rig, ids, ignore_tips=())
        if cl["floor_mm"] > float(max_floor_mm):
            rejected["table"] += 1; continue
        # the opening hand and its approach must clear the walls around the part (a tray, a drawer, a neighbour)
        sweep = _sweep_overlap(model, data0, rig, ids, pose, open_frac, gids_part)
        if sweep * 1000.0 > float(max_overlap_mm):
            rejected["approach"] += 1; continue
        pose = dict(pose)
        pose["pinch"] = {k: c.get(k) for k in ("axis", "thumb_side", "along", "fingers", "z", "grip", "contact_deg")}
        pose["family"] = "pencil_pinch" if c.get("grip") == "pencil" else f"flat_pinch_{c['axis']}"
        pose.setdefault("grip_fingers", ["thumb"] + list(c["fingers"]))
        pose.setdefault("approach", [0.0, 0.0, 1.0])
        # a flat part is pinched from a nearly closed hand with a light clamp: opened 55% the fingers drop beside the
        # part and below its edge, and a deep clamp rolls the thumb pad off its face (measured on a 120 g box and a
        # 600 g drive). The executor and the stress tests read these from the pose.
        pose["open_frac"] = float(open_frac)
        pose["press_depth_rad"] = float(press_depth_rad)
        # how the planned contacts hold the part's weight with realistic (2 cm) twisting friction: a pinch line far from
        # the centre of mass holds the gravity torque about it only by twisting the pads (a 600 g drive pinched 30-45 mm
        # from its centre twisted out of every executed pinch); hand.grip_hold_quality on the ray-cast surface contacts
        cons = [{"link": f, "point": np.asarray(c["points"][f], float), "normal": -np.asarray(c["normals"][f], float)} for f in c["points"]]
        hq = H.grip_hold_quality(cons, com, mass)
        pose["hold"] = {k: hq[k] for k in ("opposed", "opposition_deg", "need_n")}
        out.append(pose)
    # pinches that hold the weight with margin first (hand.HOLD_NEED_OK_N per contact at 1.5 g, tilted 30 deg), then the
    # static fit
    out.sort(key=lambda g: (_hold_rank(g),
                            g.get("overlap_mm", 0.0), g.get("worst_residual_mm", 0.0)))
    n_arm = None
    if bench_scene and out:
        from tool_library.common import armhand as AH
        probe = AH.ArmProbe(str(bench_scene))
        scr = AH.screen_poses(bench_model, spec.get("side", "right"), out, bench_q, probe=probe, arm_seed=arm_seed, waypoints=arm_waypoints)
        for g, s_ in zip(out, scr):
            g["arm"] = {k: s_.get(k) for k in ("ok", "residual_mm", "residual_deg", "reason", "duplicate_of")}
        out.sort(key=lambda g: (0 if g["arm"]["ok"] else (2 if g["arm"].get("duplicate_of") is not None else 1),
                                _hold_rank(g), g.get("overlap_mm", 0.0)))
        n_arm = sum(1 for g in out if g["arm"]["ok"])
    progress["lift_tests_planned"] = min(len(out), int(verify_top))
    for g in out[:int(verify_top)]:
        if over_budget():
            progress["stopped"] = "time_budget"; break
        progress["lift_tests_run"] += 1
        tried = []
        # the bench holds the hand where it is commanded (the hand scene carries the hand's weight too): pads placed
        # exactly on the surface can close without wedging the part, so the ladder ends with the wrist seated 2 mm
        # deeper along its approach (visible overlap with the part and the table capped at 4 mm)
        dq = mujoco.MjData(model); dq.qpos[:] = data0.qpos; H.apply_pose(model, dq, rig, g)
        ax = H.approach_axis(model, dq, rig, g)
        for pd, of, seat in ((press_depth_rad, open_frac, 0.0), (0.06, open_frac, 0.0), (0.25, open_frac, 0.0), (press_depth_rad, 0.2, 0.0),
                             (press_depth_rad, open_frac, 0.002), (0.25, open_frac, 0.002)):
            if tried and over_budget():
                progress["stopped"] = "time_budget"; break
            gg = dict(g, press_depth_rad=float(pd), open_frac=float(of))
            if seat:
                gg["pos"] = (np.asarray(g["pos"], float) - ax * seat).tolist()
                dq.qpos[:] = data0.qpos; H.apply_pose(model, dq, rig, gg); cl = H.clearance(model, dq, rig, ids, ignore_tips=())
                if max(cl.get("visual_object_mm", cl["object_mm"]), cl.get("visual_floor_mm", cl["floor_mm"])) > 4.0:
                    continue
            r = H.stress_rollout(model, data0, rig, ids, gg, bid, phases=("approach", "close", "lift"), lift_heights=(0.04, 0.08))
            tried.append({"press_depth_rad": pd, "open_frac": of, "seat_mm": 1000 * seat, "ok": r["ok"], "first_fail": r["first_fail"],
                          "achieved_mm": r["lifts"][-1]["achieved_mm"] if r["lifts"] else 0.0})
            if r["ok"]:
                g["press_depth_rad"], g["open_frac"] = float(pd), float(of)
                if seat:
                    g["pos"] = gg["pos"]; g["seat_m"] = float(seat)
                break
        if not tried:
            continue
        best = next((t for t in tried if t["ok"]), tried[0])
        g["lift"] = {"ok": best["ok"], "first_fail": best["first_fail"], "achieved_mm": best["achieved_mm"], "tried": tried}
    if verify_top:
        out.sort(key=lambda g: (0 if g.get("arm", {}).get("ok", True) else 1, 0 if g.get("lift", {}).get("ok") else 1,
                                _hold_rank(g), g.get("overlap_mm", 0.0)))
    res = {"grasps": out, "n_candidates": min(len(cands), int(max_candidates)), "n_feasible": len(out), "n_arm_feasible": n_arm,
           "rejected": rejected, "skipped_axes": skipped, "progress": progress,
           "footprint": {"short_axis": fp["short"].tolist(), "long_axis": fp["long"].tolist(), "half_lengths_m": fp["half"],
                         "height_m": round(height, 4), "contact_z": zs}}
    progress["elapsed_s"] = round(time.monotonic() - t0, 1)
    if progress["stopped"]:
        res["note"] = (f"time budget of {float(time_budget_s):.0f} s ran out after {progress['candidates_solved']} of "
                       f"{progress['candidates_total']} candidates and {progress['lift_tests_run']} of {progress['lift_tests_planned']} lift tests: "
                       "the grasps are the best found so far (lift-untested ones carry no 'lift'); give a larger time_budget_s, fewer "
                       "fingers/axes/height_frac/along_fracs, or a smaller max_candidates / verify_top")
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(out, indent=1, default=float))
        res["out_json"] = str(out_json)
    return res
