TOOL = {
    "name": "generate_rigid_grasps",
    "title": "Generate grasp candidates for a rigid object",
    "type": "skill",
    "category": "Grasp & Poses",
    "summary": "Enumerate and rank grasp poses for a rigid object: pocket wraps from the side and above plus fingertip pinches across its bounding box, each with open, cinched and squeezed finger targets.",
    "description": "Measures where the hand holds things (the pocket between the palm face and the curling fingers, from the hand's own kinematics) and the object's extent along its own axes, then places the pocket on the object: side wraps around the object's up axis at a chosen height fraction (palm facing the axis, fingers across it or pointing down it, every yaw step), and top grasps (palm against its top, fingers along several yaws). The up axis is the vertical for an upright part; for a part leaning 3-45 deg (a pipette in a rack) it is the part's own axis (its elongated principal axis, else its body axis nearest the vertical): wraps about the vertical forced a leaning pipette upright and pushed its rack. For objects narrower than 9 cm it also solves thumb-versus-index/middle pinches across each horizontal axis of the bounding box with the fingertip IK solver. Every candidate is screened statically: the OPEN hand must be able to retreat along its own approach axis to a pose that clears the scene by 4 mm, and the cinched hand must touch the object with at least two pads. The cinched hand's contacts are read from the geometry and judged (hold: opposed when two contact groups push at least 100 deg apart with the centre of mass between them; need_n, the force each contact must push to hold the part's weight at 1.5 g tilted 30 deg with 2 cm twisting friction). Survivors are ranked opposed first (with realistic contact a grip whose pads all press one face, the palm on another, holds only by the palm's friction while the wrist pushes: a bar held that way shot out of the hand when the arm pulled it), then by number of touching pads, shallower interpenetration and shorter retreat. With verify=True the top candidates are additionally driven through grasp_lift_test and the ones that lift are returned first. The object is never pre-fitted to a width model: the force-controlled close does the fitting at run time.",
    "inputs": [
        {"name": "scene", "type": "dict", "description": "{hand_xml, side, objects, floor}; the last object is grasped"},
        {"name": "height_fracs", "type": "list of float", "description": "object-height fractions for side grasps (default [0.5, 0.65])"},
        {"name": "yaw_step_deg", "type": "float", "description": "azimuth step around the object for side and top grasps (default 45)"},
        {"name": "finger_dirs", "type": "list", "description": "finger-axis options for side grasps: 'horizontal', 'down' (default both)"},
        {"name": "include_top", "type": "bool", "description": "also generate palm-down grasps (default True)"},
        {"name": "thumb_oppose_fracs", "type": "list of float", "description": "thumb opposition settings to try, as fractions of the opposition joints' range (default 0, 0.4, 0.8)"},
        {"name": "min_fingers", "type": "int", "description": "cinched pads that must touch (default 2)"},
        {"name": "include_pinch", "type": "bool", "description": "also solve thumb-vs-index/middle pinches across the bounding box for objects narrower than pinch_max_width_m (default True)"},
        {"name": "pinch_max_width_m / pinch_height_frac / pinch_ik_iters", "type": "numbers", "description": "pinch family limits: widest object to pinch (0.09), height fraction of the pinch, or a list of them (0.6; a pinch low on a tall part hangs the curled free fingers into its top on the way down, so every pinch is screened for the open hand's descent along +z and [0.6, 0.85] tries one nearer the top too), IK iterations per seed (240)"},
        {"name": "verify", "type": "bool", "description": "run grasp_lift_test on the top `verify_top` candidates (default False)"},
        {"name": "open_frac", "type": "float", "description": "Finger flexion fraction of the OPEN hand during seating and approach (default 0.0, fingers straight, so the palm can come up to a fat object and the fingers wrap it; 0.15 keeps them slightly curled)."},
        {"name": "verify_top", "type": "int", "description": "how many candidates to verify (default 4)"},
        {"name": "bench_scene", "type": "str | None", "description": "The Stage 2 bench scene.xml: every grasp is screened for the station's ARM (exact IK of the flange at the pre-grasp and the grasp, arm links against the table and the objects) and annotated arm {ok, residual_mm, residual_deg, reason}; arm-feasible grasps rank first. Strongly recommended: a grasp the arm cannot hold is not a grasp of this robot."},
        {"name": "arm_seed", "type": "list | None", "description": "Arm joints the screen starts from (default the scene's home)."},
        {"name": "arm_waypoints", "type": "list | None", "description": "Reach screening along the intended manipulation: legacy world wrist offsets [dx,dy,dz], or {object: free_root_body_name, target_pos: [x,y,z], target_euler_deg: [rx,ry,rz], steps: 8}. Object targets use intrinsic XYZ degrees and absolute WORLD body-frame poses (not object-bottom coordinates); omitted rotation preserves the previous orientation. The hand follows its grasp transform while the object moves, including intermediate rotations. These hypothetical IK states are never executed object motion; the full physical executor remains required."},
        {"name": "out_json", "type": "str | None", "description": "Write the ranked grasp list here (a JSON list of poses, as generate_flat_pinches writes it). Default none."},
        {"name": "keep_clear", "type": "list | dict | None", "description": "zones no part of the hand may enter, {center, radius} spheres or {min, max} boxes in world metres: the room the other hand needs (the cap above a bottle this hand holds while the other unscrews it). Default none"},
        {"name": "state", "type": "str | list | None", "description": "Current bench state for arm/collision screening: named trajectory.npz or proposal qpos. Defaults to the hand scene's recorded bench_state when available, otherwise the initial keyframe. Explicit object state must match the hand scene."},
    ],
    "outputs": [
        {"name": "grasps", "type": "list of pose dicts", "description": "ranked poses: {side, pos, euler, finger_targets (cinched), finger_targets_open, finger_targets_squeezed, grip_fingers, family, approach, retract_m, overlap_mm, n_touching, hold {opposed, opposition_deg, pair, need_n (N per contact), links}, lift (if verified), arm (with bench_scene)}"},
        {"name": "n_candidates / n_feasible", "type": "int", "description": "how many were tried and how many survived the static screen"},
        {"name": "object_axis", "type": "dict", "description": "{axis, tilt_deg}: the up axis the wraps were built about and the part's tilt from the vertical; out_json when written"},
    ],
    "status": "tested",
    "entrypoint": "generate_rigid_grasps",
    "requirements": ["mujoco>=3.1", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["grasp", "candidate-generation", "rigid-object"],
    "used_in": ["Place Cup in Bowl", "Giftbox Packing", "Battery Insertion", "Open the Bottle"],
}

import numpy as np


def _in_zone(model, d, rig, zones):
    """Any collision geom of the hand inside one of the zones: {center, radius} spheres or {min, max} boxes (world)."""
    import numpy as _np
    zones = zones if isinstance(zones, list) else [zones]
    for g in rig.collision_gids:
        c = _np.asarray(d.geom_xpos[g], float); r = float(model.geom_rbound[g])
        for z in zones:
            if "radius" in z:
                if _np.linalg.norm(c - _np.asarray(z["center"], float)) < float(z["radius"]) + 0.5 * r:
                    return True
            else:
                lo, hi = _np.asarray(z["min"], float), _np.asarray(z["max"], float)
                if _np.all(c > lo - 0.5 * r) and _np.all(c < hi + 0.5 * r):
                    return True
    return False


def _hold(model, d, rig, obj, com, mass, fingers):
    """The closed hand's contacts with the gripping fingers and the palm, read from the geometry (hand.static_grip), and
    how they hold the part (opposition, the per-pad force its weight needs): hand.grip_hold_quality."""
    from tool_library.common import hand as H
    q = H.grip_hold_quality(H.static_grip(model, d, rig, obj, band_m=0.002, fingers=fingers), com, mass)
    return {k: q[k] for k in ("opposed", "opposition_deg", "pair", "need_n", "links", "n_contacts")}


def _geom_points(model, data, gids):
    """World points of a geom set: mesh vertices, else the corners of each geom's own bounding box."""
    import mujoco
    pts = []
    for g in gids:
        R = np.asarray(data.geom_xmat[g], float).reshape(3, 3); p = np.asarray(data.geom_xpos[g], float)
        if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH) and int(model.geom_dataid[g]) >= 0:
            mid = int(model.geom_dataid[g])
            V = np.asarray(model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]], float)
        else:
            c = np.asarray(model.geom_aabb[g, :3], float); h = np.asarray(model.geom_aabb[g, 3:], float)
            V = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * h + c
        pts.append(V @ R.T + p)
    return np.vstack(pts) if pts else np.zeros((0, 3))


AXIS_MIN_TILT_DEG = 3.0     # an object axis closer than this to the vertical is taken as vertical (the wraps as before)
AXIS_MAX_TILT_DEG = 45.0    # past this the part is lying down: wraps go about the vertical through its cross-section


def object_axis(model, data, gids, bid):
    """The object's own up axis (unit, pointing up) and its tilt from the vertical in degrees: its elongated principal axis
    when it has one (largest point variance at least twice the next: a pipette, a bottle), else the axis of the grasped
    body's frame nearest the vertical (a tilted cup). World +z when that is within AXIS_MIN_TILT_DEG of the vertical or
    more than AXIS_MAX_TILT_DEG from it."""
    z = np.array([0.0, 0.0, 1.0])
    P = _geom_points(model, data, gids)
    u = None
    if len(P) >= 4:
        w, V = np.linalg.eigh(np.cov((P - P.mean(axis=0)).T))
        if w[-1] >= 2.0 * max(w[-2], 1e-12):
            u = V[:, -1]
    if u is None:
        R = np.asarray(data.xmat[bid], float).reshape(3, 3)
        u = R[:, int(np.argmax(np.abs(R[2, :])))]
    u = u / max(np.linalg.norm(u), 1e-12)
    if u[2] < 0:
        u = -u
    tilt = float(np.degrees(np.arccos(np.clip(u[2], -1.0, 1.0))))
    if tilt < AXIS_MIN_TILT_DEG or tilt > AXIS_MAX_TILT_DEG:
        return z, tilt
    return u, tilt


def generate_rigid_grasps(scene, height_fracs=(0.5, 0.65), yaw_step_deg=45.0,
                          finger_dirs=("horizontal", "down"), include_top=True,
                          thumb_oppose_fracs=(0.0, 0.4, 0.8), min_fingers=2,
                          include_pinch=True, pinch_max_width_m=0.09, pinch_height_frac=0.6,
                          pinch_ik_iters=240, verify=False, verify_top=4, open_frac=0.0, keep_clear=None,
                          bench_scene=None, arm_seed=None, arm_waypoints=None, out_json=None, state=None):
    import mujoco
    from tool_library.common import hand as H

    bench_model, bench_q = None, None
    if bench_scene:
        from tool_library.common.episode import build_execution_model
        from tool_library.skills.hand_scene_from_recon import resolve_bench_state
        bench_model, _ = build_execution_model(str(bench_scene))
        bench_q = resolve_bench_state(H.load_scene_spec(scene), bench_model, state)
        if bench_q is None:
            bench_q = np.asarray(bench_model.key_qpos[0] if bench_model.nkey else bench_model.qpos0, float).copy()
    model, data0, rig, ids = H.open_scene(scene)
    bid0 = H.grasp_body(model, scene, ids)
    root0 = int(model.body_rootid[bid0])
    com, mass = data0.subtree_com[root0].copy(), float(model.body_subtreemass[root0])
    pocket = H.grasp_pocket(model, data0, rig)
    ext = H.object_extent(model, data0, ids["grasp_gids"])
    c, lo, hi = ext["centre"], ext["lo"], ext["hi"]
    # the object's own frame: e1, e2 across it, u along its up axis. Upright, that is the world (the bounding box gives the
    # centre and the extents, as always); a part leaning in a rack (a pipette at 14 deg) is wrapped about its own axis: wraps
    # about the vertical forced it upright and pushed the rack
    u, tilt = object_axis(model, data0, sorted(ids["grasp_gids"]), bid0)
    if u[2] >= 1.0 - 1e-12:
        E = np.eye(3)
        half = 0.5 * (hi - lo)
    else:
        e1 = np.array([1.0, 0.0, 0.0]) - u * u[0]
        if np.linalg.norm(e1) < 0.3:
            e1 = np.array([0.0, 1.0, 0.0]) - u * u[1]
        e1 /= np.linalg.norm(e1)
        E = np.column_stack([e1, np.cross(u, e1), u])
        Q = (_geom_points(model, data0, sorted(ids["grasp_gids"])) - ext["centre"]) @ E
        qlo, qhi = Q.min(axis=0), Q.max(axis=0)
        c = ext["centre"] + E @ (0.5 * (qlo + qhi))
        half = 0.5 * (qhi - qlo)
        lo, hi = c - E @ half, c + E @ half          # reported extent: the oriented box's corners

    def along(fr):                                  # the point on the axis at fraction fr of the object's height
        return c + u * ((2.0 * float(fr) - 1.0) * half[2])

    cands = []
    for yaw in np.arange(0.0, 360.0, float(yaw_step_deg)):
        a = E[:, 0] * np.cos(np.radians(yaw)) + E[:, 1] * np.sin(np.radians(yaw))   # from axis outward
        for hf in height_fracs:
            pt = along(hf)
            for fd in finger_dirs:
                if fd == "horizontal":
                    # fingers across the axis, thumb toward its top: the sense depends on the hand (a left hand is the
                    # mirror of a right one); the other sense is tried too and the screen drops the
                    # one whose thumb is in the table
                    s_h = 1.0 if rig.side == "right" else -1.0
                    f_ws = [s_h * np.cross(u, a), -s_h * np.cross(u, a)]
                elif fd == "down":
                    f_ws = [-u]
                else:
                    raise ValueError(f"finger_dirs entry '{fd}' unknown (horizontal|down)")
                for f_w in f_ws:
                    pos, eul = H.pocket_pose(pocket, pt, -a, f_w)
                    cands.append({"family": f"side_{fd}", "pos": pos, "euler": eul, "approach": a.copy()})
        if include_top:
            f_w = a.copy()
            pt = along(1.0 - 0.35)
            pos, eul = H.pocket_pose(pocket, pt, -u, f_w)
            cands.append({"family": "top", "pos": pos, "euler": eul, "approach": u.copy()})

    obj = sorted(ids["grasp_gids"])
    out = []
    d = mujoco.MjData(model)

    # PINCH family: small objects are not held in the pocket but between the
    # thumb pad and the finger pads on opposite faces. Points come from the
    # bounding box (thumb on one face centre, index and middle on the
    # opposite face, a little apart along the third axis) and the pose is
    # solved by fingertip IK, the same solver pose_from_points uses.
    if include_pinch:
        # two-pad pinches first (thumb against index: balanced forces by symmetry once the servos saturate),
        # then thumb against index and middle for wider or heavier parts
        families = [["thumb", "index"]] + ([["thumb", "index", "middle"]] if "middle" in rig.fingertip_sites else [])
        hfs = [float(v) for v in np.atleast_1d(pinch_height_frac)]
        for pinch_fingers, phf in [(pf_, h_) for h_ in hfs for pf_ in families]:
            for ax in (0, 1):
                if 2.0 * half[ax] > pinch_max_width_m:
                    continue
                other = 1 - ax
                for sgn in (+1.0, -1.0):
                    n_t = E[:, ax] * sgn
                    hz = along(phf)
                    face_t = hz + n_t * half[ax]
                    face_f = hz - n_t * half[ax]
                    pts = {"thumb": face_t.copy(), "index": face_f.copy()}
                    nrm = {"thumb": n_t, "index": -n_t}
                    if "middle" in pinch_fingers:
                        off = min(0.012, 0.4 * 2.0 * half[other])
                        pts["index"] = pts["index"] + E[:, other] * off
                        pts["middle"] = face_f - E[:, other] * off
                        nrm["middle"] = -n_t
                    # the tip site sits on the pad's axis: a pad touches when the site is one pad radius out
                    standoff = H.pad_radius(model, rig, "index") + 0.001
                    targets = {f: pts[f] + nrm[f] * standoff for f in pts}
                    cen = np.mean(list(targets.values()), axis=0)
                    seeds = H.seed_sweep(cen, [u.tolist(), -n_t], yaw_step_deg=60, pitches_deg=(-60, 0, 60))
                    try:
                        worst, res, pen, qpos, face = H.solve_fingertip_ik(
                            model, data0, rig, ids, targets, seeds, face_at=c, iters=pinch_ik_iters)
                    except (RuntimeError, ValueError):
                        continue
                    if worst > 0.006 or pen > 0.006:
                        continue
                    d.qpos[:] = qpos
                    H.curl_free_fingers_clear(model, d, rig, ids, set(pts))
                    mujoco.mj_forward(model, d)
                    # The IK standoff is a contact pose, not a pre-opening: its
                    # millimetre gap disappears under ordinary calibration error.
                    # Back active flexion off before approaching, keeping idle
                    # fingers self-clear; the swept checks below validate this
                    # actual pre-opening against the scene.
                    contact_t = {k: float(d.qpos[a]) for k, a in rig.finger_qpos.items()}
                    open_t = H.open_targets(rig, contact_t, .25, curl=set(H.FINGERS) - set(pts))
                    base = H.pose_from_data(model, d, rig, rig.side)
                    # the executor brings the open hand in along the approach (the object's up axis) from 8 cm: a pinch whose
                    # curled free fingers hang into the part on the way down is no grasp (measured: a ring finger 10 mm into a bar)
                    dd = mujoco.MjData(model)
                    blocked = False
                    for back in (0.08, 0.05, 0.03, 0.015):
                        dd.qpos[:] = data0.qpos
                        H.apply_pose(model, dd, rig, {"pos": np.asarray(base["pos"], float) + u * back, "euler": base["euler"]}, open_t)
                        if H.deepest_overlap(model, dd, rig, ids, keep_tips=()) > 0.002:
                            blocked = True
                            break
                    if blocked:
                        continue
                    avoid = [g_ for g_ in ids["obj_gids"] if g_ not in ids["grasp_gids"]] + ([ids["floor_gid"]] if ids["floor_gid"] >= 0 else [])
                    rep = H.cinch_fingers(model, d, rig, obj, fingers=list(pts), avoid_gids=avoid)
                    touching = [f for f, v in rep.items() if v["gap_mm"] <= 0.5]
                    if len(touching) < 2 or "thumb" not in touching:
                        continue
                    closed = {k: float(d.qpos[a]) for k, a in rig.finger_qpos.items()}
                    # squeeze: the pads of the pinch move 2 mm into the faces with the wrist fixed (a pinch
                    # presses by closing the pads toward each other, which a flexion cinch cannot do)
                    H.finger_ik(model, d, rig, {f: pts[f] + nrm[f] * (standoff - 0.003) for f in pts})
                    squeezed = {k: float(d.qpos[a]) for k, a in rig.finger_qpos.items()}
                    d.qpos[:] = qpos
                    for k_, v_ in closed.items():
                        d.qpos[rig.finger_qpos[k_]] = v_
                    mujoco.mj_forward(model, d)
                    hold = _hold(model, d, rig, obj, com, mass, touching)
                    g = dict(base)
                    g.update({"finger_targets": closed, "finger_targets_open": open_t,
                              "finger_targets_squeezed": squeezed, "grip_fingers": touching, "hold": hold,
                              "family": "pinch" if len(pinch_fingers) == 2 else "pinch3", "approach": [float(v) for v in u], "retract_m": None,
                              "press_depth_rad": 0.12,
                              "seat_offset_m": 0.0, "overlap_mm": round(pen * 1000, 2),
                              "n_touching": len(touching), "cinch": rep,
                              "residual_mm": {f: round(v * 1000, 2) for f, v in res.items()}})
                    out.append(g)
    # the thumb's opposition is not part of the closing synergy; try it at
    # several settings and keep the one that lets the thumb reach the object
    oppose_keys = [k for k in rig.finger_range if H.joint_role(k) == "thumb_oppose"]
    expanded = []
    for cd in cands:
        for frac in thumb_oppose_fracs:
            open_t = rig.finger_targets(float(open_frac))
            for k in oppose_keys:
                jlo, jhi = rig.finger_range[k]
                open_t[k] = jlo + float(frac) * (jhi - jlo)
            expanded.append((dict(cd, pos=cd["pos"].copy()), open_t, frac))
    for cd, open_t, frac in expanded:
        # SEAT: the pocket centre is where a 55%-closed hand holds a thin
        # object; a fat one must sit further out. Back the open hand off
        # along its approach until nothing but the pads overlaps the object
        seated = None
        for k in range(0, 26):
            s = 0.002 * k
            d.qpos[:] = data0.qpos
            H.apply_pose(model, d, rig, {"pos": cd["pos"] + cd["approach"] * s, "euler": cd["euler"]}, open_t)
            if H.deepest_overlap(model, d, rig, ids, keep_tips=H.FINGERS) <= 0.002:
                seated = s
                break
        if seated is None:
            continue
        cd["pos"] = cd["pos"] + cd["approach"] * seated
        base = {"side": rig.side, "pos": [float(v) for v in cd["pos"]],
                "euler": [float(v) for v in cd["euler"]], "finger_targets": open_t}
        # can the open hand come in along its approach without hitting anything?
        retract = None
        for k in range(0, 16):
            back = 0.012 * k
            d.qpos[:] = data0.qpos
            H.apply_pose(model, d, rig, {"pos": cd["pos"] + cd["approach"] * back,
                                         "euler": cd["euler"]}, open_t)
            clr = H.clearance(model, d, rig, ids)
            # clear of the objects; the hand may brush the table by a few millimetres (a low grasp on a
            # short object rests the edge of the palm on the table, as a human's does)
            if clr["object_mm"] <= 0.0 and clr["floor_mm"] <= 3.0:
                # require a 4 mm true gap to the objects as well
                gap = min(H.hand_geom_distance(model, d, g, og, 0.05)
                          for g in rig.collision_gids
                          for og in obj)
                if gap * 1000.0 >= 4.0:
                    retract = back
                    break
        if retract is None:
            continue
        d.qpos[:] = data0.qpos
        H.apply_pose(model, d, rig, base, open_t)
        overlap = H.deepest_overlap(model, d, rig, ids, keep_tips=())
        avoid = [g_ for g_ in ids["obj_gids"] if g_ not in ids["grasp_gids"]] + ([ids["floor_gid"]] if ids["floor_gid"] >= 0 else [])
        rep = H.cinch_fingers(model, d, rig, obj, avoid_gids=avoid)
        touching = [f for f, v in rep.items() if v["gap_mm"] <= 0.5]
        if len(touching) < int(min_fingers):
            continue
        # the closed hand must not sit in the floor or in a bystander either
        if H.deepest_overlap(model, d, rig, ids, keep_tips=()) > 0.008:
            continue
        # nor inside a zone kept free for the other hand (the cap above a held bottle, a handle the other hand pulls)
        if keep_clear and _in_zone(model, d, rig, keep_clear):
            continue
        closed = {k: float(d.qpos[a]) for k, a in rig.finger_qpos.items()}
        hold = _hold(model, d, rig, obj, com, mass, touching)
        d2 = mujoco.MjData(model)
        d2.qpos[:] = data0.qpos
        H.apply_pose(model, d2, rig, base, open_t)
        H.cinch_fingers(model, d2, rig, obj, touch_m=-0.0025, max_pen_m=0.0035, avoid_gids=avoid)
        squeezed = {k: float(d2.qpos[a]) for k, a in rig.finger_qpos.items()}
        g = dict(base)
        g.update({"finger_targets": closed, "finger_targets_open": dict(open_t),
                  "finger_targets_squeezed": squeezed, "grip_fingers": touching,
                  "family": cd["family"], "approach": [float(v) for v in cd["approach"]],
                  "thumb_oppose_frac": float(frac), "hold": hold,
                  "retract_m": round(retract, 3), "seat_offset_m": round(seated, 3),
                  "overlap_mm": round(overlap * 1000, 2),
                  "n_touching": len(touching), "cinch": rep})
        out.append(g)
    # opposed grips first: with realistic contact (2 cm twisting patches) the pads must squeeze across the part; five pads
    # on one face with the palm on another hold only by the palm's friction while the wrist pushes (a bar held that way
    # shot out of the hand the moment the arm pulled it), then the number of pads and the static fit
    out.sort(key=lambda g: (0 if g["hold"]["opposed"] else 1, -g["n_touching"], g["overlap_mm"], g["retract_m"] or 0.0))
    if bench_scene and out:
        # hand and arm together: a grasp the station's arm cannot hold exactly, or holds only by running its links
        # into the table or the objects, is not a grasp of this robot. Feasible ones rank first.
        from tool_library.common import armhand as AH
        probe = AH.ArmProbe(str(bench_scene))
        side_ = H.load_scene_spec(scene).get("side", "right")
        scr = AH.screen_poses(bench_model, side_, out, bench_q, probe=probe, arm_seed=arm_seed, waypoints=arm_waypoints)
        for g, r in zip(out, scr):
            g["arm"] = {k: r.get(k) for k in ("ok", "residual_mm", "residual_deg", "reason", "duplicate_of")}
        out.sort(key=lambda g: (0 if g["arm"]["ok"] else (2 if g["arm"].get("duplicate_of") is not None else 1),
                                0 if g["hold"]["opposed"] else 1, -g["n_touching"], g["overlap_mm"], g["retract_m"] or 0.0))
    if verify and out:
        bid = H.grasp_body(model, scene, ids)
        for g in out[:int(verify_top)]:
            r = H.stress_rollout(model, data0, rig, ids, g, bid, phases=("approach", "close", "lift"),
                                 lift_heights=(0.06, 0.10))
            g["lift"] = {"ok": r["ok"], "first_fail": r["first_fail"],
                         "achieved_mm": r["lifts"][-1]["achieved_mm"] if r["lifts"] else 0.0}
        out.sort(key=lambda g: (0 if g.get("arm", {}).get("ok", True) else 1, 0 if g.get("lift", {}).get("ok") else 1,
                                0 if g["hold"]["opposed"] else 1, -g.get("lift", {}).get("achieved_mm", 0.0), -g["n_touching"]))
    res = {"grasps": out, "n_candidates": len(expanded), "n_feasible": len(out),
           "n_arm_feasible": sum(1 for g in out if g.get("arm", {}).get("ok")) if bench_scene else None,
           "object_extent": {"lo": lo.tolist(), "hi": hi.tolist()},
           "object_axis": {"axis": [round(float(v), 5) for v in u], "tilt_deg": round(tilt, 2)}}
    if out_json:
        import json
        from pathlib import Path
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(out, indent=1, default=float))      # the ranked list, as generate_flat_pinches writes it
        res["out_json"] = str(out_json)
    return res
