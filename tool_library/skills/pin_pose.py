"""Hand poses that steady (pin) a part, or one link or end of a cable, for the other hand."""
from __future__ import annotations

TOOL = {
    "name": "pin_pose",
    "title": "Pin pose",
    "type": "skill",
    "category": "Grasp & Poses",
    "summary": "Hand poses that steady a part, or one link or end of a cable, while the other hand acts: fingertip, two-finger or "
               "flat-hand presses onto its top (or given grip candidates), kept out of the zones the other hand needs and "
               "reachable by this station's arm, ranked; the input of a `pin` plan entry.",
    "description": (
        "A steadying hand must hold the part still AND leave the other hand its room: a wrap around a bottle that "
        "reaches up to the cap is in the way of the hand turning the cap, and a press on a free cable end that the "
        "station's arm cannot hold steadies nothing. Press mode samples the part's top surface (vertical rays on its "
        "own collision geoms, keeping hits whose normal points up; or the `points` given), and at each point poses "
        "the families asked for: 'flat' (flat_hand_pose: straight index and middle fingers, pads down, pitched so "
        "only the fingertips touch, in `n_dirs` finger directions), 'tip' (push_pose_from_point: the index finger "
        "pointing into the surface, straight down and tilted 30 deg toward each of four directions) and 'palm' "
        "(flat_hand_pose unpitched: the whole flat hand on a large top). Grip mode ranks the given `candidates` (any "
        "grasp tool's poses; for a rigid part without candidates generate_rigid_grasps is called with the zones as "
        "keep_clear, wrapping low on the part; the best grips are driven through their approach and close, and one that "
        "shoves the part more than 1 cm is dropped). Every candidate is judged in the hand scene: no hand geom more than "
        "max_overlap_m into the "
        "scene, including pressing pads along the approach; a grip is judged against everything but its part), the posed "
        "hand pushed along the press direction meets the part with an intended pressing finger "
        "within 1 cm (the pose is corrected to first contact and the executor presses from that contact), and its collision geometry, at the pose "
        "and along the approach, at least clear_margin_m out of every keep_clear zone: {center, radius} spheres, "
        "{min, max} boxes, {p0, p1, radius} capsules (the swept cylinder of a cap and the fingers turning it), and "
        "{hand_pose, side, sweep?: {point, axis, deg}} (the OTHER hand placed at its own pose in the bench scene, "
        "optionally turned about an axis through a range: the volume a hand turning a cap sweeps). With bench_scene "
        "and side every surviving candidate is screened for this station's arm (armhand.screen_poses: the arm holds "
        "the hand at the pose and the pre-grasp, without running into the scene) and the arm's links must also stay "
        "out of the zones. Ranked: feasible first, then the larger zone margin (up to 5 cm), a press square to the "
        "support (its normal nearest the vertical, in 10 deg steps), more pressing pads (two pads cannot roll a round part "
        "from under them as one fingertip can), the nearer to the part's centre (or to "
        "`prefer_point`; the first press point is also taken on the crest), the smaller overlap. Each pose carries pin_mode, its press normals and points, the "
        "approach, and pin {family, point, normal, keep_clear_mm, overlap_mm, press_offset_mm, reach}; a `pin` entry "
        "takes it as its pose (force_n sets the press)."
    ),
    "inputs": [
        {"name": "scene", "type": "dict | str", "description": "Hand scene (hand_scene_from_recon dict or its JSON path) of the pinning hand; the part is its grasp body unless `body` names one."},
        {"name": "body", "type": "str | None", "description": "The body to pin (hand-scene name, the bench name: a cable link such as '<rope>_ropeB_first'). Default the hand scene's grasp body."},
        {"name": "mode", "type": "str", "description": "'press' (default) or 'grip'."},
        {"name": "families", "type": "list", "description": "Press families: 'flat', 'tip', 'palm'. Default ['flat', 'tip']."},
        {"name": "points", "type": "list | None", "description": "Press points [[x, y, z], ...] on the part's top (world). Default: sampled on the top surface."},
        {"name": "n_points", "type": "int", "description": "Sampled press points. Default 4."},
        {"name": "n_dirs", "type": "int", "description": "Finger directions per point for 'flat'. Default 8."},
        {"name": "pitch_deg", "type": "float", "description": "Pitch of a 'flat' press (fingertips touch, the palm clears). Default 30."},
        {"name": "fingers", "type": "list", "description": "Pressing fingers of a 'flat' press. Default ['index', 'middle']."},
        {"name": "keep_clear", "type": "list | dict | None", "description": "Zones no part of the hand (or its arm) may come within clear_margin_m of (see the description)."},
        {"name": "clear_margin_m", "type": "float", "description": "Default 0.01."},
        {"name": "candidates", "type": "list | str | None", "description": "Grip mode: poses (or a JSON file of them) to rank."},
        {"name": "verify, verify_top, max_grip_shift_m", "type": "bool, int, float", "description": "Grip mode: drive the approach and the close of the best candidates through the actuators (hand.stress_rollout) and reject a grip that shoves the part more than max_grip_shift_m (the pin entry's own placement tolerance) or closes on it with fewer than two fingers, until verify_top pass. Defaults true, 6, 0.01."},
        {"name": "grip_height_fracs", "type": "list", "description": "Grip mode without candidates: the heights (fractions of the part's height) generate_rigid_grasps wraps at. Default [0.3, 0.45, 0.6]: a steadying grip low on the part leaves its top to the other hand."},
        {"name": "bench_scene", "type": "str | None", "description": "Bench scene MJCF for the arm screen and hand_pose zones."},
        {"name": "side", "type": "str | None", "description": "The pinning hand's side (default the hand scene's)."},
        {"name": "state", "type": "str | list | None", "description": "The bench state to screen against: trajectory.npz or qpos; default the hand scene's recorded bench state. Conflicting object states require rebuilding the hand scene."},
        {"name": "prefer_point", "type": "list | None", "description": "Rank presses nearer this world point first (default the part's centre)."},
        {"name": "max_overlap_m", "type": "float", "description": "Default 0.003."},
        {"name": "approach_m", "type": "float", "description": "Approach distance checked against the zones and screened for the arm. Default 0.05."},
        {"name": "top_k", "type": "int", "description": "Poses returned. Default 6."},
        {"name": "max_screen", "type": "int", "description": "At most this many poses (the best the hand scene admits) are screened for the arm. Default 24."},
        {"name": "out_json", "type": "str | None", "description": "Write the best pose here (and the ranked list to <stem>_all.json)."},
    ],
    "outputs": [
        {"name": "result", "type": "dict", "description": "{ok, pose (the best feasible, or None), poses [ranked, each with pin {...} and ok], n_candidates, rejected {reason: count}, body, mode}."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "bimanual", "pin", "steady", "press", "keep-out", "reach"],
    "used_in": [],
    "status": "tested",
    "entrypoint": "pin_pose",
}

import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def _unit(v):
    v = np.asarray(v, float)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def geom_points(model, data, gids, n_prim=10):
    """World points on a set of collision geoms: mesh vertices; for primitives points on their surface."""
    pts = []
    for g in gids:
        c = np.asarray(data.geom_xpos[g], float); R = np.asarray(data.geom_xmat[g], float).reshape(3, 3); sz = model.geom_size[g]
        typ = int(model.geom_type[g])
        if typ == int(mujoco.mjtGeom.mjGEOM_MESH):
            mid = int(model.geom_dataid[g]); a = int(model.mesh_vertadr[mid]); nv = int(model.mesh_vertnum[mid])
            pts.append(model.mesh_vert[a:a + nv] @ R.T + c)
        elif typ in (int(mujoco.mjtGeom.mjGEOM_SPHERE), int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
            r = float(sz[0]); hl = float(sz[1]) if typ != int(mujoco.mjtGeom.mjGEOM_SPHERE) else 0.0
            ang = np.linspace(0, 2 * np.pi, n_prim, endpoint=False)
            ring = np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], 1) * r
            loc = [ring + [0, 0, z] for z in np.linspace(-hl, hl, max(2, int(hl / 0.005) + 1))] + [np.array([[0, 0, hl + r], [0, 0, -hl - r]])]
            pts.append(np.vstack(loc) @ R.T + c)
        elif typ == int(mujoco.mjtGeom.mjGEOM_BOX):
            s = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)], float) * sz[:3]
            pts.append(s @ R.T + c)
        else:
            pts.append(c[None] + np.eye(3) * float(model.geom_rbound[g]))
    return np.vstack(pts) if pts else np.zeros((0, 3))


def _seg_dist(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def zone_clearance(P, zones):
    """Smallest signed distance (m) of points P to the zones (negative: inside one); inf with no zones. Zones: {center,
    radius} | {min, max} | {p0, p1, radius} | {points, radius} (a point cloud, e.g. the other hand)."""
    best = np.inf
    if len(P) == 0:
        return best
    for z in zones:
        if "points" in z:
            from scipy.spatial import cKDTree
            dd, _ = cKDTree(np.asarray(z["points"], float)).query(P, k=1)
            best = min(best, float(dd.min()) - float(z.get("radius", 0.0)))
        elif "p0" in z:
            best = min(best, float(_seg_dist(P, np.asarray(z["p0"], float), np.asarray(z["p1"], float)).min()) - float(z["radius"]))
        elif "radius" in z:
            best = min(best, float(np.linalg.norm(P - np.asarray(z["center"], float), axis=1).min()) - float(z["radius"]))
        elif "min" in z:
            lo, hi = np.asarray(z["min"], float), np.asarray(z["max"], float)
            c, h = 0.5 * (lo + hi), 0.5 * (hi - lo)
            q = np.abs(P - c) - h
            out = np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)
            best = min(best, float(out.min()))
    return best


def _hand_zone(bench_model, z):
    """A {hand_pose, side, sweep?} zone as a point cloud: the other hand's collision geometry at its pose in the bench
    scene, turned about sweep.axis through sweep.point over sweep.deg (both directions when deg is negative-free)."""
    from tool_library.common import plan as P
    from tool_library.common.subgoal import load_pose
    side = z["side"]
    rigs = P.scene_rigs(bench_model)
    rig = rigs[side]
    d = mujoco.MjData(bench_model)
    if bench_model.nkey:
        mujoco.mj_resetDataKeyframe(bench_model, d, 0)
    pose = load_pose(z["hand_pose"])
    p, R = P.pose_from_dict(pose)
    sw = z.get("sweep") or {}
    angs = [0.0]
    if sw:
        deg = float(sw.get("deg", 0.0))
        angs = list(np.linspace(min(0.0, deg), max(0.0, deg), max(2, int(abs(deg) / 10.0) + 1)))
    gids = [g for g in range(bench_model.ngeom) if f"_collision_hand_{side}_" in (bench_model.geom(g).name or "")]
    pts = []
    for a in angs:
        Rd = Rot.from_rotvec(np.radians(a) * _unit(sw.get("axis", [0, 0, 1]))).as_matrix() if sw else np.eye(3)
        piv = np.asarray(sw.get("point", p), float) if sw else p
        P.set_wrist(bench_model, d, rig, piv + Rd @ (p - piv), Rd @ R)
        rig.set_fingers(d, {k: float(v) for k, v in (pose.get("finger_targets") or {}).items() if k in rig.finger_qpos})
        mujoco.mj_kinematics(bench_model, d)
        pts.append(geom_points(bench_model, d, gids, n_prim=8))
    return {"points": np.vstack(pts), "radius": 0.0}


def _arm_points(bench_model, side, pose, q_base):
    """Points on the visible links of the station's arm (wrist links to shoulder) holding the hand at `pose`."""
    from tool_library.common import armhand as AH
    from tool_library.common import plan as P
    S = AH.ArmSide(bench_model, side)
    rig = P.scene_rigs(bench_model)[side]
    d = mujoco.MjData(bench_model); d.qpos[:] = q_base
    p, R = P.pose_from_dict(pose)
    P.set_wrist(bench_model, d, rig, p, R)
    q = d.qpos.copy()
    pf, Rf = S.flange_target(q)
    _, _, qa = S.solve(q, pf, Rf, S.home, multistart="force")
    d.qpos[:] = q; d.qpos[S.arm.qadr] = qa
    mujoco.mj_kinematics(bench_model, d)
    gids = [g for g in range(bench_model.ngeom) if (bench_model.body(int(bench_model.geom_bodyid[g])).name or "").startswith(f"robot_{side}_arm_")
            and int(bench_model.geom_group[g]) <= 2 and float(bench_model.geom_rgba[g][3]) > 0
            and not (bench_model.body(int(bench_model.geom_bodyid[g])).name or "").endswith("gripper")]
    return geom_points(bench_model, d, gids, n_prim=8)


def _top_points(model, data, gids, n, prefer):
    """Points on the part's top: vertical rays on a grid over its footprint against its own geoms, normals from the
    neighbouring hits; the n spread out farthest from each other, starting nearest `prefer`."""
    from tool_library.common import hand as H
    ext = H.object_extent(model, data, gids)
    lo, hi = ext["lo"], ext["hi"]
    span = hi - lo
    step = max(0.003, float(min(span[0], span[1])) / 6.0)
    xs = np.arange(lo[0] + 0.5 * step, hi[0], step); ys = np.arange(lo[1] + 0.5 * step, hi[1], step)
    if not len(xs):
        xs = np.array([0.5 * (lo[0] + hi[0])])
    if not len(ys):
        ys = np.array([0.5 * (lo[1] + hi[1])])
    from tool_library.common import episode as E
    gset = set(int(g) for g in gids)

    def ray(x, y):
        best = None
        for g in gset:
            dist = E._ray_down(model, data, g, np.array([x, y, hi[2] + 0.05]))
            if dist >= 0 and (best is None or dist < best):
                best = dist
        return None if best is None else hi[2] + 0.05 - best

    hits = []
    e = 0.0015
    for x in xs:
        for y in ys:
            z0 = ray(x, y)
            if z0 is None:
                continue
            zx, zy = ray(x + e, y), ray(x, y + e)
            if zx is None or zy is None:
                continue
            nrm = _unit(np.cross([e, 0.0, zx - z0], [0.0, e, zy - z0]))
            if nrm[2] < 0:
                nrm = -nrm
            if nrm[2] >= 0.85:
                hits.append((np.array([x, y, z0]), nrm))
    if not hits:
        return []
    pref = np.asarray(prefer, float)
    # the first point: near `prefer` and on the crest (a press off the crest of a round cable rolls it from under the pad:
    # 2 mm off the crest of a 24 mm cable tilts the normal 6 deg)
    order = [int(np.argmin([np.linalg.norm(h[0][:2] - pref[:2]) + 2.0 * (1.0 - h[1][2]) for h in hits]))]
    while len(order) < min(n, len(hits)):
        dmin = [min(np.linalg.norm(hits[i][0] - hits[j][0]) for j in order) for i in range(len(hits))]
        order.append(int(np.argmax(dmin)))
    return [hits[i] for i in order]


def _rank(j):
    """Feasible first; then room for the other hand (5 cm is plenty), a press square to the support (in 10 deg steps: a
    press 26 deg off the crest of a round cable rolled it from under the pad), more pressing pads (two pads cannot roll the
    part from under them as one fingertip can), nearer the preferred point, less overlap."""
    i = j["info"]
    clr = i["keep_clear_mm"] if i["keep_clear_mm"] is not None else 50.0
    tilt = int(float(i.get("normal_tilt_deg", 0.0)) // 10.0)
    return (j["why"] is not None, -min(clr, 50.0), tilt, -len(j["pose"].get("grip_fingers") or []), i["dist_to_prefer_mm"], i["overlap_mm"])


def pin_pose(scene, body=None, mode="press", families=("flat", "tip"), points=None, n_points=4, n_dirs=8, pitch_deg=30.0,
             fingers=("index", "middle"), keep_clear=None, clear_margin_m=0.01, candidates=None, bench_scene=None, side=None, state=None,
             prefer_point=None, max_overlap_m=0.003, approach_m=0.05, top_k=6, max_screen=24, grip_height_fracs=(0.3, 0.45, 0.6), verify=True, verify_top=6, max_grip_shift_m=0.01, out_json=None):
    from tool_library.common import hand as H
    from tool_library.common import plan as P
    from tool_library.common import bimanual as B
    from tool_library.common.subgoal import load_pose
    from tool_library.skills.flat_hand_pose import flat_hand_pose
    from tool_library.skills.push_pose_from_point import push_pose_from_point
    if isinstance(scene, (str, Path)):
        scene = H.load_scene_spec(scene)
    if mode not in ("press", "grip"):
        raise ValueError(f"mode must be 'press' or 'grip', got {mode!r}")
    if not 0 <= float(max_overlap_m) <= .003:
        raise ValueError("max_overlap_m must be between 0 and 0.003; pin safety limits cannot be relaxed")
    model, data0, rig, ids = H.open_scene(scene)
    side = side or rig.side
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body) if body else H.grasp_body(model, scene, ids)
    if b < 0:
        raise ValueError(f"no body {body!r} in the hand scene")
    part_b = B.rigid_bodies(model, b)
    part_g = sorted(g for g in ids["obj_gids"] if int(model.geom_bodyid[g]) in part_b)
    if not part_g:
        raise ValueError(f"{model.body(b).name!r} has no collision geometry the hand can touch")
    centre = np.asarray(H.object_extent(model, data0, part_g)["centre"], float)
    prefer = np.asarray(prefer_point if prefer_point is not None else centre, float)
    zones = list(keep_clear if isinstance(keep_clear, list) else ([keep_clear] if keep_clear else []))
    bench_model = mujoco.MjModel.from_xml_path(str(bench_scene)) if bench_scene else None
    if any("hand_pose" in z for z in zones) and bench_model is None:
        raise ValueError("a hand_pose zone needs bench_scene")
    zones = [(_hand_zone(bench_model, z) if "hand_pose" in z else z) for z in zones]
    q_base = None
    if bench_model is not None:
        from tool_library.skills.hand_scene_from_recon import resolve_bench_state
        q_base = resolve_bench_state(scene, bench_model, state)
        if q_base is None:
            q_base = np.asarray(bench_model.key_qpos[0] if bench_model.nkey else bench_model.qpos0, float).copy()
    # ---- candidates
    cands = []
    if mode == "press":
        pts = [(np.asarray(p_, float), np.array([0.0, 0.0, 1.0])) for p_ in points] if points else _top_points(model, data0, part_g, int(n_points), prefer)
        if not pts:
            raise ValueError(f"no upward-facing surface on {model.body(b).name!r} to press (give `points`, or mode 'grip')")
        for pt, nrm in pts:
            ref = _unit(np.cross(nrm, [0.0, 0.0, 1.0]) if abs(nrm[2]) < 0.99 else [1.0, 0.0, 0.0])
            ref2 = np.cross(nrm, ref)
            dirs = [np.cos(a) * ref + np.sin(a) * ref2 for a in np.linspace(0, 2 * np.pi, int(n_dirs), endpoint=False)]
            if "flat" in families or "palm" in families:
                for fam, pitch, fings in (("flat", float(pitch_deg), list(fingers)), ("palm", 0.0, ["index", "middle", "ring", "pinky"])):
                    if fam not in families:
                        continue
                    for fd in dirs:
                        try:
                            ps = flat_hand_pose(scene, pt.tolist(), nrm.tolist(), fd.tolist(), fingers=fings, contact="pad", pitch_deg=pitch)
                        except Exception:                       # a direction the hand cannot be laid in: skip it
                            continue
                        cands.append((ps, {"family": fam, "point": pt.round(5).tolist(), "normal": nrm.round(4).tolist(),
                                           "finger_dir": np.round(fd, 3).tolist()}))
            if "tip" in families:
                tilts = [(0.0, None)] + [(30.0, dirs[i]) for i in range(0, len(dirs), max(1, len(dirs) // 4))]
                for tilt, az in tilts:
                    nn = nrm if az is None else _unit(np.cos(np.radians(tilt)) * nrm + np.sin(np.radians(tilt)) * az)
                    try:
                        ps = push_pose_from_point(scene, pt.tolist(), nn.tolist(), finger="index")
                    except Exception:
                        continue
                    ps = dict(ps)
                    ps.setdefault("grip_fingers", ["index"])
                    ps["normals"] = {"index": nrm.tolist()}
                    ps["points"] = {"index": pt.tolist()}
                    ps["approach"] = nn.tolist()
                    cands.append((ps, {"family": "tip", "point": pt.round(5).tolist(), "normal": nrm.round(4).tolist(), "tilt_deg": tilt}))
    else:
        if candidates is None:
            from tool_library.skills.generate_rigid_grasps import generate_rigid_grasps
            # generate_rigid_grasps takes spheres and boxes: a capsule goes in as a chain of spheres, a point cloud (the other
            # hand) is left to the exact check below
            kc = []
            for z in zones:
                if "p0" in z:
                    a, b_ = np.asarray(z["p0"], float), np.asarray(z["p1"], float)
                    n = max(2, int(np.ceil(np.linalg.norm(b_ - a) / max(float(z["radius"]), 1e-3))) + 1)
                    kc += [{"center": (a + (b_ - a) * t).tolist(), "radius": float(z["radius"])} for t in np.linspace(0.0, 1.0, n)]
                elif "points" not in z:
                    kc.append(z)
            g = generate_rigid_grasps(scene, height_fracs=tuple(grip_height_fracs), keep_clear=kc or None, bench_scene=bench_scene,
                                      state=None if q_base is None else q_base.tolist())
            candidates = g.get("grasps") or []
        elif isinstance(candidates, (str, Path)):
            v = json.loads(Path(candidates).read_text())
            candidates = v.get("grasps") if isinstance(v, dict) and "grasps" in v else (v if isinstance(v, list) else [v])
        for c in candidates:
            cands.append((dict(load_pose(c)), {"family": "grip"}))
    # ---- judge each in the hand scene
    d = mujoco.MjData(model)
    rejected = {}
    judged = []
    for pose, info in cands:
        geometry, off = None, None
        if mode == "press":
            geometry = B.pin_press_geometry(model, data0, rig, pose, part_g, approach_m=approach_m,
                                            max_overlap_m=max_overlap_m)
            off = geometry.get("contact_offset_m")
            info["press_geometry"] = {k: v for k, v in geometry.items() if k not in ("posed", "opened")}
            if geometry["ok"]:
                pose = dict(pose, pos=geometry["contact_pos"], finger_targets=geometry["posed"],
                            finger_targets_open=geometry["opened"])
        d.qpos[:] = data0.qpos
        fing = {k: float(v) for k, v in (pose.get("finger_targets") or {}).items() if k in rig.finger_range}
        H.apply_pose(model, d, rig, pose, fing)
        mujoco.mj_forward(model, d)
        grip = list(pose.get("grip_fingers") or [])
        if mode == "press":
            ov = H.deepest_overlap(model, d, rig, ids, keep_tips=(), visual_floor=True)
        else:
            # a grip touches the part with the whole hand (a wrap's palm and phalanges): only the other objects and the
            # table count; how the grip sits on the part is its stress tests' business
            ov = H.deepest_overlap(model, d, rig, dict(ids, obj_gids=set(ids["obj_gids"]) - set(part_g)), keep_tips=grip)
        hp = geom_points(model, d, sorted(rig.collision_gids))
        p, R = P.pose_from_dict(pose)
        ap = P.retreat_axis(pose, R)
        clr = zone_clearance(hp, zones)
        for s_ in (0.5, 1.0):                        # the way in, as the executor reaches
            H.apply_pose(model, d, rig, {"pos": (p + ap * approach_m * s_).tolist(), "euler": pose["euler"]}, fing)
            mujoco.mj_kinematics(model, d)
            clr = min(clr, zone_clearance(geom_points(model, d, sorted(rig.collision_gids)), zones))
        why = None
        if geometry is not None and not geometry["ok"]:
            why = geometry["reason"]
        elif ov > max_overlap_m:
            why = "overlap"
        elif clr < clear_margin_m:
            why = "keep_clear"
        nz = np.asarray(info.get("normal", [0.0, 0.0, 1.0]), float)
        info.update({"keep_clear_mm": None if not np.isfinite(clr) else round(1000 * clr, 1), "overlap_mm": round(1000 * ov, 2),
                     "normal_tilt_deg": round(float(np.degrees(np.arccos(np.clip(_unit(nz)[2], -1.0, 1.0)))), 1),
                     "press_offset_mm": None if off is None else round(1000 * off, 2),
                     "dist_to_prefer_mm": round(1000 * float(np.linalg.norm(np.asarray(info.get("point", p), float)[:2] - prefer[:2])), 1)})
        judged.append({"pose": pose, "info": info, "why": why})
    # ---- a grip must close on the part without shoving it: the approach and the close driven through the actuators as
    # the executor drives them (hand.stress_rollout), the part's shift measured (a side wrap closing on a free bottle shoved
    # it 36 mm); the best few the hand scene admits, until top_k pass
    if mode == "grip" and verify:
        bid0 = int(b)
        live = sorted([j for j in judged if j["why"] is None], key=lambda j: _rank(j)[1:])
        n_ok = 0
        for j in live:
            if n_ok >= int(verify_top):
                j["why"] = "not_verified"
                continue
            r = H.stress_rollout(model, data0, rig, ids, j["pose"], bid0, phases=("approach", "close"), servo_reserve=False)
            dd = r.pop("data")
            shift = float(np.linalg.norm(dd.xpos[bid0] - data0.xpos[bid0]))
            fing = next((ph for ph in r["phases"] if ph["name"] == "seated"), {}).get("finger_contacts", 0)
            j["info"]["grip_shift_mm"] = round(1000 * shift, 1)
            j["info"]["grip_fingers_touching"] = fing
            j["info"]["grip_physics"] = {k: r.get(k) for k in ("ok", "first_fail", "penetration")}
            if not r.get("ok"):
                j["why"] = "grip_physics"
            elif shift > float(max_grip_shift_m) or fing < 2:
                j["why"] = "grip_shoves_part" if shift > float(max_grip_shift_m) else "grip_misses_part"
            else:
                n_ok += 1
    # ---- the arm: screened only for those the hand scene admits (cheap first)
    if bench_model is not None:
        from tool_library.common import armhand as AH
        # the arm screen solves the arm at the pose and the pre-grasp (seconds a pose): only the best the hand scene
        # admits are screened, in batches, until top_k pass or max_screen were tried
        live = sorted([j for j in judged if j["why"] is None], key=lambda j: _rank(j)[1:])
        probe = AH.ArmProbe(str(bench_scene)) if live else None
        n_ok, batch, k0 = 0, [], 0
        for k0 in range(0, min(len(live), int(max_screen)), 6):
            batch = live[k0:k0 + 6]
            scr = AH.screen_poses(bench_model, side, [j["pose"] for j in batch], q_base, probe=probe, approach_m=approach_m)
            for j, r in zip(batch, scr):
                j["info"]["reach"] = {k: r.get(k) for k in ("ok", "residual_mm", "residual_deg", "reason")}
                if not r.get("ok"):
                    j["why"] = "arm"
                elif zones:
                    # the arm holding this hand must leave the zones free too (a forearm across the cap is as much in the
                    # other hand's way as a finger)
                    ac = zone_clearance(_arm_points(bench_model, side, j["pose"], q_base), zones)
                    j["info"]["arm_keep_clear_mm"] = None if not np.isfinite(ac) else round(1000 * ac, 1)
                    if ac < clear_margin_m:
                        j["why"] = "arm_keep_clear"
                n_ok += j["why"] is None
            if n_ok >= int(top_k):
                break
        for j in live[k0 + len(batch):] if live else []:
            j["why"] = "not_screened"
    for j in judged:
        if j["why"]:
            rejected[j["why"]] = rejected.get(j["why"], 0) + 1
    judged.sort(key=_rank)
    out_poses = []
    for j in judged[:max(int(top_k), 1)]:
        ps = dict(j["pose"]); ps["pin_mode"] = mode; ps["pin"] = dict(j["info"], ok=j["why"] is None, rejected=j["why"])
        ps.setdefault("side", side)
        out_poses.append(ps)
    best = out_poses[0] if out_poses and out_poses[0]["pin"]["ok"] else None
    res = {"ok": best is not None, "pose": best, "poses": out_poses, "n_candidates": len(judged), "rejected": rejected,
           "body": model.body(b).name, "mode": mode}
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        if best is not None:
            Path(out_json).write_text(json.dumps(best, indent=1, default=float))
        Path(str(Path(out_json).with_suffix("")) + "_all.json").write_text(json.dumps(out_poses, indent=1, default=float))
        res["out_json"] = str(out_json)
    return res
