"""Hook a ring, strap or handle lying flat: a finger down through its opening, the thumb outside its wall."""
from __future__ import annotations

TOOL = {
    "name": "hook_grasp_pose",
    "title": "Hook grasp pose",
    "type": "skill",
    "category": "Grasp & Poses",
    "summary": "Solve a hook grasp of a part with an opening (a scissor strap, a key ring, a handle lying flat): one or more "
               "fingers pointing down through the opening press its inner wall, the thumb outside presses the same wall "
               "from the other side, both with their fingertips upright so nothing reaches under the tabletop.",
    "description": (
        "A ring or a strap lying flat is too low for pads that close from its sides with the fingers slanted (a pad's "
        "centre must stay a pad radius above the table), and no finger gets under it. Its opening is the way in: a "
        "fingertip pointing down fits through an opening wider than the fingertip (thin_grasp_diagnosis lists the "
        "openings seen from above, with the largest circle that fits), and an upright fingertip touches a wall with "
        "the side of its pad at any height above its own radius. The opening is found from above (nonprehensile."
        "openings: the part's top-down occupancy, the enclosed empty regions); `opening` picks one (default the widest). "
        "The wall to hook is the one in direction `wall_dir` from the opening's centre (default: each side in turn, the "
        "stations' side first). The hooking fingers (`fingers`, default ['index']) touch the inner face of that wall and the "
        "thumb its outer face, both at `contact_height_m` above the support (default: tried 1.5, 2.5 and 3.5 mm under the "
        "wall's top, never under a pad radius + 1 mm), pressing toward each other; an upright fingertip touches a low wall with its "
        "distal cap, so the fingertip sites go that cap's offset above the contacts (nonprehensile.tip_cap_offset: 6 mm "
        "on the Sharpa index, whose visible tip reaches 12.4 mm past its site). With thumb false the fingers alone hook "
        "the wall (a pure hook: the planner pulls along `pull_dir` while rising so the part tips onto them). The "
        "fingertip IK (pose_from_points, seeded from above) keeps every link and the visible meshes out of the table; "
        "one candidate per sector of wrist heading (`sectors`) is kept when the residual and overlap are within limits. "
        "For a round opening every pose is also turned about the opening's vertical axis in 15 deg steps (the same hook on "
        "another stretch of wall, kept where the hand stays out of the part): an arm at the bench edge holds a few hand "
        "headings only, and the turned copies offer them without another IK; the six the arm holds best are stress-tested. "
        "The hooking fingers are the anvil (they hold as posed, a millimetre off the inner wall) and the thumb's clamp "
        "presses the wall onto them, so grip_fingers is the thumb (a pure hook clamps the hooking fingers): flexing an "
        "upright fingertip swings it off a wall instead of into it. The open shape (finger_targets_open, open_frac) is the widest opening of the grip whose hooking "
        "fingertips stay inside the opening with 1.5 mm to spare and whose thumb stays outside the wall, so the hand "
        "comes straight down (approach +z) into the opening. Give the poses to the `hook` motion (or grasp_lift) and "
        "stress-test them with grasp_lift_test."
    ),
    "inputs": [
        {"name": "scene", "type": "dict | str", "description": "Hand scene (hand_scene_from_recon output or its json); the part is its last object / grasp_body."},
        {"name": "opening", "type": "int | list | None", "description": "Index into the openings found (widest first), or a world [x, y] inside the opening. Default 0."},
        {"name": "wall_dir", "type": "list | None", "description": "[dx, dy] from the opening's centre toward the wall to hook. Default None: eight directions in turn, toward the stations first, until one gives a pose the arm holds."},
        {"name": "fingers", "type": "list", "description": "Hooking fingers through the opening. Default ['index']."},
        {"name": "thumb", "type": "bool", "description": "Thumb on the wall's outside (default true; false: a pure hook)."},
        {"name": "contact_height_m", "type": "float | None", "description": "Contact height above the support (default: tried 1.5, 2.5 and 3.5 mm under the wall's top, never lower than a pad radius + 1 mm, until one gives a pose)."},
        {"name": "sectors", "type": "int", "description": "Wrist headings about the vertical are split in this many sectors, one candidate each (an arm reaches some headings only). Default 12."},
        {"name": "max_residual_mm, max_overlap_mm", "type": "float", "description": "Acceptance (defaults 3, 1.5: a finger link resting on a 10 mm wall's top pushes the part away on the way down)."},
        {"name": "iters", "type": "int", "description": "IK iterations per seed. Default 200."},
        {"name": "arm_screen", "type": "bool", "description": "With the hand scene's bench_scene, keep only the candidates the station's arm holds at the pre-grasp, the grasp and the lift (armhand.screen_poses; `arm_dropped` counts the others). Default true."},
        {"name": "screen_lift", "type": "bool", "description": "Run each candidate through the grasp_stress_test ladder in the hand scene (approach, close, lifts of 6, 10 and 15 cm, 4 cm carries in five directions, 25 deg turns, hold; hand.stress_rollout) and drop the ones that fail (listed in `dropped`). Default true."},
        {"name": "out_json", "type": "str | None", "description": "Write {pose, poses, opening, ...} here."},
    ],
    "outputs": [
        {"name": "result", "type": "dict", "description": "ok, pose, poses (ranked; pos, euler, finger_targets, finger_targets_open, open_frac, grip_fingers, points, normals, approach [0, 0, 1], pull_dir (a pure hook), anchor, hook {opening, wall_dir, contact_height_m}), openings (all found), opening (the one used), why (when nothing fits: the opening narrower than a fingertip, the wall too low for the thumb, ...)."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["grasp", "hook", "ring", "strap", "handle", "thin-object", "scissors", "stage3"],
    "status": "tested",
    "entrypoint": "hook_grasp_pose",
}

import json
from pathlib import Path

import mujoco
import numpy as np

from tool_library.common import hand as H
from tool_library.common import nonprehensile as NP


def hook_grasp_pose(scene, opening=0, wall_dir=None, fingers=("index",), thumb=True, contact_height_m=None,
                    sectors=12, max_residual_mm=3.0, max_overlap_mm=1.5, iters=200, arm_screen=True, screen_lift=True, out_json=None):
    spec = H.load_scene_spec(scene)
    fingers = [f for f in fingers if f != "thumb"]
    if not fingers:
        raise ValueError("fingers must name at least one hooking finger besides the thumb")
    model, d0, rig, ids = H.open_scene(spec)
    bid = H.grasp_body(model, spec, ids)
    bodies = NP.E.subtree_bodies(model, bid)
    gs = NP.hand_geoms(model, bodies)
    fp = NP.footprint(model, d0, bodies)
    dims = NP.hand_dims(model, rig)
    fl = ids.get("floor_gid", -1)
    support_z = float(H.floor_surface(model, d0, fl)[0][2]) if fl >= 0 else NP.support_under(model, d0, bodies, fp)[1]
    holes = NP.openings(model, d0, bodies, fp)
    out = {"ok": False, "pose": None, "poses": [], "openings": holes, "opening": None, "tried": []}
    if not holes:
        out["why"] = "no through-opening seen from above"
        return _write(out, out_json)
    if isinstance(opening, (list, tuple)):
        xy = np.asarray(opening, float)[:2]
        hole = min(holes, key=lambda o: float(np.linalg.norm(np.asarray(o["centre"]) - xy)))
    else:
        hole = holes[int(opening)]
    out["opening"] = hole
    rmin = 0.5 * dims["tip_thickness_m"] + 0.0015
    if hole["inscribed_radius_m"] < rmin:
        out["why"] = (f"the opening is {1000 * 2 * hole['inscribed_radius_m']:.0f} mm across; a fingertip needs {1000 * 2 * rmin:.0f} mm")
        return _write(out, out_json)
    wall_top = hole["wall_top_z"]
    if thumb and wall_top - support_z < NP.hook_thumb_wall(dims) - 1e-4:
        out["why"] = (f"the wall stands {1000 * (wall_top - support_z):.1f} mm: too low for the thumb's pad on its outside "
                      "(a pure hook: thumb false)")
        return _write(out, out_json)
    # contact heights tried, highest first (a finger link that rests on the wall's top on the way down pushes the part
    # away; a contact 1.5 mm under the top keeps the fingertips' caps off the table)
    lo_z = support_z + dims["pad_radius_m"] + 0.001
    heights = ([support_z + float(contact_height_m)] if contact_height_m is not None else
               sorted({round(max(lo_z, wall_top - dz_), 5) for dz_ in (0.0015, 0.0025, 0.0035)}, reverse=True))
    # the wall to hook: the one given, else each side of the opening in turn (toward the stations first) until one
    # gives a pose the arm holds
    dirs = [wall_dir] if wall_dir is not None else [(float(np.cos(a)), float(np.sin(a))) for a in np.radians([270, 315, 225, 0, 180, 45, 135, 90])]
    for wd in dirs:
        for zc in heights:
            zc = min(zc, wall_top - 0.0015)
            _hook_at_height(model, d0, rig, ids, spec, gs, hole, wd, fingers, thumb, zc, support_z, sectors, iters,
                            max_residual_mm, max_overlap_mm, screen_lift, bid, out, arm_screen)
            if out["poses"]:
                break
        if out["poses"]:
            break
    if out["poses"]:
        out.update(ok=True, pose=out["poses"][0])
    else:
        out["why"] = "no wrist heading gave a hook within the residual and overlap limits with an opening that clears the part on the way down"
    return _write(out, out_json)


def _hook_at_height(model, d0, rig, ids, spec, gs, hole, wall_dir, fingers, thumb, zc, support_z, sectors, iters, max_residual_mm,
                    max_overlap_mm, screen_lift, bid, out, arm_screen=True):
    """One contact height of hook_grasp_pose: solve, check the descent, screen; appends to out['poses'] / out['tried']."""
    c = np.asarray(hole["centre"], float)
    u = NP._unit([wall_dir[0], wall_dir[1], 0.0])
    # the wall's inner and outer faces along u, at the contact height: rays inward from far outside and outward from the
    # opening's centre, on the part's own geoms
    def ray_h(p0, vec):
        ds = [x for x in (NP._ray(model, d0, g, np.asarray(p0, float), np.asarray(vec, float)) for g in gs) if x >= 0]
        return min(ds) if ds else None
    p_c = np.array([c[0], c[1], zc])
    t_in = ray_h(p_c, u)
    if t_in is None:
        out["tried"].append({"contact_height_m": round(zc - support_z, 4), "ok": False, "why": "no wall found in wall_dir from the opening's centre"})
        return
    inner = p_c + u * t_in
    far = inner + u * 0.08
    t_out = ray_h(far, -u)
    outer = far - u * t_out
    wall_t = float((outer - inner) @ u)
    pts, nrm = {}, {}
    spread = 0.019
    offs = (np.arange(len(fingers)) - 0.5 * (len(fingers) - 1)) * spread
    tang = np.array([-u[1], u[0], 0.0])
    for f, o in zip(fingers, offs):
        p = inner + tang * o
        tin = ray_h(np.array([c[0], c[1], zc]) + tang * o, u)
        pts[f] = (np.array([c[0], c[1], zc]) + tang * o + u * tin) if tin is not None else p
        nrm[f] = -u
    if thumb:
        pts["thumb"] = outer; nrm["thumb"] = u
    # upright fingertips touch the wall with their distal caps: the sites go that cap offset above the contacts
    tg = NP.upright_tip_targets(model, rig, pts, nrm)
    anchor = NP.body_anchor(model, d0, bid)
    # the hooking fingers are the anvil: posed a millimetre off the inner wall, they hold as posed while the thumb's
    # clamp presses the wall onto them (flexing an upright fingertip swings it off the wall); a pure hook clamps them
    grip = ["thumb"] if thumb else list(fingers)
    cands, tried = NP.solve_sector_poses(spec, (model, d0, rig, ids), pts, nrm, tg, approach_dirs=((0.0, 0.0, 1.0),), sectors=int(sectors),
                                         iters=iters, max_residual_mm=max_residual_mm, max_overlap_mm=max_overlap_mm, grip=grip, caps=True)
    out["tried"] += [dict(t_, contact_height_m=round(zc - support_z, 4)) for t_ in tried]
    for pose in cands:
        closed = {k: float(v) for k, v in pose["finger_targets"].items()}
        # the widest opening whose hooking tips stay inside the opening and whose thumb stays outside, along the descent
        chosen = None
        for frac in (0.35, 0.25, 0.15, 0.08, 0.0):
            # the clamping fingers open by frac; the hooking (anvil) fingers keep their posed shape all the way down
            op = dict(closed)
            if frac > 0:
                opened = H.open_targets(rig, closed, frac)
                op.update({k: v for k, v in opened.items() if H.finger_of_joint(k) in grip})
            ok = True
            # at the pose the pads may touch (they are a millimetre off their walls); above it, on the way down, nothing may
            for dz, keep in ((0.0, tuple(grip)), (0.005, ()), (0.015, ()), (0.03, ()), (0.06, ())):
                dd = mujoco.MjData(model); dd.qpos[:] = d0.qpos
                rig.teleport(dd, np.asarray(pose["pos"]) + [0, 0, dz], pose["euler"]); rig.set_fingers(dd, op); mujoco.mj_forward(model, dd)
                if H.deepest_overlap(model, dd, rig, ids, keep_tips=keep, visual_floor=True) > max_overlap_mm / 1000.0:
                    ok = False; break
            if ok:
                chosen = (frac, op); break
        if chosen is None:
            continue
        frac, op = chosen
        pose.update({"grip_fingers": grip, "hook_fingers": list(fingers), "normals": {f: v.tolist() for f, v in nrm.items()}, "targets": {f: v.tolist() for f, v in tg.items()},
                     "approach": [0.0, 0.0, 1.0], "approach_m": 0.06, "finger_targets_open": {k: float(v) for k, v in op.items()},
                     "open_frac": float(frac), "anchor": anchor, "press_depth_rad": 0.15,
                     "hook": {"opening": hole, "wall_dir": u[:2].round(6).tolist(), "contact_height_m": round(zc - support_z, 4),
                              "wall_thickness_m": round(wall_t, 4), "thumb": bool(thumb)}})
        if not thumb:
            pose["pull_dir"] = u[:2].round(6).tolist()          # pull the hooked wall away from the opening: it tips onto the fingers
        out["poses"].append(pose)
    # a round opening looks the same all around: each pose turned about the opening's vertical axis is the same hook
    # on another stretch of wall (kept where the turned hand stays out of the part, a tab or a blade included). An arm
    # at the bench edge holds a few hand headings only; the turned copies give it those headings without a new IK
    if out["poses"] and hole["area_m2"] <= 1.6 * np.pi * hole["inscribed_radius_m"] ** 2:
        from scipy.spatial.transform import Rotation as Rot
        cen = np.array([hole["centre"][0], hole["centre"][1], 0.0])
        turned = []
        for pose in out["poses"]:
            p0 = np.asarray(pose["pos"], float); R0 = H.euler_to_mat(pose["euler"])
            for ang in range(15, 360, 15):
                Rz = Rot.from_euler("z", float(ang), degrees=True).as_matrix()
                pz = dict(pose); pz["pos"] = (cen + Rz @ (p0 - cen)).tolist(); pz["euler"] = H.mat_to_euler(Rz @ R0).tolist()
                pz["points"] = {f: (cen + Rz @ (np.asarray(v, float) - cen)).tolist() for f, v in pose["points"].items()}
                pz["normals"] = {f: (Rz @ np.asarray(v, float)).tolist() for f, v in pose["normals"].items()}
                pz["targets"] = {f: (cen + Rz @ (np.asarray(v, float) - cen)).tolist() for f, v in pose["targets"].items()}
                pz["hook"] = dict(pose["hook"], wall_dir=(Rz[:2, :2] @ np.asarray(pose["hook"]["wall_dir"], float)).round(6).tolist(), turned_deg=ang)
                ok = True
                for dz, keep in ((0.0, tuple(grip)), (0.015, ()), (0.06, ())):
                    dd = mujoco.MjData(model); dd.qpos[:] = d0.qpos
                    rig.teleport(dd, np.asarray(pz["pos"]) + [0, 0, dz], pz["euler"])
                    rig.set_fingers(dd, pz["finger_targets"] if dz == 0.0 else pz["finger_targets_open"]); mujoco.mj_forward(model, dd)
                    if H.deepest_overlap(model, dd, rig, ids, keep_tips=keep, visual_floor=True) > max_overlap_mm / 1000.0:
                        ok = False; break
                if ok:
                    turned.append(pz)
        out["poses"] += turned
    bench = spec.get("bench_scene")
    if arm_screen and bench and Path(str(bench)).exists() and out["poses"]:
        scr = NP.arm_screen(bench, rig.side, out["poses"])
        for pz, (ok_, mm, why) in zip(out["poses"], scr):
            pz["arm_screen"] = {"ok": ok_, "residual_mm": mm, "reason": why}
        out["arm_dropped"] = out.get("arm_dropped", 0) + sum(1 for pz in out["poses"] if not pz["arm_screen"]["ok"])
        out["poses"] = sorted([pz for pz in out["poses"] if pz["arm_screen"]["ok"]], key=lambda pz: pz["arm_screen"]["residual_mm"])[:6]
    if screen_lift and out["poses"]:
        # the approach-close-lift rollout in the hand scene: only candidates that lift are given to the executor
        res = NP.lift_screen(spec, out["poses"])
        for pz, (ok_, ff, mm) in zip(out["poses"], res):
            pz["lift_screen"] = {"ok": ok_, "first_fail": ff, "lift_mm": mm}
        out.setdefault("dropped", []).extend({"index": i, "contact_height_m": round(zc - support_z, 4), **pz["lift_screen"]} for i, pz in enumerate(out["poses"]) if not pz["lift_screen"]["ok"])
        out["poses"] = [pz for pz in out["poses"] if pz["lift_screen"]["ok"]]


def _write(out, out_json):
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(out, indent=1, default=float))
        out["out_json"] = str(out_json)
    return out
