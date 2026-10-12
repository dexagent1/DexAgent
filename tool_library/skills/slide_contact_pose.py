"""The hand pose that slides a part along the table: pads pressed on its top (drag), or a fingertip on a side face (push)."""
from __future__ import annotations

TOOL = {
    "name": "slide_contact_pose",
    "title": "Slide contact pose",
    "type": "skill",
    "category": "Grasp & Poses",
    "summary": "Pose the hand to slide a part across the table (to make it overhang the edge, or to bring it against a stop): "
               "flat fingertips pressed on its top that drag it by friction, or a fingertip pushing one side face; with the "
               "friction check that says whether a drag can move it at all and with how much press.",
    "description": (
        "A drag works when the pads' friction beats the support's: mu_hand N > mu_support (N + m g), so it needs a press "
        "N > mu_support m g / (mu_hand - mu_support) (the hand's contact friction is the library's 1.2, the part-table "
        "friction the larger of the two geoms', as MuJoCo combines them); the press must fit under the 40 N wrist cap "
        "with margin (slide.drag_within_wrist_cap: 1.5 x the press <= 40 N; a drag that does not fit is still returned, "
        "flagged). Mode 'drag' lays the straight index, middle and ring fingers on the part's top over its centre of "
        "mass (so the drag does not turn it), pitched `pitch_deg` so only the fingertips touch (flat_hand_pose), the "
        "fingers pointing along `finger_dir` (default: against the slide, the hand trailing behind the part's centre as "
        "a person pulls a phone toward them); candidates turn the fingers about the vertical (`yaws_deg`) and pitch them "
        "steeper (an arm at a bench edge holds a fingers-down hand far more often than a flat one), returned as `poses` "
        "for the plan entry's pose_candidates (the executor keeps the ones its arm holds). Mode 'push' stands the index and "
        "middle fingertips upright just behind the side face that faces away from the slide, their caps a pad radius "
        "plus 1 mm above the table (on a part lower than that they meet its top edge), one candidate per sector of wrist "
        "heading. Mode 'auto' (default) pushes when the contact slope allows it (tan of the angle "
        "under 0.6 / mu_support: a 5 mm part still, a 1.5 mm strip not) and drags otherwise; the drag rides on the pads' "
        "friction margin over the support's, which on this bench (1.2 against the table's 1.0) let the pads slide 6 cm "
        "over a 170 g slab pressed with 21 N. Mode 'press' gives the drag's flat fingertips at contact_point with no slide: "
        "the press pose of tilt_up. "
        "The pose carries slide_mode and the friction numbers; give it to the `slide` motion with the direction and how "
        "far (overhang_m, distance_m or target_pos)."
    ),
    "inputs": [
        {"name": "scene", "type": "dict | str", "description": "Hand scene (hand_scene_from_recon output or its json); the part is its last object / grasp_body."},
        {"name": "direction", "type": "list", "description": "[dx, dy]: where the part is to go."},
        {"name": "mode", "type": "str", "description": "'auto' (default: push when the contact slope allows, else drag), 'drag', 'push', or 'press' (the drag's flat fingertips at contact_point with no slide: the press pose of tilt_up)."},
        {"name": "finger_dir", "type": "list | None", "description": "Drag: the direction the fingers point (default minus direction)."},
        {"name": "fingers", "type": "list", "description": "Drag / press: fingers on the top. Default ['index', 'middle', 'ring'] (a flat hand); one finger gives an upright fingertip on the contact point (a strip or a small part, where a flat hand's other fingers would press the table)."},
        {"name": "pitch_deg", "type": "float", "description": "Drag: the hand pitched so only the fingertips touch (tried first; 45, 60 and 30 follow). Default 45."},
        {"name": "yaws_deg", "type": "list[float]", "description": "Drag: finger directions tried, degrees about the vertical from finger_dir. Default [0, 30, -30, 60, -60, 90, -90, 120, -120, 180]."},
        {"name": "contact_point", "type": "list | None", "description": "Drag: world xy (or xyz) to press at (default: over the part's centre of mass)."},
        {"name": "out_json", "type": "str | None", "description": "Write the pose here."},
    ],
    "outputs": [
        {"name": "pose", "type": "dict", "description": "The first candidate: hand pose (pos, euler, finger_targets, finger_targets_open, grip_fingers, points, normals, approach) plus slide_mode, slide {direction, mu_hand, mu_support, mass_kg, drag_force_n (the press a drag needs; None: a drag cannot move it), drag_within_wrist_cap, height_m, push_contact_deg, why}, and poses (every candidate, for pose_candidates)."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["slide", "push", "drag", "non-prehensile", "thin-object", "stage3"],
    "status": "tested",
    "entrypoint": "slide_contact_pose",
}

import json
from pathlib import Path

import numpy as np

from tool_library.common import hand as H
from tool_library.common import nonprehensile as NP


def slide_contact_pose(scene, direction, mode="auto", finger_dir=None, fingers=("index", "middle", "ring"), pitch_deg=45.0,
                       contact_point=None, yaws_deg=(0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0, 120.0, -120.0, 180.0), out_json=None):
    from tool_library.skills.flat_hand_pose import flat_hand_pose
    spec = H.load_scene_spec(scene)
    if mode not in ("auto", "drag", "push", "press"):
        raise ValueError("mode must be 'auto', 'drag', 'push' or 'press'")
    dvec = NP._horizontal(direction, "direction")
    model, d0, rig, ids = H.open_scene(spec)
    bid = H.grasp_body(model, spec, ids)
    bodies = NP.E.subtree_bodies(model, bid)
    gs = NP.hand_geoms(model, bodies)
    fp = NP.footprint(model, d0, bodies)
    mass = float(sum(model.body_mass[b] for b in bodies))
    mu_part = max(float(model.geom_friction[g][0]) for g in gs)
    fl = ids.get("floor_gid", -1)
    mu_sup = max(mu_part, float(model.geom_friction[fl][0]) if fl >= 0 else 1.0)
    need = NP.drag_force_need(mass, NP.HAND_MU, mu_sup)
    drag_ok = need is not None and 1.5 * need <= H.WRIST_FORCE_N
    dims = NP.hand_dims(model, rig)
    support_z = NP.support_under(model, d0, bodies, fp)[1] if fl < 0 else float(H.floor_surface(model, d0, fl)[0][2])
    h = fp["z_high"] - support_z
    push_deg = NP.push_contact_deg(dims, h)
    push_ok = np.tan(np.radians(push_deg)) <= NP.PUSH_SLOPE_FRAC / mu_sup
    use = mode if mode != "auto" else ("push" if push_ok else "drag")
    if use == "press" and contact_point is None:
        raise ValueError("mode 'press' needs contact_point (where on the part's top to press: a tilt_up presses its overhanging end)")
    info = {"direction": dvec[:2].round(6).tolist(), "mu_hand": NP.HAND_MU, "mu_support": round(mu_sup, 3), "mass_kg": round(mass, 4),
            "drag_force_n": None if need is None else round(need, 2), "drag_within_wrist_cap": bool(drag_ok),
            "height_m": round(h, 4), "push_contact_deg": round(push_deg, 1)}
    if use in ("drag", "press"):
        if need is None and use == "drag":
            raise ValueError(f"a drag cannot move it: the pads' friction {NP.HAND_MU} does not exceed the part-support friction {mu_sup:.2f}; use mode 'push'")
        com = np.asarray(d0.subtree_com[bid], float)
        xy = np.asarray(contact_point, float)[:2] if contact_point is not None else com[:2]
        z = NP.surface_z(model, d0, gs, xy, fp["z_high"] + 0.05)
        if z is None:
            raise ValueError("no top surface of the part at the contact point")
        from scipy.spatial.transform import Rotation as Rot
        fdir = -dvec if finger_dir is None else NP._horizontal(finger_dir, "finger_dir")
        # candidates: the fingers turned about the vertical from finger_dir and pitched steeper (an arm at a bench edge
        # holds a fingers-down hand far more often than a flat one); the executor screens them for its arm
        pitches = [float(pitch_deg)] + [p_ for p_ in (45.0, 60.0, 30.0) if abs(p_ - float(pitch_deg)) > 1e-6]
        poses = []
        if len(fingers) == 1:
            # one upright fingertip (a strip, a small part a flat hand would press beside, into the table): its distal cap
            # on the part's top at the contact point, the other fingers curled out of the way
            f = list(fingers)[0]
            so = H.pad_radius(model, rig, "index") + 0.001
            pt = {f: np.array([float(xy[0]), float(xy[1]), z])}; nn = {f: np.array([0.0, 0.0, 1.0])}
            poses, _ = NP.solve_sector_poses(spec, (model, d0, rig, ids), pt, nn, {f: pt[f] + nn[f] * so}, sectors=8, caps=True)
            for pz in poses:
                pz.update({"grip_fingers": [f], "approach": [0.0, 0.0, 1.0], "finger_targets_open": dict(pz["finger_targets"])})
            yaws_deg = ()
        for yaw in yaws_deg:
            fd = Rot.from_euler("z", float(yaw), degrees=True).as_matrix() @ fdir
            for pt in pitches:
                pz = flat_hand_pose(spec, [float(xy[0]), float(xy[1]), z], [0.0, 0.0, 1.0], fd.tolist(), fingers=list(fingers),
                                    contact="pad", thumb="auto", pitch_deg=pt)
                if float(pz.get("overlap_mm", 0.0) or 0.0) > 2.0:
                    continue
                pz["finger_yaw_deg"], pz["pitch_deg"] = float(yaw), pt
                poses.append(pz)
        if not poses:
            raise ValueError("no flat-hand drag pose stays out of the scene (overlap over 2 mm at every yaw and pitch)")
        pose = poses[0]
        info["why"] = ("flat fingertips pressing its top at the contact point (tilt_up)" if use == "press" else
                       f"pads pressed with at least {need:.1f} N drag it (friction {NP.HAND_MU} on the part vs {mu_sup:.2f} on the support)")
    else:
        if not push_ok:
            raise ValueError(f"a push cannot move it: a fingertip on its {1000 * h:.1f} mm side meets its top edge at {push_deg:.0f} deg, "
                             f"so it presses it down harder than it pushes (the support's friction is {mu_sup:.2f})"
                             + ("" if need is None else ": use mode 'drag'"))
        poses = _rake_poses(model, d0, rig, ids, bodies, fp, dvec, support_z, dims)
        if not poses:
            raise ValueError("no fingertips-behind-the-part pose stays out of the scene: use mode 'drag'")
        pose = poses[0]
        info["why"] = "upright fingertips behind the part push its trailing face (or its top edge, on a part lower than a pad radius)"
    for pz in poses:
        pz["slide_mode"] = use
        pz["slide"] = info
    pose = dict(pose, poses=poses)
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(pose, indent=1, default=float))
    return pose


def _rake_poses(model, d0, rig, ids, bodies, fp, dvec, support_z, dims, sectors=4, fingers=("index", "middle"), spread=0.019,
                max_residual_mm=3.0, max_overlap_mm=2.0):
    """Upright index and middle fingertips just behind the part's trailing face (the side facing away from the slide),
    their caps a millimetre above the table, the face (or its top edge, when lower) in front of them: pushing along
    dvec rakes the part ahead. One candidate per sector of wrist heading."""
    rel = (fp["points"][:, :2] - fp["centre"]) @ dvec[:2]
    back = float(rel.min())
    side = np.array([-dvec[1], dvec[0], 0.0])
    com = np.asarray(d0.subtree_com[int(min(bodies))], float)
    base = fp["centre"] + dvec[:2] * back
    # centre the push on the line of the slide through the centre of mass (no turning moment)
    base = base + side[:2] * float((com[:2] - base) @ side[:2])
    zc = support_z + dims["pad_radius_m"] + NP.TABLE_CLEAR_M
    pts, nrm = {}, {}
    for i, f in enumerate(fingers):
        o = (i - 0.5 * (len(fingers) - 1)) * spread
        xy = base + side[:2] * o
        # the distal cap a pad radius behind the face, its centre a pad radius (+1 mm) above the table: on a face lower
        # than that it meets the top edge as it pushes
        pts[f] = np.array([xy[0], xy[1], zc])
        nrm[f] = -dvec
    tg = {f: pts[f] + nrm[f] * (H.pad_radius(model, rig, "index") + 0.001) for f in pts}
    spec = {"_built": True}
    cands, _ = NP.solve_sector_poses(spec, (model, d0, rig, ids), pts, nrm, tg, approach_dirs=((0.0, 0.0, 1.0),), sectors=sectors,
                                     max_residual_mm=max_residual_mm, max_overlap_mm=max_overlap_mm, caps=True)
    out = []
    for pose in cands:
        pose.update({"grip_fingers": list(fingers), "normals": {f: v.tolist() for f, v in nrm.items()}, "approach": [0.0, 0.0, 1.0],
                     "finger_targets_open": dict(pose["finger_targets"])})
        out.append(pose)
    out.sort(key=lambda p: (round(p["worst_residual_mm"]), p["overlap_mm"]))
    return out
