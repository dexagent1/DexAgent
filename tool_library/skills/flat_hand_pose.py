"""A flat hand laid on a plane: straight fingers whose pads or backs lie on a surface."""
from __future__ import annotations

TOOL = {
    "name": "flat_hand_pose",
    "title": "Flat hand pose",
    "type": "skill",
    "category": "Grasp & Poses",
    "summary": "Lay a hand with straight fingers on a plane: the fingers point along a given direction in the plane and their "
               "pads (palm toward the surface) or their backs (palm away: fingers slid under a lid) touch it at a contact point; "
               "no inverse kinematics, the wrist is placed from the hand's measured finger thickness.",
    "description": (
        "Some contacts are made with the flat of the hand, not with a grasp: fingers slid under a lid near its hinge to "
        "lift it, a palm pressing a box flap shut, fingertips pushing a drawer front, a hand sweeping something across the "
        "table. Grasp solvers put pads on points and bend the fingers to do it; this skill keeps the fingers straight "
        "(every flexion joint at its extended end, abduction zero) and places the wrist so the chosen face of the named "
        "fingers is tangent to the plane through `point` with outward normal `normal`, the fingers pointing along "
        "`finger_dir` projected into the plane (the direction from the entry edge toward where the contact is). "
        "`contact` 'pad' faces the palm to the surface; 'back' faces it away (the lid rests on the backs of the fingers, "
        "the hand stays palm down and in the arm's reach). The thumb is 'out' (straight, in the hand's plane), 'curled' "
        "(folded under the palm; needs room below), 'up' (raised) or 'auto' (whichever penetrates the scene least). "
        "`pitch_deg` tilts the hand about the fingertips' line so only the fingertips touch and the thicker palm clears "
        "the table (a lid 35 mm above the table takes 18 mm fingers but not a 45 mm palm); a pitched hand is centred by the "
        "pads it really touches with (measured under the fingertips, about 10 mm behind the tip sites at 25 deg), so a press "
        "goes through `point` (a point contact transmits no torque: a press off the part's centre tips it). The pose carries grip_fingers, its own open shape (the straight hand, so the planner never curls "
        "the fingers into the table before the approach), points and normals (the contact under each finger) for the outcome's pad check, an approach along the normal when that space is free (a press comes from above) and "
        "else along the surface (fingers back out from under a lid), contact_side, and press_travel_m (4 mm of pad curl for a pad contact; none for a back contact, which the wrist's motion holds). "
        "Verify it like any pose (penetration against the scene, reach with optimize_pose_for_reach / arm_reach_check): "
        "the skill knows the plane, not the rest of the scene."
    ),
    "inputs": [
        {"name": "scene", "type": "dict | str", "description": "Hand scene from hand_scene_from_recon (hand_xml, side, objects...)."},
        {"name": "point", "type": "list", "description": "World contact point on the surface (under the middle of the named fingers)."},
        {"name": "normal", "type": "list", "description": "Outward surface normal at the point (for a lid's underside: pointing down)."},
        {"name": "finger_dir", "type": "list", "description": "World direction the fingers point (projected into the plane)."},
        {"name": "fingers", "type": "list", "description": "Fingers expected to touch (the outcome's pad check). Default ['index', 'middle', 'ring', 'pinky']; the face is always the whole hand's."},
        {"name": "contact", "type": "str", "description": "'pad' (palm toward the surface) or 'back' (palm away). Default 'pad'."},
        {"name": "thumb", "type": "str", "description": "'out' | 'curled' | 'up' | 'auto' (the mode that penetrates the scene least). Default 'auto'."},
        {"name": "pitch_deg", "type": "float", "description": "Tilt about the fingertips' line: positive raises the wrist (world up) so only the fingertips touch and the palm clears the table (a hand that does not fit flat under a low lid, a press on a tray's floor). Default 0."},
        {"name": "clearance_m", "type": "float", "description": "Gap left between the finger face and the plane. Default 0.001."},
        {"name": "out_json", "type": "str | None", "description": "Write the pose here."},
    ],
    "outputs": [
        {"name": "pose", "type": "dict", "description": "{side, pos, euler, finger_targets, grip_fingers, points, normals, approach, contact_side, press_travel_m, face_offset_m, thumb, pitch_deg, overlap_mm (hand vs scene at the pose)}: the pose dict every planner and verifier takes."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["pose", "flat hand", "push", "lid", "non-prehensile"],
    "used_in": ["Giftbox Packing", "Place Toy in Drawer"],
    "status": "tested",
    "entrypoint": "flat_hand_pose",
}

import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from tool_library.common import hand as H


def _unit(v):
    v = np.asarray(v, float); return v / max(np.linalg.norm(v), 1e-9)


def straight_targets(rig, thumb="out"):
    """Every flexion joint at its extended end, abduction zero; the thumb out, curled or raised."""
    t = {}
    for k, (lo, hi) in rig.finger_range.items():
        role = H.joint_role(k); fin = H.finger_of_joint(k)
        if fin == "thumb":
            if role == "thumb_oppose":
                t[k] = float(lo + (0.0 if thumb == "out" else 0.55 if thumb == "curled" else 1.0) * (hi - lo))
            elif role == "flex":
                t[k] = float(lo + (0.0 if thumb != "curled" else 0.6) * (hi - lo))
            else:
                t[k] = float(np.clip(0.0, min(lo, hi), max(lo, hi)))
        elif role in ("flex", "pinky_arch"):
            t[k] = float(lo)
        else:
            t[k] = float(np.clip(0.0, min(lo, hi), max(lo, hi)))
    return t


def flat_hand_pose(scene, point, normal, finger_dir, fingers=("index", "middle", "ring", "pinky"), contact="pad", thumb="auto", pitch_deg=0.0,
                   clearance_m=0.001, out_json=None):
    if isinstance(scene, (str, Path)):
        scene = json.loads(Path(scene).read_text())
    model, data0, rig, ids = H.open_scene(scene)
    side = rig.side
    for f in fingers:
        if f not in rig.fingertip_sites:
            raise ValueError(f"finger {f!r} not in the hand ({sorted(rig.fingertip_sites)})")
    n = _unit(normal); u = -n                                       # u: from the hand toward the surface
    fdir = np.asarray(finger_dir, float); fdir = _unit(fdir - n * (n @ fdir))
    palm = u if contact == "pad" else -u
    R0 = np.column_stack([palm, np.cross(fdir, palm), fdir])         # local x = palm normal, z = fingers
    # pitch about the fingertips' line (local y): positive raises the wrist (world up) so only the fingertips touch
    # and the thicker palm clears the table (a hand that does not fit flat under a low lid, a press inside a tray)
    R = R0
    if pitch_deg:
        cands = [Rot.from_rotvec(np.radians(float(pitch_deg)) * sgn * R0[:, 1]).as_matrix() @ R0 for sgn in (1.0, -1.0)]
        R = max(cands, key=lambda Rc: float(-(Rc @ np.array([0.0, 0.0, 0.19]))[2]))   # the wrist sits at -z_local*0.19 from the tips: pick the higher one
    # the face is the whole hand's: every non-thumb finger (a pinky that sits lower than the ring touches first)
    face_fingers = [f for f in rig.fingertip_sites if f != "thumb"]
    tip_bodies = {int(model.site_bodyid[rig.fingertip_sites[f]]) for f in face_fingers}
    fin_bodies = {b for b in range(model.nbody) if any(f"_{f}_" in model.body(b).name or model.body(b).name.endswith(f"_{f}") for f in face_fingers)} | tip_bodies
    contact_bodies = tip_bodies if pitch_deg else fin_bodies

    def place(thumb_mode):
        targets = straight_targets(rig, thumb_mode)
        d = mujoco.MjData(model); d.qpos[:] = data0.qpos
        rig.teleport(d, np.zeros(3), np.zeros(3)); rig.set_fingers(d, targets); mujoco.mj_forward(model, d)
        # geometry in the wrist frame (wrist at the origin, rotation R applied)
        ext = -np.inf
        for g in rig.collision_gids:
            if int(model.geom_bodyid[g]) not in contact_bodies:
                continue
            c = R @ d.geom_xpos[g]; r = float(model.geom_size[g][0]); typ = int(model.geom_type[g])
            if typ == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
                ax = R @ d.geom_xmat[g].reshape(3, 3)[:, 2]; hl = float(model.geom_size[g][1])
                ext = max(ext, ((c + ax * hl) @ u) + r, ((c - ax * hl) @ u) + r)
            else:
                ext = max(ext, (c @ u) + float(model.geom_rbound[g]))
        sites = {f: R @ d.site_xpos[rig.fingertip_sites[f]] for f in fingers}
        cen = np.mean(list(sites.values()), axis=0); cen_ip = cen - u * (cen @ u)
        wrist = np.asarray(point, float) + n * clearance_m - u * ext - cen_ip
        return targets, wrist, sites, ext

    modes = [thumb] if thumb != "auto" else ["out", "up", "curled"]
    best = None
    for mode in modes:
        targets, wrist, sites, ext = place(mode)
        # score the thumb mode by the hand's penetration of the scene (objects and floor) at the pose
        d = mujoco.MjData(model); d.qpos[:] = data0.qpos
        rig.teleport(d, wrist, H.mat_to_euler(R)); rig.set_fingers(d, targets); mujoco.mj_forward(model, d)
        pen = H.deepest_overlap(model, d, rig, ids, keep_tips=())
        if best is None or pen < best[0]:
            best = (pen, mode, targets, wrist, sites, ext)
    pen, mode, targets, wrist, sites, ext = best
    # seat the hand by MEASURED geometry: the extent estimate above (capsule ends, bounding spheres) left a flat hand's
    # pads 9-14 mm off a box top or wall (measured), so the hand is slid along the normal until the deepest finger
    # geometry sits `clearance_m` off the plane
    def _height(dd, g):
        typ = int(model.geom_type[g]); c = dd.geom_xpos[g]; Rg = dd.geom_xmat[g].reshape(3, 3); sz = model.geom_size[g]
        if typ == int(mujoco.mjtGeom.mjGEOM_MESH):
            mid = int(model.geom_dataid[g]); a = int(model.mesh_vertadr[mid]); nv = int(model.mesh_vertnum[mid])
            v = model.mesh_vert[a:a + nv] @ Rg.T + c
            return float(np.min((v - point_w) @ n))
        if typ == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            return float((c - point_w) @ n - abs(Rg[:, 2] @ n) * sz[1] - sz[0])
        if typ == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            return float((c - point_w) @ n - sz[0])
        if typ == int(mujoco.mjtGeom.mjGEOM_BOX):
            return float((c - point_w) @ n - np.sum(np.abs(Rg.T @ n) * sz[:3]))
        return float((c - point_w) @ n - model.geom_rbound[g])
    def _low_point(dd, g):
        """The point of geom g lowest along -n (where it meets the plane first)."""
        typ = int(model.geom_type[g]); c = dd.geom_xpos[g]; Rg = dd.geom_xmat[g].reshape(3, 3); sz = model.geom_size[g]
        if typ == int(mujoco.mjtGeom.mjGEOM_MESH):
            mid = int(model.geom_dataid[g]); a = int(model.mesh_vertadr[mid]); nv = int(model.mesh_vertnum[mid])
            v = model.mesh_vert[a:a + nv] @ Rg.T + c
            return v[int(np.argmin(v @ n))]
        if typ == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            ax = Rg[:, 2] * float(sz[1]); end = c - ax if (ax @ n) > 0 else c + ax
            return end - n * float(sz[0])
        if typ == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            return c - n * float(sz[0])
        if typ == int(mujoco.mjtGeom.mjGEOM_BOX):
            return c - Rg @ (np.sign(Rg.T @ n) * sz[:3])
        return c - n * float(model.geom_rbound[g])
    point_w = np.asarray(point, float)
    seat_geoms = [g for g in rig.collision_gids if any(f"_{f}_" in (model.geom(g).name or "") for f in fingers if f != "thumb")]
    gap_before = None
    contact_pts = None
    if seat_geoms:
        dd = mujoco.MjData(model); dd.qpos[:] = data0.qpos
        rig.teleport(dd, wrist, H.mat_to_euler(R)); rig.set_fingers(dd, targets); mujoco.mj_forward(model, dd)
        gap_before = min(_height(dd, g) for g in seat_geoms)
        wrist = wrist - n * (gap_before - float(clearance_m))
        rig.teleport(dd, wrist, H.mat_to_euler(R)); mujoco.mj_forward(model, dd)
        if pitch_deg:
            # a pitched hand touches with the pads UNDER its fingertips, not at the tip sites: measured 10 mm along the
            # fingers on a 25 deg pitch. With realistic contact a press through a point off the part's centre tips the
            # part (a stud brick pressed at its edge jammed tilted 1.9 deg), so the measured contact points, not the
            # sites, are centred on `point`; the pose's points are those contacts
            contact_pts = {}
            for f in fingers:
                gs = [g for g in seat_geoms if f"_{f}_" in (model.geom(g).name or "")]
                if gs:
                    lows = [_low_point(dd, g) for g in gs]
                    contact_pts[f] = lows[int(np.argmin([q @ n for q in lows]))]
            if contact_pts:
                cen_c = np.mean(list(contact_pts.values()), axis=0)
                shift = (point_w - cen_c) - n * ((point_w - cen_c) @ n)
                wrist = wrist + shift
                contact_pts = {f: q + shift for f, q in contact_pts.items()}
                rig.teleport(dd, wrist, H.mat_to_euler(R)); mujoco.mj_forward(model, dd)
        pen = H.deepest_overlap(model, dd, rig, ids, keep_tips=())
    # the approach: along the outward normal when that space is free (a press on a floor comes from above), else
    # along the surface, backing out along the fingers (fingers slid under a lid: the normal points into the table)
    d = mujoco.MjData(model); d.qpos[:] = data0.qpos
    def pen_at(offset):
        rig.teleport(d, wrist + offset, H.mat_to_euler(R)); rig.set_fingers(d, targets); mujoco.mj_forward(model, d)
        return H.deepest_overlap(model, d, rig, ids, keep_tips=())
    clear_normal = all(pen_at(n * t) <= 0.002 for t in (0.03, 0.06, 0.1))
    approach = n if clear_normal else -fdir
    pose = {"side": side, "pos": wrist.tolist(), "euler": H.mat_to_euler(R).tolist(), "finger_targets": targets,
            "finger_targets_open": dict(targets),          # the pre-grasp shape is the straight hand itself: a curl near a table jams the wrist
            "grip_fingers": list(fingers), "contact_side": contact, "thumb": mode, "pitch_deg": float(pitch_deg),
            "press_travel_m": 0.004 if contact == "pad" else 0.0,        # pads can curl into the surface; backs cannot
            "face_offset_m": round(float(ext), 4), "overlap_mm": round(1000 * float(pen), 1),
            "approach": approach.tolist(), "approach_kind": "normal" if clear_normal else "along_surface",
            "gap_before_m": None if gap_before is None else round(float(gap_before), 4),
            "normals": {f: n.tolist() for f in fingers}}
    pts = {}
    for f in fingers:
        s_w = contact_pts[f] if contact_pts and f in contact_pts else wrist + sites[f]
        pts[f] = (s_w - n * ((s_w - np.asarray(point, float)) @ n)).tolist()
    pose["points"] = pts
    if out_json:
        Path(out_json).write_text(json.dumps(pose, indent=1)); pose["out_json"] = str(out_json)
    return pose
