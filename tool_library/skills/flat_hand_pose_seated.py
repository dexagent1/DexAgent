TOOL = {
    "name": "flat_hand_pose_seated",
    "title": "Flat hand pose, seated on the surface",
    "type": "skill",
    "category": "Grasp & Poses",
    "summary": "Lay a hand with straight fingers on a plane like flat_hand_pose, then seat it: measure the real gap between the "
               "fingers' collision geometry (mesh hulls included) and the plane and slide the wrist along the normal until the "
               "fingers sit at the requested clearance.",
    "description": (
        "flat_hand_pose places the wrist from each finger geom's bounding-sphere radius when the geom is not a capsule; for a "
        "hand whose fingers collide as convex mesh hulls that radius is far larger than the hull's thickness, so the pads end "
        "10-15 mm off the surface and a press or a hold never touches. This skill builds the same orientation (palm toward the "
        "surface for 'pad', away for 'back'; fingers along finger_dir projected into the plane; optional pitch about the "
        "fingertips' line; thumb out, curled, up or the least penetrating) and then measures, for every collision geom of the "
        "face (all non-thumb finger links, or only the fingertip links when pitched), its exact support along the direction "
        "into the surface: mesh vertices, box corners, sphere, capsule and cylinder analytically, the bounding sphere only for "
        "other types. The wrist is moved along the normal so the deepest of them sits clearance_m outside the plane. It "
        "assumes the plane is the contact surface near `point` (a face larger than the hand's footprint); it does not check "
        "reach (use optimize_pose_for_reach) or physics (execute the subgoal). Returns the pose dict every planner takes, with "
        "gap_before_m (where the unseated placement would have left the fingers) and the measured gap after seating."
    ),
    "inputs": [
        {"name": "scene", "type": "dict | str", "description": "Hand scene from hand_scene_from_recon (hand_xml, side, objects...)."},
        {"name": "point", "type": "list", "description": "World contact point on the surface (under the middle of the named fingers)."},
        {"name": "normal", "type": "list", "description": "Outward surface normal at the point."},
        {"name": "finger_dir", "type": "list", "description": "World direction the fingers point (projected into the plane)."},
        {"name": "fingers", "type": "list", "description": "Fingers expected to touch. Default ['index', 'middle', 'ring', 'pinky']."},
        {"name": "contact", "type": "str", "description": "'pad' (palm toward the surface) or 'back' (palm away). Default 'pad'."},
        {"name": "thumb", "type": "str", "description": "'out' | 'curled' | 'up' | 'auto'. Default 'auto'."},
        {"name": "pitch_deg", "type": "float", "description": "Tilt about the fingertips' line raising the wrist; then only the fingertip links form the face. Default 0."},
        {"name": "clearance_m", "type": "float", "description": "Gap left between the deepest finger geometry and the plane after seating. Default 0.0005."},
        {"name": "out_json", "type": "str | None", "description": "Write the pose here."},
    ],
    "outputs": [
        {"name": "pose", "type": "dict", "description": "{side, pos, euler, finger_targets, finger_targets_open, grip_fingers, points, normals, approach, approach_kind, contact_side, press_travel_m, thumb, pitch_deg, overlap_mm (hand vs scene, plane contact excluded by the clearance), gap_before_m, gap_m, seated_shift_m}."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["pose", "flat hand", "push", "lid", "hold", "non-prehensile", "seating"],
    "used_in": [],
    "status": "tested",
    "entrypoint": "flat_hand_pose_seated",
}

import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from tool_library.common import hand as H

_CONTACTS = ("pad", "back")
_THUMBS = ("out", "curled", "up", "auto")


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError("zero-length direction: give a non-zero `normal` and `finger_dir`")
    return v / n


def _straight_targets(rig, thumb):
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


def _support(model, data, g, u):
    """Largest coordinate of geom g along unit direction u (world), exact for meshes and primitives."""
    typ = int(model.geom_type[g]); c = data.geom_xpos[g]; R = data.geom_xmat[g].reshape(3, 3); s = model.geom_size[g]
    if typ == int(mujoco.mjtGeom.mjGEOM_MESH):
        mid = int(model.geom_dataid[g]); a = int(model.mesh_vertadr[mid]); nv = int(model.mesh_vertnum[mid])
        v = model.mesh_vert[a:a + nv]
        return float(np.max((v @ R.T + c) @ u))
    if typ == int(mujoco.mjtGeom.mjGEOM_BOX):
        return float(c @ u + np.sum(np.abs(R.T @ u) * s[:3]))
    if typ == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return float(c @ u + s[0])
    if typ == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        return float(c @ u + abs(R[:, 2] @ u) * s[1] + s[0])
    if typ == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        a = abs(R[:, 2] @ u)
        return float(c @ u + a * s[1] + s[0] * np.sqrt(max(0.0, 1.0 - a * a)))
    return float(c @ u + model.geom_rbound[g])


def flat_hand_pose_seated(scene, point, normal, finger_dir, fingers=("index", "middle", "ring", "pinky"), contact="pad", thumb="auto",
                          pitch_deg=0.0, clearance_m=0.0005, out_json=None):
    if contact not in _CONTACTS:
        raise ValueError(f"contact must be one of {_CONTACTS}, got {contact!r}")
    if thumb not in _THUMBS:
        raise ValueError(f"thumb must be one of {_THUMBS}, got {thumb!r}")
    if point is None or len(point) != 3:
        raise ValueError("point must be a world xyz list")
    if isinstance(scene, (str, Path)):
        scene = json.loads(Path(scene).read_text())
    model, data0, rig, ids = H.open_scene(scene)
    fingers = list(fingers)
    for f in fingers:
        if f not in rig.fingertip_sites:
            raise ValueError(f"finger {f!r} not in the hand ({sorted(rig.fingertip_sites)})")
    n = _unit(normal); u = -n
    fdir = np.asarray(finger_dir, float); fdir = fdir - n * (n @ fdir)
    if np.linalg.norm(fdir) < 1e-6:
        raise ValueError("finger_dir is parallel to the normal: give a direction that lies in the surface")
    fdir = _unit(fdir)
    palm = u if contact == "pad" else -u
    R0 = np.column_stack([palm, np.cross(fdir, palm), fdir])
    R = R0
    if pitch_deg:
        cands = [Rot.from_rotvec(np.radians(float(pitch_deg)) * sgn * R0[:, 1]).as_matrix() @ R0 for sgn in (1.0, -1.0)]
        R = max(cands, key=lambda Rc: float(-(Rc @ np.array([0.0, 0.0, 0.19]))[2]))
    eul = H.mat_to_euler(R)
    face_fingers = [f for f in rig.fingertip_sites if f != "thumb"]
    tip_bodies = {int(model.site_bodyid[rig.fingertip_sites[f]]) for f in face_fingers}
    fin_bodies = {b for b in range(model.nbody) if any(f"_{f}_" in model.body(b).name or model.body(b).name.endswith(f"_{f}") for f in face_fingers)} | tip_bodies
    face_bodies = tip_bodies if pitch_deg else fin_bodies
    face_gids = [g for g in rig.collision_gids if int(model.geom_bodyid[g]) in face_bodies]
    if not face_gids:
        raise ValueError("the hand has no collision geoms on its finger links: check hand_xml")
    plane_c = float(np.asarray(point, float) @ u)

    def pose_at(mode, wrist):
        targets = _straight_targets(rig, mode)
        d = mujoco.MjData(model); d.qpos[:] = data0.qpos
        rig.teleport(d, wrist, eul); rig.set_fingers(d, targets); mujoco.mj_forward(model, d)
        deepest = max(_support(model, d, g, u) for g in face_gids)
        return targets, d, deepest

    modes = [thumb] if thumb != "auto" else ["out", "up", "curled"]
    best = None
    for mode in modes:
        # start where the fingertip sites' centroid sits on the plane, then seat by the measured support
        targets, d, _ = pose_at(mode, np.zeros(3))
        cen = np.mean([d.site_xpos[rig.fingertip_sites[f]] for f in fingers], axis=0)
        wrist = np.asarray(point, float) - cen
        targets, d, deepest = pose_at(mode, wrist)
        gap_before = plane_c - deepest
        wrist = wrist + u * (gap_before - clearance_m)
        targets, d, deepest = pose_at(mode, wrist)
        pen = H.deepest_overlap(model, d, rig, ids, keep_tips=())
        cand = (float(pen), mode, targets, wrist, float(gap_before), float(plane_c - deepest), d)
        if best is None or cand[0] < best[0] - 1e-4:
            best = cand
    pen, mode, targets, wrist, gap_before, gap_after, d = best
    sites = {f: d.site_xpos[rig.fingertip_sites[f]].copy() for f in fingers}

    dd = mujoco.MjData(model); dd.qpos[:] = data0.qpos
    def pen_at(offset):
        rig.teleport(dd, wrist + offset, eul); rig.set_fingers(dd, targets); mujoco.mj_forward(model, dd)
        return H.deepest_overlap(model, dd, rig, ids, keep_tips=())
    clear_normal = all(pen_at(n * t) <= 0.002 for t in (0.03, 0.06, 0.1))
    approach = n if clear_normal else -fdir
    pts = {f: (s - n * ((s - np.asarray(point, float)) @ n)).tolist() for f, s in sites.items()}
    pose = {"side": rig.side, "pos": np.asarray(wrist).tolist(), "euler": np.asarray(eul).tolist(), "finger_targets": targets,
            "finger_targets_open": dict(targets), "grip_fingers": fingers, "contact_side": contact, "thumb": mode,
            "pitch_deg": float(pitch_deg), "press_travel_m": 0.004 if contact == "pad" else 0.0,
            "overlap_mm": round(1000 * pen, 1), "approach": np.asarray(approach).tolist(),
            "approach_kind": "normal" if clear_normal else "along_surface",
            "normals": {f: n.tolist() for f in fingers}, "points": pts,
            "gap_before_m": round(gap_before, 4), "gap_m": round(gap_after, 4),
            "seated_shift_m": round(gap_before - clearance_m, 4)}
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(pose, indent=1)); pose["out_json"] = str(out_json)
    return pose
