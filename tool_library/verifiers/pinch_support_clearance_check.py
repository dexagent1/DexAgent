"""Static pre-physics check of a grasp pose on an object that lies on a support: does the closed hand fit above the
support, and where do the grip pads end once the support has pushed the hand up?"""
TOOL = {
    "name": "pinch_support_clearance_check",
    "title": "Pinch support clearance check",
    "type": "verifier",
    "category": "Grasp Stress Tests",
    "summary": "Pose a hand with its closed fingers in a floating-hand scene and measure how deep its collision geoms sink into "
               "the support (floor) and the other objects, then where the grip pads end once the support pushes the hand up "
               "by that depth: above the object, above a round object's centre line, or where they can hold it.",
    "description": (
        "Pose solvers that place pads on a part (handle_pinch_pose, pose_from_points) check the hand against the object "
        "but not against the support it lies on, so a pinch of a thin object lying on a table (a pen, a rod, a cable, a "
        "card) can put the fingertips centimetres below the table top with a zero overlap report. In physics the support "
        "stops the hand that much higher, the pads rise with it and close above the object's centre line (the squeeze "
        "presses the object into the table and the lift leaves it behind) or above the object altogether. This verifier "
        "applies the pose with its closed finger targets, measures the deepest penetration of every hand collision geom "
        "into the floor of the hand scene and into objects other than the grasped one (support_mm), and the deepest "
        "penetration of non-grip geoms into the grasped object (object_mm). It then shifts the grip pads up by support_mm "
        "(the vertical push of the support) and compares their heights with the grasped object's world box: a pad above "
        "the object's top cannot touch it; for a round cross-section (round_section=True: rods, pens, cables, cylinders "
        "lying on their side) a pad more than r*sin(atan(mu)) above the axis squeezes the object down instead of lifting "
        "it. Passes when support_mm and object_mm are within tol_mm and every grip pad can still hold. It does not replace "
        "the physics stress tests; it says before physics why one will fail and by how much."),
    "inputs": [
        {"name": "scene", "type": "dict | str", "description": "Floating-hand scene (hand_scene_from_recon output or its json); the last object (or grasp_body) is the grasped one."},
        {"name": "pose", "type": "dict | str", "description": "Hand pose {pos, euler, finger_targets, grip_fingers} or a json path holding it."},
        {"name": "grip_fingers", "type": "list | None", "description": "Fingers whose pads must hold (default the pose's grip_fingers, else thumb and index)."},
        {"name": "tol_mm", "type": "float", "description": "Allowed penetration into the support and of non-grip geoms into the object, mm. Default 3.0."},
        {"name": "round_section", "type": "bool", "description": "The grasped part is round in the pinch plane (a rod or pen on its side): apply the centre-line rule. Default False."},
        {"name": "mu", "type": "float", "description": "Pad-object friction coefficient for the centre-line rule. Default 0.85."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; support_mm (deepest hand penetration into the floor or other objects) and support_pair; object_mm and object_pair; support_lift_m; pads {finger: {z_m, effective_z_m, above_top, above_hold_line}}; object {lo_z, hi_z, centre_z, radius_m}; hold_line_z_m (round sections)."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["grasp", "pinch", "thin object", "support", "table", "pen", "rod", "cable", "pre-physics", "stress test"],
    "used_in": ["Drawing and Wiping"],
    "status": "tested",
    "entrypoint": "pinch_support_clearance_check",
}

import json
from pathlib import Path

import mujoco
import numpy as np

from tool_library.common import hand as H


def _load(x):
    if isinstance(x, (str, Path)):
        d = json.loads(Path(x).read_text())
        if isinstance(d, dict) and "pose" in d and "pos" not in d:
            d = d["pose"]
        if isinstance(d, list):
            d = d[0]
        return d
    return x


def pinch_support_clearance_check(scene, pose, grip_fingers=None, tol_mm=3.0, round_section=False, mu=0.85):
    if float(tol_mm) < 0:
        raise ValueError("tol_mm must be non-negative")
    if float(mu) <= 0:
        raise ValueError("mu must be positive")
    pose = _load(pose)
    if not isinstance(pose, dict) or "pos" not in pose or "euler" not in pose:
        raise ValueError("pose must be a dict (or json) with pos and euler")
    spec = H.load_scene_spec(scene)
    model, data0, rig, ids = H.open_scene(spec)
    d = mujoco.MjData(model)
    d.qpos[:] = data0.qpos
    mujoco.mj_forward(model, d)
    H.apply_pose(model, d, rig, pose)
    mujoco.mj_forward(model, d)
    grip = list(grip_fingers or pose.get("grip_fingers") or ["thumb", "index"])
    unknown = [f for f in grip if f not in rig.fingertip_sites]
    if unknown:
        raise ValueError(f"grip_fingers {unknown} are not fingers of this hand ({sorted(rig.fingertip_sites)})")
    grasp_gids = set(ids["grasp_gids"])
    others = [g for g in ids["obj_gids"] if g not in grasp_gids]
    supports = ([ids["floor_gid"]] if ids.get("floor_gid", -1) >= 0 else []) + others
    pad_gids = set()
    for f in grip:
        pad_gids |= set(rig.tip_geoms(f))
    sup_mm, sup_pair, obj_mm, obj_pair = 0.0, None, 0.0, None
    for g in sorted(rig.collision_gids):
        for og in supports:
            pen = -H.hand_geom_distance(model, d, g, og, 0.0) * 1000.0
            if pen > sup_mm:
                sup_mm, sup_pair = pen, [model.geom(g).name, model.geom(og).name]
        if g in pad_gids:
            continue
        for og in grasp_gids:
            pen = -H.hand_geom_distance(model, d, g, og, 0.0) * 1000.0
            if pen > obj_mm:
                obj_mm, obj_pair = pen, [model.geom(g).name, model.geom(og).name]
    ext = H.object_extent(model, d, sorted(grasp_gids))
    lo_z, hi_z = float(ext["lo"][2]), float(ext["hi"][2])
    centre_z = 0.5 * (lo_z + hi_z)
    radius = 0.5 * (hi_z - lo_z)
    lift = max(0.0, sup_mm) / 1000.0
    hold_line = centre_z + radius * np.sin(np.arctan(float(mu))) if round_section else hi_z
    pads, issues = {}, []
    for f in grip:
        z = float(d.site_xpos[rig.fingertip_sites[f]][2])
        ez = z + lift
        above_top = ez > hi_z
        above_line = bool(round_section) and ez > hold_line
        pads[f] = {"z_m": round(z, 4), "effective_z_m": round(ez, 4), "above_top": bool(above_top), "above_hold_line": bool(above_line)}
        if above_top:
            issues.append(f"{f} pad ends at z={ez:.4f} m once the support lifts the hand {sup_mm:.1f} mm, above the object's top "
                          f"({hi_z:.4f} m): it cannot touch it; grasp a part that stands clear of the support or approach from the side")
        elif above_line:
            issues.append(f"{f} pad ends at z={ez:.4f} m, above the hold line {hold_line:.4f} m of the round section (axis {centre_z:.4f} m): "
                          f"the squeeze presses the object into the support and the lift leaves it behind")
    if sup_mm > float(tol_mm):
        issues.insert(0, f"closed hand is {sup_mm:.1f} mm inside the support ({sup_pair[0]} in {sup_pair[1]}); in physics the support "
                         f"stops the hand {sup_mm:.1f} mm higher")
    if obj_mm > float(tol_mm):
        issues.append(f"non-grip geom {obj_pair[0]} is {obj_mm:.1f} mm inside the grasped object ({obj_pair[1]})")
    return {"ok": not issues, "issues": issues, "support_mm": round(sup_mm, 2), "support_pair": sup_pair,
            "object_mm": round(obj_mm, 2), "object_pair": obj_pair, "support_lift_m": round(lift, 4), "pads": pads,
            "object": {"lo_z": round(lo_z, 4), "hi_z": round(hi_z, 4), "centre_z": round(centre_z, 4), "radius_m": round(radius, 4)},
            "hold_line_z_m": round(float(hold_line), 4), "grip_fingers": grip}
