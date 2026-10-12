TOOL = {
    "name": "optimize_fingertip_contacts",
    "title": "Cinch fingertips onto the object",
    "type": "skill",
    "category": "Grasp & Poses",
    "summary": "Curl each finger of a posed hand inward until its pad just touches the object, producing the closed finger targets a grasp is driven to.",
    "description": "Starting from a solved pose, every finger's flexion joints advance in small steps toward the hand's closed synergy while the true gap between that finger's tip geoms and the object is measured with mj_geomDistance (the hand assets ship with contacts off, so the contact list cannot be used). A finger stops at first touch (0.8 mm of interference), and any step that would go deeper than 1.8 mm or that makes the gap worse is reverted: past a point the flexion arc carries the tip back toward the palm, away from the surface. Abduction joints are never moved, so the cinch curls and never spreads the fingers. Optionally also returns a preloaded variant solved with a deeper target interference, which is the correct way to squeeze a pinch (nudging joint angles toward a limit is not).",
    "inputs": [
        {"name": "scene", "type": "dict", "description": "{hand_xml, side, objects, floor}; the last object is the one the fingers close on"},
        {"name": "pose", "type": "dict", "description": "a pose from pose_from_points / handle_pinch_pose / generate_rigid_grasps"},
        {"name": "fingers", "type": "list", "description": "fingers to cinch (default: all with tip sites)"},
        {"name": "touch_mm", "type": "float", "description": "target interference per pad (default 0.8)"},
        {"name": "max_pen_mm", "type": "float", "description": "never let a pad go deeper than this (default 1.8)"},
        {"name": "squeeze_mm", "type": "float", "description": "if > 0, also produce finger_targets_squeezed cinched to this deeper interference (default 2.5)"},
        {"name": "step_rad, max_steps", "type": "numbers", "description": "cinch step and budget per finger"},
    ],
    "outputs": [
        {"name": "pose", "type": "dict", "description": "the input pose with finger_targets updated, finger_targets_squeezed added, and cinch: {finger: {curled_rad, gap_mm}}"},
    ],
    "status": "tested",
    "entrypoint": "optimize_fingertip_contacts",
    "requirements": ["mujoco>=3.1", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["grasp", "contact", "finger-closure"],
    "used_in": ["Place Cup in Bowl", "Knot the Rope", "Battery Insertion", "Open the Bottle", "Scissor Cutting"],
}

import copy


def optimize_fingertip_contacts(scene, pose, fingers=None, touch_mm=0.8, max_pen_mm=1.8,
                                squeeze_mm=2.5, step_rad=0.012, max_steps=90):
    import mujoco
    from tool_library.common import hand as H

    model, data0, rig, ids = H.open_scene(scene)
    obj = sorted(ids["grasp_gids"])
    out = copy.deepcopy(pose)
    fins = list(fingers or pose.get("grip_fingers") or list(rig.fingertip_sites))

    def run(touch, maxpen):
        d = mujoco.MjData(model)
        d.qpos[:] = data0.qpos
        H.apply_pose(model, d, rig, pose)
        rep = H.cinch_fingers(model, d, rig, obj, fingers=fins, step_rad=step_rad,
                              max_steps=max_steps, touch_m=-touch / 1000.0,
                              max_pen_m=maxpen / 1000.0)
        return {k: float(d.qpos[a]) for k, a in rig.finger_qpos.items()}, rep

    out["finger_targets"], out["cinch"] = run(float(touch_mm), float(max_pen_mm))
    if squeeze_mm and squeeze_mm > 0:
        sq, _ = run(float(squeeze_mm), float(squeeze_mm) + 1.0)
        out["finger_targets_squeezed"] = sq
    out["grip_fingers"] = fins
    return out
