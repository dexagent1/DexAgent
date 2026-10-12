TOOL = {
    "name": "pregrasp_clearance_test",
    "title": "Pre-grasp clearance test",
    "type": "verifier",
    "category": "Grasp Stress Tests",
    "summary": "Find how far the open hand must back off along its approach before it clears every object by a margin, and fail when no retraction within reach is clear.",
    "description": "Opening the fingers at the grasp position sweeps them through the very thing being grasped, so a grasp is only enterable if there is a nearby pose where the OPEN hand touches nothing. The fingers are opened open_frac of the way from the closed targets (idle fingers curled, an idle thumb extended), and the wrist is stepped back along the approach direction (the pose's own, else +z) in 12 mm steps until the smallest distance from any hand collision geom to any object or the floor is at least need_mm. Returns that retraction and the open targets so the caller can drive the entry from there. Failing means the grasp cannot be entered as posed, which is itself the answer.",
    "inputs": [
        {"name": "scene", "type": "dict", "description": "{hand_xml, side, objects, floor}"},
        {"name": "pose", "type": "dict", "description": "{pos, euler, finger_targets, grip_fingers?, approach?}"},
        {"name": "approach", "type": "xyz", "description": "retreat direction override"},
        {"name": "open_frac", "type": "float", "description": "how far the fingers open (default 0.55)"},
        {"name": "need_mm", "type": "float", "description": "required clearance (default 4)"},
        {"name": "tries / step_m", "type": "numbers", "description": "retraction ladder (14 steps of 0.012 m)"},
    ],
    "outputs": [
        {"name": "ok", "type": "bool", "description": "a clear pre-grasp exists within the ladder"},
        {"name": "retract_m", "type": "float or None", "description": "retraction that clears"},
        {"name": "clearance_mm", "type": "float", "description": "clearance at that retraction (or the best seen)"},
        {"name": "open_targets / approach / curled", "type": "...", "description": "what to drive to enter the grasp"},
    ],
    "status": "tested",
    "entrypoint": "pregrasp_clearance_test",
    "requirements": ["mujoco>=3.1", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["pre-grasp", "clearance", "approach"],
    "used_in": ["Place Cup in Bowl", "Giftbox Packing", "Knot the Rope", "Scissor Cutting"],
}


def pregrasp_clearance_test(scene, pose, approach=None, open_frac=0.55, need_mm=4.0, tries=14, step_m=0.012):
    import mujoco
    import numpy as np
    from tool_library.common import hand as H
    model, data0, rig, ids = H.open_scene(scene)
    d = mujoco.MjData(model)
    d.qpos[:] = data0.qpos
    closed = {k: float(v) for k, v in (pose.get("finger_targets") or {}).items() if k in rig.finger_range}
    gf = list(pose.get("grip_fingers") or [])
    idle = [f for f in H.FINGERS if gf and f not in gf]
    op = H.open_targets(rig, closed, open_frac, curl=idle)
    P = np.asarray(pose["pos"], float)
    if approach is not None:
        a = np.asarray(approach, float)
    else:
        H.apply_pose(model, d, rig, pose, op)
        a = H.approach_axis(model, d, rig, pose)
    a = a / (np.linalg.norm(a) + 1e-12)
    others = sorted(ids["obj_gids"]) + ([ids["floor_gid"]] if ids["floor_gid"] >= 0 else [])
    best = None
    for k in range(0, int(tries) + 1):
        back = float(k) * float(step_m)
        H.apply_pose(model, d, rig, {"pos": P + a * back, "euler": pose["euler"]}, op)
        worst = min(H.hand_geom_distance(model, d, g, og, 0.05) for g in rig.collision_gids for og in others)
        if best is None or worst > best[1]:
            best = (back, worst)
        if worst * 1000.0 >= float(need_mm):
            return {"ok": True, "retract_m": back, "clearance_mm": round(worst * 1000, 2),
                    "open_targets": op, "approach": [float(v) for v in a], "curled": idle}
    return {"ok": False, "retract_m": None, "best_retract_m": best[0], "clearance_mm": round(best[1] * 1000, 2),
            "open_targets": op, "approach": [float(v) for v in a], "curled": idle}
