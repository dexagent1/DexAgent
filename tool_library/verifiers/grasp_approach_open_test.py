TOOL = {
    "name": "grasp_approach_open_test",
    "title": "Open-hand approach test",
    "type": "verifier",
    "category": "Grasp Stress Tests",
    "summary": "Check that the open hand can be driven into the grasp pose without knocking the object away or burying its collision geometry in the scene.",
    "description": "A grasp is not just its final pose: opening the fingers at the grasp position sweeps them through the very thing being grasped (measured on a cord: teleporting the open hand onto the solved wrist shoved it 15 mm sideways and every later lift was the hand chasing an object it had already knocked away). The hand starts 90 mm back along its retreat axis with the pose's recorded opening (or the executor's floor-clear opening), settles, and follows approach waypoints with the executor's rate-limited motion. Passes when the object moved less than nudge_max_mm and every physics substep stays below the penetration cap, including grip pads and self-contact; visible hand meshes against the floor are checked at 60 Hz.",
    "inputs": [
        {"name": "scene", "type": "dict", "description": "{hand_xml, side, objects, floor, grasp_body?}"},
        {"name": "pose", "type": "dict", "description": "{pos, euler, finger_targets, grip_fingers?}"},
        {"name": "approach_m", "type": "float", "description": "how far back the open hand starts (default 0.09)"},
        {"name": "open_frac", "type": "float | None", "description": "optional explicit opening fraction; default honors pose opening and executor floor clearance"},
        {"name": "nudge_max_mm", "type": "float", "description": "allowed object displacement (default 20)"},
        {"name": "pen_max_mm", "type": "float", "description": "maximum contact or visible-floor penetration along the approach in mm (default 3; cannot exceed shared 3 mm cap)"},
    ],
    "outputs": [
        {"name": "ok", "type": "bool", "description": "approached without a nudge or a burial"},
        {"name": "object_nudged_mm", "type": "float", "description": "object displacement during the approach"},
        {"name": "penetration_mm", "type": "float", "description": "deepest collision-geometry penetration at arrival"},
        {"name": "approach_axis", "type": "xyz", "description": "the retreat direction that was used"},
    ],
    "status": "tested",
    "entrypoint": "grasp_approach_open_test",
    "requirements": ["mujoco>=3.1", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["grasp", "approach", "pre-grasp"],
    "used_in": ["Place Cup in Bowl", "Knot the Rope", "Scissor Cutting"],
}


def grasp_approach_open_test(scene, pose, approach_m=0.09, open_frac=None, nudge_max_mm=20.0, pen_max_mm=3.0):
    import mujoco
    import numpy as np
    from tool_library.common import hand as H
    model, data0, rig, ids = H.open_scene(scene)
    bid = H.grasp_body(model, scene, ids)
    data = mujoco.MjData(model)
    data.qpos[:] = data0.qpos
    monitor = H.GraspPenetrationMonitor(model, rig, ids)
    closed = {k: float(v) for k, v in (pose.get("finger_targets") or {}).items() if k in rig.finger_range}
    gf = list(pose.get("grip_fingers") or [])
    clamp = H.executor_clamp(rig, pose)
    own = {k: float(v) for k, v in (pose.get("finger_targets_open") or {}).items() if k in rig.finger_range}
    fraction = float(pose.get("open_frac", .55)) if open_frac is None else float(open_frac)
    opened = own if own and open_frac is None else H.approach_open_targets(
        rig, closed, fraction, gf, clamp["grip_posed"])
    floor = int(ids.get("floor_gid", -1))
    if open_frac is None and not own and floor >= 0:
        for f in [fraction] + [v for v in (.45, .35, .25, .15) if v < fraction - 1e-9]:
            opened = H.approach_open_targets(rig, closed, f, gf, clamp["grip_posed"])
            H.apply_pose(model, data, rig, pose, opened)
            if min(H.geom_distance(model, data, g, floor, .05) for g in rig.collision_gids) > -.002:
                break
    pos, eul = np.asarray(pose["pos"], float), np.asarray(pose["euler"], float)
    H.apply_pose(model, data, rig, pose, opened)
    axis = H.approach_axis(model, data, rig, pose)
    H.apply_pose(model, data, rig, {"pos": pos + axis * approach_m, "euler": eul}, opened)
    H.step_hold(model, data, rig, pos + axis * approach_m, eul, opened, 0.4, monitor, "settle")
    p0 = np.asarray(data.xpos[bid], float).copy()
    current = pos + axis * approach_m
    waypoints = list(pose.get("approach_waypoints") or [])
    for i, waypoint in enumerate(waypoints + [[0., 0., 0.]]):
        target = pos + np.asarray(waypoint, float)
        H._exec_leg(model, data, rig, current, target, eul, eul, opened, opened,
                    .6 if i < len(waypoints) else .8, monitor, "approach")
        current = target
    H.step_hold(model, data, rig, pos, eul, opened, 0.1, monitor, "arrival")
    nudged = float(np.linalg.norm(np.asarray(data.xpos[bid], float) - p0)) * 1000
    pen = H.clearance(model, data, rig, ids, ignore_tips=())
    peak = monitor.report()
    penetration_ok = bool(peak["ok"] and max(peak["max_contact_mm"], peak["max_visual_floor_mm"]) <= pen_max_mm)
    return {"ok": bool(nudged < nudge_max_mm and penetration_ok),
            "penetration": peak, "first_fail": "penetration" if not penetration_ok else ("approach_open" if nudged >= nudge_max_mm else None),
            "object_nudged_mm": round(nudged, 2), "penetration_mm": pen["object_mm"],
            "floor_penetration_mm": pen["floor_mm"], "approach_axis": [round(float(v), 4) for v in axis],
            "open_targets": opened}
