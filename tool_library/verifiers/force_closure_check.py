TOOL = {
    "name": "force_closure_check",
    "title": "Force-closure check",
    "type": "verifier",
    "category": "Grasp Stress Tests",
    "summary": "Close a posed hand on the object in physics and check that the resulting contact set has force closure (Ferrari-Canny Q1 above a threshold) rather than being a one-sided press.",
    "description": "The hand is placed open at the pose, the fingers are driven to the closed or squeezed targets, and the scene settles. The hand-object contacts from the simulator (position, normal, force, friction) form a 6-D wrench hull with 8-edge friction cones; Q1 is the largest ball around the origin inside it, and force closure means Q1 > 0. The check also fails when the contacts all push the same way (press_frac above press_max, a press that can hold nothing against gravity), when fewer than min_fingers pads are in contact, or when the close itself moved the object more than move_max_mm. This is the static companion of grasp_lift_test: cheap enough to run on every candidate, not sufficient on its own.",
    "inputs": [
        {"name": "scene", "type": "dict", "description": "{hand_xml, side, objects, floor, grasp_body?}"},
        {"name": "pose", "type": "dict", "description": "{pos, euler, finger_targets, finger_targets_squeezed?, grip_fingers?}"},
        {"name": "q1_min", "type": "float", "description": "minimum Q1 (default 0.01)"},
        {"name": "press_max", "type": "float", "description": "maximum |sum F|/sum|F| (default 0.85)"},
        {"name": "min_fingers", "type": "int", "description": "pads that must touch (default 2)"},
        {"name": "move_max_mm", "type": "float", "description": "allowed object motion during the close (default 15)"},
    ],
    "outputs": [
        {"name": "ok", "type": "bool", "description": "force closure with opposing contacts and a quiet close"},
        {"name": "q1 / press_frac / n_contacts / fingers / total_force_N / object_moved_mm", "type": "numbers", "description": "the measurements"},
    ],
    "status": "tested",
    "entrypoint": "force_closure_check",
    "requirements": ["mujoco>=3.1", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["grasp", "force-closure", "quality"],
    "used_in": ["Place Cup in Bowl", "Battery Insertion", "Open the Bottle"],
}


def force_closure_check(scene, pose, q1_min=0.01, press_max=0.85, min_fingers=2, move_max_mm=15.0):
    import mujoco
    import numpy as np
    from tool_library.common import hand as H
    model, data0, rig, ids = H.open_scene(scene)
    bid = H.grasp_body(model, scene, ids)
    radius = float(np.max(H.object_extent(model, data0, ids["grasp_gids"])["size"])) / 2.0
    d = mujoco.MjData(model)
    d.qpos[:] = data0.qpos
    closed = {k: float(v) for k, v in (pose.get("finger_targets") or {}).items() if k in rig.finger_range}
    target = {k: float(v) for k, v in (pose.get("finger_targets_squeezed") or {}).items() if k in rig.finger_range} or closed
    gf = list(pose.get("grip_fingers") or [])
    opened = H.open_targets(rig, closed, 0.55, curl=[f for f in H.FINGERS if gf and f not in gf])
    H.apply_pose(model, d, rig, pose, opened)
    H.step_hold(model, d, rig, pose["pos"], pose["euler"], opened, 0.15)
    H.ramp(model, d, rig, pose["pos"], pose["pos"], pose["euler"], pose["euler"], opened, target, 0.4)
    H.step_hold(model, d, rig, pose["pos"], pose["euler"], target, 0.5)
    cons = H.hand_object_contacts(model, d, rig, ids["grasp_gids"])
    q1 = H.q1_metric(cons, np.asarray(d.xipos[bid], float), radius=radius)
    pf = H.press_fraction(cons)
    fingers = sorted({f for c in cons for f in H.FINGERS if f"_{f}_" in (c["geom"] or "")})
    moved = float(np.linalg.norm(d.xpos[bid] - data0.xpos[bid])) * 1000
    return {"ok": bool(q1 >= q1_min and pf <= press_max and len(fingers) >= min_fingers and moved <= move_max_mm),
            "q1": round(q1, 4), "press_frac": round(pf, 3), "n_contacts": len(cons), "fingers": fingers,
            "total_force_N": round(sum(c["force"] for c in cons), 2), "object_moved_mm": round(moved, 1)}
