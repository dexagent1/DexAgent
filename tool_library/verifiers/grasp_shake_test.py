TOOL = {
    "name": "grasp_shake_test",
    "title": "Grasp carry-and-turn test",
    "type": "verifier",
    "category": "Grasp Stress Tests",
    "summary": "Hold the object, lift it, then carry it out and back in five directions and turn the wrist about three axes; pass when the object's drift in the wrist frame stays small.",
    "description": "The hand approaches with the executor's opening and motion. Every physics substep checks contact penetration, including grip pads and self-contact; visible mesh penetration of the support is checked at 60 Hz. More than 3 mm fails regardless of lift or hold success. After the close-until-held and a short lift, the wrist translates shake_m out and back along up, forward, back, left and right, then rotates turn_deg and back about roll, pitch and yaw. Drift is the change of the object's position in the WRIST frame since seating, so an object that travels with the hand has not slipped however far it moved in the world. Each move and turn must keep the drift under drift_max_mm. This is the test that catches grasps which lift but rotate in the hand.",
    "inputs": [
        {"name": "scene", "type": "dict", "description": "{hand_xml, side, objects, floor, grasp_body?}"},
        {"name": "pose", "type": "dict", "description": "{pos, euler, finger_targets, finger_targets_squeezed?, grip_fingers?}"},
        {"name": "shake_m", "type": "float", "description": "carry amplitude (default 0.04)"},
        {"name": "turn_deg", "type": "float", "description": "wrist turn about each axis (default 25)"},
        {"name": "drift_max_mm", "type": "float", "description": "allowed drift in the wrist frame (default 30)"},
        {"name": "lift_m", "type": "float", "description": "lift before shaking (default 0.08)"},
        {"name": "maintain_N", "type": "float", "description": "grip-force regulation target, 0 disables"},
    ],
    "outputs": [
        {"name": "ok", "type": "bool", "description": "drift stayed under the limit through every move and turn"},
        {"name": "max_drift_mm", "type": "float", "description": "largest drift seen"},
        {"name": "first_fail / phases", "type": "...", "description": "per-move drift and ok"},
    ],
    "status": "tested",
    "entrypoint": "grasp_shake_test",
    "requirements": ["mujoco>=3.1", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["grasp", "carry", "turn", "drift"],
    "used_in": ["Place Cup in Bowl", "Open the Bottle", "Scissor Cutting"],
}


def grasp_shake_test(scene, pose, shake_m=0.04, turn_deg=25.0, drift_max_mm=30.0, lift_m=0.08, maintain_N=0.0):
    from tool_library.common import hand as H
    model, data0, rig, ids = H.open_scene(scene)
    bid = H.grasp_body(model, scene, ids)
    r = H.stress_rollout(model, data0, rig, ids, pose, bid, phases=("approach", "close", "lift", "move", "turn"),
                         lift_heights=(float(lift_m),), shake_m=shake_m, turn_deg=turn_deg,
                         drift_max_m=drift_max_mm / 1000.0, maintain_N=maintain_N)
    moves = [p for p in r["phases"] if p["name"].startswith(("move_", "turn_"))]
    bad = [p["name"] for p in r["phases"] if not p.get("ok", True)]
    return {"ok": bool(r["ok"] and not bad), "max_drift_mm": r["max_drift_mm"], "first_fail": r["first_fail"],
            "penetration": r["penetration"],
            "phases": r["phases"], "n_moves": len(moves)}
