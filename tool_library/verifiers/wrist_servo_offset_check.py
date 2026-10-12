TOOL = {
    "name": "wrist_servo_offset_check",
    "title": "Wrist servo offset check",
    "type": "verifier",
    "category": "Scene Assertions",
    "summary": "Measure the steady-state error of the hand's wrist position servos at a pose with the fingers closed, and return the command offset that cancels it.",
    "description": "The wrist actuators are position servos with steady-state error under load: a solved grasp pose can sit 7 mm low once the hand hangs on it dynamically, which lifts the pads clear of a cord while every static measurement still says the pose is seated. The hand is driven to the pose for `steps` steps, the error between commanded and reached wrist position is read, the command is corrected by it, and this repeats for a few rounds. The offset is kept only while it helps: if the error does not shrink (the hand is blocked by the table, not lagging) the search stops and the best offset so far is returned. Passes when the residual error is below tol_mm. Measure it with the hand CLOSED: the same grasp reads -1.7 mm open and +7.3 mm closed. It calibrates the wrist servo at the pose; it does not judge whether the grasp holds, so it is not evidence for a grasp stress test.",
    "inputs": [
        {"name": "scene", "type": "dict", "description": "{hand_xml, side, objects, floor}"},
        {"name": "pose", "type": "dict", "description": "{pos, euler, finger_targets}"},
        {"name": "tol_mm", "type": "float", "description": "acceptable residual (default 1.5)"},
        {"name": "steps / rounds", "type": "int", "description": "settle steps per round (250) and rounds (3)"},
        {"name": "give_up_mm", "type": "float", "description": "offset beyond which the hand is considered blocked (default 15)"},
    ],
    "outputs": [
        {"name": "ok", "type": "bool", "description": "residual error below tol_mm"},
        {"name": "offset_m", "type": "xyz", "description": "subtract from the commanded wrist position"},
        {"name": "residual_mm", "type": "float", "description": "error after correction"},
        {"name": "raw_error_mm", "type": "float", "description": "error with no correction"},
    ],
    "status": "tested",
    "entrypoint": "wrist_servo_offset_check",
    "requirements": ["mujoco>=3.1", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["servo", "wrist", "calibration"],
    "used_in": ["Knot the Rope", "Scissor Cutting"],
}


def wrist_servo_offset_check(scene, pose, tol_mm=1.5, steps=250, rounds=3, give_up_mm=15.0):
    import mujoco
    import numpy as np
    from tool_library.common import hand as H
    model, data0, rig, ids = H.open_scene(scene)
    d = mujoco.MjData(model)
    d.qpos[:] = data0.qpos
    fingers = {k: float(v) for k, v in (pose.get("finger_targets") or {}).items() if k in rig.finger_range}
    H.apply_pose(model, d, rig, pose, fingers)
    H.step_hold(model, d, rig, pose["pos"], pose["euler"], fingers, steps * model.opt.timestep)
    raw = float(np.linalg.norm(rig.wrist_pos(d) - np.asarray(pose["pos"], float))) * 1000
    off, err = H.wrist_servo_offset(model, d, rig, pose["pos"], pose["euler"], fingers, steps=steps,
                                    rounds=rounds, tol=tol_mm / 1000.0, give_up_m=give_up_mm / 1000.0)
    return {"ok": bool(err * 1000 <= tol_mm), "offset_m": [round(float(v), 5) for v in off],
            "residual_mm": round(err * 1000, 2), "raw_error_mm": round(raw, 2)}
