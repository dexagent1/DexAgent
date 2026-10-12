TOOL = {
    "name": "grasp_hold_test",
    "title": "Grasp hold test",
    "type": "verifier",
    "category": "Grasp Stress Tests",
    "summary": "Close on the object, lift, and hold still; pass when the object's slip rate in the hand is below a threshold and at least one pad is still on it.",
    "description": "The hand approaches with the executor's opening and motion. Every physics substep checks contact penetration, including grip pads and self-contact; visible mesh penetration of the support is checked at 60 Hz. More than 3 mm fails regardless of lift or hold success. After closing and lifting, the wrist is commanded to stay put for hold_s while every simulated state is checked. Slip rate is the maximum displacement from the hold's initial relative position divided by hold duration, so a slip that later recovers cannot disappear. Rigid held links must also stay within the episode audit's 15 degree relative rotation limit. Articulated objects are measured at the contacted link and material contact point; ball-joint bending is reported without judging its angle. Also reports contact force and pads at the end.",
    "inputs": [
        {"name": "scene", "type": "dict", "description": "{hand_xml, side, objects, floor, grasp_body?}"},
        {"name": "pose", "type": "dict", "description": "{pos, euler, finger_targets, finger_targets_squeezed?, grip_fingers?}"},
        {"name": "hold_s", "type": "float", "description": "hold duration (default 0.8)"},
        {"name": "slip_max_mm_s", "type": "float", "description": "allowed slip rate (default 10)"},
        {"name": "lift_m", "type": "float", "description": "lift before holding (default 0.08)"},
        {"name": "maintain_N", "type": "float", "description": "grip-force regulation target, 0 disables"},
    ],
    "outputs": [
        {"name": "ok", "type": "bool", "description": "slip below the limit and still in contact"},
        {"name": "slip_mm_s", "type": "float", "description": "measured slip rate"},
        {"name": "contact_force_N / fingers", "type": "...", "description": "state at the end of the hold"},
    ],
    "status": "tested",
    "entrypoint": "grasp_hold_test",
    "requirements": ["mujoco>=3.1", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["grasp", "hold", "slip"],
    "used_in": ["Place Cup in Bowl", "Battery Insertion", "Open the Bottle"],
}


def grasp_hold_test(scene, pose, hold_s=0.8, slip_max_mm_s=10.0, lift_m=0.08, maintain_N=0.0):
    from tool_library.common import hand as H
    model, data0, rig, ids = H.open_scene(scene)
    bid = H.grasp_body(model, scene, ids)
    r = H.stress_rollout(model, data0, rig, ids, pose, bid, phases=("approach", "close", "lift", "hold"),
                         lift_heights=(float(lift_m),), hold_s=hold_s,
                         slip_max_mps=slip_max_mm_s / 1000.0, maintain_N=maintain_N)
    hold = next((p for p in r["phases"] if p["name"] == "hold"), {})
    lift_ok = all(p.get("ok", True) for p in r["phases"] if p["name"].startswith("lift_"))
    return {"ok": bool(r["ok"] and hold.get("ok", False) and lift_ok), "slip_mm_s": hold.get("slip_mm_s"),
            "first_fail": r["first_fail"], "penetration": r["penetration"],
            "contact_force_N": hold.get("contact_force_N"), "fingers": hold.get("finger_contacts"),
            "lifted": lift_ok, "phases": r["phases"]}
