TOOL = {
    "name": "grasp_lift_test",
    "title": "Grasp lift test",
    "type": "verifier",
    "category": "Grasp Stress Tests",
    "summary": "Close the posed hand on the object by driving its actuators and lift; pass when the object rises with the hand at every commanded height.",
    "description": "The hand approaches with the executor's opening and motion. Every physics substep checks contact penetration, including grip pads and self-contact; visible mesh penetration of the support is checked at 60 Hz. More than 3 mm fails regardless of lift or hold success. The hand approaches open, closes until the grip pads are on the object with real force (or to the full press target), seats for 0.5 s, then the wrist is driven up to each height in turn. A height passes when the object rose more than lift_frac of it. This is the single most informative grasp test: a pose that looks seated statically and still fails it is pressing, not gripping (both pads above the object's centre line, or one finger loaded and the others in the air). With maintain_N > 0 the grip fingers keep closing while the contact force is below it. While the hand alone carries the object, the finger servos' reserve is judged as finger_servo_saturation_audit judges an episode (the smallest share of the finger torque caps with which the hand's contacts carry it, every executor frame): a grasp that holds only with a finger servo at 95% of its cap or more on 60% of the hold fails at 'servo_reserve' (a drive pinched 29 mm off its centre lifted with the thumb's abduction servo at 110% of its cap).",
    "inputs": [
        {"name": "scene", "type": "dict", "description": "{hand_xml, side, objects, floor, grasp_body?}"},
        {"name": "pose", "type": "dict", "description": "{pos, euler, finger_targets, finger_targets_squeezed?, grip_fingers?}"},
        {"name": "lift_heights_m", "type": "list", "description": "default (0.06, 0.10, 0.15)"},
        {"name": "lift_frac", "type": "float", "description": "fraction of the commanded height the object must rise (default 0.5)"},
        {"name": "maintain_N", "type": "float", "description": "grip-force regulation target, 0 disables"},
        {"name": "approach", "type": "bool", "description": "include the open-hand approach phase (default True)"},
    ],
    "outputs": [
        {"name": "ok", "type": "bool", "description": "every lift raised the object"},
        {"name": "lifts", "type": "list", "description": "[{commanded_mm, achieved_mm}]"},
        {"name": "grip", "type": "dict", "description": "finger contacts and force when seated"},
        {"name": "first_fail / phases", "type": "...", "description": "the ladder's per-phase record"},
    ],
    "status": "tested",
    "entrypoint": "grasp_lift_test",
    "requirements": ["mujoco>=3.1", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["grasp", "lift"],
    "used_in": ["Place Cup in Bowl", "Giftbox Packing", "Knot the Rope", "Battery Insertion", "Open the Bottle", "Scissor Cutting"],
}


def grasp_lift_test(scene, pose, lift_heights_m=(0.06, 0.10, 0.15), lift_frac=0.5, maintain_N=0.0, approach=True):
    from tool_library.common import hand as H
    model, data0, rig, ids = H.open_scene(scene)
    bid = H.grasp_body(model, scene, ids)
    ph = ("approach", "close", "lift") if approach else ("close", "lift")
    r = H.stress_rollout(model, data0, rig, ids, pose, bid, phases=ph,
                         lift_heights=tuple(lift_heights_m), lift_frac=lift_frac, maintain_N=maintain_N)
    seated = next((p for p in r["phases"] if p["name"] == "seated"), {})
    return {"ok": bool(r["ok"]), "lifts": r["lifts"], "first_fail": r["first_fail"],
            "penetration": r["penetration"],
            "grip": {"fingers": seated.get("fingers"), "force_N": seated.get("contact_force_N")},
            "phases": r["phases"]}
