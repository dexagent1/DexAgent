TOOL = {
    "name": "grasp_stress_test_preopen",
    "title": "Grasp stress test with a chosen pre-grasp opening",
    "type": "verifier",
    "category": "Grasp Stress Tests",
    "summary": "Run the approach, close, lift, carry-shake, turn and hold phases of a posed hand on an object from a chosen pre-grasp finger opening, so grasps of thin objects lying on a support (pens, cards, cables) are tested the way they are executed when a wide-open thumb would hit the support.",
    "description": (
        "The library's other grasp stress tests open the hand as the executor does (the pose's finger_targets_open or "
        "open_frac, else 55%, closed further until the open fingers clear the floor). For a thin object lying on a "
        "table (a pen, a rod, a card edge) a wide-open thumb swings below the object's centre line, lands on the table and "
        "is held there by friction, so the close never reaches the object even though the pose itself is sound. This "
        "verifier runs the same physics rollout (the shared stress_rollout of common/hand.py: approach along the pose's "
        "approach axis, close until held, then the selected phases) with the pre-grasp opening fixed to `open_frac`, or "
        "to the pose's own `open_frac`, or 0.55, so an opening can be tried before it is written into the pose. The "
        "executor (plan_subgoal_keyframes) honours the same `open_frac` key on the pose, so a pose that passes here is "
        "executed with the opening it was tested with. Passes when every selected phase passes: seated with at least one "
        "finger in contact, each commanded lift raises the object by lift_frac of the command, drift and slip under "
        "their limits. "
        "While the hand alone carries the object, the finger servos' reserve is judged as finger_servo_saturation_audit judges an episode (the smallest share of the finger torque caps with which the hand's contacts carry it, every executor frame): a grasp that holds only with a finger servo at 95% of its cap or more on 60% of the hold fails at 'servo_reserve' (a drive pinched 29 mm off its centre lifted with the thumb's abduction servo at 110% of its cap)."
    ),
    "inputs": [
        {"name": "scene", "type": "dict", "description": "Floating-hand scene {hand_xml, side, objects, floor, grasp_body} (hand_scene_from_recon output)."},
        {"name": "pose", "type": "dict", "description": "Hand pose {pos, euler, finger_targets, grip_fingers, ...}; its `open_frac` is used when the argument is None."},
        {"name": "open_frac", "type": "float | None", "description": "Pre-grasp opening as a fraction of the way from the closed grip to fully open (0 = closed shape, 1 = fully open). Default: pose['open_frac'] or 0.55."},
        {"name": "phases", "type": "list", "description": "Subset of approach, close, lift, move, turn, hold in order. Default approach, close, lift, hold."},
        {"name": "lift_heights_m", "type": "list", "description": "Commanded lift heights. Default [0.06, 0.10]."},
        {"name": "approach_m", "type": "float", "description": "Approach distance along the pose's approach axis. Default 0.09."},
        {"name": "lift_frac", "type": "float", "description": "Fraction of each commanded lift the object must follow. Default 0.5."},
        {"name": "drift_max_mm", "type": "float", "description": "Largest drift of the object in the wrist frame during move/turn. Default 30."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues (list of strings); open_frac used; first_fail; phases (per-phase measurements); lifts; grip {fingers, force_N}."},
    ],
    "status": "tested",
    "entrypoint": "grasp_stress_test_preopen",
    "requirements": ["mujoco>=3.1", "numpy"],
    "tags": ["grasp", "stress-test", "thin-object", "pre-grasp", "pinch", "pen"],
    "used_in": ["Drawing and Wiping"],
}

PHASES = ("approach", "close", "lift", "move", "turn", "hold")


def grasp_stress_test_preopen(scene, pose, open_frac=None, phases=("approach", "close", "lift", "hold"),
                              lift_heights_m=(0.06, 0.10), approach_m=0.09, lift_frac=0.5, drift_max_mm=30.0):
    from tool_library.common import hand as H
    if not isinstance(scene, dict):
        raise ValueError("scene must be the floating-hand scene dict (hand_scene_from_recon output)")
    if not isinstance(pose, dict) or "pos" not in pose or "euler" not in pose:
        raise ValueError("pose must be a dict with pos and euler")
    of = open_frac if open_frac is not None else pose.get("open_frac", 0.55)
    of = float(of)
    if not 0.0 <= of <= 1.0:
        raise ValueError(f"open_frac must lie in [0, 1], got {of}")
    phases = tuple(phases)
    bad = [p for p in phases if p not in PHASES]
    if bad or not phases:
        raise ValueError(f"phases must be a non-empty subset of {PHASES}, got {list(phases)}")
    if "close" not in phases:
        raise ValueError("phases must include 'close' (every later phase holds the closed grip)")
    if not lift_heights_m:
        raise ValueError("lift_heights_m must list at least one height")
    model, data0, rig, ids = H.open_scene(scene)
    bid = H.grasp_body(model, scene, ids)
    r = H.stress_rollout(model, data0, rig, ids, pose, bid, phases=phases, approach_m=float(approach_m), open_frac=of,
                         lift_heights=tuple(float(h) for h in lift_heights_m), lift_frac=float(lift_frac),
                         drift_max_m=float(drift_max_mm) / 1000.0)
    seated = next((p for p in r["phases"] if p["name"] == "seated"), {})
    issues = []
    for p in r["phases"]:
        if not p.get("ok", True):
            issues.append(f"phase '{p['name']}' failed ({ {k: v for k, v in p.items() if k not in ('name', 'ok')} }): "
                          + ("no finger reached the object: lower open_frac, or move the pads off the support" if p["name"] == "seated"
                             else "the object did not follow: grip more fingers, grip lower on the object, or squeeze deeper"))
    return {"ok": bool(r["ok"]), "issues": issues, "open_frac": of, "first_fail": r.get("first_fail"),
            "phases": r["phases"], "lifts": r.get("lifts", []),
            "grip": {"fingers": seated.get("fingers"), "force_N": seated.get("contact_force_N")}}
