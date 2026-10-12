TOOL = {
    "name": "grasp_closing_rotation_test",
    "title": "Closing rotation test",
    "type": "verifier",
    "category": "Grasp Stress Tests",
    "summary": "Close the hand on the object exactly as the executor does and fail when the object turns or slides while the fingers close, before any lift.",
    "description": "A grasp can lift in the hand scene and still be wrong: if one pad reaches the object long before its opposing pad, the closing fingers push the object round and it is held in a different pose than planned (measured on the bench: a cream lego brick turned 34-49 deg during closing although every lift test passed, then slipped out of the pinch). The hand starts open on its pre-grasp (approach_m back along the executor's retreat axis, the executor's open shape), descends in 0.8 s, closes to the pose's closed targets (0.5 s + 0.2 s) and to the executor's clamp (0.3 s + 0.3 s), with the torque-capped finger servos, elliptic cones, no-slip iterations and a gravity-compensated hand of the hand scene. The grasped body's rotation and displacement are tracked from the moment the fingers start to close (the open-hand descent is grasp_approach_open_test's job) to the end of the clamp hold, together with the time each finger first touches it. Fails when the largest rotation exceeds max_rot_deg or the largest displacement exceeds max_shift_mm; the issue names the finger that touched first and the lag to its opposing pad. The defaults sit above what accepted bench grasps do (a pinch that pushes the object into place before the opposing pad meets it: up to 12.6 deg and 26.5 mm on the executed closes of ten accepted episodes, which this test reproduces with a median difference of 0.5 deg and 1.2 mm over 25 grasps) and below a grasp that turns the object out of its planned pose.",
    "inputs": [
        {"name": "scene", "type": "dict | path", "description": "hand scene: dict, hand_scene_from_recon JSON, or an objects MJCF (first free body grasped)"},
        {"name": "pose", "type": "dict | path", "description": "{pos, euler, finger_targets, finger_targets_squeezed?, finger_targets_open?, grip_fingers?, approach?} or its JSON"},
        {"name": "max_rot_deg", "type": "float", "description": "allowed rotation of the object during the close (default 20)"},
        {"name": "max_shift_mm", "type": "float", "description": "allowed displacement of the object during the close (default 30)"},
        {"name": "approach", "type": "bool", "description": "descend from the pre-grasp first (default True)"},
        {"name": "approach_m", "type": "float", "description": "pre-grasp distance along the retreat axis (default 0.08, the executor's)"},
        {"name": "side / grasp_body", "type": "str", "description": "optional overrides of the hand scene's side and grasped body"},
    ],
    "outputs": [
        {"name": "ok", "type": "bool", "description": "the object kept its pose while the fingers closed"},
        {"name": "issues", "type": "list", "description": "what moved, by how much, which finger touched first, what to change"},
        {"name": "max_rot_deg / max_shift_mm", "type": "float", "description": "largest rotation (deg) and displacement (mm) during the close"},
        {"name": "rot_deg / shift_mm", "type": "float", "description": "rotation and displacement at the end of the clamp hold"},
        {"name": "first_touch_s", "type": "dict", "description": "finger (or palm) -> seconds after closing began when it first touched the object"},
        {"name": "worst", "type": "dict", "description": "time, rotation and displacement of the worst moment"},
        {"name": "seated", "type": "dict", "description": "fingers in contact and their normal force at the end of the clamp"},
        {"name": "series", "type": "list", "description": "[t_s, rot_deg, shift_mm] every 20 ms of the close"},
    ],
    "status": "tested",
    "entrypoint": "grasp_closing_rotation_test",
    "requirements": ["mujoco>=3.1", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["grasp", "close", "rotation", "slip", "stress-test"],
}


def grasp_closing_rotation_test(scene, pose, max_rot_deg=20.0, max_shift_mm=30.0, approach=True, approach_m=0.08,
                                side=None, grasp_body=None):
    from tool_library.common import grasp_robust as GR
    if max_rot_deg <= 0 or max_shift_mm <= 0:
        raise ValueError("max_rot_deg and max_shift_mm must be positive")
    spec = GR.load_scene(scene, side=side, grasp_body=grasp_body)
    pose = GR.load_pose(pose)
    case = GR.Case(spec)
    r = GR.close_like_executor(case, pose, approach=approach, approach_m=approach_m)
    body = case.model.body(case.bid).name
    series = r["series"]
    worst_row = max(series, key=lambda s: (s[1] / max_rot_deg, s[2] / max_shift_mm)) if series else (0.0, 0.0, 0.0)
    worst = {"t_s": worst_row[0], "rot_deg": worst_row[1], "shift_mm": worst_row[2]}
    issues = []
    ok = bool(r["max_rot_deg"] <= max_rot_deg and r["max_shift_mm"] <= max_shift_mm)
    touch = dict(sorted(r["first_touch_s"].items(), key=lambda kv: kv[1]))
    if not ok:
        order = ", ".join(f"{f} {t:.2f} s" for f, t in touch.items()) or "no finger touched it"
        lag = ""
        if len(touch) >= 2:
            first, t1 = next(iter(touch.items()))
            opp = [(f, t) for f, t in touch.items() if (f == "thumb") != (first == "thumb")]
            if opp:
                lag = f"; {first} pushed alone for {opp[0][1] - t1:.2f} s before {opp[0][0]} met it"
        issues.append(
            f"{body} turned {r['max_rot_deg']:.1f} deg and slid {r['max_shift_mm']:.1f} mm while the fingers closed "
            f"(limits {max_rot_deg:g} deg / {max_shift_mm:g} mm; worst {worst['t_s']:.2f} s after closing began; first touch: {order}{lag}). "
            "The object is held in a different pose than planned. Move the wrist so the pinch line passes through the object's "
            "centre and the opposing pads reach it together (or re-solve the pose with optimize_fingertip_contacts / seat_grasp_for_lift); "
            "a lift test alone does not catch this.")
    return {"ok": ok, "issues": issues, "object": body,
            "max_rot_deg": GR.r2(r["max_rot_deg"]), "max_shift_mm": GR.r2(r["max_shift_mm"]),
            "rot_deg": GR.r2(r["rot_deg"]), "shift_mm": GR.r2(r["shift_mm"]),
            "first_touch_s": touch, "worst": worst, "seated": r["seated"],
            "limits": {"max_rot_deg": max_rot_deg, "max_shift_mm": max_shift_mm}, "series": series}
