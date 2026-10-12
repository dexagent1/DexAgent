TOOL = {
    "name": "grasp_table_support_dependence_test",
    "title": "Support dependence test",
    "type": "verifier",
    "category": "Grasp Stress Tests",
    "summary": "Close the hand as the executor does, then take the table and every neighbouring object away from the grasped object and fail when the hand alone cannot keep it in place.",
    "description": "A pose can look held only because the object still rests on something: the table under a hand that presses from above, a neighbouring brick in a stack, a lid sitting on its box. A lift test does not always expose it, since a wedged object can be dragged up with its neighbour or slide along it. Here the hand closes exactly as the executor drives it (pre-grasp, descent, close, torque-capped clamp, elliptic cones, no-slip, gravity-compensated hand); the support the object had at that moment is recorded (normal force from the floor and from each other body), then the object is made weightless, the floor and every other object stop colliding with it and with the hand (they float in place, gravity-compensated), its weight is ramped back over ramp_s so the hand takes the load as a slow lift would load it, and the wrist holds still for hold_s more. Passes when the object's drift in the wrist frame stays under drift_max_mm, its turn under turn_max_deg, and a finger still touches it at the end. A part on a fixed asset (no free joint up its tree) is held by its joint, so the test reports applicable=False with ok=True.",
    "inputs": [
        {"name": "scene", "type": "dict | path", "description": "hand scene: dict, hand_scene_from_recon JSON, or an objects MJCF"},
        {"name": "pose", "type": "dict | path", "description": "grasp pose or its JSON"},
        {"name": "hold_s", "type": "float", "description": "seconds the hand alone must hold it after the load transfer (default 1.0)"},
        {"name": "ramp_s", "type": "float", "description": "seconds over which the object's weight is handed to the hand (default 0.5)"},
        {"name": "drift_max_mm", "type": "float", "description": "allowed drift of the object in the wrist frame (default 20, the executor's in-hand slip tolerance: a pinch re-seats by up to a centimetre when the table stops pushing)"},
        {"name": "turn_max_deg", "type": "float", "description": "allowed turn of the object in the wrist frame (default 20)"},
        {"name": "approach / side / grasp_body", "type": "...", "description": "descend from the pre-grasp (default True); scene overrides"},
    ],
    "outputs": [
        {"name": "ok", "type": "bool", "description": "the hand alone keeps the object"},
        {"name": "applicable", "type": "bool", "description": "False for a part on a fixed asset"},
        {"name": "issues", "type": "list", "description": "how far it fell or turned, what carried it before, what to change"},
        {"name": "support_before", "type": "dict", "description": "normal force (N) on the object from each supporting body at the clamp, and its weight"},
        {"name": "drift_mm / turn_deg / drop_mm", "type": "float", "description": "largest drift and turn in the wrist frame, and the fall in world z, over the hold"},
        {"name": "fingers_after", "type": "dict", "description": "finger contacts at the end of the hold"},
    ],
    "status": "tested",
    "entrypoint": "grasp_table_support_dependence_test",
    "requirements": ["mujoco>=3.1", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["grasp", "support", "table", "stress-test"],
}



REPORT_KEYS = ('ok', 'applicable', 'issues', 'note', 'object', 'support_before', 'fingers_seated', 'fingers_after', 'drift_mm', 'turn_deg', 'drop_mm', 'limits')

def _support_by_body(case, d):
    import mujoco
    import numpy as np
    m = case.model
    lg = {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in case.lifted}
    buf, out = np.zeros(6), {}
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if (g1 in lg) == (g2 in lg):
            continue
        other = g2 if g1 in lg else g1
        if other in case.rig.geom_ids or c.exclude:
            continue
        mujoco.mj_contactForce(m, d, i, buf)
        nm = "floor" if other == case.ids.get("floor_gid", -1) else (m.body(int(m.geom_bodyid[other])).name or "world")
        out[nm] = out.get(nm, 0.0) + abs(float(buf[0]))
    return {k: round(v, 3) for k, v in sorted(out.items(), key=lambda kv: -kv[1])}


def grasp_table_support_dependence_test(scene, pose, hold_s=1.0, ramp_s=0.5, drift_max_mm=20.0, turn_max_deg=20.0, approach=True,
                                        side=None, grasp_body=None):
    import numpy as np
    from tool_library.common import grasp_robust as GR
    if hold_s <= 0 or ramp_s <= 0:
        raise ValueError("hold_s and ramp_s must be positive")
    spec = GR.load_scene(scene, side=side, grasp_body=grasp_body)
    pose = GR.load_pose(pose)
    case = GR.Case(spec)
    body = case.model.body(case.bid).name
    if case.root is None:
        return GR.complete({"ok": True, "applicable": False, "issues": [], "object": body,
                "note": f"{body} is a part of a fixed asset: its joint holds it, there is no support to take away"}, REPORT_KEYS)
    run = GR.close_like_executor(case, pose, approach=approach)
    d = run["data"]
    weight = case.mass * 9.81
    support = _support_by_body(case, d)
    seated = case.contacts(d)
    p0, R0 = case.obj_in_wrist(d)
    z0 = float(d.xpos[case.bid][2])
    worst = {"drift": 0.0, "turn": 0.0, "drop": 0.0}

    def track(dd):
        p, R = case.obj_in_wrist(dd)
        worst["drift"] = max(worst["drift"], float(np.linalg.norm(p - p0)))
        worst["turn"] = max(worst["turn"], GR.rot_deg(R0, R))
        worst["drop"] = max(worst["drop"], z0 - float(dd.xpos[case.bid][2]))

    # the support goes while the object is weightless and its weight comes back over ramp_s: the hand takes the load
    # quasi-statically, as a slow lift would load it, so a sound but gentle grip is not shaken loose by a sudden drop
    GR.hand_takes_weight(case, run, seconds=ramp_s, on_step=track, isolate=True, keep_hand_floor=True)
    GR.hold(case, d, run["pos"], run["R"], run["grip"]["press"], hold_s, track, run["euler"])
    after = case.contacts(d)
    drift, turn = worst["drift"] * 1000, worst["turn"]
    ok = bool(drift <= drift_max_mm and turn <= turn_max_deg and after["n_finger"] >= 1)
    issues = []
    if not ok:
        carried = ", ".join(f"{k} {v:.2f} N" for k, v in support.items()) or "nothing"
        issues.append(
            f"with the table and the neighbouring objects taken away, {body} drifted {drift:.1f} mm and turned {turn:.1f} deg in the hand "
            f"(fell {worst['drop'] * 1000:.1f} mm; fingers on it after: {after['fingers'] or 'none'}): the pose holds it only because it rests on "
            f"something (at the clamp it was carried by {carried}; its weight is {weight:.2f} N). Close opposing pads around it below its "
            "top so the hand alone can carry it (or use a different grasp family); do not count on the table or a neighbour.")
    return GR.complete({"ok": ok, "applicable": True, "issues": issues, "object": body,
            "support_before": {"by_body_N": support, "weight_N": round(weight, 3)},
            "fingers_seated": seated, "fingers_after": after,
            "drift_mm": GR.r2(drift), "turn_deg": GR.r2(turn), "drop_mm": GR.r2(worst["drop"] * 1000),
            "limits": {"drift_max_mm": drift_max_mm, "turn_max_deg": turn_max_deg, "hold_s": hold_s, "ramp_s": ramp_s}}, REPORT_KEYS)
