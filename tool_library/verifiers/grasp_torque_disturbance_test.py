TOOL = {
    "name": "grasp_torque_disturbance_test",
    "title": "Torque disturbance test",
    "type": "verifier",
    "category": "Grasp Stress Tests",
    "summary": "Hold the lifted object and twist it with an external torque ramped up to a limit about each of its principal axes, both ways; fail when the object turns or slides in the hand before the limit.",
    "description": "Using a held object (pouring, wiping, pressing a trigger, a tool meeting resistance) loads the grasp with torques that a lift never applies, and a two-pad pinch that lifts cleanly can spin like an axle about the line between its pads. After the executor's grasp (open-hand descent, close, torque-capped clamp) and a lift of lift_m, an external torque is applied at the object's centre of mass about each principal axis of its inertia, in both directions, ramped from zero to the limit over ramp_s and held for hold_s, then removed for 0.2 s before the next axis. The object's turn and drift in the wrist frame are tracked from the start of each twist; the breakaway torque is the applied torque when the turn first passes turn_max_deg. The limit is torque_nm when given, else the object's own weight acting at lever_m (default: half its largest dimension, at least 2 cm), the torque of holding it by one end. Passes when no axis breaks away below the limit and every drift stays under drift_max_mm. A part on a fixed asset is reported as applicable=False with ok=True.",
    "inputs": [
        {"name": "scene", "type": "dict | path", "description": "hand scene: dict, hand_scene_from_recon JSON, or an objects MJCF"},
        {"name": "pose", "type": "dict | path", "description": "grasp pose or its JSON"},
        {"name": "torque_nm", "type": "float | None", "description": "torque limit (N m); default mass x g x lever_m"},
        {"name": "lever_m", "type": "float | None", "description": "lever of the default limit (default half the object's largest dimension, at least 0.02)"},
        {"name": "ramp_s / hold_s", "type": "float", "description": "ramp to the limit (default 0.5) and hold at it (default 0.3)"},
        {"name": "turn_max_deg / drift_max_mm", "type": "float", "description": "pass limits in the wrist frame (default 10 deg, 10 mm)"},
        {"name": "lift_m", "type": "float", "description": "lift before twisting (default 0.08)"},
        {"name": "approach / side / grasp_body", "type": "...", "description": "descend from the pre-grasp (default True); scene overrides"},
    ],
    "outputs": [
        {"name": "ok", "type": "bool", "description": "the grasp resists the limit torque about every axis"},
        {"name": "applicable", "type": "bool", "description": "False for a part on a fixed asset"},
        {"name": "issues", "type": "list", "description": "the axes that broke away, at what torque, and what to change"},
        {"name": "torque_limit_nm", "type": "float", "description": "the limit used"},
        {"name": "axes", "type": "list", "description": "per axis and sign: axis (world, at the twist start), turn_deg, drift_mm, breakaway_nm or None, ok"},
        {"name": "lift", "type": "dict", "description": "the lift before the twists"},
    ],
    "status": "tested",
    "entrypoint": "grasp_torque_disturbance_test",
    "requirements": ["mujoco>=3.1", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["grasp", "torque", "disturbance", "tool-use", "stress-test"],
}



REPORT_KEYS = ('ok', 'applicable', 'issues', 'note', 'object', 'torque_limit_nm', 'lever_m', 'mass_kg', 'axes', 'lift', 'limits')

def grasp_torque_disturbance_test(scene, pose, torque_nm=None, lever_m=None, ramp_s=0.5, hold_s=0.3, turn_max_deg=10.0,
                                  drift_max_mm=10.0, lift_m=0.08, approach=True, side=None, grasp_body=None):
    import numpy as np
    from tool_library.common import grasp_robust as GR
    if torque_nm is not None and torque_nm <= 0:
        raise ValueError("torque_nm must be positive (or None for the object's own weight moment)")
    if ramp_s <= 0 or hold_s < 0:
        raise ValueError("ramp_s must be positive and hold_s non-negative")
    spec = GR.load_scene(scene, side=side, grasp_body=grasp_body)
    pose = GR.load_pose(pose)
    case = GR.Case(spec)
    body = case.model.body(case.bid).name
    if case.root is None:
        return GR.complete({"ok": True, "applicable": False, "issues": [], "object": body,
                "note": f"{body} is a part of a fixed asset: its joint takes the torque"}, REPORT_KEYS)
    run = GR.close_like_executor(case, pose, approach=approach)
    d, g = run["data"], run["grip"]
    lf = GR.lift_after(case, run, lift_m)
    size = GR.object_size_m(case, d)
    lever = float(lever_m) if lever_m is not None else max(0.5 * size, 0.02)
    limit = float(torque_nm) if torque_nm is not None else case.mass * 9.81 * lever
    issues = []
    if not lf["ok"]:
        issues.append(f"the grasp does not lift {body} (rose {lf['rose_mm']} of {lift_m * 1000:.0f} mm): fix it with grasp_lift_test before "
                      "testing torques")
        return GR.complete({"ok": False, "applicable": True, "issues": issues, "object": body, "torque_limit_nm": round(limit, 4), "axes": [], "lift": lf}, REPORT_KEYS)
    rows = []
    tb = case.bid
    for k in range(3):
        for sgn in (1.0, -1.0):
            p0, R0 = case.obj_in_wrist(d)
            axis0 = np.asarray(d.ximat[tb], float).reshape(3, 3)[:, k].copy()
            st = {"turn": 0.0, "drift": 0.0, "brk": None, "tau": 0.0}

            def push(dd, tau):
                ax = np.asarray(dd.ximat[tb], float).reshape(3, 3)[:, k]
                dd.xfrc_applied[tb, 3:6] = sgn * tau * ax
                st["tau"] = tau

            def track(dd):
                p, R = case.obj_in_wrist(dd)
                turn = GR.rot_deg(R0, R)
                st["turn"] = max(st["turn"], turn)
                st["drift"] = max(st["drift"], float(np.linalg.norm(p - p0)) * 1000)
                if st["brk"] is None and turn > turn_max_deg:
                    st["brk"] = st["tau"]

            n = max(int(round(ramp_s / case.dt)), 1)
            e = run["euler"]
            for i in range(n):
                push(d, limit * (i + 1) / n)
                GR.hold(case, d, run["pos"], run["R"], g["press"], case.dt, track, e)
            for _ in range(int(round(hold_s / case.dt))):
                push(d, limit)
                GR.hold(case, d, run["pos"], run["R"], g["press"], case.dt, track, e)
            d.xfrc_applied[tb, :] = 0.0
            GR.hold(case, d, run["pos"], run["R"], g["press"], 0.2, None, e)
            ok = bool(st["brk"] is None and st["drift"] <= drift_max_mm and case.contacts(d)["n_finger"] >= 1)
            rows.append({"axis": f"{'+' if sgn > 0 else '-'}{'xyz'[k]}", "axis_world": [round(float(v), 3) for v in sgn * axis0],
                         "turn_deg": round(st["turn"], 2), "drift_mm": round(st["drift"], 2),
                         "breakaway_nm": None if st["brk"] is None else round(st["brk"], 4), "ok": ok})
    bad = [r for r in rows if not r["ok"]]
    if bad:
        lst = "; ".join(f"{r['axis']} (world {r['axis_world']}): turned {r['turn_deg']} deg, drifted {r['drift_mm']} mm"
                        + (f", broke away at {r['breakaway_nm']} N m" if r["breakaway_nm"] is not None else "") for r in bad)
        issues.append(f"{body} turns or slides in the hand under a torque up to {limit:.3f} N m (its weight at {lever * 100:.1f} cm): {lst}. "
                      "A grasp that will pour, wipe or push with the object needs contacts spread across the twisting axis: add a "
                      "finger or the palm away from the line between the pads, or grip further from the object's centre along that axis.")
    return GR.complete({"ok": not bad, "applicable": True, "issues": issues, "object": body, "torque_limit_nm": round(limit, 4),
            "lever_m": round(lever, 4), "mass_kg": round(case.mass, 4), "axes": rows, "lift": lf,
            "limits": {"turn_max_deg": turn_max_deg, "drift_max_mm": drift_max_mm}}, REPORT_KEYS)
