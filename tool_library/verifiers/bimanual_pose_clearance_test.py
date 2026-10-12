TOOL = {
    "name": "bimanual_pose_clearance_test",
    "title": "Two-hand pose clearance test",
    "type": "verifier",
    "category": "Grasp Stress Tests",
    "summary": "Put the left and right hands at their grasp poses and along their open-hand approach paths in one model and fail when the two hands come closer than a minimum distance anywhere.",
    "description": "Every hand scene holds one hand, so two grasps solved separately on neighbouring parts (two magnetic rings 3 cm apart, a bottle and its cap, a box and its lid) can each pass every grasp test and still drive the hands into each other on the bench (measured on the EgoDex rings: palm into palm 14.8 mm and index into the other hand's fingers 17.9 mm, seen only in the bench physics). Both hands (with the collision hulls the bench uses) are placed by their wrist poses, which share the bench's world frame, and the smallest distance between any collision geom of one hand and any of the other is measured with mj_geomDistance: (1) both hands at their poses with the executor's closed targets and its clamp; (2) the left hand coming down its approach (approach_m back along the executor's retreat axis, open shape, n_samples steps) and closing while the right hand holds its pose, (3) the same with the roles swapped, and (4) both approaching together. Fails when any phase gets closer than min_clearance_mm; the issue names the phase, the step, the two closest geoms and how far they are apart (negative = interpenetration).",
    "inputs": [
        {"name": "left_pose / right_pose", "type": "dict | path", "description": "the two grasp poses (world frame) or their JSON"},
        {"name": "left_scene / right_scene", "type": "dict | path | None", "description": "optional hand scenes: their hand_xml are used (default setup/robot/sharpa_<side>.xml)"},
        {"name": "min_clearance_mm", "type": "float", "description": "smallest allowed hand-hand distance (default 10: two 2 mm / 2 deg calibration errors at a 0.1 m lever)"},
        {"name": "approach_m", "type": "float", "description": "pre-grasp distance along each retreat axis (default 0.08, the executor's)"},
        {"name": "n_samples", "type": "int", "description": "steps along each approach (default 12) and the close (4)"},
        {"name": "check_approach", "type": "bool", "description": "also check the approach phases (default True)"},
        {"name": "approach_sides", "type": "list", "description": "hands whose approach is checked (default both; ['right'] when the left hand is already holding and does not move)"},
    ],
    "outputs": [
        {"name": "ok", "type": "bool", "description": "the hands keep min_clearance_mm everywhere"},
        {"name": "issues", "type": "list", "description": "the phase, step and geoms that came too close, and what to change"},
        {"name": "min_clearance_mm", "type": "float", "description": "the smallest hand-hand distance over every phase"},
        {"name": "phases", "type": "list", "description": "per phase: min_mm, at (step), closest geom pair, ok"},
    ],
    "status": "tested",
    "entrypoint": "bimanual_pose_clearance_test",
    "requirements": ["mujoco>=3.1", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["bimanual", "clearance", "hand-hand", "collision", "stress-test"],
}


def _shapes(rig, pose):
    from tool_library.common import hand as H
    c = H.executor_clamp(rig, pose)                   # the executor's clamp (fingers as posed for a wrist press)
    closed, press, grip = c["closed"], c["press"], c["grip_fingers"]
    own = {k: float(v) for k, v in (pose.get("finger_targets_open") or {}).items() if k in rig.finger_range}
    opened = own or H.approach_open_targets(rig, closed, float(pose.get("open_frac", 0.55)), grip, c["grip_posed"])
    return {"opened": opened, "closed": closed, "press": press}


def bimanual_pose_clearance_test(left_pose, right_pose, left_scene=None, right_scene=None, min_clearance_mm=10.0, approach_m=0.08,
                                 n_samples=12, check_approach=True, approach_sides=("left", "right")):
    import mujoco
    import numpy as np
    from tool_library.common import grasp_robust as GR
    from tool_library.common import hand as H
    from tool_library.common import plan as P
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    approach_sides = [approach_sides] if isinstance(approach_sides, str) else list(approach_sides)
    if not approach_sides or any(s_ not in ("left", "right") for s_ in approach_sides):
        raise ValueError("approach_sides must name 'left' and/or 'right'")
    poses = {"left": GR.load_pose(left_pose, "left_pose"), "right": GR.load_pose(right_pose, "right_pose")}
    xml = {}
    for s, sc in (("left", left_scene), ("right", right_scene)):
        if sc is None:
            xml[s] = str(GR.default_hand_xml(s))
            continue
        spec = GR.load_scene(sc, side=s)
        if spec.get("side", s) != s:
            raise ValueError(f"{s}_scene: its hand is the {spec.get('side')} hand, not the {s} hand")
        xml[s] = spec["hand_xml"]
    model, rigs = GR.two_hand_model(xml["left"], xml["right"])
    d = mujoco.MjData(model)
    sh = {s: _shapes(rigs[s], poses[s]) for s in rigs}
    frame = {}
    for s, p in poses.items():
        R = H.euler_to_mat(p["euler"])
        pos = np.asarray(p["pos"], float)
        frame[s] = {"pos": pos, "euler": np.asarray(p["euler"], float), "pre": pos + P.retreat_axis(p, R) * float(approach_m)}

    def put(s, pos, fingers):
        rigs[s].teleport(d, pos, frame[s]["euler"])
        rigs[s].set_fingers(d, fingers)

    def measure():
        mujoco.mj_kinematics(model, d)
        return GR.hand_hand_distance(model, d, rigs["left"], rigs["right"])

    def interp(a, b, u):
        return {k: a.get(k, v) + (v - a.get(k, v)) * u for k, v in b.items()}

    phases = []

    def run_phase(name, steps):
        best = (1e9, None, None)
        for tag, setter in steps:
            setter()
            dist, pair = measure()
            if dist < best[0]:
                best = (dist, tag, pair)
        phases.append({"name": name, "min_mm": round(best[0] * 1000, 1), "at": best[1],
                       "pair": list(best[2]) if best[2] else None, "ok": bool(best[0] * 1000 >= min_clearance_mm)})

    def final(s, shape):
        return lambda: put(s, frame[s]["pos"], sh[s][shape])

    run_phase("both at their poses", [("closed", lambda: (final("left", "closed")(), final("right", "closed")())),
                                      ("clamp", lambda: (final("left", "press")(), final("right", "press")()))])
    if check_approach:
        us = np.linspace(0.0, 1.0, int(n_samples))

        def approach_steps(s):
            out = []
            for u in us:
                out.append((f"{s} approach {u:.2f}", lambda u=u: put(s, frame[s]["pre"] + (frame[s]["pos"] - frame[s]["pre"]) * u, sh[s]["opened"])))
            for u in np.linspace(0.25, 1.0, 4):
                out.append((f"{s} closing {u:.2f}", lambda u=u: put(s, frame[s]["pos"], interp(sh[s]["opened"], sh[s]["closed"], u))))
            return out

        for s, o in (("left", "right"), ("right", "left")):
            if s not in approach_sides:
                continue
            steps = [(tag, lambda f=fn, o=o: (final(o, "press")(), f())) for tag, fn in approach_steps(s)]
            run_phase(f"{s} approaches while {o} holds", steps)
        if len(approach_sides) == 2:
            both = [(f"both approach {u:.2f}", lambda u=u: [put(s, frame[s]["pre"] + (frame[s]["pos"] - frame[s]["pre"]) * u, sh[s]["opened"])
                                                           for s in ("left", "right")]) for u in us]
            run_phase("both approach together", both)
    worst = min(phases, key=lambda p: p["min_mm"])
    issues = []
    for ph in phases:
        if not ph["ok"]:
            what = "interpenetrate by" if ph["min_mm"] < 0 else "come within"
            issues.append(f"{ph['name']}: the hands {what} {abs(ph['min_mm']):.1f} mm (need {min_clearance_mm:g} mm clearance) at '{ph['at']}', "
                          f"between {ph['pair'][0]} and {ph['pair'][1]}. Move one grasp to the far side of its part or rotate its wrist "
                          "away from the other hand (or sequence the two grasps so they are never there together).")
    return {"ok": not issues, "issues": issues, "min_clearance_mm": worst["min_mm"], "worst_phase": worst["name"], "phases": phases,
            "limits": {"min_clearance_mm": min_clearance_mm, "approach_m": approach_m}}
