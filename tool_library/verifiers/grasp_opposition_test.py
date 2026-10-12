TOOL = {
    "name": "grasp_opposition_test",
    "title": "Contact opposition test",
    "type": "verifier",
    "category": "Grasp Stress Tests",
    "summary": "Close the hand as the executor does and check that the hand-object contacts include two opposing groups whose normals face each other across the object's centre, so the grasp is a squeeze and not a hook or a press against the table.",
    "description": "After the executor's close (pre-grasp, descent, close, torque-capped clamp) the hand-object contacts are read from the simulator over a 0.1 s hold (a torque-capped pad chatters on a light object, so one step is not a reading) and grouped by link (each finger, the palm). A group counts when its normal force reaches min_force_N; its direction is the force-weighted mean of its contact normals (pointing into the object) and its point the force-weighted contact centroid. The test takes the two groups whose directions are furthest apart and requires (1) at least two loaded groups, (2) an opposition of at least min_opposition_deg: the angle between their normals, or, when the normal forces share a common push (fingertips on a rim pressing down and in while friction carries the weight), the angle between their components across that push, (3) the object's centre of mass on the inner side of both contact planes ((c - p) . n >= -com_margin_frac x object size: the pads span the object rather than both pressing one side), and (4) while the object still rests on its support (it did not rise, or lift_m is 0), a press fraction |sum F| / sum |F| of the hand's normal forces no higher than max_press_frac (a hand whose forces all push one way, into the table, is leaning on the support, not gripping). A single-finger hook, a flat press from above and a grasp that only pins the object against the table all fail. With lift_m > 0 and a free object the hand then lifts it lift_m and the verdict is taken on the LIFTED contacts (only the hand holds it then: a claw that cradles a toy touches it only after the lift, a grasp that leans the object on the table loses it); when it does not rise (a cap still on its thread, a grasp that does not lift) the seated contacts are judged and a note says so.",
    "inputs": [
        {"name": "scene", "type": "dict | path", "description": "hand scene: dict, hand_scene_from_recon JSON, or an objects MJCF"},
        {"name": "pose", "type": "dict | path", "description": "grasp pose or its JSON"},
        {"name": "min_opposition_deg", "type": "float", "description": "smallest angle between the two opposing contact groups (default 100: two pads hold by friction while that angle exceeds 180 - 2 atan(mu), and the hand pads' mu of 1.2 cut by 30% gives 100)"},
        {"name": "min_force_N", "type": "float", "description": "normal force a link group must carry to count (default 0.3)"},
        {"name": "max_press_frac", "type": "float", "description": "largest |sum F| / sum |F| of the hand's normal forces while the object rests on its support (default 0.7: a cap pinched on its thread presses down by up to two thirds of its grip; a hand leaning on a part pushes at 0.9-1.0)"},
        {"name": "com_margin_frac", "type": "float", "description": "how far (fraction of the object size) the centre of mass may sit outside a contact plane (default 0.1)"},
        {"name": "lift_m", "type": "float", "description": "short lift before the reading that is judged, 0 judges the seated contacts (default 0.03)"},
        {"name": "approach / side / grasp_body", "type": "...", "description": "descend from the pre-grasp (default True); scene overrides"},
    ],
    "outputs": [
        {"name": "ok", "type": "bool", "description": "opposing groups span the object and the hand does not lean one way"},
        {"name": "issues", "type": "list", "description": "which condition failed and what to change"},
        {"name": "groups", "type": "list", "description": "per link: force_N, normal, point"},
        {"name": "judged", "type": "str", "description": "'lifted' or 'seated': which contact state the verdict used"},
        {"name": "best_pair / opposition_deg / com_inside_mm / press_frac", "type": "...", "description": "the measurements of the judged state"},
        {"name": "seated / lifted", "type": "dict", "description": "the readings at the clamp and after the lift (with rose_mm)"},
    ],
    "status": "tested",
    "entrypoint": "grasp_opposition_test",
    "requirements": ["mujoco>=3.1", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["grasp", "opposition", "contacts", "stress-test"],
}


def _reading(GR, case, run, min_force_N, size):
    import numpy as np
    d = run["data"]
    cons = GR.contacts_over(case, run, seconds=0.1, samples=5)
    groups = {}
    for c in cons:
        g = groups.setdefault(GR.link_of_geom(c["geom"]), {"F": 0.0, "n": np.zeros(3), "p": np.zeros(3)})
        g["F"] += c["force"]
        g["n"] += c["normal"] * c["force"]
        g["p"] += c["pos"] * c["force"]
    rows = []
    for k, g in groups.items():
        if g["F"] < float(min_force_N):
            continue
        n = g["n"] / max(np.linalg.norm(g["n"]), 1e-12)
        rows.append({"link": k, "force_N": round(g["F"], 2), "normal": [round(float(v), 3) for v in n],
                     "point": [round(float(v), 4) for v in g["p"] / g["F"]]})
    com = np.asarray(d.subtree_com[case.root] if case.root is not None else d.xipos[case.bid], float)
    # the opposition of two groups is the angle between their normals, or, when the normal forces share a common push
    # (fingertips on a rim pressing down and in, the weight carried by friction), the angle between their parts across
    # that push: pads on opposite sides of a rim squeeze it although both also press down
    N = np.asarray([r["normal"] for r in rows], float).reshape(-1, 3)
    Fw = np.asarray([r["force_N"] for r in rows], float)
    net = (N * Fw[:, None]).sum(axis=0) if len(rows) else np.zeros(3)
    u = net / np.linalg.norm(net) if len(rows) and np.linalg.norm(net) > 0.2 * Fw.sum() else None
    best = None
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            ni, nj = N[i], N[j]
            ang = float(np.degrees(np.arccos(np.clip(ni @ nj, -1.0, 1.0))))
            across = None
            if u is not None:
                pi, pj = ni - (ni @ u) * u, nj - (nj @ u) * u
                if np.linalg.norm(pi) >= 0.3 and np.linalg.norm(pj) >= 0.3:
                    across = float(np.degrees(np.arccos(np.clip((pi / np.linalg.norm(pi)) @ (pj / np.linalg.norm(pj)), -1.0, 1.0))))
            opp = max(ang, across or 0.0)
            si = float((com - np.asarray(rows[i]["point"])) @ ni)
            sj = float((com - np.asarray(rows[j]["point"])) @ nj)
            if best is None or opp > best["opposition_deg"] + 1e-9:
                best = {"pair": [rows[i]["link"], rows[j]["link"]], "opposition_deg": round(opp, 1), "normal_angle_deg": round(ang, 1),
                        "across_common_push_deg": None if across is None else round(across, 1),
                        "com_inside_mm": [round(si * 1000, 1), round(sj * 1000, 1)]}
    from tool_library.common import hand as H
    pf = H.press_fraction(cons) if cons else 1.0
    return {"groups": rows, "best_pair": best, "press_frac": round(float(pf), 3),
            "total_force_N": round(sum(c["force"] for c in cons), 2), "com": [round(float(v), 4) for v in com]}


def grasp_opposition_test(scene, pose, min_opposition_deg=100.0, min_force_N=0.3, max_press_frac=0.7, com_margin_frac=0.1,
                          lift_m=0.03, approach=True, side=None, grasp_body=None):
    from tool_library.common import grasp_robust as GR
    if not (0.0 < min_opposition_deg <= 180.0):
        raise ValueError("min_opposition_deg must be in (0, 180]")
    spec = GR.load_scene(scene, side=side, grasp_body=grasp_body)
    pose = GR.load_pose(pose)
    case = GR.Case(spec)
    run = GR.close_like_executor(case, pose, approach=approach)
    d = run["data"]
    size = GR.object_size_m(case, d)
    tol_mm = float(com_margin_frac) * size * 1000
    seat = _reading(GR, case, run, min_force_N, size)
    body = case.model.body(case.bid).name
    lifted, judged, reading = None, "seated", seat
    if lift_m and lift_m > 0 and case.root is not None:
        lf = GR.lift_after(case, run, float(lift_m))
        lifted = dict(_reading(GR, case, run, min_force_N, size), rose_mm=lf["rose_mm"], rose=lf["ok"])
        if lf["ok"]:
            # once the object hangs in the hand, only the hand holds it: the contacts that count (a claw that cradles a toy
            # touches it only after the lift; a pinch that leans the object on the table loses it here)
            judged, reading = "lifted", lifted
    issues = []
    best = reading["best_pair"]
    loaded = [g["link"] for g in reading["groups"]]
    where = "with the object lifted" if judged == "lifted" else "after the clamp, with the object on its support"
    note = None
    if lifted is not None and not lifted["rose"]:
        # a part on a thread or a hinge, or a grasp that does not lift: the opposition of the clamp is still the question
        note = (f"{body} did not rise with the hand ({lifted['rose_mm']} of {lift_m * 1000:.0f} mm), so the seated contacts are judged; "
                "grasp_lift_test says whether it should")
    if len(loaded) < 2:
        issues.append(f"only {loaded or 'no link'} pushes on {body} with {min_force_N:g} N or more {where}: a hook or a single "
                      "contact holds nothing against a pull in the other direction; put an opposing pad (the thumb against the fingers, "
                      "or the palm against the fingertips) on the far side of the object")
    else:
        if best["opposition_deg"] < min_opposition_deg:
            issues.append(f"the most opposed contacts ({best['pair'][0]} and {best['pair'][1]}) push only {best['opposition_deg']:.0f} deg "
                          f"apart {where} (need {min_opposition_deg:g}): they do not squeeze {body} between them; turn the wrist so the pads "
                          "meet opposite faces")
        if min(best["com_inside_mm"]) < -tol_mm:
            k = 0 if best["com_inside_mm"][0] < best["com_inside_mm"][1] else 1
            issues.append(f"the centre of mass of {body} lies {-best['com_inside_mm'][k]:.1f} mm outside the contact plane of "
                          f"{best['pair'][k]} {where}: the contacts do not span the object; move the grip so the pads straddle its centre")
    # a common push of the normal forces means leaning only while the object still rests on its support; once it hangs in
    # the hand the push is what friction carries (a rim held from above), and the opposition above judges the squeeze
    if judged == "seated" and reading["press_frac"] > max_press_frac and len(loaded) >= 1:
        issues.append(f"the hand's contact forces on {body} mostly push one way {where} (press fraction {reading['press_frac']:.2f} > "
                      f"{max_press_frac:g}): the object rests on the fingers or is leaned on its support instead of squeezed; close the "
                      "fingers around it from opposite sides")
    return {"ok": not issues, "issues": issues, "note": note, "object": body, "judged": judged, "groups": reading["groups"], "best_pair": best,
            "opposition_deg": best["opposition_deg"] if best else 0.0, "com_inside_mm": best["com_inside_mm"] if best else None,
            "press_frac": reading["press_frac"], "total_force_N": reading["total_force_N"], "object_size_mm": round(size * 1000, 1),
            "seated": seat, "lifted": lifted,
            "limits": {"min_opposition_deg": min_opposition_deg, "min_force_N": min_force_N, "max_press_frac": max_press_frac,
                       "com_margin_mm": round(tol_mm, 1)}}
