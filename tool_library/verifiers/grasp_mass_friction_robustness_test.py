TOOL = {
    "name": "grasp_mass_friction_robustness_test",
    "title": "Mass and friction margin test",
    "type": "verifier",
    "category": "Grasp Stress Tests",
    "summary": "Make the grasped object heavier (default x1.3) and its contacts with the hand slipperier (default x0.7), then run the executor's grasp, a lift, a hold and five carry moves; fail when the grasp no longer holds.",
    "description": "Reconstructed objects carry estimated masses and friction coefficients, and the real ones differ: a grasp that holds exactly at the estimate is one surprise away from dropping the object. The test scales the mass and inertia of every body the lift carries by mass_scale and the sliding and torsional friction of the grasped geoms and of their hand contact pairs by friction_scale, then drives the executor's grasp (open-hand descent, close, torque-capped clamp), lifts lift_m, holds hold_s (slip rate below slip_max_mm_s) and carries the object shake_m out and back along up, forward, back, left and right in move_s per leg (drift in the wrist frame below drift_max_mm). With compare_nominal the same ladder is first run unscaled, so the issue can say whether the margin or the grasp itself is the problem. A part on a fixed asset cannot be lifted, so the test reports applicable=False with ok=True.",
    "inputs": [
        {"name": "scene", "type": "dict | path", "description": "hand scene: dict, hand_scene_from_recon JSON, or an objects MJCF"},
        {"name": "pose", "type": "dict | path", "description": "grasp pose or its JSON"},
        {"name": "mass_scale / friction_scale", "type": "float", "description": "multipliers on the carried mass and the hand-object friction (default 1.3, 0.7)"},
        {"name": "lift_m / hold_s", "type": "float", "description": "lift height (default 0.10) and hold duration (default 0.8)"},
        {"name": "shake_m / move_s", "type": "float", "description": "carry amplitude (default 0.04) and seconds per leg (default 0.35)"},
        {"name": "slip_max_mm_s / drift_max_mm", "type": "float", "description": "pass limits (default 10 mm/s, 20 mm)"},
        {"name": "compare_nominal", "type": "bool", "description": "also run the unscaled ladder (default True)"},
        {"name": "approach / side / grasp_body", "type": "...", "description": "descend from the pre-grasp (default True); scene overrides"},
    ],
    "outputs": [
        {"name": "ok", "type": "bool", "description": "the scaled object is lifted, held and carried"},
        {"name": "applicable", "type": "bool", "description": "False for a part on a fixed asset"},
        {"name": "issues", "type": "list", "description": "which phase failed under the scaling and what to change"},
        {"name": "scaled / nominal", "type": "dict", "description": "per ladder: lift, hold, moves (drift_mm per direction), first_fail, ok"},
        {"name": "scaling", "type": "dict", "description": "the multipliers, the carried mass before and after, how many pairs were scaled"},
    ],
    "status": "tested",
    "entrypoint": "grasp_mass_friction_robustness_test",
    "requirements": ["mujoco>=3.1", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["grasp", "robustness", "mass", "friction", "stress-test"],
}

MOVES = (("up", (0.0, 0.0, 1.0)), ("fwd", (0.0, 1.0, 0.0)), ("back", (0.0, -1.0, 0.0)), ("left", (-1.0, 0.0, 0.0)), ("right", (1.0, 0.0, 0.0)))



REPORT_KEYS = ('ok', 'applicable', 'issues', 'note', 'object', 'scaled', 'nominal', 'scaling')

def _ladder(GR, case, pose, lift_m, hold_s, shake_m, move_s, slip_max_mm_s, drift_max_mm, approach):
    import numpy as np
    run = GR.close_like_executor(case, pose, approach=approach)
    d, g = run["data"], run["grip"]
    lf = GR.lift_after(case, run, lift_m)
    hm = GR.hold_measure(case, run, hold_s)
    out = {"lift": lf, "hold": hm, "moves": [], "first_fail": None, "seated": run["seated"]}
    if not lf["ok"]:
        out["first_fail"] = "lift"
    elif not (hm["slip_mm_s"] < slip_max_mm_s and hm["fingers"]):
        out["first_fail"] = "hold"
    p_ref, _ = case.obj_in_wrist(d)
    cur = run["pos"].copy()
    for nm, v in MOVES:
        v = np.asarray(v, float) * float(shake_m)
        e = GR.leg(case, d, cur, cur + v, run["R"], run["R"], g["press"], g["press"], move_s, None, run["euler"])
        GR.leg(case, d, cur + v, cur, run["R"], run["R"], g["press"], g["press"], move_s, None, e)
        p, _ = case.obj_in_wrist(d)
        drift = float(np.linalg.norm(p - p_ref)) * 1000
        ok = bool(drift < drift_max_mm and case.contacts(d)["n_finger"] >= 1)
        out["moves"].append({"name": nm, "drift_mm": round(drift, 1), "ok": ok})
        if not ok and out["first_fail"] is None:
            out["first_fail"] = f"move_{nm}"
    out["ok"] = out["first_fail"] is None
    return out


def grasp_mass_friction_robustness_test(scene, pose, mass_scale=1.3, friction_scale=0.7, lift_m=0.10, hold_s=0.8, shake_m=0.04,
                                        move_s=0.35, slip_max_mm_s=10.0, drift_max_mm=20.0, compare_nominal=True, approach=True,
                                        side=None, grasp_body=None):
    from tool_library.common import grasp_robust as GR
    if mass_scale <= 0 or friction_scale <= 0:
        raise ValueError("mass_scale and friction_scale must be positive")
    spec = GR.load_scene(scene, side=side, grasp_body=grasp_body)
    pose = GR.load_pose(pose)
    case = GR.Case(spec)
    body = case.model.body(case.bid).name
    if case.root is None:
        return GR.complete({"ok": True, "applicable": False, "issues": [], "object": body,
                "note": f"{body} is a part of a fixed asset and cannot be lifted"}, REPORT_KEYS)
    args = (lift_m, hold_s, shake_m, move_s, slip_max_mm_s, drift_max_mm, approach)
    nominal = _ladder(GR, case, pose, *args) if compare_nominal else None
    m0 = case.mass
    scaled_case = GR.Case(spec)
    info = GR.scale_grasped(scaled_case, mass_scale, friction_scale)
    scaled = _ladder(GR, scaled_case, pose, *args)
    issues = []
    if not scaled["ok"]:
        ff = scaled["first_fail"]
        detail = {"lift": f"rose {scaled['lift']['rose_mm']} of {lift_m * 1000:.0f} mm",
                  "hold": f"slipped {scaled['hold']['slip_mm_s']} mm/s, fingers {scaled['hold']['fingers']}"}.get(
            ff, next((f"drifted {mv['drift_mm']} mm" for mv in scaled["moves"] if not mv["ok"]), ""))
        if nominal is not None and not nominal["ok"]:
            issues.append(f"the grasp fails at {nominal['first_fail']} even at the nominal mass and friction, and at {ff} with mass "
                          f"x{mass_scale:g} / friction x{friction_scale:g} ({detail}): fix the grasp itself first (grasp_lift_test, grasp_shake_test)")
        else:
            baseline = ("although the nominal object is held" if nominal is not None
                        else "the nominal grasp was not tested")
            issues.append(f"{body} with mass x{mass_scale:g} ({m0:.3f} -> {scaled_case.mass:.3f} kg) and hand friction x{friction_scale:g} "
                          f"is lost at {ff} ({detail}); {baseline}: the grasp has no margin for the "
                          "reconstruction's mass and friction estimates. Put more pads on it (a three-finger pinch or a wrap), "
                          "move the pads toward the centre of mass, or press deeper (press_depth_rad).")
    elif nominal is not None and not nominal["ok"]:
        issues.append(f"the grasp fails at {nominal['first_fail']} at the nominal mass and friction even though the scaled "
                      "trial passes; success under changed parameters does not validate the original grasp")
    ok = bool(scaled["ok"] and (nominal is None or nominal["ok"]))
    return GR.complete({"ok": ok, "applicable": True, "issues": issues, "object": body, "scaled": scaled, "nominal": nominal,
            "scaling": {"mass_scale": mass_scale, "friction_scale": friction_scale, "mass_kg": [round(m0, 4), info["mass_kg"]],
                        "pairs_scaled": info["pairs_scaled"]}}, REPORT_KEYS)
