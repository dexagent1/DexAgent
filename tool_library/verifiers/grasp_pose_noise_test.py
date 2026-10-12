TOOL = {
    "name": "grasp_pose_noise_test",
    "title": "Grasp pose noise test",
    "type": "verifier",
    "category": "Grasp Stress Tests",
    "summary": "Re-run the executor's close, lift and hold with the hand pose off by a calibration-sized error (default 2 mm and 2 deg) in several random directions; fail when any perturbed grasp drops or slips.",
    "description": "A real arm puts the hand within a couple of millimetres and degrees of its command, so a grasp that only works at the exact pose will fail on the bench as soon as the calibration is off. The test draws n_trials wrist errors of exactly pos_noise_mm and rot_noise_deg in random directions (seeded, in antithetic pairs: +e and -e), applies each to the pose (translation, and a rotation about the wrist origin where a flange error acts, or about the pads' centroid with pivot='pads'), and runs the executor's grasp on each: open-hand descent from the pre-grasp, close, torque-capped clamp, a lift of lift_m (0.8 s, +0.3 s) and a hold of hold_s. A trial passes when the object rose more than half the lift, slipped less than slip_max_mm_s during the hold and a finger still touches it. The unperturbed pose is run first: when it fails the issue says so and points to grasp_lift_test. Passes when at least min_pass_frac of the perturbed trials pass (default all). A part on a fixed asset cannot be lifted, so the test reports applicable=False with ok=True.",
    "inputs": [
        {"name": "scene", "type": "dict | path", "description": "hand scene: dict, hand_scene_from_recon JSON, or an objects MJCF"},
        {"name": "pose", "type": "dict | path", "description": "grasp pose or its JSON"},
        {"name": "pos_noise_mm / rot_noise_deg", "type": "float", "description": "size of the wrist error (default 2 mm, 2 deg)"},
        {"name": "n_trials / seed", "type": "int", "description": "number of perturbed trials (default 6) and the random seed (default 0)"},
        {"name": "lift_m / hold_s", "type": "float", "description": "lift height (default 0.08) and hold duration (default 0.5)"},
        {"name": "slip_max_mm_s", "type": "float", "description": "allowed slip rate during the hold (default 10)"},
        {"name": "min_pass_frac", "type": "float", "description": "fraction of trials that must pass (default 1.0)"},
        {"name": "pivot", "type": "str", "description": "'wrist' (default) or 'pads': where the rotation error acts"},
        {"name": "approach / side / grasp_body", "type": "...", "description": "descend from the pre-grasp (default True); scene overrides"},
    ],
    "outputs": [
        {"name": "ok", "type": "bool", "description": "the nominal grasp and enough perturbed grasps lift and hold"},
        {"name": "applicable", "type": "bool", "description": "False for a part on a fixed asset"},
        {"name": "issues", "type": "list", "description": "which errors break the grasp and what to change"},
        {"name": "nominal", "type": "dict", "description": "the unperturbed trial"},
        {"name": "trials", "type": "list", "description": "per trial: offset_mm, rot_deg (xyz), rose_mm, slip_mm_s, fingers seated and after the hold, approach nudge_mm, hand-table floor_force_N, why (what failed), ok"},
        {"name": "pass_frac / worst", "type": "...", "description": "fraction passed and the worst trial"},
    ],
    "status": "tested",
    "entrypoint": "grasp_pose_noise_test",
    "requirements": ["mujoco>=3.1", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["grasp", "robustness", "calibration", "noise", "stress-test"],
}



REPORT_KEYS = ('ok', 'applicable', 'issues', 'note', 'object', 'nominal', 'trials', 'pass_frac', 'n_passed', 'worst', 'noise')

def _trial(GR, case, pose, lift_m, hold_s, slip_max_mm_s, approach):
    run = GR.close_like_executor(case, pose, approach=approach)
    seated = run["seated"]
    lf = GR.lift_after(case, run, lift_m)
    hm = GR.hold_measure(case, run, hold_s)
    ok = bool(lf["ok"] and hm["slip_mm_s"] < slip_max_mm_s and len(hm["fingers"]) >= 1)
    why = None
    if not ok:
        # what physically went wrong, so a flag can be checked: the open hand knocked the object on its way down, the
        # pads stopped on the table, the pads closed past the object, or the object slipped out on the lift
        if run["nudge_mm"] > 5.0:
            why = f"the descending open hand knocked the object {run['nudge_mm']:.1f} mm before the fingers closed"
        elif seated["n_finger"] < 2 and run["floor_force_N"] > 1.0:
            why = f"the fingers stopped on the table ({run['floor_force_N']:.1f} N) with {seated['fingers'] or 'no finger'} on the object"
        elif seated["n_finger"] < 2:
            why = f"only {seated['fingers'] or 'no finger'} reached the object: the pads closed beside it"
        else:
            why = f"held by {seated['fingers']} at the clamp but lost on the lift (rose {lf['rose_mm']} mm, slip {hm['slip_mm_s']} mm/s)"
    return {"rose_mm": lf["rose_mm"], "slip_mm_s": hm["slip_mm_s"], "hold_drift_mm": hm["drift_mm"], "fingers": hm["fingers"],
            "contact_force_N": hm["contact_force_N"], "seated": seated["fingers"], "nudge_mm": round(run["nudge_mm"], 1),
            "floor_force_N": run["floor_force_N"], "why": why, "ok": ok}


def grasp_pose_noise_test(scene, pose, pos_noise_mm=2.0, rot_noise_deg=2.0, n_trials=6, seed=0, lift_m=0.08, hold_s=0.5,
                          slip_max_mm_s=10.0, min_pass_frac=1.0, pivot="wrist", approach=True, side=None, grasp_body=None):
    import numpy as np
    from tool_library.common import grasp_robust as GR
    from tool_library.common import hand as H
    if pivot not in ("wrist", "pads"):
        raise ValueError("pivot must be 'wrist' or 'pads'")
    if int(n_trials) < 1 or pos_noise_mm < 0 or rot_noise_deg < 0:
        raise ValueError("n_trials must be >= 1 and the noise sizes non-negative")
    spec = GR.load_scene(scene, side=side, grasp_body=grasp_body)
    pose = GR.load_pose(pose)
    case = GR.Case(spec)
    body = case.model.body(case.bid).name
    if case.root is None:
        return GR.complete({"ok": True, "applicable": False, "issues": [], "object": body,
                "note": f"{body} is a part of a fixed asset and cannot be lifted; perturb the pose with a task-specific test instead"}, REPORT_KEYS)
    centre = None
    if pivot == "pads":
        d = case.fresh()
        case.place(d, np.asarray(pose["pos"], float), H.euler_to_mat(pose["euler"]), pose["finger_targets"])
        fins = [f for f in (pose.get("grip_fingers") or ["thumb", "index"]) if f in case.rig.fingertip_sites]
        centre = np.mean([d.site_xpos[case.rig.fingertip_sites[f]] for f in fins], axis=0)
    nominal = _trial(GR, case, pose, lift_m, hold_s, slip_max_mm_s, approach)
    rows = []
    for dp, dr in GR.random_perturbations(n_trials, pos_noise_mm, rot_noise_deg, seed):
        pp = GR.perturb_pose(pose, dp, dr, pivot=centre)
        r = _trial(GR, case, pp, lift_m, hold_s, slip_max_mm_s, approach)
        rows.append(dict(offset_mm=[round(float(v) * 1000, 2) for v in dp], rot_deg=[round(float(np.degrees(v)), 2) for v in dr], **r))
    n_ok = sum(r["ok"] for r in rows)
    frac = n_ok / len(rows)
    worst = min(rows, key=lambda r: (r["ok"], r["rose_mm"] or 0.0, -(r["slip_mm_s"] or 0.0)))
    ok = bool(nominal["ok"] and frac >= float(min_pass_frac) - 1e-9)
    issues = []
    if not nominal["ok"]:
        issues.append(f"the unperturbed grasp itself does not lift and hold {body} (rose {nominal['rose_mm']} of {lift_m * 1000:.0f} mm, "
                      f"slip {nominal['slip_mm_s']} mm/s, fingers {nominal['fingers']}): fix the pose first (grasp_lift_test, seat_grasp_for_lift)")
    if frac < float(min_pass_frac) - 1e-9:
        bad = [r for r in rows if not r["ok"]]
        lst = "; ".join(f"offset {r['offset_mm']} mm, turn {r['rot_deg']} deg: {r['why']}" for r in bad[:3])
        issues.append(f"{len(bad)} of {len(rows)} grasps with the hand off by {pos_noise_mm:g} mm / {rot_noise_deg:g} deg fail to lift and hold "
                      f"{body} ({lst}). The grasp has no margin for the arm's calibration error: seat the pads deeper on the object "
                      "(more contact patch, opposing pads below its centre line), or choose a grasp with more fingers or a wider pocket.")
    return GR.complete({"ok": ok, "applicable": True, "issues": issues, "object": body, "nominal": nominal, "trials": rows,
            "pass_frac": round(frac, 3), "n_passed": n_ok, "worst": worst,
            "noise": {"pos_mm": pos_noise_mm, "rot_deg": rot_noise_deg, "pivot": pivot, "seed": seed}}, REPORT_KEYS)
