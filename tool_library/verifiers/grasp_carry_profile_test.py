TOOL = {
    "name": "grasp_carry_profile_test",
    "title": "Carry profile replay test",
    "type": "verifier",
    "category": "Grasp Stress Tests",
    "summary": "Replay the planned carry's own wrist motion (accelerations and orientation changes, read from a trajectory segment, a keyframes JSON or a wrist path) on the grasp in its hand scene and fail when the object slips or turns in the hand.",
    "description": "The generic carry test shakes every grasp the same way; the carry the plan will actually execute can be faster, turn the object upside down or swing it round a corner. The test closes the hand as the executor does (open-hand descent, close, torque-capped clamp), lifts lift_m, then replays the carry: the world wrist path of the acting hand over the selected frames (a trajectory npz with its labels, frames [first, last] or the first run labelled '<side>: carry...'; or keyframes chained with quintic legs; or an explicit wrist path) is applied as motion RELATIVE to its first frame, expressed in the hand's frame, so the object sees the same accelerations and the same orientation sequence relative to gravity as on the bench even when the hand scene's pose was re-anchored by a yaw. The servo commands are interpolated between frames at the source's fps as the executor does, and the wrist chain is re-based when it pitches past 60 deg (as the executor does). With isolate (default) the floor and every other object stop touching the hand and the object from the lift on, and equality welds that tie the object to the rest of the scene are released (the bench carries a cap only after its thread let go), so only the grasp holds it and the replay cannot run it into the hand scene's bystanders. Fails when the object's drift in the wrist frame passes drift_max_mm, its turn passes turn_max_deg, or no finger touches it at the end. Reports the path's peak speed, acceleration and turn rate, the wrist tracking error, and the frame of the worst slip.",
    "inputs": [
        {"name": "scene", "type": "dict | path", "description": "hand scene: dict, hand_scene_from_recon JSON (its bench_scene is used to read qpos), or an objects MJCF"},
        {"name": "pose", "type": "dict | path", "description": "the grasp pose the carry holds, or its JSON"},
        {"name": "carry", "type": "path | dict", "description": "trajectory .npz, keyframes .json, or wrist path {wrist: [[x, y, z, rx, ry, rz], ...], fps}"},
        {"name": "frames", "type": "list | None", "description": "[first, last] frames of the source to replay (default the first '<side>: <phase>' run, with the frame before it)"},
        {"name": "phase / subgoal", "type": "str / int", "description": "label phase to select (default 'carry') and the trajectory's subgoal index to search in"},
        {"name": "bench_scene", "type": "path | None", "description": "scene whose qpos the source holds (default the hand scene's bench_scene; else the npz joint_names)"},
        {"name": "fps", "type": "float | None", "description": "frame rate override (keyframes default 30)"},
        {"name": "lift_m", "type": "float", "description": "lift before the replay (default 0.08)"},
        {"name": "isolate", "type": "bool", "description": "only the hand can touch the object during the replay (default True)"},
        {"name": "drift_max_mm / turn_max_deg", "type": "float", "description": "pass limits in the wrist frame (default 15 mm, 15 deg: accepted bench carries re-run with the library's pad friction drift up to 7.6 mm and turn up to 7.7 deg in the hand)"},
        {"name": "track_max_mm / track_max_deg", "type": "float", "description": "largest wrist tracking error before the replay counts as not reproduced (default 40 mm, 20 deg; a servo trails a moving target by about 30 mm per m/s)"},
        {"name": "approach / side / grasp_body", "type": "...", "description": "descend from the pre-grasp (default True); scene overrides (side also picks the hand in the source)"},
    ],
    "outputs": [
        {"name": "ok", "type": "bool", "description": "the object stays put in the hand through the replayed carry"},
        {"name": "issues", "type": "list", "description": "where it slipped (frame, drift, turn) and what to change"},
        {"name": "drift_mm / turn_deg", "type": "float", "description": "largest drift and turn in the wrist frame"},
        {"name": "worst_frame", "type": "int", "description": "source frame of the largest drift"},
        {"name": "path", "type": "dict", "description": "frames replayed, source, peak speed (m/s), acceleration (m/s^2), turn rate (deg/s), total turn"},
        {"name": "tracking", "type": "dict", "description": "worst wrist tracking error of the replay (mm, deg)"},
        {"name": "lift / fingers_after", "type": "dict", "description": "the lift before the replay and the contacts at the end"},
    ],
    "status": "tested",
    "entrypoint": "grasp_carry_profile_test",
    "requirements": ["mujoco>=3.1", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["grasp", "carry", "trajectory", "slip", "stress-test"],
}



REPORT_KEYS = ('ok', 'applicable', 'issues', 'note', 'object', 'drift_mm', 'turn_deg', 'worst_frame', 'lift', 'fingers_after', 'path', 'tracking', 'wrist_rebased', 'equalities_released', 'limits')

def grasp_carry_profile_test(scene, pose, carry, frames=None, phase="carry", subgoal=None, bench_scene=None, fps=None, lift_m=0.08,
                             isolate=True, drift_max_mm=15.0, turn_max_deg=15.0, track_max_mm=40.0, track_max_deg=20.0, approach=True, side=None,
                             grasp_body=None):
    import mujoco
    import numpy as np
    from tool_library.common import grasp_robust as GR
    from tool_library.common import hand as H
    spec = GR.load_scene(scene, side=side, grasp_body=grasp_body)
    pose = GR.load_pose(pose)
    sd = side or spec.get("side", "right")
    path = GR.wrist_path(carry, sd, frames=frames, phase=phase, subgoal=subgoal,
                         bench_scene=bench_scene or spec.get("bench_scene"), fps=fps)
    kin = GR.path_kinematics(path)
    Pp, Rs, f = path["pos"], path["R"], float(path["fps"])
    Rp = H.euler_to_mat(pose["euler"])
    R0, p0 = Rs[0], Pp[0]
    Rmap = [Rp @ R0.T @ R for R in Rs]                 # the bench's orientation sequence relative to its first frame, in the hand
    case = GR.Case(spec)
    body = case.model.body(case.bid).name
    if case.root is None:
        return GR.complete({"ok": True, "applicable": False, "issues": [], "object": body,
                "note": f"{body} is a part of a fixed asset and is not carried"}, REPORT_KEYS)
    case.set_basis([Rp] + Rmap)
    run = GR.close_like_executor(case, pose, approach=approach)
    d, g = run["data"], run["grip"]
    iso = None
    if isolate:
        # before the lift: a carried object is free by then (a cap whose thread weld the bench released, a part lifted out of
        # its neighbours); afterwards only the hand touches it. Its weight is handed to the hand gradually (a slow lift's load)
        iso = GR.hand_takes_weight(case, run, seconds=0.5, isolate=True, keep_hand_floor=False, release_equalities=True)
    lf = GR.lift_after(case, run, lift_m)
    issues = []
    if not lf["ok"]:
        issues.append(f"the grasp does not lift {body} (rose {lf['rose_mm']} of {lift_m * 1000:.0f} mm): fix it with grasp_lift_test first")
        return GR.complete({"ok": False, "applicable": True, "issues": issues, "object": body, "lift": lf, "path": dict(kin, frames=path["frames"], source=path["source"])}, REPORT_KEYS)
    start = run["pos"].copy()
    Pmap = [start + Rp @ R0.T @ (p - p0) for p in Pp]
    # the servo targets as joint values, continuous in the (possibly re-based) wrist chain
    E, prev = [], run["euler"]
    for R in Rmap:
        prev = case.euler(R, prev)
        E.append(prev)
    E = np.asarray(E)
    ref_p, ref_R = case.obj_in_wrist(d)
    n_sub = max(1, int(round(1.0 / (f * case.dt))))
    worst = {"drift": 0.0, "turn": 0.0, "frame": path["frames"][0], "track_mm": 0.0, "track_deg": 0.0}
    rig = case.rig
    for t in range(len(Pmap) - 1):
        for k in range(n_sub):
            u = (k + 1) / n_sub
            H.hold_ctrl(d, rig, (1 - u) * Pmap[t] + u * Pmap[t + 1], (1 - u) * E[t] + u * E[t + 1], g["press"])
            mujoco.mj_step(case.model, d)
        p, R = case.obj_in_wrist(d)
        dr = float(np.linalg.norm(p - ref_p)) * 1000
        tu = GR.rot_deg(ref_R, R)
        if dr > worst["drift"]:
            worst["drift"], worst["frame"] = dr, path["frames"][0] + t + 1
        worst["turn"] = max(worst["turn"], tu)
        jw = np.array([d.qpos[rig.wrist_qpos[k_]] for k_ in ("x", "y", "z")])
        worst["track_mm"] = max(worst["track_mm"], float(np.linalg.norm(jw - Pmap[t + 1])) * 1000)
        worst["track_deg"] = max(worst["track_deg"], GR.rot_deg(Rmap[t + 1], case.B @ H.euler_to_mat([d.qpos[rig.wrist_qpos[k_]] for k_ in ("rx", "ry", "rz")])))
    GR.hold(case, d, Pmap[-1], Rmap[-1], g["press"], 0.3, None, E[-1])
    p, R = case.obj_in_wrist(d)
    worst["drift"] = max(worst["drift"], float(np.linalg.norm(p - ref_p)) * 1000)
    worst["turn"] = max(worst["turn"], GR.rot_deg(ref_R, R))
    after = case.contacts(d)
    ok = bool(worst["drift"] <= drift_max_mm and worst["turn"] <= turn_max_deg and after["n_finger"] >= 1)
    if not ok:
        issues.append(
            f"{body} drifted {worst['drift']:.1f} mm and turned {worst['turn']:.1f} deg in the hand during the replayed carry "
            f"(worst at frame {worst['frame']}; the path peaks at {kin['peak_acc_m_s2']} m/s^2 and {kin['peak_ang_speed_deg_s']} deg/s and turns "
            f"{kin['turned_deg']} deg; fingers on it after: {after['fingers'] or 'none'}). Slow the carry (duration_s) or reduce its turn, or "
            "choose a grasp whose pads oppose the load: under an upside-down or fast turn the object pulls along the pinch, not into it.")
    if worst["track_mm"] > track_max_mm or worst["track_deg"] > track_max_deg:
        # a position servo trails a moving target by kv/kp x speed (about 30 mm at 1 m/s), on the bench as here; far beyond
        # that the wrist is saturated (40 N / 3 N m) and the replay does not reproduce the path the bench executed
        issues.append(f"the hand scene's wrist did not follow the replay (tracking error {worst['track_mm']:.1f} mm / {worst['track_deg']:.1f} deg, "
                      f"limits {track_max_mm:g} mm / {track_max_deg:g} deg): the path asks more than the wrist servos give (40 N, 3 N m); "
                      "slow the carry down (duration_s)")
        ok = False
    return GR.complete({"ok": ok, "applicable": True, "issues": issues, "object": body, "drift_mm": GR.r2(worst["drift"]), "turn_deg": GR.r2(worst["turn"]),
            "worst_frame": worst["frame"], "lift": lf, "fingers_after": after,
            "path": dict(kin, frames=path["frames"], source=path["source"], fps=f, n_frames=len(Pp), labels=path.get("labels")),
            "tracking": {"pos_mm": GR.r2(worst["track_mm"]), "rot_deg": GR.r2(worst["track_deg"])},
            "wrist_rebased": bool(np.abs(case.B - np.eye(3)).max() > 1e-9),
            "equalities_released": (iso or {}).get("equalities_released", []),
            "limits": {"drift_max_mm": drift_max_mm, "turn_max_deg": turn_max_deg}}, REPORT_KEYS)
