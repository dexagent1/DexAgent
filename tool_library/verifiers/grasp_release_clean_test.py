TOOL = {
    "name": "grasp_release_clean_test",
    "title": "Clean release test",
    "type": "verifier",
    "category": "Grasp Stress Tests",
    "summary": "Grasp the object, set it back down and let go exactly as the executor's release does; fail when opening the hand or backing away flings, drags, lifts or tips the object.",
    "description": "A grasp is also judged by how it ends: fingers that open INTO the object, a squeezed object that springs out of a releasing clamp, or a hand that backs away while a pad still drags on the object knock the placed object off its target (a release flick). The hand closes as the executor drives it (pre-grasp, descent, close, torque-capped clamp), lifts lift_m, comes back down until the object is set_down_m above where it rested (the executor's set-down clearance, measured on the object so in-hand slip during the lift does not ram it into its support), then releases as SubgoalPlanner._release does: the fingers go to the executor's open shape in 0.5 s (+0.3 s), the wrist backs off retreat_m along retreat_dir in 0.8 s (+0.2 s), and the scene settles for settle_s with the hand away. Measured from the moment before the fingers open: the object's horizontal slide and its spin about the vertical (an object that merely drops onto its support neither slides nor spins), its tip (its attitude against gravity at the end compared with how it rested before the grasp, so settling flat from a tilted hold is not a tip but landing on its side is), its largest rise (dragged up with the hand), its peak speed, and whether the hand still touches it at the end. Fails when any passes its limit or the hand is still on it; the issue names the phase (opening or retreat) where the object moved.",
    "inputs": [
        {"name": "scene", "type": "dict | path", "description": "hand scene: dict, hand_scene_from_recon JSON, or an objects MJCF"},
        {"name": "pose", "type": "dict | path", "description": "grasp pose or its JSON (finger_targets_open is the shape the release opens to)"},
        {"name": "lift_m / set_down_m", "type": "float", "description": "lift before the set-down (default 0.03; 0 releases where it closed) and the object's set-down clearance above where it rested (default 0.002)"},
        {"name": "retreat_m / retreat_dir", "type": "float / xyz", "description": "the retreat after opening (default 0.08 m along (0, 0, 1), the executor's)"},
        {"name": "open_targets", "type": "dict | None", "description": "finger targets the hand opens to at the let-go (default the executor's open shape: the pose's finger_targets_open, else open_targets at open_frac)"},
        {"name": "settle_s", "type": "float", "description": "seconds of settling after the retreat (default 0.5)"},
        {"name": "max_shift_mm / max_spin_deg / max_tip_deg / max_rise_mm", "type": "float", "description": "pass limits (default 25 mm, 20 deg, 20 deg, 5 mm: accepted bench releases slide up to 21 mm and spin up to 8 deg as the object drops the last few millimetres)"},
        {"name": "approach / side / grasp_body", "type": "...", "description": "descend from the pre-grasp (default True); scene overrides"},
    ],
    "outputs": [
        {"name": "ok", "type": "bool", "description": "the object stays where it was set down"},
        {"name": "issues", "type": "list", "description": "what moved in which phase and what to change"},
        {"name": "shift_mm / spin_deg / tip_deg / rise_mm / peak_speed_m_s", "type": "float", "description": "horizontal slide and spin about the vertical over the release, tilt against the object's rest attitude before the grasp, largest rise, peak speed"},
        {"name": "phases", "type": "dict", "description": "shift, spin, tip and rotation before the let-go and at the end of the opening, the retreat and the settle"},
        {"name": "touching_after", "type": "list", "description": "hand links still on the object at the end"},
    ],
    "status": "tested",
    "entrypoint": "grasp_release_clean_test",
    "requirements": ["mujoco>=3.1", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["grasp", "release", "place", "stress-test"],
}


def _twist_about_z(R):
    """Rotation angle (deg) of R about the world vertical (swing-twist decomposition)."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return float(abs(np.degrees(2.0 * np.arctan2(z, w))) % 360.0) if (abs(z) + abs(w)) > 1e-12 else 0.0


def grasp_release_clean_test(scene, pose, lift_m=0.03, set_down_m=0.002, retreat_m=0.08, retreat_dir=(0.0, 0.0, 1.0), open_targets=None, settle_s=0.5,
                             max_shift_mm=25.0, max_spin_deg=20.0, max_tip_deg=20.0, max_rise_mm=5.0, approach=True, side=None,
                             grasp_body=None):
    import mujoco
    import numpy as np
    from tool_library.common import grasp_robust as GR
    rd = np.asarray(retreat_dir, float)
    if rd.shape != (3,) or np.linalg.norm(rd) < 1e-9:
        raise ValueError("retreat_dir must be a non-zero xyz vector")
    rd = rd / np.linalg.norm(rd)
    spec = GR.load_scene(scene, side=side, grasp_body=grasp_body)
    pose = GR.load_pose(pose)
    case = GR.Case(spec)
    body = case.model.body(case.bid).name
    _, R_rest = case.obj(case.data0)
    run = GR.close_like_executor(case, pose, approach=approach)
    d, g = run["data"], run["grip"]
    z_rest = float(d.xpos[case.bid][2])
    if lift_m and lift_m > 0 and case.root is not None:
        GR.lift_after(case, run, float(lift_m))
        # back down until the object is set_down_m above where it rested (the place's clearance), measured on the object: an
        # object that slid a few millimetres in the pinch during the lift is set down, not rammed into its support
        dz = float(d.xpos[case.bid][2]) - (z_rest + float(set_down_m))
        down = run["pos"] - np.array([0.0, 0.0, dz])
        e = GR.leg(case, d, run["pos"], down, run["R"], run["R"], g["press"], g["press"], GR.T_LIFT, None, run["euler"])
        e = GR.hold(case, d, down, run["R"], g["press"], 0.1, None, e)
        run["pos"], run["euler"] = down, e
    if open_targets is not None:
        # the let-go shape under test (the executor opens to the grip's open shape, the one the approach used)
        g = dict(g, opened={k: float(v) for k, v in dict(open_targets).items() if k in case.rig.finger_range})
        if not g["opened"]:
            raise ValueError("open_targets: no finger joint of this hand in it")
    p0, R0 = case.obj(d)
    rec = {"rise": 0.0, "speed": 0.0}
    vel = np.zeros(6)
    up = np.array([0.0, 0.0, 1.0])

    def track(dd):
        rec["rise"] = max(rec["rise"], float(dd.xpos[case.bid][2]) - p0[2])
        mujoco.mj_objectVelocity(case.model, dd, mujoco.mjtObj.mjOBJ_BODY, case.bid, vel, 0)
        rec["speed"] = max(rec["speed"], float(np.linalg.norm(vel[3:])))

    def state():
        p, R = case.obj(d)
        dh = p - p0
        # tip: the object's attitude against gravity compared with how it rested before the grasp (set back on the same spot, a
        # clean release leaves it resting the same way up; settling flat from a tilted hold is not a tip). spin: its turn about
        # the vertical while the hand lets go (an object that merely drops does not spin)
        tip = float(np.degrees(np.arccos(np.clip((R_rest.T @ up) @ (R.T @ up), -1.0, 1.0))))
        return {"shift_mm": GR.r2(np.linalg.norm(dh[:2]) * 1000), "dz_mm": GR.r2(dh[2] * 1000), "spin_deg": GR.r2(_twist_about_z(R @ R0.T)),
                "tip_deg": GR.r2(tip), "rot_deg": GR.r2(GR.rot_deg(R0, R))}

    phases = {"before": state()}
    e = GR.leg(case, d, run["pos"], run["pos"], run["R"], run["R"], g["press"], g["opened"], GR.T_OPEN, track, run["euler"])
    e = GR.hold(case, d, run["pos"], run["R"], g["opened"], GR.T_OPEN_HOLD, track, e)
    phases["open"] = dict(state(), rise_mm=GR.r2(rec["rise"] * 1000))
    back = run["pos"] + rd * float(retreat_m)
    e = GR.leg(case, d, run["pos"], back, run["R"], run["R"], g["opened"], g["opened"], GR.T_RETREAT, track, e)
    e = GR.hold(case, d, back, run["R"], g["opened"], GR.T_RETREAT_HOLD, track, e)
    phases["retreat"] = dict(state(), rise_mm=GR.r2(rec["rise"] * 1000))
    GR.hold(case, d, back, run["R"], g["opened"], settle_s, track, e)
    phases["settle"] = state()
    touching = sorted({GR.link_of_geom(c["geom"]) for c in GR.hand_object_contacts(case, d, min_force=0.05)})
    end = phases["settle"]
    shift, spin, tip, rise = end["shift_mm"], end["spin_deg"], end["tip_deg"], rec["rise"] * 1000
    tipped = tip > max_tip_deg and tip > phases["before"]["tip_deg"] + 0.5 * max_tip_deg
    ok = bool(shift <= max_shift_mm and spin <= max_spin_deg and not tipped and rise <= max_rise_mm and not touching)
    issues = []
    if not ok:
        op = phases["open"]
        s_open = max(op["shift_mm"] / max_shift_mm, op["spin_deg"] / max_spin_deg, op["rise_mm"] / max_rise_mm)
        s_end = max(shift / max_shift_mm, spin / max_spin_deg, rise / max_rise_mm)
        culprit = "opening the fingers" if s_open >= 0.5 * s_end else "the retreat"
        what = []
        if shift > max_shift_mm:
            what.append(f"slid {shift:.1f} mm")
        if spin > max_spin_deg:
            what.append(f"spun {spin:.1f} deg about the vertical")
        if tipped:
            what.append(f"tipped {tip:.1f} deg from the way it rested before the grasp")
        if rise > max_rise_mm:
            what.append(f"was dragged up {rise:.1f} mm")
        if touching:
            what.append(f"is still touched by {touching}")
        hint = ("open to a shape whose pads move straight away from the object's faces (finger_targets_open: back the gripping "
                "fingers off along their flexion, never curl an idle finger into it)" if culprit == "opening the fingers" else
                "open further before backing off, or retreat along the pads' release direction rather than across them")
        issues.append(f"releasing {body}: it {', '.join(what)} (limits {max_shift_mm:g} mm, {max_spin_deg:g} deg spin, {max_tip_deg:g} deg tip, "
                      f"{max_rise_mm:g} mm rise), mostly during {culprit}; peak speed {rec['speed']:.2f} m/s. {hint[0].upper() + hint[1:]}.")
    return {"ok": ok, "issues": issues, "object": body, "shift_mm": shift, "spin_deg": spin, "tip_deg": tip, "rise_mm": GR.r2(rise),
            "peak_speed_m_s": GR.r2(rec["speed"], 3), "phases": phases, "touching_after": touching, "seated": run["seated"],
            "limits": {"max_shift_mm": max_shift_mm, "max_spin_deg": max_spin_deg, "max_tip_deg": max_tip_deg, "max_rise_mm": max_rise_mm}}
