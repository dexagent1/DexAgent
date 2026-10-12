TOOL = {
    "name": "detachable_part_release_test",
    "title": "Detachable part release test",
    "type": "verifier",
    "category": "Reconstruction Mechanism Tests",
    "summary": "Force-only trial of a part (a pen cap, a plug, a press-fit lid) that is welded to a press-fit slide until it has been pulled its release travel: it must stay seated under gravity and a brief tug, stay coaxial while welded, slide to the release travel under a hand-sized pull, and, once the weld is switched off by the release rule, leave the base freely.",
    "description": (
        "The asset carries the removable part as its own free body, welded by an equality constraint to a carrier "
        "on a slide joint of the base (the press fit is the slide's frictionloss); an executor switches the weld off "
        "once the slide has travelled release_travel_m. This verifier emulates that rule with forces only, with the "
        "base held in place (a free base is re-pinned every step; a fixed base needs nothing). Four trials, each "
        "from the initial state: (1) seated hold: gravity turned to pull the part straight off along the slide axis "
        "for hold_s, the slide must stay below a quarter of the release travel; (2) brief tug: a pull of "
        "sub_force_frac times the slide's frictionloss for sub_pull_s must not reach the release travel; (3) weld "
        "holds coaxial: a sideways force with the weld active must not move the part off the axis by more than "
        "max_coaxial_drift_m; (4) pull-off: pull_force_N along the axis until the slide reaches release_travel_m "
        "(within pull_s), then the weld is switched off and the same sideways force must carry the part at least "
        "separate_m off the axis. When expose_site is given (a nib tip, a nozzle), it must lie inside one of the "
        "part's geoms at the start (covered) and outside all of them after the pull-off (exposed). The sideways "
        "direction is the one perpendicular to the axis closest to world up, so the part is lifted, not pushed into "
        "a support. Only slide joints are judged; geoms other than box, sphere, cylinder, capsule and ellipsoid are "
        "treated as their bounding sphere for the covered/exposed test."
    ),
    "inputs": [
        {"name": "xml", "type": "path", "description": "MJCF of the asset (or a scene) containing the base, the slide, the weld and the free part."},
        {"name": "equality", "type": "str", "description": "Name of the weld equality that holds the part on the carrier (the release rule switches it off)."},
        {"name": "joint", "type": "str", "description": "Name of the press-fit slide joint the carrier moves on (positive q = off)."},
        {"name": "part_body", "type": "str", "description": "Name of the removable part's free body."},
        {"name": "release_travel_m", "type": "float", "description": "Slide travel at which the release rule frees the part, metres (> 0)."},
        {"name": "expose_site", "type": "str | None", "description": "Optional site the part covers when seated and must uncover once removed (a nib tip). Default None."},
        {"name": "pull_force_N", "type": "float", "description": "Pull along the axis in the pull-off trial, newtons. Default 4.0."},
        {"name": "pull_s", "type": "float", "description": "Time allowed to reach the release travel, seconds. Default 1.5."},
        {"name": "sub_force_frac", "type": "float", "description": "Brief tug as a fraction of the slide's frictionloss. Default 0.3."},
        {"name": "sub_pull_s", "type": "float", "description": "Duration of the brief tug, seconds. Default 0.3."},
        {"name": "hold_s", "type": "float", "description": "Seated-hold duration with gravity along the axis, seconds. Default 2.0."},
        {"name": "lateral_force_N", "type": "float", "description": "Sideways force on the part (weld-active and after release), newtons. Default 0.5."},
        {"name": "lateral_s", "type": "float", "description": "Duration of the sideways force, seconds. Default 0.5."},
        {"name": "separate_m", "type": "float", "description": "Required off-axis distance of the freed part, metres. Default 0.02."},
        {"name": "max_coaxial_drift_m", "type": "float", "description": "Allowed off-axis drift while the weld is active, metres. Default 0.003."},
        {"name": "cover_margin_m", "type": "float", "description": "Margin added to the part's geoms in the covered/exposed test, metres. Default 0.0005."},
        {"name": "out_json", "type": "path | None", "description": "Optional path for the JSON report."},
    ],
    "outputs": [
        {"name": "ok", "type": "bool", "description": "All four trials (and the cover/expose test when asked) passed."},
        {"name": "issues", "type": "list[str]", "description": "What failed and which parameter of the asset or the test to change."},
        {"name": "verdict", "type": "str", "description": "One sentence: DETACH OK or DETACH FAILED with the numbers."},
        {"name": "measured", "type": "dict", "description": "seated_hold {q_end}, brief_tug {force_N, q_max}, weld_holds_coaxial {perp_drift_m}, pull_off {released, t_release_s, q_at_release, perp_separation_m, t_separate_s, covered_before, exposed_after}, axis_world, hold_force_N."},
    ],
    "status": "tested",
    "entrypoint": "detachable_part_release_test",
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["cap", "uncap", "detach", "release", "weld", "press fit", "slide joint", "plug", "lid", "verifier"],
    "used_in": ["Drawing and Wiping"],
}

import json
import math
from pathlib import Path

import mujoco
import numpy as np


def _perp_up(axis):
    up = np.array([0.0, 0.0, 1.0])
    v = up - axis * float(axis @ up)
    if np.linalg.norm(v) < 1e-6:
        v = np.array([1.0, 0.0, 0.0]) - axis * float(axis[0])
    return v / np.linalg.norm(v)


def _inside_geom(m, d, g, p, margin):
    loc = d.geom_xmat[g].reshape(3, 3).T @ (p - d.geom_xpos[g])
    t, s = m.geom_type[g], m.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        return bool(np.all(np.abs(loc) <= s + margin))
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return bool(np.linalg.norm(loc) <= s[0] + margin)
    if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return bool(abs(loc[2]) <= s[1] + margin and math.hypot(loc[0], loc[1]) <= s[0] + margin)
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        z = float(np.clip(loc[2], -s[1], s[1]))
        return bool(np.linalg.norm(loc - np.array([0.0, 0.0, z])) <= s[0] + margin)
    if t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        return bool(np.sum((loc / (s + margin)) ** 2) <= 1.0)
    return bool(np.linalg.norm(p - d.geom_xpos[g]) <= m.geom_rbound[g] + margin)


def _covered(m, d, part_id, site_id, margin):
    if site_id is None:
        return None
    p = d.site_xpos[site_id].copy()
    return any(_inside_geom(m, d, g, p, margin) for g in range(m.ngeom) if m.geom_bodyid[g] == part_id)


def _report(out, out_json):
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(out, indent=1))
    return out


def detachable_part_release_test(xml, equality, joint, part_body, release_travel_m, expose_site=None,
                                 pull_force_N=4.0, pull_s=1.5, sub_force_frac=0.3, sub_pull_s=0.3, hold_s=2.0,
                                 lateral_force_N=0.5, lateral_s=0.5, separate_m=0.02, max_coaxial_drift_m=0.003,
                                 cover_margin_m=0.0005, out_json=None):
    for k, v in (("release_travel_m", release_travel_m), ("pull_force_N", pull_force_N), ("pull_s", pull_s),
                 ("hold_s", hold_s), ("lateral_force_N", lateral_force_N), ("lateral_s", lateral_s),
                 ("separate_m", separate_m), ("sub_pull_s", sub_pull_s)):
        if not float(v) > 0:
            raise ValueError(f"{k} must be positive, got {v}")
    if not 0.0 <= float(sub_force_frac) < 1.0:
        raise ValueError(f"sub_force_frac must be in [0, 1), got {sub_force_frac}")
    issues = []
    base_out = {"ok": False, "issues": issues, "verdict": "", "measured": {}}
    try:
        m = mujoco.MjModel.from_xml_path(str(xml))
    except Exception as e:  # a bad model is a verdict
        issues.append(f"model does not compile: {e}")
        base_out["verdict"] = "DETACH FAILED - the model does not compile"
        return _report(base_out, out_json)
    eid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, equality)
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint)
    pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, part_body)
    sid = None
    if expose_site is not None:
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, expose_site)
        if sid < 0:
            issues.append(f"expose_site '{expose_site}' not found in the model")
    if eid < 0:
        issues.append(f"equality '{equality}' not found: the part must be held by a named weld the release rule can switch off")
    if jid < 0:
        issues.append(f"joint '{joint}' not found")
    elif m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_SLIDE:
        issues.append(f"joint '{joint}' is not a slide joint: this test pulls the part along a press-fit slide")
    if pid < 0:
        issues.append(f"part_body '{part_body}' not found")
    if issues:
        base_out["verdict"] = "DETACH FAILED - " + issues[0]
        return _report(base_out, out_json)
    pfree = [j for j in range(m.njnt) if m.jnt_bodyid[j] == pid and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    if not pfree:
        issues.append(f"part_body '{part_body}' has no free joint: once released it cannot leave the base; build it as a free body welded to the carrier")
    if m.eq_type[eid] != mujoco.mjtEq.mjEQ_WELD:
        issues.append(f"equality '{equality}' is not a weld")
    jbody = m.jnt_bodyid[jid]
    if m.eq_objtype[eid] == mujoco.mjtObj.mjOBJ_BODY:
        ends = {int(m.eq_obj1id[eid]), int(m.eq_obj2id[eid])}
        if pid not in ends:
            issues.append(f"equality '{equality}' does not involve '{part_body}'")
        other = (ends - {pid}).pop() if len(ends) == 2 and pid in ends else None
        if other is not None:
            b, under = other, False
            while True:
                if b == jbody:
                    under = True
                    break
                if b == 0:
                    break
                b = m.body_parentid[b]
            if not under:
                issues.append(f"the weld's other body is not moved by joint '{joint}': pulling the slide would not carry the part")
    if issues:
        base_out["verdict"] = "DETACH FAILED - " + issues[0]
        return _report(base_out, out_json)
    root = m.body_rootid[jbody]
    bfree = [j for j in range(m.njnt) if m.jnt_bodyid[j] == root and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    badr = m.jnt_qposadr[bfree[0]] if bfree else None
    bdadr = m.jnt_dofadr[bfree[0]] if bfree else None
    qadr = m.jnt_qposadr[jid]
    hold_force = float(m.dof_frictionloss[m.jnt_dofadr[jid]])
    g0 = m.opt.gravity.copy()
    d = mujoco.MjData(m)
    dt = m.opt.timestep

    def reset():
        m.opt.gravity[:] = g0
        mujoco.mj_resetData(m, d)
        if m.nkey:
            mujoco.mj_resetDataKeyframe(m, d, 0)
        d.eq_active[eid] = 1
        mujoco.mj_forward(m, d)
        base_q = d.qpos[badr:badr + 7].copy() if badr is not None else None
        return base_q

    def step(base_q):
        mujoco.mj_step(m, d)
        if base_q is not None:
            d.qpos[badr:badr + 7] = base_q
            d.qvel[bdadr:bdadr + 6] = 0.0
            mujoco.mj_forward(m, d)
        return bool(np.all(np.isfinite(d.qpos)))

    def axis_now():
        a = d.xaxis[jid].copy()
        return a / np.linalg.norm(a)

    def perp_offset(p_ref, a):
        v = d.xpos[pid] - p_ref
        return float(np.linalg.norm(v - a * float(v @ a)))

    measured = {"hold_force_N": round(hold_force, 4), "release_travel_m": float(release_travel_m)}
    rng = m.jnt_range[jid] if m.jnt_limited[jid] else None
    if rng is not None and float(release_travel_m) > float(rng[1]) + 1e-9:
        issues.append(f"release_travel_m {release_travel_m} lies beyond the slide's upper limit {float(rng[1]):.4f}: the part can never be released; widen the joint range")

    # 1. seated hold: gravity straight along the off direction
    base_q = reset()
    a = axis_now()
    measured["axis_world"] = [round(float(x), 4) for x in a]
    m.opt.gravity[:] = a * float(np.linalg.norm(g0) or 9.81)
    finite = True
    for _ in range(int(round(hold_s / dt))):
        finite &= step(base_q)
    q_hold = float(d.qpos[qadr])
    ok1 = finite and q_hold < 0.25 * float(release_travel_m)
    measured["seated_hold"] = {"q_end": round(q_hold, 5), "ok": bool(ok1)}
    if not ok1:
        issues.append(f"under gravity along its axis the part slid {q_hold:.4f} m (limit {0.25 * float(release_travel_m):.4f}): raise the slide frictionloss (hold force) or armature")

    # 2. brief tug below the hold force
    base_q = reset()
    a = axis_now()
    f_sub = float(sub_force_frac) * hold_force
    qmax, finite = 0.0, True
    for _ in range(int(round(sub_pull_s / dt))):
        d.xfrc_applied[pid, :3] = a * f_sub
        finite &= step(base_q)
        qmax = max(qmax, float(d.qpos[qadr]))
    d.xfrc_applied[:] = 0.0
    ok2 = finite and qmax < float(release_travel_m)
    measured["brief_tug"] = {"force_N": round(f_sub, 4), "q_max": round(qmax, 5), "ok": bool(ok2)}
    if not ok2:
        issues.append(f"a brief {f_sub:.2f} N tug already reached the release travel (q {qmax:.4f}): raise the slide frictionloss or armature")

    # 3. sideways force with the weld active
    base_q = reset()
    a = axis_now()
    p_ref = d.xpos[pid].copy()
    lat = _perp_up(a)
    covered_before = _covered(m, d, pid, sid, cover_margin_m)
    drift, finite = 0.0, True
    for _ in range(int(round(lateral_s / dt))):
        d.xfrc_applied[pid, :3] = lat * float(lateral_force_N)
        finite &= step(base_q)
        drift = max(drift, perp_offset(p_ref, a))
    d.xfrc_applied[:] = 0.0
    ok3 = finite and drift <= float(max_coaxial_drift_m)
    measured["weld_holds_coaxial"] = {"lateral_force_N": float(lateral_force_N), "perp_drift_m": round(drift, 5), "ok": bool(ok3)}
    if not ok3:
        issues.append(f"with the weld active a {lateral_force_N} N sideways force moved the part {drift:.4f} m off the axis: stiffen the weld solref")

    # 4. pull-off, release rule, then carry it away sideways
    base_q = reset()
    a = axis_now()
    p_ref = d.xpos[pid].copy()
    t_rel, q_rel, finite = None, None, True
    for i in range(int(round(pull_s / dt))):
        d.xfrc_applied[pid, :3] = a * float(pull_force_N)
        finite &= step(base_q)
        if float(d.qpos[qadr]) >= float(release_travel_m):
            t_rel, q_rel = (i + 1) * dt, float(d.qpos[qadr])
            d.eq_active[eid] = 0
            break
    released = t_rel is not None
    sep, exposed_after, t_sep = 0.0, None, None
    if released:
        # the hand that pulled it carries it sideways; stop pushing once it is clear of the axis
        for i in range(int(round(lateral_s / dt))):
            d.xfrc_applied[pid, :3] = lat * float(lateral_force_N)
            finite &= step(base_q)
            sep = perp_offset(p_ref, a)
            if sep >= float(separate_m):
                t_sep = (i + 1) * dt
                break
        d.xfrc_applied[:] = 0.0
        exposed_after = None if sid is None else (not _covered(m, d, pid, sid, cover_margin_m))
    ok4 = finite and released and sep >= float(separate_m)
    if sid is not None:
        ok4 = ok4 and bool(covered_before) and bool(exposed_after)
    measured["pull_off"] = {"force_N": float(pull_force_N), "released": bool(released),
                            "t_release_s": None if t_rel is None else round(t_rel, 4),
                            "q_at_release": None if q_rel is None else round(q_rel, 5),
                            "perp_separation_m": round(sep, 4), "t_separate_s": None if t_sep is None else round(t_sep, 4),
                            "covered_before": covered_before,
                            "exposed_after": exposed_after, "ok": bool(ok4)}
    if not released:
        issues.append(f"a {pull_force_N} N pull did not slide the part to the release travel {release_travel_m} m within {pull_s} s (q reached {float(d.qpos[qadr]):.4f}; hold force {hold_force:.2f} N): lower the slide frictionloss/hold_force_N, or check the joint range")
    elif sep < float(separate_m):
        issues.append(f"after the weld was switched off the part moved only {sep:.4f} m off the axis under {lateral_force_N} N: check its contacts with the base (exclude the base-part pair) and that it is free")
    if sid is not None and not covered_before:
        issues.append(f"the seated part does not cover '{expose_site}': lengthen the part (cap) or move the site")
    if sid is not None and released and not exposed_after:
        issues.append(f"after removal '{expose_site}' is still inside the part")
    if not finite:
        issues.append("the simulation diverged: check masses, armature and the weld solref")
    ok = not issues and ok1 and ok2 and ok3 and ok4
    if ok:
        verdict = (f"DETACH OK - seated under gravity (q {q_hold * 1000:.1f} mm), held a {f_sub:.2f} N tug, stayed coaxial while welded "
                   f"({drift * 1000:.1f} mm), released at {q_rel * 1000:.1f} mm after {t_rel:.2f} s of a {pull_force_N} N pull and was {sep * 100:.1f} cm off the axis {t_sep:.2f} s after the weld was switched off"
                   + ("; the site went from covered to exposed" if sid is not None else ""))
    else:
        verdict = "DETACH FAILED - " + "; ".join(issues)
    out = {"ok": bool(ok), "issues": issues, "verdict": verdict, "measured": measured}
    return _report(out, out_json)
