"""Solver parameter sanity check: do the scene's solref / solimp values mean what their author intended at this timestep?"""
from __future__ import annotations

TOOL = {
    "name": "solver_parameter_sanity_check",
    "title": "Solver parameter sanity check",
    "type": "verifier",
    "category": "Scene Compile Checks",
    "summary": "Contact, joint-limit and equality solref/solimp values are consistent with the timestep (time constant >= 2 x "
               "timestep, positive damping ratio, no mixed-sign 'negative' solref), solimp is valid, "
               "and light objects do not rest deeper in their supports than half their own size.",
    "description": (
        "A solref time constant below twice the timestep is silently raised to 2 x timestep by MuJoCo (refsafe), so the "
        "contact the author tuned is not the one simulated; a damping ratio <= 0 bounces; a solref with one negative and one "
        "positive entry is almost always a time constant negated by accident (MuJoCo reads negative entries as direct "
        "stiffness and damping, so the positive one becomes negative damping); a direct (negative) solref needs stiffness "
        "and damping both positive and a stiffness of at least `min_direct_stiffness`. solimp must keep d0 and dmax inside "
        "(0, 1), a positive width and a valid midpoint/power. The equilibrium depth of a resting contact does not depend on "
        "the mass: it grows with the square of the time constant and with a softer impedance (about g * tc^2 * (1 - d) / d), "
        "and on a gram-scale part it can exceed the part's own features; so "
        "for each free object lighter than `light_mass_kg` the scene is settled for `settle_s` from keyframe `key` (robot "
        "joints held) and each of the object's contacts is compared with the smallest half-size of the object's geom in that "
        "contact: a part resting more than `max_depth_frac` of its size deep needs a shorter time constant or a direct (negative) solref. Checked "
        "elements: collidable geoms, explicit pairs, limited joints (solreflimit), equalities and tendon limits."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF file, XML string, or compiled model."},
        {"name": "light_mass_kg", "type": "float", "description": "Free objects below this mass get the resting-depth test. Default 0.02."},
        {"name": "max_depth_frac", "type": "float", "description": "Largest resting depth as a fraction of the part's smallest half-size. Default 0.5."},
        {"name": "settle_s", "type": "float", "description": "Settling time for the resting-depth test (0 skips it). Default 0.4."},
        {"name": "key", "type": "int | str", "description": "Keyframe the settle starts from. Default 0."},
        {"name": "min_direct_stiffness", "type": "float", "description": "Smallest direct (negative solref) stiffness, 1/s^2 (rest depth g/k). Default 100."},
        {"name": "robot_prefix", "type": "str", "description": "Name prefix of robot bodies. Default 'robot_'."},
    ],
    "outputs": [
        {"name": "report", "type": "dict",
         "description": "ok; issues; timestep; refsafe; bad: [{element, kind, solref, solimp, problem}]; light_objects: [{object, mass_kg, "
                        "depth_m, min_half_size_m, frac}]; worst."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["compile", "solref", "solimp", "timestep", "gram-scale", "contact"],
    "status": "tested",
    "entrypoint": "solver_parameter_sanity_check",
    "cli": "python -m tool_library.verifiers.solver_parameter_sanity_check scene.xml",
}

import json
import sys

import mujoco
import numpy as np

from tool_library.common import compile_checks as C


def _solref_problem(sr, dt, refsafe):
    a, b = float(sr[0]), float(sr[1])
    if a == 0 and b == 0:
        return None
    if (a < 0) != (b < 0) and not (a < 0 and b == 0):
        return (f"mixed-sign solref ({a:g}, {b:g}): MuJoCo reads negative entries as direct stiffness/damping, so this is a time "
                "constant or damping ratio negated by accident; use (timeconst, dampratio) both positive, or (-stiffness, -damping) "
                "both negative")
    if a > 0:
        if b <= 0:
            return f"damping ratio {b:g} <= 0: the constraint is undamped and bounces; use a damping ratio near 1"
        if refsafe and a < 2 * dt - 1e-12:
            return (f"time constant {a:g} s is below 2 x timestep ({2 * dt:g} s): MuJoCo silently uses {2 * dt:g} s, so the authored "
                    "stiffness is not what runs; raise the time constant or lower the timestep")
        return None
    if b == 0:
        return f"direct solref ({a:g}, 0) has no damping: it rings; give it a negative damping too"
    return None


def _solimp_problem(si):
    d0, dmax, width = float(si[0]), float(si[1]), float(si[2])
    probs = []
    if not (0 < d0 < 1) or not (0 < dmax < 1):
        probs.append(f"solimp d0={d0:g}, dmax={dmax:g} outside (0, 1) (MuJoCo clamps them)")
    if width <= 0:
        probs.append(f"solimp width {width:g} <= 0")
    if len(si) >= 5 and (not (0 < float(si[3]) < 1) or float(si[4]) < 1):
        probs.append(f"solimp midpoint/power ({float(si[3]):g}, {float(si[4]):g}) invalid (midpoint in (0,1), power >= 1)")
    return "; ".join(probs) or None


def _min_half(model, g):
    t = int(model.geom_type[g])
    s = np.asarray(model.geom_size[g], float)
    if t == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return float(s[0])
    if t in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
        return float(min(s[0], s[1] + (s[0] if t == int(mujoco.mjtGeom.mjGEOM_CAPSULE) else 0)))
    if t in (int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)):
        return float(s.min())
    if t == C.MESH:
        return float(np.min(model.geom_aabb[g][3:]))
    return None


def solver_parameter_sanity_check(scene, light_mass_kg=0.02, max_depth_frac=0.5, settle_s=0.4, key=0, min_direct_stiffness=100.0,
                                  robot_prefix=C.ROBOT_PREFIX):
    C.require_scene(scene)
    model, err = C.compile_scene(scene)
    if model is None:
        return C.compile_failure("fix the error MuJoCo names", err, bad=[])
    dt = float(model.opt.timestep)
    refsafe = not (int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_REFSAFE))
    bad, issues = [], []

    def judge(element, kind, sr, si):
        p1 = _solref_problem(sr, dt, refsafe)
        if p1 is None and float(sr[0]) < 0 and -float(sr[0]) < min_direct_stiffness:
            p1 = (f"direct stiffness {-float(sr[0]):g} 1/s^2 is too soft: a resting contact sinks about g/k = {9.81 / -float(sr[0]) * 1000:.0f} mm; "
                  f"use a stiffness of at least {min_direct_stiffness:g} (e.g. -20000 for a snap fit)")
        p2 = _solimp_problem(si) if si is not None else None
        prob = "; ".join(x for x in (p1, p2) if x)
        if prob:
            bad.append({"element": element, "kind": kind, "solref": [float(x) for x in sr],
                        "solimp": None if si is None else [float(x) for x in si], "problem": prob})

    for g in range(model.ngeom):
        if C.collidable(model, g):
            judge(C.geom_name(model, g), "geom", model.geom_solref[g], model.geom_solimp[g])
    for p in range(model.npair):
        nm = C.name_of(model, mujoco.mjtObj.mjOBJ_PAIR, p, "pair")
        judge(nm, "pair", model.pair_solref[p], model.pair_solimp[p])
    for j in range(model.njnt):
        if bool(model.jnt_limited[j]):
            judge(C.joint_name(model, j), "joint limit", model.jnt_solref[j], model.jnt_solimp[j])
    for e in range(model.neq):
        judge(C.eq_name(model, e), "equality", model.eq_solref[e], model.eq_solimp[e])
    for t in range(model.ntendon):
        if bool(model.tendon_limited[t]):
            judge(C.name_of(model, mujoco.mjtObj.mjOBJ_TENDON, t, "tendon"), "tendon limit", model.tendon_solref_lim[t], model.tendon_solimp_lim[t])
    # group repeated problems (a hand has dozens of identical geoms)
    by_prob = {}
    for b in bad:
        by_prob.setdefault((b["kind"], b["problem"]), []).append(b["element"])
    for (kind, prob), els in by_prob.items():
        issues.append(f"{len(els)} {kind}(s) ({els[:4]}{'...' if len(els) > 4 else ''}): {prob}")
    light = []
    if settle_s and settle_s > 0:
        light = _resting_depths(model, key, settle_s, light_mass_kg, robot_prefix)
        diverged = [r["object"] for r in light if r["against"] == "diverged"]
        if diverged:
            issues.append(f"the settle for the resting-depth test diverged (non-finite state within {settle_s} s) with light "
                          f"object(s) {diverged}: their resting depth cannot be measured; fix the instability first "
                          "(simulation_stability_check names it)")
        for r in light:
            if r["frac"] is not None and r["frac"] > max_depth_frac:
                issues.append(f"object '{r['object']}' ({1000 * r['mass_kg']:.1f} g) rests {1000 * r['depth_m']:.2f} mm deep in "
                              f"'{r['against']}' after {settle_s} s, {100 * r['frac']:.0f}% of its smallest half-size "
                              f"{1000 * r['min_half_size_m']:.2f} mm: the contact (time constant {r['timeconst']:g} s) is too soft for a part "
                              "this small, and the resting depth does not shrink with the mass; use a shorter time constant, a "
                              "stiffer solimp or a direct solref such as \"-20000 -400\"")
    return C.jsonable({"ok": not issues, "compiled": True, "issues": issues, "timestep": dt, "refsafe": refsafe, "bad": bad[:60],
                       "n_bad": len(bad), "light_objects": light, "worst": bad[0] if bad else None})


def _resting_depths(model, key, seconds, light_mass_kg, robot_prefix):
    data = C.key_data(model, key)
    qa, da = C.robot_kinematic_dofs(model, robot_prefix)
    q_hold = data.qpos[qa].copy() if len(qa) else None
    for s in C.servo_actuators(model):            # servos hold their joints where the keyframe put them
        data.ctrl[s["act"]] = data.qpos[s["qadr"]]
    light_roots = []
    for b in range(1, model.nbody):
        if int(model.body_parentid[b]) == 0 and int(model.body_jntnum[b]) > 0 and \
                int(model.jnt_type[int(model.body_jntadr[b])]) == C.FREE and not C.is_robot(C.body_name(model, b), robot_prefix) \
                and float(model.body_subtreemass[b]) < light_mass_kg:
            light_roots.append(b)
    if not light_roots:
        return []
    with C.capture_mj_warnings():
        for _ in range(int(round(seconds / model.opt.timestep))):
            mujoco.mj_step(model, data)
            if q_hold is not None:
                data.qpos[qa] = q_hold
                data.qvel[da] = 0.0
            if not np.isfinite(data.qpos).all():
                return [{"object": C.body_name(model, b), "mass_kg": float(model.body_subtreemass[b]), "depth_m": None,
                         "min_half_size_m": None, "frac": None, "against": "diverged", "timeconst": None} for b in light_roots]
    out = []
    for b in light_roots:
        sub = C.subtree_ids(model, b)
        depth, against, tc, frac, mh = 0.0, None, None, None, None
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            mine = g1 if int(model.geom_bodyid[g1]) in sub else (g2 if int(model.geom_bodyid[g2]) in sub else None)
            if mine is None:
                continue
            other = g2 if mine == g1 else g1
            if int(model.geom_bodyid[other]) in sub:
                continue
            d = -float(c.dist) + float(c.includemargin)
            h = _min_half(model, mine)
            f = d / h if h else None
            # judged per contact: the depth against the size of the part of the object that is in that contact
            if f is not None and (frac is None or f > frac):
                frac, depth, mh = f, d, h
                against, tc = C.body_name(model, int(model.geom_bodyid[other])), float(c.solref[0])
        out.append({"object": C.body_name(model, b), "mass_kg": float(model.body_subtreemass[b]), "depth_m": depth,
                    "min_half_size_m": mh, "frac": frac, "against": against, "timeconst": tc})
    return out


def main(argv=None):
    rep = solver_parameter_sanity_check((sys.argv[1:] if argv is None else argv)[0])
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
