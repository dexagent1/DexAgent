"""Joint armature/damping check: do the servoed joints carry the reflected inertia and damping that keep their servos stable?"""
from __future__ import annotations

TOOL = {
    "name": "joint_armature_damping_check",
    "title": "Joint armature and damping check",
    "type": "verifier",
    "category": "Scene Compile Checks",
    "summary": "Every position-servoed joint is stable at the scene's timestep (servo frequency x timestep below a limit given its "
               "inertia including armature, no anti-damping), light free bodies carry no explicit damping that outruns their inertia, "
               "and joints shared with a reference scene (the setup) keep its armature, damping, friction loss and stiffness.",
    "description": (
        "A position servo of stiffness kp on a joint with inertia I oscillates at w = sqrt(kp / I), and MuJoCo integrates the "
        "position term explicitly (implicitfast and implicit treat only velocity terms implicitly), so w * timestep must stay "
        "well below 2. The Sharpa hand ships armature 1.0 on every joint for this reason: a finger link alone is ~1e-6 "
        "kg m^2, and without armature a kp 300 servo has w * dt ~ 30 at 2 ms and the fingers blow up. The check compiles the "
        "scene, reads each servo's effective inertia from the mass matrix diagonal at keyframe `key` (link inertia plus "
        "armature), and fails w * dt > `max_omega_dt`, a negative damping ratio (compiled bias velocity term positive, or "
        "negative joint damping), and with the explicit Euler integrator a velocity gain kv with kv * dt / I > 1. A free joint "
        "with damping d on a body whose smallest rotational inertia I gives d * dt / I > 1 is reported too (0.0002 N m s on a "
        "1e-8 kg m^2 spin axis NaN'd a solver). With `reference` (the setup's scene.xml) or `expected` ({joint-name glob: "
        "{armature, damping, frictionloss, stiffness}}), each matching joint must carry those values within `rel_tol`."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF file, XML string, or compiled model."},
        {"name": "reference", "type": "str | None", "description": "Scene whose joints' armature/damping/frictionloss/stiffness shared joints must keep (e.g. setup/scene.xml)."},
        {"name": "expected", "type": "dict | None", "description": "{joint name glob: {armature, damping, frictionloss, stiffness}} the matching joints must carry."},
        {"name": "max_omega_dt", "type": "float", "description": "Largest servo frequency x timestep. Default 0.5 (explicit limit 2)."},
        {"name": "key", "type": "int | str", "description": "Keyframe the inertia is read at. Default 0."},
        {"name": "rel_tol", "type": "float", "description": "Relative tolerance of the reference comparison. Default 1e-6."},
    ],
    "outputs": [
        {"name": "report", "type": "dict",
         "description": "ok; issues; timestep; integrator; servos: [{actuator, joint, kp, kv, armature, damping, inertia, omega_dt, zeta, problem}] "
                        "(problems first); free_joint_damping: [...]; reference_mismatches: [{joint, field, scene, reference}]; worst."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["compile", "armature", "damping", "servo", "stability"],
    "status": "tested",
    "entrypoint": "joint_armature_damping_check",
    "cli": "python -m tool_library.verifiers.joint_armature_damping_check scene.xml [reference.xml]",
}

import fnmatch
import json
import sys

import mujoco
import numpy as np

from tool_library.common import compile_checks as C

_FIELDS = ("armature", "damping", "frictionloss", "stiffness")


def _joint_params(model, j):
    d = int(model.jnt_dofadr[j])
    return {"armature": float(model.dof_armature[d]), "damping": float(model.dof_damping[d]),
            "frictionloss": float(model.dof_frictionloss[d]), "stiffness": float(model.jnt_stiffness[j])}


def joint_armature_damping_check(scene, reference=None, expected=None, max_omega_dt=0.5, key=0, rel_tol=1e-6):
    C.require_scene(scene)
    if expected is not None and not isinstance(expected, dict):
        raise ValueError("expected must be a dict {joint name glob: {armature, damping, frictionloss, stiffness}}")
    model, err = C.compile_scene(scene)
    if model is None:
        return C.compile_failure("fix the error MuJoCo names", err, servos=[])
    data = C.key_data(model, key)
    dt = float(model.opt.timestep)
    euler = int(model.opt.integrator) == int(mujoco.mjtIntegrator.mjINT_EULER)
    rows, issues = [], []
    # the joint-space inertia's diagonal, read through mj_fullM: the sparse storage changed name and layout across
    # MuJoCo versions (data.qM with dof_Madr up to 3.1x, data.M later)
    full = np.zeros((model.nv, model.nv))
    try:
        mujoco.mj_fullM(model, data, full)           # MuJoCo >= 3.3: (model, data, dst)
    except TypeError:
        mujoco.mj_fullM(model, full, data.qM)        # older bindings: (model, dst, qM)
    for s in C.servo_actuators(model):
        d = s["dof"]
        inertia = float(full[int(d), int(d)])
        damp = float(model.dof_damping[d])
        kp, kv = s["kp"], s["kv"]
        w = float(np.sqrt(kp / inertia)) if inertia > 0 else float("inf")
        zeta = (kv + damp) / (2.0 * np.sqrt(kp * inertia)) if inertia > 0 else float("nan")
        probs, kinds = [], []
        if w * dt > max_omega_dt:
            kinds.append("explicitly unstable servo")
            probs.append(f"servo frequency x timestep = {w * dt:.2f} (> {max_omega_dt}): kp {kp:g} on an inertia of {inertia:.3g} kg m^2 "
                         f"(armature {float(model.dof_armature[d]):g}) is explicitly unstable at dt {dt:g}; give the joint armature "
                         f"(>= {kp * (dt / max_omega_dt) ** 2:.3g}), lower kp or the timestep")
        if kv < 0 or damp < 0:
            kinds.append("negative damping")
            probs.append(f"negative damping (servo kv {kv:g}, joint damping {damp:g}): the servo pumps energy into the joint; a position "
                         "servo's bias velocity term must be -kv (or a positive dampratio before compiling)")
        elif euler and inertia > 0 and kv * dt / inertia > 1.0:
            kinds.append("explicit velocity gain")
            probs.append(f"with the Euler integrator the servo's velocity gain is explicit: kv*dt/I = {kv * dt / inertia:.2f} > 1 is "
                         "unstable; use integrator='implicitfast' or more armature")
        rows.append({"actuator": C.actuator_name(model, s["act"]), "joint": C.joint_name(model, s["joint"]), "kp": kp, "kv": kv,
                     "armature": float(model.dof_armature[d]), "damping": damp, "inertia": inertia, "omega_dt": w * dt,
                     "zeta": zeta, "problem": "; ".join(probs) or None, "kinds": kinds})
    grouped = {}
    for r in rows:
        for k in r["kinds"]:
            grouped.setdefault(k, []).append(r)
    for k, rs in grouped.items():
        r0 = max(rs, key=lambda r: r["omega_dt"])
        issues.append(f"{k}: {len(rs)} servoed joint(s) ({[r['joint'] for r in rs][:4]}{'...' if len(rs) > 4 else ''}), worst "
                      f"'{r0['joint']}': {r0['problem']}")
    # damping on free joints of light bodies
    free_rows = []
    for j in range(model.njnt):
        if int(model.jnt_type[j]) != C.FREE:
            continue
        d0 = int(model.jnt_dofadr[j])
        dr = float(np.max(model.dof_damping[d0 + 3:d0 + 6]))
        if dr <= 0:
            continue
        b = int(model.jnt_bodyid[j])
        imin = float(np.min(model.body_inertia[b])) if np.min(model.body_inertia[b]) > 0 else 1e-12
        ratio = dr * dt / imin
        free_rows.append({"joint": C.joint_name(model, j), "body": C.body_name(model, b), "damping": dr, "min_inertia": imin, "ratio": ratio})
        if ratio > 1.0:
            issues.append(f"free joint '{C.joint_name(model, j)}' of '{C.body_name(model, b)}' has rotational damping {dr:g} on a smallest "
                          f"inertia of {imin:.3g} kg m^2 (damping*dt/I = {ratio:.1f} > 1): light bodies with free-joint damping have "
                          "NaN'd the solver; remove the damping (a plain freejoint is fine)")
    mism = []
    if reference is not None:
        rmodel, rerr = C.compile_scene(reference)
        if rmodel is None:
            raise ValueError(f"reference does not compile: {rerr}")
        for j in range(model.njnt):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            rj = mujoco.mj_name2id(rmodel, mujoco.mjtObj.mjOBJ_JOINT, nm) if nm else -1
            if rj < 0 or int(model.jnt_type[j]) != int(rmodel.jnt_type[rj]):
                continue
            a, b = _joint_params(model, j), _joint_params(rmodel, rj)
            for f in _FIELDS:
                if not np.isclose(a[f], b[f], rtol=rel_tol, atol=1e-12):
                    mism.append({"joint": nm, "field": f, "scene": a[f], "reference": b[f]})
    if expected:
        for j in range(model.njnt):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            for pat, want in expected.items():
                if not fnmatch.fnmatch(nm, pat):
                    continue
                unknown = set(want) - set(_FIELDS)
                if unknown:
                    raise ValueError(f"expected[{pat!r}] has unknown fields {sorted(unknown)}; valid: {list(_FIELDS)}")
                a = _joint_params(model, j)
                for f, v in want.items():
                    if not np.isclose(a[f], float(v), rtol=rel_tol, atol=1e-12):
                        mism.append({"joint": nm, "field": f, "scene": a[f], "reference": float(v)})
    if mism:
        by_f = {}
        for m_ in mism:
            by_f.setdefault(m_["field"], []).append(m_)
        for f, ms in by_f.items():
            less = f == "armature" and any(m_["scene"] < m_["reference"] for m_ in ms)
            issues.append(f"{len(ms)} joint(s) carry a different {f} than specified (first '{ms[0]['joint']}': {ms[0]['scene']:g} vs "
                          f"{ms[0]['reference']:g}){' - less armature than specified destabilizes servos' if less else ''}; copy "
                          "the setup's joint defaults")
    rows.sort(key=lambda r: (r["problem"] is None, -r["omega_dt"]))
    return C.jsonable({"ok": not issues, "compiled": True, "issues": issues, "timestep": dt,
                       "integrator": str(mujoco.mjtIntegrator(int(model.opt.integrator))).split(".")[-1],
                       "servos": rows[:80], "n_servos": len(rows), "free_joint_damping": free_rows, "reference_mismatches": mism[:60],
                       "worst": rows[0] if rows else None})


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    rep = joint_armature_damping_check(argv[0], reference=argv[1] if len(argv) > 1 else None)
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
