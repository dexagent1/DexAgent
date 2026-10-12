"""Keyframe dimension check: does every keyframe fit the model it belongs to, value by value?"""
from __future__ import annotations

TOOL = {
    "name": "keyframe_dimension_check",
    "title": "Keyframe dimension check",
    "type": "verifier",
    "category": "Scene Compile Checks",
    "summary": "Every keyframe gives exactly nq/nv/na/nu/nmocap values (MuJoCo silently pads a short qpos from qpos0), its "
               "quaternions are unit, its joints lie inside their ranges, its ctrl inside ctrlrange and on the servoed joints' "
               "positions, and the robot pose matches the setup's home keyframe when a reference is given.",
    "description": (
        "Keyframes go wrong when a scene grows: attaching an object adds its free joint's 7 qpos in the middle of the "
        "robot's, and a keyframe extended at the wrong index puts robot joint values into the object's slots and the "
        "object's into the robot's. MuJoCo refuses a keyframe that is too long but pads one that is too short from qpos0 "
        "without a word. The check reads each keyframe's authored lengths through MjSpec (so the silent padding is seen) "
        "and compares them with nq, nv, na, nu and nmocap, and fails a robot scene keyframe that writes no qpos at all (a "
        "<size nkey> keyframe holds qpos0); then it fails non-finite entries, free and ball joint quaternions whose authored "
        "norm is off 1 by more than `quat_tol` (MuJoCo normalizes them when compiling, hiding the slip), limited joints outside their range by more "
        "than `range_tol`, ctrl outside a limited ctrlrange, and position servos whose ctrl differs from their joint's "
        "keyframe position by more than `ctrl_tol` (the servo yanks the joint at t = 0). With `reference` (the setup's "
        "scene.xml), every joint the two share is compared with the reference keyframe (`reference_key`, default 'home') "
        "within `ref_tol`: a split at the wrong index shows as a robot that is no longer at home. A bench scene (one with "
        "robot bodies) without any keyframe fails when `require_keyframe` is set: its initial state is qpos0, hands at the "
        "world origin and arms at zero."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF file, XML string, or compiled model (a model cannot show authored lengths)."},
        {"name": "reference", "type": "str | None", "description": "Scene whose keyframe the shared joints must match (e.g. setup/scene.xml)."},
        {"name": "reference_key", "type": "str | int", "description": "Keyframe of the reference. Default 'home' (index 0 when absent)."},
        {"name": "require_keyframe", "type": "bool", "description": "A scene with robot bodies must have a keyframe. Default True."},
        {"name": "quat_tol", "type": "float", "description": "Allowed |norm - 1| of keyframe quaternions. Default 1e-3."},
        {"name": "range_tol", "type": "float", "description": "Allowed excursion past a joint range (rad or m). Default 0.01."},
        {"name": "ctrl_tol", "type": "float", "description": "Allowed |ctrl - joint position| of a position servo (rad or m). Default 0.02."},
        {"name": "ref_tol", "type": "float", "description": "Allowed difference to the reference keyframe per shared joint. Default 1e-4."},
        {"name": "robot_prefix", "type": "str", "description": "Name prefix of robot bodies. Default 'robot_'."},
    ],
    "outputs": [
        {"name": "report", "type": "dict",
         "description": "ok; issues; sizes {nq, nv, na, nu, nmocap}; keys: [{key, lengths, problems}]; reference: {key, n_shared, "
                        "mismatches: [{joint, scene, reference}]}; worst."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["compile", "keyframe", "qpos", "attach", "home"],
    "status": "tested",
    "entrypoint": "keyframe_dimension_check",
    "cli": "python -m tool_library.verifiers.keyframe_dimension_check scene.xml [reference.xml]",
}

import json
import sys

import mujoco
import numpy as np

from tool_library.common import compile_checks as C


def _authored_lengths(scene):
    """[(name, {field: authored length})] read through MjSpec, or None when the scene is a compiled model."""
    if isinstance(scene, mujoco.MjModel):
        return None
    p = C.scene_path(scene)
    try:
        with C.capture_mj_warnings():
            spec = mujoco.MjSpec.from_file(str(p)) if p is not None else mujoco.MjSpec.from_string(str(scene))
    except Exception:
        return None
    out = []
    for k in spec.keys:
        out.append((k.name or "", {f: int(len(getattr(k, f))) for f in ("qpos", "qvel", "act", "ctrl", "mpos", "mquat")},
                    np.asarray(k.qpos, float).copy()))
    return out


def _joint_values(model, qpos):
    """{joint name: value} for named hinge/slide joints."""
    out = {}
    for j in range(model.njnt):
        if int(model.jnt_type[j]) in (C.HINGE, C.SLIDE):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if nm:
                out[nm] = float(qpos[int(model.jnt_qposadr[j])])
    return out


def _key_index(model, key):
    if isinstance(key, str):
        k = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key)
        return k if k >= 0 else (0 if model.nkey else -1)
    return int(key) if 0 <= int(key) < model.nkey else -1


def keyframe_dimension_check(scene, reference=None, reference_key="home", require_keyframe=True, quat_tol=1e-3, range_tol=0.01,
                             ctrl_tol=0.02, ref_tol=1e-4, robot_prefix=C.ROBOT_PREFIX):
    C.require_scene(scene)
    model, err = C.compile_scene(scene)
    if model is None:
        hint = ("a keyframe longer than the model (MuJoCo: 'invalid ... size'): rebuild it from the scene's joint layout"
                if "keyframe" in str(err) else "fix the error MuJoCo names")
        return C.compile_failure(hint, err, keys=[])
    sizes = {"nq": int(model.nq), "nv": int(model.nv), "na": int(model.na), "nu": int(model.nu), "nmocap": int(model.nmocap)}
    expect = {"qpos": model.nq, "qvel": model.nv, "act": model.na, "ctrl": model.nu, "mpos": 3 * model.nmocap, "mquat": 4 * model.nmocap}
    issues, keys = [], []
    has_robot = bool(C.robot_body_ids(model, robot_prefix))
    if model.nkey == 0 and require_keyframe and has_robot:
        issues.append("the scene has robot bodies but no keyframe: its initial state is qpos0 (each hand at the world origin, the arms at "
                      "zero, objects where their bodies were written) and the first physics step snaps the robot; add the setup's "
                      "'home' keyframe extended with every object's initial joint values")
    authored = _authored_lengths(scene)
    servos = C.servo_actuators(model)
    for k in range(model.nkey):
        kname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, k) or f"key{k}"
        probs = []
        if authored is not None and k < len(authored):
            lengths, raw_q = authored[k][1], authored[k][2]
        elif authored is not None:          # reserved by <size nkey>: nothing authored, it is qpos0
            lengths, raw_q = {f: 0 for f in expect}, None
        else:
            lengths, raw_q = None, None
        if lengths is not None and lengths["qpos"] == 0 and has_robot:
            probs.append("no qpos is written for it, so it holds qpos0 (each hand at the world origin, the arms at zero; a <size nkey> "
                         "reserves such keyframes): write the robot's home and the objects' initial values")
        if lengths:
            for f, n in lengths.items():
                if n and n != expect[f]:
                    probs.append(f"{f} has {n} values, the model needs {expect[f]}" +
                                 (" (MuJoCo pads the missing ones from qpos0 without a warning)" if f == "qpos" and n < expect[f] else ""))
        q = np.asarray(model.key_qpos[k], float)
        # MuJoCo normalizes keyframe quaternions when it compiles: judge them on the authored values
        q_auth = raw_q if raw_q is not None and len(raw_q) == model.nq else q
        if not np.isfinite(q).all():
            probs.append(f"non-finite qpos at {np.where(~np.isfinite(q))[0][:5].tolist()}")
        bad_quat, out_range = [], []
        for j in range(model.njnt):
            t, a = int(model.jnt_type[j]), int(model.jnt_qposadr[j])
            if t in (C.FREE, C.BALL):
                qq = q_auth[a + 3:a + 7] if t == C.FREE else q_auth[a:a + 4]
                nrm = float(np.linalg.norm(qq))
                if abs(nrm - 1.0) > quat_tol:
                    bad_quat.append((C.joint_name(model, j), round(nrm, 4), [round(float(x), 4) for x in qq]))
                if t == C.FREE and float(np.linalg.norm(q[a:a + 3])) > 10.0:
                    out_range.append((C.joint_name(model, j), "position", [round(float(x), 3) for x in q[a:a + 3]]))
            elif t in (C.HINGE, C.SLIDE) and bool(model.jnt_limited[j]):
                lo, hi = [float(x) for x in model.jnt_range[j]]
                if q[a] < lo - range_tol or q[a] > hi + range_tol:
                    out_range.append((C.joint_name(model, j), round(float(q[a]), 4), [lo, hi]))
        if bad_quat:
            probs.append(f"{len(bad_quat)} quaternions are not unit (first {bad_quat[0]}): values from another joint landed in a free joint's "
                         "slots, the signature of a keyframe extended at the wrong index")
        if out_range:
            probs.append(f"{len(out_range)} joints outside their range (first {out_range[0]})")
        if model.nu:
            c = np.asarray(model.key_ctrl[k], float)
            bad_c = [(C.actuator_name(model, a), round(float(c[a]), 4), [float(x) for x in model.actuator_ctrlrange[a]])
                     for a in range(model.nu) if bool(model.actuator_ctrllimited[a])
                     and (c[a] < model.actuator_ctrlrange[a][0] - 1e-6 or c[a] > model.actuator_ctrlrange[a][1] + 1e-6)]
            if bad_c:
                probs.append(f"{len(bad_c)} ctrl values outside ctrlrange (first {bad_c[0]})")
            off = [(C.actuator_name(model, s["act"]), round(float(c[s["act"]]), 4), round(float(q[s["qadr"]]), 4))
                   for s in servos if abs(float(c[s["act"]]) - float(q[s["qadr"]])) > ctrl_tol]
            if off:
                probs.append(f"{len(off)} position servos are commanded away from their joint's keyframe position (first {off[0]}: ctrl, "
                             "qpos): they yank the joint at t = 0; set the keyframe ctrl to the joint positions")
        keys.append({"key": kname, "lengths": lengths, "problems": probs})
        for p in probs:
            issues.append(f"keyframe '{kname}': {p}")
    ref_rep = None
    if reference is not None:
        rmodel, rerr = C.compile_scene(reference)
        if rmodel is None:
            raise ValueError(f"reference does not compile: {rerr}")
        rk = _key_index(rmodel, reference_key)
        sk = _key_index(model, reference_key)
        if rk < 0:
            raise ValueError(f"reference has no keyframe '{reference_key}'")
        if sk < 0:
            ref_rep = {"key": reference_key, "n_shared": 0, "mismatches": []}
        else:
            rv = _joint_values(rmodel, rmodel.key_qpos[rk])
            sv = _joint_values(model, model.key_qpos[sk])
            shared = sorted(set(rv) & set(sv))
            mism = [{"joint": j, "scene": round(sv[j], 6), "reference": round(rv[j], 6)} for j in shared if abs(sv[j] - rv[j]) > ref_tol]
            ref_rep = {"key": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, sk) or f"key{sk}", "n_shared": len(shared), "mismatches": mism}
            if mism:
                issues.append(f"keyframe '{ref_rep['key']}': {len(mism)} of {len(shared)} joints shared with the reference differ from its "
                              f"'{reference_key}' (first {mism[0]['joint']}: {mism[0]['scene']} vs {mism[0]['reference']}): the robot is not at "
                              "the setup's home, as when a keyframe was split at the wrong index; rebuild it from the reference values by joint name")
    worst = next((k for k in keys if k["problems"]), None)
    return C.jsonable({"ok": not issues, "compiled": True, "issues": issues, "sizes": sizes, "nkey": int(model.nkey), "keys": keys,
                       "reference": ref_rep, "worst": worst})


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    rep = keyframe_dimension_check(argv[0], reference=argv[1] if len(argv) > 1 else None)
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
