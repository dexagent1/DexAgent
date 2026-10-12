"""Equality constraint check: are the scene's welds and connects well formed, satisfied at the start, and in the state intended?"""
from __future__ import annotations

TOOL = {
    "name": "equality_constraint_check",
    "title": "Equality constraint check",
    "type": "verifier",
    "category": "Scene Compile Checks",
    "summary": "Every weld and connect has a unit relative quaternion and a meaningful torquescale, every active equality is satisfied "
               "at the initial state, and each equality's active flag matches what the scene or its episode intends (a fresh "
               "MjData re-activates every weld whose eq_active0 is set).",
    "description": (
        "Equalities hold the hands on the arms and the mechanisms together, and they fail silently. A weld whose relative "
        "quaternion is zero (eq_data written as [0]*11 at run time) has no defined orientation; a weld written through "
        "MjSpec with the default data gets torquescale 0 and holds position only, so a part welded to the world spins about "
        "its anchor; only the robot's arm-to-hand welds are position-only by design (both bodies carry the robot prefix). "
        "The check compiles the scene and, at keyframe `key`, measures every weld / connect residual itself (position "
        "error of the anchor, and the rotation error of a weld whose torquescale is not 0) plus joint-equality residuals; an "
        "active equality violated by more than `pos_tol_m` / `rot_tol_deg` at the start snaps its bodies on the first step. "
        "Intended state: `expect_active` ({equality name or 'body1|body2': bool}) is compared with eq_active0; and with a "
        "`trajectory` (npz with qpos, optional eq_active / eq_active_start / eq_names), every non-robot weld/connect that the "
        "recorded qpos violates by more than `release_tol_m` / `release_tol_deg` on some frame must have been released in "
        "the recording: if the scene starts it active and the trajectory records no eq_active, replaying the episode from a "
        "fresh MjData re-engages it (a removed cap snaps back onto its bottle); if eq_active is recorded, its frames are "
        "used. A recorded eq_active_start that differs from eq_active0 is reported the same way."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF file, XML string, or compiled model."},
        {"name": "key", "type": "int | str", "description": "Keyframe of the initial state. Default 0 (qpos0 without keyframes)."},
        {"name": "expect_active", "type": "dict | None", "description": "{equality name or 'body1|body2': True/False} the scene must start with."},
        {"name": "trajectory", "type": "str | None", "description": "Episode npz (qpos [T, nq], optional eq_active [T, neq], eq_active_start, eq_names)."},
        {"name": "frames", "type": "list[int] | None", "description": "[first, last] frames of the trajectory to scan. Default all."},
        {"name": "stride", "type": "int", "description": "Scan every stride-th frame. Default 5."},
        {"name": "pos_tol_m", "type": "float", "description": "Residual an active equality may have at the start, metres. Default 0.002."},
        {"name": "rot_tol_deg", "type": "float", "description": "Rotation residual an active weld may have at the start. Default 2."},
        {"name": "release_tol_m", "type": "float", "description": "Residual on a trajectory frame that means the equality was released. Default 0.01."},
        {"name": "release_tol_deg", "type": "float", "description": "Rotation residual that means a weld was released. Default 10."},
        {"name": "robot_prefix", "type": "str", "description": "Name prefix of robot bodies. Default 'robot_'."},
    ],
    "outputs": [
        {"name": "report", "type": "dict",
         "description": "ok; issues; equalities: [{name, type, body1, body2, active0, torquescale, robot_pair, pos_err_m, rot_err_deg, problem}]; "
                        "trajectory: {n_frames, recorded_eq_active, released: [{name, first_frame, max_pos_err_m}]}; worst."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["compile", "equality", "weld", "connect", "eq_active", "torquescale"],
    "status": "tested",
    "entrypoint": "equality_constraint_check",
    "cli": "python -m tool_library.verifiers.equality_constraint_check scene.xml [trajectory.npz]",
}

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

from tool_library.common import compile_checks as C

_JOINT = int(mujoco.mjtEq.mjEQ_JOINT)
_SITE = int(mujoco.mjtObj.mjOBJ_SITE)


def _frame(model, data, objtype, i):
    if objtype == _SITE:
        return data.site_xpos[i].copy(), data.site_xmat[i].reshape(3, 3).copy()
    return data.xpos[i].copy(), data.xmat[i].reshape(3, 3).copy()


def _quat_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float) / max(np.linalg.norm(q), 1e-12))
    return m.reshape(3, 3)


def _residual(model, data, e):
    """(position error m, rotation error deg or None) of equality e at data (kinematics done)."""
    t = int(model.eq_type[e])
    d = np.asarray(model.eq_data[e], float)
    ot = int(model.eq_objtype[e])
    if t == C.CONNECT:
        p1, R1 = _frame(model, data, ot, int(model.eq_obj1id[e]))
        p2, R2 = _frame(model, data, ot, int(model.eq_obj2id[e]))
        if ot == _SITE:
            return float(np.linalg.norm(p1 - p2)), None
        return float(np.linalg.norm((p1 + R1 @ d[0:3]) - (p2 + R2 @ d[3:6]))), None
    if t == C.WELD:
        p1, R1 = _frame(model, data, ot, int(model.eq_obj1id[e]))
        p2, R2 = _frame(model, data, ot, int(model.eq_obj2id[e]))
        if ot == _SITE:
            ang = float(np.degrees(np.arccos(np.clip((np.trace(R1.T @ R2) - 1) / 2, -1, 1))))
            return float(np.linalg.norm(p1 - p2)), ang
        anchor, relpos, relq = d[0:3], d[3:6], d[6:10]
        Rrel = _quat_mat(relq)
        err_p = (p2 + R2 @ anchor) - (p1 + R1 @ (relpos + Rrel @ anchor))
        Rerr = (R1 @ Rrel).T @ R2
        ang = float(np.degrees(np.arccos(np.clip((np.trace(Rerr) - 1) / 2, -1, 1))))
        return float(np.linalg.norm(err_p)), ang
    if t == _JOINT:
        j1, j2 = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
        q1 = float(data.qpos[int(model.jnt_qposadr[j1])]) - float(model.qpos0[int(model.jnt_qposadr[j1])])
        if j2 >= 0:
            x = float(data.qpos[int(model.jnt_qposadr[j2])]) - float(model.qpos0[int(model.jnt_qposadr[j2])])
            target = d[0] + d[1] * x + d[2] * x ** 2 + d[3] * x ** 3 + d[4] * x ** 4
        else:
            target = d[0]
        return abs(q1 - target), None
    return 0.0, None


def _endpoints(model, e):
    t, ot = int(model.eq_type[e]), int(model.eq_objtype[e])
    if t in (C.WELD, C.CONNECT):
        if ot == _SITE:
            b1, b2 = int(model.site_bodyid[int(model.eq_obj1id[e])]), int(model.site_bodyid[int(model.eq_obj2id[e])])
        else:
            b1, b2 = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
        return C.body_name(model, b1) if b1 > 0 else "world", C.body_name(model, b2) if b2 > 0 else "world", b1, b2
    if t == _JOINT:
        j1, j2 = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
        return C.joint_name(model, j1), (C.joint_name(model, j2) if j2 >= 0 else ""), -1, -1
    return str(int(model.eq_obj1id[e])), str(int(model.eq_obj2id[e])), -1, -1


def equality_constraint_check(scene, key=0, expect_active=None, trajectory=None, frames=None, stride=5, pos_tol_m=0.002,
                              rot_tol_deg=2.0, release_tol_m=0.01, release_tol_deg=10.0, robot_prefix=C.ROBOT_PREFIX):
    C.require_scene(scene)
    if expect_active is not None and not isinstance(expect_active, dict):
        raise ValueError("expect_active must be a dict {equality name or 'body1|body2': bool}")
    if int(stride) < 1:
        raise ValueError("stride must be >= 1")
    model, err = C.compile_scene(scene)
    if model is None:
        return C.compile_failure("an equality MuJoCo rejects (unknown body/site/joint, or a weld between a body and itself): fix the "
                                 "equality named in the message", err, equalities=[])
    data = C.key_data(model, key)
    data.eq_active[:] = 1                    # residuals are geometric; activity does not change kinematics
    mujoco.mj_kinematics(model, data)
    tname = {int(mujoco.mjtEq.mjEQ_WELD): "weld", int(mujoco.mjtEq.mjEQ_CONNECT): "connect", _JOINT: "joint",
             int(mujoco.mjtEq.mjEQ_TENDON): "tendon"}
    rows, issues = [], []
    robot_ids = C.robot_body_ids(model, robot_prefix)
    for e in range(model.neq):
        t = int(model.eq_type[e])
        n1, n2, b1, b2 = _endpoints(model, e)
        robot_pair = b1 in robot_ids and b2 in robot_ids
        ts = float(model.eq_data[e][10]) if t == C.WELD else None
        pe, re_ = _residual(model, data, e)
        if t == C.WELD and ts == 0.0:
            re_ = None                          # a position-only weld does not hold orientation
        act = bool(model.eq_active0[e])
        probs = []
        if t == C.WELD and int(model.eq_objtype[e]) != _SITE:
            qn = float(np.linalg.norm(model.eq_data[e][6:10]))
            if abs(qn - 1.0) > 1e-3:
                probs.append(f"relative quaternion has norm {qn:.3g}: the weld's orientation is undefined (eq_data written as zeros at run "
                             "time); set relpose to the bodies' relative pose or recompile so MuJoCo computes it")
            if ts == 0.0 and not robot_pair:
                probs.append("torquescale 0: the weld holds position only and the part spins freely about its anchor (MjSpec's default "
                             "weld data); a weld to the world or a mechanism weld needs torquescale 1 (data = [0]*10 + [1])")
        if act and pe > pos_tol_m:
            probs.append(f"active but violated by {1000 * pe:.1f} mm at the initial state: the first step snaps "
                         f"'{n2}' onto '{n1}'; place the bodies where the equality holds them, or start it inactive")
        if act and re_ is not None and re_ > rot_tol_deg:
            probs.append(f"active but its orientation is off by {re_:.1f} deg at the initial state: the first step twists the bodies into "
                         "place; fix relpose or the keyframe")
        nm = C.eq_name(model, e)
        if expect_active:
            want = None
            for k, v in expect_active.items():
                if k == nm or k in (f"{n1}|{n2}", f"{n2}|{n1}"):
                    want = bool(v)
            if want is not None and want != act:
                probs.append(f"starts {'active' if act else 'inactive'} but the scene is meant to start it {'active' if want else 'inactive'}"
                             f" (a fresh MjData takes eq_active0); set active=\"{str(want).lower()}\" on the equality")
        rows.append({"name": nm, "type": tname.get(t, str(t)), "body1": n1, "body2": n2, "active0": act, "torquescale": ts,
                     "robot_pair": robot_pair, "pos_err_m": pe, "rot_err_deg": re_, "problem": "; ".join(probs) or None})
        for p in probs:
            issues.append(f"equality '{nm}' ({tname.get(t, t)} {n1} - {n2}): {p}")
    if expect_active:
        known = {r["name"] for r in rows} | {f"{r['body1']}|{r['body2']}" for r in rows} | {f"{r['body2']}|{r['body1']}" for r in rows}
        unknown = [k for k in expect_active if k not in known]
        if unknown:
            issues.append(f"expect_active names equalities that are not in the scene: {unknown}; equalities: {[r['name'] for r in rows]}")
    traj_rep = None
    if trajectory is not None:
        traj_rep = _scan_trajectory(model, trajectory, frames, int(stride), rows, robot_ids, release_tol_m, release_tol_deg, issues)
    rows.sort(key=lambda r: r["problem"] is None)
    return C.jsonable({"ok": not issues, "compiled": True, "issues": issues, "equalities": rows, "n_equalities": int(model.neq),
                       "trajectory": traj_rep, "worst": rows[0] if rows and rows[0]["problem"] else None})


def _scan_trajectory(model, trajectory, frames, stride, rows, robot_ids, tol_m, tol_deg, issues):
    p = Path(trajectory)
    if not p.exists():
        raise ValueError(f"trajectory not found: {p}")
    z = np.load(str(p), allow_pickle=True)
    Q = np.asarray(z["qpos"] if "qpos" in z.files else z["Q"], float)
    if Q.ndim != 2 or Q.shape[1] != model.nq:
        issues.append(f"trajectory qpos has shape {list(Q.shape)}, the scene has nq={model.nq}: it was recorded in another scene")
        return {"n_frames": int(len(Q)), "recorded_eq_active": False, "released": []}
    lo, hi = (0, len(Q) - 1) if frames is None else (max(0, int(frames[0])), min(len(Q) - 1, int(frames[1])))
    ea = np.asarray(z["eq_active"]) if "eq_active" in z.files else None
    names = [str(x) for x in z["eq_names"]] if "eq_names" in z.files else [C.eq_name(model, e) for e in range(model.neq)]
    col = {n: i for i, n in enumerate(names)}
    start = np.asarray(z["eq_active_start"]).astype(bool) if "eq_active_start" in z.files else None
    data = mujoco.MjData(model)
    released = []
    cand = [e for e in range(model.neq) if int(model.eq_type[e]) in (C.WELD, C.CONNECT) and not rows[e]["robot_pair"]]
    first, worst = {}, {}
    for t in list(range(lo, hi + 1, stride)) + ([hi] if (hi - lo) % stride else []):
        data.qpos[:] = Q[t]
        mujoco.mj_kinematics(model, data)
        for e in cand:
            pe, re_ = _residual(model, data, e)
            if rows[e]["torquescale"] == 0.0:
                re_ = None
            viol = pe > tol_m or (re_ is not None and re_ > tol_deg)
            if viol:
                first.setdefault(e, t)
                worst[e] = max(worst.get(e, 0.0), pe)
    for e, t0 in first.items():
        nm = rows[e]["name"]
        rec = None
        if ea is not None and nm in col and ea.ndim == 2 and ea.shape[0] == len(Q):
            rec = bool(ea[t0][col[nm]])
        released.append({"name": nm, "first_frame": int(t0), "max_pos_err_m": float(worst[e]), "recorded_active_then": rec})
        if ea is None and bool(model.eq_active0[e]):
            issues.append(f"equality '{nm}' ({rows[e]['body1']} - {rows[e]['body2']}) is violated by up to {1000 * worst[e]:.0f} mm from frame "
                          f"{t0} on, so the episode released it, but the scene starts it active and the trajectory records no eq_active: "
                          "replaying from a fresh MjData re-engages it and snaps the part back; save eq_active / eq_names with the trajectory")
        elif rec is True:
            issues.append(f"equality '{nm}' is recorded active on frame {t0} but the recorded qpos violates it by {1000 * worst[e]:.0f} mm: "
                          "the recording and its equality states disagree")
    if start is not None:
        for e in range(model.neq):
            nm = C.eq_name(model, e)
            if nm in col and col[nm] < len(start) and bool(start[col[nm]]) != bool(model.eq_active0[e]):
                issues.append(f"equality '{nm}': the episode started it {'active' if start[col[nm]] else 'inactive'} but the scene's "
                              f"eq_active0 is {bool(model.eq_active0[e])}: a fresh MjData starts the replay in the wrong state")
    return {"n_frames": int(len(Q)), "frames": [lo, hi], "recorded_eq_active": ea is not None, "released": released}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    rep = equality_constraint_check(argv[0], trajectory=argv[1] if len(argv) > 1 else None)
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
