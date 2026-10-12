"""Weld gap check: does every welded pair of bodies actually sit together?

    from tool_library.verifiers.weld_gap_check import check_weld_gap
    check_weld_gap("scene.xml", "trajectory.npz")

A hand welded to an arm flange, a cap welded to a bottle: in a kinematic
trajectory the two sides of a weld are driven separately, and nothing stops
them drifting apart. Replayed with physics the constraint then yanks them
together and the episode diverges. This check evaluates the weld's own
residual, position and angle, on every frame.
"""
from __future__ import annotations

TOOL = {
    "name": "weld_gap_check",
    "title": "Weld gap check",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "Position and angle residual of every active weld or connect constraint on every frame.",
    "description": (
        "For each active weld (or connect) equality constraint in the scene, replays the "
        "trajectory kinematically and measures how far the two anchor points are apart and, "
        "for a weld with a rotational part, the angle between the two frames. Fails when the "
        "gap exceeds 5 mm or the angle 3 degrees anywhere, the tolerance below which a "
        "physics replay starts without a visible snap. Constraints can be selected by body "
        "name pattern; by default all are checked. Works on any scene that uses equality "
        "constraints to assemble a robot or an object."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF scene."},
        {"name": "trajectory", "type": "str | ndarray [T, nq]", "description": "trajectory.npz or qpos array."},
        {"name": "gap_tol_m", "type": "float, metres", "description": "Largest allowed anchor separation. Default 0.005."},
        {"name": "angle_tol_deg", "type": "float, degrees", "description": "Largest allowed frame misalignment for torque-carrying welds. Default 3."},
        {"name": "only", "type": "list[str] | None", "description": "fnmatch patterns on either body name; others are skipped."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; per constraint: bodies, max gap (mm), max angle (deg), worst frame and label, position-only flag."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["weld", "constraint", "assembly", "replay"],
    "used_in": ["Multi-Cup Grasping", "Giftbox Packing", "Battery Insertion", "Computer Installation", "Open the Bottle"],
    "status": "tested",
    "entrypoint": "weld_gap_check",
}

import fnmatch
import json
import sys

import mujoco
import numpy as np

from tool_library.common import trajectory as TJ


def check_weld_gap(scene, trajectory, gap_tol_m=0.005, angle_tol_deg=3.0, only=None):
    model = TJ.load_model(scene)
    Q, labels, fps = TJ.as_trajectory(trajectory)
    data = mujoco.MjData(model)
    eqs = []
    for e in range(model.neq):
        kind = int(model.eq_type[e])
        if not model.eq_active0[e] or kind not in (TJ.WELD, TJ.CONNECT):
            continue
        b1, b2 = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
        n1, n2 = model.body(b1).name, model.body(b2).name
        if only and not any(fnmatch.fnmatchcase(n, p) for n in (n1, n2) for p in only):
            continue
        d = np.asarray(model.eq_data[e], float)
        if kind == TJ.CONNECT:
            eqs.append((e, b1, b2, d[0:3], d[3:6], None, True))
        else:
            # weld data: anchor on body2 (0:3), relpose (3:6 pos, 6:10 quat), torquescale (10). MuJoCo's residual compares
            # the point x1 + R1 @ relpose_pos with x2 + R2 @ anchor (engine: mj_instantiateEquality)
            rel_p, rel_q = d[3:6], d[6:10]
            if np.linalg.norm(rel_q) < 1e-9:     # relpose left to the compiler: the weld holds the qpos0 configuration
                mujoco.mj_resetData(model, data)
                mujoco.mj_kinematics(model, data)
                R1, R2 = data.xmat[b1].reshape(3, 3), data.xmat[b2].reshape(3, 3)
                Rrel = R1.T @ R2
                rel_p = R1.T @ (data.xpos[b2] + R2 @ d[0:3] - data.xpos[b1])
            else:
                Rrel = TJ.quat2mat(rel_q)
            eqs.append((e, b1, b2, rel_p, d[0:3], Rrel, float(d[10]) == 0.0))
    rep, ok = {"n_constraints": len(eqs)}, True
    gaps = np.zeros((len(Q), len(eqs)))
    angs = np.zeros((len(Q), len(eqs)))
    for t in range(len(Q)):
        data.qpos[:] = Q[t]
        mujoco.mj_kinematics(model, data)
        for k, (e, b1, b2, a1, a2, Rrel, pos_only) in enumerate(eqs):
            p1 = data.xpos[b1] + data.xmat[b1].reshape(3, 3) @ a1
            p2 = data.xpos[b2] + data.xmat[b2].reshape(3, 3) @ a2
            gaps[t, k] = np.linalg.norm(p1 - p2)
            if Rrel is not None:
                R1, R2 = data.xmat[b1].reshape(3, 3), data.xmat[b2].reshape(3, 3)
                angs[t, k] = np.degrees(TJ.angle_between(R1 @ Rrel, R2))
    for k, (e, b1, b2, a1, a2, Rrel, pos_only) in enumerate(eqs):
        i = int(np.argmax(gaps[:, k])); j = int(np.argmax(angs[:, k]))
        r = {"bodies": [model.body(b1).name, model.body(b2).name], "position_only": bool(pos_only),
             "gap_mm": {"max": round(1000 * float(gaps[i, k]), 2), "limit": 1000 * gap_tol_m,
                        "at": i, "label": labels[i], "p99": round(1000 * float(np.percentile(gaps[:, k], 99)), 2)},
             "angle_deg": {"max": round(float(angs[j, k]), 2), "limit": angle_tol_deg if not pos_only else None,
                           "at": j, "label": labels[j]}}
        r["ok"] = bool(gaps[:, k].max() <= gap_tol_m and (pos_only or angs[:, k].max() <= angle_tol_deg))
        ok &= r["ok"]
        rep[f"eq{e}"] = r
    rep["ok"] = bool(ok)
    return rep


if __name__ == "__main__":
    rep = check_weld_gap(sys.argv[1], sys.argv[2])
    print(json.dumps(rep, indent=1, default=str)); sys.exit(0 if rep["ok"] else 1)


weld_gap_check = check_weld_gap   # the library calls every tool by its own name
