"""Object disturbance check: objects the task does not touch must not move.

    from tool_library.verifiers.object_disturbance_check import check_object_disturbance
    check_object_disturbance("scene.xml", "trajectory.npz", manipulated=["cup_a"])

In a physics replay a stray finger, a sleeve of the arm or a dropped object
nudges the rest of the scene. Every free body that is not declared as
manipulated must end where it started, within 10 mm and a few degrees, and
no body (manipulated or not) may jump between frames.
"""
from __future__ import annotations

TOOL = {
    "name": "object_disturbance_check",
    "title": "Object disturbance check",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "Untouched free bodies stay within 10 mm and 5 degrees of their start; no body jumps between frames.",
    "description": (
        "Finds every free-jointed body outside the robot, removes the ones declared as "
        "manipulated, and measures each remaining body's largest displacement and rotation "
        "from its first frame over the trajectory. Fails above 10 mm or 5 degrees, the level "
        "at which a disturbed scene reads as a different task on video. Separately, every "
        "free body, manipulated or not, is checked for continuity: a per-frame translation "
        "step above 0.03 m at 30 fps (0.9 m/s, faster than any tabletop manipulation) is a "
        "teleport (a reset, a lost grasp re-seated by a constraint) and fails too. With a "
        "reference trajectory the continuity limit is the reference's own step plus 0.01 m, "
        "so an adapted episode may move an object as the original did and no faster."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF scene."},
        {"name": "trajectory", "type": "str | ndarray [T, nq]", "description": "trajectory.npz or qpos array."},
        {"name": "manipulated", "type": "list[str]", "description": "Bodies the task is allowed to move (fnmatch patterns)."},
        {"name": "reference", "type": "str | ndarray | None", "description": "Original trajectory for the relative continuity rule."},
        {"name": "move_tol_m, rot_tol_deg", "type": "float", "description": "Allowed drift of an untouched object. Defaults 0.010 m, 5 deg."},
        {"name": "step_tol_m", "type": "float, metres", "description": "Per-frame translation step above which a body jumped. Default 0.03 at 30 fps; 0.01 above the reference's step in relative mode."},
        {"name": "robot_prefix", "type": "str", "description": "Bodies with this prefix are the robot. Default 'robot_'."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; untouched: per body max drift (mm, deg) and the frame; steps: per body largest frame step (mm) and frame; lists of offenders."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["disturbance", "continuity", "teleport", "scene"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope", "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting", "Biology Experiment"],
    "status": "tested",
    "entrypoint": "object_disturbance_check",
}

import fnmatch
import json
import sys

import numpy as np

from tool_library.common import trajectory as TJ


def check_object_disturbance(scene, trajectory, manipulated=(), reference=None, move_tol_m=0.010,
                             rot_tol_deg=5.0, step_tol_m=0.03, robot_prefix="robot_"):
    model = TJ.load_model(scene)
    Q, labels, fps = TJ.as_trajectory(trajectory)
    objs = TJ.free_objects(model, robot_prefix)
    if not objs:
        return {"ok": True, "note": "no free bodies outside the robot", "untouched": {}, "steps": {}}
    Qr = TJ.as_trajectory(reference)[0] if reference is not None else None
    step_lim = step_tol_m * 30.0 / fps
    rep = {"untouched": {}, "steps": {}, "disturbed": [], "jumped": []}
    for name, adr in objs.items():
        P = Q[:, adr:adr + 3]
        step = np.linalg.norm(np.diff(P, axis=0), axis=1) if len(Q) > 1 else np.zeros(0)
        lim = np.full(len(step), step_lim)
        if Qr is not None and len(Qr) == len(Q):
            lim = 0.01 * 30.0 / fps + np.linalg.norm(np.diff(Qr[:, adr:adr + 3], axis=0), axis=1)
        over = step - lim
        i = int(np.argmax(over)) if len(over) else 0
        s = {"max_step_mm": round(1000 * float(step.max()), 2) if len(step) else 0.0,
             "limit_mm": round(1000 * float(lim[i] if len(step) else step_lim), 2),
             "at": i + 1, "label": labels[min(i + 1, len(labels) - 1)],
             "ok": bool(len(over) == 0 or over.max() <= 0)}
        rep["steps"][name] = s
        if not s["ok"]:
            rep["jumped"].append(name)
        if any(fnmatch.fnmatchcase(name, p) for p in manipulated):
            continue
        d = np.linalg.norm(P - P[0], axis=1)
        R0 = TJ.quat2mat(Q[0, adr + 3:adr + 7])
        ang = np.array([np.degrees(TJ.angle_between(R0, TJ.quat2mat(Q[t, adr + 3:adr + 7]))) for t in range(len(Q))])
        i, k = int(np.argmax(d)), int(np.argmax(ang))
        u = {"max_drift_mm": round(1000 * float(d[i]), 2), "drift_at": i, "drift_label": labels[i],
             "max_rot_deg": round(float(ang[k]), 2), "rot_at": k,
             "ok": bool(d[i] <= move_tol_m and ang[k] <= rot_tol_deg)}
        rep["untouched"][name] = u
        if not u["ok"]:
            rep["disturbed"].append(name)
    rep["ok"] = not rep["disturbed"] and not rep["jumped"]
    return rep


if __name__ == "__main__":
    rep = check_object_disturbance(sys.argv[1], sys.argv[2], manipulated=sys.argv[3:])
    print(json.dumps(rep, indent=1, default=str)); sys.exit(0 if rep["ok"] else 1)


object_disturbance_check = check_object_disturbance   # the library calls every tool by its own name
