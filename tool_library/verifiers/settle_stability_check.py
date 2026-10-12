"""Settle stability check: do the free objects stay where they were placed?"""
from __future__ import annotations

TOOL = {
    "name": "settle_stability_check",
    "title": "Settle stability check",
    "type": "verifier",
    "category": "Scene Compile Checks",
    "summary": "Simulates the scene for a few seconds from its initial state; no free body may "
               "drift, fall or tip over.",
    "description": (
        "An object reconstructed at rest must stay at rest. The scene is stepped from its "
        "initial state for `seconds` (2 s by default) and every body with a free joint is "
        "measured against where it started: translation and rotation. A body that moves more "
        "than `drift_fail` (5 cm) or turns more than `rot_fail_deg` (30 degrees) fails; a body "
        "that moves more than `drift_warn` (1.5 cm) is reported as a warning. A translation "
        "with little rotation points at a placement error (the object hovers or sinks); a "
        "rotation beyond 90 degrees points at a wrong fitted orientation (it tips over). A "
        "diverging simulation is a failure of its own. The settled state is also returned so "
        "other checks can read it."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF file, XML string, or compiled model."},
        {"name": "seconds", "type": "float", "description": "Settle time. Default 2.0."},
        {"name": "bodies", "type": "list[str] | None", "description": "Free bodies to judge; all of them when omitted."},
        {"name": "drift_fail", "type": "float, metres", "description": "Translation that fails. Default 0.05."},
        {"name": "rot_fail_deg", "type": "float, degrees", "description": "Rotation that fails. Default 30."},
        {"name": "drift_warn", "type": "float, metres", "description": "Translation that is reported as a warning. Default 0.015."},
    ],
    "outputs": [
        {"name": "report", "type": "dict",
         "description": "ok; drift: {body: {pos_m, rot_deg, verdict}}; failures and warnings as lists of "
                        "body names; diverged (bool); settled_qpos: the state after settling."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["compile", "physics", "stability", "settle"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Battery Insertion", "Open the Bottle"],
    "status": "tested",
    "entrypoint": "settle_stability_check",
    "cli": "python -m tool_library.verifiers.settle_stability_check scene.xml",
}

import json
import sys

import numpy as np

from tool_library.common import scene as S


def check_settle_stability(scene, seconds=2.0, bodies=None, drift_fail=0.05,
                           rot_fail_deg=30.0, drift_warn=0.015):
    model = S.load_model(scene)
    data = S.forward(model)
    free = S.free_bodies(model) if bodies is None else [S.body_id(model, b) for b in bodies]
    start = {b: (data.xpos[b].copy(), data.xmat[b].copy()) for b in free}
    # the first step: a keyframe that leaves a welded or jointed part off its mate snaps it there with a huge
    # acceleration; MuJoCo then warns and silently resets the state, and every later measurement lies
    snap = []
    badacc = False
    worst = {b: 0.0 for b in free}
    prev = {b: start[b][0].copy() for b in free}
    for _ in range(int(round(0.05 / model.opt.timestep))):      # the first 50 ms, step by step
        S.mujoco.mj_step(model, data)
        if int(data.warning[S.mujoco.mjtWarning.mjWARN_BADQACC].number) > 0:
            badacc = True
            break
        for b in free:
            worst[b] = max(worst[b], float(np.linalg.norm(data.xpos[b] - prev[b])) / model.opt.timestep)
            prev[b] = data.xpos[b].copy()
    for b in free:
        moved = float(np.linalg.norm(data.xpos[b] - start[b][0]))
        if moved > 0.005 or worst[b] > 5.0:                       # a 5 mm snap, or an impulsive speed
            snap.append({"body": model.body(b).name, "peak_speed_mps": round(worst[b], 2), "moved_mm": round(1000 * moved, 1)})
    if badacc or snap:
        return {"ok": False, "diverged": badacc, "keyframe_snap": snap, "drift": {}, "failures": [x["body"] for x in snap] or [model.body(b).name for b in free],
                "warnings": [], "issues": ["the initial state is not at rest: " + ("MuJoCo reset the simulation on the first step (bad acceleration); " if badacc else "")
                                           + ", ".join(f"{x['body']} snaps {x['moved_mm']} mm (peak {x['peak_speed_mps']} m/s) in the first 50 ms" for x in snap)
                                           + "; a welded or jointed part (a cap on its thread, a lid on its hinge) sits off its mate in the keyframe: re-place the asset so its parts start where the joints hold them"]}
    try:
        S.settle(model, data, seconds)
    except FloatingPointError as exc:
        return {"ok": False, "diverged": True, "error": str(exc), "drift": {},
                "failures": [model.body(b).name for b in free], "warnings": []}
    drift, failures, warnings = {}, [], []
    for b in free:
        name = model.body(b).name
        dp = float(np.linalg.norm(data.xpos[b] - start[b][0]))
        ang = S.rotation_angle_deg(start[b][1], data.xmat[b])
        if dp >= drift_fail or ang >= rot_fail_deg:
            verdict = "tips over" if ang >= 90.0 else "unstable"
            failures.append(name)
        elif dp >= drift_warn:
            verdict = "drifts"
            warnings.append(name)
        else:
            verdict = "at rest"
        drift[name] = {"pos_m": round(dp, 4), "rot_deg": round(ang, 1), "verdict": verdict}
    return {"ok": not failures, "diverged": False, "drift": drift, "failures": failures,
            "warnings": warnings, "seconds": seconds, "settled_qpos": data.qpos.tolist()}


def main(argv=None):
    rep = check_settle_stability(sys.argv[1] if argv is None else argv[0])
    rep.pop("settled_qpos", None)
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())


settle_stability_check = check_settle_stability   # the library calls every tool by its own name
