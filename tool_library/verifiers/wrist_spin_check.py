"""Wrist spin check: no turn of half a revolution or more, no turning back and forth.

    from tool_library.verifiers.wrist_spin_check import check_wrist_spin
    check_wrist_spin("scene.xml", "trajectory.npz", joints=["robot_right_arm_j6"])

Two rules every deployed episode followed. A wrist never turns 180 degrees
or more in one go: that is a spin a human never makes and a cable the arm
does not have. And it does not turn redundantly: turning one way and back
without a phase change in between is IK wandering, not the task. The check
measures the largest monotone excursion of each roll joint and counts the
reversals that are larger than noise.
"""
from __future__ import annotations

TOOL = {
    "name": "wrist_spin_check",
    "title": "Wrist spin check",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "Largest monotone turn of each wrist joint stays under 180 degrees and the joint does not wander back and forth.",
    "description": (
        "Splits each named hinge trajectory into monotone runs (ignoring reversals smaller "
        "than the noise floor) and reports the largest run in degrees, failing when it reaches "
        "the spin limit (180 degrees by default). It also counts reversals larger than "
        "redundant_deg (15 degrees by default) that happen inside one phase label; a wrist "
        "that turns 30 degrees one way and back within the same phase did not need to turn. "
        "Joints default to every hinge whose name contains 'roll' or ends in the last arm "
        "joint index; pass them explicitly for other robots."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF scene."},
        {"name": "trajectory", "type": "str | ndarray [T, nq]", "description": "trajectory.npz or qpos array; phase labels enable the redundancy rule."},
        {"name": "joints", "type": "list[str] | None", "description": "Hinge joints to check. Default: hinges with 'roll', 'j6' or 'wrist' in their name."},
        {"name": "spin_limit_deg", "type": "float, degrees", "description": "Largest monotone turn allowed. Default 180."},
        {"name": "redundant_deg", "type": "float, degrees", "description": "A reversal this large inside one phase counts as redundant turning. Default 15."},
        {"name": "noise_deg", "type": "float, degrees", "description": "Reversals below this are jitter. Default 1."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; per joint: largest turn, its frames and labels, net turn, redundant reversals (frame, size, label)."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["wrist", "spin", "redundancy", "deployability"],
    "used_in": ["Place Toy in Drawer", "Giftbox Packing", "Battery Insertion", "Open the Bottle", "Scissor Cutting"],
    "status": "tested",
    "entrypoint": "wrist_spin_check",
}

import json
import sys

import numpy as np

from tool_library.common import trajectory as TJ


def monotone_runs(q, noise):
    """[(start, end, signed turn)] of monotone stretches; a reversal smaller
    than noise does not end a run."""
    runs, s, e, direction = [], 0, 0, 0
    for t in range(1, len(q)):
        if direction == 0:
            if abs(q[t] - q[s]) > noise:
                direction, e = np.sign(q[t] - q[s]), t
            continue
        if direction * (q[t] - q[e]) >= 0:
            e = t                                # still going the same way
        elif direction * (q[e] - q[t]) > noise:  # turned back: close the run at the extreme
            runs.append((s, e, float(q[e] - q[s])))
            s, direction, e = e, -direction, t
    runs.append((s, len(q) - 1, float(q[-1] - q[s])))
    return runs


def check_wrist_spin(scene, trajectory, joints=None, spin_limit_deg=180.0, redundant_deg=15.0,
                     noise_deg=1.0):
    model = TJ.load_model(scene)
    Q, labels, fps = TJ.as_trajectory(trajectory)
    if joints is None:
        joints = [model.joint(j).name for j in range(model.njnt) if int(model.jnt_type[j]) == TJ.HINGE
                  and any(k in model.joint(j).name.lower() for k in ("roll", "j6", "wrist", "joint6"))]
        if not joints:
            raise ValueError("no wrist hinge found by name; pass joints=[...]")
    rep, ok = {}, True
    for name in joints:
        j = TJ.joint_id(model, name)
        q = np.degrees(Q[:, int(model.jnt_qposadr[j])])
        runs = monotone_runs(q, noise_deg)
        big = max(runs, key=lambda r: abs(r[2]))
        redundant = []
        for a, b in zip(runs[:-1], runs[1:]):
            if abs(a[2]) < redundant_deg or abs(b[2]) < redundant_deg:
                continue
            e = a[1]                             # the reversal frame
            # the last redundant_deg of the turn out, and the first of the turn back,
            # must both happen inside the phase the reversal is in
            t_a = max(t for t in range(a[0], e + 1) if abs(q[t] - q[e]) >= redundant_deg)
            t_b = min(t for t in range(e, b[1] + 1) if abs(q[t] - q[e]) >= redundant_deg)
            if labels[t_a] == labels[e] == labels[t_b]:
                redundant.append({"frame": e, "turn_deg": round(abs(a[2]), 1),
                                  "back_deg": round(abs(b[2]), 1), "label": labels[e]})
        r = {"largest_turn_deg": round(abs(big[2]), 1), "turn_frames": [big[0], big[1]],
             "turn_labels": [labels[big[0]], labels[big[1]]], "net_turn_deg": round(float(q[-1] - q[0]), 1),
             "n_runs": len(runs), "redundant_reversals": redundant,
             "ok": bool(abs(big[2]) < spin_limit_deg and not redundant)}
        ok &= r["ok"]
        rep[name] = r
    rep["ok"] = bool(ok)
    return rep


if __name__ == "__main__":
    rep = check_wrist_spin(sys.argv[1], sys.argv[2])
    print(json.dumps(rep, indent=1, default=str)); sys.exit(0 if rep["ok"] else 1)


wrist_spin_check = check_wrist_spin   # the library calls every tool by its own name
