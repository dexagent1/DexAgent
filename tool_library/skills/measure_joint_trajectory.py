"""Measure what a robot's joints do over an episode: ranges, speeds, stops touched.

    from tool_library.skills.measure_joint_trajectory import measure_joint_trajectory
    m = measure_joint_trajectory("scene.xml", "trajectory.npz", end_effectors={"right": "..."})
    m["groups"]["right_arm"]["joints"]["robot_right_arm_dof_joint6"]

Per joint: minimum, maximum, range used, fraction of the limit range used,
frames spent on a stop, peak speed and the frame it happens, and per group
the per-frame step profile. These are the numbers that say whether a plan
is executable before any gate is applied, and which joint to give room to
when it is not.
"""
from __future__ import annotations

TOOL = {
    "name": "measure_joint_trajectory",
    "title": "Measure joint trajectory",
    "type": "skill",
    "category": "Measurements",
    "summary": "Per joint and per group: range used, limit proximity, frames on a stop, peak speed and step profile.",
    "description": (
        "Groups the robot's scalar joints (per side: end effector, arm, fingers, discovered "
        "from the end-effector bodies and the arm welded to each; or explicit groups) and "
        "measures each joint's minimum, maximum, used range as a fraction of its limit range, "
        "frames within 0.5 degrees of a stop, peak speed in the joint's unit per second and "
        "the frame and label where it occurs. Per group it returns the largest per-frame step "
        "profile (degrees for hinges) and its 99th percentile, the same signal the "
        "joint_speed_check gates on. Also lists the free joints of the robot (a floating "
        "wrist) with their translation and rotation steps."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF scene."},
        {"name": "trajectory", "type": "str | ndarray [T, nq]", "description": "trajectory.npz or qpos array."},
        {"name": "end_effectors", "type": "dict side -> body name | None", "description": "Hand root bodies; finds the arm and fingers per side."},
        {"name": "chains", "type": "dict name -> list[str] | None", "description": "Explicit joint groups instead."},
    ],
    "outputs": [
        {"name": "measurement", "type": "dict", "description": "groups: per group peak step, p99, worst frame/label and per joint statistics; fps; n_frames."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["measurement", "joints", "speed", "limits"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope", "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting", "Biology Experiment"],
    "status": "tested",
    "entrypoint": "measure_joint_trajectory",
}

import json
import sys

import numpy as np

from tool_library.common import trajectory as TJ


def measure_joint_trajectory(scene, trajectory, end_effectors=None, chains=None, robot_prefix="robot_"):
    model = TJ.load_model(scene)
    Q, labels, fps = TJ.as_trajectory(trajectory)
    groups = TJ.resolve_chains(model, chains, end_effectors, robot_prefix)
    by_adr = {int(model.jnt_qposadr[j]): j for j in range(model.njnt)}
    out = {"fps": fps, "n_frames": len(Q), "groups": {}}
    for gname, qadr in groups.items():
        joints = {}
        steps = []
        for a in qadr:
            j = by_adr[int(a)]
            hinge = int(model.jnt_type[j]) == TJ.HINGE
            q = Q[:, a]
            conv = np.degrees if hinge else (lambda x: x)
            lo, hi = model.jnt_range[j] if model.jnt_limited[j] else (np.nan, np.nan)
            v = np.abs(np.diff(q)) * fps if len(q) > 1 else np.zeros(1)
            k = int(np.argmax(v))
            on_stop = int(((q <= lo + np.radians(0.5)) | (q >= hi - np.radians(0.5))).sum()) if model.jnt_limited[j] else 0
            joints[model.joint(j).name] = {
                "unit": "deg" if hinge else "m",
                "min": round(float(conv(q.min())), 3), "max": round(float(conv(q.max())), 3),
                "used_range": round(float(conv(q.max() - q.min())), 3),
                "limit_range": [round(float(conv(lo)), 3), round(float(conv(hi)), 3)] if model.jnt_limited[j] else None,
                "used_frac_of_limit": round(float((q.max() - q.min()) / (hi - lo)), 3) if model.jnt_limited[j] and hi > lo else None,
                "frames_on_stop": on_stop,
                "peak_speed_per_s": round(float(conv(v[k])), 3), "peak_at": {"frame": k + 1, "label": labels[min(k + 1, len(labels) - 1)]},
            }
            steps.append(np.abs(np.diff(conv(q))) if len(q) > 1 else np.zeros(0))
        S = np.max(np.stack(steps), axis=0) if steps and len(steps[0]) else np.zeros(0)
        i = int(np.argmax(S)) if len(S) else 0
        out["groups"][gname] = {
            "n_joints": len(qadr), "joints": joints,
            "max_step": round(float(S[i]), 4) if len(S) else 0.0,
            "p99_step": round(float(np.percentile(S, 99)), 4) if len(S) else 0.0,
            "max_step_at": {"frame": i + 1, "label": labels[min(i + 1, len(labels) - 1)]},
            "step_unit": "deg/frame (hinges) or m/frame (slides)",
        }
    free = {}
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == TJ.FREE and model.body(model.jnt_bodyid[j]).name.startswith(robot_prefix):
            a = int(model.jnt_qposadr[j])
            dp = 1000 * np.linalg.norm(np.diff(Q[:, a:a + 3], axis=0), axis=1)
            R = np.stack([TJ.quat2mat(q) for q in Q[:, a + 3:a + 7]])
            dr = TJ.rotation_steps_deg(R)
            free[model.joint(j).name] = {"max_step_mm": round(float(dp.max()), 2), "max_step_deg": round(float(dr.max()), 2)}
    if free:
        out["free_joints"] = free
    return out


if __name__ == "__main__":
    ee = dict(kv.split("=", 1) for kv in sys.argv[3:]) or None
    print(json.dumps(measure_joint_trajectory(sys.argv[1], sys.argv[2], end_effectors=ee), indent=1))
