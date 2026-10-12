"""Arm jerk check: does the arm command accelerate harder than a controller follows?

    from tool_library.verifiers.arm_jerk_check import check_arm_jerk
    check_arm_jerk("scene.xml", "trajectory.npz", end_effectors={"right": "..."})

The per-frame step limit (joint_speed_check) misses a command that reverses
direction at full speed. The second difference of each arm joint, in degrees
per frame squared, catches that; 3.0 deg/frame^2 at 30 fps was the budget
the deployed episodes were held to, measured as the level above which the
real arm's tracking error visibly lagged the plan.
"""
from __future__ import annotations

TOOL = {
    "name": "arm_jerk_check",
    "title": "Arm jerk check",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "Second difference of every arm joint per frame against a replayable budget.",
    "description": (
        "Takes the arm joint groups of a trajectory (discovered from the end-effector bodies "
        "or named explicitly; with neither, the scalar joints of robot bodies whose body or "
        "joint name contains '_arm', one group per side, so the hand's wrist and finger joints "
        "are never held to the arm budget), computes the frame-to-frame change of the per-frame step "
        "(the discrete acceleration) in degrees per frame squared, and fails when any joint "
        "exceeds the budget, 3.0 deg/frame^2 at 30 fps scaled to the trajectory's frame rate. "
        "Reports the worst frame and phase label per group and how many frames are over the "
        "budget, so the fix can target one transit instead of smoothing the whole episode."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF scene."},
        {"name": "trajectory", "type": "str | ndarray [T, nq]", "description": "trajectory.npz or the qpos array."},
        {"name": "end_effectors", "type": "dict side -> body name | None", "description": "Finds the arm chain per side."},
        {"name": "chains", "type": "dict name -> list[str] | None", "description": "Explicit joint groups instead."},
        {"name": "limit_deg", "type": "float", "description": "Budget in deg/frame^2 at 30 fps. Default 3.0."},
        {"name": "robot_prefix", "type": "str", "description": "Name prefix of robot bodies, used to find the arm joints when neither end_effectors nor chains is given. Default 'robot_'."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; per group: max, limit, p99, worst frame and label, frames over the limit."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["smoothness", "jerk", "replay", "arm"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope", "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting", "Biology Experiment"],
    "status": "tested",
    "entrypoint": "arm_jerk_check",
}

import json
import sys

import numpy as np

from tool_library.common import trajectory as TJ


def _default_arm_groups(model, robot_prefix):
    """Arm joints when neither end effectors nor chains are named: scalar joints of robot bodies whose body or joint
    name contains '_arm', one group per side (the hand's own wrist and finger joints are not the arm)."""
    groups = {}
    for j in range(model.njnt):
        if int(model.jnt_type[j]) not in (TJ.HINGE, TJ.SLIDE):
            continue
        body = model.body(int(model.jnt_bodyid[j])).name
        if not body.startswith(robot_prefix) or "_arm" not in f"{body} {model.joint(j).name}":
            continue
        side = next((s for s in ("left", "right") if body.startswith(f"{robot_prefix}{s}_")), None)
        groups.setdefault(f"{side}_arm" if side else "robot_arm", []).append(int(model.jnt_qposadr[j]))
    return {k: np.asarray(v, int) for k, v in groups.items()}


def check_arm_jerk(scene, trajectory, end_effectors=None, chains=None, limit_deg=None,
                   robot_prefix="robot_"):
    model = TJ.load_model(scene)
    Q, labels, fps = TJ.as_trajectory(trajectory)
    limit = (TJ.STEP_LIMITS["arm_jerk_deg"] if limit_deg is None else limit_deg) * (30.0 / fps) ** 2
    if chains or end_effectors:
        groups = TJ.resolve_chains(model, chains, end_effectors, robot_prefix)
        groups = {k: v for k, v in groups.items() if k.endswith("_arm") or chains}
    else:
        groups = _default_arm_groups(model, robot_prefix)
    if not groups:
        raise ValueError("no arm joint group found: the end effectors have no arm bolted on and no robot body or joint "
                         "name contains '_arm'; pass chains")
    rep, ok = {"fps": fps}, True
    for name, qadr in groups.items():
        acc = np.degrees(np.abs(np.diff(Q[:, qadr], n=2, axis=0)).max(axis=1)) if len(Q) > 2 else []
        w = TJ.worst_step(acc, limit, labels, offset=2)
        w["unit"] = "deg/frame^2"
        rep[name] = w
        ok &= w["ok"]
    rep["ok"] = bool(ok)
    return rep


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=TOOL["summary"])
    ap.add_argument("scene"); ap.add_argument("trajectory"); ap.add_argument("--ee", nargs="*", default=[])
    a = ap.parse_args()
    rep = check_arm_jerk(a.scene, a.trajectory, end_effectors=dict(kv.split("=", 1) for kv in a.ee) or None)
    print(json.dumps(rep, indent=1, default=str)); sys.exit(0 if rep["ok"] else 1)


arm_jerk_check = check_arm_jerk   # the library calls every tool by its own name
