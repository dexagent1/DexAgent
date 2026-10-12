"""Joint speed check: does any robot joint move faster than a replay can follow?

    from tool_library.verifiers.joint_speed_check import check_joint_speed
    check_joint_speed("scene.xml", "trajectory.npz", end_effectors={"right": "..."})

A trajectory is judged per frame, so the number that matters is the largest
step between consecutive frames of each joint group. Two ways to set the
limit: absolute (degrees per frame for hinges, with the paper's replay
budgets as defaults) or relative to a reference trajectory (scale x the
reference's own fastest step + a floor), which is how an adapted episode is
held to the verified one it was derived from.
"""
from __future__ import annotations

TOOL = {
    "name": "joint_speed_check",
    "title": "Joint speed check",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "Largest per-frame step of every robot joint group against an absolute budget "
               "or a reference trajectory's own speed.",
    "description": (
        "Reads the qpos trajectory, groups the robot's scalar joints (arm, end effector, "
        "fingers per side, found from the end-effector bodies, or given explicitly) and "
        "computes the largest change between consecutive frames of each group. Without a "
        "reference the budgets are the replay limits the paper's episodes were held to: "
        "3.5 deg/frame for arm joints, 15 mm/frame or 8 deg/frame for a floating wrist and "
        "0.30 rad/frame for fingers at 30 fps; they scale with the frame rate. With a "
        "reference trajectory the limit is scale x the reference's fastest step plus a floor, "
        "so an adapted episode may not move faster than the one it came from. Reports the "
        "worst frame and phase label per group."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF scene the trajectory belongs to."},
        {"name": "trajectory", "type": "str | ndarray [T, nq]", "description": "trajectory.npz (qpos, labels, fps) or the qpos array."},
        {"name": "end_effectors", "type": "dict side -> body name | None", "description": "Finds arm / end effector / finger joint groups per side."},
        {"name": "chains", "type": "dict name -> list[str] | None", "description": "Explicit joint-name groups instead of discovery."},
        {"name": "reference", "type": "str | ndarray | None", "description": "Reference trajectory; limits become scale x its fastest step + floor."},
        {"name": "limits", "type": "dict | None", "description": "Override the per-frame budgets: arm_deg, wrist_mm, wrist_deg, finger (rad)."},
        {"name": "scale, floor", "type": "float", "description": "Relative mode: 1.5 x reference + 0.5 rad/s (or m/s)."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; per group: max step, limit, p99, frame and label of the worst step, frames over the limit."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["smoothness", "speed", "replay", "trajectory"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope", "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting", "Biology Experiment"],
    "status": "tested",
    "entrypoint": "joint_speed_check",
    "cli": "python -m tool_library.verifiers.joint_speed_check scene.xml trajectory.npz --ee right=robot_right_right_base_yaw",
}

import json
import sys

import numpy as np

from tool_library.common import trajectory as TJ


def _group_kind(name, model, qadr):
    """arm | wrist (free pose joints of a floating hand) | finger, from the group name
    and the joint types; decides which budget applies."""
    if name.endswith("_arm") or name == "robot":
        return "arm"
    if name.endswith("_fingers"):
        return "finger"
    slides = [j for j in range(model.njnt) if int(model.jnt_qposadr[j]) in set(qadr)
              and int(model.jnt_type[j]) == TJ.SLIDE]
    return "wrist" if slides else "arm"


def check_joint_speed(scene, trajectory, end_effectors=None, chains=None, reference=None,
                      limits=None, scale=1.5, floor=0.5, robot_prefix="robot_"):
    model = TJ.load_model(scene)
    Q, labels, fps = TJ.as_trajectory(trajectory)
    lim = {**TJ.STEP_LIMITS, **(limits or {})}
    fscale = 30.0 / fps                      # budgets were measured per frame at 30 fps
    groups = TJ.resolve_chains(model, chains, end_effectors, robot_prefix)
    Qr = TJ.as_trajectory(reference)[0] if reference is not None else None
    rep, ok = {"fps": fps, "mode": "reference" if Qr is not None else "absolute"}, True
    for name, qadr in groups.items():
        kind = _group_kind(name, model, qadr)
        # one flag per column of `steps`, in qadr order (explicit chains keep the caller's joint order)
        slide_cols = {int(model.jnt_qposadr[j]) for j in range(model.njnt) if int(model.jnt_type[j]) == TJ.SLIDE}
        is_slide = np.array([int(a) in slide_cols for a in qadr], bool)
        steps = np.abs(np.diff(Q[:, qadr], axis=0)) if len(Q) > 1 else np.zeros((0, len(qadr)))
        if Qr is not None:
            v = steps.max(axis=1) * fps if len(steps) else np.zeros(0)
            v0 = (np.abs(np.diff(Qr[:, qadr], axis=0)).max() * fps) if len(Qr) > 1 else 0.0
            w = TJ.worst_step(v, scale * v0 + floor, labels)
            w["unit"] = "rad/s or m/s"
            w["reference_max"] = round(float(v0), 3)
        elif kind == "finger":
            w = TJ.worst_step(steps.max(axis=1) if len(steps) else [], lim["finger"] * fscale, labels)
            w["unit"] = "rad/frame"
        elif kind == "arm":
            w = TJ.worst_step(np.degrees(steps.max(axis=1)) if len(steps) else [], lim["arm_deg"] * fscale, labels)
            w["unit"] = "deg/frame"
        else:                                # floating wrist: slides in mm, hinges in degrees
            if len(steps) and is_slide.any():
                mm = 1000.0 * steps[:, is_slide].max(axis=1)
                w = TJ.worst_step(mm, lim["wrist_mm"] * fscale, labels)
                w["unit"] = "mm/frame"
                if (~is_slide).any():
                    w2 = TJ.worst_step(np.degrees(steps[:, ~is_slide].max(axis=1)), lim["wrist_deg"] * fscale, labels)
                    w2["unit"] = "deg/frame"
                    w["rotation"] = w2
                    w["ok"] = bool(w["ok"] and w2["ok"])
            else:
                w = TJ.worst_step(np.degrees(steps.max(axis=1)) if len(steps) else [], lim["wrist_deg"] * fscale, labels)
                w["unit"] = "deg/frame"
        w["kind"] = kind
        rep[name] = w
        ok &= w["ok"]
    rep["ok"] = bool(ok)
    return rep


def _cli(argv):
    import argparse
    ap = argparse.ArgumentParser(description=TOOL["summary"])
    ap.add_argument("scene"); ap.add_argument("trajectory")
    ap.add_argument("--ee", nargs="*", default=[], help="side=body")
    ap.add_argument("--reference")
    a = ap.parse_args(argv)
    ee = dict(kv.split("=", 1) for kv in a.ee) or None
    rep = check_joint_speed(a.scene, a.trajectory, end_effectors=ee, reference=a.reference)
    print(json.dumps(rep, indent=1, default=str))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))


joint_speed_check = check_joint_speed   # the library calls every tool by its own name
