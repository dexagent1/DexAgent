"""Wrist stop check: does the wrist roll stay inside the REAL arm's stops?

    from tool_library.verifiers.wrist_stop_check import check_wrist_stop
    check_wrist_stop("scene.xml", "trajectory.npz",
                     joints={"robot_right_arm_j6": (-20, 180), "robot_left_arm_j6": (-180, 20)})

A simulated hinge is often given a full turn of range so IK never gets stuck.
The real joint has stops, and a plan that crosses them cannot be executed.
This check holds named joints to the real ranges, with a margin, and also
reports how close each joint comes to the stop, which is where the next
plan should keep a reserve.
"""
from __future__ import annotations

TOOL = {
    "name": "wrist_stop_check",
    "title": "Wrist stop check",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "Named joints stay inside the real hardware's ranges, with a margin, on every frame.",
    "description": (
        "Compares each named joint's trajectory against the range the real hardware can "
        "reach (degrees for hinges, metres for slides), minus a safety margin, and fails on "
        "any frame outside it. By default it finds the wrist hinges by name (roll, wrist, j6) "
        "and uses the scene's own joint limits (a joint the scene leaves unlimited, with no real range given, is "
        "skipped and listed); pass joints={name: (lo, hi)} with the real "
        "stops, or real_stops={side: (lo, hi)} to apply one range per side by name. The "
        "report gives each joint's minimum and maximum, the remaining margin to each stop and "
        "the frame and phase label of the closest approach. The module carries the real "
        "wrist-roll stops the paper's arms were deployed with, right (-20, 180) and left "
        "(-180, 20) degrees, as REAL_WRIST_STOPS_DEG."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF scene."},
        {"name": "trajectory", "type": "str | ndarray [T, nq]", "description": "trajectory.npz or qpos array."},
        {"name": "joints", "type": "dict joint name -> (lo, hi) | list[str] | None",
         "description": "Joints to check with their real ranges (degrees for hinges). A list uses the model's limits; None finds wrist hinges by name."},
        {"name": "real_stops", "type": "dict side -> (lo, hi) | None", "description": "Real range applied to each default joint whose name contains the side."},
        {"name": "margin_deg", "type": "float, degrees", "description": "Reserve kept from each explicit real range. Default 2; the model's own limits get 0."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; per joint: min, max, range, margin to each stop, worst frame and label; unlimited_skipped: joints with no range to check."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["joint limits", "wrist", "deployability", "real robot"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope", "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting", "Biology Experiment"],
    "status": "tested",
    "entrypoint": "wrist_stop_check",
}

import json
import sys

import numpy as np

from tool_library.common import trajectory as TJ

# the real wrist-roll stops of the deployed arms, degrees, keyed by a name
# fragment: handy defaults for scenes that follow the robot_<side>_arm naming
REAL_WRIST_STOPS_DEG = {"right": (-20.0, 180.0), "left": (-180.0, 20.0)}


WRIST_NAME_HINTS = ("roll", "wrist", "j6", "joint6")


def check_wrist_stop(scene, trajectory, joints=None, real_stops=None, margin_deg=2.0, robot_prefix="robot_"):
    model = TJ.load_model(scene)
    Q, labels, fps = TJ.as_trajectory(trajectory)
    if joints is None:
        names = [model.joint(j).name for j in range(model.njnt)
                 if int(model.jnt_type[j]) == TJ.HINGE and model.body(model.jnt_bodyid[j]).name.startswith(robot_prefix)
                 and any(h in model.joint(j).name.lower() for h in WRIST_NAME_HINTS)]
        if not names:
            raise ValueError("no wrist hinge found by name; pass joints")
        joints = {}
        for n in names:
            side = next((s for s in (real_stops or {}) if s in n.lower()), None)
            joints[n] = tuple(real_stops[side]) if side else None
    if isinstance(joints, (list, tuple)):
        joints = {n: None for n in joints}
    rep, ok, unlimited = {}, True, []
    for name, rng in joints.items():
        j = TJ.joint_id(model, name)
        hinge = int(model.jnt_type[j]) == TJ.HINGE
        if rng is None and not model.jnt_limited[j]:   # no real range given and no model limit: nothing to hold it to
            unlimited.append(name)
            continue
        q = Q[:, int(model.jnt_qposadr[j])]
        q = np.degrees(q) if hinge else q
        if rng is None:                          # the model's own limit: being on it is legal
            lo, hi = model.jnt_range[j]
            lo, hi = (np.degrees(lo), np.degrees(hi)) if hinge else (lo, hi)
            m, source = (-0.01 if hinge else -1e-4), "model"   # clamp precision, not a reserve
        else:
            lo, hi = float(rng[0]), float(rng[1])
            m, source = (margin_deg if hinge else margin_deg / 1000.0), "real"
        i_lo, i_hi = int(np.argmin(q)), int(np.argmax(q))
        near_low = (q[i_lo] - lo) < (hi - q[i_hi])
        r = {"unit": "deg" if hinge else "m", "range": [round(float(lo), 2), round(float(hi), 2)],
             "range_source": source, "margin": m,
             "min": round(float(q[i_lo]), 2), "max": round(float(q[i_hi]), 2),
             "margin_to_low_stop": round(float(q[i_lo] - lo), 2),
             "margin_to_high_stop": round(float(hi - q[i_hi]), 2),
             "closest": {"frame": i_lo if near_low else i_hi, "label": labels[i_lo if near_low else i_hi]},
             "frames_outside": int(((q < lo + m) | (q > hi - m)).sum())}
        r["ok"] = r["frames_outside"] == 0
        ok &= r["ok"]
        rep[name] = r
    if unlimited:
        rep["unlimited_skipped"] = unlimited
    rep["ok"] = bool(ok)
    return rep


if __name__ == "__main__":
    rep = check_wrist_stop(sys.argv[1], sys.argv[2])
    print(json.dumps(rep, indent=1, default=str)); sys.exit(0 if rep["ok"] else 1)


wrist_stop_check = check_wrist_stop   # the library calls every tool by its own name
