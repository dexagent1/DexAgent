"""Put the arms behind the hands on every frame of an episode."""
from __future__ import annotations

TOOL = {
    "name": "fill_arm_joints",
    "title": "Fill arm joints",
    "type": "skill",
    "category": "Trajectory & Execution",
    "summary": "Fit each arm's joints frame by frame so its flange holds the hand it is bolted to (the hand's pose times "
               "the fixed mount), by exact 6-DoF IK with warm starts and branch continuity, and write the episode with the "
               "arm columns filled.",
    "description": (
        "The hands are driven by their virtual wrist servos during execution and the arm-hand welds are off; the arm "
        "columns of the executed trajectory still hold the home joints. This skill discovers each side's arm chain "
        "(the gripper body welded to the hand's end-effector body), reads the fixed flange-to-hand rotation and offset "
        "from the scene's home keyframe, and fits the chain to every frame with the continuous arm fitter (trust "
        "region, limit-aware, a multi-start reset when the warm chain is badly off, blended so the command stays "
        "smooth). Reports per side the position and tilt residuals, branch resets and the smallest joint-limit "
        "margin; arm_reach_check turns the residuals into a pass/fail."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "Bench scene MJCF."},
        {"name": "trajectory", "type": "str", "description": "Episode npz (qpos, labels, fps) with the hands' motion."},
        {"name": "out_npz", "type": "str", "description": "Where to write the filled episode."},
        {"name": "sides", "type": "list[str] | None", "description": "Arms to fit (default both)."},
        {"name": "max_arm_rate_deg_s", "type": "float", "description": "method='legacy' only: joint-rate ceiling of the fit (95 deg/s; a hand that leaves home faster leaves the arm behind, which arm_reach_check then reports)."},
        {"name": "method", "type": "str", "description": "'exact' (default): strict 6-DoF IK of the flange under the hand on every frame, warm-started and branch-continuous, never rate-limited (an arm that cannot keep up shows as a joint-speed failure in arm_reach_check instead of a hidden gap at the wrist); 'legacy': the old trust-region fitter with rate limit and smoothing. For new episodes prefer execute_subgoal_plan with couple_arms (the arm is planned with the hand)."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "per side: pos_err_mm {max, mean}, tilt_err_deg, roll_err_deg, branch_resets, worst_frame, limit_margin_deg; out_npz."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "arm", "ik", "episode"],
    "used_in": ["Open the Bottle", "Giftbox Packing", "Place Toy in Drawer", "Battery Insertion", "Place Cup in Bowl", "Computer Installation"],
    "status": "tested",
    "entrypoint": "fill_arm_joints",
}

import mujoco
import numpy as np

from tool_library.common import episode as E


def fill_arm_joints(scene, trajectory, out_npz, sides=None, max_arm_rate_deg_s=95.0, method="exact"):
    model = mujoco.MjModel.from_xml_path(str(scene))
    ep = E.load_episode(trajectory)
    if method == "legacy":
        Q, rep = E.fill_arms(model, ep["qpos"], sides=sides, fps=float(ep.get("fps", 30.0)), max_arm_rate_deg_s=float(max_arm_rate_deg_s))
    else:
        from tool_library.common import armhand as AH
        Q, fit = AH.fit_executed(model, ep["qpos"], sides=sides)
        fps = float(ep.get("fps", 30.0))
        parts = E.robot_parts(model)
        rep = {}
        for s_, v in fit.items():
            A = Q[:, parts[s_]["arm_qadr"]]
            sp = float(np.degrees(np.abs(np.diff(A, axis=0)).max()) * fps) if len(A) > 1 else 0.0
            rep[s_] = {"pos_err_mm": {"max": v["residual_mm_max"]}, "tilt_err_deg": {"max": v["residual_deg_max"]},
                       "frames_over_2mm_2deg": v["frames_over_2mm_2deg"], "worst_frame": v["worst_frame"], "peak_joint_speed_deg_s": round(sp, 1),
                       "mount": v["mount"]}
    extra = {k: ep[k] for k in ("subgoal", "joint_names") if k in ep}
    E.save_episode(out_npz, Q, ep["labels"], ep["fps"], **extra)
    rep["out_npz"] = str(out_npz); rep["n_frames"] = int(len(Q))
    return rep
