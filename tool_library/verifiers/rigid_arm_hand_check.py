"""A delivered hand is a fixed member of its arm, not a tracked floating body."""
from __future__ import annotations

TOOL = {
    "name": "rigid_arm_hand_check", "title": "Rigid arm hand assembly check", "type": "verifier",
    "category": "Episode Audits", "summary": "Require each delivered hand to be a fixed child of its arm flange with the trusted setup mount, and verify all recorded states.",
    "description": "Rejects floating wrists, missing hands, changed flange mounts, invalid state arrays and any measured separation. The trusted mount is derived from the setup home state, never from the delivered trajectory.",
    "inputs": [{"name": "scene", "type": "str", "description": "Actual rigid execution scene."},
               {"name": "trajectory", "type": "str", "description": "Actual recorded trajectory npz."},
               {"name": "setup", "type": "str | None", "description": "Trusted setup folder; defaults to repository setup."}],
    "outputs": [{"name": "report", "type": "dict", "description": "ok, issues, sides with maximum mount errors, frames_checked."}],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"], "simulators": ["MuJoCo"],
    "tags": ["rigid", "arm", "hand", "mount", "stage3"], "status": "tested", "entrypoint": "rigid_arm_hand_check",
}

from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def rigid_arm_hand_check(scene, trajectory, setup=None):
    from tool_library.common.rigid_robot import trusted_mounts
    model = mujoco.MjModel.from_xml_path(str(scene))
    trusted = trusted_mounts(**({"setup_scene": Path(setup) / "scene.xml"} if setup else {}))
    with np.load(str(trajectory), allow_pickle=False) as z:
        if "qpos" not in z.files or "fps" not in z.files:
            return {"ok": False, "issues": ["trajectory must contain qpos and fps"], "sides": {}, "frames_checked": 0}
        Q = np.asarray(z["qpos"], float)
        fps = float(z["fps"])
    if (Q.ndim != 2 or Q.shape[1] != model.nq or len(Q) < 2
            or not np.isfinite(Q).all() or not np.isfinite(fps) or fps <= 0):
        return {"ok": False, "issues": ["trajectory must have at least two finite states in the execution model's joint layout and positive fps"],
                "sides": {}, "frames_checked": 0}
    issues, sides = [], {}
    data = mujoco.MjData(model)
    for side in ("left", "right"):
        root_name = f"robot_{side}_{side}_base_yaw"
        flange_name = f"robot_{side}_arm_gripper"
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)
        flange = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, flange_name)
        if root < 0 or flange < 0 or side not in trusted:
            issues.append(f"{side}: hand, arm flange or trusted mount is missing")
            continue
        if int(model.body_parentid[root]) != flange or int(model.body_jntnum[root]) != 0:
            issues.append(f"{side}: hand must be a fixed child of its arm flange; floating wrists and soft welds cannot pass")
            continue
        old = [f"robot_{side}_{side}_{suffix}" for suffix in ("pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z")
               if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot_{side}_{side}_{suffix}") >= 0]
        if old:
            issues.append(f"{side}: independent wrist degrees of freedom remain: {old}")
        p, quat = trusted[side]
        R = Rotation.from_quat(np.roll(np.asarray(quat), -1)).as_matrix()
        pos_err, rot_err = [], []
        for q in Q:
            data.qpos[:] = q
            mujoco.mj_kinematics(model, data)
            F = data.xmat[flange].reshape(3, 3)
            H = data.xmat[root].reshape(3, 3)
            pos_err.append(float(np.linalg.norm(data.xpos[root] - data.xpos[flange] - F @ p)))
            rot_err.append(float(np.degrees(np.linalg.norm(Rotation.from_matrix((F @ R).T @ H).as_rotvec()))))
        sides[side] = {"max_position_error_m": max(pos_err), "max_rotation_error_deg": max(rot_err), "rigid": True}
        # MjSpec.to_xml writes quaternions to six significant digits. Its
        # bounded rounding error is about 1e-4 degrees; this checks calibration,
        # while the fixed parent/zero-joint requirement enforces no separation.
        if max(pos_err) > 1e-7 or max(rot_err) > 2e-4:
            issues.append(f"{side}: hand mount differs from trusted setup ({max(pos_err):.9g} m, {max(rot_err):.9g} deg)")
    return {"ok": not issues and len(sides) == 2, "issues": issues, "sides": sides, "frames_checked": len(Q)}
