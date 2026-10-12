"""Export an episode as a named, timestamped joint-space trajectory for policy training or a controller.

    from tool_library.skills.export_policy_trajectory import export_policy_trajectory
    out = export_policy_trajectory("scene.xml", "trajectory.npz", "episode.npz",
                                   groups={"right_arm": ["robot_right_arm_dof_joint1", ...],
                                           "right_hand": "subtree:robot_right_right_hand_C_MC"},
                                   end_effectors={"right": "robot_right_right_base_yaw"},
                                   objects=["cup"], target_fps=15)

The qpos array of a simulator is indexed by the scene, not by the robot: it
mixes objects, free joints and robot joints in compile order. A policy wants
the robot's joints by name, in a fixed order, at a fixed rate, with the end
effector poses, the object poses and the phase labels alongside. This
writes exactly that, resampled with linear interpolation (quaternions
re-normalized), in npz and json, plus a CSV when asked.
"""
from __future__ import annotations

TOOL = {
    "name": "export_policy_trajectory",
    "title": "Export policy trajectory",
    "type": "skill",
    "category": "Trajectory & Execution",
    "summary": "qpos episode -> named joint groups at a target rate with end-effector poses, object poses, labels and timestamps (npz/json/csv).",
    "description": (
        "Resolves each requested group to qpos columns: an explicit list of joint names, "
        "'subtree:<body>' for every scalar joint under a body (a hand), or 'chain:<body>' for "
        "the joints above a body (an arm). Resamples the trajectory from the source fps to "
        "target_fps by linear interpolation (free-joint quaternions re-normalized), computes "
        "end-effector world poses (position + quaternion wxyz + rotation matrix) and object "
        "free-joint poses by forward kinematics on the resampled frames, carries the phase "
        "label of the nearest source frame, and writes an npz with one array per group plus "
        "joint-name lists, a json manifest with the layout, and optionally a wide CSV. "
        "Also returns the arrays. Nothing depends on a specific robot."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF scene."},
        {"name": "trajectory", "type": "str | ndarray [T, nq]", "description": "trajectory.npz."},
        {"name": "out_path", "type": "str", "description": "Output .npz (a .json manifest is written beside it)."},
        {"name": "groups", "type": "dict name -> list[str] | 'subtree:<body>' | 'chain:<body>'", "description": "Joint groups to export, in order."},
        {"name": "end_effectors", "type": "dict side -> body | None", "description": "Bodies whose world pose is exported."},
        {"name": "objects", "type": "list[str] | None", "description": "Free bodies whose pose is exported."},
        {"name": "target_fps", "type": "float | None", "description": "Resample rate; default the source rate."},
        {"name": "csv", "type": "bool", "description": "Also write a wide CSV. Default False."},
    ],
    "outputs": [
        {"name": "result", "type": "dict", "description": "paths, n_frames, fps, per-group dimensionality, total action dimension, arrays (groups, ee poses, object poses, labels, t)."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["export", "policy", "dataset", "resample"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope", "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting", "Biology Experiment"],
    "status": "tested",
    "entrypoint": "export_policy_trajectory",
}

import json
from pathlib import Path

import numpy as np

from tool_library.common import trajectory as TJ


def _resolve(model, spec):
    if isinstance(spec, str) and spec.startswith("subtree:"):
        bid = TJ.body_id(model, spec.split(":", 1)[1])
        js = [j for j in range(model.njnt) if int(model.jnt_bodyid[j]) in TJ.subtree(model, bid)
              and int(model.jnt_type[j]) in (TJ.HINGE, TJ.SLIDE)]
    elif isinstance(spec, str) and spec.startswith("chain:"):
        js = TJ.chain_joints(model, TJ.body_id(model, spec.split(":", 1)[1]))
    else:
        js = [TJ.joint_id(model, n) for n in spec]
    return [model.joint(j).name for j in js], np.array([int(model.jnt_qposadr[j]) for j in js], int)


def resample_qpos(model, Q, fps, target_fps):
    T = len(Q)
    t_src = np.arange(T) / fps
    t_new = np.arange(0.0, t_src[-1] + 1e-9, 1.0 / target_fps)
    Qn = np.stack([np.interp(t_new, t_src, Q[:, k]) for k in range(Q.shape[1])], axis=1)
    for j in range(model.njnt):                 # keep quaternions unit length after lerp
        if int(model.jnt_type[j]) in (TJ.FREE, TJ.BALL):
            a = int(model.jnt_qposadr[j]) + (3 if int(model.jnt_type[j]) == TJ.FREE else 0)
            q = Qn[:, a:a + 4]
            Qn[:, a:a + 4] = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
    src_idx = np.clip(np.round(t_new * fps).astype(int), 0, T - 1)
    return Qn, t_new, src_idx


def export_policy_trajectory(scene, trajectory, out_path, groups, end_effectors=None, objects=None,
                             target_fps=None, csv=False):
    model = TJ.load_model(scene)
    Q, labels, fps = TJ.as_trajectory(trajectory)
    target_fps = float(target_fps or fps)
    Qn, t, src = resample_qpos(model, Q, fps, target_fps)
    arrays, manifest = {"t": t, "labels": np.array([labels[i] for i in src])}, {"fps": target_fps, "source_fps": fps,
                                                                                "n_frames": len(t), "groups": {}}
    dim = 0
    for g, spec in groups.items():
        names, qadr = _resolve(model, spec)
        arrays[g] = Qn[:, qadr]
        arrays[f"{g}_joint_names"] = np.array(names)
        manifest["groups"][g] = {"dim": len(names), "joints": names}
        dim += len(names)
    manifest["action_dim"] = dim
    if end_effectors:
        P, R = TJ.replay_poses(model, Qn, list(end_effectors.values()))
        for k, side in enumerate(end_effectors):
            arrays[f"ee_{side}_pos"] = P[:, k]
            arrays[f"ee_{side}_quat_wxyz"] = np.stack([TJ.mat2quat(r) for r in R[:, k]])
            arrays[f"ee_{side}_rot"] = R[:, k]
        manifest["end_effectors"] = end_effectors
    if objects:
        for name in objects:
            adr = TJ.free_joint_qadr(model, name)
            if adr is None:
                P, R = TJ.replay_poses(model, Qn, [name])
                arrays[f"obj_{name}_pos"] = P[:, 0]
                arrays[f"obj_{name}_quat_wxyz"] = np.stack([TJ.mat2quat(r) for r in R[:, 0]])
            else:
                arrays[f"obj_{name}_pos"] = Qn[:, adr:adr + 3]
                arrays[f"obj_{name}_quat_wxyz"] = Qn[:, adr + 3:adr + 7]
        manifest["objects"] = list(objects)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=1))
    paths = {"npz": str(out), "json": str(out.with_suffix(".json"))}
    if csv:
        cols, data = ["t", "label"], [t, arrays["labels"]]
        for g in groups:
            for i, n in enumerate(arrays[f"{g}_joint_names"]):
                cols.append(f"{g}/{n}"); data.append(arrays[g][:, i])
        for k in arrays:
            if k.startswith(("ee_", "obj_")) and arrays[k].ndim == 2:
                for i in range(arrays[k].shape[1]):
                    cols.append(f"{k}[{i}]"); data.append(arrays[k][:, i])
        with open(out.with_suffix(".csv"), "w") as f:
            f.write(",".join(cols) + "\n")
            for r in range(len(t)):
                f.write(",".join(str(d[r]) for d in data) + "\n")
        paths["csv"] = str(out.with_suffix(".csv"))
    return {"paths": paths, "n_frames": len(t), "fps": target_fps, "action_dim": dim,
            "groups": manifest["groups"], "arrays": arrays}
