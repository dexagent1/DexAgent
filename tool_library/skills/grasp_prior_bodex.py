TOOL = {
    "name": "grasp_prior_bodex",
    "title": "Grasp prior from BODex",
    "type": "skill",
    "category": "Grasp & Poses",
    "summary": "Run BODex force-closure grasp synthesis on the object's own mesh at its settled pose and translate the top seeds into wrist poses and per-joint finger targets for this hand rig.",
    "description": "BODex (cuRobo-based dexterous grasp synthesis) runs in its own environment as a subprocess, driven by a job file and returning an .npz of candidate grasps in three stages (pre, grasp, squeeze) with their force-closure error. The object is placed at its current world pose on a table whose top is z=0, so grasps from below are excluded by BODex's own collision world. Each of the best seeds is converted to this library's pose convention: BODex's base_link pose becomes the virtual-wrist pose through the fixed wrist-to-palm transform measured on the compiled hand at rest, and the joint values are mapped by name with a joint sign decided by comparing the URDF's limits with the MJCF ranges (a converted URDF can have every joint mirrored). The GRASP stage is the candidate (fingers at the surface, not pressed in); the squeeze stage rides along as finger_targets_squeezed. Use the result as a set of candidates for optimize_fingertip_contacts and the grasp verifiers, not as a final answer: an articulated target is not a free rigid body and BODex cannot model it.",
    "inputs": [
        {"name": "scene", "type": "dict", "description": "{hand_xml, side, objects, floor, grasp_body?}; the grasped object must be a mesh for BODex"},
        {"name": "mesh_path", "type": "path", "description": "metric mesh of the grasped object in its body frame (.obj/.stl)"},
        {"name": "bodex_dir", "type": "path", "description": "BODex checkout (github.com/JYChen18/BODex) with cuRobo assets installed"},
        {"name": "bodex_python", "type": "path", "description": "python of the BODex conda environment"},
        {"name": "hand", "type": "str or dict", "description": "'sharpa' or 'shadow', or {robot_yml, manip_dir, joint_prefix, base_body_suffix, joint_sign ('auto' | 1 | -1)} for another BODex robot config"},
        {"name": "n_seeds", "type": "int", "description": "random restarts (default 20)"},
        {"name": "top_k", "type": "int", "description": "candidates to return (default 8)"},
        {"name": "out_dir", "type": "path", "description": "work folder (job.json, raw BODex output, .npz)"},
        {"name": "timeout_s", "type": "int", "description": "subprocess timeout (default 900)"},
    ],
    "outputs": [
        {"name": "candidates", "type": "list of pose dicts", "description": "[{side, pos, euler, finger_targets, finger_targets_squeezed, grip_fingers, grasp_error, dist_error_mm}] best first"},
        {"name": "n_force_closure", "type": "int", "description": "seeds with grasp_error < 0.2"},
        {"name": "npz", "type": "path", "description": "the raw BODex export"},
    ],
    "status": "tested",
    "entrypoint": "grasp_prior_bodex",
    "requirements": ["mujoco>=3.1", "numpy", "BODex environment (torch, warp-lang, cuRobo, trimesh, pyyaml, coacd) with a GPU"],
    "simulators": ["MuJoCo"],
    "tags": ["grasp", "prior", "bodex", "force-closure", "external-model"],
    "used_in": ["Place Cup in Bowl"],
}

import json
import subprocess
from pathlib import Path

import numpy as np

# robot config inside BODex, the joint-name prefix BODex adds, the suffix of
# the body BODex calls base_link in this rig's naming, and the joint SIGN
# between BODex's URDF and the MJCF: "auto" compares the URDF's joint limits
# with the MJCF ranges (a URDF converted with every joint frame flipped has
# the mirrored limits [-hi, -lo], and replaying its raw values puts the
# fingertips 77-107 mm off with the back of the hand toward the object).
BODEX_HANDS = {
    ("sharpa", "right"): {"robot_yml": "sharpa_sim.yml", "manip_dir": "sim_sharpa", "joint_prefix": "right_",
                          "base_body_suffix": "hand_C_MC", "joint_sign": "auto"},
    ("sharpa", "left"): {"robot_yml": "sharpa_left_sim.yml", "manip_dir": "sim_sharpa_left", "joint_prefix": "left_",
                         "base_body_suffix": "hand_C_MC", "joint_sign": "auto"},
    ("shadow", "right"): {"robot_yml": "right_shadow_hand_sim.yml", "manip_dir": "sim_shadow", "joint_prefix": "rh_",
                          "base_body_suffix": "palm", "joint_sign": "auto"},
    ("shadow", "left"): {"robot_yml": "left_shadow_hand.yml", "manip_dir": "sim_shadow_left", "joint_prefix": "lh_",
                         "base_body_suffix": "palm", "joint_sign": "auto"},
}


def joint_sign_from_urdf(bodex_dir, robot_yml, joint_prefix, rig):
    """+1 when BODex's URDF limits match the MJCF ranges, -1 when they are
    the mirror image; decided by majority over the joints both share."""
    import xml.etree.ElementTree as et
    import yaml
    cfg = yaml.safe_load(open(Path(bodex_dir, "src/curobo/content/configs/robot", robot_yml)))
    kin = cfg["robot_cfg"]["kinematics"]
    urdf = Path(bodex_dir, "src/curobo/content/assets", kin["urdf_path"])
    votes = []
    for j in et.parse(urdf).getroot().iter("joint"):
        lim = j.find("limit")
        if lim is None:
            continue
        short = j.get("name", "")
        short = short[len(joint_prefix):] if short.startswith(joint_prefix) else short
        if short not in rig.finger_range:
            continue
        lo_u, hi_u = float(lim.get("lower", 0)), float(lim.get("upper", 0))
        lo, hi = rig.finger_range[short]
        if abs(hi - lo) < 1e-6 or abs(lo + hi) < 0.05:          # symmetric range: no vote
            continue
        same = abs(lo_u - lo) + abs(hi_u - hi)
        mirror = abs(lo_u + hi) + abs(hi_u + lo)
        votes.append(1.0 if same <= mirror else -1.0)
    return 1.0 if not votes or sum(votes) >= 0 else -1.0


def grasp_prior_bodex(scene, mesh_path, bodex_dir, bodex_python, hand="sharpa", n_seeds=20, top_k=8,
                      out_dir=None, timeout_s=900):
    import mujoco
    from tool_library.common import hand as H

    side = scene.get("side", "right")
    cfg = dict(hand) if isinstance(hand, dict) else BODEX_HANDS.get((hand, side))
    if cfg is None:
        raise ValueError(f"no BODex config for hand={hand!r} side={side!r}; pass a dict with "
                         "robot_yml, manip_dir, joint_prefix, base_body_suffix, joint_sign")
    if not Path(bodex_python).exists():
        raise FileNotFoundError(f"BODex python not found at {bodex_python}; create the BODex conda "
                                "environment (github.com/JYChen18/BODex) and pass its python")
    if not Path(bodex_dir, "example_grasp", "plan_batch_env.py").exists():
        raise FileNotFoundError(f"{bodex_dir} is not a BODex checkout (example_grasp/plan_batch_env.py missing)")
    if not Path(mesh_path).exists():
        raise FileNotFoundError(mesh_path)

    model, data0, rig, ids = H.open_scene(scene)
    bid = H.grasp_body(model, scene, ids)
    T_set = np.eye(4)
    T_set[:3, :3] = np.asarray(data0.xmat[bid], float).reshape(3, 3)
    T_set[:3, 3] = np.asarray(data0.xpos[bid], float)
    obj_id = (model.body(bid).name or "object").replace("/", "_")
    key = f"{cfg['manip_dir']}_{side}"
    out_dir = Path(out_dir or Path(mesh_path).parent / "bodex_prior")
    out_dir.mkdir(parents=True, exist_ok=True)
    job = {"bodex_dir": str(bodex_dir), "out_dir": str(out_dir), "seed_num": int(n_seeds),
           "objects": [{"obj_id": obj_id, "mesh": str(mesh_path), "T_settled": T_set.tolist()}],
           "hands": [{"key": key, "robot_yml": cfg["robot_yml"], "manip_dir": cfg["manip_dir"]}]}
    job_path = out_dir / "job.json"
    job_path.write_text(json.dumps(job, indent=1))
    npz = out_dir / f"{obj_id}__{key}.npz"
    if not npz.exists():
        worker = Path(__file__).resolve().parent.parent / "common" / "hand_bodex_worker.py"
        r = subprocess.run([str(bodex_python), str(worker), str(job_path)], cwd=str(bodex_dir),
                           capture_output=True, text=True, timeout=int(timeout_s))
        if not npz.exists():
            raise RuntimeError("BODex produced no grasps:\n" + "\n".join((r.stdout + r.stderr).splitlines()[-20:]))

    d = np.load(npz, allow_pickle=True)
    ge = np.asarray(d["grasp_error"], float)
    joint_names = [str(x) for x in d["joint_names"]]
    # fixed transform from this rig's wrist control frame to BODex's base
    # link, read off the compiled hand at rest (wrist qpos 0 = world frame)
    probe = mujoco.MjData(model)
    probe.qpos[:] = data0.qpos
    rig.teleport(probe, np.zeros(3), np.zeros(3))
    mujoco.mj_forward(model, probe)
    pb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{rig.prefix}{side}_{cfg['base_body_suffix']}")
    if pb < 0:
        raise ValueError(f"hand has no body ending in {cfg['base_body_suffix']!r} to serve as BODex's base_link")
    T_wp = np.eye(4)
    T_wp[:3, :3] = np.asarray(probe.xmat[pb], float).reshape(3, 3)
    T_wp[:3, 3] = np.asarray(probe.xpos[pb], float)
    n_stage = d["base_pose_rel"].shape[1]
    g = 1 if n_stage >= 3 else n_stage - 1
    jp = cfg["joint_prefix"]
    sign = (joint_sign_from_urdf(bodex_dir, cfg["robot_yml"], jp, rig)
            if cfg.get("joint_sign", "auto") == "auto" else float(cfg["joint_sign"]))

    def targets(si, stage):
        out = {}
        for jn, val in zip(joint_names, np.asarray(d["joints"][si, stage], float)):
            short = jn[len(jp):] if jn.startswith(jp) else jn
            if short in rig.finger_act:
                lo, hi = rig.finger_range[short]
                out[short] = float(np.clip(sign * float(val), lo, hi))
        return out

    cands = []
    for si in np.argsort(ge)[:int(top_k)]:
        si = int(si)
        T_palm = T_set @ np.asarray(d["base_pose_rel"][si, g], float)
        T_wrist = T_palm @ np.linalg.inv(T_wp)
        ft = targets(si, g)
        if not ft:
            continue
        cands.append({"side": side, "pos": T_wrist[:3, 3].tolist(), "euler": H.mat_to_euler(T_wrist[:3, :3]).tolist(),
                      "finger_targets": ft, "finger_targets_squeezed": targets(si, n_stage - 1),
                      "grip_fingers": list(rig.fingertip_sites), "grasp_error": float(ge[si]),
                      "dist_error_mm": float(d["dist_error"][si]) * 1000.0, "seed": si})
    return {"candidates": cands, "n_force_closure": int((ge < 0.2).sum()), "npz": str(npz),
            "n_seeds": int(len(ge)), "joint_sign": sign}
