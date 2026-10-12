"""DexGraspNet as a pose source: differentiable force-closure grasp synthesis on the object's mesh."""
from __future__ import annotations

TOOL = {
    "name": "grasp_prior_dexgraspnet",
    "title": "Grasp prior from DexGraspNet",
    "type": "skill",
    "category": "Grasp & Poses",
    "summary": "Run DexGraspNet's annealed force-closure grasp synthesis (its URDF hand model with our dexterous hand) on the "
               "mesh of the object or part to grasp, and translate the best grasps into this library's pose convention "
               "(wrist pos/euler in the world, finger targets by joint name) as candidates for the stress tests.",
    "description": (
        "DexGraspNet (Wang et al.) optimizes hand poses with a differentiable energy: force closure over sampled contact "
        "points, contact distance, hand-object and self penetration, joint limits, by simulated annealing over a batch "
        "of random initializations around the object. This skill exports the grasp body's collision geometry (its "
        "subtree) to an OBJ, writes it as a DexGraspNet object (normalized to its bounding box with the metric scale "
        "recorded), runs the synthesis in DexGraspNet's own Python environment through a generated driver (the hand "
        "URDF, its contact and penetration point files, batch size, iterations), filters the results by the energy "
        "thresholds, and converts each grasp: the URDF root link pose in the metric object frame becomes the wrist "
        "pose in the world through the object's current pose, the joint values are mapped by their shared names onto "
        "the MJCF hand (with a fingertip-position comparison against MuJoCo's forward kinematics reported, so a mirrored "
        "joint is seen), and `finger_targets_squeezed` is the same closure driven 2 mm deeper by the fingertip IK. "
        "Candidates come sorted by energy. Use them as a prior: run grasp_lift_test / grasp_hold_test / grasp_turn_test "
        "on them in the hand scene before any is executed. Needs a GPU and a DexGraspNet checkout prepared for the hand "
        "(DexGraspNet/grasp_generation with utils/hand_model_urdf.py and mjcf/<hand>_contact_points.json)."
    ),
    "inputs": [
        {"name": "scene", "type": "dict", "description": "Hand scene dict (hand_scene_from_recon); the grasp_body's subtree is the object."},
        {"name": "dexgraspnet_dir", "type": "str", "description": "DexGraspNet checkout (contains grasp_generation/)."},
        {"name": "dexgraspnet_python", "type": "str", "description": "Python of the DexGraspNet environment (torch, pytorch3d, TorchSDF, pytorch_kinematics)."},
        {"name": "hand_urdf", "type": "str", "description": "URDF of the hand for pytorch_kinematics (its joint names must match the MJCF hand's, e.g. right_thumb_CMC_FE)."},
        {"name": "contact_points_json", "type": "str | None", "description": "DexGraspNet contact candidate file for the hand (default grasp_generation/mjcf/sharpa_contact_points.json)."},
        {"name": "penetration_points_json", "type": "str | None", "description": "Penetration keypoint file (default grasp_generation/mjcf/sharpa_penetration_points.json)."},
        {"name": "batch_size", "type": "int", "description": "Grasps optimized in parallel. Default 32."},
        {"name": "n_iter", "type": "int", "description": "Annealing iterations. Default 3000 (about 5 minutes on one GPU); 6000 for energies that pass DexGraspNet's own thresholds."},
        {"name": "top_k", "type": "int", "description": "Candidates returned. Default 8."},
        {"name": "n_contact", "type": "int", "description": "Contact points in the force-closure term. Default 4."},
        {"name": "object_mesh", "type": "str | None", "description": "Use this OBJ (in the grasp body's frame) instead of exporting the collision geometry."},
        {"name": "out_dir", "type": "str | None", "description": "Working folder (default next to the scene's bench scene)."},
        {"name": "timeout_s", "type": "int", "description": "Default 1800."},
        {"name": "seed", "type": "int", "description": "Default 1."},
        {"name": "coacd_threshold", "type": "float", "description": "CoACD concavity threshold for the convex decomposition of the object. Default 0.05."},
        {"name": "min_wrist_z", "type": "float", "description": "Candidates whose wrist is below this height are dropped (DexGraspNet has no table). Default 0.01."},
    ],
    "outputs": [
        {"name": "result", "type": "dict", "description": "candidates: [pose dicts with pos, euler, finger_targets, finger_targets_squeezed, grip_fingers, approach, energy terms, fk_mismatch_mm], n_passing, out_dir, log."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "trimesh", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "grasp", "dexgraspnet", "force closure", "prior", "gpu"],
    "used_in": ["Open the Bottle", "Place Cup in Bowl", "Multi-Cup Grasping", "Giftbox Packing"],
    "status": "ported",
    "entrypoint": "grasp_prior_dexgraspnet",
}

import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from tool_library.common import hand as H
from tool_library.common import meshexport as MX

DRIVER = r'''
import json, os, sys, math
job = json.load(open(sys.argv[1]))
os.chdir(os.path.join(job["dexgraspnet_dir"], "grasp_generation")); sys.path.insert(0, os.getcwd())
import numpy as np, torch, transforms3d
from utils.hand_model_urdf import HandModelURDF
from utils.object_model import ObjectModel
from utils.initializations import initialize_convex_hull
from utils.energy import cal_energy
from utils.optimizer import Annealing
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d
torch.manual_seed(job["seed"]); np.random.seed(job["seed"])
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hand = HandModelURDF(urdf_path=job["hand_urdf"], contact_points_path=job["contact_points_json"], penetration_points_path=job["penetration_points_json"], device=dev, spen_dist=0.015)
A = np.stack([[1, 0, 0], [0, 0, 1], [0, 1, 0]], axis=1).astype(float)
c, s = math.cos(-math.pi / 3), math.sin(-math.pi / 3)
B = np.stack([[0, -c, -s], [0, -s, c], [1, 0, 0]], axis=1).astype(float)
hand.rotation_hand_np = B @ np.linalg.inv(A)
MU = {"CMC_FE": 1.2, "CMC_AA": 0.0, "MCP_FE": 0.2, "MCP_AA": 0.0, "IP": 0.2, "PIP": 0.6, "DIP": 0.0, "CMC": 0.1}
def mu_for(n):
    for suf in sorted(MU, key=len, reverse=True):
        if n.endswith(suf): return MU[suf]
    return 0.0
hand.joint_angles_mu = torch.tensor([mu_for(n) for n in hand.joints_names], dtype=torch.float, device=dev)
obj = ObjectModel(data_root_path=job["meshdata"], batch_size_each=job["batch_size"], num_samples=2000, device=dev)
obj.initialize([job["code"]])
obj.object_scale_tensor = torch.tensor([job["dmax"]], dtype=torch.float, device=dev).unsqueeze(1).repeat(1, job["batch_size"])
class A_: pass
args = A_()
for k, v in dict(n_contact=job["n_contact"], jitter_strength=0.1, distance_lower=0.2, distance_upper=0.3, theta_lower=-math.pi/6, theta_upper=math.pi/6).items(): setattr(args, k, v)
initialize_convex_hull(hand, obj, args)
hand_pose_st = hand.hand_pose.detach().clone()
opt = Annealing(hand, switch_possibility=0.5, starting_temperature=18, temperature_decay=0.95, annealing_period=30, step_size=0.005, stepsize_period=50, mu=0.98, device=dev)
W = dict(w_dis=100.0, w_pen=100.0, w_spen=10.0, w_joints=1.0)
energy, E_fc, E_dis, E_pen, E_spen, E_joints = cal_energy(hand, obj, verbose=True, **W)
energy.sum().backward(retain_graph=True)
for step in range(1, job["n_iter"] + 1):
    opt.try_step(); opt.zero_grad()
    ne, nfc, ndis, npen, nspen, nj = cal_energy(hand, obj, verbose=True, **W)
    ne.sum().backward(retain_graph=True)
    with torch.no_grad():
        accept, t = opt.accept_step(energy, ne)
        energy[accept] = ne[accept]; E_fc[accept] = nfc[accept]; E_dis[accept] = ndis[accept]; E_pen[accept] = npen[accept]; E_spen[accept] = nspen[accept]; E_joints[accept] = nj[accept]
    if step % 500 == 0: print("iter", step, "best energy", float(energy.min()), flush=True)
names = list(hand.joints_names); out = []
for i in range(job["batch_size"]):
    hp = hand.hand_pose[i].detach().cpu()
    rot = robust_compute_rotation_matrix_from_ortho6d(hp[3:9].unsqueeze(0))[0].numpy()
    out.append({"translation": hp[:3].tolist(), "rotation": rot.tolist(), "qpos": dict(zip(names, hp[9:].tolist())),
                "energy": float(energy[i]), "E_fc": float(E_fc[i]), "E_dis": float(E_dis[i]), "E_pen": float(E_pen[i]), "E_spen": float(E_spen[i]), "E_joints": float(E_joints[i])})
json.dump({"grasps": out, "joint_names": names, "root_link": hand.chain.get_joint_parent_frame_names()[0] if hasattr(hand.chain, "get_joint_parent_frame_names") else None}, open(job["out_json"], "w"))
print("wrote", job["out_json"])
'''


def grasp_prior_dexgraspnet(scene, dexgraspnet_dir, dexgraspnet_python, hand_urdf, contact_points_json=None, penetration_points_json=None,
                            batch_size=32, n_iter=3000, top_k=8, n_contact=4, object_mesh=None, out_dir=None, timeout_s=1800, seed=1,
                            coacd_threshold=0.05, min_wrist_z=0.01):
    dg = Path(dexgraspnet_dir).resolve()
    gen = dg / "grasp_generation"
    if not (gen / "utils" / "hand_model_urdf.py").exists():
        raise FileNotFoundError(f"{gen}/utils/hand_model_urdf.py missing: a DexGraspNet checkout prepared for a URDF hand is needed")
    model, data0, rig, ids = H.open_scene(scene)
    bid = H.grasp_body(model, scene, ids)
    out = Path(out_dir or Path(scene.get("bench_scene", ".")).parent / "dexgraspnet_prior").resolve(); out.mkdir(parents=True, exist_ok=True)
    code = f"obj_{model.body(bid).name}"
    mdir = out / "meshdata" / code / "coacd"; mdir.mkdir(parents=True, exist_ok=True)
    import trimesh
    mesh = trimesh.load(object_mesh, force="mesh") if object_mesh else MX.body_mesh(model, data0, bid, collision_only=True, frame="body")
    lo, hi = mesh.bounds; centre = 0.5 * (lo + hi); dmax = float(np.max(np.linalg.norm(mesh.vertices - centre, axis=1)) * 1.03)
    norm = mesh.copy(); norm.apply_translation(-centre); norm.apply_scale(1.0 / dmax)
    # DexGraspNet's distance and penetration terms want a convex decomposition (its own data is CoACD output); a
    # concatenation of primitives has inside-out faces where the pieces overlap and the energy never settles
    try:
        import coacd
        try:
            coacd.set_log_level("error")
        except Exception:
            pass
        cm = coacd.Mesh(np.asarray(norm.vertices, np.float64), np.asarray(norm.faces, np.int64))
        parts = coacd.run_coacd(cm, threshold=float(coacd_threshold), max_convex_hull=48, preprocess_mode="auto")
        hulls = [trimesh.Trimesh(np.asarray(v), np.asarray(f), process=False) for v, f in parts]
        dec = trimesh.util.concatenate(hulls)
        n_hulls = len(hulls)
    except Exception:                            # pragma: no cover - coacd missing: fall back to the raw union
        dec, n_hulls = norm, 0
    dec.export(str(mdir / "decomposed.obj"))
    (mdir.parent / "scale.json").write_text(json.dumps({"dmax": dmax, "centre_in_body": centre.tolist()}))
    job = {"dexgraspnet_dir": str(dg), "hand_urdf": str(Path(hand_urdf).resolve()), "meshdata": str(out / "meshdata"), "code": code, "dmax": dmax,
           "contact_points_json": str(contact_points_json or gen / "mjcf" / "sharpa_contact_points.json"),
           "penetration_points_json": str(penetration_points_json or gen / "mjcf" / "sharpa_penetration_points.json"),
           "batch_size": int(batch_size), "n_iter": int(n_iter), "n_contact": int(n_contact), "seed": int(seed), "out_json": str(out / "grasps_raw.json")}
    (out / "job.json").write_text(json.dumps(job, indent=1)); (out / "driver.py").write_text(DRIVER)
    raw = out / "grasps_raw.json"
    log = ""
    if not raw.exists():
        r = subprocess.run([str(dexgraspnet_python), str(out / "driver.py"), str(out / "job.json")], capture_output=True, text=True, timeout=int(timeout_s), cwd=str(gen))
        log = (r.stdout + r.stderr)[-4000:]
        (out / "driver.log").write_text(r.stdout + r.stderr)
        if not raw.exists():
            raise RuntimeError(f"DexGraspNet produced no grasps; log tail:\n{log}")
    res = json.loads(raw.read_text())
    # object pose in the hand scene (world): the grasp body
    mujoco.mj_forward(model, data0)
    p_obj, R_obj = data0.xpos[bid].copy(), data0.xmat[bid].reshape(3, 3).copy()
    # URDF root link = the palm body of the MJCF hand; the wrist frame sits on the palm with a fixed offset
    palm = rig.palm_body
    d = mujoco.MjData(model); d.qpos[:] = data0.qpos; rig.teleport(d, np.zeros(3), np.zeros(3)); mujoco.mj_forward(model, d)
    p_palm0, R_palm0 = d.xpos[palm].copy(), d.xmat[palm].reshape(3, 3).copy()      # palm pose when the wrist is at the identity
    grasps = sorted(res["grasps"], key=lambda g: g["energy"])
    passing = [g for g in grasps if g["E_fc"] < 0.3 and g["E_dis"] < 0.005 and g["E_pen"] < 0.001]
    cands = []
    n_below = 0
    for g in (passing or grasps):
        if len(cands) >= int(top_k):
            break
        t = np.asarray(g["translation"], float); Rr = np.asarray(g["rotation"], float)
        # metric object frame (centred at the bbox centre) -> world
        p_root = p_obj + R_obj @ (centre + t); R_root = R_obj @ Rr
        # wrist pose from the palm pose: T_wrist = T_root * T_palm0^-1
        R_w = R_root @ R_palm0.T; p_w = p_root - R_w @ p_palm0
        if p_w[2] < float(min_wrist_z):             # DexGraspNet has no table: grasps from below are dropped here
            n_below += 1
            continue
        closed = {}
        for jn, v in g["qpos"].items():
            short = jn.split(f"{rig.side}_", 1)[-1] if f"{rig.side}_" in jn else jn
            if short in rig.finger_range:
                lo_, hi_ = rig.finger_range[short]; closed[short] = float(np.clip(v, lo_, hi_))
        pose = {"side": rig.side, "pos": p_w.tolist(), "euler": H.mat_to_euler(R_w).tolist(), "finger_targets": closed}
        # which pads touch: cinch bookkeeping by distance of each tip to the object
        d.qpos[:] = data0.qpos; H.apply_pose(model, d, rig, pose, closed); mujoco.mj_forward(model, d)
        obj_g = sorted(ids["grasp_gids"]); touching = []
        for f in rig.fingertip_sites:
            gap = min(mujoco.mj_geomDistance(model, d, tg, og, 0.05, None) for tg in rig.tip_geoms(f) for og in obj_g)
            if gap < 0.004:
                touching.append(f)
        # squeeze: pads 2 mm deeper along their current contact normals, wrist fixed
        tips = {f: d.site_xpos[rig.fingertip_sites[f]].copy() for f in touching}
        c_obj = d.xpos[bid].copy()
        tgt = {f: p + 0.002 * (c_obj - p) / max(np.linalg.norm(c_obj - p), 1e-6) for f, p in tips.items()}
        if tgt:
            H.finger_ik(model, d, rig, tgt)
        squeezed = {k: float(d.qpos[a]) for k, a in rig.finger_qpos.items()}
        approach = (p_w - p_obj); approach[2] = max(approach[2], 0.0); approach = approach / max(np.linalg.norm(approach), 1e-9)
        pose.update({"finger_targets_squeezed": squeezed, "grip_fingers": touching, "approach": approach.tolist(), "press_depth_rad": 0.15,
                     "source": "dexgraspnet", "energy": {k: round(g[k], 4) for k in ("energy", "E_fc", "E_dis", "E_pen", "E_spen", "E_joints")},
                     "n_touching": len(touching)})
        cands.append(pose)
    # forward-kinematics agreement: the URDF's fingertip spread vs MuJoCo's for the first candidate (a mirrored joint shows here)
    fk = None
    if cands:
        try:
            fk = _fk_mismatch(model, d, data0, rig, cands[0], Path(hand_urdf), dexgraspnet_python)
        except Exception as e:                      # pragma: no cover
            fk = {"error": str(e)}
    (out / "candidates.json").write_text(json.dumps(cands, indent=1, default=float))
    return {"candidates": cands, "n_passing": len(passing), "n_total": len(grasps), "n_below_table": n_below, "convex_hulls": n_hulls,
            "out_dir": str(out), "object_code": code, "dmax_m": round(dmax, 4), "fk_check": fk, "log": log[-800:]}


def _fk_mismatch(model, d, data0, rig, pose, urdf, python):
    """Compare fingertip-to-palm distances between MuJoCo (this hand) and pytorch_kinematics (the URDF) for one joint
    vector: a sign-flipped joint shows as centimetres of mismatch."""
    import subprocess, json, tempfile, os
    d.qpos[:] = data0.qpos; H.apply_pose(model, d, rig, pose, pose["finger_targets"]); mujoco.mj_forward(model, d)
    palm = d.xpos[rig.palm_body].copy()
    mj = {f: float(np.linalg.norm(d.site_xpos[s] - palm)) for f, s in rig.fingertip_sites.items()}
    script = r'''
import json, sys, numpy as np, torch, pytorch_kinematics as pk
j = json.load(open(sys.argv[1])); chain = pk.build_chain_from_urdf(open(j["urdf"], "rb").read())
names = chain.get_joint_parameter_names(); q = torch.tensor([[j["qpos"].get(n, 0.0) for n in names]], dtype=torch.float)
fk = chain.forward_kinematics(q); out = {}
for link, tf in fk.items():
    out[link] = tf.get_matrix()[0, :3, 3].tolist()
print(json.dumps(out))
'''
    tmp = Path(tempfile.mkdtemp(dir=os.environ.get("TMPDIR")))
    (tmp / "fk.py").write_text(script)
    qpos_urdf = {f"{rig.side}_{k}": v for k, v in pose["finger_targets"].items()}
    (tmp / "j.json").write_text(json.dumps({"urdf": str(urdf), "qpos": qpos_urdf}))
    r = subprocess.run([str(python), str(tmp / "fk.py"), str(tmp / "j.json")], capture_output=True, text=True, timeout=300)
    links = json.loads(r.stdout.strip().splitlines()[-1])
    root = next((k for k in links if k.endswith("hand_C_MC")), None)
    out = {}
    for f in mj:
        tip = next((k for k in links if f"_{f}_" in k and k.endswith("DP")), None)
        if root and tip:
            out[f] = round(1000 * abs(float(np.linalg.norm(np.asarray(links[tip]) - np.asarray(links[root]))) - mj[f]), 1)
    return {"tip_to_palm_mismatch_mm": out, "note": "the URDF tip is the DP link origin, MuJoCo's the tip site: a constant 1-2 cm is the pad length; a finger far off the others is a mirrored joint"}
