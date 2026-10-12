"""Physics-informed retargeting iterations: annealed sampling in a GPU physics simulator refines a reference into hand
commands the objects follow under contact, written as a tracking reference the executor follows on the bench."""
from __future__ import annotations

TOOL = {
    "name": "physics_informed_retargeting",
    "title": "Physics-informed retargeting iterations",
    "type": "skill",
    "category": "Trajectory & Execution",
    "summary": "Stage 3 method module (the first after the agent): annealed sampling-based optimization in MuJoCo Warp on "
               "a GPU iterates hand commands along a reference (the agent's best physical attempt, an earlier method's "
               "tracking reference, or the demonstrator's hands) until the objects follow it under contact physics, with "
               "contact guidance; the result is checked on the bench's own rigid runtime and written as a tracking "
               "reference plus an execute_subgoal_plan plan of track_reference entries.",
    "description": (
        "Physics-informed retargeting (the sampling retargeting of Pan et al., 'SPIDER: Scalable Physics-Informed "
        "Dexterous Retargeting', 2025, run from its own checkout and environment; nothing of it is copied here): "
        "hundreds of control samples around the reference are rolled out in parallel in MuJoCo Warp, scored by how "
        "closely the object follows the reference, and annealed over iterations (DIAL-MPC), re-planned along the motion. "
        "Reference: `reference` (an earlier method's tracking reference), else the agent's best physical attempt in the "
        "workspace (tracking.reference_from_attempt: the execute_subgoal_plan folder whose executed entries cover the most "
        "Stage 1 subgoals, its commanded reference with one segment per executed entry); with source='human' the "
        "demonstrator's hands (EgoDex hands.npz, or HaMeR) mapped onto the Sharpa by matching anatomical hand frames, the "
        "objects following the plan or a given track, segmented by Stage 1's time windows. Preparation (sample-agnostic): "
        "the manipulated object grounded on what lies under it, fingertips within 2 cm of it in the interaction window put "
        "on its surface one pad radius out, the thumb re-derived at the antipode of the finger contacts (contact "
        "guidance). The optimizer runs at the bench's own 2 ms step with the bench's actuation limits (0.8 N m fingers, "
        "40 N / 3 N m wrist) and contact parameters; the sampling `profiles` are tried in order (refine: small noise "
        "around the reference, held to it; explore: larger) and each result's commands (the executed wrist plus the "
        "servo push re-expressed for the bench's wrist stiffness, the finger commands) are rolled out once on the bench's "
        "rigid runtime as the executor will run them (tracking.BenchTracker: home start, joint-rate ceilings, arms "
        "coupled); the profile whose objects stay closest to the reference there is written. Outputs: reference.npz "
        "(tracking reference), plan.json (track_reference entries with the segments' Stage 1 indices and expectations), "
        "report.json. The harness runs the plan with execute_subgoal_plan (a trusted execution), packages it with "
        "package_stage3_episode and runs finish's verifiers, exactly as for an agent's plan; the agent may call it the same "
        "way. Limits: one free rigid object per acting hand (articulated objects, cables and a second object in one hand "
        "are refused with the reason), the reference decides which contacts are tried, a GPU and the checkout "
        "(models.spider_dir, environments.spider) are needed; without them the call returns ok false with "
        "`unavailable` and the next method module runs."
    ),
    "inputs": [
        {"name": "workspace", "type": "str", "description": "Stage 3 workspace (the agent's attempts are read from it)."},
        {"name": "scene_semantic_json", "type": "str", "description": "Stage 1 scene_semantics.json."},
        {"name": "scene_recon_json", "type": "str", "description": "Stage 2 scene_recon.json."},
        {"name": "spider_dir", "type": "str | None", "description": "The optimizer's checkout (github.com/facebookresearch/spider); dexagent.yaml models.spider_dir."},
        {"name": "spider_python", "type": "str | None", "description": "Its environment's Python (mujoco, mujoco_warp, warp-lang, torch+CUDA, hydra-core); dexagent.yaml environments.spider."},
        {"name": "reference", "type": "str | None", "description": "A tracking reference to start from (an earlier method's); default the agent's best attempt in the workspace."},
        {"name": "source", "type": "str", "description": "'plan' (default: `reference` or the agent's attempt) or 'human' (the demonstrator's hands, `human`)."},
        {"name": "human", "type": "dict | None", "description": "source='human': {hands_npz (EgoDex, read_egodex_episode)} or {hamer_npz, camera_json, joint_regressor}; optional frames [first, last], fps (default 30)."},
        {"name": "object_poses", "type": "str | dict | None", "description": "source='human': 'plan' (default: the objects of the plan reference) or {body: [T, 7] world free-joint poses or an npz with qpos}."},
        {"name": "scene", "type": "str | None", "description": "Bench scene; default the stations scene of the workspace (place_arm_stations), else Stage 2's."},
        {"name": "setup_dir", "type": "str | None", "description": "Setup folder (hand MJCF); default the repository's setup/."},
        {"name": "out_dir", "type": "str | None", "description": "Default <workspace>/methods/physics_informed_retargeting."},
        {"name": "sides, objects", "type": "list[str] | dict | None", "description": "Override the acting hands and their objects ({side: body})."},
        {"name": "ik", "type": "str", "description": "'auto' (default: the reference's own joints for a robot reference, the optimizer's keypoint IK for the human's), 'direct' or 'keypoints'."},
        {"name": "profiles", "type": "list[str] | None", "description": "Sampling regimes tried in order (spider_bridge.MJWP_PROFILES); default ['refine', 'explore'] for a robot reference, ['retarget'] for the human's hands."},
        {"name": "mjwp", "type": "dict | None", "description": "Optimizer overrides on top of the profile (num_samples, max_num_iterations, horizon, ...)."},
        {"name": "device", "type": "str", "description": "CUDA device. Default 'cuda:0'."},
        {"name": "timeout_s", "type": "int", "description": "Per optimizer run. Default 7200."},
    ],
    "outputs": [
        {"name": "result", "type": "dict", "description": "ok, method, reference (tracking reference npz), plan (plan.json for execute_subgoal_plan), scene, fps, coverage {covered, missing Stage 1 indices}, bench_check (object tracking on the bench's runtime: mean / end error per object), profiles [{profile, optimizer object error, bench_check, error}], unavailable (true when the optimizer cannot run here), seconds, next_step."},
    ],
    "status": "tested",
    "entrypoint": "physics_informed_retargeting",
    "requirements": ["mujoco>=3.2", "numpy", "scipy", "trimesh", "the optimizer's checkout + its environment (torch with CUDA, warp-lang, mujoco_warp, hydra-core) and a GPU"],
    "simulators": ["MuJoCo", "MuJoCo Warp"],
    "tags": ["stage3", "method-module", "physics", "retargeting", "sampling", "external-model"],
    "used_in": ["Place Cup in Bowl"],
}

import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np

from tool_library.common import episode as E
from tool_library.common import spider_bridge as SB
from tool_library.common import tracking as TR

REPO = Path(__file__).resolve().parents[2]
NAME = TOOL["name"]


# --------------------------------------------------------------------------------------------- the reference --
def _object_track(model, obj_poses, body, T, fps_src, fps):
    v = obj_poses[body]
    if isinstance(v, (str, Path)):
        z = np.load(str(v), allow_pickle=True)
        v = z["qpos"] if "qpos" in z.files else z[z.files[0]]
    v = np.asarray(v, float)
    if v.ndim == 3 and v.shape[1:] == (4, 4):          # homogeneous matrices
        out = np.zeros((len(v), 7))
        for t, M in enumerate(v):
            out[t, :3] = M[:3, 3]; out[t, 3:] = SB._mat_quat(M[:3, :3])
        v = out
    t_src = np.arange(len(v)) / float(fps_src)
    return SB.resample(v, t_src, np.arange(T) / float(fps), quat_adr=[3])


def reference_from_human(model, human, sem, plan_ref=None, object_poses="plan", setup_dir=None, fps=30.0):
    """A bench reference from the demonstrator's hands: the wrist from the anatomical-frame match of each acting hand
    (Stage 1's hands.used), the fingers left for the optimizer's keypoint IK, the objects from the plan reference or a
    given track, and labels per Stage 1 time window ('<side>: <motion> <object>', reach before the hand's object first
    moves, retreat after it last moves). Returns the reference dict plus {human_kp: {side: {palm, tips, root}}}."""
    from tool_library.common import human_prior as HP
    setup_dir = Path(setup_dir or REPO / "setup")
    sides = [s for s in (sem.get("hands") or {}).get("used", []) if s in SB.SIDE_ORDER] or ["right"]
    frames = human.get("frames")
    src_fps = float(human.get("fps", fps))
    kp = {}
    for s in sides:
        if human.get("hands_npz"):
            with np.load(str(human["hands_npz"])) as z:
                n = len(z["rig_xyz"])
            fr = np.arange(frames[0], frames[1] + 1) if frames else np.arange(n)
            kp[s] = HP.keypoints_from_egodex(human["hands_npz"], s, fr)
        elif human.get("hamer_npz"):
            with np.load(str(human["hamer_npz"]), allow_pickle=True) as z:
                fidx = [int(v) for v in z["frame_idx"]]
            fr = [f for f in fidx if not frames or frames[0] <= f <= frames[1]]
            kp[s] = HP.keypoints_from_hamer(human["hamer_npz"], human["camera_json"], s, fr, human.get("joint_regressor"))
        else:
            raise ValueError("human needs hands_npz (EgoDex) or hamer_npz + camera_json + joint_regressor")
    T_src = len(next(iter(kp.values()))["wrist"])
    t_src = np.arange(T_src) / src_fps
    T = int(round(t_src[-1] * fps)) + 1
    for s in kp:                                      # to the bench's frame rate
        kp[s] = {k: SB.resample(np.asarray(v, float).reshape(T_src, -1), t_src, np.arange(T) / fps).reshape((T,) + np.asarray(v).shape[1:])
                 for k, v in kp[s].items()}
    hk = SB.human_keypoints(kp, {s: setup_dir / "robot" / f"sharpa_{s}.xml" for s in sides})
    d = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, d, 0)
    Q = np.tile(d.qpos.copy(), (T, 1))
    cols = SB.hand_columns(model)
    from tool_library.common import hand as H
    for s in sides:
        prev = Q[0, cols[s]["wrist"][3:]]
        for t in range(T):
            R = hk[s]["root"][t][:3, :3]
            e = H.nearest_euler_branch(H.mat_to_euler(R), prev); prev = e
            Q[t, cols[s]["wrist"][:3]] = hk[s]["root"][t][:3, 3]; Q[t, cols[s]["wrist"][3:]] = e
    free = E.object_free_joints(model)
    if object_poses == "plan":
        if plan_ref is None:
            raise ValueError("object_poses='plan' needs the plan reference; pass object_poses={body: [T, 7]} instead")
        P_ = plan_ref["qpos"]
        t_p = np.arange(len(P_)) / plan_ref["fps"]
        Pr = SB.resample(P_, t_p, np.linspace(0, t_p[-1], T), quat_adr=SB.quat_columns(model))
        for b, a in free.items():
            Q[:, a:a + 7] = Pr[:, a:a + 7]
    else:
        for b in object_poses:
            a = free[b]
            Q[:, a:a + 7] = _object_track(model, object_poses, b, T, human.get("object_fps", fps), fps)
    # labels: one Stage 1 subgoal per time window; each acting hand ('both' = every hand Stage 1 uses) is labelled with the
    # subgoal's object nearest its fingertips over the window, so a two-hand subgoal gets a two-hand label
    labels, sub = [""] * T, np.zeros(T, int)
    sgs = sem.get("subgoals") or []
    for t in range(T):
        ts = t / fps
        k = next((sg for sg in sgs if sg.get("time_window_s") and ts <= float(sg["time_window_s"][1])), sgs[-1] if sgs else None)
        sub[t] = int(k["index"]) if k else 1
    for k in sorted(set(sub.tolist())):
        sg = next((g for g in sgs if int(g["index"]) == k), {})
        win = np.where(sub == k)[0]
        hand = sg.get("acting_hand")
        acting = [s for s in sides if hand in (s, "both")] or sides[:1]
        refs = sg.get("object_refs") or []
        cands = [b for b in free if any(b == r or b.startswith(f"{r}_") for r in refs)] or list(free)
        parts = {}
        for s in acting:
            tips = hk[s]["tips"][win].mean(axis=1)                     # [n, 3] fingertip centroid per frame
            dist = {b: float(np.median(np.linalg.norm(Q[win, free[b]:free[b] + 3] - tips, axis=1))) for b in cands}
            body = min(dist, key=dist.get) if dist else "object"
            parts[s] = f"{s}: {sg.get('motion', 'move')} {body}"
        lab = E.combine_label(parts)
        for t in win:
            labels[t] = lab
    return {"qpos": Q, "labels": labels, "subgoal": sub, "fps": float(fps), "phases": [], "source": "human hands",
            "human_kp": hk}


def _human_tracking_reference(model, ref, sem, out_path):
    """The human reference as a tracking reference: one segment per Stage 1 subgoal it spans."""
    ks = sorted(set(int(k) for k in ref["subgoal"]))
    segs = []
    for k in ks:
        g = next((s for s in sem.get("subgoals", []) if int(s["index"]) == k), {})
        segs.append({"stage1_indices": [k], "motion": g.get("motion"),
                     "side": g.get("acting_hand") if g.get("acting_hand") in ("left", "right") else "both", "object": None})
    segment = np.array([ks.index(int(k)) for k in ref["subgoal"]], int)
    return TR.save_reference(out_path, model, ref["qpos"], ref["labels"], ref["fps"], segment, segs, source="human hands")


def portable_scene(src, dst):
    """A copy of the scene MJCF whose asset files are absolute paths, so it loads from any folder."""
    spec = mujoco.MjSpec.from_file(str(src))
    base = Path(src).resolve().parent
    for attr, items in (("meshdir", spec.meshes), ("texturedir", spec.textures)):
        d = base / (getattr(spec, attr) or "")
        for it in items:
            if it.file and not Path(it.file).is_absolute():
                it.file = str((d / it.file).resolve())
        setattr(spec, attr, "")
    for it in getattr(spec, "hfields", []):
        if it.file and not Path(it.file).is_absolute():
            it.file = str((base / it.file).resolve())
    Path(dst).write_text(spec.to_xml())
    return str(dst)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


def bench_check(scene, reference, work_dir, scene_recon_json=None):
    """One rollout of a tracking reference on the bench's rigid runtime as the executor will run it (no optimization):
    the objects' tracking errors. Diagnostic; the executor and finish's gates decide."""
    bt = TR.BenchTracker(scene, reference, work_dir, scene_recon_json=scene_recon_json)
    r = bt.sim.rollout(0, len(bt.sim.base) - 1, bt.sim.base)
    out = bt.sim.summary(r["qpos"])
    out.update(unstable=r["unstable"], arm_coupling=bt.couple_report)
    return out


# ------------------------------------------------------------------------------------------------- the skill --
def physics_informed_retargeting(workspace, scene_semantic_json, scene_recon_json, spider_dir=None, spider_python=None, reference=None,
                                 source="plan", human=None, object_poses="plan", scene=None, setup_dir=None, out_dir=None, sides=None,
                                 objects=None, ik="auto", profiles=None, mjwp=None, device="cuda:0", timeout_s=7200, ref_dt=0.02, fps=30.0):
    t_all = time.time()
    ws = Path(workspace).resolve()
    sem_p, recon_p = Path(scene_semantic_json).resolve(), Path(scene_recon_json).resolve()
    sem, recon = json.loads(sem_p.read_text()), json.loads(recon_p.read_text())
    setup_dir = Path(setup_dir or REPO / "setup").resolve()
    out = Path(out_dir or ws / "methods" / NAME).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rep = {"ok": False, "method": NAME, "out_dir": str(out)}

    def finish(**kw):
        rep.update(kw)
        rep["seconds"] = round(time.time() - t_all, 1)
        (out / "report.json").write_text(json.dumps(_jsonable(rep), indent=1, default=str))
        return _jsonable(rep)
    try:
        SB.check_spider(spider_dir, spider_python)
    except FileNotFoundError as e:
        return finish(unavailable=True, error=f"the GPU optimizer is not set up here: {e}")
    scene_p = TR.scene_for(ws, recon_p, scene)
    model = mujoco.MjModel.from_xml_path(str(scene_p))
    rep["scene"] = str(scene_p)

    # ---- the reference on the bench (a tracking reference, at the bench's frame rate)
    base_ref, info = TR.best_reference(ws, model, out / "reference_in.npz", given=reference, sem=sem)
    rep["reference_in"] = {"path": base_ref, **info}
    ref = TR.load_reference(base_ref, model)
    use_human = source == "human"
    if use_human:
        h = reference_from_human(model, human or {}, sem, plan_ref=ref, object_poses=object_poses, setup_dir=setup_dir, fps=fps)
        base_ref = _human_tracking_reference(model, h, sem, out / "reference_human.npz")
        hk = h["human_kp"]
        ref = TR.load_reference(base_ref, model)
    elif source != "plan":
        raise ValueError("source must be 'plan' or 'human'")
    if abs(ref["fps"] - fps) > 1e-6:
        raise ValueError(f"the reference runs at {ref['fps']} fps; the bench executes at {fps}")
    Qref, labels = ref["qpos"], ref["labels"]
    t_ref = np.arange(len(Qref)) / fps
    roles = SB.infer_roles(model, Qref, labels, sides=sides, objects=objects)
    rep["roles"] = roles

    # ---- the optimizer's inputs (its dataset folder: the bench's hand(s), scene, keypoints, kinematic trajectory)
    mjwp = {"sim_dt": float(model.opt.timestep), **(mjwp or {})}
    emb = SB.embodiment_of(roles["sides"])
    task = re.sub(r"[^A-Za-z0-9_]+", "_", ws.parent.name or "task") or "task"
    ds = out / "dataset"
    paths = SB.dataset_paths(ds, emb, task)
    robot_xml = SB.write_robot_xml(setup_dir, roles["sides"], paths["robot_xml"], sim_dt=mjwp["sim_dt"])
    cols = SB.hand_columns(model)
    free = E.object_free_joints(model)
    track = np.vstack([Qref[:, cols[s]["wrist"][:3]] for s in roles["sides"]] + [Qref[:, free[b]:free[b] + 3] for b in roles["objects"].values()])
    sc = SB.write_scene(model, Qref[0], roles, robot_xml, paths["robot_task"], paths["assets"] / "objects",
                        floor_z=0.0, track_bounds=(track.min(axis=0), track.max(axis=0)))
    sm = mujoco.MjModel.from_xml_path(sc["scene"])
    layout = sc["layout"]
    t_s = np.arange(0, t_ref[-1] + 1e-9, float(ref_dt))
    S_ref = SB.resample(SB.bench_to_spider(model, Qref, sm, layout), t_ref, t_s, SB.quat_columns(sm))
    use_ik = ik == "keypoints" or (ik == "auto" and use_human)
    if use_human:
        palm = {s: SB.resample(hk[s]["palm"], t_ref, t_s, quat_adr=[3]) for s in roles["sides"]}
        tips = {s: SB.resample(hk[s]["tips"].reshape(len(t_ref), -1), t_ref, t_s).reshape(len(t_s), 5, 3) for s in roles["sides"]}
        kp = SB.keypoints_npz(len(t_s), roles["sides"], palm=palm, tips=tips,
                              objects={s: S_ref[:, SB.spider_columns(sm, [s])[s]["object"]] for s in roles["sides"]})
    else:
        kp = SB.keypoints_from_spider_qpos(sm, S_ref, roles["sides"])
    kp, prep = SB.prep_keypoints(kp, sm, [s for s in roles["sides"] if s not in layout["dummy"]], 1.0 / ref_dt)
    rep["prep"] = prep
    dz = float(prep["dz"])
    paths["mano_trial"].mkdir(parents=True, exist_ok=True)
    np.savez(paths["mano_trial"] / "trajectory_keypoints.npz", **kp)
    info = {"ref_dt": float(ref_dt), "robot_type": SB.ROBOT, "source": ref["source"], "objects": roles["objects"], "dz": dz}
    for d_ in (paths["mano_task"], paths["robot_task"]):
        d_.mkdir(parents=True, exist_ok=True)
        (d_ / "task_info.json").write_text(json.dumps(info, indent=1))
    if use_ik:
        kin = Path(SB.run_spider_ik(spider_dir, spider_python, ds, emb, task, ref_dt, log=out / "keypoint_ik.log", timeout=timeout_s))
        z = dict(np.load(kin))
        n_short = len(t_s) - len(z["qpos"])          # the IK's moving average and first-frame drop: re-align to the keypoints
        lead = max(0, n_short - 1)
        S_kin = np.vstack([np.repeat(z["qpos"][:1], lead, 0), z["qpos"], np.repeat(z["qpos"][-1:], n_short - lead, 0)])
        for s in roles["sides"]:                     # the objects follow the (grounded) reference exactly
            oc = SB.spider_columns(sm, [s])[s]["object"]
            S_kin[:, oc] = kp[f"qpos_obj_{s}"]
    else:
        S_kin = S_ref.copy()
        for s in roles["sides"]:
            S_kin[:, SB.spider_columns(sm, [s])[s]["wrist"][2]] += dz
            S_kin[:, SB.spider_columns(sm, [s])[s]["object"][2]] += dz
    contact = np.concatenate([kp["contact_right"], kp["contact_left"]], axis=1)
    paths["trial"].mkdir(parents=True, exist_ok=True)
    np.savez(paths["trial"] / "trajectory_kinematic.npz", qpos=S_kin, qvel=SB.qvel_of(sm, S_kin, ref_dt), contact=contact,
             contact_pos=np.zeros((len(S_kin), 10, 3)), frequency=1.0 / ref_dt)

    # ---- the iterations, per sampling profile; each result checked on the bench's runtime
    if profiles is None:
        profiles = ["retarget"] if use_human else ["refine", "explore"]
    tried = []
    for profile in profiles:
        adir = out / "profiles" / profile
        adir.mkdir(parents=True, exist_ok=True)
        row = {"profile": profile}
        t0 = time.time()
        try:
            mj = SB.run_spider_mjwp(spider_dir, spider_python, ds, emb, task, overrides=mjwp, profile=profile, device=device,
                                    log=adir / "optimizer.log", timeout=timeout_s)
        except (SB.SpiderError, subprocess.TimeoutExpired) as e:
            row.update(error=f"{type(e).__name__}: {e}", seconds=round(time.time() - t0, 1)); tried.append(row)
            continue
        mj = shutil.copy(mj, adir / "trajectory_optimized.npz")
        res = SB.read_mjwp(mj, float(SB.mjwp_settings(profile, mjwp)["sim_dt"]))
        err = {}
        for s in roles["sides"]:
            if s in layout["dummy"]:
                continue
            oc = SB.spider_columns(sm, [s])[s]["object"]
            Pk = SB.resample(S_kin, t_s, res["t"], SB.quat_columns(sm))[:, oc[:3]]
            dev = np.linalg.norm(res["qpos"][:, oc[:3]] - Pk, axis=1)
            err[roles["objects"][s]] = {"mean_m": round(float(dev.mean()), 4), "max_m": round(float(dev.max()), 4)}
        Qcmd, _ = SB.bench_command_from_spider(model, Qref, sm, layout, res, t_ref, dz=dz)
        for b, a in free.items():                     # the targets stay the reference's; the commands are the optimizer's
            Qcmd[:, a:a + 7] = Qref[:, a:a + 7]
        cand = TR.save_reference(adir / "reference.npz", model, Qcmd, labels, fps, ref["segment"], ref["segments"],
                                 source=f"{NAME} ({profile}) from {ref['source']}")
        row.update(reference=cand, optimizer_object_error=err, seconds=round(time.time() - t0, 1))
        try:
            row["bench_check"] = bench_check(scene_p, cand, adir / "bench", scene_recon_json=recon_p)
        except Exception as e:                        # the executor will say; a check that cannot run is not a verdict
            row["bench_check"] = {"error": f"{type(e).__name__}: {e}"}
        tried.append(row)
        if row["bench_check"].get("worst_end_mm", 1e9) <= 10.0 and not row["bench_check"].get("unstable"):
            break
    rep["profiles"] = tried
    ran = [r for r in tried if r.get("reference")]
    if not ran:
        return finish(error="the optimizer produced no result: " + "; ".join(str(r.get("error"))[:300] for r in tried))
    best = min(ran, key=lambda r: (r["bench_check"].get("worst_mean_mm", 1e9), r["bench_check"].get("worst_end_mm", 1e9)))
    shutil.copy(best["reference"], out / "reference.npz")
    plan = TR.write_plan(out / "plan.json", out / "reference.npz", model, sem, recon, scene_recon_json=recon_p, method=NAME)
    return finish(ok=True, reference=str(out / "reference.npz"), plan=str(out / "plan.json"), scene=str(scene_p), fps=float(fps),
                  profile=best["profile"], bench_check=best["bench_check"], coverage=TR.coverage(out / "reference.npz", model, sem),
                  n_entries=len(plan["subgoals"]),
                  next_step=f"execute_subgoal_plan(scene={str(scene_p)!r}, plan={str(out / 'plan.json')!r}, out_dir={str(out / 'execution')!r}) "
                            "then package_stage3_episode and finish")
