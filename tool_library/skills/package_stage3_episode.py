"""Package one passing physical execution as the Stage 3 deliverable."""
from __future__ import annotations

TOOL = {
    "name": "package_stage3_episode",
    "title": "Package a Stage 3 episode",
    "type": "skill",
    "category": "Trajectory & Execution",
    "summary": "Turn one passing execute_subgoal_plan run into the Stage 3 deliverable: trajectory.npz and phases.json, the "
               "manifest built from the executor's own outcomes (one entry per executed segment, bound to the pose each "
               "segment actually used), grasp stress tests run on those poses with their evidence, the episode audits, "
               "stage4_task/ from write_stage4_config, sim_video.mp4; then the manifest check and the Stage 1 goal gate.",
    "description": (
        "Everything in the manifest comes from the execution, not from the caller: frames, motion, side, object, release, "
        "expectations and measured outcomes from the executor's report and outcomes.json, the plan entry each segment ran, "
        "and for every segment that holds a part in the fingers the executor's selected pose and the hand scene of its "
        "actual starting state, on which the stress tests are run (default grasp_stress_test; `stress_tests` maps a motion "
        "or a subgoal index to other library stress tests, e.g. a turn or slide test). The audits episode_physics_check, "
        "robot_self_collision_check and arm_reach_check run on the executed scene and trajectory, stage4_handoff_check on "
        "the written task folder. A failed test or audit is written as failed and reported; nothing is forced to pass. "
        "trajectory_manifest_check and stage1_goal_gate are run at the end and returned, so a result with ok true is ready "
        "for finish (which re-checks everything independently). Two-hand segments (motion 'both') are packaged per hand. "
        "Deformables simulated in Isaac Sim: `isaac_episode` is the execute_isaac_deformables run on this very execution "
        "(its report's sha256 binding must name this scene and trajectory); its recording, deformables.json, MuJoCo mirror "
        "and video are copied into the workspace and stage4_task/, the manifest gets a deformables block, and the goal gate "
        "runs on the mirror."
    ),
    "inputs": [
        {"name": "execution_report", "type": "str", "description": "execution_report.json of a passing execute_subgoal_plan run (in its out folder)."},
        {"name": "plan", "type": "str | dict", "description": "The plan given to that run (path or dict with `subgoals`)."},
        {"name": "scene_semantic_json", "type": "str", "description": "Stage 1 scene_semantics.json."},
        {"name": "scene_recon_json", "type": "str", "description": "Stage 2 scene_recon.json."},
        {"name": "human_video", "type": "str", "description": "The human demonstration video (for stage4_task)."},
        {"name": "workspace", "type": "str", "description": "The Stage 3 workspace to write the deliverable into."},
        {"name": "stress_tests", "type": "dict | None", "description": "Overrides: {motion or subgoal index: [stress test names]}. Default grasp_stress_test."},
        {"name": "num_samples", "type": "int", "description": "Stage 4 samples requested (write_stage4_config). Default 3."},
        {"name": "notes", "type": "list[str] | None", "description": "Free-text notes added to the manifest."},
        {"name": "isaac_episode", "type": "str | None", "description": "out_dir of execute_isaac_deformables run on this execution; required when Stage 2 has deformables simulated in Isaac Sim."},
    ],
    "outputs": [
        {"name": "result", "type": "dict", "description": "ok (manifest check and goal gate both pass), manifest path, per-entry stress test verdicts, "
                                                         "audits {name: ok}, manifest_check {ok, issues}, goal_gate {ok, issues}, issues."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo", "Isaac Sim"],
    "tags": ["stage3", "manifest", "package", "handoff"],
    "used_in": ["Place Cup in Bowl"],
    "status": "tested",
    "entrypoint": "package_stage3_episode",
}

import json
import shutil
from pathlib import Path

DEFAULT_TESTS = ["grasp_stress_test"]


def _load(p):
    return json.loads(Path(p).read_text())


def _call_inprocess(lib, mod, args):
    """The dispatch an agent's call goes through (the harness's agent.library.call_tool_inprocess); without the
    harness (the library used on its own) the same steps here: a .json path given for an argument the tool declares
    as a dict is loaded, the entrypoint runs, and the result is made JSON-safe."""
    try:
        from agent.library import call_tool_inprocess
    except ImportError:
        call_tool_inprocess = None
    if call_tool_inprocess is not None:
        return call_tool_inprocess(lib, mod.TOOL, args)
    types = {n.strip(): str(spec.get("type", "")) for spec in mod.TOOL.get("inputs") or []
             for n in str(spec.get("name", "")).replace("/", ",").split(",")}
    kw = {}
    for k, v in args.items():
        t = types.get(k, "")
        if isinstance(v, str) and v.endswith(".json") and "dict" in t and "str" not in t and Path(v).exists():
            v = json.loads(Path(v).read_text())
        kw[k] = v
    result = getattr(mod, mod.TOOL.get("entrypoint") or mod.TOOL["name"])(**kw)
    return json.loads(json.dumps(result, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def _stress(lib, ws, name, pose, scene, cache):
    """Run one library stress test the way the harness dispatch does, recording its evidence."""
    import importlib
    from tool_library.common.evidence import file_digest, record_tool_evidence, tool_inputs
    key = (name, file_digest(pose), file_digest(scene))
    if key in cache:
        return cache[key]
    mod = importlib.import_module(f"tool_library.verifiers.{name}")
    if mod.TOOL.get("category") != "Grasp Stress Tests":
        raise ValueError(f"{name} is not a grasp stress test")
    args = {"pose": str(pose), "scene": str(scene)}
    inputs = tool_inputs(args, ws)
    try:
        result = _call_inprocess(lib, mod, args)
        result = record_tool_evidence(lib, mod.TOOL, args, result, ws, inputs)
    except Exception as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    cache[key] = result
    return result


def _pose_source(plan_entry, side=None):
    p = (plan_entry or {}).get("pose")
    if isinstance(p, dict) and side:
        p = p.get(side)
    if isinstance(p, list) and p:
        p = p[0]
    try:
        rec = _load(p) if isinstance(p, str) else (p if isinstance(p, dict) else {})
    except (OSError, ValueError):
        rec = {}
    src = rec.get("source") or rec.get("tool") or (rec.get("meta") or {}).get("tool") if isinstance(rec, dict) else None
    return str(src or "plan pose, executed and re-anchored by execute_subgoal_plan (pose_used)")


ISAAC_FILES = ("isaac_episode_report.json", "deformables.json", "scene_mirror.xml", "trajectory_mirror.npz", "isaac/isaac_recording.npz",
               "isaac/isaac_events.json", "isaac/job.json", "isaac/isaac_episode.log")


def _isaac_episode(isaac_episode, scene, traj_src, scene_recon_json, issues):
    """The Isaac deformables run bound to this execution, or None when Stage 2 has no Isaac deformable."""
    from tool_library.common.evidence import file_digest
    recon = _load(scene_recon_json)
    cords = [o.get("id") for o in recon.get("objects", []) if o.get("type") == "deformable" and str(o.get("simulator", "")).lower() == "isaac"]
    if not cords:
        return None
    if not isaac_episode or not (Path(isaac_episode) / "isaac_episode_report.json").is_file():
        issues.append(f"the deformables {cords} are simulated in Isaac Sim: run execute_isaac_deformables on this execution "
                      f"(scene_xml={scene}, trajectory={traj_src}) and pass its out_dir as isaac_episode")
        return False
    src = Path(isaac_episode).resolve()
    rep = _load(src / "isaac_episode_report.json")
    b = rep.get("binding") or {}
    if b.get("scene_sha256") != file_digest(scene) or b.get("trajectory_sha256") != file_digest(traj_src):
        issues.append("isaac_episode was simulated from another scene or trajectory than this execution: run execute_isaac_deformables again")
        return False
    if not rep.get("ok"):
        issues.append(f"the Isaac deformables run failed its own checks: {rep.get('issues', [])[:3]}")
    return src


def _isaac_stage4(task_dir, scene_semantic_json, scene_recon_json):
    """The Stage 4 task of an episode with Isaac cords: each cord is a movable object of the layout draws (by its cord
    name), the hand phases that name it are anchored to it, its goals are judged on the mirror's cable bodies, and the
    plate step erases it from the human video."""
    from tool_library.common import isaac_cord as IC
    spec = _load(task_dir / "deformables.json")
    sem, recon = _load(scene_semantic_json), _load(scene_recon_json)
    names = {o.get("id"): o.get("name") for o in sem.get("objects", [])}
    ids = {c["name"]: next((o.get("id") for o in recon.get("objects", []) if o.get("simulator") == "isaac"
                            and Path(str((o.get("isaac") or {}).get("cord_json", ""))).name.endswith(f"_{c['name']}_cord.json")), c["name"])
           for c in spec["cords"]}
    cfg_p = task_dir / "stage4.json"
    cfg = _load(cfg_p)
    cfg["deformables"] = "deformables.json"
    objs = cfg.setdefault("randomize", {}).setdefault("objects", {})
    rng = next(iter(objs.values()), None) or {"dx": [-0.03, 0.03], "dy": [-0.03, 0.03], "dyaw_deg": [-10.0, 10.0]}
    for c in spec["cords"]:
        objs.setdefault(c["name"], {k: rng[k] for k in ("dx", "dy", "dyaw_deg") if k in rng})
    for side, segs in (cfg.get("segments") or {}).items():
        for seg in segs:
            text = " ".join(seg.get("phases") or []).lower()
            hit = [c["name"] for c in spec["cords"] if c["name"].lower() in text or str(ids[c["name"]]).lower() in text]
            if hit:
                seg.pop("blend", None)
                seg["anchor"] = hit[0]
    erase = cfg.setdefault("inpaint", {}).setdefault("erase_prompts", [])
    for c in spec["cords"]:
        nm = names.get(ids[c["name"]]) or ids[c["name"]]
        if nm and nm not in erase:
            erase.append(nm)
    cfg_p.write_text(json.dumps(cfg, indent=1))
    go_p = task_dir / "goal_objects.json"
    go = _load(go_p) if go_p.exists() else {"objects": []}
    go["objects"] = [o for o in go.get("objects", []) if o.get("id") not in ids.values()]
    for c in spec["cords"]:
        bodies = IC.body_names(c["name"], c["n_links"])
        go["objects"].append({"id": ids[c["name"]], "type": "deformable", "simulator": "isaac", "root_body": bodies[0], "bodies": bodies})
    go_p.write_text(json.dumps(go, indent=1))


def package_stage3_episode(execution_report, plan, scene_semantic_json, scene_recon_json, human_video, workspace,
                           stress_tests=None, num_samples=3, notes=None, isaac_episode=None):
    from tool_library.verifiers.trajectory_manifest_check import GRASP_CONTINUATIONS, GRASP_MOTIONS_TESTED
    lib = Path(__file__).resolve().parents[1]
    ws = Path(workspace).resolve()
    ws.mkdir(parents=True, exist_ok=True)
    rep_p = Path(execution_report).resolve()
    rep = _load(rep_p)
    issues = []
    if rep.get("ok") is not True or rep.get("dry_run"):
        return {"ok": False, "issues": ["the execution did not pass (or was a dry run): package a passing execute_subgoal_plan run"]}
    files = rep.get("files") or {}
    exec_dir = rep_p.parent
    scene = Path(files["scene_physics"]).resolve()
    traj_src, phases_src = Path(files["trajectory"]).resolve(), Path(files["phases"]).resolve()
    outcomes = _load(files["outcomes"])
    plan = _load(plan) if isinstance(plan, (str, Path)) else (plan or {})
    sem = _load(scene_semantic_json)
    stage1 = {int(s["index"]): s for s in sem.get("subgoals", [])}
    traj, phases = ws / "trajectory.npz", ws / "phases.json"
    isaac_src = _isaac_episode(isaac_episode, scene, traj_src, scene_recon_json, issues)
    if isaac_src is False:
        return {"ok": False, "issues": issues}
    mirror_assets = []
    if isaac_src is not None:                        # the mirror scene's meshes (relative, by content) travel with it
        mirror_assets = list((_load(isaac_src / "isaac_episode_report.json").get("mirror") or {}).get("assets") or [])
        for f in (*ISAAC_FILES, *mirror_assets):
            if (isaac_src / f).is_file() and (isaac_src / f).resolve() != (ws / f).resolve():
                (ws / f).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(isaac_src / f, ws / f)
    if traj_src != traj:
        shutil.copy2(traj_src, traj)
    if phases_src != phases:
        shutil.copy2(phases_src, phases)
    overrides = {str(k): v for k, v in (stress_tests or {}).items()}
    plan_rows = list(plan.get("subgoals") or [])
    cache, verdicts, entries = {}, {}, []

    def plan_entry(row):
        same = [p for p in plan_rows if p.get("index") == row.get("index") and p.get("step") == row.get("step")]
        same = same or [p for p in plan_rows if p.get("index") == row.get("index")]
        return same[0] if same else {}

    def tests_for(row):
        return overrides.get(str(row.get("index"))) or overrides.get(str(row.get("motion"))) or DEFAULT_TESTS

    def pose_block(row, motion, side, selected, pe):
        if not isinstance(selected, dict) or not selected.get("path"):
            return None
        hs = selected.get("hand_scene_used") or selected.get("hand_scene_path")
        pose = {"file": selected["path"], "hand_scene": hs, "source": _pose_source(pe, side)}
        if motion in GRASP_MOTIONS_TESTED or motion in GRASP_CONTINUATIONS:
            pose["stress_tests"] = {}
            for name in tests_for(row):
                r = _stress(lib, ws, name, selected["path"], hs, cache)
                pose["stress_tests"][name] = r
                verdicts[f"{row.get('index')}/{side}/{name}"] = bool(r.get("ok"))
                if not r.get("ok"):
                    issues.append(f"subgoal {row.get('index')} ({side}): {name} failed on the executed pose: "
                                  f"{str(r.get('first_fail') or r.get('error') or r.get('issues'))[:240]}")
        return pose

    for row, oc in zip(rep.get("subgoals") or [], outcomes):
        pe = plan_entry(row)
        idx = row.get("stage1_indices") or [row["index"]]
        s1 = stage1.get(int(idx[0]), {})
        entry = {"index": row["index"], "step": row.get("step"), "stage1_indices": idx, "frames": row["frames"],
                 "motion": row["motion"], "release": bool(row.get("release", False)),
                 "action": s1.get("action") or s1.get("goal"), "plan": pe, "expect": oc.get("expect"),
                 "outcome": {"ok": oc.get("ok"), "failed": oc.get("failed") or [], "checks": oc.get("checks"),
                             "measured": oc.get("measured")}}
        if entry["step"] is None:
            entry.pop("step")
        selected = row.get("selected_pose_evidence")
        if row.get("motion") == "both" or row.get("side") == "both":
            hands = {}
            for side, h in sorted((row.get("hands") or {}).items()):
                sub = {"object": (h or {}).get("object"), "motion": (h or {}).get("motion")}
                pose = pose_block({**row, "motion": sub["motion"]}, sub["motion"], side,
                                  (selected or {}).get(side) if isinstance(selected, dict) else None, pe)
                if pose:
                    sub["pose"] = pose
                hands[side] = sub
            entry["hands"] = hands
        else:
            entry["side"], entry["object"] = row["side"], row["object"]
            sel = selected.get(row["side"]) if isinstance(selected, dict) and row["side"] in selected else selected
            pose = pose_block(row, row["motion"], row["side"], sel, pe)
            if pose:
                entry["pose"] = pose
        entries.append(entry)

    # the episode audits on what actually ran
    audits = {}
    for name, call in (("episode_physics_check", lambda f: f(str(scene), str(traj))),
                       ("robot_self_collision_check", lambda f: f(str(scene), str(traj))),
                       ("arm_reach_check", lambda f: f(str(scene), str(traj)))):
        try:
            import importlib
            fn = getattr(importlib.import_module(f"tool_library.verifiers.{name}"), name)
            r = call(fn)
            audits[name] = {"ok": bool(r.get("ok")), "issues": [str(i)[:300] for i in (r.get("issues") or r.get("failed") or [])][:6]}
        except Exception as exc:
            audits[name] = {"ok": False, "issues": [f"{type(exc).__name__}: {exc}"]}
        if not audits[name]["ok"]:
            issues.append(f"{name}: {audits[name]['issues'][:2]}")
    # the simulation video through the video camera (Isaac Sim's render when the episode has deformables)
    sim = ws / "sim_video.mp4"
    if isaac_src is not None and (isaac_src / "sim_video.mp4").is_file():
        if (isaac_src / "sim_video.mp4").resolve() != sim.resolve():
            shutil.copy2(isaac_src / "sim_video.mp4", sim)
    else:
        try:
            from tool_library.skills.render_trajectory_video import render_trajectory_video
            out = render_trajectory_video(str(scene), str(traj), str(ws / ".sim_render"), cameras=["video_cam"])
            vids = [Path(v) for v in (out.get("videos") or {}).values()] if isinstance(out, dict) else []
            src = next((v for v in vids if v.name.startswith("video_cam")), None) or next(iter(sorted((ws / ".sim_render").glob("video_cam*.mp4"))), None)
            if src:
                shutil.copy2(src, sim)
        except Exception as exc:
            issues.append(f"sim video: {type(exc).__name__}: {exc}")
    # Stage 4 hand-off
    task_dir = ws / "stage4_task"
    try:
        from tool_library.skills.write_stage4_config import write_stage4_config
        write_stage4_config(str(scene_recon_json), str(exec_dir), str(task_dir), str(human_video),
                            sim_video=str(sim) if sim.exists() else None, scene_semantic_json=str(scene_semantic_json),
                            num_samples=int(num_samples), scene=str(scene), plan=plan or None)
        if isaac_src is not None:                    # the cords' replay inputs and the episode's recording travel with the task
            for f in (*ISAAC_FILES, *mirror_assets):
                if (ws / f).is_file() and not (task_dir / f).exists():
                    (task_dir / f).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(ws / f, task_dir / f)
            _isaac_stage4(task_dir, scene_semantic_json, scene_recon_json)
        from tool_library.verifiers.stage4_handoff_check import stage4_handoff_check
        h = stage4_handoff_check(str(task_dir))
        audits["stage4_handoff_check"] = {"ok": bool(h.get("ok")), "issues": [str(i)[:300] for i in (h.get("issues") or [])][:6]}
    except Exception as exc:
        audits["stage4_handoff_check"] = {"ok": False, "issues": [f"{type(exc).__name__}: {exc}"]}
    if not audits["stage4_handoff_check"]["ok"]:
        issues.append(f"stage4_handoff_check: {audits['stage4_handoff_check']['issues'][:2]}")
    stations = {"moved": False}
    if (ws / "place_arm_stations.json").exists():
        stations = {"tool": "place_arm_stations", "report": "place_arm_stations.json"}
    import numpy as np
    z = np.load(traj, allow_pickle=True)
    man = {"task_prompt": sem.get("task_prompt"), "scene_semantic": str(Path(scene_semantic_json).resolve()),
           "scene_recon": str(Path(scene_recon_json).resolve()), "scene_executed": str(scene), "stations": stations,
           "subgoals": entries,
           "episode": {"trajectory": str(traj), "phases": str(phases), "sim_video": str(sim), "fps": float(z["fps"]),
                       "n_frames": int(len(z["qpos"])), "execution_dir": str(exec_dir), "execution_report": str(rep_p)},
           "stage4": {"task_dir": str(task_dir), "config": str(task_dir / "stage4.json")},
           "checks": audits, "notes": list(notes or []) + ["packaged by package_stage3_episode from the executor's own records"]}
    if isaac_src is not None:
        man["deformables"] = {"simulator": "isaac", "report": "isaac_episode_report.json", "spec": "deformables.json",
                              "recording": "isaac/isaac_recording.npz", "scene_mirror": "scene_mirror.xml",
                              "trajectory_mirror": "trajectory_mirror.npz",
                              "cords": [c["name"] for c in _load(ws / "deformables.json")["cords"]]}
    mp = ws / "trajectory_manifest.json"
    mp.write_text(json.dumps(man, indent=1, default=str))
    from tool_library.verifiers.trajectory_manifest_check import trajectory_manifest_check
    from tool_library.verifiers.stage1_goal_gate import stage1_goal_gate
    mc = trajectory_manifest_check(str(mp), scene_semantic_json=str(scene_semantic_json), library_index=str(lib / "index.json"))
    gscene, gtraj = (ws / "scene_mirror.xml", ws / "trajectory_mirror.npz") if isaac_src is not None else (scene, traj)
    gg = stage1_goal_gate(str(scene_semantic_json), str(mp), str(gscene), str(gtraj), scene_recon_json=str(scene_recon_json))
    return {"ok": bool(mc.get("ok")) and bool(gg.get("ok")), "manifest": str(mp), "stress_tests": verdicts,
            "audits": {k: v["ok"] for k, v in audits.items()},
            "manifest_check": {"ok": mc.get("ok"), "issues": [str(i)[:300] for i in mc.get("issues", [])][:12]},
            "goal_gate": {"ok": gg.get("ok"), "issues": [str(i)[:300] for i in gg.get("issues", [])][:12]},
            "issues": issues[:20]}
