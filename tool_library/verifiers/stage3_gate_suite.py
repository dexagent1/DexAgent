"""Every check a delivered Stage 3 episode must pass, in one call: the harness runs it at `finish`.

    from tool_library.verifiers.stage3_gate_suite import stage3_gate_suite
    r = stage3_gate_suite("runs/x/stage3", "runs/x/stage1/scene_semantics.json", "runs/x/stage2/scene_recon.json")
    r["ok"], r["issues"]

Nothing the agent wrote about its own results is trusted: the suite re-measures the delivered trajectory in the scene
it was executed in. It rejects low-level errors (penetration, the hand off its arm, teleports, joints past the real
limits, objects that move without contact, a start the scene cannot hold), goals missed against the Stage 1 sentence
(not the agent's own expectation), a robot or physics changed to make a step easier, and a hand-off Stage 4 cannot
randomize.
"""
from __future__ import annotations

TOOL = {
    "name": "stage3_gate_suite",
    "title": "Stage 3 gate suite",
    "type": "verifier",
    "category": "Task & Rollout Gates",
    "summary": "Re-measures a delivered Stage 3 episode with every episode audit, goal gate, anti-tampering scene check and "
               "the Stage 4 hand-off (with randomization draws) and returns one verdict; incomplete diagnostic runs cannot pass.",
    "description": (
        "Reads trajectory_manifest.json in the workspace for the trajectory, the scene it was executed in "
        "(`scene_executed`, else the Stage 2 scene) and the subgoal entries. Groups: (1) episode physics: "
        "episode_physics_check, robot_self_collision_check, arm_reach_check and the AUDITS (teleports, joint jumps, silent "
        "resets, swept collisions, visual-mesh penetration, dynamics consistency, contactless motion, joint limits of the "
        "setup, start and end states, release and slip, labels, ...); (2) goals: subgoal_outcome_check on every executed "
        "entry, stage1_goal_gate (each subgoal against its Stage 1 goal sentence, so a weak `expect` cannot pass a missed "
        "goal) and bimanual_sync_gate; (3) the executed scene against the setup: robot_model_integrity_check, "
        "robot_joint_range_check, actuator_limit_integrity_check, contact_exclusion_audit, physics_options_check, "
        "stage_station_integrity_check (the supplied setup is fixed), initial_keyframe_state_check; "
        "(4) Stage 4: stage4_handoff_check and, with draws > 0, stage4_randomization_gate (two thirds of the randomized "
        "layouts, 2 of 3, must pass Stage 4's own checks). Deformables simulated in Isaac Sim (the manifest's deformables "
        "block from execute_isaac_deformables): the goal gates and the audits that read object states run on the "
        "episode's MuJoCo mirror (the executed scene plus each cord's recorded links), the checks that re-simulate "
        "MuJoCo physics on the executed recording, and isaac_recording_check (binding, PhysX, the cord's own physics, an "
        "Isaac re-simulation) is added. Quality audits block final acceptance. Disabling quality, "
        "omitting randomization, skipping a check, or a missing/crashing check cannot produce a final pass."
    ),
    "inputs": [
        {"name": "workspace", "type": "str", "description": "Stage 3 workspace with trajectory_manifest.json."},
        {"name": "scene_semantic_json", "type": "str", "description": "Stage 1 deliverable."},
        {"name": "scene_recon_json", "type": "str", "description": "Stage 2 scene_recon.json (the scene, camera.json next to it)."},
        {"name": "setup_dir", "type": "str | None", "description": "Setup folder the robot and physics must match; default the repository's setup/."},
        {"name": "draws", "type": "int", "description": "Stage 4 randomization draws to try. Default 3; zero is diagnostic only and cannot pass."},
        {"name": "strict_quality", "type": "bool", "description": "Default True. False is diagnostic only and cannot pass."},
        {"name": "skip", "type": "list[str] | None", "description": "Diagnostic check omissions; any requested omission prevents final acceptance."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok, issues (prefixed by check name), warnings, reports {check: report}, skipped, seconds {check: s}, scene, trajectory."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo", "Isaac Sim"],
    "tags": ["stage3", "finish", "gate", "suite", "audit", "anti-tampering"],
    "status": "tested",
    "entrypoint": "stage3_gate_suite",
}

import importlib
import json
import time
import uuid
from pathlib import Path

# whole-episode audits: name -> blocking (False = quality, a warning unless strict_quality)
AUDITS = {
    "object_teleport_audit": True, "robot_joint_jump_audit": True, "badqacc_reset_audit": True, "arm_branch_flip_audit": True,
    "swept_collision_audit": True, "object_visual_penetration_audit": True, "dynamics_consistency_audit": True,
    "contactless_motion_audit": True, "free_fall_plausibility_audit": True, "object_spin_rate_audit": True,
    "contact_impulse_spike_audit": True, "finger_servo_saturation_audit": True, "hand_object_relative_jitter_audit": True,
    "joint_limit_margin_audit": True, "joint_acceleration_audit": True, "hand_joint_rate_audit": True, "start_state_audit": True,
    "end_state_rest_audit": True, "released_contact_audit": True, "held_object_slip_audit": True,
    "phase_label_continuity_audit": True, "table_workspace_bounds_audit": True, "wrist_singularity_audit": True,
    "deformable_integrity_audit": True,
    "timing_realism_audit": False, "camera_visibility_audit": False, "redundant_motion_audit": False,
}


# what an entry does to the objects its hand holds: takes one (replacing what it held), takes one more with free fingers
# (grasp_more), keeps them (carry), or puts one down
_TAKES = ("grasp_lift", "hold", "regrasp", "lift_together")
_TAKES_MORE = ("grasp_more",)
_KEEPS = ("carry", "carry_together")
_PUTS_DOWN = ("place", "insert", "release", "drop", "put_down", "lay", "drag")   # a rope lay/drag ends with the hand open


def _held_at_end(man):
    """Bodies an episode legitimately ends holding, followed hand by hand through the executed entries: an entry that
    grasps, holds or regrasps an object makes it what that hand holds, grasp_more adds a second object to the same hand,
    a carry keeps what the hand holds, and a place (or any entry with release: true) puts its object down."""
    held = {}
    for sg in man.get("subgoals", []):
        if sg.get("skipped"):
            continue
        hands = sg.get("hands") if isinstance(sg.get("hands"), dict) else None
        subs = [(s_, e or {}) for s_, e in hands.items()] if hands else [(sg.get("side") or sg.get("acting_hand"), sg)]
        for side, s in subs:
            m, obj = s.get("motion", sg.get("motion")), s.get("object")
            cur = held.setdefault(side, [])
            if m in _TAKES and isinstance(obj, str):
                held[side] = [obj]
            elif m in _TAKES_MORE and isinstance(obj, str):
                cur.append(obj) if obj not in cur else None
            elif m in _KEEPS:
                if isinstance(obj, str) and obj not in cur:
                    cur.append(obj)
            elif m in _PUTS_DOWN or s.get("release") is True:
                held[side] = [o for o in cur if o != obj] if isinstance(obj, str) else []
    return [o for side in held for o in held[side]]


# Deformables run in Isaac Sim (execute_isaac_deformables): the checks that read object states run on the episode's
# MuJoCo mirror (the executed scene and trajectory plus each cord's recorded links), the same code as for a rigid
# object; the checks that re-simulate MuJoCo physics stay on the executed recording (the robot and rigid objects); the
# cord's own physics is isaac_recording_check (its re-simulation in Isaac Sim included).
MIRROR_CHECKS = ("stage1_goal_gate", "bimanual_sync_gate", "object_teleport_audit", "object_visual_penetration_audit",
                 "contactless_motion_audit", "free_fall_plausibility_audit", "object_spin_rate_audit",
                 "hand_object_relative_jitter_audit", "released_contact_audit", "held_object_slip_audit",
                 "table_workspace_bounds_audit", "deformable_integrity_audit", "camera_visibility_audit")

# Stage 4 randomization and physics-replay gates (run when draws > 0)
RANDOMIZATION_GATE = "stage4_randomization_gate"
PHYSICS_GATE = "stage4_physics_replay_gate"


def _call(name, kwargs):
    mod = importlib.import_module(f"tool_library.verifiers.{name}")
    return getattr(mod, name)(**kwargs)


def stage3_gate_suite(workspace, scene_semantic_json, scene_recon_json, setup_dir=None, draws=3, strict_quality=True, skip=None):
    ws = Path(workspace)
    man_p = ws / "trajectory_manifest.json"
    if not man_p.exists():
        return {"ok": False, "issues": ["trajectory_manifest.json is missing"], "warnings": [], "reports": {}, "skipped": {}, "seconds": {}}
    man = json.loads(man_p.read_text())
    recon_p = Path(scene_recon_json)
    recon = json.loads(recon_p.read_text())
    scene = (recon_p.parent / recon.get("scene_xml", "scene.xml")).resolve()
    pre_warnings, pre_issues = [], []
    if man.get("scene_executed"):
        se = Path(man["scene_executed"]); se = se if se.is_absolute() else ws / se
        if se.exists():
            scene = se.resolve()
        else:
            pre_issues.append(f"scene_executed does not exist: {se}")
    else:
        pre_issues.append("scene_executed is required: final checks need the actual rigid execution scene")
    if not man.get("scene_executed") and (ws / "place_arm_stations.json").exists():
        # the stations were moved but the manifest does not say where the episode ran: use the scene the report wrote
        rep = json.loads((ws / "place_arm_stations.json").read_text())
        se = ws / rep.get("scene", "scene_stations.xml")
        if se.exists():
            scene = se.resolve()
            pre_warnings.append(f"the manifest has no scene_executed; using {se.name} from place_arm_stations.json")
    epi = man.get("episode") if isinstance(man.get("episode"), dict) else {}
    traj = ws / (epi.get("trajectory") or "trajectory.npz")      # a null path (an unfinished run) is the default, never a crash
    if not traj.exists():
        return {"ok": False, "issues": [f"{traj} is missing"], "warnings": [], "reports": {}, "skipped": {}, "seconds": {}}
    camera = recon_p.parent / recon.get("camera_json", "camera.json")
    task_dir = ws / ((man.get("stage4") or {}).get("task_dir") or "stage4_task")
    stations_report = next((p for p in (ws / "place_arm_stations.json", ws / "stations.json") if p.exists()), None)
    S = {"setup": str(setup_dir)} if setup_dir else {}
    sc, tj, sem = str(scene), str(traj), str(scene_semantic_json)
    skip = set(skip or [])
    deform = man.get("deformables") if isinstance(man.get("deformables"), dict) else None
    isaac_objects = [o.get("id") for o in recon.get("objects", []) if o.get("type") == "deformable" and str(o.get("simulator", "")).lower() == "isaac"]
    msc, mtj = sc, tj
    if isaac_objects and not deform:
        pre_issues.append(f"the Stage 2 deformables {isaac_objects} are simulated in Isaac Sim, but the manifest has no deformables "
                          f"block: run execute_isaac_deformables on the executed episode and package its output")
    if deform:
        msc = str((ws / deform.get("scene_mirror", "scene_mirror.xml")).resolve())
        mtj = str((ws / deform.get("trajectory_mirror", "trajectory_mirror.npz")).resolve())
        for p_ in (msc, mtj):
            if not Path(p_).exists():
                pre_issues.append(f"deformables: {p_} is missing")

    plan = [  # (check, kwargs, blocking)
        ("episode_physics_check", {"scene": sc, "trajectory": tj, "stride": 1}, True),
        ("robot_self_collision_check", {"scene": sc, "trajectory": tj, "stride": 1}, True),
        ("rigid_arm_hand_check", {"scene": sc, "trajectory": tj, **S}, True),
        ("physics_recording_check", {"scene": sc, "trajectory": tj, "scene_recon_json": str(recon_p)}, True),
        ("execution_scene_integrity_check", {"scene": sc, "scene_recon_json": str(recon_p), "trajectory": tj}, True),
        ("arm_reach_check", {"scene": sc, "trajectory": tj, **S}, True),
        ("stage1_goal_gate", {"scene_semantic_json": sem, "trajectory_manifest_json": str(man_p), "scene": msc, "trajectory": mtj,
                              "scene_recon_json": str(recon_p)}, True),
        ("bimanual_sync_gate", {"scene_semantic_json": sem, "trajectory_manifest_json": str(man_p), "scene": msc, "trajectory": mtj,
                                "scene_recon_json": str(recon_p)}, True),
        ("robot_model_integrity_check", {"scene": sc, **S}, True),
        ("robot_joint_range_check", {"scene": sc, **S}, True),
        ("actuator_limit_integrity_check", {"scene": sc, **S}, True),
        ("contact_exclusion_audit", {"scene": sc, **S}, True),
        ("physics_options_check", {"scene": sc, **S}, True),
        ("stage_station_integrity_check", {"scene": sc, **S, "unrecorded_shift_ok": False,
                                           **({"stations_report": str(stations_report)} if stations_report else {})}, True),
        ("initial_keyframe_state_check", {"scene": sc, **S}, True),
        ("stage4_handoff_check", {"task_dir": str(task_dir), "camera_json": str(camera)}, True),
        ("uncovered_frames_check", {"trajectory": tj}, True),     # every frame labelled, no flicker phases
    ]
    extra = {
        "camera_visibility_audit": {"camera": str(camera)} if camera.exists() else {},
        "timing_realism_audit": {"scene_semantic": sem, "manifest": str(man_p)},
        "end_state_rest_audit": {"allow_held": _held_at_end(man)} if _held_at_end(man) else {},
        "joint_limit_margin_audit": {"reference": str(Path(setup_dir) / "scene.xml")} if setup_dir else {},
        "object_visual_penetration_audit": {"stride": 1},
        # Stable grasp quality cannot be waived in an agent-written expect.
        "held_object_slip_audit": {"words": ["lift", "carry", "transport", "pour", "tilt", "set down"]},
    }
    for name, blocking in AUDITS.items():
        on = (msc, mtj) if name in MIRROR_CHECKS else (sc, tj)
        plan.append((name, {"scene": on[0], "trajectory": on[1], **extra.get(name, {})}, blocking or strict_quality))
    if deform:
        plan.append(("isaac_recording_check", {"workspace": str(ws), "scene_semantic_json": sem, "scene_recon_json": str(recon_p),
                                               "replay": True}, True))
    augmentation_plan = []
    if int(draws) > 0:
        augmentation_plan.append((RANDOMIZATION_GATE, {"task_dir": str(task_dir), "draws": int(draws)}, True))
        # the original replayed in Stage 4's physics, and up to 3 accepted layouts of which one must hold
        augmentation_plan.append((PHYSICS_GATE, {"task_dir": str(task_dir), "draws": int(draws), "layouts": min(3, int(draws))}, True))
    else:
        pre_issues.append("Stage 4 randomization and physics replay are required; draws must be positive")
    if not strict_quality:
        pre_issues.append("strict_quality=False is diagnostic only; final acceptance requires all quality audits")

    issues, warnings, reports, skipped, seconds = list(pre_issues), list(pre_warnings), {}, {}, {}
    if skip:
        issues.append("skip requests are diagnostic only; final acceptance requires the complete suite")

    # Long, integration-step audits must be observable without interrupting the
    # run. This file is diagnostic only: no verifier reads it as evidence and a
    # new invocation always starts from scratch, irrespective of its contents.
    progress_path = ws / "gate_progress.json"
    invocation = uuid.uuid4().hex
    started = time.time()

    def progress(active=None, complete=False):
        payload = {"invocation": invocation, "diagnostic_only": True,
                   "started_unix_s": started, "updated_unix_s": time.time(),
                   "elapsed_s": round(time.time() - started, 2), "active_check": active,
                   "complete": complete,
                   "checks": {k: {"ok": v["ok"], "seconds": seconds[k],
                                  "issues": v.get("issues", [])} for k, v in reports.items()},
                   "issues": issues, "skipped": skipped}
        temp = progress_path.with_name(f".{progress_path.name}.{invocation}.tmp")
        try:
            temp.write_text(json.dumps(payload, indent=2, default=str))
            temp.replace(progress_path)
        except OSError:
            # A read-only diagnostic workspace does not change physical verdicts.
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass

    progress()

    def run(name, kwargs, blocking, label=None):
        label = label or name
        if name in skip:
            skipped[label] = "skipped by request"
            issues.append(f"{label}: required check skipped; incomplete validation is not a pass")
            progress()
            return
        progress(active=label)
        t0 = time.time()
        try:
            r = _call(name, kwargs)
            if not isinstance(r, dict) or type(r.get("ok")) is not bool:
                raise ValueError("verifier must return a dictionary with a boolean ok")
            if r.get("skipped") or r.get("unavailable"):
                r = {**r, "ok": False, "issues": list(r.get("issues") or []) + ["required validation was skipped or unavailable"]}
        except Exception as e:
            r = {"ok": False, "issues": [f"crashed: {type(e).__name__}: {e}"]}
        seconds[label] = round(time.time() - t0, 2)
        reports[label] = r
        msgs = [f"{label}: {i}" for i in (r.get("issues") or ([] if r.get("ok") else [json.dumps(r.get("failed") or "failed", default=str)[:300]]))]
        if not r.get("ok"):
            (issues if blocking else warnings).extend(msgs)
        warnings.extend(f"{label}: {w}" for w in (r.get("warnings") or []))
        progress()

    for name, kwargs, blocking in plan:
        run(name, kwargs, blocking)
    # every executed entry's own expectation, re-measured over its frames
    from tool_library.common.task_gates import expand_entries
    for sg in expand_entries(man.get("subgoals", [])):
        if sg.get("skipped") or not sg.get("expect"):
            continue
        win = {"frames": sg["frames"]} if sg.get("frames") else {"subgoal": sg.get("index")}
        run("subgoal_outcome_check", {"scene": sc, "trajectory": tj, "expect": sg["expect"], **win}, True,
            label=f"subgoal_outcome_check[{sg.get('index')}{'.' + str(sg['step']) if sg.get('step') is not None else ''}]")
    # Finish all original-episode diagnostics first. An already rejected motion
    # cannot become publishable through augmentation, so do not spend minutes
    # replaying its randomized copies before the agent sees the actual blocker.
    # Successful acceptance still requires every augmentation check below.
    deferred = {}
    for name, kwargs, blocking in augmentation_plan:
        if issues:
            deferred[name] = "required validation pending: repair preceding failed checks first"
        else:
            run(name, kwargs, blocking)
    progress(complete=True)
    return {"ok": not issues and not deferred, "issues": issues, "warnings": warnings, "reports": reports, "skipped": skipped,
            "deferred": deferred, "seconds": seconds,
            "scene": sc, "trajectory": tj}
