"""Bookkeeping check of a Stage 3 deliverable: trajectory_manifest.json."""
from __future__ import annotations

TOOL = {
    "name": "trajectory_manifest_check",
    "title": "Trajectory manifest check",
    "type": "verifier",
    "category": "Task & Rollout Gates",
    "summary": "Check a Stage 3 trajectory_manifest.json: every Stage 1 subgoal is executed (or merged/skipped with a reason), "
               "each subgoal names its hand pose source and the stress tests that passed on it, every executed subgoal's "
               "outcome passed, the episode files exist and agree, and the Stage 4 task folder is referenced.",
    "description": (
        "The floor for finishing Stage 3. Reads the manifest and the Stage 1 scene_semantics.json it points to, and "
        "verifies: `subgoals` covers every Stage 1 index (an entry may list several Stage 1 indices it merges, or carry "
        "`skipped` with a reason and `skip_kind` 'no_robot_action' (the human only looks or waits) or 'merged' (done inside "
        "another entry); a manipulation step skipped for any other reason fails: the task is not complete); each entry has motion, side, object (body in the scene), and for grasping motions "
        "a `pose` with `source` (the skill that produced it) and, for grasping motions (grasp_lift, hold, unscrew, "
        "screw, open_joint, close_joint, pull, regrasp), `stress_tests` naming at least one library verifier with ok true; "
        "grasp_more requires combined_hold_state_test bound to its actual segment-end frame and every retained object "
        "in the trusted execution history (a push or "
        "press is proved by its outcome alone; a `cut` is proved by its outcome's `severed` check -- the rope's join switched "
        "off by the blades during the subgoal, read from the recorded equality state -- and, when present, `cut_ends_apart`, "
        "and then the trajectory must carry eq_active / eq_names with that join off at its last frame); each has `outcome.ok` true; `episode.trajectory` exists, has the scene's nq columns and 'side: "
        "phase' labels; `episode.sim_video` exists; `stage4.task_dir` holds stage4.json; the checks block carries the "
        "episode verifiers (episode_physics_check, robot_self_collision_check, arm_reach_check, stage4_handoff_check); "
        "when Stage 2 simulates a deformable in Isaac Sim, the deformables block names the Isaac report, deformables.json, the "
        "recording and the MuJoCo mirror scene and trajectory (of matching width) "
        "with ok true. Returns ok and the list of issues."
    ),
    "inputs": [
        {"name": "path", "type": "str", "description": "trajectory_manifest.json."},
        {"name": "scene_semantic_json", "type": "str | None", "description": "Stage 1 manifest (default from the manifest)."},
        {"name": "library_index", "type": "str | None", "description": "index.json to validate verifier names against."},
        {"name": "trusted_verifiers", "type": "dict | None", "description": "Run-start snapshot mapping independently reviewed grasp verifier names to source SHA-256; new tools cannot certify their own result."},
        {"name": "trusted_execution_reports", "type": "list[str] | None", "description": "Harness-captured SHA-256 digests of successful direct executor report files. An empty list rejects all executions; None permits standalone diagnostics."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok, issues, counts {subgoals, executed, skipped, stress_tests}."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "manifest", "bookkeeping"],
    "used_in": ["Open the Bottle", "Giftbox Packing", "Place Toy in Drawer", "Battery Insertion", "Place Cup in Bowl", "Computer Installation"],
    "status": "tested",
    "entrypoint": "trajectory_manifest_check",
}

import json
import importlib
import inspect
from pathlib import Path

SKIP_KINDS = {"no_robot_action", "merged"}

import mujoco
import numpy as np

# need a pose with a source (the two-hand, pin and thin-part motions too)
POSE_MOTIONS = {"grasp_lift", "hold", "unscrew", "screw", "open_joint", "close_joint", "pull", "push", "press", "regrasp",
                "grasp_more", "lift_together", "pin", "slide", "tilt_up", "edge_pinch", "overhang_grasp", "hook", "strip_pinch",
                "lay", "drag"}
# and a passing stress test (every motion that holds a part in the fingers)
GRASP_MOTIONS_TESTED = {"grasp_lift", "hold", "unscrew", "screw", "open_joint", "close_joint", "pull", "regrasp",
                        "grasp_more", "lift_together", "edge_pinch", "overhang_grasp", "hook", "strip_pinch"}
GRASP_CONTINUATIONS = {"carry", "carry_together", "place", "insert", "pour", "tilt", "release"}
CUT_MOTIONS = {"cut"}                    # proved by outcome.checks.severed and the trajectory's recorded eq_active


def _stage1_scope(entry):
    indices = entry.get("stage1_indices") or ([entry["index"]] if "index" in entry else [])
    if (not isinstance(indices, list) or not indices
            or any(type(i) is not int for i in indices) or len(indices) != len(set(indices))):
        raise ValueError("Stage 1 index scope must be a nonempty list of distinct integers")
    return sorted(indices)


def _bind_execution_identity(entry, executed, tag, issues):
    """Generator metadata cannot change the obligations of a trusted execution."""
    if executed.get("ok") is not True or executed.get("dry_run"):
        issues.append(f"{tag}: bound executor entry did not pass a real physical execution")
    hands = entry.get("hands")
    actual = {"index": entry.get("index"), "motion": entry.get("motion"), "side": "both" if isinstance(hands, dict) else entry.get("side"),
              "object": {side: (part or {}).get("object") for side, part in hands.items()} if isinstance(hands, dict) else entry.get("object"),
              "step": entry.get("step")}
    for key, value in actual.items():
        if value != executed.get(key) or type(value) is not type(executed.get(key)):
            issues.append(f"{tag}: manifest {key} differs from the bound executor entry")
    for record, name in ((entry, "manifest"), (executed, "executor")):
        if type(record.get("index")) is not int:
            issues.append(f"{tag}: {name} index must be a literal integer")
        frames = record.get("frames")
        if (not isinstance(frames, list) or len(frames) != 2 or any(type(n) is not int for n in frames)
                or frames[0] < 0 or frames[1] < frames[0]):
            issues.append(f"{tag}: {name} frames must be an ordered inclusive pair of nonnegative integers")
        if type(record.get("release", False)) is not bool:
            issues.append(f"{tag}: {name} release flag must be a literal boolean")
    if entry.get("release", False) != executed.get("release", False):
        issues.append(f"{tag}: manifest release differs from the bound executor entry")
    if executed.get("motion") == "both":
        actions = executed.get("hands")
        if not isinstance(actions, dict) or set(actions) != {"left", "right"}:
            issues.append(f"{tag}: executor lacks trusted per-hand action identity; rerun this simultaneous entry")
        elif isinstance(hands, dict):
            for side in ("left", "right"):
                declared, performed = hands.get(side), actions[side]
                if not isinstance(declared, dict) or not isinstance(performed, dict):
                    issues.append(f"{tag} ({side}): missing per-hand action identity")
                    continue
                if declared.get("side", side) != side:
                    issues.append(f"{tag} ({side}): per-hand side differs from its trusted hand key")
                for key in ("motion", "object", "release"):
                    default = False if key == "release" else None
                    given, actual_value = declared.get(key, default), performed.get(key, default)
                    if (given != actual_value or type(given) is not type(actual_value)
                            or (key == "release" and type(given) is not bool)):
                        issues.append(f"{tag} ({side}): manifest {key} differs from the bound per-hand executor action")
    try:
        if _stage1_scope(entry) != _stage1_scope(executed):
            issues.append(f"{tag}: Stage 1 index scope differs from the bound executor entry")
    except (ValueError, TypeError) as exc:
        issues.append(f"{tag}: invalid execution scope: {exc}")


def trajectory_manifest_check(path, scene_semantic_json=None, library_index=None, trusted_verifiers=None, trusted_execution_reports=None):
    p = Path(path).resolve(); ws = p.parent
    if not p.exists():
        return {"ok": False, "issues": [f"{p} missing"], "counts": {}}
    man = json.loads(p.read_text())
    issues = []
    stress_cache = {}
    def rel(x):
        q = Path(x)
        return q if q.is_absolute() else (ws / q)
    sem_p = Path(scene_semantic_json) if scene_semantic_json else rel(man.get("scene_semantic", "../stage1/scene_semantics.json"))
    sem = json.loads(sem_p.read_text()) if sem_p.exists() else None
    if sem is None:
        issues.append("scene_semantics.json not found")
    recon_p = rel(man.get("scene_recon", "../stage2/scene_recon.json"))
    scene_xml = None
    if recon_p.exists():
        recon = json.loads(recon_p.read_text()); scene_xml = (recon_p.parent / recon.get("scene_xml", "scene.xml")).resolve()
    else:
        issues.append("scene_recon.json not found")
    lib = Path(library_index).resolve().parent if library_index else Path(__file__).resolve().parents[1]
    names = None
    if library_index and Path(library_index).exists():
        names = {t["name"] for t in json.loads(Path(library_index).read_text())["tools"]}
    if not man.get("scene_executed") and (ws / "place_arm_stations.json").exists():
        issues.append("the stations were moved (place_arm_stations.json) but scene_executed is missing: name the scene the episode "
                      "ran in (scene_stations.xml) so the audits re-run there")
    se = man.get("scene_executed")
    if se and rel(se).exists():
        scene_xml = rel(se).resolve()             # bodies and qpos width are checked in the scene the episode ran in
    elif se:
        issues.append(f"scene_executed does not exist: {se}")
    model = mujoco.MjModel.from_xml_path(str(scene_xml)) if scene_xml and scene_xml.exists() else None
    epi = man.get("episode") if isinstance(man.get("episode"), dict) else {}
    traj = rel(epi.get("trajectory") or "trajectory.npz")
    execution = None
    execution_path = rel(epi["execution_report"]) if epi.get("execution_report") else traj.parent / "execution_report.json"
    try:
        from tool_library.common.evidence import file_digest
        if trusted_execution_reports is not None and file_digest(execution_path) not in trusted_execution_reports:
            raise ValueError("execution report was not captured from a trusted executor invocation by this harness")
        execution = json.loads(execution_path.read_text())
        if execution.get("ok") is not True or execution.get("dry_run"):
            raise ValueError("executor did not complete a real, successful rollout")
        if execution.get("trajectory_sha256") != file_digest(traj):
            raise ValueError("execution report belongs to a different trajectory")
        if execution.get("scene_sha256") != file_digest(scene_xml):
            raise ValueError("execution report belongs to a different execution scene")
    except Exception as exc:
        issues.append(f"episode.execution_report: {type(exc).__name__}: {exc}")
        execution = None
    subs = man.get("subgoals", [])
    covered = set(); n_exec = n_skip = n_tests = 0
    bound_entries = set()
    cut_joins = []
    for i, sg in enumerate(subs):
        idx = sg.get("stage1_indices") or ([sg["index"]] if "index" in sg else [])
        covered |= {int(x) for x in idx}
        if sg.get("skipped"):
            n_skip += 1
            if not sg.get("reason"):
                issues.append(f"subgoal {idx}: skipped without a reason")
            elif sg.get("skip_kind") not in SKIP_KINDS:
                # a manipulation step left out is an unfinished task, whatever the reason: only a step with no robot
                # action (the human looks, waits) or one done inside another entry may be skipped
                issues.append(f"subgoal {idx}: skipped ({str(sg.get('reason'))[:120]}) - the task is not complete; skipping is "
                              f"only for a step with no robot action or one merged into another entry (skip_kind one of {sorted(SKIP_KINDS)})")
            elif sg.get("skip_kind") == "no_robot_action":
                from tool_library.common.task_gates import no_robot_action_allowed
                stages = {int(s["index"]): s for s in (sem or {}).get("subgoals", [])}
                if not idx or any(int(k) not in stages or not no_robot_action_allowed(stages[int(k)]) for k in idx):
                    issues.append(f"subgoal {idx}: no_robot_action is not supported by the independent Stage 1 input")
            elif sg.get("skip_kind") == "merged":
                others = {int(k) for other in subs if other is not sg and not other.get("skipped")
                          for k in (other.get("stage1_indices") or [other.get("index")]) if k is not None}
                if not set(int(k) for k in idx) <= others:
                    issues.append(f"subgoal {idx}: merged indices must be covered by an executed entry")
            continue
        if sg.get("executed") is False:
            # an unfinished run lists the steps it never executed: one issue each, not counted as executed
            issues.append(f"subgoal {idx}: not executed ({str(sg.get('blocked') or sg.get('reason') or 'no outcome')[:160]})")
            continue
        n_exec += 1
        executed_entry = None
        if execution is not None:
            matches = [(n, r) for n, r in enumerate(execution.get("subgoals", []))
                       if r.get("frames") == sg.get("frames") and r.get("index") == sg.get("index")]
            if len(matches) != 1:
                issues.append(f"subgoal {idx}: frames and index do not identify one entry in the bound execution report")
            else:
                number, record = matches[0]
                if number in bound_entries:
                    issues.append(f"subgoal {idx}: bound executor entry is duplicated in the manifest")
                bound_entries.add(number)
                executed_entry = {**record, "_bound_runtime_scene_sha256": execution["scene_sha256"]}
                _bind_execution_identity(sg, executed_entry, f"subgoal {idx}", issues)
        if sg.get("motion") == "both" or isinstance(sg.get("hands"), dict):
            # a two-hand entry (both, lift_together): each hand's sub-entry carries its own object, pose and stress tests
            # (and its own motion in a 'both' entry; the entry's motion otherwise)
            hands = sg.get("hands") if isinstance(sg.get("hands"), dict) else {}
            if set(hands) != {"left", "right"}:
                issues.append(f"subgoal {idx}: a '{sg.get('motion')}' entry needs hands.left and hands.right")
            for side, sub in sorted(hands.items()):
                sub = dict(sub or {}) if sg.get("motion") == "both" else {**(sub or {}), "motion": sg.get("motion")}
                n_tests += _entry_checks(sub, f"subgoal {idx} ({side})", model, names, issues, side_default=side,
                                         workspace=ws, library=lib, stress_cache=stress_cache, trusted_verifiers=trusted_verifiers,
                                         execution_context={"execution": execution, "selected": executed_entry,
                                                            "scene": scene_xml, "trajectory": traj})
                _bind_execution_pose(sub, executed_entry, side, ws, f"subgoal {idx} ({side})", issues)
        else:
            n_tests += _entry_checks(sg, f"subgoal {idx}", model, names, issues,
                                     workspace=ws, library=lib, stress_cache=stress_cache, trusted_verifiers=trusted_verifiers,
                                     execution_context={"execution": execution, "selected": executed_entry,
                                                        "scene": scene_xml, "trajectory": traj})
            _bind_execution_pose(sg, executed_entry, sg.get("side"), ws, f"subgoal {idx}", issues)
        oc = sg.get("outcome") or {}
        if oc.get("ok") is not True:
            issues.append(f"subgoal {idx}: outcome not ok ({oc.get('failed')})")
        ck = oc.get("checks") or {}
        if sg.get("motion") in CUT_MOTIONS or "severed" in ck:
            # a cut is proved by the recorded equality state, not by a pose: the rope's join must have been switched off
            # by the blades during this subgoal (outcome severed) and the cut ends must have come apart
            if not ck.get("severed"):
                issues.append(f"subgoal {idx}: a cut is proved by outcome.checks.severed (the join equality switched off by "
                              f"scissors.try_sever during the subgoal, read from the recorded eq_active); it is missing or false")
            elif "cut_ends_apart" in ck and not ck["cut_ends_apart"]:
                issues.append(f"subgoal {idx}: the rope's join was switched off but the cut ends did not come apart")
            cut_joins += [str(n) for n in ((oc.get("measured") or {}).get("severed") or {})]
    if execution is not None:
        omitted = sorted(set(range(len(execution.get("subgoals", [])))) - bound_entries)
        if omitted:
            issues.append(f"manifest omits bound executor entries at positions {omitted}; every physical segment must be represented once")
    if sem is not None:
        want = {int(s["index"]) for s in sem.get("subgoals", [])}
        missing = sorted(want - covered)
        if missing:
            issues.append(f"Stage 1 subgoals not covered: {missing}")
    if not traj.exists():                              # epi / traj as read above (a null episode is an issue, not a crash)
        issues.append("episode.trajectory missing")
    else:
        z = np.load(traj, allow_pickle=True)
        if model is not None and z["qpos"].shape[1] != model.nq:
            issues.append(f"trajectory has {z['qpos'].shape[1]} columns, scene nq={model.nq}")
        if "labels" not in z.files:
            issues.append("trajectory has no labels")
        elif any(": " not in str(l) for l in z["labels"]):
            issues.append("labels must read 'side: phase' (or 'left: phase | right: phase' for a two-hand step)")
        if "fps" not in z.files:
            issues.append("trajectory has no fps")
        if cut_joins:
            # the cut is run-time state no qpos holds: the trajectory must carry it for Stage 4 and any physics replay
            if "eq_active" not in z.files or "eq_names" not in z.files:
                issues.append("a subgoal cuts the rope but trajectory.npz has no eq_active / eq_names (run execute_subgoal_plan "
                              "with the scene_recon_json that carries the rope's cut_joins)")
            else:
                eq_names = [str(x) for x in z["eq_names"]]
                for nm in sorted(set(cut_joins)):
                    if nm not in eq_names:
                        issues.append(f"cut join {nm!r} is not a column of the trajectory's eq_names")
                    elif bool(z["eq_active"][-1][eq_names.index(nm)]):
                        issues.append(f"cut join {nm!r} is whole again at the trajectory's last frame")
    if not rel(epi.get("sim_video", "sim_video.mp4")).exists():
        issues.append("episode.sim_video missing")
    s4 = man.get("stage4") or {}
    if not s4.get("task_dir") or not (rel(s4["task_dir"]) / "stage4.json").exists():
        issues.append("stage4.task_dir with stage4.json missing")
    # deformables simulated in Isaac Sim: the recording and its MuJoCo mirror travel with the episode
    isaac_cords = [o.get("id") for o in (recon.get("objects", []) if recon_p.exists() else [])
                   if o.get("type") == "deformable" and str(o.get("simulator", "")).lower() == "isaac"]
    blk = man.get("deformables")
    if isaac_cords and not isinstance(blk, dict):
        issues.append(f"Stage 2 simulates {isaac_cords} in Isaac Sim: the manifest needs the deformables block (run "
                      f"execute_isaac_deformables on the execution and pass its out_dir to package_stage3_episode as isaac_episode)")
    if isinstance(blk, dict):
        for key in ("report", "spec", "recording", "scene_mirror", "trajectory_mirror"):
            if not blk.get(key) or not rel(blk[key]).is_file():
                issues.append(f"deformables.{key} missing")
        if blk.get("scene_mirror") and blk.get("trajectory_mirror") and rel(blk["scene_mirror"]).is_file() and rel(blk["trajectory_mirror"]).is_file():
            mm = mujoco.MjModel.from_xml_path(str(rel(blk["scene_mirror"])))
            with np.load(rel(blk["trajectory_mirror"]), allow_pickle=True) as zm:
                if zm["qpos"].shape[1] != mm.nq:
                    issues.append(f"deformables: the mirror trajectory has {zm['qpos'].shape[1]} columns, its scene nq={mm.nq}")
    checks = man.get("checks") or {}
    for n in ("episode_physics_check", "robot_self_collision_check", "arm_reach_check", "stage4_handoff_check"):
        r = checks.get(n)
        if not isinstance(r, dict):
            issues.append(f"checks.{n} missing")
        elif r.get("ok") is not True:
            issues.append(f"checks.{n} not ok")
    return {"ok": not issues, "issues": issues, "counts": {"subgoals": len(subs), "executed": n_exec, "skipped": n_skip, "stress_tests": n_tests}}


def _bind_execution_pose(entry, executed, side, workspace, tag, issues):
    """The independently stress-tested input must be the executor's selected pose.

    A candidate may be changed by variant search or re-anchoring. Candidate
    evidence cannot certify the changed grasp even when both files say `ok`.
    Continuous carry entries need no new pose certification when no pose is used.
    """
    selected = executed.get("selected_pose_evidence") if isinstance(executed, dict) else None
    if isinstance(selected, dict) and ("left" in selected or "right" in selected):
        selected = selected.get(side)
    if entry.get("motion") not in POSE_MOTIONS and not entry.get("pose") and not selected:
        return
    if not isinstance(executed, dict):
        issues.append(f"{tag}: no bound execution entry identifies the pose actually selected")
        return
    selected = executed.get("selected_pose_evidence") or {}
    if side in selected:
        selected = selected[side]
    if not isinstance(selected, dict) or not selected.get("path") or not selected.get("sha256"):
        issues.append(f"{tag}: execution report lacks selected_pose_evidence for {side}")
        return
    from tool_library.common.evidence import file_digest, input_record
    def resolve(value):
        p = Path(value)
        return p if p.is_absolute() else Path(workspace) / p
    pose = entry.get("pose") or {}
    try:
        expected_scene = executed.get("_bound_runtime_scene_sha256")
        if expected_scene is not None and selected.get("runtime_scene_sha256") != expected_scene:
            raise ValueError("selected pose evidence belongs to a different runtime scene")
        actual = resolve(selected["path"])
        if file_digest(actual) != selected["sha256"]:
            raise ValueError("the selected pose file changed after execution")
        if not pose.get("file") or input_record(str(resolve(pose["file"])))['sha256'] != input_record(str(actual))['sha256']:
            raise ValueError("pose.file is not the actual selected/re-anchored grasp; rerun stress tests on pose_used")
        hs = selected.get("hand_scene_used") or selected.get("hand_scene_path")
        hs_sha = selected.get("hand_scene_used_sha256") or selected.get("hand_scene_sha256")
        if not hs or not hs_sha:
            raise ValueError("executor lacks hand_scene_used evidence for the actual starting object state")
        actual_scene = resolve(hs)
        if file_digest(actual_scene) != hs_sha:
            raise ValueError("the selected hand scene changed after execution")
        if selected.get("start_state_proposal"):
            if file_digest(resolve(selected["start_state_proposal"])) != selected.get("start_state_proposal_sha256"):
                raise ValueError("the recorded starting state of the selected grasp changed after execution")
        supplied = pose.get("hand_scene") or entry.get("hand_scene")
        if not supplied or input_record(str(resolve(supplied)))['sha256'] != input_record(str(actual_scene))['sha256']:
            raise ValueError("pose.hand_scene is not the actual selected grasp's starting hand scene")
    except Exception as exc:
        issues.append(f"{tag}: executed pose binding failed: {type(exc).__name__}: {exc}")


def _entry_checks(sg, tag, model, names, issues, side_default=None, workspace=None, library=None, stress_cache=None, trusted_verifiers=None,
                  execution_context=None) -> int:
    """Motion, side, object and the hand pose's source and stress tests of one executed entry; returns the number of
    stress tests it lists."""
    for k in ("motion", "side"):
        if not (sg.get(k) or (k == "side" and side_default)):
            issues.append(f"{tag}: missing {k}")
    obj = sg.get("object")
    if isinstance(obj, str) and obj and model is not None and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj) < 0:
        issues.append(f"{tag}: object body {obj!r} not in the scene")
    elif obj is not None and not isinstance(obj, str):
        issues.append(f"{tag}: object must be one body name (a two-hand step names each hand's object in hands.<side>)")
    n = 0
    if sg.get("motion") in POSE_MOTIONS or sg.get("pose"):
        pose = sg.get("pose") or {}
        stress_required = (sg.get("motion") in GRASP_MOTIONS_TESTED or
                           (bool(pose) and sg.get("motion") in GRASP_CONTINUATIONS))
        if not pose.get("source"):
            issues.append(f"{tag}: pose.source (the skill that produced the hand pose) missing")
        tests = pose.get("stress_tests") or sg.get("stress_tests") or {}
        passed = [k for k, r in tests.items() if (r.get("ok") if isinstance(r, dict) else bool(r))]
        n = len(tests)
        if not passed and stress_required:
            issues.append(f"{tag}: no passing stress test on its hand pose")
        if passed and stress_required:
            evidence_errors, verified = [], []
            for name in passed:
                try:
                    _verify_stress_evidence(name, tests[name], pose, sg, side_default, workspace, library, stress_cache, trusted_verifiers,
                                            execution_context=execution_context)
                    verified.append(name)
                except Exception as exc:
                    evidence_errors.append(f"{name}: {type(exc).__name__}: {exc}")
            if not verified:
                issues.append(f"{tag}: no independently passing stress evidence ({'; '.join(evidence_errors)})")
        if names is not None:
            for k in tests:
                if k.split("@")[0] not in names:
                    issues.append(f"{tag}: stress test {k!r} is not a library tool")
    return n


def _verify_stress_evidence(name, report, pose, entry, side_default, workspace, library, cache=None, trusted_verifiers=None,
                            execution_context=None):
    """Check content identity and independently execute the verifier at its defaults.

    Generator-supplied result booleans and hashes are never an authority. Defaults
    are used so a caller cannot obtain acceptance by weakening stress parameters.
    Required tool-specific arguments (e.g. the joint being turned) are retained.
    """
    from tool_library.common.evidence import file_digest, input_record, value_digest
    ws, lib = Path(workspace), Path(library)
    def resolve(path):
        p = Path(path)
        return p if p.is_absolute() else ws / p
    evidence = report.get("evidence") if isinstance(report, dict) else None
    if not isinstance(evidence, dict) or not evidence.get("path") or not evidence.get("sha256"):
        raise ValueError("missing evidence.path and evidence.sha256 from an actual library invocation")
    ep = resolve(evidence["path"])
    if file_digest(ep) != evidence["sha256"]:
        raise ValueError("evidence file hash mismatch")
    rec = json.loads(ep.read_text())
    name = name.split("@")[0]
    if entry.get("motion") == "grasp_more" and name != "combined_hold_state_test":
        raise ValueError("grasp_more requires combined_hold_state_test on the actual recorded combined hold; "
                         "a single-object test or a regrasp from the original scene cannot certify retained grips")
    source = lib / "verifiers" / f"{name}.py"
    metadata = rec.get("tool") or {}
    if trusted_verifiers is not None and trusted_verifiers.get(name) != file_digest(source):
        raise ValueError("verifier was not trusted with this source at run start")
    if (rec.get("schema") != "dexagent.tool_evidence.v1" or metadata.get("name") != name
            or metadata.get("type") != "verifier" or metadata.get("category") != "Grasp Stress Tests"
            or metadata.get("source_sha256") != file_digest(source)):
        raise ValueError("evidence does not identify the current grasp stress verifier")
    mod = importlib.import_module(f"{lib.name}.verifiers.{name}")
    if mod.TOOL.get("category") != "Grasp Stress Tests" or mod.TOOL.get("type") != "verifier":
        raise ValueError("the named library tool is not a grasp stress verifier")
    if name == "combined_hold_state_test":
        return _verify_combined_state_evidence(rec, mod, entry, side_default, ws, execution_context, cache)
    actual = {}
    for key, path in (("pose", pose.get("file")), ("scene", pose.get("hand_scene") or entry.get("hand_scene"))):
        if not path:
            raise ValueError(f"pose.{ 'file' if key == 'pose' else 'hand_scene'} is required")
        actual[key] = input_record(str(resolve(path)))
        bound = (rec.get("inputs") or {}).get(key) or {}
        if bound.get("sha256") != actual[key]["sha256"] or value_digest(bound.get("value")) != actual[key]["sha256"]:
            raise ValueError(f"stress evidence belongs to a different {key}")
    scene = actual["scene"]["value"]
    if scene.get("side") != (entry.get("side") or side_default) or scene.get("grasp_body") != entry.get("object"):
        raise ValueError("stress evidence tests a different hand or object")
    if (rec.get("result") or {}).get("ok") is not True:
        raise ValueError("the recorded test did not pass")
    fn = getattr(mod, mod.TOOL.get("entrypoint") or name)
    args = {key: actual[key]["value"] for key in ("pose", "scene")}
    supplied = rec.get("arguments") or {}
    for key, param in inspect.signature(fn).parameters.items():
        if key not in args and param.default is inspect.Parameter.empty and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            if key not in supplied:
                raise ValueError(f"recorded call is missing required argument {key}")
            args[key] = supplied[key]
    key = value_digest({"tool": metadata, "args": args})
    if cache is not None and key in cache:
        result = cache[key]
    else:
        result = fn(**args)
        if cache is not None:
            cache[key] = result
    if not isinstance(result, dict) or result.get("ok") is not True:
        raise ValueError(f"independent default-parameter stress replay failed: {str(result)[:600]}")
    for inp, record in actual.items():
        if input_record(record["path"])["sha256"] != record["sha256"]:
            raise ValueError(f"stress replay changed its {inp}")


def _combined_objects_from_execution(context, side):
    """Derive retained bodies from the bound executor, never the manifest's claim.

    The current executor summary does not expose per-hand motion history for a
    simultaneous ``both`` entry. Ambiguous histories fail instead of guessing.
    """
    if not isinstance(context, dict) or not isinstance(context.get("execution"), dict) or not isinstance(context.get("selected"), dict):
        raise ValueError("combined stress requires a bound successful execution report")
    execution, selected = context["execution"], context["selected"]
    if execution.get("ok") is not True or execution.get("dry_run") or selected.get("motion") != "grasp_more":
        raise ValueError("combined stress must bind a real successful grasp_more execution")
    held, found, last_end = [], False, -1
    for item in execution.get("subgoals", []):
        frames = item.get("frames")
        if (not isinstance(frames, list) or len(frames) != 2 or any(isinstance(n, bool) or not isinstance(n, int) for n in frames)
                or frames[0] < last_end or frames[1] < frames[0] or item.get("ok") is not True or item.get("dry_run")):
            raise ValueError("combined stress needs an ordered successful physical execution history")
        last_end = frames[1]
        motion, acting, obj = item.get("motion"), item.get("side"), item.get("object")
        if acting == "both" or motion == "both" or isinstance(obj, dict):
            raise ValueError("combined stress cannot yet authenticate retained grips across simultaneous hand entries")
        if acting == side:
            if not isinstance(obj, str):
                raise ValueError("combined stress history lacks the acted-on object")
            if motion in ("release", "place"):
                held = [name for name in held if name != obj]
            elif motion == "grasp_more":
                if not held or obj in held:
                    raise ValueError("invalid retained-object acquisition history")
                held.append(obj)
            elif motion in ("grasp_lift", "hold", "regrasp", "edge_pinch", "overhang_grasp", "hook", "strip_pinch"):
                if held and held != [obj]:
                    raise ValueError("combined stress history changes retained grips without an explicit release")
                held = [obj]
            elif motion not in ("carry", "carry_together", "insert", "pour", "tilt"):
                raise ValueError(f"combined stress cannot infer retention through motion {motion!r}")
        if item.get("frames") == selected.get("frames") and item.get("index") == selected.get("index"):
            if acting != side or len(held) < 2:
                raise ValueError("selected entry does not identify a combined hold by this hand")
            found = True
            break
    if not found:
        raise ValueError("selected combined grasp is absent from the execution history")
    return held


def _verify_combined_state_evidence(rec, mod, entry, side_default, ws, context, cache):
    from tool_library.common.evidence import input_record, value_digest, file_digest
    side = entry.get("side") or side_default
    objects = _combined_objects_from_execution(context, side)
    selected = context["selected"]
    if entry.get("object") != objects[-1] or entry.get("motion") != "grasp_more":
        raise ValueError("combined evidence belongs to a different acquisition")
    scene, trajectory = Path(context["scene"]), Path(context["trajectory"])
    execution = context["execution"]
    if (file_digest(scene) != execution.get("scene_sha256")
            or file_digest(trajectory) != execution.get("trajectory_sha256")):
        raise ValueError("combined stress execution scene or trajectory changed")
    args = {"scene": str(scene), "trajectory": str(trajectory), "frame": selected["frames"][1],
            "side": side, "objects": objects}
    actual = {key: input_record(value) for key, value in args.items()}
    supplied = rec.get("arguments") or {}
    for key, record in actual.items():
        bound = (rec.get("inputs") or {}).get(key) or {}
        if bound.get("sha256") != record["sha256"] or bound.get("value") != record["value"]:
            raise ValueError(f"combined stress evidence belongs to a different {key}")
        # The immutable input record and actual invocation must agree too.
        if key in ("scene", "trajectory"):
            value = Path(supplied.get(key) or "")
            value = value if value.is_absolute() else Path(rec.get("cwd") or ws) / value
            if value.resolve() != Path(record["value"]).resolve():
                raise ValueError(f"combined stress invocation used a different {key}")
        elif supplied.get(key) != args[key]:
            raise ValueError(f"combined stress invocation used a different {key}")
    if (rec.get("result") or {}).get("ok") is not True:
        raise ValueError("recorded combined hold test did not pass")
    key = value_digest({"tool": rec["tool"], "inputs": actual})
    if cache is not None and key in cache:
        result = cache[key]
    else:
        result = mod.combined_hold_state_test(**args)
        if cache is not None:
            cache[key] = result
    if not isinstance(result, dict) or result.get("ok") is not True:
        raise ValueError(f"independent actual-state combined stress failed: {str(result)[:600]}")
    for key in ("scene", "trajectory"):
        if input_record(args[key])["sha256"] != actual[key]["sha256"]:
            raise ValueError(f"combined stress replay changed its {key}")
