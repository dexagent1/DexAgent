"""Every scene check a Stage 2 deliverable must pass, in one call: the harness runs it at `finish`.

    from tool_library.verifiers.stage2_gate_suite import stage2_gate_suite
    r = stage2_gate_suite("runs/x/stage2", "runs/x/stage1/scene_semantics.json")
    r["ok"], r["issues"], r["reports"]["robot_joint_range_check"]

The suite reads scene_recon.json in the workspace (scene, camera, masks, objects) and runs the library's scene
assertions and compile checks on the scene the run delivers: the robot and the physics must be the setup's (no widened
ranges, stronger actuators, disabled contacts or moved stations), the initial state must hold, the objects must be
physically plausible and seen where the video shows them, and the scene must compile cleanly and simulate stably.
"""
from __future__ import annotations

TOOL = {
    "name": "stage2_gate_suite",
    "title": "Stage 2 gate suite",
    "type": "verifier",
    "category": "Scene Compile Checks",
    "summary": "Runs source-reference provenance, every scene assertion and compile check a Stage 2 scene must pass (robot and physics untouched, initial "
               "state settled, plausible objects, camera and Stage 1 consistency, stable compile) and returns one verdict.",
    "description": (
        "Reads `scene_recon.json` in the workspace for the scene, camera, masks and objects, and the Stage 1 file for the "
        "objects' roles and properties. Source masks (including the hands exclusion) require trusted harness receipts tied to the original video and matching reference frame; standalone/legacy masks and written provenance are unverified. Blocking checks (any failure fails the suite): source_reference_provenance_check, robot_model_integrity_check, "
        "robot_joint_range_check, actuator_limit_integrity_check, contact_exclusion_audit, physics_options_check, "
        "stage_station_integrity_check, initial_keyframe_state_check, hand_contact_pair_mask_check, "
        "object_mass_plausibility_check, object_inertia_plausibility_check, friction_parameter_check, "
        "free_body_policy_check, video_frustum_check, camera_registration_check, collision_visual_alignment_check, "
        "object_count_vs_stage1_check, articulation_axis_semantics_check, appearance_texture_check, asset_scene_consistency_check, "
        "deformable_topology_check and deformable_rest_check (when a deformable is a MuJoCo cable: one a subgoal cuts), "
        "isaac_deformable_rest_check (when a deformable is simulated in Isaac Sim: its cord specification, settled poses, "
        "settling report and placement re-measured from the files), hand_object_contact_consistency_check (when the "
        "workspace has metric hand joints: hands.npz in the workspace, egocam/ or episode/, or scene_recon's hands_npz), object_scale_vs_footprint_check (when masks exist; "
        "an Isaac Sim cord is drawn as a tube along its settled centreline; "
        "objects declared hidden_at_frame0 have no frame-0 footprint and are left out of it and of camera_registration_check), and the scene "
        "compile checks listed in COMPILE_CHECKS. camera_hand_overlay_check is reported as a warning only (demonstrators' "
        "hands rarely rest exactly at the robot's home pose). A check that crashes counts as failed. Each check's full "
        "report is returned under `reports`."
    ),
    "inputs": [
        {"name": "workspace", "type": "str", "description": "Stage 2 workspace holding scene_recon.json."},
        {"name": "scene_semantic_json", "type": "str", "description": "Stage 1 deliverable."},
        {"name": "setup_dir", "type": "str | None", "description": "Setup folder the robot and physics must match; default the repository's setup/."},
        {"name": "skip", "type": "list[str] | None", "description": "Check names to leave out (reported as skipped; for a custom setup without some data)."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok, issues (prefixed by check name), warnings, reports {check: report}, skipped, seconds {check: s}."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo", "Isaac Sim"],
    "tags": ["stage2", "finish", "gate", "suite", "scene", "anti-tampering"],
    "status": "tested",
    "entrypoint": "stage2_gate_suite",
}

import importlib
import json
import time
from pathlib import Path

# scene compile checks on every Stage 2 scene: name -> blocking (asset paths only warn here: a run's scene may point into
# the repository; the Stage 4 task folder and saved examples must be portable and are checked there)
COMPILE_CHECKS = {"mesh_validity_check": True, "inertia_validity_check": True, "name_uniqueness_check": True,
                  "keyframe_dimension_check": True, "equality_constraint_check": True, "solver_parameter_sanity_check": True,
                  "contact_margin_gap_check": True, "simulation_stability_check": True, "actuator_mapping_check": True,
                  "joint_armature_damping_check": True, "texture_material_resolution_check": True, "geom_count_budget_check": True,
                  "compile_determinism_check": True, "mujoco_version_compat_check": True, "free_joint_object_check": True,
                  "scene_size_budget_check": True, "asset_path_portability_check": False}
WITH_OBJECTS = {"name_uniqueness_check", "geom_count_budget_check", "free_joint_object_check"}
WITH_REFERENCE = {"keyframe_dimension_check", "actuator_mapping_check", "joint_armature_damping_check"}


def _call(name, kwargs):
    mod = importlib.import_module(f"tool_library.verifiers.{name}")
    return getattr(mod, name)(**kwargs)


def task_reach_requirements(semantic, reconstruction):
    """Resolve acting-arm prerequisites from frozen semantics, including destinations."""
    roots = {o.get('id'): o.get('root_body') for o in reconstruction.get('objects', [])
             if o.get('type') != 'deformable' and o.get('root_body')}
    needed = {}
    for step in semantic.get('subgoals', []):
        side = step.get('acting_hand')
        sides = ['left', 'right'] if side == 'both' else [side] if side in ('left', 'right') else []
        for ref in step.get('object_refs', []):
            body = roots.get(str(ref).split(':')[0])
            if body:
                needed.setdefault(body, set()).update(sides)
    return {body: sorted(sides) for body, sides in needed.items()}


def stage2_gate_suite(workspace, scene_semantic_json, setup_dir=None, skip=None):
    ws = Path(workspace)
    recon_p = ws / "scene_recon.json"
    if not recon_p.exists():
        return {"ok": False, "issues": ["scene_recon.json is missing"], "warnings": [], "reports": {}, "skipped": {}, "seconds": {}}
    recon = json.loads(recon_p.read_text())
    scene = str((ws / recon.get("scene_xml", "scene.xml")).resolve())
    camera = ws / recon.get("camera_json", "camera.json")
    sem_p = Path(scene_semantic_json)
    sem = json.loads(sem_p.read_text()) if sem_p.exists() else {}
    setup = str(setup_dir) if setup_dir else None
    skip = set(skip or [])
    objs = recon.get("objects", [])
    # an object hidden at frame 0 has no frame-0 footprint to measure (its empty mask is checked by scene_qc and provenance)
    from tool_library.common.hidden_objects import is_hidden
    masks = {o["id"]: str(ws / o["mask"]) for o in objs if o.get("mask") and (ws / o["mask"]).exists() and o.get("type") != "deformable"
             and not is_hidden(o)}
    hands_mask = ws / "masks" / "hands.png"
    # metric hand joints: where the manifest says, else where read_egodex_episode / place_egodex_camera_for_reach write them
    hands_npz = next((q for q in ([ws / recon["hands_npz"]] if recon.get("hands_npz") else []) +
                      [ws / "hands.npz", ws / "egocam" / "hands.npz", ws / "episode" / "hands.npz"] +
                      sorted(ws.glob("*/hands.npz"), key=lambda q: -q.stat().st_mtime) if q.exists()), ws / "hands.npz")
    # deformables simulated in Isaac Sim are re-measured from their cord files (isaac_deformable_rest_check); MuJoCo cables
    # (a cord a subgoal cuts) by the MuJoCo topology and rest checks, as is a Stage 1 deformable without any entry
    from tool_library.common.scene_assert import is_isaac_only
    isaac_deformables = [o for o in objs if o.get("type") == "deformable" and is_isaac_only(o)]
    entered = {o.get("id") for o in objs}
    has_deformable = any(o.get("type") == "deformable" and not is_isaac_only(o) for o in objs) or \
        any(o.get("type") == "deformable" and o.get("id") not in entered for o in sem.get("objects", []))
    footprint_masks = dict(masks, **{o["id"]: str(ws / o["mask"]) for o in isaac_deformables if o.get("mask") and (ws / o["mask"]).exists()})
    S = {"setup": setup} if setup else {}

    plan = [  # (check, kwargs, blocking)
        ("robot_model_integrity_check", {"scene": scene, **S}, True),
        ("robot_joint_range_check", {"scene": scene, **S}, True),
        ("actuator_limit_integrity_check", {"scene": scene, **S}, True),
        ("contact_exclusion_audit", {"scene": scene, **S}, True),
        ("physics_options_check", {"scene": scene, **S}, True),
        # A solver's own station-shift report is not authorization to change the
        # supplied hardware setup. Custom layouts must be supplied as setup_dir.
        ("stage_station_integrity_check", {"scene": scene, **S, "unrecorded_shift_ok": False}, True),
        ("initial_keyframe_state_check", {"scene": scene, **S}, True),
        ("hand_contact_pair_mask_check", {"scene": scene}, True),
        ("object_mass_plausibility_check", {"scene": scene, "recon": str(recon_p), "semantic": str(sem_p)}, True),
        ("object_inertia_plausibility_check", {"scene": scene, "recon": str(recon_p)}, True),
        ("friction_parameter_check", {"scene": scene, **S}, True),
        ("free_body_policy_check", {"scene": scene, "semantic": str(sem_p), "recon": str(recon_p)}, True),
        ("collision_visual_alignment_check", {"scene": scene, "recon": str(recon_p)}, True),
        ("object_count_vs_stage1_check", {"scene": scene, "semantic": str(sem_p), "recon": str(recon_p)}, True),
        ("articulation_axis_semantics_check", {"scene": scene, "semantic": str(sem_p), "recon": str(recon_p), **({"camera": str(camera)} if camera.exists() else {})}, True),
        ("appearance_texture_check", {"scene": scene, "recon": str(recon_p)}, True),
        ("asset_scene_consistency_check", {"scene": scene, "recon": str(recon_p)}, True),   # Stage 3 builds hand scenes from the assets
    ]
    reach = task_reach_requirements(sem, recon)
    if reach:
        plan.append(("object_reach_check", {"scene": scene, "objects": list(reach), "required_sides": reach, "requirement_mode": "necessary"}, True))
    if camera.exists():
        plan.append(("video_frustum_check", {"scene": scene, "semantic": str(sem_p), "recon": str(recon_p), "camera": str(camera)}, True))
        plan.append(("camera_registration_check", {"camera": str(camera), "scene": scene, **({"masks": masks} if masks else {}), **S}, True))
        if hands_mask.exists():
            plan.append(("camera_hand_overlay_check", {"camera": str(camera), "hands_mask": str(hands_mask), "scene": scene, **S}, False))
        if footprint_masks:
            plan.append(("object_scale_vs_footprint_check", {"scene": scene, "camera": str(camera), "masks": footprint_masks, "recon": str(recon_p),
                                                             **({"ignore_mask": str(hands_mask)} if hands_mask.exists() else {})}, True))
    if has_deformable:
        plan.append(("deformable_topology_check", {"scene": scene, "recon": str(recon_p)}, True))
        plan.append(("deformable_rest_check", {"scene": scene, "recon": str(recon_p)}, True))
    if isaac_deformables:
        plan.append(("isaac_deformable_rest_check", {"scene": scene, "recon": str(recon_p), **({"camera": str(camera)} if camera.exists() else {}),
                                                     **({"ignore_mask": str(hands_mask)} if hands_mask.exists() else {})}, True))
    if hands_npz.exists():
        plan.append(("hand_object_contact_consistency_check", {"scene": scene, "hands_npz": str(hands_npz), "scene_semantic_json": str(sem_p),
                                                               "scene_recon_json": str(recon_p)}, True))
    ref_scene = str(Path(setup) / "scene.xml") if setup else None
    for name, blocking in COMPILE_CHECKS.items():
        kw = {"scene": scene}
        if name in WITH_OBJECTS:
            kw["objects"] = str(recon_p)
            if name == "name_uniqueness_check" and isaac_deformables:
                # a cord simulated in Isaac Sim has no MuJoCo body to find by name
                kw["objects"] = {"objects": [o for o in objs if o not in isaac_deformables]}
        if name in WITH_REFERENCE and ref_scene:
            kw["reference"] = ref_scene
        plan.append((name, kw, blocking))
    usd = next(iter(sorted((ws / "isaac").glob("*.usd*"))), None) if (ws / "isaac").is_dir() else None
    if usd is not None:
        plan.append(("isaac_export_check", {"usd": str(usd), "scene": scene}, True))

    from agent.reference_evidence import check_workspace
    source_report = check_workspace(ws)
    issues = [f"source_reference_provenance_check: {i}" for i in source_report.get("issues", [])]
    warnings, reports, skipped, seconds = [], {"source_reference_provenance_check": source_report}, {}, {}
    if skip:
        issues.append("checks skipped by request: diagnostic evaluation cannot certify Stage 2")
    if not camera.exists():
        skipped.update({n: "no camera.json in the workspace" for n in ("video_frustum_check", "camera_registration_check", "object_scale_vs_footprint_check")})
        warnings.append("no camera.json: the camera checks were skipped (scene_recon_manifest_check requires one)")
    for name, kwargs, blocking in plan:
        if name in skip:
            skipped[name] = "skipped by request"; continue
        t0 = time.time()
        try:
            r = _call(name, kwargs)
        except Exception as e:                       # a check that cannot run on this scene is a failed check
            r = {"ok": False, "issues": [f"crashed: {type(e).__name__}: {e}"]}
        seconds[name] = round(time.time() - t0, 2)
        reports[name] = r
        msgs = [f"{name}: {i}" for i in (r.get("issues") or ([] if r.get("ok") else ["failed"]))]
        if r.get("ok") is not True:
            (issues if blocking else warnings).extend(msgs)
        warnings.extend(f"{name}: {w}" for w in (r.get("warnings") or []))
    for row in (reports.get("free_body_policy_check") or {}).get("objects", []) or []:
        if row.get("anchored_to"):
            warnings.append(f"free_body_policy_check: {row['id']} is welded to its declared anchor {row['anchored_to']} "
                            f"(scene_recon.json); Stage 1 must describe it so")
    for row in (reports.get("deformable_topology_check") or {}).get("ropes", []) or []:
        if row.get("declared"):
            warnings.append(f"deformable_topology_check: {row['id']} is held to its declared topology {row['declared']} "
                            f"(scene_recon.json); Stage 1 must describe it so")
    if has_egodex_hands(ws) and not hands_npz.exists():
        warnings.append("hand_object_contact_consistency_check skipped: the workspace has an EgoDex episode but no hands.npz")
    return {"ok": not issues, "issues": issues, "warnings": warnings, "reports": reports, "skipped": skipped, "seconds": seconds,
            "scene": scene, "hands_npz": str(hands_npz) if hands_npz.exists() else None}


def has_egodex_hands(ws):
    return (ws / "episode.json").exists() or any(ws.glob("*/episode.json"))
