"""Read a failed Stage 3 attempt and say what to do next: failure classes with their evidence, ranked concrete repairs."""
from __future__ import annotations

TOOL = {
    "name": "diagnose_stage3_failure",
    "title": "Diagnose a Stage 3 failure",
    "type": "skill",
    "category": "Trajectory & Execution",
    "summary": "Reads a failed finish verdict (or failure_report.json / gate_suite_report.json), the executor's outcomes and the "
               "workspace's poses, classifies the failure (unreachable, no stress-test pass, slip in carry, hand-hand or arm "
               "collision, joint speed, goal not met, Stage 4 cannot randomize, physics audit, upstream suspect, thin object, "
               "needs a steadying hand) with the evidence for each, and returns ranked concrete repairs: library tool names "
               "with arguments filled from the workspace, report_blocked with upstream for an earlier stage's error.",
    "description": (
        "The Stage 3 harness appends this diagnosis to every failed finish verdict, so the agent always gets next steps; the "
        "agent may also call it on an executor failure. Sources, all optional: the verdict ({ok, issues, reports}: the "
        "manifest check and stage3_gate_suite findings, or a failure_report.json with last_verdict / last_finish and the "
        "blocked reason, or a gate_suite_report.json), the newest executor outcomes.json files in the workspace (each failed "
        "entry's failed checks and measurements), the manifest, and the pose files. Text findings are matched against each "
        "class's vocabulary (gate names and the executor's check names, e.g. arm_unreachable, plan_collision, arm_speed, "
        "no_slip, stage4_physics_replay_gate). Executor checks are classed by the motion they failed in: rise / in_hand / "
        "pads_touching / servo_tracking in a grasp are a grasp that does not hold (no stress-test pass), in_hand or no_slip "
        "in a carry is slip, a joint that did not travel while the object's base moved, or a disturbed body of the same "
        "object, needs a steadying hand. Two classes are MEASURED from the scene, not read: thin_object (the top of the part "
        "the failing grasp is on, or of each manipulated object when nothing executed, is lower above the table than two "
        "fingertip pad radii of the setup hand plus 2 mm, so the pads cannot get under it) and upstream_suspect (a hand scene "
        "in poses/ whose object bodies differ from the bench scene's same bodies in collision geoms or extent: the Stage 2 "
        "asset the grasp tools see is not the object the robot meets; human prior warnings that the demonstration and the "
        "reconstruction disagree; a consistency check that says an earlier stage must be revised). Each class gets a score "
        "(measured evidence weighs most, the latest outcomes next, gate findings and the blocked reason after) and its "
        "repairs: optimize_grasp_pose with the failing pose as the start, human_grasp_seeds, optimize_pose_for_reach, "
        "whole-motion reach screening, a slower leg and execute_subgoal_plan(resume=True, from_subgoal=k), a park of the idle arm, a "
        "steadying hold first, generate_flat_pinches or a non-prehensile push for a thin part, report_blocked(upstream=...) "
        "with independently measured evidence. Finite failed grasp/contact searches do not establish unreachable upstream "
        "geometry. Preserve the configured stations and task goals. It changes nothing; it reads and advises."
    ),
    "inputs": [
        {"name": "workspace", "type": "path | None", "description": "Stage 3 workspace (failure_report.json, gate_suite_report.json, trajectory_manifest.json, outcomes.json files, poses/, plans)."},
        {"name": "verdict", "type": "dict | path | None", "description": "A finish verdict {ok, issues, reports}, or a failure_report.json / gate_suite_report.json / run.json path, a .jsonl log of finish verdicts (its last line), or a dict; default the workspace's failure_report.json, else its gate_suite_report.json."},
        {"name": "reason", "type": "str | None", "description": "A blocked reason to read as well (default the failure report's)."},
        {"name": "outcomes", "type": "path | list | None", "description": "Executor outcomes.json file(s) or entries; default the 3 newest under the workspace."},
        {"name": "scene_semantic_json", "type": "path | None", "description": "Stage 1 scene_semantics.json (objects, subgoals)."},
        {"name": "scene_recon_json", "type": "path | None", "description": "Stage 2 scene_recon.json (object root bodies; its scene_xml is the default bench scene)."},
        {"name": "scene", "type": "path | None", "description": "Bench scene the episode ran in (default the manifest's scene_executed, else the recon's scene_xml)."},
        {"name": "setup_dir", "type": "path | None", "description": "Setup folder (the hand's pad radius); default the repository's setup/."},
        {"name": "check_hand_scenes", "type": "bool", "description": "Compare the hand scenes in poses/ with the bench scene (Stage 2 asset vs scene). Default true."},
        {"name": "include_workspace_history", "type": "bool", "description": "Read prior gate reports and pose warnings as additional evidence. Default true; false diagnoses only the supplied verdict/outcomes, as used for a current executor failure."},
        {"name": "plan", "type": "dict | list | path | None", "description": "The exact plan passed to the failed executor, including inline plans. Takes precedence over workspace history; matches the failing index and step so repair suggestions retain its mechanism and grasp."},
        {"name": "max_repairs", "type": "int", "description": "Default 8."},
    ],
    "outputs": [
        {"name": "classes", "type": "list", "description": "[{class, score, evidence: [...], subgoals, objects}] most likely first."},
        {"name": "repairs", "type": "list", "description": "[{rank, class, tool, args, why}]: concrete next calls (args filled where the workspace says)."},
        {"name": "upstream", "type": "dict | None", "description": "{stage: 'stage1' | 'stage2', evidence} when an earlier stage is suspect."},
        {"name": "summary", "type": "str", "description": "A few lines for the agent (what the harness appends to a failed verdict)."},
        {"name": "sources", "type": "list", "description": "The files read."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "diagnosis", "repair", "failure", "agent search"],
    "used_in": [],
    "status": "tested",
    "entrypoint": "diagnose_stage3_failure",
}

import json
import re
from pathlib import Path

import numpy as np

DEFAULT_SETUP = Path(__file__).resolve().parents[2] / "setup"
CLASSES = ("upstream_suspect", "thin_object", "needs_steadying_hand", "unreachable", "no_stress_pass", "slip_in_carry", "collision",
           "joint_speed", "physics_audit", "stage4_randomize", "goal_not_met", "state_requires_physics", "deformable_topology")
GRASP_MOTIONS = ("grasp_lift", "hold", "regrasp", "pull", "open_joint", "close_joint", "unscrew", "screw", "unscrew_bite", "press_seat",
                 "pin", "grasp_more", "lift_together")
CARRY_MOTIONS = ("carry", "place", "insert", "pour", "tilt", "lift", "cut")
# vocabulary of each class in gate findings and blocked reasons (lower case); the first class that matches a finding takes it
ISSUE_PATTERNS = [
    ("state_requires_physics", r"state_requires_physics"),
    ("deformable_topology", r"rope_knot_check|\bunknot\b|\btrefoil\b|\bknot\b|\btopolog(?:y|ical)\b|loop_aperture|endpoint_passage"),
    ("upstream_suspect", r"must be revised|re-run stage|revise stage|stage ?[12] (defect|revision|must|is wrong|error)|needs stage ?[12]|"
                         r"misplaced|mis-?scaled|hand_object_contact_consistency|disagree|asset_xml|occlusion as a pick|not in the demonstration"),
    ("stage4_randomize", r"stage4_randomization_gate|stage4_physics_replay_gate|stage4_handoff_check|stage4\b"),
    ("joint_speed", r"arm_speed|robot_joint_jump_audit|joint_acceleration_audit|hand_joint_rate_audit|arm_branch_flip_audit|"
                    r"wrist_singularity_audit|too fast|races|flip(s|ped)? (its|the) configuration"),
    ("unreachable", r"arm_unreachable|reach_turn_too_large|cannot hold the hand on its flange|out of reach|unreachable|object_reach_check|"
                    r"arm_weld|stage_station_integrity_check|arm_reach_check"),
    ("collision", r"arm_collision|hand_clearance|plan_collision|hands_apart|robot_self_collision_check|swept_collision_audit|runs into|"
                  r"through each other|hand.{0,20}into the"),
    ("slip_in_carry", r"no_slip|slip|held_object_slip_audit|hand_object_relative_jitter|grasp_shake_test|loses the grasp|drops"),
    ("no_stress_pass", r"no passing stress test|stress test|grasp_lift_test|seated|no pose|no grasp|pads_touching|\b0/\d+"),
    ("physics_audit", r"deformable_integrity_audit|episode_physics_check|object_teleport_audit|contactless_motion_audit|"
                      r"free_fall_plausibility_audit|dynamics_consistency_audit|contact_impulse_spike_audit|object_visual_penetration_audit|"
                      r"badqacc|finger_servo_saturation_audit|start_state_audit|initial_keyframe_state_check|released_contact_audit"),
    ("goal_not_met", r"stage1_goal_gate|not covered|no executed manifest entry|skipped|outcome not ok|at_target|joint_reached|turned|knot|"
                     r"severed|displaced|released|end_state_rest_audit|bimanual_sync_gate|missing|uncovered|timing_realism|goal"),
]
# extra cues read from a blocked reason (a reason is free text: every class it names counts)
REASON_PATTERNS = {
    "deformable_topology": r"rope_knot_check|\bunknot\b|\btrefoil\b|\bknot\b|\btopolog(?:y|ical)\b|loop_aperture|endpoint_passage",
    "thin_object": r"\bthin\b|lying flat|flat strip|pad radi|pads? (cannot|can't) get under|box links|mm thick|\d+(\.\d+)? ?mm (tall|high|thick)",
    "needs_steadying_hand": r"steady|anchor|drag|slid(es|ing)?\b|tip(s|ping)? over|base_moved|force went into the base|hold the base|"
                            r"re-?tangle|nothing (anchors|holds)",
    "upstream_suspect": r"stage ?[12] (defect|error|must|revision)|needs stage ?[12] revision|misplaced|mis-?scaled|occlusion",
    "unreachable": r"unreachable|out of reach|cannot reach|reach",
    "no_stress_pass": r"no pose|no grasp|stress test|0/\d+|seated|cannot synthesize",
    "collision": r"collid|runs into|hull into|through the",
}
EXECUTOR_MAP = {"arm_unreachable": "unreachable", "reach_turn_too_large": "unreachable", "arm_weld": "unreachable",
                "knot": "deformable_topology",
                "state_requires_physics": "state_requires_physics",
                "arm_collision": "collision", "hand_clearance": "collision", "plan_collision": "collision", "hands_apart": "collision",
                "hand_penetration": "collision", "arm_speed": "joint_speed", "reference_too_long": "joint_speed",
                "object_penetration": "physics_audit", "no_slip": "slip_in_carry"}
GOAL_CHECKS = ("at_target", "joint_reached", "turned", "yaw_turned", "tilted", "stays_tilted", "knot", "severed", "cut_ends_apart",
               "displaced", "seated", "released", "at_rest", "supported", "upright", "stays_put", "thread_released", "joint_moved", "planning")


def _load(x):
    if x is None:
        return None
    if isinstance(x, (str, Path)):
        p = Path(x)
        if not p.exists():
            return None
        if p.suffix == ".jsonl":
            lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
            return json.loads(lines[-1]) if lines else None
        return json.loads(p.read_text())
    return x


def _issues_of(v):
    """(issues, reports ok-map, blocked reason) from any of the verdict formats."""
    if not isinstance(v, dict):
        return [], {}, None
    reason = v.get("reason") if isinstance(v.get("reason"), str) else None
    if "last_verdict" in v and isinstance(v["last_verdict"], dict):
        iss, rep, _ = _issues_of(v["last_verdict"])
        return iss, rep, reason
    if "verdict" in v and isinstance(v["verdict"], dict):
        iss, rep, r2 = _issues_of(v["verdict"])
        return iss, rep, reason or r2
    issues = list(v.get("issues") or []) + [str(x) for x in (v.get("last_finish") or [])]
    reports = {}
    for name, r in (v.get("reports") or {}).items():
        if isinstance(r, dict):
            reports[name] = r.get("ok")
            sub = r.get("report")
            if isinstance(sub, dict):
                reports.update({k: (x.get("ok") if isinstance(x, dict) else x) for k, x in sub.items()})
        else:
            reports[name] = r
    return [str(i) for i in issues], reports, reason


def _subgoal_of(text):
    m = re.search(r"subgoal \[?(\d+)", text)
    return int(m.group(1)) if m else None


def _newest(ws, pattern, n):
    try:
        fs = [p for p in Path(ws).glob(pattern) if p.is_file()]
    except OSError:
        return []
    return sorted(fs, key=lambda p: p.stat().st_mtime, reverse=True)[:n]


def _pad_radius(setup_dir):
    import mujoco
    from tool_library.common import hand as H
    xml = Path(setup_dir or DEFAULT_SETUP) / "robot" / "sharpa_right.xml"
    m = mujoco.MjModel.from_xml_path(str(xml))
    return float(H.pad_radius(m, H.HandRig.build(m, "", "right")))


def _body_top(model, data, bid):
    """Highest point (world z) of a body's own collision geoms (its subtree when it has none)."""
    from tool_library.common import episode as E
    gs = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == bid and (int(model.geom_contype[g]) or int(model.geom_conaffinity[g]))]
    if not gs:
        sub = E.subtree_bodies(model, bid)
        gs = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in sub and (int(model.geom_contype[g]) or int(model.geom_conaffinity[g]))]
    if not gs:
        return None
    tops = []
    for g in gs:
        R = np.asarray(data.geom_xmat[g], float).reshape(3, 3)
        c = np.asarray(data.geom_xpos[g], float) + R @ np.asarray(model.geom_aabb[g, :3], float)
        tops.append(float(c[2] + (np.abs(R) @ np.asarray(model.geom_aabb[g, 3:], float))[2]))
    return max(tops)


def _thin_objects(scene_xml, bodies, pad_r):
    """{body: top height above the table (m)} of the bodies whose top is below two pad radii + 2 mm (the table is z = 0)."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(scene_xml))
    d = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    out = {}
    for b in bodies:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, str(b))
        if bid < 0:
            continue
        top = _body_top(m, d, bid)
        if top is not None and top < 2.0 * pad_r + 0.002:
            out[str(b)] = round(top, 4)
    return out


def _geom_signature(model, bid):
    """(collision geom count, extent [3] in the body's frame of its own collision geoms) of one body."""
    import mujoco
    gs = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == bid and (int(model.geom_contype[g]) or int(model.geom_conaffinity[g]))]
    if not gs:
        gs = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == bid]
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in gs:
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.asarray(model.geom_quat[g], float))
        R = R.reshape(3, 3)
        c = np.asarray(model.geom_pos[g], float) + R @ np.asarray(model.geom_aabb[g, :3], float)
        h = np.abs(R) @ np.asarray(model.geom_aabb[g, 3:], float)
        lo, hi = np.minimum(lo, c - h), np.maximum(hi, c + h)
    return len(gs), ((hi - lo) if gs else np.zeros(3))


def _hand_scene_mismatch(hs_files, scene_xml, limit=8):
    """Bodies of a hand scene's objects whose collision geometry differs from the same body in the bench scene."""
    import mujoco
    from tool_library.common import hand as H
    bm = mujoco.MjModel.from_xml_path(str(scene_xml))
    found, seen = [], set()
    for f in hs_files[:limit]:
        try:
            spec = H.load_scene_spec(str(f))
            if not isinstance(spec, dict) or "objects" not in spec or "hand_xml" not in spec:
                continue
            key = json.dumps([o.get("xml") for o in spec["objects"]], sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            model, data0, rig, ids = H.open_scene(spec)
        except Exception:
            continue
        hand_bodies = set(rig.body_ids)
        for b in range(1, model.nbody):
            if b in hand_bodies:
                continue
            name = model.body(b).name
            bb = mujoco.mj_name2id(bm, mujoco.mjtObj.mjOBJ_BODY, name)
            if not name or bb < 0:
                continue
            n1, e1 = _geom_signature(model, b)
            n2, e2 = _geom_signature(bm, bb)
            if n1 != n2 or float(np.max(np.abs(e1 - e2))) > 0.003:
                found.append({"hand_scene": str(f), "body": name, "hand_scene_geoms": n1, "bench_geoms": n2,
                              "hand_scene_extent_mm": [round(1000 * float(v), 1) for v in e1],
                              "bench_extent_mm": [round(1000 * float(v), 1) for v in e2]})
    return found


def _latest_failed_outcomes(paths):
    out = []
    for p in paths:
        recs = _load(p)
        if isinstance(recs, dict):
            recs = recs.get("subgoals") or [recs]
        for r in recs or []:
            if isinstance(r, dict) and r.get("ok") is False:
                out.append(dict(r, _file=str(p)))
    return out


def _entry_from_plan(plan, index, step=None, side=None):
    subs = plan.get("subgoals", []) if isinstance(plan, dict) else plan
    for entry in subs if isinstance(subs, list) else []:
        if not isinstance(entry, dict) or str(entry.get("index")) != str(index):
            continue
        if step is not None and entry.get("step") is not None and str(entry["step"]) != str(step):
            continue
        if isinstance(entry.get("hands"), dict) and side in entry["hands"]:
            return {**entry, **entry["hands"][side], "side": side}
        return entry
    return None


def _plan_entry(ws, index, step=None, side=None):
    """The newest plan json in the workspace that has an entry for this subgoal index."""
    if ws is None or index is None:
        return None, None
    for p in _newest(ws, "plan*.json", 6) + _newest(ws, "plans/*.json", 6):
        try:
            plan = json.loads(p.read_text())
        except Exception:
            continue
        entry = _entry_from_plan(plan, index, step, side)
        if entry is not None:
            return entry, str(p)
    return None, None


def _split_hand_failures(records, plan=None, workspace=None):
    """Resolve each failed hand against its own primitive and measured state.

    The executor deliberately keeps a simultaneous entry in one record. Repair
    tools take one concrete side/pose, so stripping a check's `left:` prefix
    without retaining that context produced invalid `side='both'` searches.
    Unprefixed arm failures use the measured arm identity when available.
    """
    supplied = _load(plan) if plan is not None else None
    out = []
    for record in records:
        if record.get("side") != "both" and not isinstance(record.get("object"), dict):
            out.append(record)
            continue
        parent = (_entry_from_plan(supplied, record.get("index"), record.get("step")) if supplied is not None else
                  _plan_entry(workspace, record.get("index"), record.get("step"))[0]) or {}
        hands = parent.get("hands") or {}
        objects = record.get("object") if isinstance(record.get("object"), dict) else {}
        measured = record.get("measured") or {}
        implicated = {row.get("side") for row in measured.get("infeasible", []) if isinstance(row, dict)}
        implicated.update((measured.get("branch_jumps") or {}).keys())
        for hit in measured.get("arm_collisions", []):
            for side in ("left", "right"):
                if any(f"robot_{side}_" in str(g) for g in hit.get("geoms", [])):
                    implicated.add(side)
        available = [side for side in ("left", "right") if side in hands or side in objects]
        inferred = [side for side in available if side in implicated] or available
        checks = {}
        for check in record.get("failed") or []:
            prefix, _, remainder = str(check).partition(":")
            sides = [prefix] if prefix in ("left", "right") else inferred
            for side in sides:
                checks.setdefault(side, []).append(remainder if prefix == side else str(check))
        if not checks:
            out.append(record)
            continue
        for side, failures in checks.items():
            entry = {**parent, **(hands.get(side) or {}), "side": side}
            motion = entry.get("motion") or record.get("motion")
            # Shared-object primitives omit a separate motion inside hands.
            if motion == "lift_together":
                motion = "grasp_lift"
            elif motion == "carry_together":
                motion = "carry"
            local = {k: v for k, v in measured.items() if k != "hands"}
            local.update((measured.get("hands") or {}).get(side) or {})
            out.append({**record, "side": side, "motion": motion,
                        "object": objects.get(side) or entry.get("object"),
                        "failed": failures, "measured": local, "_plan_entry": entry,
                        "_parent_motion": record.get("motion")})
    return out


def diagnose_stage3_failure(workspace=None, verdict=None, reason=None, outcomes=None, scene_semantic_json=None, scene_recon_json=None,
                            scene=None, setup_dir=None, check_hand_scenes=True, max_repairs=8, include_workspace_history=True, plan=None):
    ws = Path(workspace) if workspace else None
    sources = []
    v = _load(verdict)
    if v is None and ws is not None and include_workspace_history:
        for name in ("failure_report.json", "gate_suite_report.json"):
            if (ws / name).exists():
                v = _load(ws / name)
                sources.append(str(ws / name))
                break
    elif isinstance(verdict, (str, Path)):
        sources.append(str(verdict))
    issues, reports, reason0 = _issues_of(v)
    if include_workspace_history and ws is not None and (ws / "gate_suite_report.json").exists() and str(ws / "gate_suite_report.json") not in sources:
        g_iss, g_rep, _ = _issues_of(_load(ws / "gate_suite_report.json"))
        issues += [i for i in g_iss if i not in issues]
        reports.update({k: x for k, x in g_rep.items() if k not in reports})
        sources.append(str(ws / "gate_suite_report.json"))
    reason = reason or reason0
    if reason and reason.startswith("blocked: "):
        reason = reason[len("blocked: "):]
    ev = {c: [] for c in CLASSES}
    score = {c: 0.0 for c in CLASSES}
    subgoals = {c: set() for c in CLASSES}
    objects = {c: set() for c in CLASSES}
    class_failures = {c: [] for c in CLASSES}
    current_failure = None

    def add(c, text, w, sg=None, obj=None):
        if len(ev[c]) < 8:
            ev[c].append(text[:300])
        score[c] += w
        if sg is not None:
            subgoals[c].add(int(sg))
        if obj:
            objects[c].add(str(obj))
        if current_failure is not None and not any(r is current_failure for r in class_failures[c]):
            class_failures[c].append(current_failure)

    # 1. the verdict's findings: a gate counts fully once per class, its further findings a quarter each
    seen_gate = set()
    for text in issues:
        low = text.lower()
        gate = (re.match(r"\s*([a-z0-9_]+(?:\[\d+\])?):", low) or re.match(r"\s*(subgoal \[?\d+\]?)", low))
        gate = gate.group(1) if gate else low[:30]
        for c, pat in ISSUE_PATTERNS:
            if re.search(pat, low):
                if c == "unreachable" and "arm_reach_check" in low and re.search(r"race|speed|flip", low):
                    c = "joint_speed"
                add(c, text, 0.25 if (c, gate) in seen_gate else 1.0, _subgoal_of(text))
                seen_gate.add((c, gate))
                break
    # 2. the blocked reason (free text: every class it names counts)
    if reason:
        low = reason.lower()
        for c, pat in REASON_PATTERNS.items():
            hits = re.findall(pat, low)
            if hits:
                add(c, f"blocked reason: ...{_snippet(reason, pat)}...", min(3.0, 1.0 * len(hits)))
    # 3. the executor's outcomes
    files = []
    if outcomes is not None:
        files = outcomes if isinstance(outcomes, list) and outcomes and not isinstance(outcomes[0], dict) else [outcomes]
    elif ws is not None:
        files = [str(p) for p in _newest(ws, "**/outcomes.json", 3)]
    sources += [str(f) for f in files if isinstance(f, (str, Path))]
    failed = []
    for f in files:
        failed += _latest_failed_outcomes([f]) if not isinstance(f, list) else [r for r in f if isinstance(r, dict) and r.get("ok") is False]
    n_failed_outcomes = len(failed)
    failed = _split_hand_failures(failed, plan, ws if include_workspace_history else None)
    for k, r in enumerate(failed):
        current_failure = r
        w = 1.5 if k < 3 else 0.5
        motion = str(r.get("motion") or "")
        sg, obj, side = r.get("index"), r.get("object"), r.get("side")
        meas = r.get("measured") or {}
        checks = [str(c) for c in (r.get("failed") or [])]
        for chk in checks:
            base = chk.split(":", 1)[-1] if ":" in chk and chk.split(":", 1)[0] in ("left", "right") else chk
            tag = f"subgoal {sg} {motion} ({side}, {obj}): {base}"
            if base in EXECUTOR_MAP:
                add(EXECUTOR_MAP[base], tag + (f" - {str(r.get('error'))[:160]}" if r.get("error") else ""), w, sg, obj)
            elif base.startswith("untouched"):
                body = base.split(":", 1)[1] if ":" in base else ""
                same = bool(obj) and body.split("_")[0] == str(obj).split("_")[0]
                add("needs_steadying_hand" if same else "collision", tag + (" (a body of the same object moved: steady it)" if same else
                                                                             " (a bystander was disturbed)"), w, sg, obj)
            elif (base in ("rise", "in_hand", "pads_touching", "servo_tracking") or base.startswith("link_in_hand:")) and motion in GRASP_MOTIONS:
                add("no_stress_pass", tag + " (the grasp did not hold on the bench)", w, sg, obj)
            elif (base == "in_hand" or base.startswith("link_in_hand:")) and motion in CARRY_MOTIONS:
                add("slip_in_carry", tag, w, sg, obj)
            elif base == "servo_tracking":
                add("collision", tag + " (the hand was driven into something)", w, sg, obj)
            elif base in GOAL_CHECKS:
                add("goal_not_met", tag, w, sg, obj)
            else:
                add("goal_not_met", tag, 0.5 * w, sg, obj)
        if meas.get("base_moved") and any(c in checks for c in ("joint_reached", "joint_moved", "turned")):
            add("needs_steadying_hand", f"subgoal {sg} {motion}: the joint did not travel while the base moved "
                                        f"{meas['base_moved']}: the force went into the base", 2.0, sg, obj)
    current_failure = None
    # 4. measured: thin parts and Stage 2 assets that differ from the bench
    man = _load(ws / "trajectory_manifest.json") if ws is not None and (ws / "trajectory_manifest.json").exists() else None
    recon = _load(scene_recon_json) if scene_recon_json else None
    sem = _load(scene_semantic_json) if scene_semantic_json else None
    scene_xml = scene
    if scene_xml is None and isinstance(man, dict) and man.get("scene_executed") and ws is not None and (ws / man["scene_executed"]).exists():
        scene_xml = str(ws / man["scene_executed"])
    if scene_xml is None and isinstance(recon, dict) and scene_recon_json:
        cand = Path(scene_recon_json).resolve().parent / recon.get("scene_xml", "scene.xml")
        scene_xml = str(cand) if cand.exists() else None
    grasp_trouble = score["no_stress_pass"] > 0 or bool(ev["thin_object"])
    if scene_xml and grasp_trouble:
        bodies = {str(o) for o in objects["no_stress_pass"] if o}
        if not bodies and isinstance(recon, dict):
            ids = None
            if isinstance(sem, dict):
                ids = {o.get("id") for o in sem.get("objects") or [] if o.get("role") in ("manipulated", "tool")}
            for o in recon.get("objects") or []:
                if ids is None or o.get("id") in ids:
                    bodies |= {str(b) for b in ([o.get("root_body")] + list(o.get("bodies") or [])) if b}
        try:
            pad_r = _pad_radius(setup_dir)
            for b, top in _thin_objects(scene_xml, sorted(bodies), pad_r).items():
                add("thin_object", f"{b}: its top is {1000 * top:.1f} mm above the table, below two pad radii + 2 mm "
                                   f"({1000 * (2 * pad_r + 0.002):.1f} mm): the pads cannot get under it", 3.0, None, b)
        except Exception as e:
            ev["thin_object"].append(f"(thickness not measured: {type(e).__name__}: {e})")
    if scene_xml and check_hand_scenes and ws is not None:
        hs_files = [p for p in _newest(ws, "poses/hs*.json", 12)] + [p for p in _newest(ws, "poses/*hand_scene*.json", 4)]
        try:
            for mm in _hand_scene_mismatch(hs_files, scene_xml):
                add("upstream_suspect", f"Stage 2 asset differs from the bench: body {mm['body']} has {mm['hand_scene_geoms']} collision geom(s), "
                                        f"extent {mm['hand_scene_extent_mm']} mm, in the hand scene ({Path(mm['hand_scene']).name}) but "
                                        f"{mm['bench_geoms']}, extent {mm['bench_extent_mm']} mm, in the bench scene: the grasp tools "
                                        "see another object than the robot meets", 3.5, None, mm["body"])
        except Exception as e:
            ev["upstream_suspect"].append(f"(hand scenes not compared: {type(e).__name__}: {e})")
    if ws is not None and include_workspace_history:
        for p in _newest(ws, "poses/*.json", 30):
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            ws_list = d.get("warnings") if isinstance(d, dict) else None
            if isinstance(d, dict) and "subgoals" in d and isinstance(d["subgoals"], list):
                ws_list = [w for s in d["subgoals"] if isinstance(s, dict) for w in (s.get("warnings") or [])]
            for w in ws_list or []:
                if re.search(r"disagree|misplaced", str(w).lower()):
                    add("upstream_suspect", f"{p.name}: {str(w)[:220]}", 1.0)
                    break
    # 5. rank, upstream stage, repairs
    ranked = sorted([c for c in CLASSES if score[c] > 0], key=lambda c: -score[c])
    upstream = None
    if "upstream_suspect" in ranked:
        txt = " ".join(ev["upstream_suspect"]).lower() + " " + (reason or "").lower()
        st1 = len(re.findall(r"stage ?1|occlusion|not in the demonstration|window|subgoal \d+ should", txt))
        st2 = len(re.findall(r"stage ?2|asset|bench scene|misplaced|mis-?scaled|reconstruct|placement", txt))
        # A blocked model's hypothesis is not an independent geometry check.
        verified = any(e.startswith("Stage 2 asset differs from the bench:") for e in ev["upstream_suspect"])
        upstream = {"stage": "stage1" if st1 > st2 else "stage2", "evidence": " | ".join(ev["upstream_suspect"][:3])[:900],
                    "verified": verified}
    repairs = []
    for c in ranked:
        contexts = class_failures[c] or [None]
        for failure in contexts:
            ctx = _context(ws, [failure] if failure else failed, recon, scene_recon_json, scene_semantic_json,
                           scene_xml, plan, include_workspace_history)
            for rp in _repairs(c, ctx, subgoals[c], objects[c], upstream, ev[c]):
                repairs.append(dict(rp, **{"class": c, "side": ctx["side"], "subgoal": ctx["index"]}))
    repairs = repairs[:int(max_repairs)]
    for i, r in enumerate(repairs):
        r["rank"] = i + 1
    classes = [{"class": c, "score": round(score[c], 2), "evidence": ev[c], "subgoals": sorted(subgoals[c]), "objects": sorted(objects[c])}
               for c in ranked]
    return {"classes": classes, "repairs": repairs, "upstream": upstream, "summary": _summary_text(classes, repairs, upstream),
            "n_issues": len(issues), "n_failed_outcomes": n_failed_outcomes, "sources": sources,
            "reach_search_status": "inconclusive" if score["unreachable"] else None}


def _snippet(text, pat, width=110):
    m = re.search(pat, text.lower())
    if not m:
        return text[:width]
    a = max(0, m.start() - width // 2)
    return text[a:a + width].replace("\n", " ")


def _context(ws, failed, recon, recon_path, sem_path, scene_xml, supplied_plan=None, include_workspace_history=True):
    ctx = {"scene": scene_xml or "<scene.xml>", "scene_recon_json": str(recon_path) if recon_path else "<scene_recon.json>",
           "scene_semantic_json": str(sem_path) if sem_path else "<scene_semantics.json>", "workspace": str(ws) if ws else None}
    f = failed[0] if failed else {}
    ctx.update({"index": f.get("index"), "side": f.get("side") or "<side>", "object": f.get("object") or "<object body>",
                "motion": f.get("motion"), "out_dir": str(Path(f["_file"]).parent) if f.get("_file") else "<the executor's out_dir>"})
    if supplied_plan is not None:
        entry = _entry_from_plan(_load(supplied_plan), f.get("index"), f.get("step"), f.get("side"))
        plan = supplied_plan
    else:
        entry, plan = _plan_entry(ws if include_workspace_history else None, f.get("index"), f.get("step"), f.get("side"))
    entry = f.get("_plan_entry") or entry
    ctx["plan"] = plan or "<plan.json>"
    ctx["pose"] = (entry or {}).get("pose") or "<pose json of the subgoal>"
    ctx["hand_scene"] = (entry or {}).get("hand_scene") or "<hand scene json of the subgoal>"
    ctx["duration_s"] = (entry or {}).get("duration_s")
    ctx["entry"] = entry or {}
    hands = None
    if recon_path and Path(recon_path).exists():
        cand = Path(recon_path).resolve().parent / "hands.npz"
        hands = str(cand) if cand.exists() else None
    ctx["hands_npz"] = hands
    # Read the state produced by the failing executor or final assembly, not
    # a planned wrist reference or an unrelated newest experiment.
    trajectory_candidates = ([Path(f["_file"]).parent / "trajectory.npz"] if f.get("_file") else [])
    if ws is not None:
        trajectory_candidates.append(ws / "trajectory.npz")
    ctx["trajectory"] = next((str(p) for p in trajectory_candidates if p.exists()), "<trajectory.npz from the physically executed prefix>")
    return ctx


def _repairs(c, x, sgs, objs, upstream, evidence=()):
    k = x["index"] if x["index"] is not None else (min(sgs) if sgs else "<k>")
    resume = {"tool": "execute_subgoal_plan", "args": {"scene": x["scene"], "plan": x["plan"], "out_dir": x["out_dir"], "resume": True, "from_subgoal": k},
              "why": "re-run from the fixed subgoal only: finished subgoals are kept"}
    opt = {"tool": "optimize_grasp_pose", "args": {"hand_scene": x["hand_scene"], "starts": x["pose"], "family": "auto", "bench_scene": x["scene"],
                                                    "out_json": f"poses/opt_sg{k}.json"},
           "why": "one call: arm screen and the real stress tests over many variants of the failing pose, in parallel"}
    if c == "upstream_suspect":
        if not (upstream or {}).get("verified"):
            return []  # Other measured failure classes determine the repairs.
        st = (upstream or {}).get("stage", "stage2")
        return [{"tool": "report_blocked", "args": {"reason": f"{st} must be revised: <what is wrong and what it should be>", "upstream": st,
                                                    "evidence": (upstream or {}).get("evidence", "<measured evidence>")},
                 "why": f"an earlier stage's result is wrong; the run re-opens {st} once with this reason and re-runs the stages after it"}]
    if c == "deformable_topology":
        return [{"tool": "inspect_rope_state", "args": {"scene": x["scene"], "trajectory": x["trajectory"]},
                 "why": "inspect the actual material centreline, spatial crossings and loop apertures. Passing in_hand/no_slip only proves the grasp. "
                        "A visible crossing or end motion is not a knot; no finite unsuccessful plan proves the source task impossible"},
                {"tool": "inspect_rope_state", "args": {"scene": x["scene"], "trajectory": x["trajectory"],
                    "loop_bounds": "<material bounds of the observed loop>", "history_start": "<frame before the attempted passage>"},
                 "why": "track the chosen endpoint relative to the moving material aperture across the executed attempt; distinguish a passage "
                        "inside the loop from motion around its outside. Use the measured state for subsequent contact/reach planning; passage diagnostics do not waive the topology gate"},
                dict(resume, why="revise the missing deformation or threading operation from the measured executed state and resume from the earliest changed subgoal; "
                                 "then check rope_knot_check through release and settling and run the complete final gates")]
    if c == "state_requires_physics":
        return [{"tool": "execute_subgoal_plan", "args": {"scene": x["scene"], "plan": x["plan"],
                                                           "out_dir": x["out_dir"], "resume": True, "dry_run": False},
                 "why": "execute the contact prefix physically; downstream deformable geometry cannot be inferred from a dry-run "
                        "wrist reference. Resume may retain physically verified segments, never promote a dry-run state"},
                {"tool": "hand_scene_from_recon", "args": {"scene_recon_json": x["scene_recon_json"], "side": x["side"],
                    "target": x["object"], "state": "<trajectory.npz from the physically executed prefix>"},
                 "why": "rebuild the hand scene from the measured current state, and use that same state for downstream "
                        "contact/reach search. This is state uncertainty, not evidence of an upstream geometry error"}]
    if c == "thin_object":
        o = sorted(objs)[0] if objs else x["object"]
        return [{"tool": "generate_flat_pinches", "args": {"scene": x["hand_scene"], "bench_scene": x["scene"]},
                 "why": f"{o} is lower than the pads: a nearly closed thumb-and-finger pinch across a part lying flat (1-3 cm)"},
                {"tool": "push_pose_from_point", "args": {"scene": x["hand_scene"], "point": "<a side face point>", "normal": "<its outward normal>"},
                 "why": "a non-prehensile primitive: slide the part (plan motion 'push' with direction and distance_m) until an edge "
                        "overhangs the table, then pinch that edge (plate_edge_pinch_pose)"},
                {"tool": "optimize_grasp_pose", "args": {"hand_scene": x["hand_scene"], "family": "flat", "bench_scene": x["scene"]},
                 "why": "search the flat-pinch family with the stress tests"}]
    if c == "needs_steadying_hand":
        other = {"left": "right", "right": "left"}.get(x["side"], "<other side>")
        return [{"tool": "hand_scene_from_recon", "args": {"scene_recon_json": x["scene_recon_json"], "side": other, "target": "<the part to steady>"},
                 "why": f"steady the part with the {other} hand first: a 'hold' subgoal (generate_rigid_grasps with keep_clear = the "
                        "acting hand's zone, or flat_hand_pose pads down) before the acting subgoal"},
                {"tool": "execute_subgoal_plan", "args": {"scene": x["scene"], "plan": x["plan"], "out_dir": x["out_dir"], "resume": True, "from_subgoal": k},
                 "why": "for one articulated part, insert hold before the acting entry; the executor couples co-holders. "
                        "Use both only for different independently manipulated parts. Screen both hands throughout the shared motion."}]
    if c == "unreachable":
        entry = x.get("entry") or {}
        args = {"scene": x["scene"], "side": x["side"], "pose": x["pose"], "hand_scene": x["hand_scene"]}
        if entry.get("joint") and entry.get("target") is not None:
            args["sweep"] = {"joint": entry["joint"], "to": entry["target"], "hand_follows": entry.get("hand_follows", "rigid")}
        grasp_repairs = [] if entry.get("joint") else [dict(opt, why="search contact families with the complete intended arm_waypoints, including held-object translation and "
                             "orientation targets ({object, target_pos, target_euler_deg, steps}); an initial grasp alone is insufficient")]
        contact_args = {"scene": x["scene"], "side": x["side"], "hand_scene": x["hand_scene"]}
        if "sweep" in args:
            contact_args["sweep"] = args["sweep"]
        return grasp_repairs + [{"tool": "optimize_pose_for_reach", "args": args,
                 "why": "screen the whole manipulation from the current state: joint sweep for an articulated part or held-object "
                        "waypoints for a carry/rotation. Compare rigid versus level contacts where physically appropriate, then stress-test and execute"},
                {"tool": "reachable_contact_search", "args": contact_args,
                 "why": "supply a measured geom/face or contact candidates and the planned sweep; multiple contact locations/headings "
                        "and sweep.hand_follows='auto' explore distinct strategies. Use best.hand_follows in the plan. No finite failure "
                        "proves upstream impossibility; keep stations and task goals fixed"}]
    if c == "no_stress_pass":
        out = []
        if x.get("hands_npz"):
            out.append({"tool": "human_grasp_seeds", "args": {"scene_semantic_json": x["scene_semantic_json"], "hands_npz": x["hands_npz"],
                                                               "scene_recon_json": x["scene_recon_json"], "out_dir": "poses"},
                        "why": "the human's own grasp at the contact window, retargeted and snapped onto the part, as starts"})
        out.append(opt)
        out.append({"tool": "generate_rigid_grasps", "args": {"scene": x["hand_scene"], "bench_scene": x["scene"], "out_json": f"poses/grasps_sg{k}.json"},
                    "why": "more starts of another family; give the json to optimize_grasp_pose(starts=...)"})
        return out
    if c == "slip_in_carry":
        return [dict(opt, args=dict(opt["args"], tests=["grasp_lift_test", "grasp_hold_test", "grasp_shake_test"]),
                     why="a grip that also survives the carry's accelerations (shake test) before re-executing"),
                dict(resume, why="with the carry's duration_s raised (" + (f"{x['duration_s']} s" if x.get("duration_s") else "its default")
                                 + " x 1.5): slower legs load the grip less")]
    if c == "collision":
        return [{"tool": "preview_keyframes", "args": {"keyframes": "<the subgoal's keyframes json from plan_subgoal_keyframes>"},
                 "why": "inspect the reported collision pair and phase; distinguish a destination/object obstruction from hand-hand contact"},
                dict(opt, why="search a grasp with clearance over the complete arm_waypoints including the destination and release approach; "
                             "a pose that fits at pickup may not fit during placement"),
                dict(resume, why="repair the measured obstruction with a clear approach or transfer path and resume from the changed entry; "
                                 "park another hand only if it is idle and its task role permits it, never while it holds or steadies a part")]
    if c == "joint_speed":
        return [dict(resume, why="with the failing entry's duration_s raised (x 1.5) or a lower max_rate_deg_s: the arm was asked to move too fast"),
                {"tool": "optimize_pose_for_reach", "args": {"scene": x["scene"], "side": x["side"], "pose": x["pose"], "hand_scene": x["hand_scene"]},
                 "why": "a wrist orientation the arm reaches without flipping its configuration mid-motion"}]
    if c == "physics_audit":
        return [dict(resume, why="slow the leg at the frame the audit names (duration_s), lower a press force, or lift a pinch "
                                 "that drags a cable into itself; the audit names the frame and the phase")]
    if c == "stage4_randomize":
        return [dict(opt, args=dict(opt["args"], tests=["grasp_lift_test", "grasp_hold_test", "grasp_pose_noise_test"]),
                     why="Stage 4 replays the episode in new layouts: a grasp that only holds at the exact pose fails there; "
                         "search one that survives pose noise, then re-execute"),
                {"tool": "write_stage4_config", "args": {"scene_recon_json": x["scene_recon_json"], "episode_dir": x["out_dir"],
                                                "out_dir": str(Path(x["workspace"]) / "stage4_task") if x["workspace"] else "<workspace>/stage4_task",
                                                "human_video": "<the human video>"},
                 "why": "re-package after the re-execution and run stage4_handoff_check"}]
    # goal_not_met
    return [dict(resume, why="fix the failing subgoal's parameters (target_pos / on / into, joint target, turns, distance) to what its "
                             "Stage 1 goal sentence asks, then re-run from it"),
            {"tool": "stage3_gate_suite", "args": {"workspace": x["workspace"] or "<workspace>", "scene_semantic_json": x["scene_semantic_json"],
                                                   "scene_recon_json": x["scene_recon_json"]},
             "why": "inspect which measured goal clauses remain unmet; plan and execute every required clause. Additional predicates "
                    "cannot replace the source goal, and a deviation cannot waive missing evidence"}]


def _summary_text(classes, repairs, upstream):
    if not classes:
        return "Diagnosis: no failure class recognised in the verdict, outcomes or workspace."
    lines = ["Diagnosis (diagnose_stage3_failure), most likely first:"]
    for c in classes[:4]:
        lines.append(f"- {c['class']} (score {c['score']}" + (f", subgoals {c['subgoals']}" if c["subgoals"] else "") + f"): {c['evidence'][0] if c['evidence'] else ''}")
    if upstream and upstream.get("verified"):
        lines.append(f"- an earlier stage is suspect ({upstream['stage']}): fix it with report_blocked(reason, upstream='{upstream['stage']}', evidence).")
    elif upstream:
        lines.append("- upstream error is an unverified hypothesis; failed finite search does not prove a reconstruction defect.")
    lines.append("Next steps:")
    for r in repairs[:5]:
        args = json.dumps(r["args"], default=str)
        lines.append(f"{r['rank']}. {r['tool']}({args[:260]}) - {r['why'][:200]}")
    return "\n".join(lines)
