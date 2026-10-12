"""Plan, execute and check a whole subgoal list, one subgoal at a time, each from the physics state the last one left."""
from __future__ import annotations

TOOL = {
    "name": "execute_subgoal_plan",
    "title": "Execute subgoal plan",
    "type": "skill",
    "category": "Trajectory & Execution",
    "summary": "Run a list of subgoals end to end: plan each one's keyframes from the executed state of the previous one, "
               "roll it out in contact physics, check its expectations, and stop at the first subgoal that fails; writes the "
               "actual rigid robot episode with phases, recorded controls and per-subgoal outcomes.",
    "description": (
        "The Stage 3 driver. The hand is bolted to its arm, so the arm is planned with the hand (couple_arms) and a subgoal "
        "the arm cannot hold is re-solved (next grasp candidate, pose turned for reach, carried object turned) before any "
        "physics. The arm fill is collision aware (common/armplan.py): a free-space leg whose straight joint path runs the arm, "
        "its hand or what it carries into the table, an object, the other station or its own hand is replaced by an RRT-Connect "
        "path (shortcut, corners smoothed, 20/10/5 mm clearance) with the same ends, an idle station (its hand still and touching "
        "nothing) that comes within "
        "3 cm of the active one is parked at a clearance configuration and comes back on its own next reach, and legs too fast "
        "for the arm (90 deg/s, 12 rad/s^2 smoothed, 4.8 m/s^2 at the flange) are slowed with contact phases kept at their pace; "
        "if an otherwise passing physical trial exceeds the arm speed cap, the same joint path is retried more slowly "
        "at most twice from the identical complete initial state. Every gate is rerun, integration-step velocities "
        "determine the speed peak, and failed timing trials remain in the segment's timing_attempts folder. "
        "a clean straight plan is left exactly as it was, and the subgoal's measured.arm.plan lists what was planned, parked or "
        "could not be. A one-hand entry whose planned hand runs more than 3 mm into the other (still) hand fails as "
        "hand_clearance before physics. For subgoal k it calls plan_subgoal_keyframes from the physics state at the end of "
        "subgoal k-1 (hand commands carried over so a holding hand keeps pressing), rollout_keyframes_physics on the "
        "keyframes, and the outcome evaluation on the executed frames (rise, in hand, no slip, at target, released, at "
        "rest, supported, joint reached, turned, tilted, untouched objects, penetration, servo tracking). A required "
        "stable-grasp check cannot be disabled with expect.no_slip=False, redirected to a different object/hand, "
        "or shortened with slip_from; the executed phases determine its window. The slip tolerance can only be tightened. "
        "A failing subgoal stops the run with its measurements so the pose or the parameters can be changed and the run resumed "
        "from that subgoal (`resume=True` keeps the finished segments on disk; every plan entry has its own segment folder, "
        "subgoal_NN, or subgoal_NN_SS for entries that share an index: their `step`, else the next free step number in plan "
        "order, so entries never overwrite each other). A two-hand entry ({motion: 'both', hands: {left: entry, right: "
        "entry}, sync}) runs both hands' plans in ONE rollout, time-aligned: both arms are planned with their hands (each "
        "with its own free-space legs, each planned against the other arm and hand where they are at the same time: a leg that "
        "would meet them first waits along its own straight path, else takes a planned path, retimed together), the planned "
        "hands must stay hand_clearance_m apart (else the "
        "attempt fails as hand_clearance before physics), and each hand's expectations are checked with the bystanders "
        "and hands_apart; lift_together and carry_together (two hands on one cable) run the same way. An isolated "
        "grasp stress pass is a candidate screen: mounted-arm contact dynamics and another hand acting on the same "
        "object can still change the grasp. A rise/contact/slip failure requires a changed grasp, approach or "
        "timing followed by a fresh physical trial from the same valid prefix. For lift_together, approach_controls "
        "reports the waypoint offsets and synchronized contact timings used in the reference; an agent can verify "
        "that its proposed repair changed the plan. A pin (one hand "
        "steadying a part while the other acts) lasts through the following entries: the pinning hand keeps its commands "
        "and, for a press, its wrist force cap, and every such entry is judged on the pinned part staying put on every "
        "frame (`pinned`). A press_seat entry runs with the wrist force cap its plan asks for (at most the 40 N default). "
        "A grip clamps the grip fingers press_depth_rad past their closed targets (torque-capped); a negative press_depth_rad "
        "(on the pose or the entry) drives them that far toward extension instead, spreading what they hold (scissor rings). "
        "The episode is written as "
        "trajectory.npz paired with scene_physics.xml: the actual rigid robot's named joint states and controls, including "
        "every integration step. Proposal diagnostics use separate files. Failed executions have no formal trajectory. "
        "execution_report.json binds the scene, trajectory, and each selected/re-anchored pose to its starting hand scene."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "Bench scene MJCF (Stage 2 scene.xml)."},
        {"name": "plan", "type": "dict | str | list", "description": "{subgoals: [...] (one-hand entries, or two-hand entries {index, motion: 'both', hands: {left: {...}, right: {...}}, sync: 'start'|'end'|'keyframes', hand_clearance_m}), scene_recon_json?: path (screw-cap release rules, scissor cut rules and connector latch rules are read from it: an object's `cut_joins` from construct_cuttable_rope against the scissors' `cutter`; an object's `latch` from add_connector_latch, which engages the plug where it bottoms out in its port and lets it go above release_N, carried from subgoal to subgoal), rules?: [...]} or a JSON path or the list; each subgoal as plan_subgoal_keyframes takes it. With cut rules every physics step checks scissors.try_sever on each join, trajectory.npz / trajectory_hands.npz carry eq_active [T, neq] (the equality state after each frame), eq_active_start and eq_names, and the summary lists `cuts` [{equality, frame, subgoal}]; with latch rules the summary lists `latches` [{equality, event: engage | release, frame, subgoal}] read from that record and each subgoal's outcome its `latches` events; scenes without such rules are run and written exactly as before."},
        {"name": "out_dir", "type": "str", "description": "Output folder."},
        {"name": "fps", "type": "float", "description": "Default 30."},
        {"name": "resume", "type": "bool", "description": "Keep finished segments in out_dir and start from the first unfinished or failed subgoal. Default false."},
        {"name": "from_subgoal", "type": "int | None", "description": "With resume: restart from this subgoal index even if it passed."},
        {"name": "stop_on_fail", "type": "bool", "description": "Default true."},
        {"name": "fill_arms", "type": "bool", "description": "Deprecated compatibility argument. The runtime always uses physical rigid arms; post-run fitting is forbidden."},
        {"name": "wrist_kp", "type": "float", "description": "Wrist servo stiffness. Default 10000."},
        {"name": "max_rate_deg_s", "type": "float", "description": "Joint-rate ceiling for the hands' reference. Default 90."},
        {"name": "settle_s", "type": "float", "description": "Must be 0 (default). Settling must be an explicit recorded initial hold, so the episode starts at the trusted Stage 2 home state."},
        {"name": "require_reach", "type": "bool", "description": "Fail a subgoal before physics when the station's arm cannot hold its planned hand on the flange or runs into the scene. Default true."},
        {"name": "reach_tol_m", "type": "float", "description": "Legacy gate (couple_arms=False only). Default 0.03."},
        {"name": "couple_arms", "type": "bool", "description": "Must be true (default): arm IK converts floating planning proposals into rigid robot joint references, bounded arm motors execute them, and the hand remains fixed to its flange. Infeasible references fail before physics."},
        {"name": "arm_speed_deg_s, project_tol_mm, project_tol_deg, weld_tol_mm, weld_tol_deg, arm_collision_tol_m", "type": "float", "description": "Defaults 90, 6, 4, 2, 2, 0.003."},
        {"name": "dry_run", "type": "bool", "description": "Screen arm reach without physics; this is not a physical success verdict. Rigid proposals can be chained. A deformable contact has no valid predicted terminal state, so downstream screening stops with state_requires_physics: execute the physical prefix, then resume from its recorded state. No trajectory is written. Default false."},
        {"name": "max_attempts", "type": "int", "description": "Total attempts per subgoal, including physical timing retries: grasp candidates (a `pose` that is a ranked list or json list, or `pose_candidates`), screened for the arm first; on an arm failure the pose turned about its contact (auto_reach_pose; at most 15 deg, or the entry's max_reach_turn_deg: a larger turn is another grasp, so it is reported, not tried) and, for carries/places without a prescribed orientation, the carried object turned about the vertical (auto_turn). Default 6."},
        {"name": "auto_turn, auto_reach_pose", "type": "bool", "description": "Enable those re-solves. Default true."},
        {"name": "plan_arm_paths", "type": "bool", "description": "Collision-aware arm fill (couple_arms only): planned free-space legs, idle-arm parking, dual-arm timing. Default true; false keeps every leg straight."},
    ],
    "outputs": [
        {"name": "summary", "type": "dict", "description": "ok, n_frames, duration_s, subgoals: [{index, tag (the segment folder), motion, ok, failed, measured, frames}], first_failure, files {trajectory, trajectory_hands, phases, outcomes, state, scene_physics}, arm_fit, cuts / equalities_released / latches (when the scene has those mechanisms). A two-hand entry's failed checks are named '<side>:<check>' (plus untouched:*, hands_apart, hand_clearance); its frames carry 'left: ... | right: ...' labels and phases.json lists each two-hand phase once per hand."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "driver", "subgoals", "episode"],
    "used_in": ["Open the Bottle", "Giftbox Packing", "Place Toy in Drawer", "Battery Insertion", "Place Cup in Bowl", "Computer Installation"],
    "status": "tested",
    "entrypoint": "execute_subgoal_plan",
}

import copy
import hashlib
import json
import shutil
from pathlib import Path

import mujoco
import numpy as np

from tool_library.common import armhand as AH
from tool_library.common import armplan as AP
from tool_library.common import arm_timing as AT
from tool_library.common import bimanual as B
from tool_library.common import episode as E
from tool_library.common import rigid_robot as RR
from tool_library.common import outcome as O
from tool_library.common import plan as P
from tool_library.common import trajectory as TJ
from tool_library.skills.chain_subgoals import chain_subgoals
from tool_library.skills.plan_subgoal_keyframes import plan_subgoal_keyframes
from tool_library.skills.rollout_keyframes_physics import rollout_keyframes_physics


def _load_plan(plan):
    rules = []
    base = None
    if isinstance(plan, (str, Path)):
        base = Path(plan).resolve().parent             # relative paths in a plan file are relative to the file
        plan = json.loads(Path(plan).read_text())
    if isinstance(plan, dict):
        rules = list(plan.get("rules") or [])
        if plan.get("scene_recon_json"):
            rules += thread_rules_from_recon(_rel(plan["scene_recon_json"], base))
            rules += cut_rules_from_recon(_rel(plan["scene_recon_json"], base))
            rules += latch_rules_from_recon(_rel(plan["scene_recon_json"], base))
        plan = plan.get("subgoals", plan)
    out = []
    for i, sg in enumerate(plan):
        sg = _resolve_paths(dict(sg), base); sg.setdefault("index", i + 1)
        if isinstance(sg.get("hands"), dict):
            sg["hands"] = {s_: _resolve_paths(dict(e), base) for s_, e in sg["hands"].items()}
        out.append(sg)
    from tool_library.common.subgoal import expand_subgoals
    return _assign_tags(expand_subgoals(out)), rules


def _resolve_paths(sg, base):
    for key in ("pose", "pose_candidates", "hand_scene", "regrasp_pose", "reference"):
        if isinstance(sg.get(key), str):
            sg[key] = _rel(sg[key], base)
    if isinstance(sg.get("pose_candidates"), list):
        sg["pose_candidates"] = [_rel(c, base) if isinstance(c, str) else c for c in sg["pose_candidates"]]
    return sg


def _step_of(sg):
    st = sg.get("step")
    return int(st) if st else None


def _assign_tags(subgoals):
    """Every plan entry its own segment folder (`_tag`): subgoal_NN for an index used once (subgoal_NN_SS when it gives a
    step); entries that share an index keep their own distinct `step`, and the others take the next free step number in
    plan order. Entries sharing an index without a step used to write the same folder, each overwriting the last, and
    resume then read another entry's rollout. `_legacy_tag` is the folder the old naming gave the entry (resume may
    still read it, see _legacy_segment)."""
    groups = {}
    for sg in subgoals:
        groups.setdefault(int(sg["index"]), []).append(sg)
    for k, grp in groups.items():
        for sg in grp:
            st = _step_of(sg)
            sg["_legacy_tag"] = f"subgoal_{k:02d}" + (f"_{st:02d}" if st else "")
        if len(grp) == 1:
            grp[0]["_tag"] = grp[0]["_legacy_tag"]
            continue
        used, pending = set(), []
        for sg in grp:
            st = _step_of(sg)
            if st is not None and st not in used:
                used.add(st); sg["_seg_step"] = st
            else:
                pending.append(sg)
        nxt = 1
        for sg in pending:
            while nxt in used:
                nxt += 1
            used.add(nxt); sg["_seg_step"] = nxt
        for sg in grp:
            sg["_tag"] = f"subgoal_{k:02d}_{int(sg['_seg_step']):02d}"
    return subgoals


def _signature(sg):
    return (sg.get("motion"), sg.get("side") or sg.get("acting_hand") or ("both" if B.is_two_hand(sg) else None))


def _execution_implementation_digest():
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "common").glob("*.py")) + [root / "skills" / name for name in
            ("execute_subgoal_plan.py", "plan_subgoal_keyframes.py", "rollout_keyframes_physics.py", "chain_subgoals.py")]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _resume_signature(sg, rules, options, scene_sha256, _binding_cache=None):
    """Invalidate cached segments when code, plans, input poses or controls change.

    This is cache identity, not verification evidence. Reused recordings still
    undergo the independent physics and task gates before publication.
    """
    workspace = Path.cwd()
    compiled = {} if _binding_cache is None else _binding_cache

    def bound(value, base=workspace, seen=()):
        if isinstance(value, dict):
            return {k: bound(v, base, seen) for k, v in value.items() if k not in ("_tag", "_legacy_tag", "_seg_step")}
        if isinstance(value, (list, tuple)):
            return [bound(v, base, seen) for v in value]
        if isinstance(value, (str, Path)):
            named = Path(value)
            # Candidate wrappers can name further pose/hand-scene files. The
            # executor uses workspace paths first; some readers fall back to
            # the containing file. Bind both so a real repair cannot disappear
            # behind an unchanged wrapper, including same-named alternatives.
            candidates = [named] if named.is_absolute() else [workspace / named, base / named]
            bindings, visited = [], set()
            for path in candidates:
                try:
                    if not path.is_file():
                        continue
                    path = path.resolve()
                    if path in visited:
                        continue
                    visited.add(path)
                    raw = path.read_bytes()
                    digest = hashlib.sha256(raw).hexdigest()
                except OSError:
                    continue  # normal input loading will report unreadable files
                item = {"path": str(path), "sha256": digest}
                if path in seen:
                    item["cycle"] = True
                elif path.suffix.lower() == ".json":
                    try:
                        decoded = json.loads(raw)
                    except (ValueError, UnicodeError):
                        pass  # malformed inputs remain content-bound
                    else:
                        item["contents"] = bound(decoded, path.parent, (*seen, path))
                elif path.suffix.lower() == ".xml":
                    # The XML bytes omit included files and mesh data. Bind the
                    # compiled model too; compile each asset once per execution.
                    key = (str(path), digest)
                    if key not in compiled:
                        try:
                            model = mujoco.MjModel.from_xml_path(str(path))
                            buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
                            mujoco.mj_saveModel(model, buffer=buffer)
                            compiled[key] = hashlib.sha256(buffer.tobytes()).hexdigest()
                        except (ValueError, OSError):
                            compiled[key] = None  # e.g. a non-model XML sidecar
                    item["compiled_sha256"] = compiled[key]
                bindings.append(item)
            if bindings:
                return {"resolved_inputs": bindings}
            return str(value)
        return value
    payload = {"schema": "dexagent.segment-input.v3", "scene_sha256": scene_sha256,
               "implementation_sha256": _execution_implementation_digest(),
               "subgoal": bound(sg), "rules": bound(rules), "options": options}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False, default=float).encode()).hexdigest()


def _legacy_segment(out, sg, subgoals):
    """A finished segment the old naming wrote for this entry (entries sharing an index all wrote subgoal_NN, the last
    one winning): used only when that folder's outcome is ok, its rollout has the recorded length, and its motion and
    side match this entry and no other entry of the same old folder."""
    lt = sg.get("_legacy_tag")
    if not lt or lt == sg.get("_tag"):
        return None
    d = Path(out) / lt
    if not all((d / f).exists() for f in ("rollout.npz", "outcome.json", "phases.json")):
        return None
    try:
        rec = json.loads((d / "outcome.json").read_text())
    except Exception:
        return None
    if not rec.get("ok") or rec.get("dry_run"):
        return None
    mine = _signature(sg)
    if (rec.get("motion"), rec.get("side")) != mine:
        return None
    rivals = [x for x in subgoals if x is not sg and x.get("_legacy_tag") == lt and _signature(x) == mine]
    if rivals:
        return None
    try:
        n = len(E.load_episode(d / "rollout.npz")["qpos"])
    except Exception:
        return None
    if rec.get("n_frames") is not None and int(rec["n_frames"]) != n:
        return None
    return dict(rec, tag=sg["_tag"], legacy_folder=lt), d


def _pose_candidates(sg):
    """All supplied poses, including singletons, receive the same arm screen."""
    c = sg.get("pose_candidates")
    if isinstance(c, list) and c:
        return list(c)[:48]
    p = c if isinstance(c, (str, Path)) else sg.get("pose")
    if isinstance(p, list):
        return p[:48]
    if isinstance(p, (str, Path)) and str(p).endswith(".json") and Path(p).exists():
        try:
            v = json.loads(Path(p).read_text())
        except Exception:
            return []
        if isinstance(v, list):
            return v[:48]
        if isinstance(v, dict) and isinstance(v.get("grasps"), list):
            return v["grasps"][:48]
        if isinstance(v, dict):
            return [p]
    if isinstance(p, dict):
        return [p]
    return []


def _screen_entry_poses(model, side, poses, sg, q_phys, **kwargs):
    """Rank at the current object state, using the primitive's approach distance.

    This inexpensive ranking is advisory; the planner's complete, possibly
    contact-corrected path and actual physics remain the acceptance checks.
    """
    from tool_library.common.subgoal import load_pose, pose_anchor, reanchor_pose
    data = mujoco.MjData(model)
    data.qpos[:] = q_phys
    mujoco.mj_forward(model, data)
    materialized, approaches = [], []
    for candidate in poses:
        try:
            pose = dict(load_pose(candidate))
            for key in ("open_frac", "finger_targets_open", "press_depth_rad", "press_travel_m"):
                if sg.get(key) is not None:
                    pose[key] = sg[key]
            pose = reanchor_pose(model, data, pose, pose_anchor(sg, pose), mode=sg.get("reanchor", "full"))
        except (OSError, ValueError, TypeError, KeyError):
            # Preserve candidate indices and let the screen report an unreadable
            # pose. A malformed alternative must not abort all other candidates.
            pose = {}
        motion = sg.get("motion")
        approach = .08
        if motion == "pin":
            grip = list(pose.get("grip_fingers") or [])
            mode = sg.get("mode") or pose.get("pin_mode") or ("press" if grip and "thumb" not in grip else "grip")
            approach = .05 if mode == "press" else .06
        elif motion == "hold" or motion in ("open_joint", "close_joint") or sg.get("joint"):
            approach = .06
        elif motion in ("grasp_more", "press_seat"):
            approach = .05
        approaches.append(float(sg.get("approach_m", approach)))
        materialized.append(pose)
    return AH.screen_poses(model, side, materialized, q_phys, approach_m=approaches, **kwargs)


# the largest turn of a stress-tested pose the executor makes on its own to reach it (a plan entry may set
# max_reach_turn_deg); a larger turn is a different grasp the agent turns and proves itself
MAX_REACH_TURN_DEG = 15.0


def _reach_variants(scene, sg, auto_turn=True, auto_reach_pose=True):
    """Re-solves for a subgoal whose arm cannot hold the planned hand: the grasp turned about its own contact for
    reach (optimize_pose_for_reach, strict tolerance), and for carries and places without a prescribed orientation the
    carried object turned about the vertical (smallest turn first)."""
    out = []
    motion = str(sg.get("motion", ""))
    if auto_reach_pose and sg.get("pose") is not None and motion in ("grasp_lift", "hold", "regrasp", "rotate", "push", "pull", "press", "press_seat", "pin", "edge_pinch", "overhang_grasp", "hook", "strip_pinch", "lay", "drag"):
        try:
            from tool_library.skills.optimize_pose_for_reach import optimize_pose_for_reach
            wp = [[0.0, 0.0, float(sg.get("lift_m", 0.08))]] if motion == "grasp_lift" else None
            if motion in ("lay", "drag"):              # a rope move: the pinch must also hold at its lift
                wp = [[0.0, 0.0, float(sg.get("lift_m") or (0.008 if motion == "drag" else 0.04))]]
            r = optimize_pose_for_reach(scene, sg["side"], sg["pose"], hand_scene=sg.get("hand_scene"), pos_tol_m=0.004, tilt_tol_deg=3.0, waypoints=wp)
            rot = r.get("rotation_deg") or {}
            turn = float(np.sqrt(sum(float(v or 0.0) ** 2 for v in rot.values())))
            limit = float(sg.get("max_reach_turn_deg", MAX_REACH_TURN_DEG))
            if r.get("pose") is not None and turn > 1e-6:
                if turn <= limit:
                    out.append((dict(sg, pose=r["pose"]), f"pose turned {rot} deg about its contact for reach"))
                else:
                    # a larger turn is another grasp: its stress tests were run on the pose as planned, not on this one
                    out.append((dict(sg, _skip=f"the reachable turn of the pose ({turn:.0f} deg, {rot}) exceeds "
                                               f"max_reach_turn_deg {limit:.0f}: turn it with optimize_pose_for_reach and stress-test the turned pose"),
                                f"pose turned {rot} deg about its contact for reach (not tried)"))
        except Exception as e:                   # the re-solve is a best effort; the failure is reported either way
            out.append((dict(sg), f"reach re-orientation failed: {type(e).__name__}: {e}"[:200]))
            out.pop()
    if motion in ("carry", "place") and sg.get("arm_path") != "joint":
        # the same target and orientation, the transfer planned in arm joint space: an arm parked in a configuration
        # that cannot follow the straight carry (it would flip mid-way) changes configuration in free space instead
        out.append((dict(sg, arm_path="joint"), "carried along a joint-space arm path (the arm may change configuration in the air)"))
        if motion == "carry" and sg.get("via_height") is None:
            # the same, lifted 12 cm first: a straight joint-space swing can sweep the forearm through the held part
            out.append((dict(sg, arm_path="joint", via_rise_m=0.12), "carried along a joint-space arm path after rising 12 cm"))
    out += _low_carry_variants(sg)
    if auto_turn and motion in ("carry", "place") and sg.get("target_euler_deg") is None and not sg.get("turn"):
        for deg in (20, -20, 40, -40, 60, -60, 90, -90):
            out.append((dict(sg, turn={"axis": [0.0, 0.0, 1.0], "deg": float(deg)}), f"carried object turned {deg:+d} deg about the vertical for reach"))
    return out


def _low_carry_variants(sg):
    """A place carries the part 5 cm above the higher of the wrist and the target by default; when the wrist is already
    high that straightens the elbow (out of reach, or arm joint speeds near the singularity): carry it lower."""
    if str(sg.get("motion", "")) != "place" or sg.get("via_height") is not None or sg.get("carry_rise_m") is not None:
        return []
    return [(dict(sg, carry_rise_m=0.02), "carried 3 cm lower (rise 2 cm)"), (dict(sg, carry_rise_m=0.0), "carried at the wrist's height (no rise)")]


def _side_of(sg):
    return "both" if B.is_two_hand(sg) else sg.get("side")


def _object_of(sg):
    if B.is_two_hand(sg):
        return {s_: (e or {}).get("object") for s_, e in (sg.get("hands") or {}).items()}
    return sg.get("object")


def _deformable_body_ids(model, recon_path=None):
    """Recognize articulated/constraint cables, declared deformables and flexes.

    Reuse the integrity audit's structural recognition so legacy scenes need
    neither an object-name convention nor a reconstruction manifest.
    """
    from types import SimpleNamespace
    from tool_library.verifiers.deformable_integrity_audit import _find_chains
    robot = {b for b in range(1, model.nbody) if model.body(b).name.startswith("robot_")}
    bodies = {b for ch in _find_chains(SimpleNamespace(model=model, robot_bodies=robot), 3)
              for b in ch["bodies"]}
    if getattr(model, "nflexvert", 0):
        bodies.update(int(b) for b in model.flex_vertbodyid if int(b) > 0)
    if recon_path:
        recon = json.loads(Path(recon_path).read_text())
        for obj in recon.get("objects", []):
            if str(obj.get("type", "")).lower() != "deformable":
                continue
            for name in [obj.get("root_body"), *(obj.get("bodies") or [])]:
                if name:
                    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(name))
                    if bid >= 0:
                        bodies.add(int(bid))
    # Include attached handles and other descendants of a deformable root.
    for b in range(1, model.nbody):
        if int(model.body_parentid[b]) in bodies:
            bodies.add(b)
    return bodies - robot


def _dry_deformable_contacts(model, bodies, planned, sgv, previous_state):
    names = _allowed_bodies(planned, sgv) + list((previous_state.get("held") or {}).values())
    return sorted({name for name in names if isinstance(name, str)
                   and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) in bodies})


def _allowed_bodies(planned, sgv):
    """Bodies the hands may touch in the kinematic plan: the handled parts, their supports, what the hands hold (per hand
    for a two-hand entry)."""
    exp = planned["expect"]
    entries = [(exp, sgv)]
    if exp.get("hands"):
        entries = [(e, (sgv.get("hands") or {}).get(s_, {})) for s_, e in exp["hands"].items()]
    out = []
    for e, sub in entries:
        out += [e.get("object"), sub.get("object"), sub.get("on"), sub.get("over")] + list(e.get("involved", []) or [])
    out += list(exp.get("involved", []) or []) + list((planned["state"].get("held") or {}).values())
    return [b for b in dict.fromkeys(out) if b]


def _both_variants(model, sg, q_phys, probe, arm_state, ah_cfg, parts, couple_arms):
    """Variants of a two-hand entry: each hand's grasp candidates (its `pose` list / `pose_candidates`), screened for
    its own arm first when the arms are coupled, combined in order of rank (best pair first). Returns (variants,
    per-hand screens)."""
    per, screens = {}, {}
    for s_, e in sg["hands"].items():
        c = _pose_candidates(e)
        if not c:
            per[s_] = [None]
            continue
        order = list(range(len(c)))
        if couple_arms:
            screen_entry = dict(e)
            if "approach_m" not in screen_entry and "approach_m" in sg:
                screen_entry["approach_m"] = sg["approach_m"]
            scr = _screen_entry_poses(model, s_, c, screen_entry, q_phys, probe=probe,
                                      arm_seed=arm_state.get(s_), cfg=ah_cfg, parts=parts)
            screens[s_] = scr
            order = [r["index"] for r in scr if r["ok"]]
            order += [r["index"] for r in sorted((r for r in scr if not r["ok"] and r.get("duplicate_of") is None and "residual_mm" in r),
                                                 key=lambda r: (bool(r.get("arm_hits")), r["residual_mm"] + r["residual_deg"]))]
        per[s_] = [(i, c[i]) for i in order] or [None]
    sides = list(sg["hands"])
    pairs = sorted(((i, j) for i in range(len(per[sides[0]])) for j in range(len(per[sides[1]]))), key=lambda ij: (ij[0] + ij[1], ij[0]))[:24]
    out = []
    for i, j in pairs:
        hands, words = {}, []
        for s_, n in zip(sides, (i, j)):
            pick = per[s_][n]
            if pick is None:
                hands[s_] = dict(sg["hands"][s_])
            else:
                hands[s_] = dict(sg["hands"][s_], pose=pick[1]); words.append(f"{s_} pose candidate {pick[0] + 1}")
        out.append((dict(sg, hands=hands), ", ".join(words) or "as planned"))
    return out, (screens or None)


def _both_reach_variants(scene, sg, sides, auto_turn, auto_reach_pose):
    """The one-hand re-solves (_reach_variants) of the hand(s) whose arm failed, put back into the two-hand entry: when
    both arms failed, the two hands' k-th re-solves together first, then each hand's alone."""
    per = {}
    if not sg.get("hands"):                         # carry_together: nothing to re-solve per hand
        return []
    for s_ in sorted(sides or sg["hands"]):
        if s_ in sg["hands"]:
            per[s_] = _reach_variants(scene, dict(sg["hands"][s_], side=s_), auto_turn=auto_turn, auto_reach_pose=auto_reach_pose)
    out = []
    if len(per) == 2:
        (sa, va), (sb, vb) = per.items()
        for (a, na), (b, nb) in zip(va, vb):
            skip = a.get("_skip") or b.get("_skip")
            out.append((dict(sg, hands={**sg["hands"], sa: a, sb: b}, **({"_skip": skip} if skip else {})), f"{sa}: {na}; {sb}: {nb}"))
    for s_, vs in per.items():
        out += [(dict(sg, hands={**sg["hands"], s_: v}, **({"_skip": v["_skip"]} if v.get("_skip") else {})), f"{s_}: {note}") for v, note in vs]
    return out


# a one-hand entry fails before physics when its planned hand overlaps the other (still) hand deeper than this, the
# tolerance robot_self_collision_check holds the delivered episode to: the arm fill already routed the free legs around
# the other hand and parked it when it was idle, so what is left is the pose itself (two hands on rings 3 cm apart)
ONE_HAND_OVERLAP_M = 0.003


def _hand_gap_failure(model, Qc, labels, expect, note, need=None):
    """The two-hand entry's planned clearance gate: the hands' collision geoms must stay hand_clearance_m apart on every
    frame of the (arm-coupled) reference. With `need` (a one-hand entry: -ONE_HAND_OVERLAP_M) only an overlap deeper
    than that fails. Returns the failed attempt, or None."""
    one_hand = need is not None
    need = float(expect.get("hand_clearance_m", 0.005)) if need is None else float(need)
    clr = E.hand_hand_clearance(model, Qc, stride=2, cutoff_m=max(0.05, 2 * need))
    if clr is None or clr["min_m"] >= need:
        if clr is not None and not one_hand:
            expect["planned_hand_gap_mm"] = round(1000 * clr["min_m"], 1)
        return None
    lab = str(labels[clr["frame"]]) if clr["frame"] is not None and clr["frame"] < len(labels) else ""
    if one_hand:
        return {"variant": note, "failed": ["hand_clearance"], "hand_clearance": clr,
                "error": (f"the planned hand runs {-1000 * clr['min_m']:.1f} mm into the other hand ({clr['geoms'][0]} / {clr['geoms'][1]}) at "
                          f"'{lab}' (frame {clr['frame']}); the other hand holds or steadies a part there (an idle hand is parked out of "
                          "the way and free legs are routed around it), so no arm path avoids it: choose a grasp or an approach for this "
                          "hand away from the other, or let the other hand go first")}
    return {"variant": note, "failed": ["hand_clearance"], "hand_clearance": clr,
            "error": (f"the planned hands come {1000 * clr['min_m']:.1f} mm apart ({clr['geoms'][0]} / {clr['geoms'][1]}) at '{lab}' (frame "
                      f"{clr['frame']}), under hand_clearance_m {1000 * need:.0f} mm: move one hand's grasp, approach or destination away "
                      "from the other, or change the timing (sync 'end' or 'keyframes', a longer duration_s on one side)")}


def _couple_two_hands(model, Qref, fps, arm_start=None, labels=None, keyframe_frames=None, cfg=None, parts=None, max_frames=None,
                      planner=None):
    """armhand.couple for a two-hand entry. couple() gives an arm its free-space legs where the frame's label starts with
    its side, so a two-hand label ('left: reach for a | right: reach for b') would leave the right arm without its free
    reach. Each arm is filled on its own hand's part of the labels (retime off). With a planner both arms are then
    planned together (armplan.plan_arms: each arm's legs against the table, the objects and the OTHER arm and hand where
    they are at the same time; a second pass re-checks the first arm against the second's final path), then both are
    retimed TOGETHER (a leg one arm must take slower is slowed for both hands, so they stay time-aligned) and every
    hand is put back on its flange. Same return as couple()."""
    parts = parts or E.robot_parts(model)
    Qref = np.asarray(Qref, float)
    T = len(Qref)
    labels = list(labels) if labels is not None else [""] * T
    sides = [s_ for s_ in parts if parts[s_]["gripper"] >= 0 and parts[s_]["arm_joints"]]
    Q = Qref.copy()
    arms, report, arm_side, legs, starts = {}, {"ok": True, "infeasible": []}, {}, {}, {}
    for s_ in sides:
        r = AH.couple(model, Qref, fps, arm_start=arm_start, labels=E.side_view(labels, s_), keyframe_frames=keyframe_frames,
                      sides=[s_], cfg=cfg, parts=parts, retime=False)
        arm_side[s_] = AH.ArmSide(model, s_, parts=parts, cfg=cfg)
        cols = list(parts[s_]["arm_qadr"]) + [int(a) for a in arm_side[s_].hand.qadr]
        Q[:, cols] = np.asarray(r["qpos"])[:, cols]
        arms[s_] = np.asarray(r["arm"][s_], float)
        legs[s_], starts[s_] = r["legs"][s_], r["arm_start_used"][s_]
        report[s_] = r["report"][s_]
        report["infeasible"] += r["report"]["infeasible"]
        report["ok"] = bool(report["ok"] and r["report"]["ok"])
    changed = False
    if planner is not None and arms and T > 1:
        from tool_library.common import armplan as AP
        pl = AP.plan_arms(model, planner, Q, arms, legs, starts, labels, fps, park=False)
        report["plan"] = pl["report"]
        Q, arms, labels = pl["Qref"], pl["arms"], pl["labels"]
        changed = bool(pl["report"]["planned_legs"])
    Q, arms, labels, src, report["retimed_s"] = AH.retime_arms(model, Q, arms, labels, fps, cfg, parts, max_frames)
    if changed or report["retimed_s"]:
        for s_, A in arms.items():
            for t in range(len(Q)):
                Q[t] = arm_side[s_].put_hand(Q[t], A[t])
    for s_, A in arms.items():
        report[s_]["peak_speed_deg_s"] = round(float(np.degrees(np.abs(np.diff(A, axis=0)).max()) * float(fps)) if len(A) > 1 else 0.0, 1)
    return {"qpos": Q, "arm": arms, "labels": labels, "src_index": src, "report": report}


def _two_hand_phase_order(phases, labels):
    """phases.json for Stage 4 with two-hand phases. write_stage4_config walks the phases per hand: a phase another hand
    owns keeps this hand's current anchor, a change of its own anchor is a blend, and when a label is listed twice the
    later entry wins. So a two-hand phase lists first the hand whose anchor stays and last the hand whose anchor
    changes; when both hands' anchors change in the same phase (both reach for their parts at once), its frames are
    split in two: the first half carries one hand's change, the second half (the same phases named in the other order,
    'right: ... | left: ...') the other's. One-hand phases are left exactly as they are."""
    labels = list(labels)
    out, cur = [], {}
    norm = lambda p_: "world" if "return home" in str(p_["label"]).lower() else (p_.get("anchor") or "world")
    i = 0
    while i < len(phases):
        grp = [phases[i]]
        while i + len(grp) < len(phases) and phases[i + len(grp)]["label"] == phases[i]["label"] and phases[i + len(grp)].get("subgoal") == phases[i].get("subgoal"):
            grp.append(phases[i + len(grp)])
        i += len(grp)
        lab = grp[0]["label"]
        if len(grp) == 1 or " | " not in str(lab):
            for p_ in grp:
                out.append(p_); cur[p_.get("hand")] = norm(p_)
            continue
        change = [p_ for p_ in grp if norm(p_) != cur.get(p_.get("hand"), "world")]
        stay = [p_ for p_ in grp if p_ not in change]
        run = []
        if len(change) >= 2:
            t0 = next((t for t, l in enumerate(labels) if l == lab), None)
            if t0 is not None:
                run = [t0]
                while run[-1] + 1 < len(labels) and labels[run[-1] + 1] == lab:
                    run.append(run[-1] + 1)
        if len(change) < 2 or len(run) < 2:
            out += stay + change
        else:
            a, b = change[0], change[1]
            base, sfx = (str(lab).rsplit(" #", 1) + [None])[:2]
            parts_ = E.split_label(base)
            lab2 = E.combine_label(parts_, order=(b.get("hand"), a.get("hand"))) + (f" #{sfx}" if sfx else "")
            for t in run[len(run) // 2:]:
                labels[t] = lab2
            out += stay + [dict(b, anchor=None if cur.get(b.get("hand"), "world") == "world" else cur.get(b.get("hand"))), a]
            out += stay + [dict(a, label=lab2), dict(b, label=lab2)]
        for p_ in grp:
            cur[p_.get("hand")] = norm(p_)
    return out, labels


def _reach_message(sg, rep, jumps):
    inf = rep.get("infeasible") or []
    if inf:
        w = max(inf, key=lambda x: x["worst_mm"])
        msg = (f"the {w['side']} arm cannot hold the hand on its flange at '{w['label']}' (frames {w['frames'][0]}-{w['frames'][1]}: "
               f"{w['worst_mm']} mm / {w['worst_deg']} deg off)")
    else:
        s, j = next(iter(jumps.items()))
        msg = f"the {s} arm would have to flip its configuration mid-motion (frames {j[:3]})"
    return msg + ("; re-solve: a different grasp candidate (pose_candidates), optimize_pose_for_reach on the pose, a target nearer the "
                  "station's palm-down band, or turn the carried object (turn {axis, deg})")


def _pose_execution_evidence(model, q_start, sg, recon_path, sdir, runtime, variant):
    """Resolve exactly the planner's selected/re-anchored input for fresh tests."""
    from tool_library.common import subgoal as SG
    from tool_library.skills.hand_scene_from_recon import hand_scene_from_recon
    pose = copy.deepcopy(SG.load_pose(sg["pose"]))
    for key in ("open_frac", "finger_targets_open", "press_depth_rad", "press_travel_m"):
        if sg.get(key) is not None:
            pose[key] = sg[key]
    d = mujoco.MjData(model); d.qpos[:] = q_start; mujoco.mj_forward(model, d)
    anchor = SG.pose_anchor(sg, pose)
    pose = SG.reanchor_pose(model, d, pose, anchor, mode=sg.get("reanchor", "full"))
    if anchor:
        b = model.body(anchor["body"]).id
        quat = np.zeros(4); mujoco.mju_mat2Quat(quat, d.xmat[b])
        pose["anchor"] = {"body": anchor["body"], "pos": d.xpos[b].tolist(), "quat_wxyz": quat.tolist()}
    side = str(sg["side"])
    pose_path = sdir / f"pose_used_{side}.json"
    pose_path.write_text(json.dumps(pose, indent=1, default=float))
    state_path = sdir / "start_state_proposal.npz"
    names = np.array(E.joint_names(model))
    same_state = False
    if state_path.exists():
        try:
            with np.load(state_path, allow_pickle=False) as previous:
                same_state = (np.array_equal(previous["qpos"], np.asarray(q_start)[None])
                              and np.array_equal(previous["joint_names"], names)
                              and str(previous["artifact_role"]) == "proposal_start_state")
        except (ValueError, OSError, KeyError):
            pass
    # A changed earlier segment can move the object before this grasp. Replace
    # its old start-state evidence; the two hands in one segment share exactly
    # the same state file and digest when their starting coordinates agree.
    if not same_state:
        np.savez_compressed(state_path, qpos=np.asarray(q_start)[None], joint_names=names,
                            artifact_role=np.array("proposal_start_state"))
    evidence = {"path": str(pose_path), "sha256": hashlib.sha256(pose_path.read_bytes()).hexdigest(),
                "variant": variant, "reanchored": pose.get("reanchored"), "runtime_scene_sha256": runtime.scene_sha256,
                "start_state_proposal": str(state_path), "start_state_proposal_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
                "stress_validation_required": True,
                "reason": "fresh stress evidence must use this selected pose in its actual starting object state"}
    try:
        if not recon_path:
            raise ValueError("plan.scene_recon_json is required to generate the executed hand scene")
        rp = Path(recon_path).resolve(); recon = json.loads(rp.read_text())
        rm = mujoco.MjModel.from_xml_path(str((rp.parent / recon.get("scene_xml", "scene.xml")).resolve()))
        rq = RR.map_joints(q_start, model, rm)
        hp = sdir / f"hand_scene_used_{side}.json"
        target = sg.get("object") or (anchor or {}).get("body")
        hand_scene_from_recon(str(rp), side, target, grasp_body=(anchor or {}).get("body"), state=rq.tolist(), out_json=str(hp))
        evidence.update(hand_scene_used=str(hp), hand_scene_used_sha256=hashlib.sha256(hp.read_bytes()).hexdigest())
    except (ValueError, OSError, KeyError) as exc:
        evidence["hand_scene_error"] = str(exc)
    return evidence


def _join_runtime_segments(out, segs, labels, subgoals, fps, runtime):
    """Concatenate recorded physical states; remove duplicate boundary samples.

    Every integration state/control is retained unchanged. Absolute times are
    formed by adding the elapsed time of the earlier segments, not by guessing
    a video-frame duration for a 16/17-substep interval.
    """
    per_frame = ("qpos", "qvel", "ctrl", "act", "reference", "time_s", "eq_active", "track_pos_m", "track_rot_deg",
                 "hand_pen_m", "object_pen_m", "wrist_force_n", "wrist_torque_nm", "hand_object_force_n", "arm_tracking_rad")
    per_step = ("substep_qpos", "substep_qvel", "substep_ctrl", "substep_time_s", "substep_eq_active",
                "substep_act", "substep_eq_active_before", "substep_eq_data")
    rows = {k: [] for k in per_frame + per_step}
    base, previous, first = 0., None, None
    for i, (tag, _ep) in enumerate(segs):
        # A legacy segment can have a new semantic tag while its immutable
        # recording still lives in the old folder. Read the file we resumed.
        recording_path = _ep.get("_runtime_path", out / tag / "rollout_runtime.npz")
        with np.load(recording_path, allow_pickle=False) as z:
            rec = {k: z[k].copy() for k in z.files}
        if str(rec.get("robot_schema")) != RR.SCHEMA or str(rec.get("scene_sha256")) != runtime.scene_sha256:
            raise ValueError(f"{tag}: physical scene/schema changed between segments")
        if first is None:
            first = rec
        if previous is not None:
            if not np.array_equal(rec["initial_qpos"], previous["qpos"][-1]) or not np.array_equal(rec["initial_qvel"], previous["qvel"][-1]):
                raise ValueError(f"{tag}: physical state or velocity is discontinuous at the segment boundary")
        for k in per_frame:
            value = rec[k][1 if i else 0:].copy()
            if k == "time_s":
                value += base
            rows[k].append(value)
        for k in per_step:
            value = rec[k].copy()
            if k == "substep_time_s":
                value += base
            rows[k].append(value)
        base += float(rec["time_s"][-1])
        previous = rec
    arrays = {k: np.concatenate(v, axis=0) for k, v in rows.items()}
    for k in ("initial_qpos", "initial_qvel", "initial_act", "initial_state", "state_spec", "initial_time_s", "eq_data_start",
              "eq_active_start", "eq_names", "joint_names", "actuator_names", "scene_sha256", "robot_schema"):
        arrays[k] = first[k]
    arrays["final_state"] = previous["final_state"]
    arrays["mechanism_initial"] = first["mechanism_initial"]
    arrays["mechanism_final"] = previous["mechanism_final"]
    arrays.update(labels=np.array(labels), subgoal=np.array(subgoals), fps=float(fps))
    if len(arrays["qpos"]) != len(labels):
        raise ValueError("runtime frames and semantic labels do not match")
    return arrays


def _rel(path, base):
    p = Path(path)
    if base is not None and not p.is_absolute() and not p.exists() and (base / p).exists():
        return str(base / p)
    return str(path)


def _label_suffix(tag):
    """Phase-label suffix of an executed segment: '#3' for subgoal_03, '#3.2' for its step subgoal_03_02."""
    parts = str(tag).split("_")
    return f"#{int(parts[1])}" if len(parts) < 3 else f"#{int(parts[1])}.{int(parts[2])}"


def thread_rules_from_recon(scene_recon_json):
    """Screw-cap release rules from a Stage 2 manifest: objects that expose the bottle contract
    (equalities.engaged, joints.lift, params thread_pitch x thread_turns)."""
    man = json.loads(Path(scene_recon_json).read_text())
    rules = []
    for o in man.get("objects", []):
        eqs, js = o.get("equalities") or {}, o.get("joints") or {}
        if not isinstance(eqs, dict) or not isinstance(js, dict):         # a plain list of joint names has no lift/engaged roles
            continue
        eq = eqs.get("engaged"); lj = js.get("lift")
        pr = o.get("params") or {}
        if eq and lj and pr.get("thread_pitch") and pr.get("thread_turns"):
            rules.append({"engaged": eq, "lift_joint": lj, "travel_m": float(pr["thread_pitch"]) * float(pr["thread_turns"]), "frac": 0.96})
        elif eq and lj and pr.get("release_travel_m"):    # a press-fit part (a marker cap) let go after its travel
            rules.append({"engaged": eq, "lift_joint": lj, "travel_m": float(pr["release_travel_m"]), "frac": 0.96})
    return rules


def cut_rules_from_recon(scene_recon_json):
    """Scissor cut rules from a Stage 2 manifest: every object's `cut_joins` (construct_cuttable_rope's
    scene_fields: [{eq, body, anchor_local, diameter_m}]) against every pair of scissors in it (a construct_scissors
    entry: its recorded `cutter`, or one derived from its params, with the scene prefix of its root body). Empty when
    the scene has no cut joins or no scissors."""
    from tool_library.common import scissors as SC
    man = json.loads(Path(scene_recon_json).read_text())
    objs = [o for o in man.get("objects", []) if isinstance(o, dict)]
    joins = [dict(j) for o in objs for j in (o.get("cut_joins") or []) if isinstance(j, dict) and j.get("eq") and j.get("body")]
    if not joins:
        return []
    cutters = [c for c in (SC.cutter_from_recon(o) for o in objs) if c]
    return [{"kind": "cut", "cutter": c, "joins": joins, "gates": "cord"} for c in cutters]


def latch_rules_from_recon(scene_recon_json):
    """Connector latch rules from a Stage 2 manifest: every object entry's `latch` (add_connector_latch's scene_fields:
    {eq, site_a, site_b, seat_m, release_N}, or a list of them for a cable with two plugs) as {kind: 'latch', ...}. Empty
    when no entry carries one."""
    man = json.loads(Path(scene_recon_json).read_text())
    return E.latch_rules_from_objects(man.get("objects", []))


def _eq_record(segs, model):
    """The whole episode's equality state (eq_active [T, neq] after each frame, eq_active_start, eq_names) from the
    segments that recorded it, or None when none did."""
    if not any("eq_active" in ep for _, ep in segs):
        return None
    names = next(list(map(str, ep["eq_names"])) for _, ep in segs if "eq_names" in ep)
    rows, last = [], None
    start = next((np.asarray(ep["eq_active_start"], np.int8) for _, ep in segs if "eq_active_start" in ep), None)
    for _, ep in segs:
        if "eq_active" in ep:
            a = np.asarray(ep["eq_active"], np.int8)
        else:                                     # a segment run before cutting was configured: the state carried
            a = np.tile(last if last is not None else model.eq_active0.astype(np.int8), (len(ep["qpos"]), 1))
        rows.append(a); last = a[-1]
    EQ = np.vstack(rows)
    if start is None:
        start = EQ[0]
    return {"eq_active": EQ, "eq_active_start": start, "eq_names": np.array(names)}


def execute_subgoal_plan(scene, plan, out_dir, fps=30.0, resume=False, from_subgoal=None, stop_on_fail=True, fill_arms=True,
                         wrist_kp=10000.0, max_rate_deg_s=90.0, settle_s=0.0, require_reach=True, reach_tol_m=0.03,
                         couple_arms=True, arm_speed_deg_s=90.0, project_tol_mm=6.0, project_tol_deg=4.0, weld_tol_mm=2.0, weld_tol_deg=2.0,
                         arm_collision_tol_m=0.003, max_attempts=6, auto_turn=True, auto_reach_pose=True, dry_run=False, plan_arm_paths=True):
    # Bind the controls and planner settings, independently of output paths or
    # which completed prefix the caller asks to retain.
    resume_options = {k: v for k, v in locals().copy().items()
                      if k not in ("scene", "plan", "out_dir", "resume", "from_subgoal", "stop_on_fail", "dry_run")}
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    if settle_s:
        raise ValueError("unrecorded pre-episode settling is forbidden; include settling as an explicit initial hold subgoal")
    subgoals, rules = _load_plan(plan)
    # Reusing an output folder starts a new result. A failure before the first
    # physical segment (or a dry run) must not leave an earlier trajectory at
    # the current result path. Keep it as explicitly unvalidated diagnostics.
    prior_trajectory = out / "trajectory.npz"
    if prior_trajectory.exists():
        import time
        prior_trajectory.rename(out / f"trajectory.previous_unvalidated.{time.time_ns()}.npz")
    plan_config = json.loads(Path(plan).read_text()) if isinstance(plan, (str, Path)) else plan
    recon_path = (plan_config.get("scene_recon_json") if isinstance(plan_config, dict) else None)
    if recon_path:
        recon_path = _rel(recon_path, Path(plan).resolve().parent if isinstance(plan, (str, Path)) else None)
    scene = str(Path(scene).resolve())
    if not couple_arms:
        raise ValueError("Stage 3 requires the rigid arm-hand robot; couple_arms=False is only a legacy proposal mode")
    model, n_pairs = E.build_execution_model(scene, out_xml=str(out / "scene_proposal.xml"), scene_recon_json=recon_path)
    if RR.is_rigid(model):
        raise ValueError("execute_subgoal_plan takes a proposal scene; a rigid scene is replayed with servo_rollout")
    runtime = RR.compile_rigid(model)
    deformable_bodies = _deformable_body_ids(model, recon_path) if dry_run else set()
    binding_cache = {}
    segment_signatures = {sg["_tag"]: _resume_signature(sg, rules, resume_options, runtime.scene_sha256, binding_cache) for sg in subgoals}
    (out / "scene_physics.xml").write_text(runtime.xml)
    qvel_phys = np.zeros(runtime.model.nv)
    physics_state = None
    mechanism_state = None
    parts = E.robot_parts(model)
    d = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, d, 0)
    q_home = d.qpos.copy()
    # ---- resume bookkeeping (segments are keyed by their folder tag, unique per plan entry: see _assign_tags)
    def tag_of(sg):
        return sg["_tag"]
    done = {}
    state = {"held": {}, "grip": {}, "eq_inactive": [], "latched": {}}
    q_phys, q_cmd = q_home.copy(), q_home.copy()
    segs = []
    start_k = 0
    if resume and (out / "outcomes.json").exists():
        prev = {r.get("tag") or f"subgoal_{int(r['index']):02d}": r for r in json.loads((out / "outcomes.json").read_text())}
        for sg in subgoals:
            tg = tag_of(sg); rec = prev.get(tg)
            seg_d = out / tg
            if not (rec and rec.get("ok") and (seg_d / "rollout.npz").exists()):
                leg = _legacy_segment(out, sg, subgoals)       # a run written before every entry had its own folder
                if leg is not None:
                    rec, seg_d = leg
            if (rec and rec.get("ok") and not rec.get("dry_run")
                    and rec.get("resume_signature") == segment_signatures[tg]
                    and (seg_d / "rollout.npz").exists() and (seg_d / "rollout_runtime.npz").exists()
                    and (from_subgoal is None or int(sg["index"]) < int(from_subgoal))):
                rec = dict(rec, tag=tg)
                done[tg] = rec
                ep = E.load_episode(seg_d / "rollout.npz"); ep["phases"] = json.loads((seg_d / "phases.json").read_text())
                ep["_runtime_path"] = str(seg_d / "rollout_runtime.npz")
                segs.append((tg, ep)); q_phys = ep["qpos"][-1].copy(); q_cmd = ep["reference"][-1].copy()
                with np.load(seg_d / "rollout_runtime.npz", allow_pickle=False) as rz:
                    if str(rz["scene_sha256"]) != runtime.scene_sha256:
                        raise ValueError("resume runtime scene changed; old physical segments cannot be reused")
                    qvel_phys = rz["qvel"][-1].copy()
                    physics_state = rz["final_state"].copy()
                    mechanism_state = json.loads(str(rz["mechanism_final"])) if "mechanism_final" in rz else None
                st = seg_d / "state.json"
                if st.exists():
                    state = json.loads(st.read_text())
                start_k += 1
            else:
                break
    settle_resets = 0
    outcomes = [done[t] for t, _ in segs]
    first_fail = None
    probe = AH.ArmProbe(scene, tol_m=arm_collision_tol_m) if couple_arms else None
    # collision-aware arm planning: free-space legs routed around what a straight joint path runs into, idle arms parked
    # out of the active hand's way (armplan; a no-op where the straight plan is clean)
    arm_planner = AP.ArmPlanner(scene) if couple_arms and plan_arm_paths else None
    ah_cfg = {"project_pos_m": float(project_tol_mm) / 1000.0, "project_rot_deg": float(project_tol_deg), "max_speed_deg_s": float(arm_speed_deg_s)}
    arm_state = {s: q_phys[pt["arm_qadr"]].copy() for s, pt in parts.items() if pt["gripper"] >= 0 and pt["arm_joints"]}
    bite_pose = {}                                   # subgoal index -> the pose its first bite succeeded with
    unknown_dry_state = None
    for sg in subgoals[start_k:]:
        k = int(sg["index"]); tag = tag_of(sg); sdir = out / tag; sdir.mkdir(exist_ok=True)
        if unknown_dry_state is not None:
            rec = {"index": k, "step": sg.get("step"), "tag": tag, "motion": sg.get("motion"),
                   "side": _side_of(sg), "object": _object_of(sg), "ok": False, "dry_run": True,
                   "failed": ["state_requires_physics"], "state_dependency": unknown_dry_state,
                   "error": "Dry-run cannot predict a deformable's state after contact. Execute the physical prefix "
                            "and resume from its recorded state before screening the next subgoal.",
                   "resume_signature": segment_signatures[tag]}
            (sdir / "outcome.json").write_text(json.dumps(rec, indent=1))
            outcomes.append(rec); first_fail = rec
            break  # later collision checks also depend on the unknown geometry
        if str(sg.get("motion", "")).endswith("_bite") and int(sg.get("step") or 1) > 1:
            # later bites re-grasp with the pose the first bite used (a ranked candidate list picks one there)
            first_bite = next((s for s in subgoals if int(s["index"]) == k and int(s.get("step") or 1) == 1), None)
            used = out / tag_of(first_bite) / "pose_used.json" if first_bite is not None else None
            bp = bite_pose.get(k) or (json.loads(used.read_text()) if used is not None and used.exists() else None)
            if bp is not None:
                sg = dict(sg, regrasp_pose=bp)
        both = B.is_two_hand(sg)                     # both, lift_together, carry_together: two hands in one rollout
        cands = [] if both else _pose_candidates(sg)
        screen = None
        if both and not sg.get("hands"):
            variants = [(dict(sg), "as planned")]
        elif both:
            variants, screen = _both_variants(model, sg, q_phys, probe, arm_state, ah_cfg, parts, couple_arms)
        elif cands and couple_arms:
            # Reachable candidates first; the complete planner can still repair
            # a failed advisory screen with contact correction/path planning.
            screen = _screen_entry_poses(model, sg["side"], cands, sg, q_phys, probe=probe,
                                         arm_seed=arm_state.get(sg["side"]), cfg=ah_cfg, parts=parts)
            order = [r["index"] for r in screen if r["ok"]]
            order += [r["index"] for r in sorted((r for r in screen if not r["ok"] and r.get("duplicate_of") is None and "residual_mm" in r),
                                                   key=lambda r: (bool(r.get("arm_hits")), r["residual_mm"] + r["residual_deg"]))]
            variants = [(dict(sg, pose=cands[i]), f"pose candidate {i + 1}") for i in order]
        elif cands:
            variants = [(dict(sg, pose=c), f"pose candidate {i + 1}") for i, c in enumerate(cands)]
        else:
            variants = [(dict(sg), "as planned")]
        attempts, reach_added, done_ok, last = [], False, None, None
        low_added = False
        dry_done = None
        vi = 0
        while vi < len(variants) and len(attempts) < int(max_attempts):
            sgv, note = variants[vi]; vi += 1
            if sgv.get("_skip"):
                attempts.append({"variant": note, "failed": ["reach_turn_too_large"], "error": sgv["_skip"]})
                continue
            try:
                planned = plan_subgoal_keyframes(model, sgv, qpos0=q_phys.tolist(), state=copy.deepcopy(state), command_qpos=q_cmd.tolist(), fps=fps,
                                                 out_json=str(sdir / "keyframes.json"), auto_via=not couple_arms,
                                                 max_rate_deg_s=max_rate_deg_s, max_rate_m_s=0.4)
            except Exception as e:                       # a planning error is a result, not a crash
                attempts.append({"variant": note, "failed": ["planning"], "error": f"{type(e).__name__}: {e}"})
                continue
            # a track_reference entry whose rows' arms hold their hands is already timed (by the execution or the method
            # that wrote it): its rows run one per frame, unstretched; the arm speed is judged after physics as always
            given_arms = (planned.get("rollout") or {}).get("arms") == "reference"
            ref = chain_subgoals(model, planned["keyframes"], fps=fps, max_rate_deg_s=None if given_arms else max_rate_deg_s,
                                 max_rate_m_s=None if given_arms else 0.4)
            cp = None
            if couple_arms:
                # the arm is part of the plan: dense IK along the hand reference, free legs in arm joint space, the hand
                # put on the flange, too-fast legs slowed down; what the arm cannot hold is re-solved, not executed
                if given_arms:
                    # a track_reference entry whose rows' arms hold their hands: run as given, the hands put on the flanges
                    from tool_library.common.tracking import couple_as_given
                    cp = couple_as_given(model, ref["qpos"], ref["labels"], fps, parts)
                else:
                    cp = (_couple_two_hands if both else AH.couple)(model, ref["qpos"], fps, arm_start=arm_state, labels=ref["labels"],
                                                                    keyframe_frames=ref["keyframe_frames"], cfg=ah_cfg, parts=parts, max_frames=3000,
                                                                    planner=arm_planner)
                rep_c = cp["report"]
                if both or len(parts) > 1:
                    # the two hands first: a plan that drives one hand through the other is not fixed by any arm re-solve
                    gap = _hand_gap_failure(model, cp["qpos"], cp["labels"], planned["expect"], note, need=None if both else -ONE_HAND_OVERLAP_M)
                    if gap is not None:
                        attempts.append(gap)
                        continue
                jumps = {s: v["branch_jumps"] for s, v in rep_c.items() if isinstance(v, dict) and v.get("branch_jumps")}
                if require_reach and (rep_c["infeasible"] or jumps):
                    att = {"variant": note, "failed": ["arm_unreachable"], "infeasible": rep_c["infeasible"][:4], "branch_jumps": jumps,
                           "error": _reach_message(sgv, rep_c, jumps)}
                    attempts.append(att)
                    if not reach_added:
                        reach_added = True
                        variants += (_both_reach_variants(scene, sgv, {i_["side"] for i_ in rep_c["infeasible"]} | set(jumps), auto_turn, auto_reach_pose)
                                     if both else _reach_variants(scene, sgv, auto_turn=auto_turn, auto_reach_pose=auto_reach_pose))
                    continue
                hits = probe.check(cp["qpos"], cp["labels"], stride=3)
                if require_reach and hits:
                    attempts.append({"variant": note, "failed": ["arm_collision"], "arm_collisions": hits,
                                     "error": f"the arm runs into {hits[0]['geoms'][1] if 'arm' in hits[0]['geoms'][0] else hits[0]['geoms'][0]} "
                                              f"({hits[0]['depth_mm']} mm) at '{hits[0]['label']}': approach from another side or re-orient the pose"})
                    if not reach_added:
                        reach_added = True
                        variants += (_both_reach_variants(scene, sgv, {s_ for h_ in hits for s_ in ("left", "right") if any(f"robot_{s_}" in g_ for g_ in h_["geoms"])},
                                                          auto_turn, auto_reach_pose)
                                     if both else _reach_variants(scene, sgv, auto_turn=auto_turn, auto_reach_pose=auto_reach_pose))
                    continue
                reference = {"qpos": cp["qpos"], "labels": cp["labels"], "stretched": ref["stretched"]}
                arm_arg = cp["arm"]
                if dry_run:
                    # the executor's own arm feasibility (plan, couple, re-solve variants, arm screen) without physics:
                    # the next subgoal starts where this plan ends, objects where the plan carried them
                    dry_done = {"variant": note, "planned": planned, "cp": cp, "report": {s_: {k_: v_ for k_, v_ in r_.items() if k_ in ("ik_residual_mm_max", "ik_residual_deg_max", "projected_frames", "free_leg_frames")}
                                                                                      for s_, r_ in rep_c.items() if isinstance(r_, dict)}}
                    break
            else:
                reference = {"qpos": ref["qpos"], "labels": ref["labels"], "stretched": ref["stretched"]}
                arm_arg = "home"
                if both:
                    gap = _hand_gap_failure(model, ref["qpos"], ref["labels"], planned["expect"], note)
                    if gap is not None:
                        attempts.append(gap)
                        continue
            # Retry only timing, from exactly the same simulator/mechanism state.
            # The pristine expectation must be resolved against each trial's labels.
            timing_trial, timing_scale = 0, 1
            timing_note = note
            expect_template = copy.deepcopy(planned["expect"])
            trial_state, trial_mechanism = physics_state, mechanism_state
            trial_eq_data = runtime.model.eq_data.copy()
            while True:
                runtime.model.eq_data[:] = trial_eq_data
                try:
                    # the wrist force cap: the entry's own, else what the motion asks for (press_seat), else a light lateral cap
                    # for screw bites
                    lateral = sgv.get("wrist_force_n")
                    if lateral is None:
                        lateral = (planned.get("rollout") or {}).get("wrist_force_n")
                    if lateral is None and sgv.get("motion", "").endswith("_bite"):
                        lateral = [10.0, 10.0, 40.0]
                    lateral = B.with_pin_caps(lateral, state)   # a pressing pin keeps its press cap while the other hand acts
                    roll = rollout_keyframes_physics(model, reference=reference, qpos_start=q_phys.tolist(), fps=fps, wrist_kp=wrist_kp,
                                                     max_rate_deg_s=max_rate_deg_s, out_npz=str(sdir / "rollout.npz"), subgoal_index=k, model=model, return_diag=True,
                                                     eq_inactive=state.get("eq_inactive", []), latched=state.get("latched") or {}, rules=rules, arm=arm_arg,
                                                     finger_force_nm=sgv.get("finger_force_nm"), wrist_force_n=lateral, wrist_torque_nm=sgv.get("wrist_torque_nm"),
                                                     allowed_bodies=_allowed_bodies(planned, sgv), qvel_start=qvel_phys, state_start=trial_state,
                                                     mechanism_state_start=trial_mechanism)
                except ValueError as e:                    # an over-long reference: report it as the subgoal's failure
                    attempts.append({"variant": note, "failed": ["reference_too_long"], "error": str(e)})
                    roll = None
                    break
                diag, labels = roll.pop("_diag"), roll.pop("_labels")
                Q = diag["qpos"]
                # phase-named frames (the pads' check frame, the first lift/carry frame for slip), relative to the subgoal's first
                # frame so they hold on the full trajectory too; per hand for a two-hand entry
                O.resolve_label_frames(planned["expect"], labels)
                if roll.get("released") and sgv.get("motion") in ("unscrew", "unscrew_bite") and "turned_rad" in planned["expect"]:
                    # the thread let the cap go during this bite: the bite is judged by the release, not by a full turn
                    rel = {r_["equality"] for r_ in roll["released"]}
                    rr = next((r_ for r_ in rules if r_["engaged"] in rel), None)
                    if rr is not None:
                        planned["expect"]["released_by"] = {"joint": rr["lift_joint"], "travel_m": round(float(rr["travel_m"]) * float(rr.get("frac", 0.96)), 6)}
                ev = O.evaluate(model, Q, fps, list(range(len(Q))), planned["expect"], diagnostics=diag, parts=parts)
                if roll.get("plan_collisions"):
                    # the kinematic plan sweeps a hand through something. When the physics shows the hand was really
                    # forced (tracking error, penetration, large force) it is a failure; when the contact resolved by
                    # compliance (a thumb sliding off a wall as the lid closes) it is a warning the agent should still read.
                    forced = (ev["measured"].get("max_track_pos_mm", 0) > 20) or (ev["measured"].get("max_hand_pen_mm", 0) > 4) or (not ev["ok"])
                    ev["measured"]["plan_collisions"] = roll["plan_collisions"]
                    if forced:
                        ev["ok"] = False; ev["failed"].append("plan_collision")
                    else:
                        ev["measured"]["plan_collision_warning"] = "the plan sweeps a hand through a bystander; physics resolved it by compliance"
                if roll.get("unstable"):
                    ev["ok"] = False; ev["failed"].append("simulation_unstable"); ev["measured"]["unstable"] = roll["unstable"]
                if couple_arms:
                    # These are actual motor-driven arm states. The compatibility
                    # hand coordinates came from FK of this same rigid tree.
                    W = AH.weld_residuals(model, Q, parts=parts)
                    speeds = AT.runtime_speed_peaks(runtime.model, diag, fps)
                    fit = {}
                    for side_, (pe, re, src) in W.items():
                        fit[side_] = {"residual_mm_max": float(pe.max()), "residual_deg_max": float(re.max()),
                                      **speeds[side_], "mount": "fixed runtime flange"}
                    ev["measured"]["arm"] = {"plan": {s: v for s, v in cp["report"].items() if isinstance(v, dict)},
                                              "retimed_s": cp["report"]["retimed_s"], "runtime": fit,
                                              "timing_retries": cp["report"].get("timing_retries", [])}
                    if any(v["residual_mm_max"] > weld_tol_mm or v["residual_deg_max"] > weld_tol_deg for v in fit.values()):
                        ev["ok"] = False; ev["failed"].append("arm_weld")
                    fast = {s: v["peak_speed_deg_s"] for s, v in fit.items() if v["peak_speed_deg_s"] > 1.5 * arm_speed_deg_s}
                    if fast:
                        ev["ok"] = False; ev["failed"].append("arm_speed"); ev["measured"]["arm_speed"] = fast
                    hits = probe.check(Q, labels, stride=1)
                    if hits:
                        ev["ok"] = False; ev["failed"].append("arm_collision"); ev["measured"]["arm_collisions"] = hits
                factor = AT.retry_factor(ev, cp, 1.5 * arm_speed_deg_s, timing_trial,
                                         int(max_attempts) - len(attempts) - 1)
                if factor is None:
                    break
                try:
                    slower = AT.slow_reference(model, cp, fps, factor, max_frames=3000)
                except ValueError as error:
                    ev["measured"]["timing_retry_unavailable"] = str(error)
                    break
                archive = AT.archive_failed_trial(sdir, ev, cp, factor, len(attempts) + 1)
                attempts.append({"variant": note, "failed": list(ev["failed"]),
                                 "measured": copy.deepcopy(ev["measured"]), "files": {"failed_trial": archive}})
                history = list(cp["report"].get("timing_retries", []))
                history.append({"factor": factor, "failed_speed_deg_s": dict(ev["measured"]["arm_speed"]),
                                "failed_trial": archive})
                slower["report"]["timing_retries"] = history
                cp = slower
                reference = {"qpos": cp["qpos"], "labels": cp["labels"], "stretched": ref["stretched"]}
                arm_arg = cp["arm"]
                # Includes velocities, warmstart, controls, actuator state, time,
                # equality flags and the mechanism's independent bookkeeping.
                trial_state = diag["initial_state"].copy()
                trial_mechanism = copy.deepcopy(diag["mechanism_initial"])
                trial_eq_data = diag["eq_data_start"].copy()
                timing_trial += 1
                timing_scale *= factor
                planned["expect"] = AT.scale_expectation_frames(expect_template, timing_scale)
                note = f"{timing_note}; timing retry {timing_trial} ({timing_scale}x duration)"
            if roll is None:
                break
            if ev["ok"]:
                done_ok = (sgv, note, planned, roll, ev, Q, labels)
                break
            attempts.append({"variant": note, "failed": list(ev["failed"]), "measured": {kk: v for kk, v in ev["measured"].items() if not isinstance(v, (list, dict))}})
            if "arm_speed" in ev["failed"] and not low_added:
                low_added = True
                variants += _low_carry_variants(sgv)
            last = (sgv, note, planned, roll, ev, Q, labels)
        if dry_run:
            if dry_done is not None:
                planned_d, cp_d = dry_done["planned"], dry_done["cp"]
                rec = {"index": k, "step": sg.get("step"), "tag": tag, "motion": sg.get("motion"), "side": _side_of(sg), "object": _object_of(sg),
                       "ok": True, "dry_run": True, "variant": dry_done["variant"], "attempts": attempts, "pose_screen": screen,
                       "arm": dry_done["report"], "n_frames": int(len(cp_d["qpos"])), "phases": [p_["label"] for p_ in planned_d["phases"]]}
                if planned_d.get("approach_controls"):
                    rec["approach_controls"] = planned_d["approach_controls"]
                contacts = _dry_deformable_contacts(model, deformable_bodies, planned_d, sg, state)
                rec["validation_scope"] = "kinematic_arm_screen_only"
                rec["terminal_state_valid"] = not bool(contacts)
                if contacts:
                    unknown_dry_state = {"tag": tag, "objects": contacts}
                    rec["next_step"] = "Execute this prefix in physics before planning from its terminal state."
                else:
                    state = planned_d["state"]
                    q_phys = np.asarray(cp_d["qpos"][-1], float).copy(); q_cmd = q_phys.copy()
                    for s_ in arm_state:
                        arm_state[s_] = q_phys[parts[s_]["arm_qadr"]].copy()
            else:
                a = attempts[-1] if attempts else {"failed": ["planning"], "error": "no variant"}
                rec = {"index": k, "step": sg.get("step"), "tag": tag, "motion": sg.get("motion"), "side": _side_of(sg), "object": _object_of(sg),
                       "ok": False, "dry_run": True, "failed": a["failed"], "error": a.get("error"), "attempts": attempts, "pose_screen": screen}
                first_fail = first_fail or rec
            rec["resume_signature"] = segment_signatures[tag]
            (sdir / "outcome.json").write_text(json.dumps(rec, indent=1, default=float))
            outcomes.append(rec)
            if not rec["ok"] and stop_on_fail:
                break
            continue
        result = done_ok or last
        if result is None:
            # nothing reached physics (or the last attempt failed before it): report the last attempt
            a = attempts[-1] if attempts else {"failed": ["planning"], "error": "no variant"}
            rec = {"index": k, "step": sg.get("step"), "tag": tag, "motion": sg.get("motion"), "side": _side_of(sg), "object": _object_of(sg), "ok": False,
                   "failed": a["failed"], "error": a.get("error"), "frames": None, "attempts": attempts, "pose_screen": screen,
                   "measured": {kk: a[kk] for kk in ("infeasible", "branch_jumps", "arm_collisions", "hand_clearance") if kk in a}}
            rec["resume_signature"] = segment_signatures[tag]
            (sdir / "outcome.json").write_text(json.dumps(rec, indent=1, default=float))
            outcomes.append(rec); first_fail = rec
            if stop_on_fail or "reference_too_long" in a["failed"]:
                break
            continue
        sgv, note, planned, roll, ev, Q, labels = result
        reach = planned.get("arm_reach")
        if reach and not reach.get("ok", True) and not couple_arms:
            ev["measured"]["arm_reach_warning"] = {"worst_pos_err_mm": reach["worst_pos_err_mm"], "worst_rot_err_deg": reach["worst_rot_err_deg"], "unreachable": reach["unreachable"][:4]}
        rec = {"index": k, "step": sg.get("step"), "tag": tag, "motion": sg.get("motion"), "side": _side_of(sg),
               "object": planned["expect"].get("object") if not both else planned["expect"].get("objects"), "ok": ev["ok"],
               "failed": ev["failed"], "checks": ev["checks"], "measured": ev["measured"], "n_frames": len(Q), "arm_reach": reach,
               "expect": planned["expect"], "phases": [p["label"] for p in planned["phases"]], "stretched": roll["stretched"],
               "variant": note, "attempts": attempts, "pose_screen": screen}
        if planned.get("approach_controls"):
            rec["approach_controls"] = planned["approach_controls"]
        rec["stage1_indices"] = list(sgv.get("stage1_indices") or [k])
        # Certification must use the executed action, not a later editable
        # manifest label. In particular, each hand of `both` has its own grasp
        # stress requirements and release semantics.
        rec["release"] = sgv.get("release", False)
        if str(sgv.get("motion")) == "both":
            rec["hands"] = {
                side: {"motion": entry["motion"],
                       "object": (planned["expect"].get("objects") or {}).get(side) or entry.get("object"),
                       "release": entry.get("release", False)}
                for side, entry in sgv["hands"].items()}
        rec["resume_signature"] = segment_signatures[tag]
        if sgv.get("pose") is not None:
            proof = _pose_execution_evidence(model, q_phys, sgv, recon_path, sdir, runtime, note)
            rec["pose_used"], rec["selected_pose_evidence"] = proof["path"], proof
            # Compatibility name for multi-bite continuation; still the exact
            # selected/re-anchored pose, never the rejected original candidate.
            shutil.copyfile(proof["path"], sdir / "pose_used.json")
        if both:
            for s_, e_ in (sgv.get("hands") or {}).items():
                if e_.get("pose") is not None:
                    proof = _pose_execution_evidence(model, q_phys, dict(e_, side=s_), recon_path, sdir, runtime, note)
                    rec.setdefault("pose_used", {})[s_] = proof["path"]
                    rec.setdefault("selected_pose_evidence", {})[s_] = proof
        if note != "as planned" and sgv.get("turn"):
            rec["turn_used"] = sgv["turn"]
        if ev["ok"] and str(sg.get("motion", "")).endswith("_bite") and int(sg.get("step") or 1) == 1 and isinstance(sgv.get("pose"), dict):
            bite_pose[k] = sgv["pose"]
        (sdir / "outcome.json").write_text(json.dumps(rec, indent=1, default=float))
        # run-time mechanism state the next subgoal starts from: released constraints (and pulled-out latches, disarmed
        # until their plug has left the port) and the latches engaged at the end, with their relpose
        state = planned["state"]; state["eq_inactive"] = roll["eq_inactive"]; state["latched"] = roll.get("latched", {})
        rec["released"] = roll["released"]
        if roll.get("latch_events"):
            rec["latches"] = roll["latch_events"]
        (sdir / "state.json").write_text(json.dumps(state, indent=1, default=float))
        (sdir / "phases.json").write_text(json.dumps(planned["phases"], indent=1))
        outcomes.append(rec)
        ep = E.load_episode(sdir / "rollout.npz"); ep["phases"] = planned["phases"]
        segs.append((tag, ep))
        q_phys = Q[-1].copy(); q_cmd = np.asarray(ep["reference"])[-1].copy()
        with np.load(sdir / "rollout_runtime.npz") as rz:
            qvel_phys = rz["qvel"][-1].copy()
            physics_state = rz["final_state"].copy()
            mechanism_state = json.loads(str(rz["mechanism_final"]))
        for s_ in arm_state:
            arm_state[s_] = q_phys[parts[s_]["arm_qadr"]].copy()
        if not ev["ok"]:
            first_fail = rec
            if stop_on_fail or roll.get("unstable"):      # nothing after an unstable segment means anything
                break
    (out / "outcomes.json").write_text(json.dumps(outcomes, indent=1, default=float))
    # ---- concatenate
    summary = {"ok": first_fail is None and len(outcomes) == len(subgoals), "dry_run": bool(dry_run), "n_subgoals": len(subgoals), "subgoals": [
        {k: v for k, v in r.items() if k in ("index", "step", "stage1_indices", "tag", "motion", "side", "object", "hands", "release", "ok", "failed", "measured", "n_frames", "frames", "error", "arm_reach", "dry_run", "variant", "pose_used", "selected_pose_evidence", "validation_scope", "terminal_state_valid", "state_dependency", "next_step", "approach_controls")} for r in outcomes],
        "first_failure": None if first_fail is None else {k: v for k, v in first_fail.items() if k in ("index", "step", "motion", "side", "object", "failed", "measured", "error", "expect", "approach_controls")},
        "files": {"scene_physics": str(out / "scene_physics.xml"), "outcomes": str(out / "outcomes.json")}, "contact_pairs": n_pairs,
        "settle_resets": settle_resets}
    if segs:
        # Each rollout includes its initial state. Only the first is a new
        # observation; a later segment starts at the preceding segment's end.
        for i, (tg, ep) in enumerate(segs):
            if i:
                for field in ("qpos", "labels", "subgoal", "reference", "eq_active"):
                    if field in ep:
                        ep[field] = ep[field][1:]
        Qs, Ls, Ss, Rs = [], [], [], []
        phases = []
        for tg, ep in segs:
            k = int(tg.split("_")[1])
            # steps of one subgoal (subgoal_01_05) get their own suffix: two steps doing the same phase on the same part
            # ('left: close lid' before and after the other hand lets go) are different phases for Stage 4
            sfx = _label_suffix(tg)
            Qs.append(ep["qpos"]); Ls += [f"{l} {sfx}" for l in ep["labels"]]; Ss += [k] * len(ep["qpos"]); Rs.append(np.asarray(ep["reference"]))
            for p in ep["phases"]:
                phases.append({**p, "label": f"{p['label']} {sfx}", "subgoal": k})
        Q = np.vstack(Qs); R = np.vstack(Rs)
        # every executed entry's own frames in the concatenated trajectory: a manifest judges each entry on its window,
        # not on its whole Stage 1 subgoal
        t0_, span = 0, {}
        for tg, ep in segs:
            span[tg] = [t0_, t0_ + len(ep["qpos"]) - 1]; t0_ += len(ep["qpos"])
        for r_ in outcomes:
            if r_.get("tag") in span:
                r_["frames"] = span[r_["tag"]]
        for r_ in summary["subgoals"]:
            if r_.get("tag") in span:
                r_["frames"] = span[r_["tag"]]
        (out / "outcomes.json").write_text(json.dumps(outcomes, indent=1, default=float))
        phases, Ls = _two_hand_phase_order(phases, Ls)
        jn = np.array(E.joint_names(model))
        # the equality state per frame travels with the trajectory (a cut or a thread that ran out is run-time state)
        eqr = _eq_record(segs, model) or {}
        if eqr:
            EQ = eqr["eq_active"]; prev = np.vstack([eqr["eq_active_start"][None], EQ[:-1]])
            cut_eqs = {str(j.get("eq")) for r in rules if r.get("kind") == "cut" for j in r.get("joins", [])}
            latch_eqs = {str(r.get("eq")) for r in rules if r.get("kind") == "latch"}
            offs = [{"equality": str(eqr["eq_names"][e]), "frame": int(t), "subgoal": int(Ss[t])}
                    for t, e in zip(*np.nonzero((prev == 1) & (EQ == 0)))]
            summary["cuts"] = [o for o in offs if o["equality"] in cut_eqs]
            summary["equalities_released"] = [o for o in offs if o["equality"] not in cut_eqs | latch_eqs]
            if latch_eqs:
                # a latch's state per frame is the recorded eq_active: engaged where it switches on, pulled out where it
                # switches off (a latch carried across subgoals stays on, so a segment boundary is no event)
                ons = [{"equality": str(eqr["eq_names"][e]), "event": "engage", "frame": int(t), "subgoal": int(Ss[t])}
                       for t, e in zip(*np.nonzero((prev == 0) & (EQ == 1)))]
                summary["latches"] = sorted([o for o in ons if o["equality"] in latch_eqs]
                                            + [dict(o, event="release") for o in offs if o["equality"] in latch_eqs], key=lambda o: o["frame"])
        E.save_episode(out / "trajectory_proposal.npz", Q, Ls, fps, subgoal=np.array(Ss), joint_names=jn, reference=R, artifact_role="proposal_diagnostics", **eqr)
        # phases by label, in order, with frame ranges
        ranges, last = {}, None
        for t, (l, s) in enumerate(zip(Ls, Ss)):
            key = (l, s)
            if key != last:
                ranges.setdefault(key, []).append([t, t]); last = key
            else:
                ranges[key][-1][1] = t
        ph_out = []
        seen = set()
        for p in phases:
            key = (p["label"], p["subgoal"])
            if key + (p.get("hand"),) in seen:      # a two-hand phase is listed once per hand
                continue
            seen.add(key + (p.get("hand"),))
            ph_out.append({"label": p["label"], "hand": p.get("hand"), "objects": p.get("objects", []), "anchor": p.get("anchor"), "subgoal": p["subgoal"], "frames": ranges.get(key, [])})
        (out / "phases.json").write_text(json.dumps(ph_out, indent=1))
        (out / "state.json").write_text(json.dumps(state, indent=1, default=float))
        summary.update({"n_frames": int(len(Q)), "duration_s": round(len(Q) / fps, 2)})
        summary["files"].update({"trajectory_proposal": str(out / "trajectory_proposal.npz"), "phases": str(out / "phases.json"), "state": str(out / "state.json")})
        summary["robot_schema"] = RR.SCHEMA
        summary["files"]["scene_proposal"] = str(out / "scene_proposal.xml")
        # A failed or partial execution only has diagnostics. Final trajectory
        # creation is conditional, and the independent release gate still has
        # to accept it before the harness can deliver/render it.
        if summary["ok"] and not dry_run and len(segs) == len(subgoals):
            arrays = _join_runtime_segments(out, segs, Ls, Ss, fps, runtime)
            np.savez_compressed(out / "trajectory.npz", **arrays)
            summary["files"]["trajectory"] = str(out / "trajectory.npz")
            summary["files"]["runtime_scene"] = str(out / "scene_physics.xml")
            summary["scene_sha256"] = runtime.scene_sha256
            summary["trajectory_sha256"] = hashlib.sha256((out / "trajectory.npz").read_bytes()).hexdigest()
    summary["files"]["execution_report"] = str(out / "execution_report.json")
    (out / "execution_report.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary
