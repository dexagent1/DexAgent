"""Automatic grasp search: one call screens reach and drives the real stress tests over many grasp variants in parallel."""
from __future__ import annotations

TOOL = {
    "name": "optimize_grasp_pose",
    "title": "Optimize grasp pose",
    "type": "skill",
    "category": "Grasp & Poses",
    "summary": "Cross-entropy search over a grasp's pose parameters (wrist roll about the approach, tilts about the pinch line and "
               "across it, offsets along the approach and the pads' line, the opening, the press depth), starting from "
               "human seeds, given poses or a grasp family's own candidates (rigid wraps and pinches, bar/cable-link pinches, "
               "plate-edge pinches, flat pinches), with a static and arm-reach screen first and the REAL grasp stress tests "
               "run in parallel processes; returns the best passing pose, or the failure classes of the best tries.",
    "description": (
        "Replaces the hand search of one candidate per call. The agent names the hand scene (the object, the part and the "
        "hand), the grasp family and the stress tests the subgoal needs; the skill does the rest. Starts: `starts` (a pose, a "
        "list, or a json file of poses: human_grasp_seeds' seeds, an earlier result, a deliberately rough guess) and, unless "
        "family is 'seeds', the family's generator run in the worker pool: 'rigid' = generate_rigid_grasps (pocket wraps and "
        "thumb pinches), 'bar' = handle_pinch_pose at several points along the bar and tilts about it (handles, rods, a cable "
        "or rope link: the grasp body's own capsule/cylinder geom, never the chain hanging from it), 'plate' = "
        "plate_edge_pinch_pose on each in-plane edge of the part's thinnest box geom, 'flat' = generate_flat_pinches (parts "
        "lying flat 1-3 cm tall); 'auto' picks by the grasp body as a whole (thinner than 24 mm across and twice as long, "
        "with a capsule/cylinder of at most 12 mm radius -> bar; a slab at most 12 mm thick with sides three times that -> plate; "
        "else rigid). Each candidate is a start "
        "perturbed by nine parameters in a frame at the grip pads (a = approach, p = pinch line, q = a x p): roll about a, "
        "tilt about p (keeps the pinch contact), tilt about q, offsets along a, p and q, the way in tipped toward above or "
        "the side (approach_up), the approach opening (open_frac, replacing the start's own open shape) and press_depth_rad. A perturbed hand re-closes its grip fingers from the "
        "posed shape until the pads touch the grasped body (hand.cinch_fingers, table and bystanders avoided). Generated "
        "starts cover distinct grasp families within the budget; after the first batch, isolated squeeze/opening/approach "
        "probes preserve seated contact geometry before the broader coupled search. Screens, cheap "
        "first: the closed hand overlaps the scene by at most max_overlap_m (grip pads excused against the part), the thumb "
        "and another grip pad within 2 mm of it (prehensile grips), the open hand 3 and 6 cm back along the approach clear "
        "of everything (the executor reaches in from there), outside keep_clear; then, with bench_scene, the station's arm "
        "(armhand.screen_poses: exact flange IK at the grasp, the pre-grasp and every arm_waypoints offset, arm links "
        "against the table and objects). Survivors run the stress tests by calling the library verifiers themselves "
        "(default grasp_lift_test and grasp_hold_test; any Grasp Stress Test by name with test_args) in `workers` "
        "worker processes of its own, or through `launcher` (a command prefix such as a cluster wrapper) for more CPUs. The score rewards "
        "passed tests and partial progress (the share of the commanded lift achieved, low slip and drift) and penalises "
        "screen failures by how far off they are; a Gaussian per start is refit to the elite candidates each iteration "
        "and the search stops at the first iteration with a passing pose (or at max_evals). Nothing is loosened: a pose "
        "passes only when every requested test passes unchanged. The result is a pose json the executor and stress tests "
        "take (with stress_tests to copy into the manifest) and the ranked passing poses as pose_candidates; when nothing "
        "passes, the failure classes of the best tries (screen reasons, arm reasons, each test's first failing phase) and "
        "a hint say what to change: another family, a non-prehensile motion for a part too thin for the pads, the "
        "stations, or the scene."
    ),
    "inputs": [
        {"name": "hand_scene", "type": "dict | str", "description": "Hand scene (hand_scene_from_recon's json or dict): the object, its grasp_body and the hand side."},
        {"name": "family", "type": "str", "description": "'auto' (default), 'rigid', 'bar', 'plate', 'flat', or 'seeds' (only the given starts)."},
        {"name": "starts", "type": "dict | list | str | None", "description": "Start poses: a pose, a list, or a json file holding a pose or a list (human_grasp_seeds' seeds_sgNN_<side>.json, generate_rigid_grasps' out_json)."},
        {"name": "tests", "type": "list[str]", "description": "Grasp Stress Tests every pose must pass. Default ['grasp_lift_test', 'grasp_hold_test']; add 'grasp_shake_test' for a carry, 'grasp_turn_test' (with test_args joint) for a turn."},
        {"name": "test_args", "type": "dict | None", "description": "{test name: keyword arguments} for the tests (e.g. {'grasp_turn_test': {'joint': 'cap_hinge'}})."},
        {"name": "lift_heights_m", "type": "list", "description": "Heights for the lift tests. Default [0.06, 0.10, 0.15]."},
        {"name": "bench_scene", "type": "str | None", "description": "Stage 2 bench scene.xml (or the station-moved scene): screen every candidate for the station's arm before its stress tests."},
        {"name": "arm_waypoints", "type": "list | None", "description": "Reach screening along the intended manipulation: legacy world wrist offsets [dx,dy,dz], or {object: free_root_body_name, target_pos: [x,y,z], target_euler_deg: [rx,ry,rz], steps: 8}. Object targets use intrinsic XYZ degrees and absolute WORLD body-frame poses (not object-bottom coordinates); omitted rotation preserves the previous orientation. The hand follows its grasp transform while the object moves, including intermediate rotations. These hypothetical IK states are never executed object motion; the full physical executor remains required."},
        {"name": "state", "type": "str | list | None", "description": "Bench state for the arm screen: trajectory.npz (last frame) or qpos. Defaults to the hand scene's recorded bench_state when available, otherwise the initial keyframe. Explicit object state must agree with the hand scene; rebuild hand_scene_from_recon after physical motion."},
        {"name": "bar", "type": "str | dict | None", "description": "Family 'bar': the capsule/cylinder geom name or {p0, p1, radius}; default the grasp body's own thinnest capsule/cylinder."},
        {"name": "plate_geom", "type": "str | None", "description": "Family 'plate': the box geom; default the grasp body's thinnest box."},
        {"name": "keep_clear", "type": "list | dict | None", "description": "Zones ({center, radius} or {min, max}, world m) no part of the hand may enter (room for the other hand)."},
        {"name": "bounds", "type": "dict | None", "description": "Override parameter ranges: roll_deg, tilt_pinch_deg, tilt_cross_deg, off_approach_m, off_pinch_m, off_cross_m, approach_up, open_frac, press_depth_rad (each [lo, hi])."},
        {"name": "iterations / population / elite_frac / max_evals / max_starts", "type": "numbers", "description": "Search budget: 8 iterations of 16 candidates, the best quarter refit, at most 128 candidate evaluations, 6 starts."},
        {"name": "seed_time_budget_s", "type": "float | None", "description": "Per bar-family IK solve budget (default 30 seconds; None for exhaustive seeds). A running seed may finish after the budget. Reports searched seed scope; every accepted pose still passes all requested physical tests. Supplied starts are verified before fallback family generation."},
        {"name": "max_overlap_m", "type": "float", "description": "Closed-hand overlap allowed by the screen (pads excused against the part; the executor fails a hand 8 mm in). Default 0.008."},
        {"name": "prehensile", "type": "bool", "description": "Require the thumb and another pad on the part (default true; false for a one-sided push or press pose)."},
        {"name": "workers", "type": "int | None", "description": "Parallel processes (default: CPUs available - 1, at most 8)."},
        {"name": "launcher", "type": "list[str] | None", "description": "Command prefix that runs a worker batch elsewhere (a cluster wrapper); the batch runs `python -m tool_library.skills.optimize_grasp_pose --worker in.json out.json` there with its own pool. Needs a shared filesystem."},
        {"name": "launcher_batches", "type": "int", "description": "Batches sent through the launcher at once. Default 2."},
        {"name": "seed", "type": "int", "description": "Random seed. Default 0."},
        {"name": "out_json", "type": "str | None", "description": "Write the best passing pose here and the ranked passing poses to <stem>_passing.json; when nothing passes, the best try goes to <stem>_best_try.json (a start for the next attempt, not a pose to execute)."},
        {"name": "checkpoint_json", "type": "str | None", "description": "Optional atomic search journal. Saves completed evaluations, generated starts, pending jobs and CEM/RNG state after bounded batches. This agent-writable file is not verification evidence; resumed passing candidates are physically rechecked."},
        {"name": "resume", "type": "bool", "description": "Resume checkpoint_json with identical inputs (default false). Changed parameters, input files, compiled hand/bench physics, tool-library code or runtime reject the checkpoint. Budgets are total across resumptions, not added per call."},
        {"name": "stop_file", "type": "str | None", "description": "With checkpoint_json, an existing marker requests a cooperative stop. The current batch finishes and workers close normally. Returns ok=false, status=incomplete, pose=null, passing=[]; no passing pose file is written. Remove the marker before resuming. An individual verifier or seed solve is not interrupted, so this is not a wall-clock deadline."},
        {"name": "checkpoint_batch_size", "type": "int | None", "description": "Maximum jobs per checkpoint batch (default workers). Requires checkpoint_json. Changes execution grouping only; the population, budgets and every requested check remain unchanged."},
    ],
    "outputs": [
        {"name": "ok", "type": "bool", "description": "A pose passed every requested test (and the arm screen when bench_scene was given)."},
        {"name": "status / complete", "type": "str / bool", "description": "passed or budget_exhausted with complete=true; cooperative stop returns incomplete with complete=false. Budget exhaustion is not proof that the task is impossible."},
        {"name": "pose", "type": "dict", "description": "The best passing pose (else the best try): pos, euler, finger_targets, finger_targets_squeezed, grip_fingers, approach, open_frac?, press_depth_rad, source, family, stress_tests {test: {ok, first_fail, ...}}, optimize {theta, start, score, arm}."},
        {"name": "passing", "type": "list", "description": "Ranked passing poses (give them as pose_candidates)."},
        {"name": "failure_classes", "type": "dict", "description": "{class: count} over every evaluated candidate: screen:<reason>, arm:<reason>, <test>:<first failing phase>."},
        {"name": "best_tries", "type": "list", "description": "The best candidates' scores, parameters, start and per-test outcomes."},
        {"name": "hint", "type": "str | None", "description": "When nothing passed: what the failure classes point to."},
        {"name": "n_evals / n_screened_out / n_arm_rejected / n_tested / family / starts / seconds", "type": "numbers", "description": "Search record."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["grasp", "optimization", "cem", "stress test", "parallel", "stage 3"],
    "used_in": [],
    "status": "tested",
    "entrypoint": "optimize_grasp_pose",
}

import json
import hashlib
import importlib.metadata
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

PARAMS = ("roll_deg", "tilt_pinch_deg", "tilt_cross_deg", "off_approach_m", "off_pinch_m", "off_cross_m", "approach_up", "open_frac",
          "press_depth_rad")
DEFAULT_BOUNDS = {"roll_deg": (-45.0, 45.0), "tilt_pinch_deg": (-35.0, 35.0), "tilt_cross_deg": (-20.0, 20.0),
                  "off_approach_m": (-0.012, 0.02), "off_pinch_m": (-0.02, 0.02), "off_cross_m": (-0.02, 0.02),
                  "approach_up": (-0.6, 1.2), "open_frac": (0.15, 0.75), "press_depth_rad": (0.08, 0.45)}
DEFAULT_TESTS = ("grasp_lift_test", "grasp_hold_test")
BAR_MAX_RADIUS_M = 0.012       # a capsule/cylinder this thin is pinched across, not wrapped (a handle, a rod, a cable link)
PLATE_MAX_HALF_M = 0.006       # a box this thin (12 mm) with in-plane sides three times its thickness is a plate
_CACHE = {}                    # per worker process: opened hand scenes and bench probes, by their description


# ------------------------------------------------------------------ helpers --
def _load_poses(x):
    if x is None:
        return []
    if isinstance(x, (str, Path)):
        x = json.loads(Path(x).read_text())
    if isinstance(x, dict) and "grasps" in x:
        x = x["grasps"]
    if isinstance(x, dict) and "pose" in x and "pos" not in x:
        x = x["pose"]
    if isinstance(x, dict):
        return [x]
    return [p for p in x if isinstance(p, dict) and "pos" in p and "euler" in p]


def _jsonable(v):
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items() if k != "data"}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


def _scene_key(scene):
    return json.dumps(scene, sort_keys=True, default=str)


def _opened(scene):
    from tool_library.common import hand as H
    k = ("scene", _scene_key(scene))
    if k not in _CACHE:
        _CACHE[k] = H.open_scene(scene)
    return _CACHE[k]


def _bench(bench_scene, state):
    import mujoco
    from tool_library.common import armhand as AH
    from tool_library.common import episode as E
    k = ("bench", str(bench_scene), json.dumps(state, default=str))
    if k not in _CACHE:
        bm, _ = E.build_execution_model(str(bench_scene))
        bd = mujoco.MjData(bm)
        if bm.nkey:
            mujoco.mj_resetDataKeyframe(bm, bd, 0)
        q = bd.qpos.copy()
        if state is not None:
            q = E.state_qpos(bm, state)
        _CACHE[k] = (bm, q, AH.ArmProbe(str(bench_scene)))
    return _CACHE[k]


def _rot_about(axis, deg):
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_rotvec(math.radians(float(deg)) * np.asarray(axis, float)).as_matrix()


def _theta(u, start, bounds):
    """Normalized u in [-1, 1]^9 -> parameters; u = 0 is the start as it is (no turn, no offset, its own opening and
    press depth)."""
    th = {}
    for i, k in enumerate(PARAMS):
        lo, hi = bounds[k]
        if k == "open_frac":
            ref = float(start.get("open_frac", 0.55))
        elif k == "press_depth_rad":
            ref = float(start.get("press_depth_rad", 0.3))
        else:
            ref = 0.0
        ref = float(np.clip(ref, lo, hi))
        x = float(u[i])
        th[k] = ref + x * (hi - ref) if x >= 0 else ref + x * (ref - lo)
    return th


def _pads_frame(model, data0, rig, pose):
    """Centre of the grip pads, the approach a, the pinch line p (thumb to the other pads, made normal to a) and q."""
    import mujoco
    from tool_library.common import hand as H
    d = mujoco.MjData(model)
    d.qpos[:] = data0.qpos
    H.apply_pose(model, d, rig, pose)
    grip = [f for f in (pose.get("grip_fingers") or ["thumb", "index"]) if f in rig.fingertip_sites]
    tips = {f: np.asarray(d.site_xpos[rig.fingertip_sites[f]], float).copy() for f in grip}
    c = np.mean(list(tips.values()), axis=0) if tips else np.asarray(pose["pos"], float)
    a = H.approach_axis(model, d, rig, pose)
    R = H.euler_to_mat(pose["euler"])
    if "thumb" in tips and len(tips) >= 2:
        p = tips["thumb"] - np.mean([v for f, v in tips.items() if f != "thumb"], axis=0)
    else:
        p = R[:, 1].copy()
    p = p - a * float(a @ p)
    if np.linalg.norm(p) < 1e-6:
        p = np.cross(a, [1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.cross(a, [0.0, 1.0, 0.0])
    p /= np.linalg.norm(p)
    return c, a, p, np.cross(a, p)


def perturb(scene, start, theta):
    """The start moved by theta in the pads' frame; a moved hand re-closes its grip fingers onto the part."""
    import mujoco
    from tool_library.common import hand as H
    model, data0, rig, ids = _opened(scene)
    pose = json.loads(json.dumps(_jsonable(start)))
    moved = any(abs(theta[k]) > 1e-9 for k in PARAMS[:6])
    if moved:
        c, a, p, q = _pads_frame(model, data0, rig, start)
        Rm = _rot_about(a, theta["roll_deg"]) @ _rot_about(p, theta["tilt_pinch_deg"]) @ _rot_about(q, theta["tilt_cross_deg"])
        pos = np.asarray(start["pos"], float)
        new_pos = c + Rm @ (pos - c) + a * theta["off_approach_m"] + p * theta["off_pinch_m"] + q * theta["off_cross_m"]
        pose["pos"] = [float(v) for v in new_pos]
        pose["euler"] = [float(v) for v in H.mat_to_euler(Rm @ H.euler_to_mat(start["euler"]))]
        pose["approach"] = [float(v) for v in Rm @ a]
    if abs(theta["approach_up"]) > 1e-9:
        # the way in, apart from the hand's orientation: tipped toward coming down from above (or in from the side)
        model, data0, rig, ids = _opened(scene)
        import mujoco as _mj
        d_ = _mj.MjData(model)
        d_.qpos[:] = data0.qpos
        a0 = H.approach_axis(model, d_, rig, pose)
        a1 = a0 + float(theta["approach_up"]) * np.array([0.0, 0.0, 1.0])
        if np.linalg.norm(a1) > 1e-6 and (a1 / np.linalg.norm(a1))[2] > -0.2:
            pose["approach"] = [float(v) for v in a1 / np.linalg.norm(a1)]
    if moved:
        # re-close: from the start's closed shape (opening the fingers first and curling them back loses a thin part: the
        # tip's arc leaves it before it arrives) each grip finger backs off where it now sinks into the part and curls on
        # where it now floats, until its pad just touches
        closed = {k: float(v) for k, v in (start.get("finger_targets") or {}).items() if k in rig.finger_range}
        own_open = {k: float(v) for k, v in (start.get("finger_targets_open") or {}).items() if k in rig.finger_range}
        grip = [f for f in (start.get("grip_fingers") or ["thumb", "index"]) if f in rig.fingertip_sites]
        d = mujoco.MjData(model)
        d.qpos[:] = data0.qpos
        H.apply_pose(model, d, rig, pose, closed)
        obj = sorted(ids["grasp_gids"])
        avoid = [g for g in ids["obj_gids"] if g not in ids["grasp_gids"]] + ([ids["floor_gid"]] if ids["floor_gid"] >= 0 else [])
        H.cinch_fingers(model, d, rig, obj, fingers=grip, avoid_gids=avoid)
        new_closed = {k: float(d.qpos[a_]) for k, a_ in rig.finger_qpos.items()}
        d2 = mujoco.MjData(model)
        d2.qpos[:] = data0.qpos
        H.apply_pose(model, d2, rig, pose, new_closed)
        H.cinch_fingers(model, d2, rig, obj, fingers=grip, touch_m=-0.0025, max_pen_m=0.0035, avoid_gids=avoid)
        pose["finger_targets"] = new_closed
        pose["finger_targets_squeezed"] = {k: float(d2.qpos[a_]) for k, a_ in rig.finger_qpos.items()}
        if own_open:
            pose["finger_targets_open"] = {k: (own_open.get(k, v) if H.finger_of_joint(k) in grip else v) for k, v in new_closed.items()}
    if abs(theta["open_frac"] - float(start.get("open_frac", 0.55))) > 1e-6:
        pose.pop("finger_targets_open", None)            # the executor's opening by fraction, idle fingers curled
        pose["open_frac"] = float(theta["open_frac"])
    pose["press_depth_rad"] = float(theta["press_depth_rad"])
    return pose


def static_screen(scene, pose, keep_clear=None, max_overlap_m=0.008, prehensile=True):
    """Cheap geometric checks before any physics: (ok, reason, numbers)."""
    import mujoco
    from tool_library.common import hand as H
    model, data0, rig, ids = _opened(scene)
    d = mujoco.MjData(model)
    d.qpos[:] = data0.qpos
    H.apply_pose(model, d, rig, pose)
    grip = [f for f in (pose.get("grip_fingers") or ["thumb", "index"]) if f in rig.fingertip_sites]
    ov = H.deepest_overlap(model, d, rig, ids, keep_tips=tuple(grip))
    gaps = {}
    for f in grip:
        gaps[f] = min((H.geom_distance(model, d, g, og, 0.05) for g in rig.tip_geoms(f) for og in ids["grasp_gids"]), default=1.0)
    touching = [f for f, g in gaps.items() if g <= 0.002]
    num = {"overlap_mm": round(1000 * ov, 2), "pad_gaps_mm": {f: round(1000 * g, 1) for f, g in gaps.items()}, "touching": touching}
    if ov > max_overlap_m:
        return False, "overlap", num
    if prehensile and ("thumb" not in touching or len(touching) < 2):
        return False, "pads_off", num
    if keep_clear:
        from tool_library.skills.generate_rigid_grasps import _in_zone
        if _in_zone(model, d, rig, keep_clear):
            return False, "keep_clear", num
    a = H.approach_axis(model, d, rig, pose)
    closed = {k: float(v) for k, v in (pose.get("finger_targets") or {}).items() if k in rig.finger_range}
    own = {k: float(v) for k, v in (pose.get("finger_targets_open") or {}).items() if k in rig.finger_range}
    clamp = H.executor_clamp(rig, pose)
    fraction = float(pose.get("open_frac", 0.55))
    opened = own or H.approach_open_targets(rig, closed, fraction, grip, clamp["grip_posed"])
    floor = int(ids.get("floor_gid", -1))
    if not own and floor >= 0:
        for f in [fraction] + [v for v in (.45, .35, .25, .15) if v < fraction - 1e-9]:
            opened = H.approach_open_targets(rig, closed, f, grip, clamp["grip_posed"])
            H.apply_pose(model, d, rig, pose, opened)
            if min(H.geom_distance(model, d, g, floor, .05) for g in rig.collision_gids) > -.002:
                break
    pos = np.asarray(pose["pos"], float)
    for back in (0.03, 0.06):
        H.apply_pose(model, d, rig, {"pos": pos + a * back, "euler": pose["euler"]}, opened)
        o = H.deepest_overlap(model, d, rig, ids, keep_tips=())
        if o > 0.003:
            num["approach_overlap_mm"] = round(1000 * o, 1)
            return False, "approach_blocked", num
    return True, None, num


def _partial(r):
    """Progress of a failed test toward passing, 0..1, from the numbers it reports."""
    terms = []
    lifts = r.get("lifts") or []
    if lifts:
        terms.append(float(np.mean([np.clip(float(x["achieved_mm"]) / max(float(x["commanded_mm"]), 1e-6), 0.0, 1.0) for x in lifts])))
    if r.get("slip_mm_s") is not None:
        terms.append(1.0 - float(np.clip(float(r["slip_mm_s"]) / 20.0, 0.0, 1.0)))
    if r.get("max_drift_mm") is not None:
        terms.append(1.0 - float(np.clip(float(r["max_drift_mm"]) / 40.0, 0.0, 1.0)))
    if r.get("first_fail") == "seated":
        terms.append(0.0)
    return float(np.mean(terms)) if terms else 0.0


def _summary(r):
    keep = ("ok", "first_fail", "lifts", "slip_mm_s", "max_drift_mm", "object_nudged_mm", "grip", "penetration")
    result = {k: _jsonable(r[k]) for k in keep if k in r}
    if "penetration" not in result:
        report = next((p for p in r.get("phases", []) if p.get("name") == "penetration"), None)
        if report is not None:
            result["penetration"] = _jsonable(report)
    return result


def run_tests(scene, pose, tests, test_args, lift_heights_m):
    import importlib
    import inspect
    out = {}
    for name in tests:
        mod = importlib.import_module(f"tool_library.verifiers.{name}")
        fn = getattr(mod, name)
        kw = dict((test_args or {}).get(name) or {})
        if "lift_heights_m" in inspect.signature(fn).parameters and "lift_heights_m" not in kw:
            kw["lift_heights_m"] = tuple(lift_heights_m)
        try:
            r = fn(scene, pose, **kw)
        except Exception as e:
            r = {"ok": False, "first_fail": f"error: {type(e).__name__}: {e}"}
        if not r.get("ok") and not r.get("first_fail"):
            # tests that report phases without naming the first failure (grasp_hold_test): the first phase that failed
            r["first_fail"] = next((ph.get("name") for ph in (r.get("phases") or []) if not ph.get("ok", True)), "failed")
        out[name] = r
        if not r.get("ok"):
            break                                          # the tests are a ladder: the first failure decides
    return out


def evaluate(job):
    """One candidate: perturb, screen, arm, stress tests. Also runs a family's start generator (job kind 'start')."""
    if job.get("kind") == "start":
        return _start_job(job)
    t0 = time.time()
    scene, start, theta = job["scene"], job["start"], job["theta"]
    res = {"index": job["index"], "start": job["start_index"], "theta": theta, "stage": "static"}
    try:
        pose = perturb(scene, start, theta)
    except Exception as e:
        res.update(score=-300.0, reason=f"perturb: {type(e).__name__}: {e}", seconds=round(time.time() - t0, 2))
        return res
    res["pose"] = pose
    ok, why, num = static_screen(scene, pose, job.get("keep_clear"), float(job.get("max_overlap_m", 0.008)), bool(job.get("prehensile", True)))
    res["static"] = num
    if not ok:
        missing = 0 if why != "pads_off" else (2 - len([f for f in num["touching"]]))
        res.update(score=-100.0 - min(50.0, num["overlap_mm"]) - 10.0 * max(missing, 0), reason=f"screen:{why}", seconds=round(time.time() - t0, 2))
        return res
    if job.get("bench_scene"):
        from tool_library.common import armhand as AH
        from tool_library.common import hand as H
        bm, q, probe = _bench(job["bench_scene"], job.get("state"))
        side = H.load_scene_spec(scene).get("side", "right")
        r = AH.screen_poses(bm, side, [pose], q, probe=probe, waypoints=job.get("arm_waypoints"))[0]
        res["arm"] = {k: r.get(k) for k in ("ok", "residual_mm", "residual_deg", "reason")}
        if not r.get("ok"):
            kind = "arm_collision" if "runs into" in str(r.get("reason")) else "arm_unreachable"
            res.update(stage="arm", score=-60.0 - min(30.0, float(r.get("residual_mm") or 0.0) / 5.0), reason=f"arm:{kind}",
                       seconds=round(time.time() - t0, 2))
            return res
    res["stage"] = "tests"
    tr = run_tests(scene, pose, job["tests"], job.get("test_args"), job.get("lift_heights_m", (0.06, 0.10, 0.15)))
    res["tests"] = {k: _summary(v) for k, v in tr.items()}
    all_ok = len(tr) == len(job["tests"]) and all(v.get("ok") for v in tr.values())
    score = sum(10.0 if v.get("ok") else 6.0 * _partial(v) for v in tr.values())
    if all_ok:
        lifts = [x for v in tr.values() for x in (v.get("lifts") or [])]
        margin = np.mean([np.clip(float(x["achieved_mm"]) / max(float(x["commanded_mm"]), 1e-6), 0.0, 1.0) for x in lifts]) if lifts else 1.0
        slip = max([float(v.get("slip_mm_s") or 0.0) for v in tr.values()] or [0.0])
        score += 5.0 + 2.0 * float(margin) - 0.05 * slip
    first = next((f"{k}:{v.get('first_fail')}" for k, v in tr.items() if not v.get("ok")), None)
    res.update(ok=bool(all_ok), score=round(float(score), 3), reason=first, seconds=round(time.time() - t0, 2))
    return res


def _start_job(job):
    """A family's start generator, run in a worker: returns {'starts': [poses], 'error'?}."""
    fam, scene, a = job["family"], job["scene"], dict(job.get("args") or {})
    try:
        if fam == "rigid":
            from tool_library.skills.generate_rigid_grasps import generate_rigid_grasps
            r = generate_rigid_grasps(scene, **a)
            return {"starts": [dict(g, source=g.get("source", "generate_rigid_grasps")) for g in r["grasps"]]}
        if fam == "bar":
            from tool_library.skills.handle_pinch_pose import handle_pinch_pose
            p = handle_pinch_pose(scene, **a)
            return {"starts": [dict(p, source="handle_pinch_pose", family="bar")]}
        if fam == "plate":
            from tool_library.skills.plate_edge_pinch_pose import plate_edge_pinch_pose
            p = plate_edge_pinch_pose(scene, **a)
            return {"starts": [dict(p, source="plate_edge_pinch_pose", family="plate")]}
        if fam == "flat":
            from tool_library.skills.generate_flat_pinches import generate_flat_pinches
            r = generate_flat_pinches(scene, **a)
            gs = r.get("grasps") if isinstance(r, dict) else r
            return {"starts": [dict(g, source="generate_flat_pinches", family="flat") for g in (gs or [])]}
    except Exception as e:
        return {"starts": [], "error": f"{fam}: {type(e).__name__}: {str(e)[:300]}"}
    return {"starts": [], "error": f"unknown family {fam}"}


# ------------------------------------------------------------ the families --
def _body_geoms(model, bid):
    return [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == int(bid) and int(model.geom_contype[g] | model.geom_conaffinity[g]) != 0] or \
           [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == int(bid)]


def auto_family(scene):
    """'bar' for a grasp body that is a thin capsule/cylinder as a whole, 'plate' for one that is a thin slab as a whole,
    else 'rigid', with the geom it found. The whole body decides, not one of its geoms: a stud brick's thin top plate does
    not make the brick a plate."""
    import mujoco
    from tool_library.common import hand as H
    model, data0, rig, ids = _opened(scene)
    bid = H.grasp_body(model, scene, ids)
    gs = _body_geoms(model, bid)
    ext = H.object_extent(model, data0, gs)
    size = np.sort(np.asarray(ext["size"], float))
    bars = [g for g in gs if int(model.geom_type[g]) in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER))
            and float(model.geom_size[g][0]) <= BAR_MAX_RADIUS_M and float(model.geom_size[g][1]) >= float(model.geom_size[g][0])]
    if bars and size[1] <= 2.0 * BAR_MAX_RADIUS_M + 1e-6 and size[2] >= 2.0 * size[1]:
        return "bar", model.geom(max(bars, key=lambda g: float(model.geom_size[g][1]))).name
    boxes = [g for g in gs if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_BOX)]
    if boxes and size[0] <= 2.0 * PLATE_MAX_HALF_M + 1e-6 and size[1] >= 3.0 * size[0]:
        return "plate", model.geom(max(boxes, key=lambda g: float(np.prod(model.geom_size[g])))).name
    return "rigid", None


def family_jobs(scene, family, bar=None, plate_geom=None, keep_clear=None, bench_scene=None, arm_waypoints=None, state=None,
                seed_time_budget_s=30.0):
    """Start-generator jobs of a family (each run in a worker)."""
    import mujoco
    from tool_library.common import hand as H
    jobs = []
    if family == "rigid":
        a = {"yaw_step_deg": 45.0, "keep_clear": keep_clear}
        if bench_scene:
            a.update(bench_scene=bench_scene, arm_waypoints=arm_waypoints, state=state)
        jobs.append({"kind": "start", "family": "rigid", "scene": scene, "args": a})
    elif family == "bar":
        if bar is None:
            fam, bar = auto_family(scene)
            if fam != "bar":
                model, data0, rig, ids = _opened(scene)
                bid = H.grasp_body(model, scene, ids)
                cyl = [g for g in _body_geoms(model, bid) if int(model.geom_type[g]) in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER))]
                if not cyl:
                    raise ValueError("family 'bar' needs a capsule/cylinder geom on the grasp body, or `bar` = {p0, p1, radius}")
                bar = model.geom(cyl[0]).name
        for s in (0.5, 0.3, 0.7):
            for tilt in (0.0, 30.0, -30.0):
                jobs.append({"kind": "start", "family": "bar", "scene": scene,
                             "args": {"bar": bar, "s": s, "tilt_deg": tilt, "fingers": ["thumb", "index"], "yaw_step_deg": 45,
                                      "pitches_deg": [-45, 0, 45], "iters": 200, "time_budget_s": seed_time_budget_s}})
    elif family == "plate":
        model, data0, rig, ids = _opened(scene)
        if plate_geom is None:
            fam, plate_geom = auto_family(scene)
            if fam != "plate":
                raise ValueError("family 'plate' needs a thin box geom on the grasp body (or plate_geom)")
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, plate_geom)
        thin = int(np.argmin(model.geom_size[g])) if g >= 0 else 2
        for ax in [i for i in range(3) if i != thin]:
            for sgn in "+-":
                jobs.append({"kind": "start", "family": "plate", "scene": scene,
                             "args": {"geom": plate_geom, "edge": sgn + "xyz"[ax], "min_out_m": 0.01, "inset_m": 0.018}})
    elif family == "flat":
        a = {"time_budget_s": 600.0}
        if bench_scene:
            a.update(bench_scene=bench_scene, arm_waypoints=arm_waypoints, state=state)
        jobs.append({"kind": "start", "family": "flat", "scene": scene, "args": a})
    elif family != "seeds":
        raise ValueError(f"family must be one of auto, rigid, bar, plate, flat, seeds (not {family!r})")
    return jobs


# --------------------------------------------------------------- the runner --
class _Pool:
    """Worker processes of our own (`python -m tool_library.skills.optimize_grasp_pose --serve`, one JSON job per line in,
    one result per line out), or batches sent through a launcher command prefix. Not multiprocessing: its spawned workers
    re-run the caller's main script, so a call from a script without a main guard (an agent's run_python snippet) broke
    the pool."""

    def __init__(self, workers, launcher=None, launcher_batches=2):
        self.launcher, self.batches = (list(launcher) if launcher else None), max(1, int(launcher_batches))
        self.workers = int(workers)
        self.procs = []
        if not self.launcher and self.workers > 1:
            root = Path(__file__).resolve().parents[2]
            env = dict(os.environ)
            env["PYTHONPATH"] = f"{root}{os.pathsep}" + env.get("PYTHONPATH", "")
            env.setdefault("OMP_NUM_THREADS", "1")
            for _ in range(self.workers):
                log = tempfile.TemporaryFile(mode="w+")
                p = subprocess.Popen([sys.executable, "-m", "tool_library.skills.optimize_grasp_pose", "--serve"], cwd=str(root), env=env,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
                self.procs.append((p, log))

    def _ask(self, k, job):
        p, log = self.procs[k]
        try:
            p.stdin.write(json.dumps(_jsonable(job)) + "\n")
            p.stdin.flush()
            line = p.stdout.readline()
            if line:
                return json.loads(line)
        except (BrokenPipeError, OSError, ValueError):
            pass
        log.seek(0)
        err = log.read()[-400:]
        return ({"starts": [], "error": f"worker failed: {err}"} if job.get("kind") == "start" else
                {"index": job.get("index"), "start": job.get("start_index"), "theta": job.get("theta"), "score": -300.0, "stage": "worker",
                 "reason": f"worker failed: {err[-200:]}"})

    def map(self, jobs, search_state=None):
        if not jobs:
            return []
        if self.launcher:
            return _launch(jobs, self.launcher, self.batches, self.workers)
        if not self.procs:
            return [evaluate(j) for j in jobs]
        import queue
        import threading
        todo, out = queue.Queue(), [None] * len(jobs)
        for i, j in enumerate(jobs):
            todo.put(i)

        def run(k):
            while True:
                try:
                    i = todo.get_nowait()
                except queue.Empty:
                    return
                out[i] = self._ask(k, jobs[i])

        threads = [threading.Thread(target=run, args=(k,), daemon=True) for k in range(len(self.procs))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return out

    def close(self):
        for p, log in self.procs:
            try:
                p.stdin.close()
                p.wait(timeout=10)
            except Exception:
                p.kill()
                p.wait()
            finally:
                for stream in (p.stdin, p.stdout, log):
                    if stream is not None:
                        stream.close()
        self.procs = []


def _serve():
    """A worker process: one JSON job per line on stdin, one JSON result per line on stdout. Anything a tool prints goes to
    stderr, so the result channel stays clean."""
    out = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            r = evaluate(json.loads(line))
        except Exception as e:
            r = {"score": -300.0, "stage": "worker", "reason": f"worker error: {type(e).__name__}: {e}", "starts": [], "error": str(e)}
        out.write(json.dumps(_jsonable(r)) + "\n")
        out.flush()


def _launch(jobs, launcher, batches, workers):
    """Split the jobs into batches, run each through the launcher (`<launcher> python -m ... --worker in out`) at once."""
    root = Path(__file__).resolve().parents[2]
    tmp = Path(tempfile.mkdtemp(prefix="optimize_grasp_", dir=os.environ.get("TMPDIR")))
    chunks = [list(enumerate(jobs))[i::batches] for i in range(batches) if jobs[i::batches]]
    procs = []
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{root}{os.pathsep}" + env.get("PYTHONPATH", "")
    for k, ch in enumerate(chunks):
        fin, fout = tmp / f"in_{k}.json", tmp / f"out_{k}.json"
        fin.write_text(json.dumps({"jobs": _jsonable([j for _, j in ch]), "workers": workers}))
        cmd = list(launcher) + [sys.executable, "-m", "tool_library.skills.optimize_grasp_pose", "--worker", str(fin), str(fout)]
        procs.append((subprocess.Popen(cmd, cwd=str(root), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True), ch, fout))
    out = [None] * len(jobs)
    for p, ch, fout in procs:
        _, err = p.communicate()
        if fout.exists():
            results = json.loads(fout.read_text())
            if len(results) != len(ch):
                err = "launcher returned the wrong number of candidate results"
                results = None
        else:                                             # the batch never ran: its candidates count as errors, not passes
            results = None
        if results is None:
            results = [{"index": j.get("index"), "start": j.get("start_index"), "theta": j.get("theta"), "score": -300.0, "stage": "launcher",
                     "reason": f"launcher: {err[-300:] if err else 'no output'}"} if j.get("kind") != "start" else
                    {"starts": [], "error": f"launcher: {err[-300:] if err else 'no output'}"} for _, j in ch]
        for (index, _), result in zip(ch, results):
            out[index] = result
    return out


def _default_workers():
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 2
    return max(1, min(8, n - 1))


def _checkpoint_binding(inputs, scene, bench_scene, state):
    """Bind a search journal to inputs, compiled physics and the executing code.

    This is an invalidation checksum, not an authority signature: the agent can
    edit the journal. Cached passing candidates must be evaluated again.
    """
    import mujoco
    from tool_library.common import hand as H

    def digest_model(model):
        buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
        mujoco.mj_saveModel(model, buffer=buffer)
        return hashlib.sha256(buffer.tobytes()).hexdigest()

    def dependencies(value):
        if isinstance(value, dict):
            for item in value.values():
                dependencies(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                dependencies(item)
        elif isinstance(value, (str, Path)):
            try:
                p = Path(value)
                if p.is_file():
                    files[str(p.resolve())] = hashlib.sha256(p.read_bytes()).hexdigest()
            except OSError:
                pass  # a string parameter need not be a filesystem path

    files = {}
    dependencies(inputs)
    model, data, _, _ = H.open_scene(scene)
    physics = {"hand_model": digest_model(model), "hand_qpos": data.qpos.tolist(), "hand_qvel": data.qvel.tolist()}
    if bench_scene:
        from tool_library.common import episode as E
        bench_model, _ = E.build_execution_model(bench_scene)
        physics["bench_model"] = digest_model(bench_model)
    root = Path(__file__).resolve().parents[1]
    code = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*.py")) if "tests" not in p.relative_to(root).parts}
    runtime = {name: importlib.metadata.version(name) for name in ("mujoco", "numpy", "scipy")}
    runtime["python"] = sys.version
    value = _jsonable({"inputs": inputs, "files": files, "physics": physics, "code": code, "runtime": runtime})
    return {"sha256": hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest(),
            "details": value}


class _SearchStopped(Exception):
    pass


class _CheckpointPool:
    """Bounded worker batches plus a deterministic, replayable search journal.

    Replay recomputes CEM/RNG bookkeeping from the persisted evaluations. The
    recorded optimizer state is diagnostic; exact job comparison rejects a
    different replay. A journal never certifies a passing evaluation.
    """
    def __init__(self, path, resume, stop_file, batch_size, binding, workers, launcher, launcher_batches):
        self.path = Path(path)
        self.stop_file = Path(stop_file) if stop_file else None
        self.batch_size = batch_size
        self.pool_args = (workers, launcher, launcher_batches)
        self.pool = None
        self.cursor = 0
        self.reused = 0
        self.rechecked = 0
        if resume:
            self.record = json.loads(self.path.read_text())
            if self.record.get("version") != 1 or self.record.get("binding") != binding:
                raise ValueError("search checkpoint rejected: inputs, compiled physics, code or runtime changed")
            if not isinstance(self.record.get("calls"), list):
                raise ValueError("search checkpoint rejected: invalid call journal")
        else:
            if self.path.exists():
                raise ValueError("checkpoint_json already exists; use resume=True or a new path")
            self.record = {"version": 1, "binding": binding, "calls": [], "status": "incomplete",
                           "authority": "untrusted search journal; cached PASS evaluations are rerun"}
        self.save()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(_jsonable(self.record), stream, sort_keys=True, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def check_stop(self):
        if self.stop_file is not None and self.stop_file.exists():
            self.record["status"] = "incomplete"
            self.record["stop_reason"] = "stop_file"
            self.save()
            raise _SearchStopped()

    def _run(self, jobs):
        if self.pool is None:
            self.pool = _Pool(*self.pool_args)
        return self.pool.map(jobs)

    def map(self, jobs, search_state=None):
        self.check_stop()
        jobs = _jsonable(jobs)
        if self.cursor < len(self.record["calls"]):
            call = self.record["calls"][self.cursor]
            if call.get("jobs") != jobs:
                raise ValueError("search checkpoint rejected: replayed candidate jobs differ")
            results = call.get("results")
            if not isinstance(results, list) or len(results) > len(jobs):
                raise ValueError("search checkpoint rejected: invalid completed batch")
            # A forged score can bias proposal efficiency, but a forged PASS
            # cannot become evidence: evaluate the exact requested job afresh.
            for i, result in enumerate(results):
                if result.get("ok") and jobs[i].get("kind") == "eval":
                    self.check_stop()
                    results[i] = self._run([jobs[i]])[0]
                    self.rechecked += 1
                    self.save()
                    self.check_stop()
                else:
                    self.reused += 1
        else:
            call = {"jobs": jobs, "results": [], "optimizer_state": _jsonable(search_state)}
            self.record["calls"].append(call)
            results = call["results"]
            self.save()
        while len(results) < len(jobs):
            self.check_stop()
            batch = jobs[len(results):len(results) + self.batch_size]
            new_results = self._run(batch)
            if len(new_results) != len(batch):
                raise RuntimeError("worker returned the wrong number of candidate results")
            results.extend(new_results)
            self.save()
            self.check_stop()
        self.cursor += 1
        # The optimizer annotates these dicts; keep journal records immutable.
        return json.loads(json.dumps(results))

    def stopped_result(self, seconds):
        completed = [(j, r) for call in self.record["calls"] for j, r in zip(call["jobs"], call["results"])]
        return {"ok": False, "status": "incomplete", "complete": False, "reason": "cooperative_stop",
                "pose": None, "passing": [], "checkpoint_json": str(self.path), "stop_file": str(self.stop_file),
                "n_evals": sum(j.get("kind") == "eval" for j, _ in completed),
                "completed_seed_jobs": sum(j.get("kind") == "start" for j, _ in completed),
                "seconds": round(seconds, 1), "hint": "Remove stop_file and repeat identical inputs with resume=True. "
                "The checkpoint is search progress, not a verified pose or an infeasibility certificate."}

    def close(self):
        if self.pool is not None:
            self.pool.close()


def _hint(classes, family, evals=()):
    if not classes:
        return None
    top = max(classes, key=classes.get)
    gaps = [min(r["static"]["pad_gaps_mm"].values()) for r in evals
            if r.get("reason") == "screen:pads_off" and (r.get("static") or {}).get("pad_gaps_mm")]
    if top == "screen:pads_off" and gaps and float(np.median(gaps)) > 10.0:
        return (f"the starts are far from the part (the nearest grip pad {float(np.median(gaps)):.0f} mm off, median): seeds from a "
                "demonstration that disagrees with the reconstruction, or a rough guess; give a family (family='auto' adds the "
                "family's own candidates), and if the human hand never reaches the part, check Stage 1's window and Stage 2's placement")
    if top.startswith("arm:"):
        return ("sampled candidates fail the station's arm: try optimize_pose_for_reach on the best try, another approach "
                "side, or reachable_contact_search over the full motion. A finite search is inconclusive; preserve the "
                "configured robot stations")
    if top in ("screen:pads_off", "screen:overlap") or top.endswith(":seated"):
        return ("the pads never close on the part from these starts: when the part is thinner than about two pad radii (or lies "
                "flat), use a family for it ('flat', 'plate') or a non-prehensile motion; else start from another family or seeds")
    if top == "screen:approach_blocked" or top.endswith(":approach_open"):
        return "the open hand runs into the scene on its way in: approach from another side, or keep_clear the obstacle"
    if ":lift" in top or ":hold" in top or ":servo_reserve" in top:
        return ("the grip closes but does not hold the weight: a grip with more opposed pads (family 'rigid' pinch3/wraps), or a "
                "steadying hand / support for a heavy or long part")
    return f"most frequent failure: {top}"


def optimize_grasp_pose(hand_scene, family="auto", starts=None, tests=DEFAULT_TESTS, test_args=None, lift_heights_m=(0.06, 0.10, 0.15),
                        bench_scene=None, arm_waypoints=None, state=None, bar=None, plate_geom=None, keep_clear=None, bounds=None,
                        iterations=8, population=16, elite_frac=0.25, max_evals=128, max_starts=6, max_overlap_m=0.008, prehensile=True,
                        workers=None, launcher=None, launcher_batches=2, seed=0, out_json=None, seed_time_budget_s=30.0,
                        checkpoint_json=None, resume=False, stop_file=None, checkpoint_batch_size=None):
    from tool_library.common import hand as H
    # The harness invokes tools repeatedly in one process. An agent may have
    # replaced a state/scene file at the same path since its previous search.
    # Keep caches within a search, never across independent tool invocations.
    _CACHE.clear()
    t0 = time.time()
    source_inputs = {"hand_scene": hand_scene, "starts": starts, "state": state}
    if (resume or stop_file is not None or checkpoint_batch_size is not None) and checkpoint_json is None:
        raise ValueError("resume, stop_file and checkpoint_batch_size require checkpoint_json")
    if checkpoint_json is not None and any(value is not None and Path(value).resolve() == Path(checkpoint_json).resolve()
                                           for value in (stop_file, out_json)):
        raise ValueError("checkpoint_json must differ from stop_file and out_json")
    batch_size = int(checkpoint_batch_size) if checkpoint_batch_size is not None else int(workers or _default_workers())
    if batch_size < 1:
        raise ValueError("checkpoint_batch_size must be positive")
    # the worker processes run from the repository, not from the caller's folder: file arguments travel as absolute paths
    if isinstance(bench_scene, (str, Path)) and bench_scene:
        bench_scene = str(Path(bench_scene).resolve())
    if isinstance(state, (str, Path)) and state and Path(state).exists():
        state = str(Path(state).resolve())
    scene = H.load_scene_spec(hand_scene)
    if not isinstance(scene, dict) or "objects" not in scene:
        raise ValueError("hand_scene must be a hand-scene dict or its json path (hand_scene_from_recon's out_json)")
    bench_state_source = None
    if bench_scene:
        from tool_library.skills.hand_scene_from_recon import resolve_bench_state
        bench_state_source = "explicit state" if state is not None else (scene.get("bench_state") or {}).get("source", "initial keyframe (legacy hand scene)")
        bench_model, _, _ = _bench(bench_scene, state)
        resolved = resolve_bench_state(scene, bench_model, state)
        if resolved is not None:
            state = resolved.tolist()
    tests = [t for t in (tests or DEFAULT_TESTS)]
    for t in tests:
        if not (Path(__file__).resolve().parents[1] / "verifiers" / f"{t}.py").exists():
            raise ValueError(f"no stress test named {t!r} in the library")
    bnd = {k: tuple(float(x) for x in (bounds or {}).get(k, v)) for k, v in DEFAULT_BOUNDS.items()}
    given = _load_poses(starts)
    if seed_time_budget_s is not None and (not np.isfinite(float(seed_time_budget_s)) or float(seed_time_budget_s) <= 0):
        raise ValueError("seed_time_budget_s must be positive or None")
    if family == "seeds" and not given:
        raise ValueError("family 'seeds' needs starts")
    side = scene.get("side", "right")
    fam = family
    note = None
    if family == "auto":
        fam, geom = auto_family(scene)
        note = f"auto: {fam}" + (f" ({geom})" if geom else "")
        if fam == "bar" and bar is None:
            bar = geom
        if fam == "plate" and plate_geom is None:
            plate_geom = geom
    rng = np.random.default_rng(int(seed))
    errors = []
    common = {"scene": scene, "tests": tests, "test_args": test_args, "lift_heights_m": list(lift_heights_m), "keep_clear": keep_clear,
              "max_overlap_m": max_overlap_m, "prehensile": prehensile, "bench_scene": bench_scene, "arm_waypoints": arm_waypoints,
              "state": state}
    binding_inputs = {"source_inputs": source_inputs, "common": common, "family": family, "resolved_family": fam,
                      "given": given, "bar": bar, "plate_geom": plate_geom, "bounds": bnd, "iterations": iterations,
                      "population": population, "elite_frac": elite_frac, "max_evals": max_evals, "max_starts": max_starts,
                      "seed": seed, "seed_time_budget_s": seed_time_budget_s, "workers": workers or _default_workers(),
                      "launcher": launcher, "launcher_batches": launcher_batches}
    binding = _checkpoint_binding(binding_inputs, scene, bench_scene, state) if checkpoint_json is not None else None
    pool = (_CheckpointPool(checkpoint_json, resume, stop_file, batch_size, binding,
                            workers or _default_workers(), launcher, launcher_batches)
            if checkpoint_json is not None else _Pool(workers or _default_workers(), launcher, launcher_batches))
    D = len(PARAMS)
    evals, it, preflight_count, generation = [], 0, 0, []
    try:
        # Existing candidates already carry task-independent geometric work.
        # Prove them before spending minutes generating a fallback family.
        if given and fam != "seeds":
            preflight_count = min(len(given), max(1, int(max_starts) - int(max_starts) // 2), int(max_evals))
            jobs = [dict(common, kind="eval", index=s, start_index=s, start=given[s],
                         theta=_theta(np.zeros(D), given[s], bnd), u=np.zeros(D).tolist())
                    for s in range(preflight_count)]
            evals = pool.map(jobs, search_state={"phase": "supplied_starts", "rng_state": rng.bit_generator.state})
            for j, r in zip(jobs, evals):
                r["u"] = j["u"]
        gen = []
        if fam != "seeds" and len(evals) < int(max_evals) and not any(r.get("ok") for r in evals):
            for r in pool.map(family_jobs(scene, fam, bar=bar, plate_geom=plate_geom, keep_clear=keep_clear, bench_scene=bench_scene,
                                          arm_waypoints=arm_waypoints, state=state, seed_time_budget_s=seed_time_budget_s),
                              search_state={"phase": "seed_generation", "evals": evals, "rng_state": rng.bit_generator.state}):
                gen += r.get("starts") or []
                generation += [p["ik_search"] for p in r.get("starts", []) if p.get("ik_search")]
                if r.get("error"):
                    errors.append(r["error"])
        # the generators' own ranking is kept (generate_rigid_grasps: arm-feasible, opposed, most pads first); bar and plate
        # pinches, one per call, rank by their pad residual and overlap
        if fam in ("bar", "plate"):
            gen.sort(key=lambda g: float(g.get("worst_residual_mm", 0.0) or 0.0) / 2.0 + float(g.get("overlap_mm", 0.0) or 0.0))
        gen.sort(key=lambda g: 0 if (g.get("arm") or {}).get("ok", True) else 1)
        # A finite start budget must cover distinct grasp families. Taking the
        # first six by static rank can spend every start on almost identical
        # wraps, excluding a feasible pinch entirely. Preserve rank within each
        # family and preserve arm-feasible candidates ahead of rejected ones.
        diverse = []
        for feasible in (True, False):
            groups = {}
            for candidate in gen:
                if bool((candidate.get("arm") or {}).get("ok", True)) == feasible:
                    groups.setdefault(candidate.get("family", fam), []).append(candidate)
            while any(groups.values()):
                for group in groups.values():
                    if group:
                        diverse.append(group.pop(0))
        gen = diverse
        # given starts first, but with a family asked for, at least half the starts are the family's own
        n_given = len(given) if not gen else max(1, min(len(given), int(max_starts) - min(len(gen), int(max_starts) // 2)))
        all_starts = [dict(p, side=p.get("side", side)) for p in given[:n_given]] + [dict(p, side=p.get("side", side)) for p in gen]
        all_starts = all_starts[:int(max_starts)]
        if not all_starts:
            raise RuntimeError("no start pose: the family generator found none" + (f" ({errors[0]})" if errors else "") +
                               "; give starts (human_grasp_seeds, a rough pose) or another family")
        mu = [np.zeros(D) for _ in all_starts]
        sig = [np.full(D, 0.45) for _ in all_starts]
        while len(evals) < int(max_evals) and it < int(iterations) and not any(r.get("ok") for r in evals):
            batch = []
            capacity = min(int(population), int(max_evals) - len(evals))
            if it == 0:
                for s in range(preflight_count, min(len(all_starts), preflight_count + capacity)):
                    batch.append((s, np.zeros(D)))
            elif it == 1:
                # A seated grasp may need only a different squeeze, opening, or
                # approach. Simultaneous perturbations of all nine dimensions
                # usually destroy its contact geometry before that is tested.
                # Probe controls individually around each best start, then use
                # the remaining budget for the full geometric search.
                best_per_start = {}
                for result in sorted(evals, key=lambda r: -r["score"]):
                    best_per_start.setdefault(result["start"], result)
                for key in ("press_depth_rad", "open_frac", "approach_up"):
                    for result in list(best_per_start.values())[:3]:
                        for direction in (-.4, .4):
                            if len(batch) >= capacity:
                                break
                            u = np.asarray(result["u"], float).copy()
                            axis = PARAMS.index(key)
                            u[axis] = np.clip(u[axis] + direction, -1., 1.)
                            batch.append((result["start"], u))
            n_more = max(0, capacity - len(batch))
            elite = sorted(evals, key=lambda r: -r["score"])[:max(2, int(round(elite_frac * max(len(evals), population))))]
            w = np.ones(len(all_starts))
            for r in elite:
                w[r["start"]] += 3.0
            w /= w.sum()
            for _ in range(n_more):
                s = int(rng.choice(len(all_starts), p=w))
                u = np.clip(rng.normal(mu[s], sig[s]), -1.0, 1.0)
                batch.append((s, u))
            jobs = [dict(common, kind="eval", index=len(evals) + i, start_index=s, start=all_starts[s], theta=_theta(u, all_starts[s], bnd), u=u.tolist())
                    for i, (s, u) in enumerate(batch)]
            res = pool.map(jobs, search_state={"phase": "cem", "iteration": it, "evals": evals, "starts": all_starts,
                                               "mu": mu, "sig": sig, "rng_state": rng.bit_generator.state})
            for j, r in zip(jobs, res):
                r["u"] = j["u"]
            evals += res
            it += 1
            if any(r.get("ok") for r in evals):
                break
            # refit each start's Gaussian to its elites
            elite = sorted(evals, key=lambda r: -r["score"])[:max(2, int(round(elite_frac * len(evals))))]
            for s in range(len(all_starts)):
                U = np.array([r["u"] for r in elite if r["start"] == s])
                if len(U) >= 2:
                    mu[s] = U.mean(axis=0)
                    sig[s] = np.maximum(0.6 * sig[s] + 0.4 * U.std(axis=0), 0.08)
                elif len(U) == 1:
                    mu[s] = U[0]
                    sig[s] = np.maximum(0.8 * sig[s], 0.1)
                else:
                    sig[s] = np.minimum(sig[s] * 1.15, 0.7)          # nothing good near this start: look wider
    except _SearchStopped:
        return pool.stopped_result(time.time() - t0)
    finally:
        pool.close()
    if checkpoint_json is not None:
        if _checkpoint_binding(binding_inputs, scene, bench_scene, state) != binding:
            raise ValueError("search inputs, compiled physics, code or runtime changed during this search")
        try:
            pool.check_stop()
        except _SearchStopped:
            return pool.stopped_result(time.time() - t0)
        pool.record["status"] = "complete"
        pool.record.pop("stop_reason", None)
        pool.save()
    ranked = sorted(evals, key=lambda r: -r["score"])
    passing = [r for r in ranked if r.get("ok")]
    classes = {}
    for r in evals:
        if not r.get("ok") and r.get("reason"):
            k = str(r["reason"]).split(" (")[0]
            classes[k] = classes.get(k, 0) + 1
    best = passing[0] if passing else (ranked[0] if ranked else None)
    pose = None
    if best is not None and best.get("pose") is not None:
        pose = dict(best["pose"])
        start = all_starts[best["start"]]
        pose["source"] = "optimize_grasp_pose"
        pose["family"] = pose.get("family") or start.get("family") or fam
        pose["stress_tests"] = best.get("tests") or {}
        pose["optimize"] = {"theta": best["theta"], "start_index": best["start"], "start_source": start.get("source"),
                            "score": best["score"], "arm": best.get("arm"), "static": best.get("static"), "passed": bool(best.get("ok"))}
    out = {"ok": bool(passing), "pose": pose, "passing": [dict(r["pose"], stress_tests=r.get("tests") or {}, source="optimize_grasp_pose")
                                                         for r in passing[:6]],
           "failure_classes": dict(sorted(classes.items(), key=lambda kv: -kv[1])),
           "best_tries": [{"score": r["score"], "ok": bool(r.get("ok")), "reason": r.get("reason"), "start": r["start"],
                           "theta": {k: round(float(v), 4) for k, v in r["theta"].items()}, "static": r.get("static"), "arm": r.get("arm"),
                           "tests": r.get("tests")} for r in ranked[:5]],
           "hint": None if passing else _hint(classes, fam, evals), "family": fam, "family_note": note,
           "starts": [{"source": s.get("source"), "family": s.get("family"), "frame": s.get("frame")} for s in all_starts],
           "start_errors": errors[:4], "n_evals": len(evals), "iterations": it,
           "n_screened_out": sum(1 for r in evals if r.get("stage") == "static"),
           "n_arm_rejected": sum(1 for r in evals if r.get("stage") == "arm"),
           "n_tested": sum(1 for r in evals if r.get("stage") == "tests"), "tests": tests, "seconds": round(time.time() - t0, 1),
           "bench_state_source": bench_state_source,
           "supplied_starts_tested_first": preflight_count, "start_generation": generation,
           "search_scope": "supplied starts passed requested checks" if passing and not gen and preflight_count else "bounded candidate search"}
    out.update(status="passed" if passing else "budget_exhausted", complete=True)
    if checkpoint_json is not None:
        out.update(checkpoint_json=str(pool.path), resumed=bool(resume), cached_evaluations_rechecked=pool.rechecked)
    if bench_scene and not scene.get("bench_state"):
        out["state_warning"] = ("This legacy hand scene has no bench-state provenance. Its object state cannot be checked "
                                "against the arm screen; after motion, rebuild it from the actual physical recording.")
    if out_json and pose is not None:
        # only a passing pose goes to out_json; the best failed try is kept beside it as a start for the next attempt
        p = Path(out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        if passing:
            p.write_text(json.dumps(_jsonable(pose), indent=1))
            out["out_json"] = str(p)
            q = p.with_name(p.stem + "_passing.json")
            q.write_text(json.dumps(_jsonable(out["passing"]), indent=1))
            out["passing_json"] = str(q)
        else:
            q = p.with_name(p.stem + "_best_try.json")
            q.write_text(json.dumps(_jsonable(pose), indent=1))
            out["best_try_json"] = str(q)
    return _jsonable(out)


def _worker_main(fin, fout):
    """A launcher batch: run the jobs with a local pool and write the results."""
    spec = json.loads(Path(fin).read_text())
    pool = _Pool(int(spec.get("workers") or _default_workers()))
    try:
        res = pool.map(spec["jobs"])
    finally:
        pool.close()
    Path(fout).write_text(json.dumps(_jsonable(res)))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        _worker_main(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 2 and sys.argv[1] == "--serve":
        _serve()
