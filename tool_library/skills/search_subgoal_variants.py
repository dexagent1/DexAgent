"""Search a failing subgoal's free parameters in parallel physics trials, then dig into the next failure."""
from __future__ import annotations

TOOL = {
    "name": "search_subgoal_variants",
    "title": "Search subgoal variants",
    "type": "skill",
    "category": "Trajectory & Execution",
    "summary": "From an execution whose earlier subgoals passed, run many variants of the failing subgoal's parameters as "
               "parallel physical trials (Latin hypercube, then cross-entropy refits on the best), rank them on subgoals "
               "passed, the Stage 1 goals and the measured shortfall of the failed checks; when a variant passes, continue "
               "the search on the next failing subgoal from its state (backtracking to another passing variant if that one "
               "dead-ends). Returns the best plan and its execution folder to re-execute and package.",
    "description": (
        "Each trial is an ordinary execute_subgoal_plan run with resume=True from a copy of the prefix execution, restarting "
        "at the target subgoal (from_subgoal), so the subgoals before it are not simulated again and everything after it is "
        "run as planned until the next failure. A variant changes only named keys of the target plan entry: `knobs` maps a "
        "key (for a two-hand entry 'left.<key>' / 'right.<key>') to [lo, hi] (absolute value), {low: [...], high: [...]} "
        "(absolute vector), {delta_low: [...], delta_high: [...], default: [...]} (added to the entry's value, e.g. a place "
        "offset; `pose_shift_m` with a delta translates the entry's grasp pose file, so the hand lands elsewhere on the part), {scale: [lo, hi], default: x} (multiplies the entry's value, e.g. duration_s) or {choices: [...]} (e.g. "
        "arm_path [null, 'joint'], or alternative pose files). Without knobs they are proposed from the failed checks and "
        "the motion (the grasp pose's position, timing and press depth for rise/in_hand/pads_touching/no_slip; place offset, set-down time, contact force and "
        "release timing for at_target/at_rest/released/upright; approach and carry heights and a joint-space transfer for "
        "penetration and reach; durations, travel and force for joints, slides and seats); when no variant of them passes, "
        "a second level searches what led into the failure together with them: the destination of the carry before it and "
        "the descent, closing time and press of the grasp that holds the same object (trials restart at that grasp, since "
        "how the object sits in the hand decides where it lands). The first round is a Latin "
        "hypercube of `population` variants; each further round samples a Gaussian (categoricals: frequencies) refitted to "
        "the `elite` best, with a variance floor. A trial is ranked by (plan subgoals passed, Stage 1 goals met when every "
        "subgoal passed and scene_semantic_json is given, fewer failed checks, smaller normalized shortfall: e.g. target "
        "error / 5 mm, slip / 5 mm, penetration / 1 mm, missing rise / 10 mm). With dig=true, once a variant passes the "
        "target, the next failing subgoal of the best such variant becomes the target (its knobs proposed from its own "
        "failure), up to max_depth subgoals; when a deeper target finds no passing variant, the search backtracks once to "
        "the next passing variant of the level above. Budgets: max_evals trials, max_minutes wall time, eval_timeout_s per "
        "trial, `workers` trials at a time. Gates are not changed or relaxed: every trial is judged by the executor's own "
        "checks, and the result is a candidate: re-execute the best plan with execute_subgoal_plan (resume=True in the "
        "returned folder reuses its segments) so the harness records the execution, then package_stage3_episode and finish."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "The scene the prefix was executed with (as given to execute_subgoal_plan)."},
        {"name": "plan", "type": "str | dict", "description": "The plan of the prefix execution (a JSON path: variants are written beside it so relative paths resolve the same; or a dict)."},
        {"name": "out_dir", "type": "str", "description": "Search folder: one sub-folder per trial, plans/, search_report.json."},
        {"name": "prefix_dir", "type": "str | None", "description": "out_dir of the execute_subgoal_plan run to start from (its passing segments are reused). None: every trial runs from the start."},
        {"name": "subgoal", "type": "int | None", "description": "Plan index of the target subgoal. Default: the first failure in prefix_dir."},
        {"name": "step", "type": "int | None", "description": "For entries sharing an index: which one (1-based). Default: the failing one."},
        {"name": "knobs", "type": "dict | None", "description": "Key -> range spec (see description) for the target entry, or {'<index>': {key: spec}} per subgoal (earlier indices make the trials restart there). Default (recommended unless you know which parameters matter): proposed from the failure, then from the carry and grasp that led into it. Given knobs that find no pass are followed by the proposed ones."},
        {"name": "population", "type": "int", "description": "Trials per round. Default 6."},
        {"name": "rounds", "type": "int", "description": "Rounds per search level (own knobs, then upstream knobs). Default 2."},
        {"name": "elite", "type": "int", "description": "Best trials the next round is refitted to. Default 2."},
        {"name": "workers", "type": "int", "description": "Parallel trials. Default 3."},
        {"name": "max_evals", "type": "int", "description": "Total trial budget. Default 24."},
        {"name": "max_minutes", "type": "float", "description": "Wall-time budget; no new round starts after it. Default 40."},
        {"name": "eval_timeout_s", "type": "float", "description": "One trial's time limit. Default 900."},
        {"name": "dig", "type": "bool", "description": "Continue on the next failing subgoal once one passes. Default true."},
        {"name": "max_depth", "type": "int", "description": "Subgoals searched in one call when digging. Default 4."},
        {"name": "exec_options", "type": "dict | None", "description": "execute_subgoal_plan options used by the prefix run (must match, or its segments are simulated again)."},
        {"name": "scene_semantic_json, scene_recon_json", "type": "str | None", "description": "Stage 1 / Stage 2 files: complete trials are also scored by stage1_goal_gate."},
        {"name": "seed", "type": "int", "description": "Sampling seed. Default 0."},
    ],
    "outputs": [
        {"name": "result", "type": "dict", "description": "ok (a trial completed the plan), best {plan, out_dir, subgoals_passed, n_subgoals, complete, goals_ok, failed, knobs}, baseline, improved, evaluations, targets (per searched subgoal: knobs, trials, passed), table (best trials), report (search_report.json), next_step."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "search", "cem", "parallel", "subgoals"],
    "used_in": [],
    "status": "tested",
    "entrypoint": "search_subgoal_variants",
}

import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
GRASP = {"grasp_lift", "hold", "pin", "grasp_more", "regrasp", "edge_pinch", "strip_pinch", "tilt_up", "hook", "lift_together"}
PLACE = {"place", "carry", "insert", "release", "carry_together", "lay", "drag"}
JOINT = {"push", "pull", "press", "open_joint", "close_joint", "rotate", "tilt", "pour", "slide", "unscrew", "screw", "turn"}
SEAT = {"press_seat"}
ANY = None
# failed check -> [(key, spec, motions)]: the parameters that change what the check measures
SHIFT = ("pose_shift_m", {"delta_low": [-0.012, -0.012, -0.012], "delta_high": [0.012, 0.012, 0.012], "default": [0.0, 0.0, 0.0]},
         GRASP | JOINT | SEAT)                      # any motion that puts a pose on a part
GRIP = [("t_close", [0.5, 1.6], GRASP), ("press_depth_rad", [0.1, 0.5], GRASP | JOINT), ("t_grip", [0.3, 1.2], GRASP),
        ("approach_m", [0.02, 0.08], GRASP | JOINT | SEAT)]
SETDOWN = [("set_down_s", [0.6, 2.5], PLACE), ("placement_force_n", [6.0, 25.0], PLACE), ("t_open", [0.3, 1.5], ANY),
           ("settle_s", [0.3, 1.5], ANY)]
CLEAR = [("approach_m", [0.03, 0.10], ANY), ("carry_rise_m", [0.02, 0.14], PLACE), ("via_rise_m", [0.03, 0.15], PLACE),
         ("arm_path", {"choices": [None, "joint"]}, PLACE)]
AUTO_KNOBS = {
    "rise": GRIP[:2] + [SHIFT] + GRIP[2:] + [("lift_m", [0.03, 0.10], GRASP)],
    "in_hand": [SHIFT] + GRIP, "pads_touching": [SHIFT, GRIP[1], GRIP[0], GRIP[3]],
    "no_slip": [("press_depth_rad", [0.1, 0.5], ANY), ("t_grip", [0.3, 1.2], GRASP), ("duration_s", {"scale": [1.0, 2.2]}, ANY)],
    "at_target": [("offset", {"delta_low": [-0.015, -0.015, -0.004], "delta_high": [0.015, 0.015, 0.010], "default": [0.0, 0.0, 0.0]}, PLACE),
                  ("set_down_s", [0.6, 2.5], PLACE), ("carry_rise_m", [0.02, 0.12], PLACE)],
    "at_target_orientation": SETDOWN, "at_rest": SETDOWN, "release_settle": SETDOWN, "upright": SETDOWN, "stays_put": SETDOWN,
    "released": [("t_open", [0.3, 1.5], ANY), ("t_retreat", [0.5, 1.6], ANY), ("settle_s", [0.3, 1.5], ANY)],
    "supported": [("offset", {"delta_low": [-0.008, -0.008, -0.006], "delta_high": [0.008, 0.008, 0.002], "default": [0.0, 0.0, 0.0]}, PLACE)] + SETDOWN[:2],
    "hand_penetration": CLEAR, "arm_collision": [SHIFT] + CLEAR, "hand_clearance": CLEAR, "plan_collision": CLEAR,
    "untouched": CLEAR[:2] + [("offset", {"delta_low": [-0.01, -0.01, 0.0], "delta_high": [0.01, 0.01, 0.008], "default": [0.0, 0.0, 0.0]}, PLACE)],
    "object_penetration": [("offset", {"delta_low": [0.0, 0.0, 0.0], "delta_high": [0.0, 0.0, 0.010], "default": [0.0, 0.0, 0.0]}, PLACE)] + CLEAR[:2],
    "arm_unreachable": [("arm_path", {"choices": [None, "joint"]}, PLACE), SHIFT, ("max_reach_turn_deg", [15.0, 45.0], ANY), ("approach_m", [0.03, 0.10], ANY)],
    "reach_turn_too_large": [("max_reach_turn_deg", [15.0, 60.0], ANY)],
    "servo_tracking": [("duration_s", {"scale": [1.2, 2.5]}, ANY)],
    "joint_reached": [("duration_s", {"scale": [1.0, 2.0]}, ANY), SHIFT, ("angle_deg", {"scale": [1.0, 1.3]}, JOINT), ("press_depth_rad", [0.1, 0.5], ANY)],
    "joint_moved": [("duration_s", {"scale": [1.0, 2.0]}, ANY), ("press_depth_rad", [0.1, 0.5], ANY), ("approach_m", [0.02, 0.08], ANY)],
    "displaced": [("distance_m", {"scale": [1.0, 1.4]}, ANY), ("force_n", [15.0, 40.0], ANY), ("duration_s", {"scale": [1.0, 2.0]}, ANY)],
    "seated": [("force_n", [20.0, 40.0], SEAT), ("travel_m", [0.01, 0.05], SEAT), ("press_s", [0.8, 2.0], SEAT)],
    "turned": [("bite_deg", [30.0, 120.0], JOINT), ("duration_s", {"scale": [1.0, 2.0]}, ANY)],
}
GENERIC = [("duration_s", {"scale": [0.8, 2.0]}, ANY), ("approach_m", [0.02, 0.08], ANY)]
# failed check -> (measured key, kind, scale): the shortfall a trial is ranked by
RESIDUAL = {"rise": ("rise_m", "deficit", 0.01), "no_slip": ("slip_in_hand_m", "excess", 0.005),
            "at_target": ("target_err_m", "excess", 0.005), "at_rest": ("moved_in_last_window_m", "excess", 0.002),
            "hand_penetration": ("max_hand_pen_mm", "excess", 1.0), "object_penetration": ("max_object_pen_mm", "excess", 1.0),
            "servo_tracking": ("max_track_pos_mm", "excess", 5.0), "displaced": ("displacement_along_m", "deficit", 0.01),
            "seated": ("seat_remaining_mm", "abs", 1.0), "joint_reached": ("joint_travel_fraction", "deficit", 0.1),
            "upright": ("object_tilt_end_deg", "excess", 5.0), "stays_put": ("object_moved_m", "excess", 0.005),
            "arm_unreachable": ("position_error_mm", "excess", 5.0), "turned": ("turned_deg", "deficit", 10.0),
            "at_target_orientation": ("target_rotation_error_deg", "excess", 10.0),
            "release_settle": ("release_settle.moved_since_release_m", "excess", 0.005),
            "plan_collision": ("plan_collisions", "count", 1.0), "pads_touching": ("pads", "pads_missing", 1.0)}


# ----------------------------------------------------------------------------- plan entries and knobs
def _plan_doc(plan):
    if isinstance(plan, (str, Path)):
        return json.loads(Path(plan).read_text()), Path(plan).resolve().parent
    return copy.deepcopy(plan), None


def _entries(doc):
    return doc["subgoals"] if isinstance(doc, dict) else doc


def _target_entry(doc, index, step):
    """The plan entry a knob changes: the entry with this index (the step-th of several)."""
    same = [i for i, e in enumerate(_entries(doc)) if int(e.get("index", i + 1)) == int(index)]
    if not same:
        same = [int(index) - 1] if 0 < int(index) <= len(_entries(doc)) else []
    if not same:
        raise ValueError(f"no plan entry with index {index}")
    return same[min(max(int(step or 1), 1), len(same)) - 1]


def _get(entry, key):
    if "." in key:
        side, k = key.split(".", 1)
        return ((entry.get("hands") or {}).get(side) or {}).get(k)
    return entry.get(key)


def _set(entry, key, value):
    if "." in key:
        side, k = key.split(".", 1)
        entry.setdefault("hands", {}).setdefault(side, {})[k] = value
    else:
        entry[key] = value


def _shift_pose(entry, key, delta, plan_dir, dst):
    """The `pose_shift_m` knob: the entry's grasp pose translated by delta (world metres, the frame its `pos` is in),
    written as a new pose file the entry then names; finger targets and orientation are kept."""
    side = key.split(".", 1)[0] if "." in key else None
    holder = ((entry.get("hands") or {}).get(side) or {}) if side else entry
    ref = holder.get("pose")
    if not isinstance(ref, str):
        return                                            # an inline pose or a candidate list is left as it is
    src = Path(ref)
    if not src.is_absolute():
        src = src if src.exists() else (Path(plan_dir) / src if plan_dir else src)
    pose = json.loads(src.read_text())
    poses = pose if isinstance(pose, list) else [pose]  # a ranked candidate list: every candidate moves the same way
    if not all(isinstance(q, dict) and isinstance(q.get("pos"), list) and len(q["pos"]) == 3 for q in poses):
        return
    for q in poses:
        q["pos"] = [float(a) + float(b) for a, b in zip(q["pos"], delta)]
        q["shifted_from"] = {"pose": str(src.resolve()), "delta_m": [float(x) for x in delta]}
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text(json.dumps(pose, indent=1))
    holder["pose"] = str(Path(dst).resolve())


def _motion_of(entry, key):
    if "." in key:
        return str(((entry.get("hands") or {}).get(key.split(".", 1)[0]) or {}).get("motion") or "")
    return str(entry.get("motion") or "")


def _split_check(check):
    """'left:rise' -> ('left', 'rise'); 'untouched:bowl' -> (None, 'untouched'); 'rise' -> (None, 'rise')."""
    head, _, tail = str(check).partition(":")
    if tail and head in ("left", "right"):
        return head, tail.partition(":")[0]
    return None, head


def auto_knobs(entry, failure, limit=5, per_check=2):
    """Knobs for an entry from its failed checks and motion (two-hand checks 'left:rise' -> 'left.<key>'): up to
    per_check for each failed check first, then the rest up to limit."""
    picks = []
    failed = [str(f) for f in (failure or {}).get("failed") or []]
    for check in failed:
        side, name = _split_check(check)
        rows = []
        for key, spec, motions in AUTO_KNOBS.get(name, []):
            k = f"{side}.{key}" if side else key
            if side is None and entry.get("motion") == "both":
                continue                                  # a two-hand entry's keys live under its hands
            if motions is not ANY and _motion_of(entry, k) not in motions:
                continue
            if isinstance(spec, dict) and "scale" in spec and _get(entry, k) is None and "default" not in spec:
                continue                                  # nothing to scale
            rows.append((k, spec))
        picks.append(rows)
    out = {}
    for rows in picks:
        for k, spec in rows[:per_check]:
            out.setdefault(k, spec)
    for rows in picks:
        for k, spec in rows[per_check:]:
            if len(out) < limit:
                out.setdefault(k, spec)
    if not out:
        for key, spec, _ in GENERIC:
            if not (isinstance(spec, dict) and "scale" in spec and _get(entry, key) is None):
                out[key] = spec
    return dict(list(out.items())[:limit])


def upstream_knobs(doc, ei, failure, limit=6):
    """When the failing entry's own parameters do not help, what led into it: the carry before it (its destination)
    and the grasp of the same object (descent, closing time, press), plus the failing entry's first two knobs.
    {entry position: {key: spec}}."""
    es = _entries(doc)
    entry = es[ei]
    obj, side = entry.get("object"), entry.get("side")
    out = {ei: dict(list(auto_knobs(entry, failure).items())[:2])}
    for j in range(ei - 1, -1, -1):
        e = es[j]
        if e.get("object") != obj or (side and e.get("side") not in (None, side)):
            continue
        m = str(e.get("motion") or "")
        if m in ("carry", "carry_together") and j not in out:
            key = "target_pos" if e.get("target_pos") is not None else "offset"
            out[j] = {key: {"delta_low": [-0.01, -0.01, 0.0], "delta_high": [0.01, 0.01, 0.02], "default": [0.0, 0.0, 0.0]}}
        elif m in GRASP:
            out[j] = {"t_descend": [0.8, 3.0], "t_close": [0.5, 1.6], "press_depth_rad": [0.05, 0.4]}
            break                                         # the grasp that holds it: nothing earlier shapes the hold
    n, kept = 0, {}
    for j in sorted(out, reverse=True):
        for k, v in out[j].items():
            if n < limit:
                kept.setdefault(j, {})[k] = v; n += 1
    return kept


class Space:
    """Knob specs ({entry position: {key: spec}}) as dimensions in [0, 1] (continuous) and categoricals."""

    def __init__(self, knobs_by_entry, doc):
        self.dims, self.cats = [], []                    # dims: (key, component|None, lo, hi, mode, base)
        knobs = {f"{ei}@{k}": v for ei, spec in knobs_by_entry.items() for k, v in spec.items()}
        for key, spec in knobs.items():
            ei, _, short = key.partition("@")
            cur = _get(_entries(doc)[int(ei)], short)
            if isinstance(spec, (list, tuple)) and len(spec) == 2 and all(isinstance(v, (int, float)) for v in spec):
                self.dims.append((key, None, float(spec[0]), float(spec[1]), "abs", None))
            elif isinstance(spec, dict) and "choices" in spec:
                self.cats.append((key, list(spec["choices"])))
            elif isinstance(spec, dict) and "scale" in spec:
                base = cur if cur is not None else spec.get("default")
                if isinstance(base, (int, float)):
                    self.dims.append((key, None, float(spec["scale"][0]), float(spec["scale"][1]), "scale", float(base)))
            elif isinstance(spec, dict) and ("delta_low" in spec or "low" in spec):
                lo, hi = (spec["delta_low"], spec["delta_high"]) if "delta_low" in spec else (spec["low"], spec["high"])
                base = cur if cur is not None else spec.get("default", [0.0] * len(lo))
                for c in range(len(lo)):
                    if float(hi[c]) != float(lo[c]):
                        self.dims.append((key, c, float(lo[c]), float(hi[c]), "delta" if "delta_low" in spec else "abs",
                                          list(map(float, base)) if "delta_low" in spec else [float(v) for v in lo]))
            else:
                raise ValueError(f"knob {key!r}: unsupported spec {spec!r}")

    def lhs(self, n, rng):
        u = np.empty((n, len(self.dims)))
        for j in range(len(self.dims)):
            u[:, j] = (rng.permutation(n) + rng.random(n)) / n
        c = [[rng.integers(len(ch)) for _ in range(n)] for _, ch in self.cats]
        return [(u[i], [c[j][i] for j in range(len(self.cats))]) for i in range(n)]

    def refit(self, elite_points, n, rng, floor=0.08):
        U = np.array([p[0] for p in elite_points]) if self.dims else np.zeros((len(elite_points), 0))
        mu, sd = U.mean(0), np.maximum(U.std(0), floor)
        out = []
        for _ in range(n):
            u = np.clip(rng.normal(mu, sd), 0.0, 1.0)
            cs = []
            for j, (_, ch) in enumerate(self.cats):
                cnt = np.ones(len(ch))
                for p in elite_points:
                    cnt[p[1][j]] += 2.0
                cs.append(int(rng.choice(len(ch), p=cnt / cnt.sum())))
            out.append((u, cs))
        return out

    def values(self, point):
        """{"<entry position>@<key>": value} of a point, vectors assembled per key."""
        u, cs = point
        vals, vec = {}, {}
        for j, (key, comp, lo, hi, mode, base) in enumerate(self.dims):
            x = lo + float(u[j]) * (hi - lo)
            if comp is None:
                vals[key] = round(base * x if mode == "scale" else x, 5)
            else:
                v = vec.setdefault(key, list(base))
                v[comp] = round((base[comp] + x) if mode == "delta" else x, 5)
        vals.update(vec)
        for j, (key, ch) in enumerate(self.cats):
            vals[key] = ch[cs[j]]
        return vals


# ----------------------------------------------------------------------------- scoring
def residual(failure):
    tot = 0.0
    measured = (failure or {}).get("measured") or {}
    for check in (failure or {}).get("failed") or []:
        side, check = _split_check(check)
        data = ((measured.get("hands") or {}).get(side) or {}) if side else measured
        key, kind, scale = RESIDUAL.get(check, (None, None, 1.0))
        v = data
        for part in (key or "").split("."):
            v = v.get(part) if isinstance(v, dict) and part else None
        if kind == "count":
            v = len(v) if isinstance(v, (list, dict)) else v
        elif kind == "pads_missing":                     # pads off the part, plus how far off they are (20 mm = one pad)
            off = [p for p in v.values() if isinstance(p, dict) and not p.get("touching")] if isinstance(v, dict) else None
            v = None if off is None else len(off) + sum(abs(float(p.get("offset_mm") or 0.0)) for p in off) / 20.0
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not np.isfinite(v):
            tot += 1.0
        elif kind == "deficit":
            tot += -float(v) / scale
        elif kind == "abs":
            tot += abs(float(v)) / scale
        else:
            tot += float(v) / scale
    return round(tot, 4)


def score_of(row):
    """Larger is better: (plan subgoals passed, Stage 1 goals met (-1 unless measured), -failed checks, -shortfall)."""
    if row.get("error"):
        return (-1, -1, -99, -1e9)
    ff = row.get("first_failure") or {}
    return (row["subgoals_passed"], row.get("goals_ok", -1), -len(ff.get("failed") or []), -residual(ff))


# ----------------------------------------------------------------------------- one trial (a child process)
def _copy_prefix(prefix, dst, keep_before):
    """The prefix execution's passing segments before the target (by plan index), without its results."""
    prefix, dst = Path(prefix), Path(dst)
    outcomes = json.loads((prefix / "outcomes.json").read_text()) if (prefix / "outcomes.json").exists() else []
    keep = {r.get("tag") for r in outcomes if r.get("ok") and int(r["index"]) < int(keep_before)}
    dst.mkdir(parents=True, exist_ok=True)
    for p in prefix.iterdir():
        if p.is_dir() and p.name in keep:
            shutil.copytree(p, dst / p.name)
    (dst / "outcomes.json").write_text(json.dumps([r for r in outcomes if r.get("tag") in keep], indent=1))


def _trial(spec):
    from tool_library.skills.execute_subgoal_plan import execute_subgoal_plan
    out = Path(spec["out_dir"])
    if spec.get("prefix_dir"):
        _copy_prefix(spec["prefix_dir"], out, spec["from_subgoal"])
    t0 = time.time()
    r = execute_subgoal_plan(spec["scene"], spec["plan"], str(out), resume=bool(spec.get("prefix_dir")),
                             from_subgoal=spec["from_subgoal"] if spec.get("prefix_dir") else None, **(spec.get("exec_options") or {}))
    keep = ("index", "step", "tag", "motion", "side", "object", "ok", "failed", "frames", "stage1_indices")
    res = {"ok": bool(r.get("ok")), "n_subgoals": r.get("n_subgoals"), "seconds": round(time.time() - t0, 1),
           "subgoals": [{k: g.get(k) for k in keep if k in g} for g in r.get("subgoals") or []],
           "first_failure": {k: (r.get("first_failure") or {}).get(k) for k in ("index", "step", "tag", "motion", "failed", "measured", "error")}
           if isinstance(r.get("first_failure"), dict) else None, "files": r.get("files") or {}}
    (out / "trial_result.json").write_text(json.dumps(res, indent=1, default=float))


def _run_trials(specs, workers, timeout_s, cwd):
    """Run trial specs as child processes, `workers` at a time; a trial past timeout_s is killed."""
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO}{os.pathsep}" + env.get("PYTHONPATH", "")
    env.setdefault("MUJOCO_GL", "egl")
    pending, running, done = list(specs), [], []
    while pending or running:
        while pending and len(running) < max(1, int(workers)):
            s = pending.pop(0)
            Path(s["out_dir"]).mkdir(parents=True, exist_ok=True)
            (Path(s["out_dir"]) / "trial_spec.json").write_text(json.dumps(s, indent=1, default=str))
            log = open(Path(s["out_dir"]) / "trial.log", "w")
            p = subprocess.Popen([sys.executable, "-m", "tool_library.skills.search_subgoal_variants", "--trial",
                                  str(Path(s["out_dir"]) / "trial_spec.json")], cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT)
            running.append((p, s, time.time(), log))
        time.sleep(0.5)
        for item in list(running):
            p, s, t0, log = item
            if p.poll() is None and time.time() - t0 < timeout_s:
                continue
            if p.poll() is None:
                p.kill(); p.wait()
            log.close()
            running.remove(item)
            res_path = Path(s["out_dir"]) / "trial_result.json"
            if res_path.exists():
                res = json.loads(res_path.read_text())
            else:
                tail = (Path(s["out_dir"]) / "trial.log").read_text()[-1500:]
                res = {"error": "timeout" if time.time() - t0 >= timeout_s else f"trial crashed: {tail}"}
            done.append((s, res))
    return done


# ----------------------------------------------------------------------------- the search
def _goal_score(row, semantic, recon, folder):
    """Stage 1 goals met by a complete trial (stage1_goal_gate on its executed frames)."""
    from tool_library.verifiers.stage1_goal_gate import stage1_goal_gate
    files = row.get("files") or {}
    executed = [g for g in row.get("subgoals") or [] if g.get("frames")]
    if not (files.get("trajectory") and files.get("scene_physics") and executed):
        return -1, ["no recording"]
    man = {"scene_recon": str(recon), "subgoals": [{"index": g["index"], "stage1_indices": g.get("stage1_indices") or [g["index"]],
                                                     "frames": g["frames"], "motion": g.get("motion"), "side": g.get("side"),
                                                     "object": g.get("object")} for g in executed]}
    mp = Path(folder) / "goal_manifest.json"
    mp.write_text(json.dumps(man, indent=1))
    try:
        rep = stage1_goal_gate(str(semantic), str(mp), str(files["scene_physics"]), str(files["trajectory"]), str(recon),
                               out_json=str(Path(folder) / "goal_gate.json"))
    except Exception as e:
        return -1, [f"goal gate crashed: {type(e).__name__}: {e}"]
    goals = sum(1 for r in rep.get("subgoals") or [] if r.get("ok"))
    return goals + (1 if rep.get("ok") else 0), [str(i) for i in rep.get("issues") or []][:4]


def search_subgoal_variants(scene, plan, out_dir, prefix_dir=None, subgoal=None, step=None, knobs=None, population=6,
                            rounds=2, elite=2, workers=3, max_evals=24, max_minutes=40.0, eval_timeout_s=900.0, dig=True,
                            max_depth=4, exec_options=None, scene_semantic_json=None, scene_recon_json=None, seed=0):
    t_start, cwd = time.time(), Path.cwd()
    out = Path(out_dir).resolve()
    (out / "plans").mkdir(parents=True, exist_ok=True)
    doc0, plan_dir = _plan_doc(plan)
    plan_home = plan_dir if plan_dir is not None else out / "plans"     # variants beside the plan: relative paths resolve the same
    stem = Path(plan).stem if isinstance(plan, (str, Path)) else "plan"
    rng = np.random.default_rng(seed)
    n_entries = len(_entries(doc0))
    rows, targets = [], []

    baseline = None
    if prefix_dir and (Path(prefix_dir) / "execution_report.json").exists():
        rep = json.loads((Path(prefix_dir) / "execution_report.json").read_text())
        baseline = {"id": "baseline", "out_dir": str(Path(prefix_dir).resolve()), "plan": str(plan) if plan_dir else None,
                    "subgoals_passed": sum(1 for g in rep.get("subgoals") or [] if g.get("ok")), "subgoals": rep.get("subgoals"),
                    "first_failure": rep.get("first_failure") if isinstance(rep.get("first_failure"), dict) else None,
                    "ok": bool(rep.get("ok")), "files": rep.get("files") or {}, "knobs": {}}
    if subgoal is None:
        ff = (baseline or {}).get("first_failure") or {}
        if ff.get("index") is None:
            raise ValueError("no target: give `subgoal`, or a prefix_dir whose execution failed")
        subgoal, step = int(ff["index"]), step or ff.get("step")

    def time_left():
        return max_minutes * 60.0 - (time.time() - t_start)

    def entry_index(doc, j):
        return int(_entries(doc)[j].get("index", j + 1))

    def search_level(level_doc, prefix, index, stp, failure, knob_map, mode):
        """Rounds of trials whose target is subgoal `index`. knob_map {entry position: {key: spec}} may name earlier
        entries too: the trials then restart at the earliest of them."""
        ei = _target_entry(level_doc, index, stp)
        space = Space(knob_map, level_doc)
        start = min([entry_index(level_doc, j) for j in knob_map] + [int(index)])
        level = {"subgoal": index, "step": stp, "motion": _entries(level_doc)[ei].get("motion"), "mode": mode, "restart_at": start,
                 "knobs": {f"{entry_index(level_doc, j)}:{k}": v for j, spec in knob_map.items() for k, v in spec.items()},
                 "prefix": str(prefix) if prefix else None, "trials": 0, "passed": 0, "failure_before": (failure or {}).get("failed")}
        targets.append(level)
        if not space.dims and not space.cats:
            level["note"] = "no knob applies to this failure; change the plan (grasp, approach, support) instead"
            return []
        level_rows, points = [], space.lhs(population, rng)
        for rnd in range(int(rounds)):
            budget = int(max_evals) - len(rows)
            if budget <= 0 or time_left() <= 0:
                break
            batch = []
            for point in points[:budget]:
                vals, shown = space.values(point), {}
                d = copy.deepcopy(level_doc)
                rid = f"t{len(rows) + len(batch):03d}"
                for ck, v in vals.items():
                    pos, _, key = ck.partition("@")
                    if key.rpartition(".")[2] == "pose_shift_m":
                        _shift_pose(_entries(d)[int(pos)], key, v, plan_dir, out / "poses" / f"{rid}_{pos}.json")
                    else:
                        _set(_entries(d)[int(pos)], key, v)
                    shown[f"{entry_index(d, int(pos))}:{key}"] = v
                pf = plan_home / f"{stem}.search_{out.name}_{rid}.json"
                pf.write_text(json.dumps(d, indent=1))
                batch.append(({"scene": str(scene), "plan": str(pf), "out_dir": str(out / rid), "prefix_dir": str(prefix) if prefix else None,
                               "from_subgoal": start, "exec_options": exec_options or {}}, rid, point, shown, d))
            done = _run_trials([b[0] for b in batch], workers, eval_timeout_s, cwd)
            by_dir = {s_["out_dir"]: res for s_, res in done}
            for spec_, rid, point, shown, d in batch:
                res = by_dir.get(spec_["out_dir"]) or {"error": "not run"}
                row = {"id": rid, "target": index, "step": stp, "mode": mode, "round": rnd, "knobs": shown, "plan": spec_["plan"],
                       "out_dir": spec_["out_dir"], "doc": d, "point": point, **res}
                row["subgoals_passed"] = sum(1 for g in res.get("subgoals") or [] if g.get("ok"))
                row["complete"] = bool(res.get("ok")) and row["subgoals_passed"] >= int(res.get("n_subgoals") or n_entries)
                if row["complete"] and scene_semantic_json and scene_recon_json:
                    row["goals_ok"], row["goal_issues"] = _goal_score(row, scene_semantic_json, scene_recon_json, spec_["out_dir"])
                ff = row.get("first_failure") or {}
                row["passed_target"] = row["complete"] or (ff.get("index") is not None and
                                                           (int(ff["index"]), int(ff.get("step") or 1)) > (int(index), int(stp or 1)))
                rows.append(row); level_rows.append(row)
            level["trials"] = len(level_rows)
            level["passed"] = sum(1 for r in level_rows if r["passed_target"])
            if any(r["complete"] for r in level_rows) or level["passed"]:
                break                                     # a pass: complete, or dig from it
            ranked = sorted(level_rows, key=score_of, reverse=True)
            points = space.refit([r["point"] for r in ranked[:max(1, int(elite))]], population, rng)
        return level_rows

    given_by_index = isinstance(knobs, dict) and any(str(k).isdigit() for k in knobs)

    def knob_map_for(doc, index, stp, failure, depth):
        ei = _target_entry(doc, index, stp)
        if given_by_index and (depth == 0 or str(index) in knobs):
            return {(ei if int(k) == int(index) else _target_entry(doc, int(k), None)): v for k, v in knobs.items()
                    if depth == 0 or int(k) == int(index)}
        if knobs and not given_by_index and depth == 0:
            return {ei: knobs}
        return {ei: auto_knobs(_entries(doc)[ei], failure)}

    def descend(doc, prefix, index, stp, failure, depth):
        level_rows = search_level(doc, prefix, index, stp, failure, knob_map_for(doc, index, stp, failure, depth), "own")
        if not any(r["passed_target"] for r in level_rows) and knobs and depth == 0 and len(rows) < int(max_evals) and time_left() > 0:
            # the caller's knobs did not carry it: the knobs proposed from the failure (for a grasp, also where the hand sits)
            ei = _target_entry(doc, index, stp)
            proposed = {ei: {k: v for k, v in auto_knobs(_entries(doc)[ei], failure).items()}}
            given = knob_map_for(doc, index, stp, failure, depth).get(ei, {})
            if set(proposed[ei]) - set(given):
                level_rows = level_rows + search_level(doc, prefix, index, stp, failure, proposed, "proposed")
        if not any(r["passed_target"] for r in level_rows):
            # its own parameters do not carry it: search what led into it (the carry and the grasp of the same object)
            up = upstream_knobs(doc, _target_entry(doc, index, stp), failure)
            if len(up) > 1 and len(rows) < int(max_evals) and time_left() > 0:
                level_rows = level_rows + search_level(doc, prefix, index, stp, failure, up, "upstream")
        passing = sorted([r for r in level_rows if r["passed_target"] and not r.get("error")], key=score_of, reverse=True)
        if any(r["complete"] for r in passing) or not dig or depth + 1 >= int(max_depth):
            return
        for r in passing[:2]:                             # the best passing variant, then one backtrack
            if len(rows) >= int(max_evals) or time_left() <= 0:
                return
            ff = r.get("first_failure") or {}
            before = len([x for x in rows if x["complete"]])
            descend(r["doc"], r["out_dir"], int(ff["index"]), ff.get("step"), ff, depth + 1)
            deeper = [x for x in rows if x.get("target") == int(ff["index"]) and x.get("step") == ff.get("step") and x["passed_target"]]
            if deeper or len([x for x in rows if x["complete"]]) > before:
                return

    descend(doc0, Path(prefix_dir).resolve() if prefix_dir else None, int(subgoal), step, (baseline or {}).get("first_failure"), 0)

    ranked = sorted(rows, key=score_of, reverse=True)
    best = ranked[0] if ranked else None
    improved = bool(best and (baseline is None or score_of(best) > score_of(baseline)))
    for r in rows:                                        # plans of trials that are not the best are removed from beside the plan
        if r is not best and plan_dir is not None:
            Path(r["plan"]).unlink(missing_ok=True)
    keep_plan = None
    if best is not None:
        keep_plan = out / "best_plan.json"
        keep_plan.write_text(json.dumps(best["doc"], indent=1))
        if plan_dir is not None:                          # the copy that resolves relative paths like the original
            keep_plan = Path(best["plan"])

    def brief(r):
        ff = r.get("first_failure") or {}
        return {"id": r["id"], "target": r.get("target"), "knobs": r.get("knobs"), "subgoals_passed": r.get("subgoals_passed"),
                "complete": r.get("complete"), "goals_ok": r.get("goals_ok"), "failed_at": ff.get("index"), "failed": ff.get("failed"),
                "shortfall": residual(ff) if ff else 0.0, "seconds": r.get("seconds"), "error": (r.get("error") or "")[:300] or None}
    report = {"baseline": brief(baseline) if baseline else None, "targets": targets, "trials": [brief(r) for r in rows],
              "best": brief(best) if best else None, "elapsed_s": round(time.time() - t_start, 1)}
    (out / "search_report.json").write_text(json.dumps(report, indent=1, default=float))
    result = {"ok": bool(best and best["complete"]), "evaluations": len(rows), "improved": improved,
              "elapsed_s": report["elapsed_s"], "baseline": report["baseline"], "targets": targets,
              "table": [brief(r) for r in ranked[:8]], "report": str(out / "search_report.json")}
    if best is not None:
        result["best"] = {**brief(best), "plan": str(keep_plan), "out_dir": best["out_dir"], "n_subgoals": best.get("n_subgoals"),
                          "goal_issues": best.get("goal_issues")}
        result["next_step"] = (f"execute_subgoal_plan(scene={str(scene)!r}, plan={str(keep_plan)!r}, out_dir={best['out_dir']!r}, "
                               f"resume=True{', ' + ', '.join(f'{k}={v!r}' for k, v in (exec_options or {}).items()) if exec_options else ''})"
                               + (" reuses this trial's segments and records the execution; then package_stage3_episode and finish."
                                  if best["complete"] else
                                  f" continues from the best trial (it passed {best['subgoals_passed']} subgoals; it fails at subgoal "
                                  f"{(best.get('first_failure') or {}).get('index')}: {(best.get('first_failure') or {}).get('failed')}); "
                                  f"search again from that folder with other knobs, or change that subgoal's plan."))
    return result


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--trial":
        _trial(json.loads(Path(sys.argv[2]).read_text()))
    else:
        print(json.dumps(search_subgoal_variants(**json.loads(sys.argv[1])), indent=1, default=str))
