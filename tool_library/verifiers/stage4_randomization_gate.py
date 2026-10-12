"""Does Stage 4 produce randomized episodes from this task folder?"""
from __future__ import annotations

TOOL = {
    "name": "stage4_randomization_gate",
    "title": "Stage 4 randomization gate",
    "type": "verifier",
    "category": "Task & Rollout Gates",
    "summary": "Run N randomize draws of a Stage 4 task folder with Stage 4's own sampler, trajectory adaptation and "
               "feasibility checks, and pass when enough of them would be kept (by default two thirds: 2 of 3, so a Stage 3 "
               "episode leaves Stage 4 a margin, not one lucky layout); reports why the others were rejected.",
    "description": (
        "A Stage 3 episode can satisfy every Stage 3 check and still yield nothing in Stage 4: the randomize step "
        "discards every layout whose adapted trajectory fails reach, joint speed, object continuity or the "
        "no-new-penetration check (an EgoDex lego stack passed Stage 3 and Stage 4 kept 0 of 60 layouts, because "
        "stacked bricks were released relative to the wrong brick). This gate loads the task folder with Stage 4's "
        "config loader, builds its Episode (the anchor schedule, carriers, releases, rests_on / inside relations and "
        "their cycle check run here with Stage 4's messages) and draws `draws` layouts exactly as the randomize step "
        "does (the same seeds and adaptive sampler, the reach pre-screen of the anchored keyframes, the warp, retiming "
        "of a layout rejected only on joint speed, the same checks), optionally with the physics replay of every "
        "kinematically accepted layout (physics_replay=true; it costs one contact-physics replay of the episode per "
        "accepted layout plus one of the original); reexecute=true asks for the re-execution of the handed-off Stage 3 "
        "plan on layouts the warp rejects, which the rigid runtime does not offer (it stays off). Nothing is written. It passes "
        "when accepted >= min_accepted, by default ceil(min_fraction x draws) with min_fraction 2/3: one layout in three "
        "passing is luck, not an episode Stage 4 can randomize. The report counts the rejections per check, gives the "
        "first concrete example of "
        "each (the colliding pair and frame, the jumping object, the arm that cannot reach) and says what to change "
        "in the config (narrower ranges, attached_to for objects that touch, rests_on / inside for an object put on "
        "another one). Assumes the folder was written by write_stage4_config (or by hand in the same format). A cord "
        "simulated in Isaac Sim (stage4.json `deformables`) is drawn like an object by its name and the hands anchored "
        "to it move with it; with physics_replay the cords of every kinematically accepted layout also run in Isaac Sim "
        "and the goals are judged on its mirror (stage4_physics_replay_gate)."
    ),
    "inputs": [
        {"name": "task_dir", "type": "str", "description": "Folder with stage4.json and the files it names."},
        {"name": "draws", "type": "int", "description": "Layouts to draw (default 10)."},
        {"name": "min_accepted", "type": "int | None", "description": "Kept layouts needed to pass (default: ceil(min_fraction x draws), 2 of 3)."},
        {"name": "min_fraction", "type": "float", "description": "Share of the draws that must be kept when min_accepted is not given (default 2/3)."},
        {"name": "seed", "type": "int | None", "description": "Seed of the draws (default: the config's seed, as the randomize step uses)."},
        {"name": "physics_replay", "type": "bool", "description": "Also replay every kinematically accepted layout in contact physics (default false)."},
        {"name": "reexecute", "type": "bool", "description": "Ask for the re-execution of the handed-off Stage 3 plan (plan.json) on layouts the warp rejects (default false; unavailable on the rigid runtime, where it stays off)."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok, issues, accepted, draws, min_accepted, rate, rejections {check: count}, examples {check: detail}, "
                                                          "methods {warp | warp+retime | reexecuted: count}, rows [per draw: ok, failed, layout, method, detail], "
                                                          "seconds_per_draw, relations, warnings, reexecution (whether a plan was handed off)."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo", "Isaac Sim"],
    "tags": ["stage3", "stage4", "handoff", "randomization", "gate", "deformable"],
    "status": "tested",
    "entrypoint": "stage4_randomization_gate",
}

from pathlib import Path

from tool_library.common import stage4 as S4


MIN_FRACTION = 2.0 / 3.0       # share of the draws a Stage 3 episode must leave randomizable: 2 of 3


def stage4_randomization_gate(task_dir, draws=10, min_accepted=None, seed=None, physics_replay=False, min_fraction=MIN_FRACTION,
                              reexecute=False):
    draws = int(draws)
    if draws < 1:
        raise ValueError(f"draws must be at least 1, got {draws}")
    if not 0.0 < float(min_fraction) <= 1.0:
        raise ValueError(f"min_fraction must be in (0, 1], got {min_fraction}")
    if min_accepted is None:
        import math
        min_accepted = max(1, min(draws, math.ceil(float(min_fraction) * draws - 1e-9)))
    min_accepted = int(min_accepted)
    if not 0 <= min_accepted <= draws:
        raise ValueError(f"min_accepted must be between 0 and draws ({draws}), got {min_accepted}")
    task = Path(task_dir)
    if not (task / "stage4.json").exists():
        return {"ok": False, "issues": [f"{task / 'stage4.json'} is missing: write it with write_stage4_config"],
                "accepted": 0, "draws": 0}
    try:
        res = S4.run_draws(task, draws=draws, seed=seed, physics_replay=bool(physics_replay), reexecute=bool(reexecute))
    except (AssertionError, ValueError, FileNotFoundError, KeyError) as e:
        return {"ok": False, "issues": [f"Stage 4 refuses the task folder before drawing: {e}"], "accepted": 0, "draws": 0}
    issues = []
    if res["accepted"] < min_accepted:
        order = sorted(res["rejections"].items(), key=lambda kv: -kv[1])
        head = f"only {res['accepted']} of {res['draws']} layouts pass Stage 4's checks (need {min_accepted})"
        if order:
            head += "; rejections: " + ", ".join(f"{k} {v}x" for k, v in order)
        issues.append(head)
        for k, _ in order:
            issues.append(f"{k}: {res['examples'].get(k, '')} -> {S4.HINTS.get(k, '')}")
    base = res.get("physics_original")
    if base is not None and (base.get("ok") is not True or base.get("failed") or base.get("not_judged")):
        issues.append(f"the original layout fails in physics replay: {(base.get('failed') or base.get('not_judged') or ['missing explicit pass'])[:4]}; fix the Stage 3 episode")
    out = {k: res[k] for k in ("accepted", "draws", "rejections", "examples", "methods", "seconds_per_draw", "physics_seconds", "relations",
                               "warnings", "reexecution")}
    out["min_accepted"] = min_accepted
    out["rate"] = round(res["accepted"] / max(1, res["draws"]), 3)
    out["rows"] = [{k: r.get(k) for k in ("attempt", "ok", "failed", "layout", "method", "detail")} for r in res["rows"]]
    if base is not None:
        out["physics_original"] = {k: base.get(k) for k in ("not_judged", "seconds", "end_error_m", "unstable")}
    ok = res["accepted"] >= min_accepted and not issues
    return {"ok": bool(ok), "issues": issues, **out}
