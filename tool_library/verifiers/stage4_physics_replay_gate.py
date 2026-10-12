"""Does a randomized Stage 4 episode still do the task when it is replayed in contact physics?"""
from __future__ import annotations

TOOL = {
    "name": "stage4_physics_replay_gate",
    "title": "Stage 4 physics replay gate",
    "type": "verifier",
    "category": "Task & Rollout Gates",
    "summary": "Replay the original layout and one randomized layout of a Stage 4 task folder in contact physics and "
               "re-measure every Stage 3 subgoal expectation on both; passes when the original holds them all and the "
               "randomized episode holds every one the original holds.",
    "description": (
        "Stage 4's kinematic checks prove the robot can follow an adapted trajectory without new collisions; they do "
        "not prove the objects still do what the task needs (a brick pushed into a neighbour that moved stops after 3 "
        "of its 15 mm; a cube left where it was when its support moved falls). This gate builds Stage 3's execution "
        "model of the task scene (hand collision hulls paired with every object and the table), replays the original "
        "servo commands (the `reference` stored in trajectory.npz; without it the executed joint values are replayed "
        "as commands, which presses no harder than the fingers touched) segment by segment as Stage 3 executed its "
        "subgoals, and judges the Stage 3 manifest's expectations (`manifest` in stage4.json: rise, in hand, no slip, "
        "at target, released, at rest, supported on, untouched, penetration, servo tracking ...) with the library's "
        "outcome evaluation, plus stability and every randomized object ending within physics_end_tol_m of where the "
        "plan puts it. Then it draws layouts exactly as Stage 4's randomize step does (its adaptive sampler, the reach "
        "pre-screen, the warp, retiming of a layout rejected only on joint speed) until one passes the kinematic checks "
        "(or uses the `layout` given), adapts the commands with the hands (retimed with the warp), moves world-frame targets with their objects, "
        "replays it and judges the same checks (with layouts > 1, that many accepted layouts, passing when one holds). A check that fails on the original is an issue on its own (the "
        "Stage 3 episode does not hold up in physics); failures cannot be removed from the new layout's requirements. "
        "Instability of the original blocks acceptance. Cost: two replays of the episode (4 to 150 ms per "
        "frame on the two-hand bench, depending on how many contacts the scene has). DEFORMABLES (stage4.json "
        "`deformables`): the robot and rigid objects replay in MuJoCo as above, then the cords run in Isaac Sim driven by "
        "the replayed robot, moved by the layout's draw for the cord (the original layout reuses the Stage 3 Isaac "
        "recording when its replay is the recorded trajectory bit for bit); the Isaac episode must pass (no PhysX "
        "error, no missed grasp, no link through the table or faster than the limit) and the Stage 1 goals and the "
        "expectations that name a cord are judged on the Isaac mirror with the same code; each Isaac episode adds "
        "its own run time (minutes for a long episode)."
    ),
    "inputs": [
        {"name": "task_dir", "type": "str", "description": "Folder with stage4.json (from write_stage4_config)."},
        {"name": "draws", "type": "int", "description": "Layouts to try until one passes the kinematic checks (default 10)."},
        {"name": "seed", "type": "int | None", "description": "Seed of the draws (default: the config's seed)."},
        {"name": "layout", "type": "dict | None", "description": "A layout to replay instead of drawing one: {objects: {name: [dx, dy, dyaw_deg]}, robot: {}}."},
        {"name": "layouts", "type": "int", "description": "Kinematically accepted layouts to replay (default 1); the gate passes when at least one holds."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok, issues, warnings, original {checks, not_held, seconds}, layout, kinematic, physics {ok, failed, "
                                                          "subgoals, end_error_m, seconds} of the layout shown, layouts_replayed, layouts_held, draws_tried."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy", "Isaac Sim 5.0 for tasks with deformables"],
    "simulators": ["MuJoCo", "Isaac Sim"],
    "tags": ["stage3", "stage4", "physics", "replay", "randomization", "gate", "deformable"],
    "status": "tested",
    "entrypoint": "stage4_physics_replay_gate",
}

import time
from pathlib import Path


from tool_library.common import stage4 as S4

def stage4_physics_replay_gate(task_dir, draws=10, seed=None, layout=None, layouts=1):
    draws, layouts = int(draws), int(layouts)
    if layouts < 1:
        raise ValueError(f"layouts must be at least 1, got {layouts}")
    if draws < 1 and layout is None:
        raise ValueError(f"draws must be at least 1 (or give a layout), got {draws}")
    if layout is not None and not isinstance(layout, dict):
        raise ValueError("layout must be a dict {objects: {name: [dx, dy, dyaw_deg]}, robot: {...}}")
    task = Path(task_dir)
    t_all = time.time()
    try:
        cfg, ep = S4.load_task(task)
        phys = S4.stage4_module("physics").Replay(ep, cfg, log=lambda *_: None)
    except (AssertionError, ValueError, FileNotFoundError, KeyError, ImportError) as e:
        return {"ok": False, "issues": [f"Stage 4 refuses the task folder: {e}"]}
    issues = []
    base = phys.original()
    n_checks = len(base["checks"])
    original = {"checks": n_checks, "not_held": base.get("failed", base.get("not_judged", [])), "seconds": base["seconds"],
                "end_error_m": base["end_error_m"], "subgoals_failed": [r for r in base["subgoals"] if not r.get("ok")]}
    warnings = []
    failing = list(base.get("failed") or base.get("not_judged") or [])
    if base.get("ok") is not True and not failing:
        failing = ["original replay did not explicitly pass"]
    if failing:
        issues.append(f"the original layout fails {len(failing)} of {n_checks} checks when replayed in physics: "
                      f"{failing[:5]}; the Stage 3 episode does not hold up in contact physics there (re-execute "
                      "that subgoal, or give the task folder the servo commands: trajectory.npz `reference`)")
    if not phys.subgoals:
        issues.append("no subgoal expectations to replay against: stage4.json has no `manifest` (write_stage4_config copies "
                      "the Stage 3 trajectory_manifest.json); only stability and end poses were judged")
    if seed is not None:
        cfg["seed"] = int(seed)
    # the layouts are drawn and warped exactly as the randomize step does (adaptive sampler, reach pre-screen, retiming
    # of a layout rejected only on joint speed); the physics replay below is this gate's own
    rz = S4.stage4_module("randomize").Randomizer(ep, cfg, log=lambda *_: None, physics_replay=False, reexecute=False)
    chosen, kin, tried = [], None, 0
    if layout is not None:
        lay = {"objects": {k: [float(x) for x in v] for k, v in layout.get("objects", {}).items()},
               "robot": layout.get("robot") or {}}
        for name in ep.objects:
            lay["objects"].setdefault(name, [0.0, 0.0, 0.0])
        res, _pl = rz.warp(lay)
        kin, tried = res["report"], 1
        if res["ok"]:
            chosen.append((lay, res["Q"], res["overrides"], res["timing"]))
    else:
        for attempt in range(draws):
            res = rz.attempt(attempt)
            kin = res["report"]
            tried += 1
            if res["ok"]:
                chosen.append((res["layout"], res["Q"], res["overrides"], res["timing"]))
                if len(chosen) >= layouts:
                    break
    out = {"original": original, "draws_tried": tried, "mujoco_version": phys.mujoco_version}
    if phys.mujoco_version["stage3"] and phys.mujoco_version["stage3"] != phys.mujoco_version["replay"]:
        warnings.append(f"replayed with MuJoCo {phys.mujoco_version['replay']}, the episode was executed with "
                        f"{phys.mujoco_version['stage3']}: outcomes can differ for that reason alone")
    if not chosen:
        det = S4.rejection_detail(kin, ep) if kin is not None else {}
        issues.append(f"no layout passed Stage 4's kinematic checks in {tried} draws ({det}); run stage4_randomization_gate "
                      "to see why, then fix the ranges")
        out["kinematic"] = {k: kin.get(k) for k in ("ok", "failed")} if kin is not None else None
        return {"ok": False, "issues": issues, "warnings": warnings, **out, "seconds": round(time.time() - t_all, 1)}
    replays = []
    for lay, Q, ov, timing in chosen:
        prep, _ = phys.check(lay, Q, ov, timing=timing)
        rec = {k: prep.get(k) for k in ("ok", "failed", "not_judged", "end_error_m", "unstable", "stopped_at_frame",
                                        "max_hand_pen_mm", "max_object_pen_mm", "seconds", "reference_ik_m")}
        rec["layout"], rec["subgoals"] = lay["objects"], prep["subgoals"]
        rec["retimed"] = timing is not None
        replays.append(rec)
    good = [r for r in replays if r["ok"]]
    pick = good[0] if good else replays[0]
    out["layout"] = pick["layout"]
    out["kinematic"] = {"ok": True, "failed": []}
    out["physics"] = pick
    out["layouts_replayed"], out["layouts_held"] = len(replays), len(good)
    if not good:
        bad = [r for r in pick["subgoals"] if not r.get("ok")]
        where = f" (first: {bad[0]['subgoal']} frames {bad[0].get('frames')} failed {bad[0]['failed']} with {bad[0].get('measured')})" if bad else ""
        issues.append(f"{'the randomized layout' if len(replays) == 1 else f'none of {len(replays)} randomized layouts, e.g.'} "
                      f"{pick['layout']} fails in the physics replay: {pick['failed'][:6]}{where}; "
                      "narrow the randomize ranges of the objects involved, move objects that interact together "
                      "(attached_to), or declare what a placed object rests on (rests_on / inside)")
    return {"ok": not issues, "issues": issues, "warnings": warnings, **out, "seconds": round(time.time() - t_all, 1)}
