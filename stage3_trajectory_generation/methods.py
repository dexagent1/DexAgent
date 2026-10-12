"""Stage 3's method modules: when the agent's search ends without a verified trajectory, non-agentic methods from the
library continue from what it left, in order, until one passes finish's verifiers.

    stage3:
      methods: [physics_informed_retargeting, sampling_mpc_refinement, reinforcement_learning_tracking]   # [] = off
      methods_max_minutes: 180          # all of them together; each gets what is left as its max_minutes / timeout_s
      method_options: {sampling_mpc_refinement: {max_minutes: 30}, reinforcement_learning_tracking: {max_minutes: 60}}

Each method is a library skill with one contract: (workspace, scene_semantic_json, scene_recon_json, reference=None,
out_dir, ...) -> {ok, reference (a tracking reference, common/tracking.py), plan (track_reference entries for
execute_subgoal_plan), scene}. The harness calls them through the stage's own dispatch and records them in the trace and
in methods/methods_report.json, without printing them; then, for a method that returned a plan, it calls
execute_subgoal_plan (the trusted execution the verifiers require), package_stage3_episode (the deliverable, Stage 4's
task folder included) and finish's verifiers. The first method whose episode passes is published like any agent result;
the next method starts from the best reference so far (the previous method's output, else the agent's best attempt).
No model call is made and no gate is changed.

Methods run only when the agent's search used its whole budget: its rounds (stage3.max_rounds), the run's money cap
(agent.budget_usd) or the stage's wall-clock budget. A stalled search, a report_blocked, a model that stopped submitting
actions, an administrative stop or a backend error ends the stage without them.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

DEFAULT_METHODS = ("physics_informed_retargeting", "sampling_mpc_refinement", "reinforcement_learning_tracking")
BUDGET_STOPS = ("budget_cap", "wall_time_budget")   # the run's money cap, the stage's wall clock
# the files a Stage 3 deliverable consists of (stage3_trajectory_generation/README.md); the agent's own are kept aside
DELIVERABLE = ("trajectory.npz", "phases.json", "sim_video.mp4", "trajectory_manifest.json", "stage4_task", "scene_executed.xml")
# options the harness passes to every method (the configuration's compute settings)
COMPUTE_ARGS = {"workers": ("compute", "workers")}
DEFAULT_MAX_MINUTES = 180.0                    # stage3.methods_max_minutes when unset: all modules together


def configured_methods(cfg: dict) -> list[str]:
    """stage3.methods: the method skills to run in order (default the three; [] or 'none' turns them off)."""
    v = ((cfg or {}).get("stage3") or {}).get("methods", list(DEFAULT_METHODS))
    if v in (None, "", False, "none", "off"):
        return []
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, (list, tuple)) or not all(isinstance(x, str) and x for x in v):
        raise ValueError(f"stage3.methods must be a list of library skill names, got {v!r}")
    return list(v)


def method_options(cfg: dict, name: str) -> dict:
    return dict((((cfg or {}).get("stage3") or {}).get("method_options") or {}).get(name) or {})


def should_run(out: dict) -> str | None:
    """Why the methods run after this agent result (its budget is used up), or None when they do not."""
    if out.get("finished") or out.get("upstream"):
        return None
    if out.get("stop_reason") in BUDGET_STOPS:
        return out["stop_reason"]
    if not out.get("stopped") and str(out.get("failure") or "").startswith("budget exhausted"):
        return "max_rounds"
    return None


def _keep_agent_attempt(ws: Path) -> list:
    """The agent's own deliverable files (unverified) to agent_attempt/, so a method's package starts clean."""
    moved = []
    keep = ws / "agent_attempt"
    for name in DELIVERABLE:
        src = ws / name
        if src.exists() and not src.is_symlink():
            keep.mkdir(exist_ok=True)
            dst = keep / name
            if dst.exists():
                shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
            shutil.move(str(src), str(dst)); moved.append(name)
    return moved


def _call(runner, name, args, rounds):
    """One skill call through the runner's own dispatch (trusted executor receipts), recorded in the trace."""
    blocks, raw = runner.run_tool(name, args)
    runner._trace({"event": "tool", "round": rounds, "phase": "method_modules", "name": name, "args": args, "result": raw})
    return raw.get("result") if raw.get("ok") and isinstance(raw.get("result"), dict) else None, raw


def run_method_modules(runner, out: dict, cfg: dict, semantic, recon, video=None) -> dict | None:
    """Run the configured method modules after an unfinished agent search (see the module docstring). Returns the
    record {reason, methods [{method, ok, ...}], ok, method (the one that passed), verdict} or None when none run."""
    names = configured_methods(cfg)
    reason = should_run(out)
    if not names or reason is None:
        return None
    ws = Path(runner.ws)
    rounds = int(out.get("rounds") or 0)
    s3 = (cfg.get("stage3") or {})
    record = {"reason": reason, "methods": [], "ok": False, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "agent_attempt_moved": _keep_agent_attempt(ws)}
    runner._trace({"event": "method_modules_start", "round": rounds, "reason": reason, "methods": names})
    compute = {k: (cfg.get(b) or {}).get(key) for k, (b, key) in COMPUTE_ARGS.items() if (cfg.get(b) or {}).get(key)}
    cap_min = float(s3.get("methods_max_minutes") or DEFAULT_MAX_MINUTES)
    t_start = time.time()
    best_ref = None
    from agent.library import find_tool
    for i, name in enumerate(names, 1):
        t0 = time.time()
        row = {"method": name, "ok": False}
        left = cap_min - (t0 - t_start) / 60.0
        if left <= 1.0:
            row["error"] = f"stage3.methods_max_minutes ({cap_min:g} min) spent before this method"
            record["methods"].append(row); continue
        tool = find_tool(runner.index, name)
        if tool is None:
            row["error"] = f"no library skill named {name!r}"
            record["methods"].append(row); continue
        declared = {n.strip() for inp in tool.get("inputs") or [] for n in str(inp.get("name", "")).split(",")}
        mdir = ws / "methods" / f"{i:02d}_{name}"
        args = {"workspace": str(ws), "scene_semantic_json": str(semantic), "scene_recon_json": str(recon), "out_dir": str(mdir),
                **({"reference": best_ref} if best_ref else {}),
                **{k: v for k, v in compute.items() if k in declared}, **method_options(cfg, name)}
        # what is left of the modules' time, as the method's own limit
        if "max_minutes" in declared:
            args["max_minutes"] = min(float(args.get("max_minutes") or left), left)
        if "timeout_s" in declared:
            args["timeout_s"] = int(min(float(args.get("timeout_s") or left * 60), left * 60))
        r, raw = _call(runner, name, args, rounds)
        if not r or not r.get("ok") or not r.get("plan"):
            row.update(error=(r or {}).get("error") or raw.get("error") or "the method returned no plan",
                       unavailable=bool((r or {}).get("unavailable")), seconds=round(time.time() - t0, 1))
            record["methods"].append(row); continue
        best_ref = r.get("reference") or best_ref
        row.update(reference=r.get("reference"), plan=r.get("plan"), coverage=r.get("coverage"), improved=r.get("improved"))
        ex_args = {"scene": r["scene"], "plan": r["plan"], "out_dir": str(mdir / "execution"), "fps": float(r.get("fps", 30.0))}
        ex, raw = _call(runner, "execute_subgoal_plan", ex_args, rounds)
        row["execution"] = {"ok": bool(ex and ex.get("ok")), "subgoals": [{k: g.get(k) for k in ("index", "ok", "failed")} for g in (ex or {}).get("subgoals", [])],
                            "report": ((ex or {}).get("files") or {}).get("execution_report"), "error": None if ex else raw.get("error")}
        if not (ex and ex.get("ok")):
            row["seconds"] = round(time.time() - t0, 1)
            record["methods"].append(row); continue
        pk_args = {"execution_report": ex["files"]["execution_report"], "plan": r["plan"], "scene_semantic_json": str(semantic),
                   "scene_recon_json": str(recon), "workspace": str(ws),
                   "num_samples": int((cfg.get("stage4") or {}).get("num_samples", 3))}
        if video:
            pk_args["human_video"] = str(video)
        pk, raw = _call(runner, "package_stage3_episode", pk_args, rounds)
        row["package"] = {"ok": bool(pk and pk.get("ok")), "issues": ((pk or {}).get("issues") or [raw.get("error")])[:8]}
        if not (pk and pk.get("ok")):
            row["seconds"] = round(time.time() - t0, 1)
            record["methods"].append(row); continue
        verdict = runner._verify()                    # finish: every verifier, then publication
        runner._trace({"event": "verify", "round": rounds, "phase": "method_modules", "method": name, "verdict": verdict})
        row["verdict"] = {"ok": bool(verdict.get("ok")), "issues": (verdict.get("issues") or [])[:12]}
        row["ok"] = bool(verdict.get("ok"))
        row["seconds"] = round(time.time() - t0, 1)
        record["methods"].append(row)
        if row["ok"]:
            record.update(ok=True, method=name, verdict=verdict)
            break
    record["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (ws / "methods").mkdir(exist_ok=True)
    (ws / "methods" / "methods_report.json").write_text(json.dumps(record, indent=1, default=str))
    runner._trace({"event": "method_modules_end", "round": rounds, "ok": record["ok"], "method": record.get("method")})
    return record
