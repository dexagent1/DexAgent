"""DexAgent end to end: a video and a task prompt in, randomized robot videos out, with every stage's progress on the
terminal (the round out of its budget, the tools called and what they returned, what the model is thinking, the
verifiers' verdicts, the tools it created) and a closing table.

    python run_dexagent.py dexagent.yaml                      # stages 1 -> 4
    python run_dexagent.py dexagent.yaml --from stage3        # resume from a stage (earlier deliverables must exist)
    python run_dexagent.py dexagent.yaml --to stage3          # stop after a stage
    python run_dexagent.py dexagent.yaml --from stage2 --to stage3 --resume-stage3-attempt runs/demo/stage3/.candidates/<id>
    python run_dexagent.py dexagent.yaml --name demo --video v.mp4 --prompt "close the box"
    python run_dexagent.py dexagent.yaml --yes                # do not stop between stages to ask

After every stage the run reports what the stage produced (and where to look at it: Stage 1's subgoals, Stage 2's
reconstructed scene and its frame-0 comparison, Stage 3's robot video, Stage 4's outputs), the rounds, the tools it used
and built, and the money spent; when stages remain it asks whether to continue, so a stage's result can be checked
before the next one starts. Answering no stops the run there; it continues later with --from <next stage>. Without a
terminal to ask on (stdin is not a TTY) the run stops after each stage unless --yes is given. A stage that asks to
re-open an earlier one (cross-stage repair, below) is also confirmed first.

Each stage also runs on its own (stage1_semantic_understanding/run_stage1.py, ...). Progress of the running stage is in
runs/<name>/<stage>/progress.json; the run's summary in runs/<name>/run_summary.json.

Cross-stage repair: a stage that ends with report_blocked(upstream='stage1' | 'stage2', evidence) re-opens that earlier stage
with the reason in its first message, and the stages after it run again (repair.max_loops_per_stage re-openings per stage,
default 1). Every loop is recorded in run_summary.json and, when the run fails, in runs/<name>/failure_report.json.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from agent import repair  # noqa: E402
from agent.budget import BudgetError  # noqa: E402
from agent.config import load_config, run_dir, semantics_file  # noqa: E402
from agent.console import STAGES, Console, summary_table  # noqa: E402

ORDER = ["stage1", "stage2", "stage3", "stage4"]


def run_agent_stage(stage: str, cfg_path: str, extra: list[str], cfg: dict) -> dict:
    mod = {"stage1": "stage1_semantic_understanding.run_stage1", "stage2": "stage2_scene_reconstruction.run_stage2",
           "stage3": "stage3_trajectory_generation.run_stage3"}[stage]
    import importlib
    m = importlib.import_module(mod)
    # the stage's main prints the console itself; we read its result back from the trace
    t0 = time.time()
    ws = run_dir(cfg, stage)
    try:
        rc = m.main([cfg_path] + extra)
    except BudgetError as exc:                     # a model API run without a spending cap is refused before any call
        print(f"  {exc}", flush=True)
        return {"stage": stage, "finished": False, "workspace": str(ws), "failure": str(exc)}
    out = {"stage": stage, "finished": rc == 0, "workspace": str(ws)}
    rj = ws / "run.json"
    if rj.exists() and rj.stat().st_mtime >= t0 - 1:
        try:
            rec = json.loads(rj.read_text())
            out["reused_example"] = rec.get("reused_example")
            for k in ("failure", "upstream", "methods", "agent_search", "stopped", "stop_reason", "search_checkpoint", "spent_usd",
                      "run_spent_usd", "budget_usd", "tools_used"):
                if rec.get(k):
                    out[k] = rec[k]
        except Exception:
            pass
    prog = ws / "progress.json"
    if prog.exists():
        p = json.loads(prog.read_text())
        out.update({"rounds": p.get("round"), "tool_calls": p.get("tool_calls"), "new_tools": p.get("new_tools"), "elapsed_s": p.get("elapsed_s")})
    return out


def apply_stage4_settings(task: Path, s4: dict) -> dict:
    """The `stage4` block of dexagent.yaml (samples, seed, views, render quality) written into the task's stage4.json."""
    c = json.loads(task.read_text())
    for k in ("num_samples", "seed"):
        if s4.get(k) is not None:
            c[k] = s4[k]
    if s4.get("views") is not None:
        views = (c.get("cameras") or {}).get("views") or {}
        c.setdefault("cameras", {})["views"] = {k: v for k, v in views.items() if k in set(s4["views"])}
    for k, v in (s4.get("render") or {}).items():
        c.setdefault("render", {})[k] = v
    task.write_text(json.dumps(c, indent=1))
    return c


def run_stage4(cfg: dict) -> dict:
    """Stage 4 is not agentic: a fixed five-step pipeline in its own environment, driven by the task folder Stage 3 wrote
    (runs/<name>/stage3/stage4_task/, the copy of the accepted delivery's; its outputs go to stage4_task/outputs/)."""
    ws3 = run_dir(cfg, "stage3"); ws4 = run_dir(cfg, "stage4")
    try:
        from agent.artifacts import accepted_delivery
        accepted = accepted_delivery(ws3)
    except (OSError, ValueError, KeyError) as exc:
        return {"stage": "stage4", "finished": False, "workspace": str(ws4),
                "failure": f"no intact accepted Stage 3 delivery: {exc}"}
    # Stage 4 writes settings and its outputs (stage4_task/outputs/) only to the copy of the sealed input at the top of
    # the Stage 3 folder; a copy that is missing or no longer the accepted one is refreshed (the old one is kept aside)
    from agent.artifacts import expose_accepted
    expose_accepted(ws3, accepted)                 # a fresh copy for this run; an earlier one (and its outputs) is kept aside
    task_dir = ws3 / "stage4_task"
    task = task_dir / "stage4.json"
    con = Console("stage4", str(cfg["run"]["name"]), ws4, max_rounds=0)
    con.banner(note=f"no model rounds: randomized episodes from {task_dir}")
    if not task.exists():
        if (accepted / "twin_episode.json").exists():   # a reused Isaac Sim twin episode: no rigid runtime recording to randomize
            msg = ("Stage 3 published a recorded twin episode (twin_episode.json), not a rigid runtime recording of the "
                   "setup: Stage 4 does not randomize it")
            con._out(f"  {msg}", "yellow")
            return {"stage": "stage4", "finished": False, "skipped": True, "workspace": str(ws4), "failure": msg}
        con._out(f"  Stage 3 left no stage4_task/stage4.json in {ws3}", "red")
        return {"stage": "stage4", "finished": False, "workspace": str(ws4)}
    apply_stage4_settings(task, cfg.get("stage4") or {})
    python = (cfg.get("environments") or {}).get("stage4") or sys.executable
    cmd = [python, str(REPO / "stage4_robot_data_generation" / "run_stage4.py"), str(task)]
    t0 = time.time()
    con._out("  ▶ " + " ".join(cmd), "bold")
    env = dict(os.environ); env.setdefault("MUJOCO_GL", "egl"); env.setdefault("PYTHONUNBUFFERED", "1")   # progress lines as they happen
    models = cfg.get("models") or {}
    if models.get("void_dir"):                     # the VOID checkout the plate step runs (default: third_party/void-model)
        env["VOID_DIR"] = str(models["void_dir"])
    if models.get("hf_home"):                      # where SAM 3, VOID and CogVideoX weights are cached (as for the tools)
        env.setdefault("HF_HOME", str(models["hf_home"]))
    if (cfg.get("environments") or {}).get("isaac"):   # deformables are replayed in Isaac Sim
        env.setdefault("DEXAGENT_ISAAC_PYTHON", str(cfg["environments"]["isaac"]))
    proc = subprocess.Popen(cmd, cwd=str(REPO / "stage4_robot_data_generation"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    for line in proc.stdout:
        line = line.rstrip()
        if line.startswith("[stage4]") or line.startswith("[render]") or "finished" in line or "error" in line.lower():
            con._out("    " + line[:200], "dim")
    rc = proc.wait()
    out = {"stage": "stage4", "finished": rc == 0, "workspace": str(ws4), "elapsed_s": round(time.time() - t0, 1), "rounds": "-", "tool_calls": 5}
    if rc != 0:
        out["failure"] = f"run_stage4.py exited with code {rc} (its log lines above; outputs in {task_dir / 'outputs'})"
    con.done({"finished": rc == 0, "rounds": "-"})
    return out


def stage_outputs(cfg: dict, stage: str) -> list[str]:
    """What a finished stage produced, with the files to open to check it."""
    ws = run_dir(cfg, stage)
    lines = []
    try:
        if stage == "stage1":
            sem = semantics_file(ws)
            if sem.exists():
                d = json.loads(sem.read_text())
                lines.append(f"subgoals ({len(d.get('subgoals') or [])}): " + "; ".join(
                    f"{g.get('index')}. {g.get('action') or g.get('goal') or g.get('motion')}" for g in (d.get("subgoals") or [])[:8])[:600])
                lines.append(f"scene semantics: {sem}")
        elif stage == "stage2":
            rec = ws / "scene_recon.json"
            if rec.exists():
                d = json.loads(rec.read_text())
                lines.append("objects: " + ", ".join(str(o.get("id")) for o in d.get("objects") or []))
                lines.append(f"scene: {ws / d.get('scene_xml', 'scene.xml')}")
            for name in ("frame0_compare.png", "scene_recon.json"):
                if (ws / name).exists():
                    lines.append(f"check: {ws / name}")
        elif stage == "stage3":
            for name in ("sim_video.mp4", "stage4_task/stage4.json", "trajectory_manifest.json"):
                if (ws / name).exists():
                    lines.append(f"{'check' if name.endswith('.mp4') else 'output'}: {ws / name}")
        elif stage == "stage4":
            outs = sorted((run_dir(cfg, "stage3") / "stage4_task" / "outputs").glob("*"))
            if outs:
                lines.append(f"outputs ({len(outs)}): {outs[0].parent}")
    except (OSError, ValueError) as exc:
        lines.append(f"(could not read the stage's outputs: {exc})")
    return lines


def report_stage(r: dict, cfg: dict):
    """The block printed when a stage ends: verdict, effort, tools used and built, money, what to check."""
    num, name = STAGES[r["stage"]]
    verdict = "PASS" if r.get("finished") else "FAIL"
    print(f"\n══ Stage {num} · {name}: {verdict} ══", flush=True)
    bits = []
    if r.get("reused_example"):
        bits.append(f"reused the verified example {r['reused_example']}")
    if r.get("rounds") not in (None, "-"):
        bits.append(f"{r['rounds']} rounds")
    if r.get("tool_calls"):
        bits.append(f"{r['tool_calls']} tool calls")
    if r.get("elapsed_s"):
        bits.append(f"{float(r['elapsed_s']) / 60:.1f} min")
    if r.get("spent_usd") is not None:
        bits.append(f"${float(r['spent_usd']):.2f} (run: ${float(r.get('run_spent_usd') or 0):.2f} of ${float(r.get('budget_usd') or 0):g})")
    if bits:
        print("   " + " · ".join(bits))
    used = r.get("tools_used") or {}
    if used:
        top = sorted(used.items(), key=lambda kv: -kv[1])
        print("   tools used: " + ", ".join(f"{k}×{v}" for k, v in top[:12]) + (f", ... ({len(top)} distinct)" if len(top) > 12 else ""))
    if r.get("new_tools"):
        print("   tools built: " + ", ".join(f"{t.get('type', 'tool')} {t.get('name')}" for t in r["new_tools"]))
    if not r.get("finished") and r.get("failure"):
        print(f"   why: {str(r['failure'])[:400]}")
    for line in stage_outputs(cfg, r["stage"]) if r.get("finished") else []:
        print("   " + line)
    sys.stdout.flush()


def continue_command(a, nxt: str) -> str:
    """The command that continues this run from stage `nxt`, with every flag this run was given."""
    import shlex
    parts = ["python", "run_dexagent.py", a.config, "--from", nxt]
    if a.stop != "stage4":
        parts += ["--to", a.stop]
    for flag, value in (("--name", a.name), ("--video", a.video), ("--prompt", a.prompt), ("--max-rounds", a.max_rounds),
                        ("--llm", a.llm if a.llm != "config" else None)):
        if value:
            parts += [flag, str(value)]
    return " ".join(shlex.quote(p) for p in parts)


def ask(question: str, auto: bool) -> bool:
    """Yes when --yes was given, else the user's answer on the terminal; no answer possible (no TTY) is no."""
    if auto:
        return True
    if not sys.stdin.isatty():
        print(f"{question} -- no terminal to answer on: stopping here (add --yes to run every stage without asking)", flush=True)
        return False
    while True:
        try:
            ans = input(f"{question} [y/n] ").strip().lower()
        except EOFError:
            return False
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no", "q", "quit", "stop"):
            return False


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="DexAgent end to end")
    ap.add_argument("config", nargs="?", default=os.environ.get("DEXAGENT_CONFIG", str(REPO / "dexagent.yaml")))
    ap.add_argument("--from", dest="start", default="stage1", choices=ORDER)
    ap.add_argument("--to", dest="stop", default="stage4", choices=ORDER)
    ap.add_argument("--name"); ap.add_argument("--video"); ap.add_argument("--prompt")
    ap.add_argument("--max-rounds", type=int, help="round budget for every agent stage (default: each stage's own)")
    ap.add_argument("--resume-stage3-attempt", type=Path,
                    help="forward this unfinished candidate to Stage 3 --resume-attempt, including after upstream repair; all current input and acceptance checks still apply")
    ap.add_argument("--llm", default="config", help="config (the YAML's agent block) or mock:<turns.json> for a scripted dry run")
    ap.add_argument("--yes", "-y", action="store_true", help="run every stage without asking between stages")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.name:
        cfg["run"]["name"] = a.name
    if a.video:
        cfg["inputs"]["video"] = str(Path(a.video).resolve())
    name = str(cfg["run"]["name"])
    if ORDER.index(a.start) > ORDER.index(a.stop):
        ap.error(f"--from {a.start} comes after --to {a.stop}")
    stages = ORDER[ORDER.index(a.start): ORDER.index(a.stop) + 1]
    print(f"DexAgent run '{name}': {' -> '.join(STAGES[s][1] for s in stages)}", flush=True)
    results, ok = [], True
    max_loops = int((cfg.get("repair") or {}).get("max_loops_per_stage", 1))
    loops, reopened = [], {}
    queue = list(stages)
    stopped_after, unanswered = None, False
    while queue:
        stage = queue.pop(0)
        if stage == "stage4":
            r = run_stage4(cfg)
        else:
            extra = []
            if a.name:
                extra += ["--name", a.name]
            if a.video and stage in ("stage1", "stage2", "stage3"):
                extra += ["--video", a.video]
            if a.prompt and stage == "stage1":
                extra += ["--prompt", a.prompt]
            if a.max_rounds:
                extra += ["--max-rounds", str(a.max_rounds)]
            if a.llm != "config":
                extra += ["--llm", a.llm]
            if stage == "stage3" and a.resume_stage3_attempt is not None:
                extra += ["--resume-attempt", str(a.resume_stage3_attempt)]
            r = run_agent_stage(stage, a.config, extra, cfg)
        if loops:
            r["after_repair_loop"] = len(loops)
        results.append(r)
        report_stage(r, cfg)
        if r.get("finished"):
            for lp in loops:
                if lp["requested_by"] == stage and lp.get("resolved") is None:
                    lp["resolved"] = True              # the stage that asked for the repair passes now
            if queue:
                nxt = queue[0]
                if not ask(f"Stage {STAGES[stage][0]} is done. Continue with Stage {STAGES[nxt][0]} ({STAGES[nxt][1]})?", a.yes):
                    stopped_after = stage
                    unanswered = not sys.stdin.isatty()
                    print(f"stopped after {STAGES[stage][1]}; to continue: {continue_command(a, nxt)}", flush=True)
                    for later in queue:
                        results.append({"stage": later, "skipped": True, "stopped_by_user": True, "finished": False, "workspace": ""})
                    break
            continue
        up = (r.get("upstream") or {}).get("stage")
        if up and repair.valid_upstream(stage, up) is None:
            loop = {"loop": len(loops) + 1, "requested_by": stage, "stage": up, "reason": str(r.get("failure") or "")[:2000],
                    "evidence": (r.get("upstream") or {}).get("evidence"), "resolved": None}
            if reopened.get(up, 0) >= max_loops:
                loop["refused"] = f"{up} was already re-opened {reopened[up]} time(s) in this run (repair.max_loops_per_stage = {max_loops})"
                loop["resolved"] = False
                loops.append(loop)
                print(f"repair refused: {loop['refused']}", flush=True)
            elif not ask(f"{STAGES[stage][1]} found {STAGES[up][1]} wrong ({str(loop['evidence'] or loop['reason'])[:300]}). "
                         f"Re-open Stage {STAGES[up][0]} and run the stages after it again?", a.yes):
                loop["refused"] = ("the user stopped the run instead of re-opening the stage" if sys.stdin.isatty()
                                   else "no terminal to confirm the repair on (run with --yes to re-open stages automatically)")
                loop["resolved"] = False
                loops.append(loop)
            else:
                reopened[up] = reopened.get(up, 0) + 1
                loops.append(loop)
                k = loop["loop"]
                loop["archived_failure"] = repair.archive_failure(run_dir(cfg, stage), k)
                repair.write_request(run_dir(cfg, up), k, stage, up, loop["reason"], loop["evidence"] or "")
                rerun = ORDER[ORDER.index(up): ORDER.index(a.stop) + 1]
                for later in rerun[1:]:
                    if later != "stage4":
                        repair.write_revised(run_dir(cfg, later), k, up, stage, loop["reason"])
                loop["reruns"] = rerun
                print(f"repair loop {k}: {stage} found {up} wrong; re-opening {up}, then {' -> '.join(rerun[1:]) or 'nothing else'}", flush=True)
                queue = rerun
                continue
        ok = False
        for lp in loops:
            if lp.get("resolved") is None:
                lp["resolved"] = False
        for later in [s_ for s_ in ORDER[ORDER.index(stage) + 1: ORDER.index(a.stop) + 1]]:
            results.append({"stage": later, "skipped": True, "finished": False, "workspace": ""})
        break
    summary_table(results)
    out_dir = Path(cfg["run"]["output_dir"]) / name
    # asset reuse: a run whose stages all passed joins examples/asset_reuse (examples.save_assets in the config)
    ran3 = any(r.get("stage") == "stage3" and r.get("finished") for r in results)
    if ok and ran3 and not any(r.get("reused_example") for r in results):
        try:
            from agent.reuse import save_run
            saved = save_run(cfg, out_dir, name=name, video=cfg["inputs"].get("video"))
            if saved:
                print(f"saved as a reusable example: {saved['example']}")
        except Exception as e:
            print(f"could not save the run as an example: {type(e).__name__}: {e}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_summary.json").write_text(json.dumps({"run": name, "ok": ok, "stages": results, "repair_loops": loops,
                                                          **({"stopped_by_user_after": stopped_after} if stopped_after else {}),
                                                          "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1, default=str))
    if not ok:
        # the run's own failure report: the stage that failed, why, and every repair loop tried on the way
        failed = next((r for r in reversed(results) if not r.get("finished") and not r.get("skipped")), {})
        (out_dir / "failure_report.json").write_text(json.dumps({
            "run": name, "failed_stage": failed.get("stage"), "reason": failed.get("failure"), "upstream": failed.get("upstream"),
            "stage_failure_report": str(Path(failed["workspace"]) / "failure_report.json") if failed.get("workspace") else None,
            "repair_loops": loops, "stages": results}, indent=1, default=str))
    elif (out_dir / "failure_report.json").exists():
        (out_dir / "failure_report.json").replace(out_dir / "failure_report_previous.json")
    print(f"summary: {out_dir / 'run_summary.json'}")
    if ok and unanswered:
        return 2                                   # stopped after a stage because nobody could be asked; not a full run
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
