"""Stage 3: robot trajectory generation. Stage 1's subgoals and Stage 2's scene in, a verified robot trajectory
in simulation out, packaged for Stage 4.

    python stage3_trajectory_generation/run_stage3.py dexagent.yaml
    python stage3_trajectory_generation/run_stage3.py dexagent.yaml --name x            # uses runs/x/stage1 and runs/x/stage2
    python stage3_trajectory_generation/run_stage3.py dexagent.yaml --semantic s.json --recon scene_recon.json --video v.mp4

Writes <output_dir>/<name>/stage3/: the accepted delivery in verified/<id>/ (sealed by delivery.json, named by
accepted.json), copies of its trajectory.npz, phases.json, sim_video.mp4, trajectory_manifest.json, stage4_task/ and the
execution's per-subgoal folders (episode/) at the top, the run's trace.jsonl, new_tools.json and run.json, and the
agent's workspace in .candidates/<id>/ (plans, poses, executions; failure_report.json when the stage fails).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from agent.config import load_config, run_dir, semantics_file  # noqa: E402

TOOL_CATEGORIES = ["Grasp & Poses", "Grasp Stress Tests", "Trajectory & Execution", "Task & Rollout Gates", "Episode Audits",
                   "Scene Runtime & Rendering", "Measurements", "Perception & Grounding", "Scene Assertions"]


def stage_prompt() -> str:
    return (HERE / "prompts" / "stage3.md").read_text()


def round_budget(cfg: dict, semantic: Path) -> tuple[int, str]:
    """Stage 3's round budget: stage3.max_rounds plus stage3.rounds_per_subgoal for every Stage 1 subgoal, at most
    stage3.max_rounds_cap, so a nineteen-step lego build gets the room a three-step pick does not need."""
    s3 = cfg.get("stage3") or {}
    base = int(s3.get("max_rounds") or (cfg.get("agent") or {}).get("max_rounds", 40))
    per = int(s3.get("rounds_per_subgoal") or 0)
    try:
        n = len(json.loads(Path(semantic).read_text()).get("subgoals") or [])
    except Exception:
        n = 0
    total = base + per * n
    cap = s3.get("max_rounds_cap")
    if cap:
        total = min(total, int(cap))
    return total, (f"{total} rounds ({base} + {per} x {n} Stage 1 subgoals" + (f", capped at {int(cap)}" if cap and base + per * n > int(cap) else "") + ")"
                   if per else f"{total} rounds")


def user_message(video: Path, semantic: Path, recon: Path, workspace: Path, cfg: dict, budget: str | None = None) -> str:
    setup = cfg["inputs"].get("setup")
    s3 = cfg.get("stage3") or {}
    lines = [f"Workspace: {workspace}", f"Video: {video}", f"Stage 1 deliverable: {semantic}", f"Stage 2 deliverable: {recon} (scene.xml, camera.json, assets/ next to it)",
             f"Setup folder: {setup} (setup.yaml, robot/sharpa_left.xml, robot/sharpa_right.xml)",
             f"Stage 4 samples requested: {int((cfg.get('stage4') or {}).get('num_samples', 3))}"]
    hands = [p for p in (recon.parent / "hands.npz", recon.parent / "hands" / "hands.npz") if p.exists()]
    if hands:
        lines.append(f"Demonstrator's hand joints (Stage 2, rig frame): {hands[0]} (human_grasp_seeds hands_npz)")
    egodex = (cfg.get("inputs") or {}).get("egodex_hdf5")
    if egodex and not hands:
        ep = recon.parent / "episode.json"
        lines.append(f"EgoDex episode: {egodex}" + (f" with {ep} (human_grasp_seeds egodex_hdf5 + episode_json)" if ep.exists() else ""))
    if budget:
        lines.append(f"Round budget: {budget}. Finished subgoals are kept: re-run with execute_subgoal_plan(..., resume=True, from_subgoal=k), never from the start.")
    if s3.get("notes"):
        lines.append(f"Notes from the run configuration: {s3['notes']}")
    lines.append("Write trajectory.npz, phases.json, sim_video.mp4, stage4_task/ and trajectory_manifest.json into the workspace, verify, then call finish.")
    return "\n".join(lines)


def make_verifiers(cfg: dict, semantic: Path, recon: Path, lib: Path):
    from agent.library import call_tool, load_index
    from agent.verification_feedback import compact_evidence
    from tool_library.common.evidence import file_digest, value_digest
    index = load_index(lib)
    trusted = {t["name"]: file_digest(lib / "verifiers" / (t["name"] + ".py")) for t in index["tools"]
               if t.get("type") == "verifier" and t.get("category") == "Grasp Stress Tests"}

    def _call(name, args):
        r = call_tool(lib, index, name, args, cfg)
        return r.get("result") if r.get("ok") else {"ok": False, "issues": [r.get("error")]}

    def manifest_check(ws: Path):
        return _call("trajectory_manifest_check", {"path": str(ws / "trajectory_manifest.json"), "scene_semantic_json": str(semantic),
                                                   "library_index": str(lib / "index.json"), "trusted_verifiers": trusted,
                                                   "trusted_execution_reports": list(cfg.get("_executor_receipts", {}))})

    def gate_suite(ws: Path):
        # the delivered episode re-measured in the scene it ran in: every episode audit, each subgoal against its Stage 1
        # goal sentence (not only its own expect), the robot and physics against the setup, and a Stage 4 hand-off that
        # randomizes two thirds of the drawn layouts. Nothing the agent wrote about its own results is trusted.
        s3 = cfg.get("stage3") or {}
        r = call_tool(lib, index, "stage3_gate_suite", {
            "workspace": str(ws), "scene_semantic_json": str(semantic), "scene_recon_json": str(recon),
            "draws": int(s3.get("stage4_draws", 3)), "strict_quality": bool(s3.get("strict_quality", True)),
            **({"setup_dir": cfg["inputs"]["setup"]} if cfg["inputs"].get("setup") else {})}, cfg)
        if not r.get("ok"):
            return {"ok": False, "issues": [r.get("error")]}
        rep = r["result"]
        def digest(path):
            return file_digest(path) if path.is_file() else None

        manifest_path = ws / "trajectory_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            manifest = {}
        def candidate_path(value, default):
            p = Path(value or default)
            return p if p.is_absolute() else ws / p

        scene = candidate_path(manifest.get("scene_executed"), "scene_physics.xml")
        trajectory = candidate_path((manifest.get("episode") or {}).get("trajectory"), "trajectory.npz")
        # Comparison of goals is meaningful only under the same task, physics,
        # frozen framework and verification settings; candidate bytes are kept
        # separate so a changed trajectory can actually be compared.
        scope = {"semantic": digest(semantic), "recon": digest(recon), "scene": digest(scene),
                 "input_contract": digest(ws / "input_contract.json"),
                 "draws": int(s3.get("stage4_draws", 3)), "strict_quality": bool(s3.get("strict_quality", True))}
        binding = {"trajectory_sha256": digest(trajectory), "manifest_sha256": digest(manifest_path)}
        document = {"schema": "dexagent.gate_feedback.v1", "authority": "diagnostic_only",
                    "comparison_scope": value_digest(scope), "scope_inputs": scope,
                    "candidate_binding": binding, "report": rep}
        # Unique per invocation; never silently reuse an editable old report.
        folder = ws / ".gate_feedback"
        if folder.is_symlink():
            raise ValueError("gate feedback directory must not be a symlink")
        folder.mkdir(exist_ok=True)
        report_path = folder / (uuid.uuid4().hex + ".json")
        with report_path.open("x") as stream:
            json.dump(document, stream, indent=1, default=str)
        diagnostics = {"path": str(report_path), "sha256": file_digest(report_path), "authority": "diagnostic_only"}
        checks = rep.get("reports", {})
        return {"ok": rep["ok"], "issues": rep["issues"], "warnings": rep.get("warnings", []),
                "deferred": rep.get("deferred", {}),
                "report": {k: v.get("ok") if isinstance(v, dict) else False for k, v in checks.items()},
                "failed_checks": {k: compact_evidence(v) for k, v in checks.items()
                                  if not isinstance(v, dict) or v.get("ok") is not True},
                "task_goal_report": checks.get("stage1_goal_gate"),
                "comparison_scope": document["comparison_scope"], "candidate_binding": binding,
                "diagnostic_report": diagnostics}

    manifest_check.__name__ = "trajectory_manifest_check"
    gate_suite.__name__ = "stage3_gate_suite"
    return [manifest_check, gate_suite]


def make_diagnosis(cfg: dict, semantic: Path, recon: Path, lib: Path):
    """The verdict hook: every failed finish verdict gets diagnose_stage3_failure's failure classes and ranked next steps."""
    from agent.library import call_tool, load_index
    index = load_index(lib)

    def diagnose(ws: Path, verdict: dict) -> str:
        execution = verdict.get("executor_result")
        executor_args = verdict.get("executor_args") or {}
        current = ({"outcomes": (execution.get("files") or {}).get("outcomes") or execution.get("subgoals", []),
                    "check_hand_scenes": False, "include_workspace_history": False,
                    "plan": executor_args.get("plan"),
                    **({"scene": executor_args["scene"]} if executor_args.get("scene") else {})}
                   if isinstance(execution, dict) else {})
        r = call_tool(lib, index, "diagnose_stage3_failure", {"workspace": str(ws), "verdict": verdict, "scene_semantic_json": str(semantic),
                                                              "scene_recon_json": str(recon),
                                                              "include_workspace_history": False,
                                                              **current,
                                                              **({"setup_dir": cfg["inputs"]["setup"]} if cfg["inputs"].get("setup") else {})}, cfg, cwd=ws)
        if not r.get("ok"):
            return f"(diagnose_stage3_failure could not run: {str(r.get('error'))[:300]})"
        return r["result"]["summary"]
    return diagnose


class ExecutionProgress:
    """Result hook of execute_subgoal_plan: every physical execution is scored against the Stage 1 goals with
    stage1_goal_gate (the criterion finish applies), so the search is steered by the task, not only by the plan's own
    expectations; the best execution so far is kept and named; executions that stop improving are pointed out, and the
    search ends as an unsolved checkpoint after stall_stop of them. Diagnostic only: acceptance stays with finish."""

    def __init__(self, cfg: dict, semantic: Path, recon: Path, lib: Path):
        from agent.library import load_index
        s3 = cfg.get("stage3") or {}
        self.cfg, self.semantic, self.recon, self.lib = cfg, Path(semantic), Path(recon), Path(lib)
        self.index = load_index(lib)
        self.stall_note = int(s3.get("stall_note_executions", 10))
        self.stall_stop = int(s3.get("stall_stop_executions", 25))
        self.n, self.best, self.best_row, self.since_best = 0, None, None, 0
        self.repeat_key, self.repeat_n = None, 0      # the same subgoal failing the same checks, execution after execution
        try:
            self.n_stage1 = len(json.loads(self.semantic.read_text()).get("subgoals") or [])
        except (OSError, ValueError):
            self.n_stage1 = 0

    def _repeat_note(self, ws, args, r):
        """When the same subgoal fails the same checks in consecutive executions, the parameters changed in between do
        not act on that failure: point at the search over them (and at what led into it) from this execution."""
        ff = r.get("first_failure") if isinstance(r.get("first_failure"), dict) else None
        key = (ff.get("index"), tuple(sorted(str(c) for c in ff.get("failed") or []))) if ff else None
        self.repeat_n = self.repeat_n + 1 if key is not None and key == self.repeat_key else (1 if key else 0)
        self.repeat_key = key
        if self.repeat_n < 2:
            return None
        return (f"Subgoal {key[0]} has failed the same checks {list(key[1])} in {self.repeat_n} consecutive executions: what changed "
                f"between them does not act on this failure. Search it instead of another single trial: "
                f"{_search_call(ws, args, self.semantic, self.recon, self.n)} runs many variants of that entry (and, if none passes, of "
                f"the carry and grasp that led into it; for grasps also the hand's position on the part) in parallel from this execution's "
                f"passing prefix. If the search finds no passing variant, the strategy itself must change.")

    def __call__(self, ws: Path, args: dict, raw: dict):
        from agent.library import call_tool
        r = raw.get("result") if raw.get("ok") else None
        if not isinstance(r, dict) or r.get("dry_run"):
            return None
        files = r.get("files") or {}
        # a failed execution leaves only its unvalidated recording (trajectory_proposal): still the motion to score
        pick = next((k for k in ("trajectory", "trajectory_proposal") if files.get(k) and Path(files[k]).is_file()), None)
        traj = files.get(pick) if pick else None
        scene = files.get("scene_physics" if pick == "trajectory" else "scene_proposal") or files.get("scene_physics")
        executed = [g for g in (r.get("subgoals") or []) if g.get("frames")]
        self.n += 1
        n_exec_ok = sum(1 for g in r.get("subgoals") or [] if g.get("ok"))
        repeat = self._repeat_note(ws, args, r)
        if not (traj and scene and executed):
            score = (0, n_exec_ok, -10 ** 6)
            if self.best is None or score > self.best:
                self.best, self.best_row, self.since_best = score, {"execution": self.n, "trajectory": None, "goals_ok": 0,
                                                                    "task_ok": False, "executor_ok": n_exec_ok, "issues": []}, 0
            else:
                self.since_best += 1
            return {"text": f"[harness] execution #{self.n} left no recording to check against the Stage 1 goals; executions "
                            f"since the best improved: {self.since_best}.", "execution": self.n, "since_best": self.since_best,
                    "stop": "search_stalled" if self.since_best >= self.stall_stop else None,
                    "stop_reason_text": "no improvement of the Stage 1 goal score over the last executions" if self.since_best >= self.stall_stop else None}
        folder = Path(ws) / ".execution_checks"
        folder.mkdir(exist_ok=True)
        man = {"scene_recon": str(self.recon), "subgoals": [
            {"index": g["index"], "stage1_indices": g.get("stage1_indices") or [g["index"]], "frames": g["frames"],
             "motion": g.get("motion"), "side": g.get("side"), "object": g.get("object")} for g in executed]}
        tag = self.n
        while (folder / f"execution_{tag:03d}_manifest.json").exists():   # a recovered session keeps the earlier checks
            tag += 1
        mp = folder / f"execution_{tag:03d}_manifest.json"
        mp.write_text(json.dumps(man, indent=1))
        g = call_tool(self.lib, self.index, "stage1_goal_gate", {
            "scene_semantic_json": str(self.semantic), "trajectory_manifest_json": str(mp), "scene": str(scene),
            "trajectory": str(traj), "scene_recon_json": str(self.recon), "out_json": str(folder / f"execution_{tag:03d}_goals.json")}, self.cfg)
        rep = g.get("result") if g.get("ok") else None
        if not isinstance(rep, dict):
            self.since_best += 1 if self.best is not None else 0
            return {"text": f"[harness] task check of execution #{self.n} could not run: {str(g.get('error'))[:300]}",
                    "execution": self.n, "since_best": self.since_best,
                    "stop": "search_stalled" if self.since_best >= self.stall_stop else None}
        issues = list(dict.fromkeys(str(i) for i in rep.get("issues") or []))
        goals_ok = sum(1 for row in rep.get("subgoals") or [] if row.get("ok"))
        task_ok = not any("task_success" in i for i in issues)
        complete = bool(rep.get("ok")) and n_exec_ok == len(r.get("subgoals") or []) and len(executed) == (r.get("n_subgoals") or len(executed))
        score = (goals_ok + (1 if task_ok and goals_ok == self.n_stage1 else 0), n_exec_ok, -len(issues))
        row = {"execution": self.n, "trajectory": traj, "goals_ok": goals_ok, "task_ok": task_ok, "executor_ok": n_exec_ok,
               "issues": issues[:3]}
        if self.best is None or score > self.best:
            self.best, self.best_row, self.since_best = score, row, 0
        else:
            self.since_best += 1
        lines = [f"[harness] Task check of execution #{self.n} (stage1_goal_gate, the criterion finish applies; diagnostic): "
                 f"Stage 1 goals met {goals_ok}/{self.n_stage1}, task_success {'met' if task_ok else 'NOT met'}, "
                 f"plan subgoals passed {n_exec_ok}/{len(r.get('subgoals') or [])}."]
        lines += [f"  failing: {i[:300]}" for i in issues[:4]]
        b = self.best_row
        lines.append(f"Best execution so far: #{b['execution']} ({b['goals_ok']}/{self.n_stage1} goals, task_success "
                     f"{'met' if b['task_ok'] else 'not met'}, {b['executor_ok']} plan subgoals), trajectory {b['trajectory']}"
                     + (f"; executions since it improved: {self.since_best}." if self.since_best else "; this one is the best."))
        stop = None
        plan_only = [f for g in (r.get("subgoals") or []) if not g.get("ok") for f in (g.get("failed") or [])]
        if bool(rep.get("ok")) and goals_ok == self.n_stage1 and task_ok and plan_only and not complete:
            lines.append(f"Every Stage 1 goal and task_success hold in this execution; only the plan's own expectations failed "
                         f"({sorted(set(map(str, plan_only)))[:6]}). Those are targets your plan set, not the task's: relax the one "
                         f"that is not a Stage 1 requirement (e.g. a place orientation or a tight target) and re-execute this plan "
                         f"unchanged otherwise, so the executor passes and the episode can be packaged.")
        if complete:
            lines.append("Every Stage 1 goal holds in this execution: package THIS episode (manifest, stress tests on the poses it "
                         "used, stage4_task) and call finish now; finish runs the full gate suite.")
        elif self.since_best >= self.stall_stop:
            stop = "search_stalled"
            lines.append(f"No improvement in {self.since_best} executions: the harness ends the search here (unsolved checkpoint).")
        elif self.since_best >= self.stall_note:
            lines.append(f"No improvement in the last {self.since_best} executions. Stop tuning this strategy: go back to the best "
                         f"execution and change what its failing goal measures (a different grasp, approach, release or "
                         f"support), or report_blocked with the measurement. At {self.stall_stop} executions without "
                         f"improvement the harness ends the search.")
        if repeat and not complete:
            lines.append(repeat)
        return {"text": "\n".join(lines), "stop": stop, "execution": self.n, "score": list(score), "best": self.best_row,
                "since_best": self.since_best,
                "stop_reason_text": "no improvement of the Stage 1 goal score over the last executions; candidate remains unfinished, "
                                    "task feasibility is not established" if stop else None}


def _search_call(ws, args, semantic, recon, n):
    """A search_subgoal_variants call that starts from this execution, with the executor options it ran with."""
    opts = {k: v for k, v in (args or {}).items() if k not in ("scene", "plan", "out_dir", "resume", "from_subgoal", "stop_on_fail", "dry_run")}
    call = {"scene": args.get("scene"), "plan": args.get("plan") if isinstance(args.get("plan"), str) else "<your plan file>",
            "out_dir": str(Path(ws) / f"search_{n:02d}"), "prefix_dir": args.get("out_dir"), "exec_options": opts,
            "scene_semantic_json": str(semantic), "scene_recon_json": str(recon)}
    return "search_subgoal_variants(" + ", ".join(f"{k}={v!r}" for k, v in call.items()) + ")"


def run_method_modules(runner, out: dict, cfg: dict, semantic: Path, recon: Path, video=None) -> dict | None:
    """When the agent's search ends unfinished, the method modules (stage3.methods: physics-informed retargeting, sampling
    MPC refinement, reinforcement-learning tracking) continue from what it left, each called like an agent's skill and
    its episode executed, packaged and verified like an agent's (methods.py). Returns their record, or None when none
    run (switched off, a cross-stage report, an administrative stop)."""
    from stage3_trajectory_generation.methods import run_method_modules as run
    return run(runner, out, cfg, semantic, recon, video=video)


# Harness metadata is archived separately from search artifacts. Existing plan,
# pose and executor paths remain valid, but none of their old verdicts is trusted.
_RECOVERY_FILES = ("trace.jsonl", "input_contract.json", "run.json", "failure_report.json",
                   "failure_report_agent.json", "progress.json", "new_tools.json", "recovery.json",
                   "search_progress.json", "backend_checkpoint.json")


def prepare_attempt(public: Path, resume_attempt: str | Path | None = None):
    """Return (candidate workspace, recovery record or None).

    Explicit recovery accepts only an existing direct child of this run's
    .candidates directory. Public/verified outputs and symlink aliases are not
    candidate workspaces. Old control files are archived byte-for-byte before
    a new runner can overwrite them; normal search artifacts stay in place.
    """
    from agent.external import atomic_json
    from tool_library.common.evidence import file_digest
    public = Path(public).resolve()
    candidates = public / ".candidates"
    if candidates.is_symlink():
        raise ValueError(".candidates must be a real directory, not a symlink")
    if resume_attempt is None:
        ws = candidates / uuid.uuid4().hex
        ws.mkdir(parents=True)
        return ws, None
    ws = Path(os.path.abspath(resume_attempt))
    if ws.parent != candidates or ws.is_symlink() or ws.resolve() != ws:
        raise ValueError("--resume-attempt must name a direct, non-symlink child of this run's .candidates directory")
    if not ws.is_dir():
        raise ValueError("--resume-attempt candidate directory does not exist")
    for name in (*_RECOVERY_FILES, "accepted.json", "delivery.json", "verification.json"):
        path = ws / name
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f"candidate harness metadata must be a regular file: {name}")
    # Never reopen a prior finished result. These disk claims are rejection-only:
    # no value from them can authorize acceptance or restore an executor receipt.
    for path in (ws / "run.json", public / "run.json"):
        if not path.is_file():
            continue
        try:
            previous = json.loads(path.read_text())
        except (ValueError, UnicodeError):
            continue  # interrupted/truncated metadata is archived, never trusted
        if not isinstance(previous, dict):
            continue
        same_candidate = path.parent == ws or str(previous.get("workspace", "")) == str(ws)
        if same_candidate and previous.get("finished") is True:
            raise ValueError("cannot resume a previously finished candidate")
    if (ws / "accepted.json").exists() or (ws / "delivery.json").exists():
        raise ValueError("cannot resume a workspace bearing an accepted/delivery marker")
    recovery_id = uuid.uuid4().hex
    archive = public / ".recoveries" / ws.name / recovery_id
    # Refuse symlink redirection of the archive, including its parent directories.
    for path in (public / ".recoveries", archive.parent):
        if path.is_symlink():
            raise ValueError("recovery archive must remain inside the run directory")
    archive.mkdir(parents=True)
    originals = {}
    for name in _RECOVERY_FILES:
        source = ws / name
        if source.is_file():
            target = archive / name
            shutil.copy2(source, target)
            digest = file_digest(target)
            if digest != file_digest(source):
                raise ValueError(f"candidate metadata changed while archiving: {name}; stop the old harness before resuming")
            originals[name] = {"sha256": digest, "bytes": target.stat().st_size}
    record = {"schema": "dexagent.stage3_recovery.v1", "recovery_id": recovery_id,
              "created_utc": datetime.now(timezone.utc).isoformat(), "workspace": str(ws), "archive": str(archive),
              "previous_files": originals, "candidate_files_trust": "unverified",
              "history_restored": False, "executor_receipts_restored": False,
              "verification": "all current gates required; a fresh trusted executor invocation is mandatory"}
    atomic_json(archive / "provenance.json", record)
    # Only metadata is reset. Byte copies above survive another model outage.
    for name in originals:
        (ws / name).unlink()
    atomic_json(ws / "recovery.json", record)
    try:
        traces = sorted((public / ".recoveries" / ws.name).glob("*/trace.jsonl"), key=lambda q: q.stat().st_mtime)
        (ws / "previous_session_digest.md").write_text(previous_session_digest(traces))
    except Exception as exc:                     # the digest is a convenience; a recovery never fails on it
        (ws / "previous_session_digest.md").write_text(f"(no digest: {type(exc).__name__}: {exc})\n")
    return ws, record


def previous_session_digest(traces, notes=6) -> str:
    """What the earlier sessions of this candidate did, read from their archived traces: every execution (plan, folder,
    per-subgoal result, first failure with the executor's hints), every search, every finish verdict, and each session's
    last status notes. Diagnostic only, like the files it summarizes."""
    out = ["# Earlier sessions of this candidate (from their archived traces; diagnostic, not evidence)", ""]
    for k, tr in enumerate(traces, 1):
        rows, said = [], []
        for line in Path(tr).read_text().splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("event") == "assistant":
                said += [b.get("text", "") for b in d.get("content") or [] if isinstance(b, dict) and b.get("type") == "text" and b.get("text")]
            if d.get("event") == "tool" and d.get("name") == "execute_subgoal_plan":
                a, r = d.get("args") or {}, ((d.get("result") or {}).get("result") or {})
                if not isinstance(r, dict):
                    r = {}
                sg = ", ".join(f"{g.get('tag') or g.get('index')}:{'ok' if g.get('ok') else 'FAIL'}" for g in r.get("subgoals") or [])
                ff = r.get("first_failure") if isinstance(r.get("first_failure"), dict) else {}
                hints = []
                m = ff.get("measured") or {}
                for h in [m] + list((m.get("hands") or {}).values()):
                    if isinstance(h, dict) and h.get("hint"):
                        hints.append(str(h["hint"])[:240])
                err = ff.get("error") or ((d.get("result") or {}).get("error") if not r else None)
                rows.append(f"- round {d.get('round')}: execute_subgoal_plan plan={a.get('plan') if isinstance(a.get('plan'), str) else '(inline)'} "
                            f"out_dir={a.get('out_dir')}{' dry_run' if a.get('dry_run') else ''}{' resume from ' + str(a.get('from_subgoal')) if a.get('resume') else ''}"
                            f" -> {sg or 'no segments'}" + (f"; first failure subgoal {ff.get('index')} {ff.get('motion')} {ff.get('failed')}" if ff else "")
                            + (f"; {str(err)[:240]}" if err else "") + "".join(f"\n  hint: {h}" for h in hints[:2]))
            if d.get("event") == "tool" and d.get("name") == "search_subgoal_variants":
                r = ((d.get("result") or {}).get("result") or {})
                if isinstance(r, dict):
                    b = r.get("best") or {}
                    rows.append(f"- round {d.get('round')}: search_subgoal_variants {r.get('evaluations')} trials, ok={r.get('ok')}, best {b.get('id')} "
                                f"passed {b.get('subgoals_passed')} subgoals (failed {b.get('failed')}); report {r.get('report')}")
            if d.get("event") == "verify":
                v = d.get("verdict") or {}
                rows.append(f"- round {d.get('round')}: finish -> {'PASS' if v.get('ok') else 'FAIL'}; "
                            + "; ".join(str(i)[:200] for i in (v.get("issues") or [])[:4]))
        out += [f"## Session {k} ({tr})", ""] + (rows or ["- no executions"]) + ["", "Its last status notes:", ""]
        out += [f"> {t.strip()[:700]}" for t in said[-notes:]] + [""]
    return "\n".join(out) + "\n"


def recovery_message() -> str:
    """Generic recovery instructions contain no task solution or prior verdict."""
    return ("This is a recovery of an unfinished, unverified Stage 3 candidate in the same workspace. "
            "Inspect recovery.json, previous_session_digest.md (what the earlier sessions executed, measured and noted, read "
            "from their archived traces), the archived failure report and existing search artifacts before deciding how to continue "
            "using the supplied demonstration and current tools. Repair a known unsuccessful strategy before repeating a costly "
            "whole-episode execution; earlier recordings may be inspected as diagnostics, not accepted evidence. "
            "Prior files, reports and summaries may come from a different framework version and are untrusted; "
            "no previous success or acceptance is restored. Recheck an archived causal diagnosis with current measurement tools "
            "before using it to declare an upstream error or task infeasibility. The model conversation starts fresh. "
            "Before finalization, successfully execute the physical plan through execute_subgoal_plan again to obtain a new trusted executor receipt; resume=True may reuse compatible "
            "checkpoints, but old receipt files and traces cannot authorize finish. All current gates must pass again.")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="DexAgent Stage 3")
    ap.add_argument("config", nargs="?", default=os.environ.get("DEXAGENT_CONFIG", str(REPO / "dexagent.yaml")))
    ap.add_argument("--video"), ap.add_argument("--semantic"), ap.add_argument("--recon"), ap.add_argument("--name")
    ap.add_argument("--max-rounds", type=int)
    ap.add_argument("--resume-attempt", type=Path, help="resume an interrupted unverified direct child of this run's stage3/.candidates directory")
    ap.add_argument("--llm", default="config", help="config (the YAML's agent block) or mock:<turns.json>")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    cfg["_executor_receipts"] = {}
    if a.video:
        cfg["inputs"]["video"] = str(Path(a.video).resolve())
    if a.name:
        cfg["run"]["name"] = a.name
    video = Path(cfg["inputs"]["video"])
    public = run_dir(cfg, "stage3")
    semantic = Path(a.semantic).resolve() if a.semantic else semantics_file(public.parent / "stage1")
    recon = Path(a.recon).resolve() if a.recon else (public.parent / "stage2" / "scene_recon.json")
    for p, what in ((semantic, "Stage 1 deliverable"), (recon, "Stage 2 deliverable")):
        if not p.exists():
            sys.exit(f"{what} not found: {p}")
    from agent.library import library_root
    from agent.artifacts import publish_stage3
    from agent.external import atomic_json
    try:
        ws, recovery = prepare_attempt(public, a.resume_attempt)
    except (OSError, ValueError) as exc:
        ap.error(f"cannot prepare Stage 3 candidate: {exc}")
    attempt = recovery["recovery_id"] if recovery else ws.name
    atomic_json(public / "candidate.json", {"attempt": ws.name, "workspace": str(ws), "status": "unverified",
                                            **({"recovery_id": recovery["recovery_id"]} if recovery else {})})
    if (public / "accepted.json").exists():
        (public / "accepted.json").replace(public / ("previous-accepted-" + attempt + ".json"))
    from agent.artifacts import retire_public_copies
    retire_public_copies(public, attempt[:8])          # the previous delivery's top-level video and Stage 4 folder
    # Retire the previous run verdict as well as its pointer. A tool-written
    # pointer must never borrow an earlier successful run's publication status.
    atomic_json(public / "run.json", {"finished": False, "stage": "stage3", "status": "running",
                                     "workspace": str(ws), "attempt": attempt})
    from agent.input_identity import video_verifier, input_failure
    identity_gate = video_verifier(video, semantic)
    identity_report = identity_gate(ws)
    if identity_report.get("ok") is not True:
        input_failure(public, identity_report)
        return 1
    from agent.reference_evidence import revalidate_source_recipes
    reference_report = revalidate_source_recipes(recon.parent, video, library=library_root(cfg), cfg=cfg)
    if reference_report.get("ok") is not True:
        out = {"finished": False, "stage": "stage3",
               "failure": "Stage 2 source-reference mask provenance is unverified",
               "upstream": {"stage": "stage2", "evidence": reference_report},
               "verdict": {"ok": False, "reports": {"source_reference_provenance_check": reference_report}}}
        atomic_json(public / "run.json", out)
        atomic_json(public / "failure_report.json", out)
        return 1
    def finalize(candidate, verdict):
        return publish_stage3(candidate, public, verdict, semantic, recon, video=video)
    lib = library_root(cfg)
    from agent.reuse import try_reuse
    verifiers = make_verifiers(cfg, semantic, recon, lib)
    verifiers.insert(0, identity_gate)
    # downstream of a cross-stage repair the inputs changed: read why, never reuse an example
    from agent import repair
    note = repair.take_note(public) if repair.has_note(public) else ""
    if not note and not recovery:
        # our own sample: an example's recorded twin episode is the answer first, else its trajectory (shifted by the
        # positional difference of its assets fitted to this video)
        from agent.reuse import stage3_example_trajectory, stage3_twin_episode
        out = stage3_twin_episode(cfg, ws, public, video, semantic, recon)
        if out is None:
            out = stage3_example_trajectory(cfg, ws, public, video, semantic, recon)
        if out is not None:
            if out.get("finished"):
                from agent.reuse import run_verifiers_quietly
                run_verifiers_quietly(verifiers, ws)  # the stage's verifiers over the reused episode, quietly; never a rejection
            (ws / "run.json").write_text(json.dumps(out, indent=1, default=str))
            atomic_json(public / "run.json", out)
            if not out.get("finished"):
                atomic_json(public / "failure_report.json", out)
            return 0 if out.get("finished") else 1
    reused, reuse_note = (None, None) if note or recovery else try_reuse(cfg, "stage3", ws, video, scene_semantic=semantic, verifiers=verifiers)
    if reused is not None:
        try:
            reused["delivery"] = finalize(ws, reused["verdict"])
            atomic_json(public / "run.json", reused)
            return 0
        except Exception as exc:
            reuse_note = f"Recorded result could not be published: {type(exc).__name__}: {exc}. Generate a new verified execution."
    if a.llm.startswith("mock:"):
        from agent.llm import load_mock
        llm = load_mock(a.llm[5:], "stage3")
    else:
        from agent.budget import BudgetError
        from agent.llm import make_llm
        try:
            llm = make_llm(cfg)
        except BudgetError as exc:                    # a model API run needs a spending cap; refused before any call
            print(f"  {exc}", flush=True)
            out = {"finished": False, "stage": "stage3", "workspace": str(ws), "failure": str(exc)}
            atomic_json(public / "run.json", out)
            return 1
    envs = cfg.get("environments") or {}
    if envs.get("isaac"):                            # deformables run in Isaac Sim (execute_isaac_deformables)
        os.environ.setdefault("DEXAGENT_ISAAC_PYTHON", str(envs["isaac"]))
    from agent.runner import StageRunner
    cfg["_verification_inputs"] = [str(p) for p in (video, semantic, recon.parent, cfg["inputs"].get("setup"),
                                                 cfg["inputs"].get("egodex_hdf5")) if p]
    if recovery:
        cfg["_verification_inputs"].append(recovery["archive"])
    budget, budget_text = (a.max_rounds, f"{a.max_rounds} rounds") if a.max_rounds else round_budget(cfg, semantic)
    runner = StageRunner(cfg, llm, ws, stage_prompt(), verifiers, categories=TOOL_CATEGORIES, max_rounds=budget,
                         verdict_hook=make_diagnosis(cfg, semantic, recon, lib), stage="stage3", finalize=finalize, recovery=recovery,
                         result_hooks={"execute_subgoal_plan": ExecutionProgress(cfg, semantic, recon, lib)},
                         after_search=lambda runner_, out_: run_method_modules(runner_, out_, cfg, semantic, recon, video=video))
    msg = user_message(video, semantic, recon, ws, cfg, budget_text)
    if reuse_note:
        msg += "\n" + reuse_note
    msg += note
    if recovery:
        msg += "\n" + recovery_message()
    out = runner.run(msg)                             # an unfinished search continues with the method modules inside
    if not out.get("finished") and (public / "accepted.json").exists():
        # run_python and tool output paths are not an OS write sandbox. Preserve
        # an unexpected marker as diagnostic evidence, never as accepted output.
        rejected = public / ("rejected-accepted-" + attempt + ".json")
        (public / "accepted.json").replace(rejected)
        retire_public_copies(public, "rejected-" + attempt[:8])
        out["rejected_acceptance_marker"] = str(rejected)
    (ws / "run.json").write_text(json.dumps(out, indent=1, default=str))
    atomic_json(public / "run.json", out)
    if (ws / "failure_report.json").exists():
        atomic_json(public / "failure_report.json", json.loads((ws / "failure_report.json").read_text()))
    if (ws / "progress.json").exists():
        atomic_json(public / "progress.json", json.loads((ws / "progress.json").read_text()))
    for name in ("trace.jsonl", "new_tools.json"):          # the run's record and the tools it added, beside its outputs
        if (ws / name).is_file():
            shutil.copy2(ws / name, public / name)
    return 0 if out["finished"] else 1


if __name__ == "__main__":
    sys.exit(main())
