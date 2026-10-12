"""Diagnostic search memory; never an execution receipt or acceptance authority."""
from __future__ import annotations

import errno
import hashlib
import json
import shutil
import uuid
from functools import wraps
from pathlib import Path


def _digest(value, diagnostic=False):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=diagnostic, default=str).encode()).hexdigest()


def _diagnostic_only(method):
    """Optional search bookkeeping must never replace real executor feedback."""
    @wraps(method)
    def guarded(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except Exception:
            return None
    return guarded


def _bound(value, base, seen=(), workspace=None):
    """Bind nested JSON inputs and compiled XML assets, including external meshes.

    The executor resolves plan paths in the workspace first, then beside the
    plan. Nested candidate lists and hand-scene records also use workspace paths.
    Bind both locations conservatively: a changed actual input must never be
    hidden by a same-named file beside its containing JSON. Extra dependencies
    can permit another trial; they cannot suppress a genuine repair.

    Output directories are excluded by the caller. Cycles are represented by
    content identity instead of recursing. Unreadable inputs disable suppression.
    """
    workspace = base if workspace is None else workspace
    if isinstance(value, dict):
        return {str(k): _bound(v, base, seen, workspace) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_bound(v, base, seen, workspace) for v in value]
    if not isinstance(value, (str, Path)):
        return value
    named = Path(value)
    candidates = [named] if named.is_absolute() else [workspace / named, base / named]
    bindings, visited = [], set()
    for candidate in candidates:
        try:
            exists = candidate.is_file()
        except OSError as exc:
            # JSON also contains prose. A filename component or total path
            # beyond the filesystem limit cannot resolve here, but its literal
            # content still belongs in the input identity. Do not swallow
            # permission/I/O failures for real dependencies.
            if exc.errno == errno.ENAMETOOLONG:
                continue
            raise
        if not exists:
            continue
        path = candidate.resolve()
        if path in visited:
            continue
        visited.add(path)
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if path in seen:
            content = {"cycle_sha256": digest}
        elif path.suffix.lower() == ".json":
            content = _bound(json.loads(data), path.parent, (*seen, path), workspace)
        elif path.suffix.lower() == ".xml":
            import mujoco
            from agent.artifacts import model_digest
            content = {"xml_sha256": digest, "compiled": model_digest(mujoco.MjModel.from_xml_path(str(path)))}
        else:
            content = {"file_sha256": digest}
        bindings.append({"path": str(path), "contents": content})
    return {"resolved_inputs": bindings} if bindings else str(value)


# Explicit directions from common.outcome's checks, not guessed from metric names.
# Unsupported checks remain incomparable until their evaluator defines a measure.
_CHECK_METRICS = {
    "arm_unreachable": {"position_error_mm": "min"},
    "hand_penetration": {"max_hand_pen_mm": "min"},
    "object_penetration": {"max_object_pen_mm": "min"},
    "servo_tracking": {"max_track_pos_mm": "min", "max_track_rot_deg": "min"},
    "at_target": {"target_err_m": "min"},
    "at_rest": {"moved_in_last_window_m": "min"},
    "no_slip": {"slip_in_hand_m": "min", "slip_in_hand_deg": "min"},
    "rise": {"rise_m": "max"},
    "displaced": {"displacement_along_m": "max"},
    "seated": {"seat_remaining_mm": "abs_min"},
}


def _metrics(failure):
    """Return losses only when every failed check has a supported finite measure."""
    import math
    measured, values = failure.get("measured") or {}, {}
    for check in failure.get("failed") or []:
        name, data, prefix = check, measured, ""
        if check.startswith(("left:", "right:")):
            side, name = check.split(":", 1)
            data = (measured.get("hands") or {}).get(side) or {}
            prefix = side + ":"
        if name == "arm_unreachable" and "infeasible" in data:
            infeasible = data["infeasible"]
            if not isinstance(infeasible, list):
                return {}
            for entry in infeasible:
                if not isinstance(entry, dict) or not isinstance(entry.get("side"), str) or not entry["side"]:
                    return {}
                for metric in ("worst_mm", "worst_deg"):
                    value = entry.get(metric)
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                        return {}
                    key = prefix + "reach:" + entry["side"] + ":" + metric
                    values[key] = max(values.get(key, value), value)
            if "branch_jumps" in data:
                jumps = data["branch_jumps"]
                if not isinstance(jumps, dict):
                    return {}
                for side, entries in jumps.items():
                    if not isinstance(side, str) or not side or not isinstance(entries, list):
                        return {}
                    values[prefix + "reach:" + side + ":branch_jump_count"] = len(entries)
            if not values:
                return {}
            continue
        spec = _CHECK_METRICS.get(name)
        if not spec:
            return {}
        for metric, direction in spec.items():
            value = data.get(metric)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                return {}
            values[prefix + metric] = (abs(value) if direction == "abs_min" else -value if direction == "max" else value)
    return values


def _relation(candidate, best):
    """Pareto comparison: trading one failure for another is never improvement."""
    if not candidate or candidate.keys() != best.keys():
        return "unsupported_measurements"
    lower = any(candidate[k] < best[k] for k in candidate)
    higher = any(candidate[k] > best[k] for k in candidate)
    return "mixed" if lower and higher else "improved_measurements" if lower else "regressed_measurements" if higher else "unchanged"


def _plan_checks(value):
    """Allow grasp repairs, keeping targets, expectations, order and subdivision fixed."""
    if isinstance(value, dict):
        return {k: _plan_checks(v) for k, v in value.items()
                if k not in {"pose", "pose_candidates", "regrasp_pose"}}
    if isinstance(value, list):
        return [_plan_checks(v) for v in value]
    return value


class SearchProgress:
    """Stop identical failed execution loops, and expose measured search history.

    Two executions of identical inputs are allowed for confirmation. This does
    not suppress search skills (which may sample randomly), successful trials,
    revised poses/plans, or a changed resume checkpoint. State is session-local;
    a generator-written progress file cannot seed this memory.
    """
    def __init__(self, workspace):
        self.ws = Path(workspace).resolve()
        self.path = self.ws / "search_progress.json"
        self.failed_inputs = {}
        self.signatures = {}
        self.events = []
        self.best_physical_prefix = 0
        self.gate_failures = 0
        self.task_scope = None
        self.contract = None
        self.local_scopes = {}
        self.semantic_scopes = {}
        self.latest_execution = None
        self.latest_gate = None

    def bind_contract(self, contract):
        # Capture trusted inputs in memory, never reload a generator-written ledger.
        self.contract = contract
        self.task_scope = _digest({"files": contract.hashes, "models": contract.models})

    def _local_scope(self, args):
        try:
            if self.contract is not None and self.contract.check().get("ok") is not True:
                return None
            scene = Path(args["scene"])
            scene = scene if scene.is_absolute() else self.ws / scene
            if not scene.is_file():
                return None
            plan = args["plan"]
            base = self.ws
            if isinstance(plan, (str, Path)):
                path = Path(plan)
                path = path if path.is_absolute() else self.ws / path
                plan, base = json.loads(path.read_text()), path.parent
            return _digest({"task": self.task_scope,
                "physics_and_execution": _bound({k: v for k, v in args.items()
                    if k not in {"plan", "out_dir", "resume"}}, self.ws),
                "checks_and_topology": _bound(_plan_checks(plan), base, workspace=self.ws)})
        except Exception:
            return None

    def _retain_prefix(self, args, result, prefix, scope):
        """Preserve actual segment proposals; no receipt or automatic restore is issued."""
        if not prefix or not args.get("out_dir"):
            return None
        out = Path(args["out_dir"])
        out = (out if out.is_absolute() else self.ws / out).resolve()
        if not out.is_relative_to(self.ws):
            return {"unavailable": "execution output is outside candidate workspace"}
        rows = result.get("subgoals", [])[:prefix]
        try:
            recorded = json.loads((out / "outcomes.json").read_text())[:prefix]
            if len(recorded) != prefix or any(r.get("ok") is not True or r.get("dry_run")
                    or r.get("tag") != row.get("tag") or not r.get("resume_signature")
                    for r, row in zip(recorded, rows)):
                return {"unavailable": "passed prefix lacks compatible physical outcome records"}
        except (OSError, ValueError, TypeError, AttributeError):
            return {"unavailable": "passed prefix outcome records unavailable"}
        folders = []
        for row in rows:
            tag = row.get("tag")
            if not isinstance(tag, str) or Path(tag).name != tag or tag in {".", ".."}:
                return {"unavailable": "passed segment has no safe artifact tag"}
            folder = out / tag
            if folder.is_symlink() or not folder.resolve().is_relative_to(out):
                return {"unavailable": "segment path escapes execution output"}
            required = ("rollout.npz", "rollout_runtime.npz", "outcome.json", "state.json", "phases.json")
            if not all((folder / name).is_file() for name in required):
                return {"unavailable": "passed segment has incomplete physical recording files"}
            folders.append(folder)
        destination = self.ws / ".search_prefixes" / uuid.uuid4().hex
        if not destination.resolve().is_relative_to(self.ws) or destination.parent.is_symlink():
            return {"unavailable": "prefix archive directory escapes candidate workspace"}
        files = []
        for folder in folders:
            files.extend(p for p in folder.rglob("*") if p.is_file()
                         and p.suffix.lower() not in {".mp4", ".gif", ".png", ".jpg"})
        files.extend(out / name for name in ("scene_physics.xml", "scene_proposal.xml") if (out / name).is_file())
        if any(p.is_symlink() or not p.resolve().is_relative_to(out) for p in files):
            return {"unavailable": "artifact symlink or dependency escapes execution output"}
        try:
            digests = {}
            for source in files:
                relative = source.relative_to(out)
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                digests[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
            from agent.external import atomic_json
            atomic_json(destination / "outcomes.json", recorded)
            plan = args.get("plan")
            if isinstance(plan, (str, Path)):
                plan_path = Path(plan)
                plan_path = plan_path if plan_path.is_absolute() else self.ws / plan_path
                plan = json.loads(plan_path.read_text())
            atomic_json(destination / "plan_original.json", plan)
            atomic_json(destination / "prefix_manifest.json", {
                "authority": "unverified_diagnostic_proposals", "comparison_scope": scope,
                "execution_args": args, "bound_inputs": _bound(args, self.ws),
                "passed_prefix_entries": prefix, "sha256": digests,
                "reuse": "Inspect/copy proposals explicitly; changed dependencies require fresh execution. Full independent gates remain required."})
            return {"path": str(destination), "manifest": str(destination / "prefix_manifest.json"),
                    "passed_prefix_entries": prefix, "authority": "unverified_diagnostic_proposals",
                    "reuse": "No automatic restore or acceptance; original external dependencies must remain compatible."}
        except Exception:
            shutil.rmtree(destination, ignore_errors=True)
            return {"unavailable": "could not preserve physical prefix artifacts"}

    def _compare_execution(self, args, result, prefix, failure, physical):
        scope = self._local_scope(args)
        metrics = _metrics(failure)
        reach_screen = (not physical and not args.get("dry_run") and not result.get("dry_run")
                        and failure.get("failed") == ["arm_unreachable"] and bool(metrics))
        evidence_kind = "kinematic_reach_screen" if reach_screen else "physical_execution"
        if scope and reach_screen:
            scope = _digest({"input_scope": scope, "evidence_kind": evidence_kind})
        report = {"scope": scope, "status": "incomparable", "semantic_progress": False,
                  "evidence_kind": evidence_kind,
                  "scope_note": "Only fixed plan targets/checks/topology and physics are compared; changed inputs are not improvement."}
        if not scope or not (physical or reach_screen):
            report["reason"] = "missing comparable input identity or physical recording"
            return report
        failure_scope = _digest({k: failure.get(k) for k in ("index", "step", "motion", "side", "object", "failed")})
        candidate = {"prefix": prefix, "failure_scope": failure_scope, "metrics": metrics,
                     "local_execution_ok": result.get("ok") is True, "event": len(self.events)}
        state = self.local_scopes.get(scope)
        status = "baseline"
        if state:
            best = state["best"]
            if prefix != best["prefix"]:
                status = "improved_local_prefix" if prefix > best["prefix"] else "regressed_local_prefix"
            elif candidate["local_execution_ok"] and not best["local_execution_ok"]:
                status = "improved_local_completion"
            elif candidate["local_execution_ok"] != best["local_execution_ok"]:
                status = "regressed_local_completion"
            elif failure_scope != best["failure_scope"]:
                status = "different_failure"
            elif candidate["local_execution_ok"]:
                status = "unchanged"
            else:
                status = _relation(metrics, best["metrics"])
        improves = status.startswith("improved_")
        if state is None:
            state = {"best": candidate, "since_improvement": 0}
            self.local_scopes[scope] = state
        elif improves:
            state.update(best=candidate, since_improvement=0)
        else:
            state["since_improvement"] += 1
        if physical and (status == "baseline" or improves):
            archived = self._retain_prefix(args, result, prefix, scope)
            if archived and archived.get("path"):
                state["retained_prefix"] = archived
            state["retention_attempt"] = archived
        state["latest"] = candidate
        report.update(status=status, best=state["best"], latest=candidate,
                      attempts_since_supported_improvement=state["since_improvement"],
                      retained_prefix=state.get("retained_prefix"), retention_attempt=state.get("retention_attempt"))
        if status == "baseline" and len(self.local_scopes) > 1:
            report["reason"] = "new comparison scope; no improvement over other plans/physics is established"
        if state["since_improvement"] >= 2:
            report["next_step"] = "Compare best and latest measurements and retained proposals; use library skills to test another hypothesis or revise earlier choices. Exploratory worse steps remain allowed."
        return report

    def execution_key(self, args):
        try:
            scene = Path(args["scene"])
            scene = scene if scene.is_absolute() else self.ws / scene
            if not scene.is_file():
                return None
            payload = {k: v for k, v in args.items() if k != "out_dir"}
            payload = _bound(payload, self.ws)
            if args.get("resume") and args.get("out_dir"):
                out = Path(args["out_dir"])
                out = out if out.is_absolute() else self.ws / out
                # These inputs can affect a resumed run. Diagnostic logs, video
                # files, timestamps and the new search ledger cannot.
                cache = {}
                outcomes = out / "outcomes.json"
                prior = json.loads(outcomes.read_text()) if outcomes.is_file() else []
                if not isinstance(prior, list) or any(not isinstance(entry, dict) for entry in prior):
                    return None
                for entry in prior:
                    if not entry.get("ok") or entry.get("dry_run"):
                        break
                    tag = entry.get("tag")
                    if not tag or not entry.get("resume_signature"):
                        return None  # legacy cache identity is not established
                    folder = out / tag
                    cache[tag] = {"resume_signature": entry["resume_signature"],
                                  "index": entry.get("index"), "motion": entry.get("motion")}
                    for name in ("rollout.npz", "rollout_runtime.npz", "state.json", "phases.json", "pose_used.json"):
                        p = folder / name
                        if p.is_file():
                            cache[str(p.relative_to(out))] = hashlib.sha256(p.read_bytes()).hexdigest()
                payload["resume_checkpoint_contents"] = cache
            return _digest(payload)
        except Exception:
            # This is a compute-saving guard, not a new failure gate. Let the
            # actual tool diagnose inputs whose identity cannot be established.
            return None

    @_diagnostic_only
    def before_execution(self, key):
        previous = self.failed_inputs.get(key) if key else None
        if not previous or previous["count"] < 2:
            return None
        note = {"ok": False, "error": "unchanged_failed_execution",
                "execution_performed": False, "identical_failed_trials": previous["count"],
                "previous_failure": previous["failure"],
                "next_steps": [
                    "Read the measured failure and compare it with the recorded attempts.",
                    "Use library_search and an appropriate measurement/search skill to test a different repair hypothesis.",
                    "Change the relevant pose, path, physical prefix or execution parameters before retrying.",
                    "Revisit earlier choices or create and test a reusable skill if existing search is insufficient."],
                "scope": "duplicate failed-input guard; no claim of task infeasibility or convergence",
                "progress_file": str(self.path)}
        self.events.append({"event": "duplicate_execution_suppressed", "input_key": key})
        if not self.write():
            return None
        return note

    @_diagnostic_only
    def observe_execution(self, key, args, raw):
        result = raw.get("result") if raw.get("ok") else None
        if not isinstance(result, dict):
            event = {"event": "execution_error", "input_key": key, "task_accepted": False,
                     "error": raw.get("error", "executor did not return a structured result"),
                     "comparison": {"status": "unassessed", "reason": "tool error is not physical infeasibility or progress"}}
            self.latest_execution = event
            self.events.append(event)
            return event if self.write() else None
        physical = not args.get("dry_run") and not result.get("dry_run") and bool(result.get("n_frames"))
        prefix = 0
        for entry in result.get("subgoals", []):
            if entry.get("ok") is not True:
                break
            prefix += 1
        if physical:
            self.best_physical_prefix = max(self.best_physical_prefix, prefix)
        failure = result.get("first_failure") or next((v for v in result.get("subgoals", []) if v.get("ok") is False), {})
        identity = {k: failure.get(k) for k in ("index", "step", "motion", "side", "object", "failed")}
        category = _digest(identity)
        if result.get("ok") is False:
            previous = self.signatures.get(category, {})
            count = previous.get("count", 0) + 1
            measured = _digest(failure.get("measured", {}), diagnostic=True)
            changed = bool(previous) and measured != previous.get("measured")
            self.signatures[category] = {"count": count, "measured": measured, "failure": identity}
            if key:
                self.failed_inputs[key] = {"count": self.failed_inputs.get(key, {}).get("count", 0) + 1,
                                           "failure": identity}
        else:
            count, changed = 0, False
        event = {"event": "execution", "input_key": key, "physical_recording": physical,
                 "passed_prefix_entries": prefix, "best_physical_prefix_entries": self.best_physical_prefix,
                 "prefix_scope": "diagnostic entry count; plan subdivision is not semantic progress or convergence",
                 "local_execution_ok": result.get("ok") is True, "task_accepted": False,
                 "failure": identity if result.get("ok") is False else None,
                 "same_failure_occurrences": count, "measurements_changed": changed}
        event["comparison"] = self._compare_execution(args, result, prefix, failure, physical)
        event["latest_final_gate"] = self.latest_gate
        event["final_gate_association"] = "historical last gate; this execution still needs fresh final verification"
        event["failure_recurrence_scope"] = "historical same symptoms; comparison field determines measured improvement"
        if count >= 3:
            event["next_step"] = ("This failure recurred. Compare measurements across the distinct attempts; use a different "
                "skill/search hypothesis or revise an earlier choice. Different numeric arguments alone are not evidence "
                "of improvement. Local search failure does not prove the task impossible.")
        if result.get("ok") is True:
            event["next_step"] = "Check original semantic goals and complete independent gates; local execution alone is not task acceptance."
        self.latest_execution = {k: event[k] for k in ("input_key", "local_execution_ok", "failure", "comparison")}
        self.events.append(event)
        if not self.write():
            return None
        return event

    @_diagnostic_only
    def observe_gate(self, verdict):
        if not verdict.get("ok"):
            self.gate_failures += 1
        event = {"event": "final_gates", "ok": verdict.get("ok") is True,
                 "failed_attempts": self.gate_failures, "issues": list(verdict.get("issues") or [])[:12]}
        suite = (verdict.get("reports") or {}).get("stage3_gate_suite") or {}
        event["semantic_comparison"] = self._compare_semantic(suite)
        event["latest_local_execution_ok"] = (self.latest_execution or {}).get("local_execution_ok")
        event["task_accepted"] = verdict.get("ok") is True
        if event["latest_local_execution_ok"] and not event["task_accepted"]:
            event["next_step"] = "Local execution passed but final task gates failed. Repair the latest independent failed checks; a retained prefix is not an accepted task."
        self.latest_gate = {k: event[k] for k in ("ok", "issues", "semantic_comparison", "task_accepted")}
        self.events.append(event)
        return event if self.write() else None

    def _compare_semantic(self, suite):
        report = suite.get("task_goal_report") or {}
        rows = report.get("subgoals") or []
        binding = suite.get("comparison_scope")
        if not binding or not rows:
            return {"status": "unassessed", "reason": "independent goal measurements or fixed scope unavailable"}
        definitions, passed, failed, unsupported = [], [], [], []
        for row in [*rows, dict(report.get("task_success") or {}, index="final_task")]:
            key = str(row.get("index"))
            predicates = row.get("predicates") or []
            definitions.append({"index": key, "goal": row.get("goal"), "required_classes": row.get("required_classes"),
                                "predicates": [{k: p.get(k) for k in ("class", "A", "B", "joint", "source")}
                                               for p in predicates]})
            supported = bool(predicates) and all(isinstance(p.get("ok"), bool) and p.get("class") for p in predicates)
            if not supported:
                unsupported.append(key)
            elif row.get("ok") is True and all(p["ok"] for p in predicates):
                passed.append(key)
            else:
                failed.append(key)
        scope = _digest({"contract": self.task_scope, "gate_scope": binding, "goals": definitions})
        candidate = {"passed_goals": passed, "failed_goals": failed, "unsupported_goals": unsupported,
                     "candidate_binding": suite.get("candidate_binding"), "diagnostic_report": suite.get("diagnostic_report")}
        state = self.semantic_scopes.get(scope)
        if unsupported:
            status = "unsupported_goal_measurements"
        elif state is None:
            status = "baseline"
        else:
            current, best = set(passed), set(state["best"]["passed_goals"])
            status = ("improved_goals" if current > best else "regressed_goals" if current < best else
                      "unchanged" if current == best else "mixed_goals")
        if state is None:
            state = {"best": candidate, "since_improvement": 0}
            self.semantic_scopes[scope] = state
        elif status == "improved_goals":
            state.update(best=candidate, since_improvement=0)
        else:
            state["since_improvement"] += 1
        state["latest"] = candidate
        return {"status": status, "scope": scope, "latest": candidate, "best": state["best"],
                "attempts_since_supported_improvement": state["since_improvement"],
                "scope_note": "Independent original-goal evidence only; no convergence guarantee or acceptance authority."}

    @_diagnostic_only
    def write(self):
        from agent.external import atomic_json
        atomic_json(self.path, {"schema": "dexagent.search_progress.v2", "authority": "diagnostic_only",
            "convergence_guaranteed": False, "best_physical_prefix_entries": self.best_physical_prefix,
            "prefix_scope": "diagnostic entry count; plan subdivision is not semantic progress or convergence",
            "gate_failures": self.gate_failures, "latest_execution": self.latest_execution,
            "latest_final_gate": self.latest_gate, "local_comparisons": self.local_scopes,
            "semantic_comparisons": self.semantic_scopes, "events": self.events})
        return True
