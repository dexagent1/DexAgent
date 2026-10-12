"""The generation-verification loop of one stage."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from agent import repair
from agent.budget import BUDGET_WARN
from agent.console import Console
from agent.devtools import BUILTIN_SCHEMAS, DevTools
from agent.library import call_tool, configured_args, find_tool, library_root, load_index, tool_catalog, tool_schema
from agent.llm import image_block

PROMPTS = Path(__file__).resolve().parent / "prompts"
MAX_RESULT_CHARS = 16000
BUDGET_MARKS = (0.5, 0.75, 0.9)                      # shares of the round budget at which the agent is told what is left
STAGE3_NO_TOOL_LIMIT = 3
WALL_GRACE_ROUNDS = 3


def system_prompt(stage_prompt: str, extra: str = "") -> str:
    parts = [(PROMPTS / "system.md").read_text(), (PROMPTS / "building_tools.md").read_text(), stage_prompt]
    if extra:
        parts.append(extra)
    return "\n\n".join(p.strip() for p in parts)


class StageRunner:
    def __init__(self, cfg: dict, llm, workspace: Path, stage_prompt: str, verifiers: list, categories=None,
                 tool_types=("skill", "verifier"), max_rounds: int | None = None, log=print, stage: str | None = None, console: Console | None = None,
                 verdict_hook=None, finalize=None, recovery: dict | None = None, result_hooks: dict | None = None, after_search=None):
        self.cfg, self.llm, self.ws = cfg, llm, Path(workspace)
        self.ws.mkdir(parents=True, exist_ok=True)
        self.stage = stage or self.ws.name
        if recovery is not None and self.stage != "stage3":
            raise ValueError("candidate recovery is supported only for Stage 3")
        self.recovery = dict(recovery) if recovery is not None else None
        from agent.search_progress import SearchProgress
        self.search_progress = SearchProgress(self.ws) if self.stage == "stage3" else None
        if self.stage == "stage3":
            # Every new model run starts with an empty in-memory authority set.
            # Configuration, traces and report files cannot import old receipts.
            self.cfg["_executor_receipts"] = {}
        self.lib = library_root(cfg)
        self.index = load_index(self.lib)
        self.dev = DevTools(self.lib, self.ws, cfg)
        self.verifiers = verifiers                      # callables: (workspace) -> {"ok": bool, "issues": [...]}
        self.verdict_hook = verdict_hook                # (workspace, verdict) -> text appended to a failed verdict (Stage 3: the diagnosis)
        self.finalize = finalize                        # trusted publication step, invoked only after every gate passes
        # tool name -> (workspace, args, raw result) -> {"text": appended to that tool's result, "stop": reason or None}
        # (Stage 3: every physical execution is scored against the Stage 1 goals and the search's best so far)
        self.result_hooks = dict(result_hooks or {})
        # (runner, out) -> record or None: what the stage does when the search ends unfinished (Stage 3: the method
        # modules); a record with ok true carries the verdict of a published deliverable
        self.after_search = after_search
        # a wall-clock budget besides the round budget: past it the agent gets WALL_GRACE_ROUNDS more rounds to finish
        wall = ((cfg.get(self.stage) or {}).get("max_wall_minutes") if isinstance(cfg.get(self.stage), dict) else None)
        self.max_wall_s = float(wall) * 60. if wall else None
        self.last_diagnosis = None
        self.latest_executor_evidence = None
        self.categories, self.tool_types = categories, tool_types
        self.dev.preferred_categories = categories
        self.tool_discovery = (cfg.get("agent") or {}).get("tool_discovery", "search")
        if self.tool_discovery not in ("search", "all"):
            raise ValueError("agent.tool_discovery must be 'search' or 'all'")
        self.active_tools = {}
        catalog = {t["name"] for t in tool_catalog(self.index, self.categories, self.tool_types)}
        for name in ("probe_video", "make_contact_sheet", "extract_frames",
                     "scene_semantic_schema_check", "scene_semantic_consistency_check"):
            if name in catalog:
                self.active_tools[name] = None
        self.max_rounds = int(max_rounds or cfg.get("agent", {}).get("max_rounds", 40))
        # the run's money cap (agent.budget_usd) for a priced model API; the external transport and mocks are capped by rounds
        from agent.budget import Budget
        self.budget = Budget.for_run(cfg) if getattr(llm, "priced", False) else None
        self.system = system_prompt(stage_prompt)
        self.trace = self.ws / "trace.jsonl"
        self.log = log
        self.console = console or Console(self.stage, str((cfg.get("run") or {}).get("name", self.ws.parent.name)), self.ws, self.max_rounds)
        self.console.budget = self.budget
        self.console.kinds.update({t["name"]: t.get("type", "skill") for t in self.index["tools"]})
        self.console.kinds.update({sch["name"]: "harness" for sch in BUILTIN_SCHEMAS})
        from agent.integrity import InputContract
        self.contract = InputContract(self.lib, cfg.get("_verification_inputs", []))
        self.contract.write(self.ws / "input_contract.json")
        if self.search_progress is not None:
            self.search_progress.bind_contract(self.contract)
        self.reference_evidence = None
        if self.stage == "stage2" and (cfg.get("inputs") or {}).get("video"):
            from agent.reference_evidence import begin
            self.reference_evidence = begin(self.ws, cfg["inputs"]["video"])
        if self.recovery is not None:
            from agent.external import atomic_json
            from tool_library.common.evidence import file_digest
            self.recovery["current_input_contract_sha256"] = file_digest(self.ws / "input_contract.json")
            atomic_json(self.ws / "recovery.json", self.recovery)

    # ------------------------------------------------------------ tools
    def tools(self) -> list[dict]:
        self.index = self.dev.index
        active = [t for t in self.index["tools"] if t["name"] in self.active_tools]
        if self.tool_discovery == "all":           # the stage's categories, plus what was searched, read or created outside them
            catalog = tool_catalog(self.index, self.categories, self.tool_types)
            catalog += [t for t in active if t["name"] not in {c["name"] for c in catalog}]
        else:
            catalog = active
        lib_tools = [tool_schema(t, self.cfg) for t in catalog]
        return BUILTIN_SCHEMAS + lib_tools

    def _activate_tools(self, names):
        """Expose requested capabilities, with a bounded recent schema window."""
        for name in names:
            self.active_tools.pop(name, None)
            self.active_tools[name] = None
        while len(self.active_tools) > 64:
            self.active_tools.pop(next(iter(self.active_tools)))

    def _cooperative_search_args(self, name, args):
        """Make direct Stage 3 searches stoppable without changing their budget.

        Calls imported inside run_python bypass this dispatcher and must supply
        the tool's checkpoint arguments explicitly. Older library schemas stay
        callable without unsupported keyword injection.
        """
        if (self.stage != "stage3" or name != "optimize_grasp_pose"
                or (self.cfg.get("agent") or {}).get("cooperative_search_checkpoints", True) is False):
            return args, None
        tool = find_tool(self.index, name)
        props = tool_schema(tool, self.cfg)["input_schema"]["properties"] if tool else {}
        if "checkpoint_json" not in props:
            return args, None
        filled = configured_args(self.cfg, tool)
        effective = {**filled, **{k: v for k, v in args.items() if not (v is None and k in filled)}}
        # Never guess which journal a caller intended to resume. In particular,
        # keep resume=True without a path as the tool's normal validation error.
        if "checkpoint_json" not in effective and not effective.get("resume"):
            effective["checkpoint_json"] = str((self.ws / ".search_checkpoints" / (uuid.uuid4().hex + ".json")).resolve())
        if effective.get("checkpoint_json") is not None and "stop_file" in props and "stop_file" not in effective:
            effective["stop_file"] = str((self.ws / ".stop_search").resolve())

        def absolute(value):
            if not isinstance(value, (str, Path)):
                return value  # leave malformed arguments to ordinary tool validation
            try:
                path = Path(value)
                return str((path if path.is_absolute() else self.ws / path).resolve())
            except (OSError, ValueError):
                return str(value)

        info = {"checkpoint_json": absolute(effective.get("checkpoint_json")),
                "stop_file": absolute(effective.get("stop_file")), "resume": bool(effective.get("resume")),
                "scope": "direct tool call; run_python imports require explicit checkpoint arguments",
                "authority": "unverified search progress; no acceptance or executor receipts restored"}
        return effective, info

    def run_tool(self, name: str, args: dict) -> tuple[list[dict], dict]:
        """Returns (content blocks for the model, raw record for the trace)."""
        if name == "look":
            r = self.dev.look(**args)
            blocks = []
            for im in r["images"]:
                if "error" in im:
                    blocks.append({"type": "text", "text": f"{im['path']}: {im['error']}"})
                else:
                    blocks.append({"type": "text", "text": im["path"]})
                    blocks.append(image_block(im["path"]))
            return blocks, {"images": [im["path"] for im in r["images"]]}
        if hasattr(self.dev, name) and name in {s["name"] for s in BUILTIN_SCHEMAS}:
            try:
                r = getattr(self.dev, name)(**args)
                if name == "library_search":
                    self._activate_tools(t["name"] for t in r.get("tools", []))
                elif name == "library_read" and not r.get("error"):
                    self._activate_tools([args["name"]])
                elif name == "library_create" and r.get("ok"):
                    self._activate_tools([r["created"]["name"]])
                self.index = self.dev.index
            except Exception as e:
                r = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                if isinstance(e, TypeError):
                    r["input_schema"] = next(s["input_schema"] for s in BUILTIN_SCHEMAS if s["name"] == name)
            return [{"type": "text", "text": _clip(json.dumps(r, default=str, indent=1))}], r
        requested_args = args
        args, search_checkpoint = self._cooperative_search_args(name, args)
        if search_checkpoint is not None:
            self._trace({"event": "tool_dispatch", "name": name, "args": requested_args,
                         "effective_args": args, "search_checkpoint": search_checkpoint})
        search_key = None
        if name == "execute_subgoal_plan" and self.search_progress is not None:
            search_key = self.search_progress.execution_key(args)
            duplicate = self.search_progress.before_execution(search_key)
            if duplicate is not None:
                return [{"type": "text", "text": json.dumps(duplicate, indent=1)}], duplicate
        reference_token = None
        if self.reference_evidence is not None:
            try:
                reference_token = self.reference_evidence.prepare(name, args)
                if reference_token is not None and not self.contract.check()["ok"]:
                    raise ValueError("trusted implementation or inputs changed before source tool dispatch")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                r = {"ok": False, "error": f"source-reference evidence: {exc}"}
                return [{"type": "text", "text": json.dumps(r)}], r
        # Generated code must not share the process-local acceptance authorities.
        # Baseline library code is frozen by InputContract; new tools run in a
        # child process just like run_python and cannot mint harness receipts.
        r = call_tool(self.lib, self.index, name, args, cfg=self.cfg, cwd=self.ws,
                      **({"force_subprocess": True} if name not in self.contract.tools else {}))
        if search_checkpoint is not None:
            r = dict(r, effective_args=args, search_checkpoint=search_checkpoint)
        if reference_token is not None:
            try:
                if not self.contract.check()["ok"]:
                    raise ValueError("trusted implementation or inputs changed")
                self.reference_evidence.complete(reference_token, r)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                r = {"ok": False, "error": f"reference evidence was not certified: {exc}", "tool_result": r}
        self._activate_tools([name])
        if (name == "execute_subgoal_plan" and r.get("ok") is True and isinstance(r.get("result"), dict)
                and r["result"].get("ok") is True and not args.get("dry_run")
                and not r["result"].get("dry_run") and r["result"].get("trajectory_sha256")):
            report = (r["result"].get("files") or {}).get("execution_report")
            # Kept in harness memory, never reconstructed from generator-written
            # manifests or traces. The independent gate also replays the controls.
            if report and self.contract.check()["ok"]:
                from tool_library.common.evidence import file_digest
                path = Path(report)
                path = path if path.is_absolute() else self.ws / path
                self.cfg.setdefault("_executor_receipts", {})[file_digest(path)] = str(path.resolve())
        self.index = self.dev.index                      # a created tool becomes callable at once
        if name == "execute_subgoal_plan":
            result = r.get("result") if r.get("ok") and isinstance(r.get("result"), dict) else None
            if result is not None:
                self.latest_executor_evidence = {k: result.get(k) for k in
                    ("ok", "first_failure", "files", "dry_run", "n_frames")}
            progress = self.search_progress.observe_execution(search_key, args, r) if self.search_progress else None
            diagnosis = ""
            if result is None or result.get("ok") is False:
                diagnosis = self._diagnose({"ok": False, "issues": [r.get("error")] if result is None else [], "reports": {},
                                           "executor_result": result, "executor_args": args})
                self._trace({"event": "executor_diagnosis", "diagnosis": diagnosis,
                             "execution_report": ((result or {}).get("files") or {}).get("execution_report")})
            limit = MAX_RESULT_CHARS - min(len(diagnosis), 6000)
            blocks = [{"type": "text", "text": _executor_feedback(r, limit) if result is not None else _clip(json.dumps(r, default=str, indent=1), limit)}]
            if diagnosis:
                blocks.append({"type": "text", "text": _clip(diagnosis, 6000)})
            if progress:
                blocks.append({"type": "text", "text": "Search progress (diagnostic only):\n" + json.dumps(progress, indent=1)})
            return blocks, r
        blocks = [{"type": "text", "text": _clip(json.dumps(r, default=str, indent=1))}]
        if search_checkpoint is not None:
            blocks.append({"type": "text", "text": "Cooperative search checkpoint (diagnostic only):\n" + json.dumps(search_checkpoint, indent=1)})
        return blocks, r

    # ------------------------------------------------------------ loop
    def _operator_checkpoint(self):
        """A neutral operator stop is administrative, never an acceptance input."""
        locations = [self.ws]
        transport = getattr(self.llm, "directory", None) or (self.cfg.get("agent") or {}).get("transport_dir")
        if transport:
            locations.insert(0, Path(transport))
        for directory in locations:
            path = Path(directory) / "operator_checkpoint.json"
            try:
                value = json.loads(path.read_text())
                if not isinstance(value, dict):
                    continue
                explicit = value.get("kind") == "administrative_checkpoint"
                legacy = str(value.get("reason", "")).lower().startswith("administrative") and bool(str(value.get("instruction", "")).strip())
                if explicit or legacy:
                    return {"path": str(path), "authority": "administrative stop only; no acceptance authority"}
            except (OSError, ValueError):
                continue
        return None

    def run(self, user_message: str) -> dict:
        messages = [{"role": "user", "content": [{"type": "text", "text": user_message}]}]
        self._trace({"event": "start", "system_chars": len(self.system), "user": user_message,
                     **({"recovery": self.recovery} if self.recovery is not None else {})})
        self.console.banner()
        rounds, finished, verdict, blocked, upstream = 0, False, None, None, None
        stopped_search = None
        no_tool_turns, backend_stop = 0, None
        search_stop = None
        wall_note, wall_last_round = None, None
        budget_noted = False
        while rounds < self.max_rounds:
            rounds += 1
            if self.budget is not None and self.budget.exceeded():
                search_stop = {"stop_reason": "budget_cap", "remaining_rounds": self.max_rounds - rounds + 1,
                               "spent_usd": round(self.budget.spent_usd, 4), "budget_usd": self.budget.cap_usd,
                               "reason": f"the run's spending cap (agent.budget_usd = ${self.budget.cap_usd:g}) is spent "
                                         f"(${self.budget.spent_usd:.2f}, {self.budget.path}); candidate remains unfinished, task "
                                         f"feasibility is not established. To continue, raise agent.budget_usd"}
                self._trace({"event": "budget_stop", "round": rounds, **search_stop})
                rounds -= 1                                 # no model call was made in this round
                break
            self.console.round_start(rounds)
            if self.budget is not None and not budget_noted and self.budget.fraction() >= BUDGET_WARN:
                budget_noted = True
                note = (f"[harness] The run's spending cap is {100 * self.budget.fraction():.0f}% spent (${self.budget.spent_usd:.2f} of "
                        f"${self.budget.cap_usd:g}); the stage stops when it is spent. Call finish on your best verified deliverable, "
                        f"or report_blocked with the measured obstruction.")
                messages[-1]["content"] = list(messages[-1]["content"]) + [{"type": "text", "text": note}]
                self._trace({"event": "budget_warning", "round": rounds, "spent_usd": round(self.budget.spent_usd, 4)})
            if self.max_wall_s and wall_last_round is None and time.time() - self.console.t0 > self.max_wall_s:
                wall_last_round = rounds + WALL_GRACE_ROUNDS - 1      # this round and the next ones: WALL_GRACE_ROUNDS model calls
                wall_note = (f"[harness] The stage's wall-clock budget ({self.max_wall_s / 60:.0f} min) is spent. You have "
                             f"{WALL_GRACE_ROUNDS} more rounds: call finish on your best verified deliverable, or report_blocked "
                             f"with the measured obstruction.")
                messages[-1]["content"] = list(messages[-1]["content"]) + [{"type": "text", "text": wall_note}]
                self._trace({"event": "wall_budget_spent", "round": rounds, "elapsed_s": round(time.time() - self.console.t0, 1)})
            if wall_last_round is not None and rounds > wall_last_round:
                search_stop = {"stop_reason": "wall_time_budget", "remaining_rounds": self.max_rounds - rounds + 1,
                               "reason": "wall-clock budget spent; candidate remains unfinished, task feasibility is not established"}
                self._trace({"event": "wall_time_stop", "round": rounds, **search_stop})
                break
            tools = self.tools()
            t0 = time.time()
            try:
                content = self.llm.complete(self.system, messages, tools)
            except Exception as exc:
                blocked = f"model backend failed: {type(exc).__name__}: {exc}"
                self._trace({"event": "backend_error", "round": rounds, "error": blocked})
                break
            usd = self.budget.charge(getattr(self.llm, "last_usage", None), self.stage) if self.budget is not None else None
            self._trace({"event": "assistant", "round": rounds, "content": content, "seconds": round(time.time() - t0, 1),
                         **({"usage": getattr(self.llm, "last_usage", None), "usd": round(usd, 5)} if usd is not None else {})})
            messages.append({"role": "assistant", "content": content})
            calls = [c for c in content if c["type"] == "tool_use"]
            for c in content:
                if c["type"] == "text" and c["text"].strip():
                    self.console.thought(c["text"])
            if not calls:
                if self.stage == "stage3":
                    no_tool_turns += 1
                    guidance = (
                        "[harness] No tool action was submitted. Tools remain available under their advertised names. "
                        "Submit an actual tool call: use library_search to discover skills, call an advertised tool, "
                        "or explicitly call finish when ready for verification. After a failed finish, report_blocked "
                        "is available if you cannot continue. A text reply does not execute a tool or request verification. "
                        f"Consecutive replies without tool actions: {no_tool_turns}/{STAGE3_NO_TOOL_LIMIT}; "
                        "at the limit the harness saves the unfinished candidate and ends this backend run normally.")
                    # There was no assistant tool_use; use ordinary user text,
                    # never fabricate an orphan tool_result or an implicit finish.
                    messages.append({"role": "user", "content": [{"type": "text", "text": guidance}]})
                    self._trace({"event": "backend_no_tool_action", "round": rounds,
                                 "consecutive_turns": no_tool_turns, "guidance": guidance})
                    if no_tool_turns >= STAGE3_NO_TOOL_LIMIT:
                        from agent.external import atomic_json
                        backend_stop = {
                            "path": str(self.ws / "backend_checkpoint.json"), "workspace": str(self.ws),
                            "trace": str(self.trace), "round": rounds, "consecutive_no_tool_turns": no_tool_turns,
                            "available_tool_names": [tool["name"] for tool in tools],
                            "authority": "unverified candidate and diagnostic history; no acceptance authority",
                            "next_step": "Resume the candidate with a working backend; inspect the currently advertised tools and reverify all acceptance evidence."}
                        atomic_json(backend_stop["path"], backend_stop)
                        break
                    continue
                # a reply without tool calls is treated as a finish request
                calls = [{"type": "tool_use", "id": f"auto_finish_{rounds}", "name": "finish", "input": {"summary": "(no tool calls)"}}]
            no_tool_turns = 0
            results = []
            for c in calls:
                name, args = c["name"], c.get("input") or {}
                if name == "finish":
                    verdict = self._verify()
                    progress = None
                    if self.search_progress:
                        progress = self.search_progress.observe_gate(verdict)
                    if verdict["ok"]:
                        finished = True
                        results.append({"type": "tool_result", "tool_use_id": c["id"], "content": [{"type": "text", "text": "All verifiers passed. Stage complete."}]})
                    else:
                        from agent.verification_feedback import verification_feedback
                        diag = self._diagnose(verdict)
                        txt = "Verifiers FAILED. Fix the deliverable and call finish again.\n" + verification_feedback(verdict, limit=MAX_RESULT_CHARS - min(len(diag), 6000) - 100)
                        if diag:
                            # the next steps go after the verdict and survive the clipping of a long one
                            txt = _clip(txt, MAX_RESULT_CHARS - min(len(diag), 6000) - 4) + "\n\n" + diag[:6000]
                        blocks = [{"type": "text", "text": _clip(txt)}]
                        if progress:
                            blocks.append({"type": "text", "text": "Search progress (diagnostic only):\n" + json.dumps(progress, indent=1)})
                        results.append({"type": "tool_result", "tool_use_id": c["id"], "content": blocks})
                    self._trace({"event": "verify", "round": rounds, "verdict": verdict, "diagnosis": self.last_diagnosis if not verdict["ok"] else None})
                    self.console.verdict(verdict)
                    if finished:
                        break  # a verified artifact cannot be changed by later calls in this turn
                    continue
                if name == "report_blocked":
                    operator = self._operator_checkpoint() if self.stage == "stage3" else None
                    if operator:
                        search_stop = {"stop_reason": "operator_checkpoint", "operator_checkpoint": operator,
                                       "reason": "administrative operator checkpoint; candidate remains unfinished"}
                        results.append({"type": "tool_result", "tool_use_id": c["id"], "content": [{"type": "text", "text": search_stop["reason"]}]})
                        self._trace({"event": "operator_checkpoint", "round": rounds, **search_stop})
                        break
                    if verdict is None or verdict.get("ok"):
                        results.append({"type": "tool_result", "tool_use_id": c["id"], "content": [{"type": "text", "text":
                                        "report_blocked is accepted only after finish has returned verifier failures: call finish first."}]})
                        continue
                    up = args.get("upstream") or None
                    bad = repair.valid_upstream(self.stage, up)
                    if bad is None and up and not str(args.get("evidence") or "").strip():
                        bad = "upstream needs evidence: what you measured or saw that shows the earlier stage is wrong"
                    if bad:
                        results.append({"type": "tool_result", "tool_use_id": c["id"], "content": [{"type": "text", "text":
                                        f"report_blocked refused: {bad}."}]})
                        continue
                    blocked = str(args.get("reason") or "")[:4000]
                    if up:
                        upstream = {"stage": up, "evidence": str(args.get("evidence"))[:4000]}
                    results.append({"type": "tool_result", "tool_use_id": c["id"], "content": [{"type": "text", "text":
                                    "Stage ended as blocked." + (f" The run re-opens {up} with your reason (once per run)." if up else "")}]})
                    self._trace({"event": "blocked", "round": rounds, "reason": blocked, "upstream": upstream})
                    break
                self.console.tool_call(name, args)
                blocks, raw = self.run_tool(name, args)
                self._trace({"event": "tool", "round": rounds, "name": name, "args": args, "result": raw,
                             **({"effective_args": raw["effective_args"]} if "effective_args" in raw else {})})
                self.console.tool_result(name, raw)
                hook = self.result_hooks.get(name)
                if hook is not None:
                    try:
                        extra = hook(self.ws, args, raw) or {}
                    except Exception as exc:              # a feedback hook never breaks the run
                        extra = {"text": f"(result hook failed: {type(exc).__name__}: {exc})"}
                    if extra.get("text"):
                        blocks = list(blocks) + [{"type": "text", "text": _clip(extra["text"], 6000)}]
                        self.console.thought("[harness] " + extra["text"].splitlines()[0][:200])
                    self._trace({"event": "result_hook", "round": rounds, "name": name, **{k: v for k, v in extra.items() if k != "text"},
                                 "text": extra.get("text")})
                    if extra.get("stop") and search_stop is None:
                        search_stop = {"stop_reason": extra["stop"], "remaining_rounds": self.max_rounds - rounds,
                                       "reason": extra.get("stop_reason_text") or "search stopped by the harness; candidate remains unfinished"}
                results.append({"type": "tool_result", "tool_use_id": c["id"], "content": blocks})
                if search_stop is not None:
                    break
                result = raw.get("result")
                if (self.stage == "stage3" and name == "optimize_grasp_pose" and raw.get("ok") is True
                        and isinstance(result, dict) and result.get("ok") is False
                        and result.get("status") == "incomplete" and result.get("reason") == "cooperative_stop"):
                    stopped_search = {**(raw.get("search_checkpoint") or {}),
                                      "checkpoint_json": result.get("checkpoint_json"),
                                      "stop_file": result.get("stop_file"),
                                      "effective_args": raw.get("effective_args", args),
                                      "n_evals": result.get("n_evals"), "authority": "unverified search progress"}
                    self._trace({"event": "cooperative_search_stop", "round": rounds, "checkpoint": stopped_search})
                    break
            note = self._budget_note(rounds)
            if note and results and not (finished or blocked or stopped_search is not None or search_stop is not None):
                results[-1]["content"] = list(results[-1]["content"]) + [{"type": "text", "text": note}]
            messages.append({"role": "user", "content": results})
            if finished or blocked or stopped_search is not None or search_stop is not None:
                break
        out = {"finished": finished, "rounds": rounds, "verdict": verdict, "new_tools": self.dev.created, "workspace": str(self.ws),
               "tool_calls": self.console.n_calls, "tools_used": dict(self.console.calls), "elapsed_s": round(time.time() - self.console.t0, 1), "stage": self.stage}
        if self.budget is not None:
            out["spent_usd"] = round(float(self.budget.state["by_stage"].get(self.stage, {}).get("spent_usd", 0.0)), 4)
            out["run_spent_usd"], out["budget_usd"] = round(self.budget.spent_usd, 4), self.budget.cap_usd
        if self.recovery is not None:
            out["recovery"] = self.recovery
        if upstream:
            out["upstream"] = upstream
        if stopped_search is not None:
            out.update(stopped=True, stop_reason="cooperative_search_checkpoint", search_checkpoint=stopped_search)
        if backend_stop is not None:
            out.update(stopped=True, stop_reason="backend_no_tool_progress", backend_checkpoint=backend_stop)
        if search_stop is not None:
            from agent.external import atomic_json
            checkpoint = dict(search_stop, workspace=str(self.ws), trace=str(self.trace),
                              authority="unverified candidate and diagnostic history; no acceptance authority")
            checkpoint["path"] = str(self.ws / "search_stalled_checkpoint.json")
            atomic_json(checkpoint["path"], checkpoint)
            out.update(stopped=True, stop_reason=search_stop["stop_reason"], search_checkpoint=checkpoint)
        if not finished:
            # the budget ran out (or the agent reported the stage blocked) without a deliverable that passes the verifiers:
            # the stage FAILED. Nothing it left in the workspace counts as a result; the report says why, so the run can be
            # continued, an earlier stage re-opened (upstream), or the input changed.
            out["failure"] = (search_stop["reason"] if search_stop is not None else
                              "backend submitted repeated replies without tool actions; unfinished candidate saved, task feasibility is unassessed" if backend_stop is not None else
                              "cooperative search stopped at a saved checkpoint; task feasibility is unassessed" if stopped_search is not None else
                              f"blocked: {blocked}" if blocked else
                              "budget exhausted without passing the stage verifiers" if verdict else "budget exhausted before finish was called")
            rep = {"stage": self.stage, "rounds": rounds, "max_rounds": self.max_rounds, "tool_calls": self.console.n_calls,
                   "reason": out["failure"], "last_verdict": verdict, "new_tools": self.dev.created}
            if upstream:
                rep["upstream"] = upstream["stage"]
                rep["upstream_evidence"] = upstream["evidence"]
            if (self.last_diagnosis and stopped_search is None and backend_stop is None
                    and (search_stop is None or search_stop["stop_reason"] == "search_stalled")):
                rep["diagnosis"] = self.last_diagnosis
            if stopped_search is not None:
                rep.update(stopped=True, stop_reason=out["stop_reason"], search_checkpoint=stopped_search)
            if backend_stop is not None:
                rep.update(stopped=True, stop_reason=out["stop_reason"], backend_checkpoint=backend_stop)
            if search_stop is not None:
                rep.update(stopped=True, stop_reason=out["stop_reason"], search_checkpoint=out["search_checkpoint"])
            (self.ws / "failure_report.json").write_text(json.dumps(rep, indent=1, default=str))
        if not finished and self.after_search is not None:
            out = self._after_search(out)
        self._trace({"event": "end", **out})
        self.console.done(out)
        if hasattr(self.llm, "close"):
            self.llm.close(out)
        return out

    def _after_search(self, out: dict) -> dict:
        """The after_search hook on an unfinished result: its record goes into the result; when it published a
        deliverable the stage is finished and the agent's own failure report is kept aside as failure_report_agent.json."""
        try:
            record = self.after_search(self, out)
        except Exception as exc:                          # the hook failing is a reason, not a crash of the run
            record = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if record is None:
            return out
        out["methods"] = record
        out.update(tool_calls=self.console.n_calls, tools_used=dict(self.console.calls), elapsed_s=round(time.time() - self.console.t0, 1))
        report = self.ws / "failure_report.json"
        if record.get("ok"):
            agent = {k: out.pop(k) for k in ("failure", "stopped", "stop_reason", "search_checkpoint") if k in out}
            out.update(finished=True, verdict=record.get("verdict"), agent_search=agent)
            if report.exists():
                report.replace(self.ws / "failure_report_agent.json")
        elif report.exists():
            rep = json.loads(report.read_text())
            rep["methods"] = record
            report.write_text(json.dumps(rep, indent=1, default=str))
        return out

    def _diagnose(self, verdict: dict) -> str:
        """The stage's verdict hook (Stage 3: diagnose_stage3_failure's ranked next steps), never fatal."""
        self.last_diagnosis = None
        if self.verdict_hook is None:
            return ""
        try:
            txt = self.verdict_hook(self.ws, verdict) or ""
        except Exception as e:
            txt = f"(the failure diagnosis could not run: {type(e).__name__}: {e})"
        self.last_diagnosis = txt or None
        return txt

    def _budget_note(self, rounds: int) -> str:
        """At half, three quarters and nine tenths of the round budget: how many rounds are left."""
        n = self.max_rounds
        marks = sorted({max(1, int(round(f * n))) for f in BUDGET_MARKS})
        if n < 8 or rounds not in marks:
            return ""
        left = n - rounds
        return (f"[harness] {rounds} of {n} rounds used, {left} left. Reuse unchanged compatible passed prefixes; rerun affected dependencies when inputs change. Batch independent "
                "calls in one round, and call finish (or report_blocked) before the budget ends.")

    def _verify(self) -> dict:
        issues, reports = [], {}
        integrity = self.contract.check()
        if not integrity["ok"]:
            return {"ok": False, "issues": integrity["issues"], "reports": {"input_contract": integrity}}
        if self.reference_evidence is not None:
            source_report = self.reference_evidence.check()
            if source_report.get("ok") is not True:
                return {"ok": False, "issues": source_report["issues"],
                        "reports": {"source_reference_provenance_check": source_report}}
        if self.recovery is not None and not self.cfg.get("_executor_receipts"):
            return {"ok": False, "issues": ["recovered candidate requires a fresh trusted execute_subgoal_plan invocation that successfully completes the physical plan; dry runs and failed or partial executions do not qualify, and prior receipts are not restored. Repair the latest executor failure before finalizing."],
                    "reports": {"recovery_executor_receipt": {"ok": False}}}
        if not self.verifiers:
            return {"ok": False, "issues": ["no verifiers configured"], "reports": {}}
        for v in self.verifiers:
            try:
                r = v(self.ws)
            except Exception as e:
                r = {"ok": False, "issues": [f"verifier {getattr(v, '__name__', v)} crashed: {type(e).__name__}: {e}"]}
            reports[getattr(v, "__name__", str(v))] = r
            if not isinstance(r, dict) or r.get("ok") is not True:
                issues.extend((r.get("issues", []) if isinstance(r, dict) else []) or [f"{getattr(v, '__name__', v)} failed"])
        if not issues and self.finalize is not None:
            try:
                result = self.finalize(self.ws, {"ok": True, "issues": [], "reports": reports})
            except Exception as exc:
                result = {"ok": False, "issues": [f"delivery failed: {type(exc).__name__}: {exc}"]}
            reports["verified_delivery"] = result
            if not isinstance(result, dict) or result.get("ok") is not True:
                issues.extend((result.get("issues", []) if isinstance(result, dict) else []) or ["delivery failed"])
        return {"ok": not issues, "issues": issues, "reports": reports}

    def _trace(self, rec: dict):
        rec["t"] = round(time.time(), 2)
        with self.trace.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")


def _clip(s: str, n: int = MAX_RESULT_CHARS) -> str:
    return s if len(s) <= n else s[: n // 2] + f"\n... [{len(s) - n} characters cut] ...\n" + s[-n // 2:]


def _executor_feedback(record: dict, limit: int = MAX_RESULT_CHARS) -> str:
    """Prioritize the failed segment; keep full measurements in the trace/report.

    Successful prefixes can contain thousands of per-frame reach measurements.
    They must not crowd out the failing segment or produce sliced, invalid JSON.
    This is a presentation view only, never the verifier or receipt input.
    """
    result = record["result"]
    identifiers = {"index", "step", "tag", "motion", "side", "object", "ok", "failed", "frames", "n_frames"}
    view = {"ok": record["ok"], "result": {
        "ok": result.get("ok"), "first_failure": result.get("first_failure"),
        "files": result.get("files"),
        **{k: v for k, v in result.items() if k not in {"ok", "first_failure", "files", "subgoals"}},
        "subgoals": [{k: v for k, v in sg.items() if k in identifiers} if sg.get("ok") is True else sg
                     for sg in result.get("subgoals", [])],
    }, "feedback_note": "Successful segment measurements omitted here; full result is in execution_report and the harness trace."}
    text = json.dumps(view, default=str, indent=1)
    if len(text) <= limit:
        return text
    # Progressively bound nested diagnostics without changing the authoritative
    # result. Every omission is explicit and the full report remains reachable.
    def bound(value, depth, items, chars):
        if isinstance(value, str):
            return value if len(value) <= chars else value[:chars] + " [see full report]"
        if isinstance(value, (list, dict)) and depth == 0:
            return {"omitted_entries": len(value), "see": "result.files.execution_report"}
        if isinstance(value, list):
            out = [bound(v, depth - 1, items, chars) for v in value[:items]]
            return out + ([{"omitted_entries": len(value) - items}] if len(value) > items else [])
        if isinstance(value, dict):
            out = {k: bound(v, depth - 1, items, chars) for k, v in list(value.items())[:items]}
            if len(value) > items:
                out["omitted_fields"] = len(value) - items
            return out
        return value
    # A failure's identity and failed checks are authoritative repair cues,
    # not a collection to truncate by insertion order. In real executor
    # records `failed` follows index/motion/side/object, so a four-field bound
    # used to discard the actual failure while keeping only its identifiers.
    failure = result.get("first_failure")
    failure_core = ({k: failure[k] for k in ("index", "motion", "side", "object", "failed", "error") if k in failure}
                    if isinstance(failure, dict) else failure)

    def bound_failure(depth, items, chars):
        if not isinstance(failure, dict):
            return failure
        return {**failure_core, **{k: bound(failure[k], depth, items, chars)
                                 for k in ("measured", "expect") if k in failure}}

    for depth, items, chars in ((7, 16, 1500), (6, 8, 800), (5, 4, 400), (4, 2, 180)):
        compact = {"ok": view["ok"], "result": {
            "ok": result.get("ok"),
            "first_failure": bound_failure(depth, items, chars),
            "files": result.get("files"),
            "n_subgoals": result.get("n_subgoals"),
            "subgoals": bound(view["result"]["subgoals"], depth, items, chars),
        }, "feedback_truncated": True, "feedback_note": view["feedback_note"]}
        text = json.dumps(compact, default=str, indent=1)
        if len(text) <= limit:
            return text
    return json.dumps({"ok": record["ok"], "result_ok": result.get("ok"), "first_failure": failure_core,
                       "feedback_truncated": True,
                       "files": result.get("files"), "feedback_note": "Read execution_report for the complete failure diagnostics."}, default=str)


def _short(args: dict, n: int = 160) -> str:
    s = json.dumps(args, default=str)
    return s if len(s) <= n else s[:n] + "..."
