"""What a run shows while it runs: the stage, the round out of the budget (and the money spent against the run's cap),
every tool call marked as a library skill, verifier or harness tool with what it returned, what the model is thinking,
the verifiers' verdicts, the tools it created — on the terminal (rich when available, plain text otherwise) and in
<workspace>/progress.json for anything that watches the run."""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

STAGES = {"stage1": ("1", "Semantic understanding"), "stage2": ("2", "Scene reconstruction"),
          "stage3": ("3", "Trajectory generation"), "stage4": ("4", "Robot data generation")}
QUIET = os.environ.get("DEXAGENT_QUIET", "") == "1"


def _rich():
    if QUIET or os.environ.get("DEXAGENT_PLAIN", "") == "1" or not sys.stdout.isatty():
        return None
    try:
        from rich.console import Console
        return Console(highlight=False, soft_wrap=True)
    except Exception:
        return None


def _fmt_args(args: dict, n: int = 220) -> str:
    """Arguments on one line: file paths by their last two parts, long dicts and lists by their shape."""
    parts = []
    for k, v in (args or {}).items():
        if isinstance(v, str) and ("/" in v or v.endswith((".json", ".xml", ".npz", ".png", ".mp4"))) and len(v) > 28:
            v = ".../" + "/".join(v.rstrip("/").split("/")[-2:])
        elif isinstance(v, dict):
            keys = list(v)[:6]
            v = "{" + ", ".join(f"{kk}: {_fmt_val(v[kk])}" for kk in keys) + (", ..." if len(v) > 6 else "") + "}"
        elif isinstance(v, (list, tuple)):
            v = _fmt_val(v)
        parts.append(f"{k}={v}")
    s = ", ".join(parts)
    return s if len(s) <= n else s[:n] + "..."


def _fmt_val(v, n: int = 60) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, (list, tuple)):
        if len(v) <= 4 and all(isinstance(x, (int, float)) for x in v):
            return "[" + ", ".join(f"{x:.3g}" if isinstance(x, float) else str(x) for x in v) + "]"
        return f"[{len(v)} items]"
    if isinstance(v, dict):
        return f"{{{len(v)} keys}}"
    s = json.dumps(v, default=str) if not isinstance(v, str) else v
    return s if len(s) <= n else s[:n] + "..."


def _result_line(name: str, raw, seconds: float) -> tuple[str, bool | None]:
    """One line about what a tool returned: ok / failed, the verdict or error, the numbers that matter."""
    r = raw.get("result", raw) if isinstance(raw, dict) else raw
    ok = None
    if isinstance(raw, dict) and raw.get("ok") is False and "error" in raw:
        return f"error: {str(raw['error']).strip().splitlines()[-1][:200]}", False
    if isinstance(r, dict):
        if "ok" in r and isinstance(r["ok"], bool):
            ok = r["ok"]
        bits = []
        for k in ("verdict", "error", "hint"):
            if r.get(k):
                bits.append(str(r[k])[:220])
        if r.get("failed"):
            bits.append("failed: " + _fmt_val(r["failed"], 120))
        if r.get("issues"):
            bits.append("issues: " + _fmt_val(r["issues"], 160))
        for k in ("n_feasible", "n_grasps", "n_frames", "duration_s", "worst_pos_err_mm", "worst_residual_mm", "overlap_mm", "iou", "rounds", "n_subgoals", "out_json", "out_png", "out_xml", "out_dir", "xml", "video"):
            if k in r and not isinstance(r[k], (dict, list)):
                bits.append(f"{k}={_fmt_val(r[k], 48)}")
        # the structures the stages return most: a search, a plan, an executed plan, a check sheet
        if isinstance(r.get("tools"), list) and r["tools"] and isinstance(r["tools"][0], dict):
            bits.append(f"{len(r['tools'])} tools: " + ", ".join(str(t.get("name")) for t in r["tools"][:6]) + (", ..." if len(r["tools"]) > 6 else ""))
        ar = r.get("arm_reach")
        if isinstance(ar, dict) and "worst_pos_err_mm" in ar:
            bits.append(f"arm reach {'ok' if ar.get('ok') else 'FAIL'} ({ar['worst_pos_err_mm']} mm / {ar.get('worst_rot_err_deg', 0)} deg)")
        if isinstance(r.get("subgoals"), list) and r["subgoals"] and isinstance(r["subgoals"][0], dict) and "ok" in r["subgoals"][0]:
            bits.append("subgoals: " + ", ".join(f"{sg.get('index')}:{'ok' if sg.get('ok') else 'FAIL ' + _fmt_val(sg.get('failed'), 40)}" for sg in r["subgoals"]))
        if isinstance(r.get("checks"), dict) and r["checks"] and all(isinstance(v, bool) for v in r["checks"].values()):
            bad = [k for k, v in r["checks"].items() if not v]
            bits.append("checks failed: " + ", ".join(bad) if bad else f"{len(r['checks'])} checks passed")
        if isinstance(r.get("reach"), dict) and "worst_pos_err_mm" in r["reach"]:
            bits.append(f"reach {'ok' if r['reach'].get('ok') else 'FAIL'} {r['reach']['worst_pos_err_mm']} mm after {_fmt_val(r.get('rotation_deg'), 60)}")
        if not bits:
            keys = [k for k in r if k not in ("ok",)][:6]
            bits.append(", ".join(f"{k}={_fmt_val(r[k], 36)}" for k in keys))
        text = "; ".join(b for b in bits if b)
    else:
        text = _fmt_val(r, 200)
    return f"{text} ({seconds:.1f}s)", ok


class Console:
    """One per stage run. The runner calls `round_start`, `thought`, `tool_call`, `tool_result`, `verdict`,
    `created`, `done`; a multi-stage driver calls `banner` and `summary`."""

    def __init__(self, stage: str, run_name: str, workspace: Path | None = None, max_rounds: int = 0, total_stages: int = 4, log=None):
        self.stage, self.run_name, self.ws = stage, run_name, Path(workspace) if workspace else None
        self.max_rounds, self.total = int(max_rounds), int(total_stages)
        self.rich = _rich()
        self.t0 = time.time()
        self.round = 0
        self.calls = Counter(); self.n_calls = 0; self.new_tools = []
        self.failures = 0
        self.last_thought = ""
        self.status = "running"
        self.verdicts = []
        self._call_t0 = None
        self.log = log
        self.budget = None                    # the run's spending cap (agent/budget.py), shown in the round header
        self.kinds = {}                       # tool name -> 'skill' | 'verifier' | 'harness', shown on every call

    # ---------------------------------------------------------------- output
    def _out(self, text: str, style: str = ""):
        if QUIET:
            return
        if self.rich:
            self.rich.print(f"[{style}]{text}[/{style}]" if style else text, markup=True)
        else:
            print(text, flush=True)

    def banner(self, title: str | None = None, note: str | None = None):
        num, name = STAGES.get(self.stage, (self.stage, self.stage))
        title = title or f"DexAgent  ·  Stage {num}/{self.total}  ·  {name}  ·  run '{self.run_name}'"
        line = "═" * max(60, len(title) + 4)
        budget = note or (f"budget: {self.max_rounds} rounds" if self.max_rounds else "no model rounds: reusing the verified example")
        if self.rich:
            from rich.panel import Panel
            self.rich.print(Panel(f"[bold]{title}[/bold]\nworkspace: {self.ws}\n{budget}", border_style="cyan", expand=False))
        else:
            self._out(f"\n{line}\n  {title}\n  workspace: {self.ws}\n  {budget}\n{line}")
        self._progress()

    def round_start(self, round_no: int):
        self.round = round_no
        el = time.time() - self.t0
        spend = f" · ${self.budget.spent_usd:.2f} of ${self.budget.cap_usd:g}" if self.budget is not None else ""
        head = (f"── round {round_no}/{self.max_rounds} · {self.n_calls} tool calls · {len(self.calls)} distinct tools · "
                f"{len(self.new_tools)} new · {self.failures} failed · {int(el) // 60}m{int(el) % 60:02d}s{spend} ──")
        self._out(head, "bold cyan")
        self._progress()

    def section(self, text: str):
        """A line that starts a new part of the stage (Stage 3: the method modules after the agent's search)."""
        self._out(f"── {text} ──", "bold magenta")
        self._progress()

    def thought(self, text: str, max_chars: int = 1200):
        text = text.strip()
        if not text:
            return
        self.last_thought = text
        shown = text if len(text) <= max_chars else text[:max_chars] + f" … [{len(text) - max_chars} more chars in trace.jsonl]"
        if self.rich:
            from rich.padding import Padding
            # wrapped at the terminal width (the console soft-wraps plain lines; a padded block would be cropped)
            self.rich.print(Padding(f"[italic dim]💭 {shown}[/italic dim]", (0, 0, 0, 2)), markup=True, soft_wrap=False)
        else:
            self._out("  💭 " + shown.replace("\n", "\n     "))
        self._progress()

    def tool_call(self, name: str, args: dict):
        self._call_t0 = time.time()
        self.calls[name] += 1; self.n_calls += 1
        kind = self.kinds.get(name)
        self._out(f"  ▶ {kind + ' ' if kind else ''}{name}({_fmt_args(args)})", "bold")

    def tool_result(self, name: str, raw):
        dt = time.time() - (self._call_t0 or time.time())
        line, ok = _result_line(name, raw, dt)
        if ok is False:
            self.failures += 1
        mark = "✓" if ok else ("✗" if ok is False else "·")
        style = "green" if ok else ("red" if ok is False else "dim")
        self._out(f"    {mark} {line}", style)
        if name == "library_create" and isinstance(raw, dict):
            r = raw.get("result", raw)
            if isinstance(r, dict) and r.get("ok") and isinstance(r.get("created"), dict):
                rec = r["created"]
                self.created(rec.get("name", "?"), rec.get("type") or rec.get("kind") or "tool")
        self._progress()

    def created(self, name: str, kind: str):
        self.new_tools.append({"name": name, "type": kind})
        self.kinds[name] = kind
        self._out(f"  ＋ new {kind} '{name}' joined the library", "bold magenta")

    def verdict(self, verdict: dict):
        ok = bool(verdict.get("ok"))
        self.verdicts.append(ok)
        if self.rich:
            from rich.table import Table
            t = Table(title=f"finish → verifiers {'PASS' if ok else 'FAIL'}", show_header=True, header_style="bold", expand=False, title_style="bold green" if ok else "bold red")
            t.add_column("verifier"); t.add_column("verdict"); t.add_column("detail", overflow="fold")
            for name, rep in (verdict.get("reports") or {}).items():
                rok = bool((rep or {}).get("ok"))
                issues = (rep or {}).get("issues") or []
                t.add_row(name, "[green]PASS[/green]" if rok else "[red]FAIL[/red]", "; ".join(str(i)[:160] for i in issues[:3]))
            self.rich.print(t)
        else:
            self._out(f"  finish → verifiers {'PASS' if ok else 'FAIL'}", "bold")
            for name, rep in (verdict.get("reports") or {}).items():
                rok = bool((rep or {}).get("ok")); issues = (rep or {}).get("issues") or []
                self._out(f"    {'PASS' if rok else 'FAIL'}  {name}  {'; '.join(str(i)[:160] for i in issues[:3])}")
        self._progress()

    def done(self, out: dict):
        self.status = "finished" if out.get("finished") else "unfinished"
        el = time.time() - self.t0
        num, name = STAGES.get(self.stage, (self.stage, self.stage))
        top = ", ".join(f"{n}×{c}" for n, c in self.calls.most_common(8))
        verdict = "PASS" if out.get("finished") else "FAIL"
        self._out(f"── stage {num} {name}: {verdict} in {out.get('rounds', self.round)} rounds, {self.n_calls} tool calls ({len(self.calls)} distinct: {top}), "
                  f"{len(self.new_tools)} new tools, {el / 60:.0f}m{el % 60:02.0f}s ──", "bold green" if out.get("finished") else "bold red")
        if self.new_tools:
            self._out("    built: " + ", ".join(f"{t['type']} {t['name']}" for t in self.new_tools), "magenta")
        if self.budget is not None:
            self._out(f"    spent: ${self.budget.state['by_stage'].get(self.stage, {}).get('spent_usd', 0.0):.2f} in this stage, "
                      f"${self.budget.spent_usd:.2f} of ${self.budget.cap_usd:g} in the run")
        if not out.get("finished") and out.get("verdict"):
            for i in (out["verdict"].get("issues") or [])[:8]:
                self._out(f"    · {str(i)[:200]}", "red")
        self._progress(final=True)

    # ---------------------------------------------------------------- progress file
    def snapshot(self) -> dict:
        return {"stage": self.stage, "run": self.run_name, "status": self.status, "round": self.round, "max_rounds": self.max_rounds,
                "tool_calls": self.n_calls, "tools_used": dict(self.calls), "failed_calls": self.failures, "new_tools": self.new_tools,
                "last_thought": self.last_thought[:2000], "verifier_passes": self.verdicts, "elapsed_s": round(time.time() - self.t0, 1),
                **({"spent_usd": round(self.budget.spent_usd, 4), "budget_usd": self.budget.cap_usd} if self.budget is not None else {}),
                "updated": time.strftime("%Y-%m-%d %H:%M:%S")}

    def _progress(self, final: bool = False):
        if self.ws is None:
            return
        try:
            (self.ws / "progress.json").write_text(json.dumps(self.snapshot(), indent=1))
        except OSError:
            pass


def summary_table(results: list[dict]):
    """The closing table of a multi-stage run: one row per stage."""
    rich = _rich()
    rows = [(STAGES.get(r["stage"], (r["stage"], r["stage"]))[0], STAGES.get(r["stage"], ("", r["stage"]))[1],
             "PASS" if r.get("finished") else ("SKIPPED" if r.get("skipped") else "FAIL"), str(r.get("rounds", "")), str(r.get("tool_calls", "")),
             str(len(r.get("new_tools") or [])), f"{(r.get('elapsed_s') or 0) / 60:.1f} min", str(r.get("workspace", ""))) for r in results]
    if rich:
        from rich.table import Table
        t = Table(title="DexAgent run", show_header=True, header_style="bold")
        for c in ("stage", "name", "result", "rounds", "tool calls", "new tools", "time", "workspace"):
            t.add_column(c, overflow="fold")
        for row in rows:
            t.add_row(*[("[green]" + v + "[/green]" if v == "PASS" else "[red]" + v + "[/red]" if v == "FAIL" else v) for v in row])
        rich.print(t)
    else:
        print("\nstage  name                      result   rounds  calls  new  time")
        for row in rows:
            print(f"{row[0]:5s}  {row[1]:24s}  {row[2]:7s}  {row[3]:6s}  {row[4]:5s}  {row[5]:3s}  {row[6]}  {row[7]}")
