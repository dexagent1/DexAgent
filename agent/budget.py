"""The run's money cap: what the model calls of a run have cost, across all its stages, against agent.budget_usd.

A model API (agent.provider anthropic or openai) needs `agent.budget_usd` before it is called: every call's tokens are
priced (agent.price_usd_per_mtok, else the list prices of the models below) and added to runs/<name>/budget.json, so
the cap holds across stages and across separate stage runs of the same run name (delete the file to start the count
again). A stage whose run has spent its cap stops before the next model call, so the last call can pass the cap by at
most its own cost; at BUDGET_WARN of it the agent is told to finish. The external transport (a coding
agent answering through files, e.g. a forked subagent on a subscription) and scripted mocks cost nothing here: their
limit is the round budget (stageN.max_rounds) and the turn limit the user gives that agent.

    budget = Budget.for_run(cfg)           # None when the provider is not a priced API
    budget.charge(usage)                   # usage = {"input_tokens", "output_tokens"} of one call
    budget.spent_usd, budget.cap_usd, budget.exceeded(), budget.fraction()
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

PRICED_PROVIDERS = ("anthropic", "openai")
BUDGET_WARN = 0.9
# USD per million tokens (input, output): list prices of the models named in dexagent.yaml's comments. A model that is
# not listed needs agent.price_usd_per_mtok; check your provider's pricing page either way.
LIST_PRICES = {
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-5-5": (4.0, 20.0),
}


class BudgetError(ValueError):
    """The configuration does not let the run's spending be capped."""


def prices(cfg: dict) -> tuple[float, float]:
    a = (cfg or {}).get("agent") or {}
    given = a.get("price_usd_per_mtok") or {}
    if given.get("input") is not None and given.get("output") is not None:
        return float(given["input"]), float(given["output"])
    model = str(a.get("model") or "")
    if model in LIST_PRICES:
        return LIST_PRICES[model]
    raise BudgetError(f"no price for model {model!r}: set agent.price_usd_per_mtok {{input, output}} (USD per million tokens) "
                      f"in dexagent.yaml so the run's spending can be capped")


def require_cap(cfg: dict):
    """Raise BudgetError unless a priced API run has a positive agent.budget_usd and a price for its model."""
    a = (cfg or {}).get("agent") or {}
    if str(a.get("provider") or "anthropic").lower() not in PRICED_PROVIDERS:
        return
    cap = a.get("budget_usd")
    try:
        cap = None if cap is None else float(cap)
    except (TypeError, ValueError):
        raise BudgetError(f"agent.budget_usd must be a number of US dollars, got {a.get('budget_usd')!r}")
    if cap is None or cap <= 0:
        raise BudgetError("agent.budget_usd is not set: a model API run needs a spending cap in USD (dexagent.yaml, agent block). "
                          "Set it, together with each stage's max_rounds; the run stops when either is spent.")
    prices(cfg)


class Budget:
    def __init__(self, path: Path, cap_usd: float, price_in: float, price_out: float):
        self.path, self.cap_usd = Path(path), float(cap_usd)
        self.price_in, self.price_out = float(price_in), float(price_out)
        self.state = {"cap_usd": self.cap_usd, "spent_usd": 0.0, "input_tokens": 0, "output_tokens": 0, "calls": 0, "by_stage": {}}
        try:
            old = json.loads(self.path.read_text())
            for k in ("spent_usd", "input_tokens", "output_tokens", "calls", "by_stage"):
                if k in old:
                    self.state[k] = old[k]
        except (OSError, ValueError):
            pass

    @classmethod
    def for_run(cls, cfg: dict):
        """The run's budget (runs/<name>/budget.json), or None for a provider that is not a priced API."""
        a = (cfg or {}).get("agent") or {}
        if str(a.get("provider") or "anthropic").lower() not in PRICED_PROVIDERS:
            return None
        require_cap(cfg)
        p_in, p_out = prices(cfg)
        run = cfg.get("run") or {}
        path = Path(run.get("output_dir", "runs")) / str(run.get("name", "run")) / "budget.json"
        return cls(path, float(a["budget_usd"]), p_in, p_out)

    @property
    def spent_usd(self) -> float:
        return float(self.state["spent_usd"])

    def fraction(self) -> float:
        return self.spent_usd / self.cap_usd if self.cap_usd > 0 else 1.0

    def exceeded(self) -> bool:
        return self.spent_usd >= self.cap_usd

    def charge(self, usage: dict | None, stage: str | None = None) -> float:
        """Add one call's tokens; returns its cost in USD."""
        if not usage:
            return 0.0
        n_in, n_out = int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)
        usd = (n_in * self.price_in + n_out * self.price_out) / 1e6
        s = self.state
        s["spent_usd"] = round(s["spent_usd"] + usd, 6)
        s["input_tokens"] += n_in; s["output_tokens"] += n_out; s["calls"] += 1
        if stage:
            b = s["by_stage"].setdefault(stage, {"spent_usd": 0.0, "calls": 0})
            b["spent_usd"] = round(b["spent_usd"] + usd, 6); b["calls"] += 1
        s["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.save()
        return usd

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(self.state, indent=1))
        tmp.replace(self.path)
