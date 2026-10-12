"""The run's spending cap (agent/budget.py): a model API run without agent.budget_usd is refused before any call, every
call's tokens are priced and summed across the run's stages in runs/<name>/budget.json, the agent is told at 90% and the
stage stops when the cap is spent; the external transport and mocks are capped by rounds only.

    python -m unittest agent.tests.test_budget
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("DEXAGENT_QUIET", "1")

from agent.budget import Budget, BudgetError, require_cap  # noqa: E402
from agent.config import load_config  # noqa: E402
from agent.llm import MockLLM, make_llm  # noqa: E402
from agent.runner import StageRunner  # noqa: E402


class PricedMock(MockLLM):
    """A scripted model whose every call costs `tokens` (as an API client reports them)."""
    priced = True

    def __init__(self, turns, tokens=(100_000, 10_000)):
        super().__init__(turns)
        self.tokens = tokens

    def complete(self, system, messages, tools):
        self.last_usage = {"input_tokens": self.tokens[0], "output_tokens": self.tokens[1]}
        return super().complete(system, messages, tools)


def turn(name, **inp):
    return [{"type": "tool_use", "name": name, "input": inp}]


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = load_config()
        self.cfg["run"].update(output_dir=self.tmp.name, name="b")

    def test_api_run_needs_a_cap_and_a_price(self):
        self.cfg["agent"].update(provider="anthropic", model="claude-opus-5-5", budget_usd=None)
        with self.assertRaises(BudgetError):
            make_llm(self.cfg)
        self.cfg["agent"].update(budget_usd=10, model="some-other-model", price_usd_per_mtok={"input": None, "output": None})
        with self.assertRaises(BudgetError):
            require_cap(self.cfg)
        self.cfg["agent"]["price_usd_per_mtok"] = {"input": 1, "output": 2}
        require_cap(self.cfg)
        self.cfg["agent"].update(provider="external", budget_usd=None)
        require_cap(self.cfg)                        # a forked coding agent is capped by rounds

    def test_charges_add_up_across_stages(self):
        self.cfg["agent"].update(provider="anthropic", model="claude-opus-5", budget_usd=20)
        b = Budget.for_run(self.cfg)
        self.assertAlmostEqual(b.charge({"input_tokens": 1_000_000, "output_tokens": 100_000}, "stage1"), 5.0 + 2.5)
        again = Budget.for_run(self.cfg)             # another stage of the same run reads what was spent
        again.charge({"input_tokens": 1_000_000, "output_tokens": 0}, "stage2")
        state = json.loads((Path(self.tmp.name) / "b" / "budget.json").read_text())
        self.assertAlmostEqual(state["spent_usd"], 12.5)
        self.assertEqual(sorted(state["by_stage"]), ["stage1", "stage2"])
        self.assertFalse(again.exceeded())

    def test_stage_is_told_at_ninety_percent_and_stops_at_the_cap(self):
        # each call: 100k in + 10k out at Opus 5 list prices = $0.75; cap $1.60: after two calls ($1.50, 94%) the agent is
        # told, the third call ($2.25) spends the cap, and the stage stops before a fourth (the cap is checked before each
        # call, so the last call can pass it by at most one call's cost)
        self.cfg["agent"].update(provider="anthropic", model="claude-opus-5", budget_usd=1.6)
        llm = PricedMock([turn("list_dir", path=".") for _ in range(10)])
        ws = Path(self.tmp.name) / "b" / "stage1"
        out = StageRunner(self.cfg, llm, ws, "test stage", [lambda w: {"ok": False, "issues": ["x"]}], max_rounds=10, stage="stage1").run("go")
        self.assertFalse(out["finished"])
        self.assertEqual(out["stop_reason"], "budget_cap")
        self.assertEqual(out["rounds"], 3)
        self.assertAlmostEqual(out["run_spent_usd"], 2.25)
        told = [b["text"] for m in llm.seen[-1] if m["role"] == "user" for b in m["content"]
                if b.get("type") == "text" and "spending cap" in b.get("text", "")]
        self.assertTrue(told)
        report = json.loads((ws / "failure_report.json").read_text())
        self.assertIn("spending cap", report["reason"])

    def test_mock_and_external_runs_are_not_priced(self):
        self.cfg["agent"].update(provider="anthropic", budget_usd=None)
        out = StageRunner(self.cfg, MockLLM([turn("list_dir", path=".")]), Path(self.tmp.name) / "m" / "stage1", "t",
                          [lambda w: {"ok": False, "issues": ["x"]}], max_rounds=2, stage="stage1").run("go")
        self.assertNotIn("spent_usd", out)


if __name__ == "__main__":
    unittest.main()
