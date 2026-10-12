"""The Stage 3 harness around the agent's search: the round budget scales with the Stage 1 subgoals, every failed finish
verdict carries diagnose_stage3_failure's ranked next steps, the agent is told how much budget is left, and an unfinished
search continues with the method modules (stage3.methods), whose episodes must pass the same verifiers.

    python agent/tests/test_stage3_search_harness.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DEXAGENT_QUIET", "1")

import stage3_trajectory_generation.run_stage3 as S3  # noqa: E402
from agent.config import load_config  # noqa: E402
from agent.library import library_root  # noqa: E402
from agent.llm import MockLLM  # noqa: E402
from agent.runner import StageRunner  # noqa: E402

FAILS = 0
FIX = ROOT / "tool_library" / "tests" / "fixtures" / "diagnose_stage3_failure"


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def T(name, **inp):
    return [{"type": "tool_use", "name": name, "input": inp}]


def semantics(tmp, n):
    p = tmp / f"sem_{n}.json"
    p.write_text(json.dumps({"subgoals": [{"index": i + 1, "acting_hand": "right", "motion": "grasp_lift", "object_refs": ["brick"]} for i in range(n)]}))
    return p


def main():
    tmp = Path(tempfile.mkdtemp(prefix="dexagent_s3harness_", dir=os.environ.get("TMPDIR")))
    cfg = load_config(ROOT / "dexagent.yaml")
    cfg["run"]["output_dir"] = str(tmp / "runs")
    # -- the budget ----------------------------------------------------------------------------------------------------
    d0, _ = S3.round_budget(cfg, semantics(tmp, 19))
    check("by default every Stage 3 run has 100 rounds, whatever its subgoal count", d0 == 100, str(d0))
    c10 = json.loads(json.dumps(cfg)); c10["stage3"]["rounds_per_subgoal"] = 10
    s3 = c10["stage3"]
    b19, t19 = S3.round_budget(c10, semantics(tmp, 19))
    b3, _ = S3.round_budget(c10, semantics(tmp, 3))
    b40, t40 = S3.round_budget(c10, semantics(tmp, 40))
    c0 = json.loads(json.dumps(c10)); c0["stage3"]["rounds_per_subgoal"] = 0
    b0, _ = S3.round_budget(c0, semantics(tmp, 19))
    base, per, cap = int(s3["max_rounds"]), int(s3["rounds_per_subgoal"]), int(s3["max_rounds_cap"])
    check("the round budget is the base plus a share per Stage 1 subgoal, capped; 0 per subgoal keeps the base",
          b19 == min(base + 19 * per, cap) and b3 == base + 3 * per and b40 == cap and b0 == base and "capped" in t40 and "19 Stage 1 subgoals" in t19,
          f"3 -> {b3}, 19 -> {b19} ({t19}), 40 -> {b40}, per 0 -> {b0}")
    # -- the search's CPUs come from the configuration ---------------------------------------------------------------------
    from agent.library import configured_args, find_tool, load_index
    c2 = json.loads(json.dumps(cfg)); c2["compute"] = {"launcher": ["srun", "--jobid=1", "--overlap"], "workers": 6}
    tool = find_tool(load_index(library_root(cfg)), "optimize_grasp_pose")
    got = configured_args(c2, tool) if tool else {}
    check("compute.launcher and compute.workers reach optimize_grasp_pose as a list and a number",
          got.get("launcher") == ["srun", "--jobid=1", "--overlap"] and got.get("workers") == 6 and not configured_args(cfg, tool), str(got))
    # -- every failed verdict carries the diagnosis --------------------------------------------------------------------
    verdict = json.loads((FIX / "braid_finish_verdict.json").read_text())

    def failing(ws):
        return verdict

    failing.__name__ = "stage3_gate_suite"
    hook = S3.make_diagnosis(cfg, tmp / "missing_sem.json", tmp / "missing_recon.json", library_root(cfg))
    llm = MockLLM([T("finish", summary="done"), T("report_blocked", reason="the strands cannot be pinched")])
    ws = tmp / "runs" / "diag" / "stage3"
    out = StageRunner(cfg, llm, ws, "test stage", [failing], max_rounds=6, stage="stage3", verdict_hook=hook).run("go")
    res = [b for m in llm.seen[-1] if m["role"] == "user" for b in m["content"] if b.get("type") == "tool_result"]
    txt = res[0]["content"][0]["text"] if res else ""
    rep = json.loads((ws / "failure_report.json").read_text())
    check("a failed finish verdict ends with the diagnosis: failure classes and ranked next steps with tool names",
          "Verifiers FAILED" in txt and "Diagnosis (diagnose_stage3_failure)" in txt and "Next steps:" in txt and "optimize_grasp_pose" in txt,
          txt[txt.find("Diagnosis"):][:300].replace("\n", " | "))
    check("the stage's failure_report.json keeps the last diagnosis", "Diagnosis" in str(rep.get("diagnosis")) and not out["finished"])
    # -- the budget notes ----------------------------------------------------------------------------------------------
    llm = MockLLM([T("list_dir", path=".") for _ in range(10)])
    StageRunner(cfg, llm, tmp / "runs" / "notes" / "stage3", "test stage", [failing], max_rounds=8, stage="stage3").run("go")
    notes = [b["text"] for m in llm.seen[-1] if m["role"] == "user" for b in m["content"] if b.get("type") == "tool_result"
             for b in b["content"] if b.get("type") == "text" and b["text"].startswith("[harness]")]
    check("the agent is told the rounds left at half, three quarters and nine tenths of the budget",
          [n.split(" rounds used")[0] for n in notes] == ["[harness] 4 of 8", "[harness] 6 of 8", "[harness] 7 of 8"], str(notes))
    # -- the method modules after an unfinished search ------------------------------------------------------------------
    import stage3_trajectory_generation.methods as M
    try:
        M.configured_methods({"stage3": {"methods": 3}}); bad = False
    except ValueError:
        bad = True
    check("method modules: the three by default, [] or 'none' turns them off, anything but skill names is an error",
          M.configured_methods({}) == list(M.DEFAULT_METHODS) == ["physics_informed_retargeting", "sampling_mpc_refinement", "reinforcement_learning_tracking"]
          and M.configured_methods({"stage3": {"methods": []}}) == [] and M.configured_methods({"stage3": {"methods": "none"}}) == [] and bad)
    check("they run only after the agent's round, money or wall-time budget is spent; not after a stall, an upstream report, an "
          "administrative stop, a model backend error or a finished stage",
          M.should_run({"failure": "budget exhausted without passing the stage verifiers"}) == "max_rounds"
          and M.should_run({"stopped": True, "stop_reason": "budget_cap"}) == "budget_cap"
          and M.should_run({"stopped": True, "stop_reason": "search_stalled"}) is None
          and M.should_run({"stopped": True, "stop_reason": "wall_time_budget"}) == "wall_time_budget"
          and M.should_run({"upstream": {"stage": "stage2"}}) is None and M.should_run({"stopped": True, "stop_reason": "operator_checkpoint"}) is None
          and M.should_run({"stopped": True, "stop_reason": "cooperative_search_checkpoint"}) is None
          and M.should_run({"failure": "blocked: model backend failed: RuntimeError: x"}) is None and M.should_run({"finished": True}) is None)
    # through the runner: the agent's budget runs out; the first method cannot run here, the second returns a plan that is
    # executed, packaged and verified exactly as an agent's, and published
    wsm = tmp / "runs" / "mm" / "stage3"
    calls, published = [], []

    def fake_tool(self, name, args):
        calls.append(name)
        r = {"physics_informed_retargeting": {"ok": False, "unavailable": True, "error": "the GPU optimizer is not set up here"},
             "sampling_mpc_refinement": {"ok": True, "plan": str(wsm / "methods" / "02_sampling_mpc_refinement" / "plan.json"),
                                         "reference": str(wsm / "methods" / "02_sampling_mpc_refinement" / "reference.npz"), "scene": "scene.xml", "fps": 30.0},
             "execute_subgoal_plan": {"ok": True, "subgoals": [{"index": 1, "ok": True}], "files": {"execution_report": str(wsm / "execution_report.json")}},
             "package_stage3_episode": {"ok": True, "issues": []}}.get(name, {"ok": False})
        if name == "package_stage3_episode":
            (wsm / "verified.txt").write_text("method")
        return [{"type": "text", "text": "x"}], {"ok": True, "result": r}

    def marker_ws(ws):
        return {"ok": (Path(ws) / "verified.txt").exists(), "issues": [] if (Path(ws) / "verified.txt").exists() else ["not verified"]}
    cm = json.loads(json.dumps(cfg))
    runner = StageRunner(cm, MockLLM([T("finish", summary="nothing yet")]), wsm, "test stage", [marker_ws], max_rounds=1, stage="stage3",
                         finalize=lambda ws_, verdict_: published.append(Path(ws_)) or {"ok": True, "issues": []},
                         after_search=lambda r_, o_: M.run_method_modules(r_, o_, cm, tmp / "sem.json", tmp / "recon.json"))
    with patch.object(StageRunner, "run_tool", fake_tool):
        out = runner.run("go")
    trace = [json.loads(line) for line in (wsm / "trace.jsonl").read_text().splitlines()]
    check("the agent's budget ran out; the unavailable method is skipped, the next one's plan is executed, packaged and verified "
          "like an agent's episode and published; the stage passes, the agent's failure report kept aside, every call in the trace "
          "and none in the agent's own tool counts (the method modules run silently)",
          out["finished"] and out["methods"]["ok"] and out["methods"]["method"] == "sampling_mpc_refinement"
          and calls == ["physics_informed_retargeting", "sampling_mpc_refinement", "execute_subgoal_plan", "package_stage3_episode"]
          and published == [wsm] and (wsm / "failure_report_agent.json").exists() and not (wsm / "failure_report.json").exists()
          and out["agent_search"].get("failure", "").startswith("budget exhausted") and "failure" not in out
          and any(e.get("phase") == "method_modules" and e.get("name") == "sampling_mpc_refinement" for e in trace)
          and not out["tools_used"].get("sampling_mpc_refinement") and (wsm / "methods" / "methods_report.json").exists(),
          json.dumps({k: out.get(k) for k in ("finished", "agent_search")}, default=str)[:300] + f" calls={calls}")
    # an upstream report re-opens that stage instead; nothing runs
    calls.clear()
    wsu = tmp / "runs" / "mu" / "stage3"
    runner = StageRunner(cm, MockLLM([T("finish", summary="x"), T("report_blocked", reason="the box", upstream="stage2", evidence="no gap to grip")]),
                         wsu, "test stage", [marker_ws], max_rounds=6, stage="stage3",
                         after_search=lambda r_, o_: M.run_method_modules(r_, o_, cm, tmp / "sem.json", tmp / "recon.json"))
    with patch.object(StageRunner, "run_tool", fake_tool):
        out = runner.run("go")
    check("a report naming Stage 2 sends the run back there: no method module runs", out.get("upstream") and "methods" not in out and calls == [])
    # through the stage's own main: the agent runs out of budget and the method modules' episode passes the stage verifiers
    y = tmp / "mm.yaml"
    import numpy as np
    from tool_library.common.video import write_video
    from agent.input_identity import video_identity
    import agent.artifacts as delivery
    # Publication is a stub only in this routing test. Actual delivery is tested with physical recordings in the separate
    # delivery/runtime tests.
    delivery.publish_stage3 = lambda *a, **k: {"ok": True, "issues": [], "test_stub": "method module routing"}
    write_video([np.zeros((32, 32, 3), dtype=np.uint8)] * 3, tmp / "v.mp4", fps=30)
    (tmp / "runs" / "mmrun" / "stage1").mkdir(parents=True)
    (tmp / "runs" / "mmrun" / "stage2").mkdir(parents=True)
    semantics(tmp, 2).replace(tmp / "runs" / "mmrun" / "stage1" / "scene_semantics.json")
    semantic_path = tmp / "runs" / "mmrun" / "stage1" / "scene_semantics.json"
    semantic_data = json.loads(semantic_path.read_text())
    semantic_data["input_identity"] = video_identity(tmp / "v.mp4")
    semantic_path.write_text(json.dumps(semantic_data))
    (tmp / "runs" / "mmrun" / "stage2" / "scene_recon.json").write_text(json.dumps({"objects": []}))
    (tmp / "runs" / "mmrun" / "stage2" / "scene.xml").write_text('<mujoco><worldbody><geom type="plane" size="1 1 .1"/></worldbody></mujoco>')
    y.write_text(f"run:\n  name: mmrun\n  output_dir: {tmp / 'runs'}\nexamples:\n  reuse: false\ninputs:\n  video: {tmp / 'v.mp4'}\n")
    S3.make_verifiers = lambda *a, **k: [lambda ws: {"ok": (Path(ws) / "verified.txt").exists(), "issues": ["not verified"]}]
    S3.make_diagnosis = lambda *a, **k: None
    mock = tmp / "mock.json"
    mock.write_text(json.dumps({"stage3": [T("finish", summary="nothing yet")]}))
    seen = []

    def methods_stub(runner_, out_, cfg_, semantic_, recon_, video=None):
        seen.append(out_.get("failure"))
        (Path(runner_.ws) / "verified.txt").write_text("method")
        verdict_ = runner_._verify()
        return {"ok": verdict_["ok"], "method": "stub", "verdict": verdict_}
    # This fixture tests the routing only, with synthetic Stage 2 and mocked publication. Real source-recipe validation has
    # separate tests.
    with patch("agent.reference_evidence.revalidate_source_recipes", return_value={"ok": True}), patch.object(S3, "run_method_modules", methods_stub):
        rc = S3.main([str(y), "--max-rounds", "3", "--llm", f"mock:{mock}"])
    wsr = tmp / "runs" / "mmrun" / "stage3"
    run = json.loads((wsr / "run.json").read_text())
    candidate = Path(run["workspace"])
    check("the agent ran out of budget; the method modules' episode passed the stage verifiers, so the stage passes and the "
          "agent's failure report is kept aside", rc == 0 and run["finished"] and run["methods"]["ok"] and seen and seen[0].startswith("budget exhausted")
          and (candidate / "failure_report_agent.json").exists() and not (wsr / "failure_report.json").exists(),
          json.dumps({k: run.get(k) for k in ("finished", "rounds")}))
    print("ALL PASS" if not FAILS else f"{FAILS} FAILED")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
