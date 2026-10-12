"""Stage 1: semantic understanding. One command: a video and a prompt in, scene_semantics.json out.

    python stage1_semantic_understanding/run_stage1.py dexagent.yaml
    python stage1_semantic_understanding/run_stage1.py dexagent.yaml --video clip.mp4 --prompt "Close the box"

Reads agent.*, inputs.video, inputs.task, inputs.library and run.* from the YAML, runs the agent
with the Stage 1 prompt and the tool library, verifies the deliverable, and writes
<output_dir>/<name>/stage1/scene_semantics.json plus trace.jsonl and any new tools.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from agent.config import load_config, run_dir, semantics_file  # noqa: E402

EXAMPLES = ["place_toy_in_drawer", "knot_the_rope", "computer_installation"]
STAGE_VERIFIERS = ["scene_semantic_schema_check", "scene_semantic_consistency_check", "goal_language_coverage_check"]
# the library categories Stage 1 can call directly; anything else is reachable through run_python
TOOL_CATEGORIES = ["Perception & Grounding", "Measurements", "Scene Assertions"]


def stage_prompt(examples=EXAMPLES) -> str:
    text = (HERE / "prompts" / "stage1.md").read_text()
    schema = json.dumps(json.loads((HERE / "schema.json").read_text()), indent=1)
    blocks = []
    for name in examples:
        d = HERE / "examples" / name
        prompt = (d / "prompt.txt").read_text().strip()
        js = semantics_file(d).read_text().strip()
        blocks.append(f"### Example: {name.replace('_', ' ')}\n\nPrompt: \"{prompt}\"\n\n```json\n{js}\n```")
    return text.replace("__SCHEMA__", schema).replace("__EXAMPLES__", "\n\n".join(blocks))


def user_message(video: Path, prompt: str, workspace: Path, duration_s: float | None, setup_note: str) -> str:
    return (f"Workspace: {workspace}\n"
            f"Video: {video}" + (f" ({duration_s:.1f} s)" if duration_s else "") + "\n"
            f"Task prompt: \"{prompt}\"\n" + ("No task prompt was supplied. Infer the task from the video; use an empty prompt for prompt-coverage verification.\n" if not prompt else "") +
            f"{setup_note}\n"
            f"Write the deliverable to {workspace / 'scene_semantics.json'}, verify it, then call finish.")


def setup_note(cfg: dict) -> str:
    return ("Robot setup: two stations, left and right hand, see setup.yaml; both human hands can be mapped."
            if cfg["inputs"].get("setup") else "")


def configured_examples(cfg: dict) -> list[str]:
    """The worked examples shown in the prompt: stage1.examples of the configuration, those that exist."""
    names = (cfg.get("stage1") or {}).get("examples", EXAMPLES)
    return [e for e in names if semantics_file(HERE / "examples" / e).exists()]


def make_verifiers(cfg: dict, prompt: str, duration_s: float | None, lib: Path):
    from agent.library import call_tool, load_index
    index = load_index(lib)

    def schema_check(ws: Path):
        r = call_tool(lib, index, "scene_semantic_schema_check", {"path": str(semantics_file(ws))}, cfg)
        return r.get("result") if r.get("ok") else {"ok": False, "issues": [r.get("error")]}

    def consistency_check(ws: Path):
        r = call_tool(lib, index, "scene_semantic_consistency_check",
                      {"path": str(semantics_file(ws)), "prompt": prompt, "video_duration_s": duration_s}, cfg)
        return r.get("result") if r.get("ok") else {"ok": False, "issues": [r.get("error")]}

    def goal_language_check(ws: Path):
        # every goal clause must be one Stage 3's goal gate can measure, or Stage 3 cannot pass whatever the robot does
        r = call_tool(lib, index, "goal_language_coverage_check", {"path": str(semantics_file(ws))}, cfg)
        return r.get("result") if r.get("ok") else {"ok": False, "issues": [r.get("error")]}

    schema_check.__name__ = "scene_semantic_schema_check"
    consistency_check.__name__ = "scene_semantic_consistency_check"
    goal_language_check.__name__ = "goal_language_coverage_check"
    return [schema_check, consistency_check, goal_language_check]


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="DexAgent Stage 1")
    ap.add_argument("config", nargs="?", default=os.environ.get("DEXAGENT_CONFIG", str(REPO / "dexagent.yaml")))
    ap.add_argument("--video"), ap.add_argument("--prompt"), ap.add_argument("--name")
    ap.add_argument("--max-rounds", type=int)
    ap.add_argument("--llm", default="config", help="config (the YAML's agent block) or mock:<turns.json>")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.video:
        cfg["inputs"]["video"] = str(Path(a.video).resolve())
    if a.prompt is not None:
        cfg["inputs"]["task"] = a.prompt
    if a.name:
        cfg["run"]["name"] = a.name
    video = Path(cfg["inputs"]["video"])
    if not video.exists():
        sys.exit(f"inputs.video does not exist: {video}")
    prompt = (cfg["inputs"].get("task") or "").strip()
    ws = run_dir(cfg, "stage1")
    from agent.library import library_root
    lib = library_root(cfg)
    from agent.input_identity import video_identity, video_verifier
    from agent.external import atomic_json
    identity = video_identity(video)
    dur = identity["duration_s"]
    atomic_json(ws / "input_video.json", identity)
    from agent.reuse import try_reuse
    verifiers = make_verifiers(cfg, prompt, dur, lib)
    verifiers.insert(0, video_verifier(video, None, expected_identity=identity))
    def finalize(candidate, verdict):
        path = semantics_file(candidate)
        semantics = json.loads(path.read_text())
        semantics["input_identity"] = identity
        atomic_json(path, semantics)
        return {"ok": True, "issues": [], "video_sha256": identity["sha256"]}
    # a stage re-opened by a cross-stage repair (or downstream of one) reads why first and never reuses an example: the input
    # it would reuse is the one found wrong
    from agent import repair
    note = repair.take_note(ws) if repair.has_note(ws) else ""
    reused, reuse_note = (None, None) if note else try_reuse(cfg, "stage1", ws, video, verifiers=verifiers, prompt=cfg["inputs"].get("task") or "")
    if reused is not None:
        finalize(ws, reused.get("verdict", {}))
        return 0
    if a.llm.startswith("mock:"):
        from agent.llm import load_mock
        llm = load_mock(a.llm[5:], "stage1")
    else:
        from agent.budget import BudgetError
        from agent.llm import make_llm
        try:
            llm = make_llm(cfg)
        except BudgetError as exc:                    # a model API run needs a spending cap; refused before any call
            print(f"  {exc}", flush=True)
            (ws / "run.json").write_text(json.dumps({"finished": False, "stage": "stage1", "workspace": str(ws), "failure": str(exc)}, indent=1))
            return 1
    from agent.runner import StageRunner
    s1 = cfg.get("stage1") or {}
    cfg["_verification_inputs"] = [str(video), str(ws / "input_video.json")]
    runner = StageRunner(cfg, llm, ws, stage_prompt(configured_examples(cfg)), verifiers, categories=TOOL_CATEGORIES,
                         max_rounds=a.max_rounds or s1.get("max_rounds"), finalize=finalize, stage="stage1")
    msg = user_message(video, prompt, ws, dur, setup_note(cfg))
    if reuse_note:
        msg += "\n" + reuse_note
    msg += note
    out = runner.run(msg)
    (ws / "run.json").write_text(json.dumps(out, indent=1, default=str))
    return 0 if out["finished"] else 1


if __name__ == "__main__":
    sys.exit(main())
