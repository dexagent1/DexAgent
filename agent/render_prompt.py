"""Write the exact prompt pack a stage sends to the model: system prompt, tool list and first
user message. For reading, for review, and for running the stage by hand or through another model.

    python -m agent.render_prompt stage1 dexagent.yaml out_dir/ [--video v.mp4 --prompt "..."]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stage1", "stage2", "stage3"])
    ap.add_argument("config")
    ap.add_argument("out_dir")
    ap.add_argument("--video"), ap.add_argument("--prompt"), ap.add_argument("--name"), ap.add_argument("--semantic"), ap.add_argument("--recon"), ap.add_argument("--egodex")
    a = ap.parse_args(argv)
    from agent.config import load_config, run_dir, semantics_file
    from agent.devtools import BUILTIN_SCHEMAS
    from agent.library import library_root, load_index, tool_catalog, tool_schema
    from agent.runner import system_prompt
    cfg = load_config(a.config)
    if a.video:
        cfg["inputs"]["video"] = str(Path(a.video).resolve())
    if a.prompt is not None:
        cfg["inputs"]["task"] = a.prompt
    if a.name:
        cfg["run"]["name"] = a.name
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    if a.stage == "stage1":
        sys.path.insert(0, str(REPO / "stage1_semantic_understanding"))
        import run_stage1 as S
        video = Path(cfg["inputs"]["video"])
        prompt = (cfg["inputs"].get("task") or "").strip() or "(no prompt given: infer the task from the video)"
        ws = run_dir(cfg, "stage1")
        try:
            from tool_library.common.video import video_info
            dur = video_info(video)["duration_s"]
        except Exception:
            dur = None
        system = system_prompt(S.stage_prompt(S.configured_examples(cfg)))
        user = S.user_message(video, prompt, ws, dur, S.setup_note(cfg))
    if a.stage == "stage2":
        sys.path.insert(0, str(REPO / "stage2_scene_reconstruction"))
        import run_stage2 as S
        video = Path(cfg["inputs"]["video"]); ws = run_dir(cfg, "stage2")
        semantic = Path(a.semantic).resolve() if a.semantic else semantics_file(ws.parent / "stage1")
        system = system_prompt(S.stage_prompt())
        egodex = a.egodex or cfg["inputs"].get("egodex_hdf5")
        user = S.user_message(video, semantic, ws, cfg, Path(egodex).resolve() if egodex else None)
    if a.stage == "stage3":
        sys.path.insert(0, str(REPO / "stage3_trajectory_generation"))
        import run_stage3 as S
        video = Path(cfg["inputs"]["video"]); ws = run_dir(cfg, "stage3")
        semantic = Path(a.semantic).resolve() if a.semantic else semantics_file(ws.parent / "stage1")
        recon = Path(a.recon).resolve() if a.recon else (ws.parent / "stage2" / "scene_recon.json")
        system = system_prompt(S.stage_prompt())
        user = S.user_message(video, semantic, recon, ws, cfg, S.round_budget(cfg, semantic)[1] if semantic.exists() else None)
    # the paragraph the harness appends when a verified example of the same task exists (an identical video never
    # reaches the model: the stage reuses the example instead)
    if a.stage in ("stage2", "stage3"):
        from agent.reuse import lookup, same_task_note
        m = lookup(cfg, video, semantic if semantic.exists() else None)
        if m.get("match") == "identical":
            print(f"note: this video is identical to the example {m['example']}; the stage would reuse it without the model")
        elif m.get("match") == "same_task":
            user += "\n" + same_task_note(a.stage, m)
    lib = library_root(cfg)
    cats = getattr(S, 'TOOL_CATEGORIES', None)
    tools = BUILTIN_SCHEMAS + [tool_schema(t, cfg) for t in tool_catalog(load_index(lib), cats)]
    (out / "system.md").write_text(system)
    (out / "user.txt").write_text(user)
    (out / "tools.json").write_text(json.dumps(tools, indent=1))
    print(f"system {len(system)} chars, {len(tools)} tools -> {out}")


if __name__ == "__main__":
    main()
