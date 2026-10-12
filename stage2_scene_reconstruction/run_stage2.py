"""Stage 2: property-based simulation reconstruction. The video and Stage 1's scene_semantics.json in,
a simulation scene with the reconstructed, verified objects placed as in the video out.

    python stage2_scene_reconstruction/run_stage2.py dexagent.yaml
    python stage2_scene_reconstruction/run_stage2.py dexagent.yaml --video clip.mp4 --semantic runs/x/stage1/scene_semantics.json --name x
    python stage2_scene_reconstruction/run_stage2.py dexagent.yaml --egodex episode.hdf5 --name egodex_bottle

Writes <output_dir>/<name>/stage2/{scene.xml, camera.json, masks/, assets/, isaac/, frame0_compare.png,
scene_recon.json, trace.jsonl, new_tools.json}.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from agent.config import egodex_video, load_config, run_dir, semantics_file  # noqa: E402

TOOL_CATEGORIES = ["Perception & Grounding", "Measurements", "Object Reconstruction", "Scene Runtime & Rendering",
                   "Reconstruction Mechanism Tests", "Scene Assertions", "Scene Compile Checks"]


def stage_prompt() -> str:
    return (HERE / "prompts" / "stage2.md").read_text()


def user_message(video: Path, semantic: Path, workspace: Path, cfg: dict, egodex: Path | None) -> str:
    setup = cfg["inputs"].get("setup")
    cam = cfg.get("camera") or {}
    lines = [f"Workspace: {workspace}", f"Video: {video}", f"Stage 1 deliverable: {semantic}", f"Setup folder: {setup} (setup.yaml, scene.xml with the home keyframe and camera 'ego')",
             f"Camera source from the run configuration: {cam.get('source', 'setup')}"]
    if cam.get("source") == "given":
        lines.append("Camera given: " + json.dumps({k: cam.get(k) for k in ("width", "height", "fovy_deg", "intrinsics", "extrinsics")}))
    if egodex:
        lines.append(f"EgoDex episode: {egodex} (use read_egodex_episode for the frames and the camera; the video above is its mp4)")
    envs = cfg.get("environments") or {}
    if envs.get("isaac"):
        lines.append(f"Isaac Sim Python for build_isaac_scene: {envs['isaac']} (pass it as isaac_python)")
    try:
        deformable = [o.get("id") for o in json.loads(Path(semantic).read_text()).get("objects", []) if o.get("type") == "deformable"]
    except (OSError, ValueError):
        deformable = []
    if deformable:
        lines.append(f"Deformable objects {deformable}: reconstructed and simulated in Isaac Sim (build_isaac_scene, step 9)"
                     + ("" if envs.get("isaac") else "; no Isaac Sim Python is configured (dexagent.yaml environments.isaac), so they "
                        "cannot be built here: say so and stop"))
    lines.append(f"Write scene.xml, camera.json, masks/, assets/, frame0_compare.png and scene_recon.json into the workspace, verify, then call finish.")
    return "\n".join(lines)


def make_verifiers(cfg: dict, semantic: Path, lib: Path):
    from agent.library import call_tool, load_index
    index = load_index(lib)

    def source_reference_provenance_check(ws: Path):
        from agent.reference_evidence import check_workspace
        return check_workspace(ws, cfg["inputs"]["video"])

    def manifest_check(ws: Path):
        r = call_tool(lib, index, "scene_recon_manifest_check", {"path": str(ws / "scene_recon.json"), "scene_semantic_json": str(semantic),
                                                                  "library_index": str(lib / "index.json")}, cfg)
        return r.get("result") if r.get("ok") else {"ok": False, "issues": [r.get("error")]}

    def scene_check(ws: Path):
        man_p = ws / "scene_recon.json"
        if not man_p.exists():
            return {"ok": False, "issues": ["scene_recon.json is missing"]}
        man = json.loads(man_p.read_text())
        scene = ws / man.get("scene_xml", "scene.xml")
        objs = man.get("objects", [])
        # keyed by the object's root body (an id need not name its body); a deformable with a MuJoCo body is checked too
        # (thin objects are scored by their centreline). A group object (`parts`: its separate pieces) is keyed by its id
        # and scored by the union silhouette of its pieces against its one mask; every piece must exist
        from tool_library.common.piece_groups import group_bodies, is_group
        groups = {o["id"]: group_bodies(o) for o in objs if is_group(o)}
        masks = {(o["id"] if o["id"] in groups else (o.get("root_body") or o["id"])): str(ws / o["mask"]) for o in objs
                 if o.get("mask") and (ws / o["mask"]).exists() and (o.get("type") != "deformable" or o.get("root_body"))}
        required = list(dict.fromkeys([o["root_body"] for o in objs if o.get("root_body")] + [b for g in groups.values() for b in g]))
        articulated = [o["root_body"] for o in objs if o.get("type") == "articulated" and o.get("root_body")]
        args = {"scene": str(scene), "required": required, "articulated": articulated}
        if masks:
            args["masks"] = masks; args["camera"] = "video_cam"
            if groups:
                args["groups"] = groups
            # an object declared hidden at frame 0 must render (near) invisible instead of matching its (empty) mask
            from tool_library.common.hidden_objects import is_hidden
            hidden = [(o["id"] if o["id"] in groups else (o.get("root_body") or o["id"])) for o in objs if is_hidden(o) and o.get("mask")]
            if hidden:
                args["hidden"] = hidden
            if (ws / "masks" / "hands.png").exists():
                args["ignore_mask"] = str(ws / "masks" / "hands.png")      # the demonstrator's hands cover objects at frame 0
        r = call_tool(lib, index, "scene_qc", args, cfg)
        if not r.get("ok"):
            return {"ok": False, "issues": [r.get("error")]}
        rep = r["result"]
        issues = []
        for name in rep.get("failed", []):
            sub = rep.get("checks", {}).get(name, {})
            detail = sub.get("issues") or sub.get("failures") or sub.get("missing") or sub.get("message") or ""
            why = {k: v.get("reason") for k, v in (sub.get("hidden") or {}).items() if not v.get("ok")}
            if why:                                  # a hidden-at-frame-0 object: say why its visibility test failed
                detail = {"failures": detail, "hidden_at_frame0": why}
            issues.append(f"scene_qc.{name} failed: {json.dumps(detail, default=str)[:700 if why else 400]}")
        return {"ok": rep.get("ok", False) and not issues, "issues": issues, "report": {k: v for k, v in rep.items() if k != "checks"}}

    def gate_suite(ws: Path):
        # every scene assertion and compile check: the robot and physics are the setup's, the initial state holds, the
        # objects are plausible and seen where the video shows them
        r = call_tool(lib, index, "stage2_gate_suite", {"workspace": str(ws), "scene_semantic_json": str(semantic),
                                                        **({"setup_dir": cfg["inputs"]["setup"]} if cfg["inputs"].get("setup") else {})}, cfg)
        if not r.get("ok"):
            return {"ok": False, "issues": [r.get("error")]}
        rep = r["result"]
        return {"ok": rep["ok"], "issues": rep["issues"], "warnings": rep.get("warnings", []),
                "report": {k: v.get("ok") for k, v in rep.get("reports", {}).items()}}

    manifest_check.__name__ = "scene_recon_manifest_check"
    scene_check.__name__ = "scene_qc"
    gate_suite.__name__ = "stage2_gate_suite"
    return [source_reference_provenance_check, manifest_check, scene_check, gate_suite]


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="DexAgent Stage 2")
    ap.add_argument("config", nargs="?", default=os.environ.get("DEXAGENT_CONFIG", str(REPO / "dexagent.yaml")))
    ap.add_argument("--video"), ap.add_argument("--semantic"), ap.add_argument("--name"), ap.add_argument("--egodex")
    ap.add_argument("--max-rounds", type=int)
    ap.add_argument("--llm", default="config", help="config (the YAML's agent block) or mock:<turns.json>")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.video:
        cfg["inputs"]["video"] = str(Path(a.video).resolve())
    if a.name:
        cfg["run"]["name"] = a.name
    if a.egodex:
        egodex_video(cfg, a.egodex)
    video = Path(cfg["inputs"]["video"])
    if not video.exists():
        sys.exit(f"inputs.video does not exist: {video}")
    ws = run_dir(cfg, "stage2")
    semantic = Path(a.semantic).resolve() if a.semantic else semantics_file(ws.parent / "stage1")
    if not semantic.exists():
        sys.exit(f"Stage 1 deliverable not found: {semantic} (run Stage 1 first or pass --semantic)")
    from agent.input_identity import video_verifier, input_failure
    identity_gate = video_verifier(video, semantic)
    identity_report = identity_gate(ws)
    if identity_report.get("ok") is not True:
        input_failure(ws, identity_report)
        return 1
    from agent.reference_evidence import begin, revalidate_source_recipes
    from agent.library import library_root
    lib = library_root(cfg)
    begin(ws, video)
    # A disk recipe is only a proposal: independently recompute supported
    # source masks before example reuse or fresh-session carryover is trusted.
    revalidate_source_recipes(ws, video, library=lib, cfg=cfg)
    egodex = Path(a.egodex).resolve() if a.egodex else (Path(cfg["inputs"]["egodex_hdf5"]).resolve() if cfg["inputs"].get("egodex_hdf5") else None)
    from agent.reuse import try_reuse
    verifiers = make_verifiers(cfg, semantic, lib)
    verifiers.insert(0, identity_gate)
    # a stage re-opened by a cross-stage repair (or downstream of one) reads why first and never reuses an example: the input
    # it would reuse is the one found wrong
    from agent import repair
    note = repair.take_note(ws) if repair.has_note(ws) else ""
    reused, reuse_note = (None, None) if note else try_reuse(cfg, "stage2", ws, video, scene_semantic=semantic, verifiers=verifiers)
    if reused is not None:
        return 0
    if a.llm.startswith("mock:"):
        from agent.llm import load_mock
        llm = load_mock(a.llm[5:], "stage2")
    else:
        from agent.budget import BudgetError
        from agent.llm import make_llm
        try:
            llm = make_llm(cfg)
        except BudgetError as exc:                    # a model API run needs a spending cap; refused before any call
            print(f"  {exc}", flush=True)
            (ws / "run.json").write_text(json.dumps({"finished": False, "stage": "stage2", "workspace": str(ws), "failure": str(exc)}, indent=1))
            return 1
    envs = cfg.get("environments") or {}
    if envs.get("isaac"):
        os.environ.setdefault("DEXAGENT_ISAAC_PYTHON", str(envs["isaac"]))
    from agent.runner import StageRunner
    s2 = cfg.get("stage2") or {}
    cfg["_verification_inputs"] = [str(p) for p in (video, semantic, cfg["inputs"].get("setup"), egodex) if p]
    runner = StageRunner(cfg, llm, ws, stage_prompt(), verifiers, categories=TOOL_CATEGORIES,
                         max_rounds=a.max_rounds or s2.get("max_rounds"), stage="stage2")
    msg = user_message(video, semantic, ws, cfg, egodex)
    if reuse_note:
        msg += "\n" + reuse_note
    msg += note
    out = runner.run(msg)
    (ws / "run.json").write_text(json.dumps(out, indent=1, default=str))
    return 0 if out["finished"] else 1


if __name__ == "__main__":
    sys.exit(main())
