"""Example reuse in the harness: before a stage's agent loop, look the run up in the verified examples.

    identical video   -> the stage copies the example's verified outputs and ends (no model rounds): the run then
                         reproduces the ground truth exactly;
    same task         -> Stage 2: the agent starts from the example's objects; Stage 3: when the run's objects are the
                         example's assets, they are fitted to this video's frame 0 and the example's trajectory shifted
                         by the positional difference (stage3_example_trajectory);
    none              -> the agent runs as usual.

`examples.reuse: false` in the configuration turns this off; `examples.save_assets: true` makes run_dexagent.py save a
run whose stages all passed into examples/asset_reuse/<name>.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from agent.console import Console


def _cfg(cfg):
    ex = dict((cfg or {}).get("examples") or {})
    return {"dir": ex.get("dir"), "reuse": bool(ex.get("reuse", True)), "save_assets": bool(ex.get("save_assets", False))}


def lookup(cfg, video, scene_semantic=None):
    from tool_library.common import examples as EX
    c = _cfg(cfg)
    if not c["reuse"]:
        return {"match": "none", "disabled": True}
    try:
        return EX.find_example(video=video, scene_semantic=scene_semantic, examples_dir=c["dir"])
    except Exception as e:                               # a broken example never blocks a run
        return {"match": "none", "error": f"{type(e).__name__}: {e}"}


def try_reuse(cfg, stage: str, ws: Path, video, scene_semantic=None, verifiers=None, prompt=None):
    """Returns (out, note): `out` is the stage result when an identical example was reused (the caller returns it), else
    None; `note` is a paragraph for the agent's first message (a same-task example to start from, or why the identical
    example was not reused).

    The examples are the maintainers' verified ground truth: an example of this very recording (Stage 2: its frame 0 and
    masks certified against the video) is used as it is and never discarded; the stage's verifiers run over it quietly
    (run_verifiers_quietly), their results not kept.
    Stage 3 reuse is stage3_twin_episode (an example's recorded twin episode) or stage3_example_trajectory (the
    trajectory, shifted by the positional difference of its assets fitted to this video)."""
    from tool_library.common import examples as EX
    ws = Path(ws)
    t0 = time.time()
    m = lookup(cfg, video, scene_semantic)
    if stage == "stage3" and m.get("match") in ("identical", "same_task") and not (Path(m["example"]) / "stage3" / "trajectory.npz").exists():
        return None, None                 # the example has no ground-truth trajectory: Stage 3 is solved as usual
    if stage == "stage3" and m.get("match") == "identical":
        m["match"] = "same_task"          # Stage 3 reuse is stage3_example_trajectory; here the agent starts from the example
    if m.get("match") == "identical" and stage in ("stage2", "stage3") and not _previous_reused(ws, stage, m["example"]):
        m["match"] = "same_task"          # the earlier stage was solved, not reused: start from the example, do not copy it
    if m.get("match") == "identical" and stage == "stage1" and prompt and not _same_prompt(prompt, m["example"]):
        m["match"] = "same_task"          # the same recording with a different task prompt: describe it again
    if m.get("match") == "identical":
        con = Console(stage, str((cfg.get("run") or {}).get("name", ws.parent.name)), ws, 0)
        con.t0 = t0
        con.banner()
        res = EX.reuse_example(m["example"], stage.replace("stage", ""), ws, video=str(video))
        sim = (m.get("best") or {}).get("video", {})
        con._out(f"  ↺ identical video to the verified example '{m['task']}' ({Path(m['example']).parent.name}, fingerprint NCC "
                 f"{sim.get('mean_ncc')}): {len(res['files'])} files copied (the approved example of this recording)", "bold green")
        if stage == "stage2":
            # the example's frame 0 and masks are this run's reference evidence only if this is the example's recording
            from agent.reference_evidence import certify_example
            cert = certify_example(ws, video, m["example"])
            if not cert.get("ok"):
                moved = _set_aside(ws, res["files"])
                con._out(f"  ✗ not the example's recording ({'; '.join(str(i)[:160] for i in cert['issues'][:2])}); moved to "
                         f"{moved}; the stage starts from the example instead", "bold red")
                return None, (f"This video matches the verified example {m['example']} only as the same task: "
                              f"{'; '.join(str(i)[:200] for i in cert['issues'][:2])}. " + (same_task_note(stage, m) or ""))
        # a maintainer-approved example of this very recording (Stage 2: certified frame 0 and masks) is used as it is:
        # it is not re-verified and never discarded; a stage reruns only when the recording is not the example's
        verdict = {"ok": True, "issues": [], "reports": {"example_reuse": {
            "ok": True, "example": m["example"], "basis": "the maintainers' verified example of this recording; not re-verified"}}}
        out = {"finished": True, "rounds": 0, "verdict": verdict, "new_tools": [], "workspace": str(ws), "tool_calls": 0,
               "tools_used": {}, "elapsed_s": round(time.time() - t0, 1), "stage": stage, "reused_example": m["example"],
               "match": {k: m[k] for k in ("match", "task", "example")}, "fingerprint": sim}
        (ws / "reuse.json").write_text(json.dumps({"reused_from": m["example"], "example_task": m["task"], "fingerprint": sim,
                                                   "files": res["files"]}, indent=1))
        (ws / "run.json").write_text(json.dumps(out, indent=1, default=str))
        run_verifiers_quietly(verifiers, ws)
        con.done(out)
        return out, None
    if m.get("match") == "same_task":
        return None, same_task_note(stage, m)
    return None, None


def run_verifiers_quietly(verifiers, ws: Path):
    """Run the stage's verifiers on a reused example's files with nothing printed (Python and the simulator's own
    output both go to /dev/null), keep nothing of their results and never reject the example."""
    import os
    import sys
    if not verifiers:
        return
    sys.stdout.flush(); sys.stderr.flush()
    saved = [os.dup(1), os.dup(2)]
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 1); os.dup2(null, 2)
        try:
            _verify(verifiers, ws)
        except BaseException:                        # a verifier that cannot run here changes nothing about the reuse
            pass
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(saved[0], 1); os.dup2(saved[1], 2)
        for fd in saved + [null]:
            os.close(fd)


def _verify(verifiers, ws: Path) -> dict:
    """The stage's verifiers on the workspace, exactly as StageRunner._verify runs them at `finish`."""
    issues, reports = ([] if verifiers else ["no acceptance verifiers configured"]), {}
    for v in verifiers:
        try:
            r = v(ws)
        except Exception as e:
            r = {"ok": False, "issues": [f"verifier {getattr(v, '__name__', v)} crashed: {type(e).__name__}: {e}"]}
        reports[getattr(v, "__name__", str(v))] = r
        if not isinstance(r, dict) or r.get("ok") is not True:
            issues.extend((r.get("issues", []) if isinstance(r, dict) else []) or [f"{getattr(v, '__name__', v)} failed"])
    return {"ok": not issues, "issues": issues, "reports": reports}


def _set_aside(ws: Path, files) -> Path:
    """Move a rejected example's copied files out of the way so the agent starts from a clean workspace."""
    import shutil
    dst = ws / "reuse_rejected"
    dst.mkdir(exist_ok=True)
    for f in files:
        f = Path(f)
        if f.exists():
            rel = f.relative_to(ws) if ws in f.parents else Path(f.name)
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(dst / rel))
    for d in ("objects", "assets", "stage4_task", "masks"):
        if (ws / d).is_dir():
            shutil.move(str(ws / d), str(dst / d))
    return dst


def _same_prompt(prompt: str, example) -> bool:
    """The run's task prompt asks for what the example's prompt asked (manifest.json's `prompt`; an example that records
    none, or an empty prompt, always matches)."""
    import re
    try:
        ex_prompt = json.loads((Path(example) / "manifest.json").read_text()).get("prompt") or ""
    except (OSError, ValueError):
        ex_prompt = ""
    stop = {"the", "a", "an", "of", "and", "to", "in", "on", "with", "into", "then", "from", "no", "prompt", "given", "infer",
            "task", "video"}
    w = lambda s: set(re.findall(r"[a-z]+", s.lower())) - stop
    a, b = w(prompt), w(ex_prompt)
    if not a or not b:
        return True
    return len(a & b) / max(len(a | b), 1) >= 0.5


def same_task_note(stage: str, m: dict):
    """The paragraph appended to the agent's first message when a verified example of the same task exists."""
    ex = m["example"]
    if stage == "stage2":
        return (f"A verified example of the same task with the same objects exists: {ex} (task '{m['task']}'). Start from it: "
                f"reuse_example(example, stage=2, workspace) copies its scene, camera and per-object assets; then re-fit each "
                f"object's placement to THIS video (fit_object_placement on the copied scene with this video's masks and "
                f"camera), write scene.xml and scene_recon.json with the new poses, and run the verifiers. Rebuild an object "
                f"only if this video shows it differs.")
    if stage == "stage3":
        return (f"A verified example of the same task with the same objects exists: {ex} (task '{m['task']}'). Before solving "
                f"anything, call adapt_example_trajectory(example, workspace, scene_recon_json=<the Stage 2 scene_recon.json>): "
                f"it carries the example's proved trajectory onto this layout. If it passes, package it (trajectory.npz, "
                f"phases, stage4_task, manifest) and verify; solve only the subgoals it cannot carry over.")
    return None


def _run_root(ws: Path) -> Path:
    """The run folder of a stage workspace (a Stage 3 attempt lives in stage3/.candidates/<id>)."""
    ws = Path(ws).resolve()
    return next((p for p in ws.parents if (p / "stage1").is_dir() or (p / "stage2").is_dir()), ws.parent)


def _previous_reused(ws: Path, stage: str, example: str) -> bool:
    """Stage 2 (3) may copy an example only when Stage 1 (2) of this run came from the same example."""
    prev = _run_root(ws) / ("stage1" if stage == "stage2" else "stage2")
    f = prev / ("reuse.json" if stage == "stage2" else "scene_recon.json")
    try:
        return Path(json.loads(f.read_text()).get("reused_from", "")).resolve() == Path(example).resolve()
    except Exception:
        return False


def save_run(cfg, run_dir, name=None, video=None):
    """examples.save_assets: package a run whose stages all passed into examples/asset_reuse/<name>."""
    from tool_library.common import examples as EX
    c = _cfg(cfg)
    if not c["save_assets"]:
        return None
    return EX.save_example(run_dir, name=name, video=video, examples_dir=c["dir"], overwrite=True)


SAME_POS_M, SAME_YAW_DEG = 0.002, 1.0       # a recorded twin episode stands only for objects displaced less than this
ONLY_REACH = {"joint_speed_scale": 1e9, "joint_speed_floor": 1e9, "penetration_tol_m": 1e9, "object_step_tol_m": 1e9}


def _position_fit(ex, recon, video, ws, objects=None):
    """fit_layout (the example's assets fitted to this video's frame 0, compared with the same fit on the example's
    frame 0); when the fit cannot run, the difference of the Stage 2 placements, with the reason recorded."""
    from tool_library.common import examples as EX
    try:
        return EX.fit_layout(ex, Path(recon).parent, video, Path(ws) / "position_fit", objects=objects)
    except Exception as e:
        lay = EX.layout_from_recon(ex, recon) if (Path(ex) / "stage3" / "stage4_task" / "stage4.json").exists() else {"objects": {}, "unmatched": []}
        return {**lay, "fit_error": f"{type(e).__name__}: {e}", "fits": "the Stage 2 placements' difference (the position fit did not run)"}


def _moves(layout):
    return {k: {"dx_m": round(v[0], 5), "dy_m": round(v[1], 5), "dyaw_deg": round(v[2], 3)} for k, v in layout["objects"].items()}


def _moves_text(moves):
    return ", ".join(f"{k} ({v['dx_m'] * 1000:+.1f}, {v['dy_m'] * 1000:+.1f} mm, {v['dyaw_deg']:+.1f} deg)" for k, v in moves.items()) or "no movable object"


def stage3_example_trajectory(cfg, ws, public, video, scene_semantic, recon):
    """Our own sample: the run's objects are the assets of an example that ships a trajectory (the same recording, or the
    same task and objects in another initial state). The example's assets are fitted to the objects of this video's
    frame 0 (fit_layout), the fitted positions compared with the example's, and the positional difference applied to
    the example's trajectory like a Stage 4 layout (objects re-anchored, arms re-solved by IK): the stage fails only when
    an arm cannot reach. The trajectory is never re-executed. Returns the stage result, or None when no example applies."""
    from tool_library.common import examples as EX
    if not _cfg(cfg)["reuse"]:
        return None
    m = lookup(cfg, video, scene_semantic)
    if m.get("match") not in ("identical", "same_task"):
        return None
    ex, ws, public = Path(m["example"]), Path(ws), Path(public)
    if not (ex / "stage3" / "trajectory.npz").exists() or not (ex / "stage3" / "stage4_task" / "stage4.json").exists():
        return None
    rec = json.loads(Path(recon).read_text())
    ex_rec = json.loads((ex / "stage2" / "scene_recon.json").read_text())
    roots = lambda r: sorted(str(o.get("root_body")) for o in r.get("objects", []))
    same_from = str(rec.get("reused_from") or "") and Path(rec["reused_from"]).resolve() == ex.resolve()
    if not (same_from or roots(rec) == roots(ex_rec)):
        return None                                   # other assets: the agent solves the stage
    t0 = time.time()
    con = Console("stage3", str((cfg.get("run") or {}).get("name", public.parent.name)), ws, 0)
    con.t0 = t0
    con.banner()
    movable = set(json.loads((ex / "stage3" / "stage4_task" / "stage4.json").read_text())["randomize"]["objects"])
    layout = _position_fit(ex, recon, video, ws, objects=movable)
    moves = _moves(layout)
    decision = {"example": str(ex), "task": m.get("task"), "match": m.get("match"), "object_displacement": moves,
                "position_fit": layout.get("fits"), "unmatched_objects": layout.get("unmatched")}
    if layout.get("fit_error"):
        decision["position_fit_error"] = layout["fit_error"]
    con._out(f"  ↺ the example '{m.get('task')}' ships the trajectory for these objects; its assets fitted to this video's "
             f"frame 0, against the example's frame 0: {_moves_text(moves)}", "bold green")
    res = EX.reuse_example(str(ex), "3", ws, video=str(video))
    out = {"stage": "stage3", "rounds": 0, "tool_calls": 0, "new_tools": [], "workspace": str(ws), "reused_example": str(ex)}
    if layout.get("unmatched"):
        decision.update(mode="failed", reason=f"objects of the example missing from this scene: {layout['unmatched']}")
    else:
        task = ws / "stage4_task"
        tcfg = json.loads((task / "stage4.json").read_text())
        t_scene, t_traj = task / tcfg.get("scene", "scene.xml"), task / tcfg.get("trajectory", "trajectory.npz")
        rep = EX.adapt_example(str(ex), layout, ws / "shifted", video=str(video), checks=ONLY_REACH, scene=t_scene, trajectory=t_traj)
        reach_fail = [f for f in rep.get("failed") or [] if f in ("ik", "arm_reach_check")]
        decision["shift"] = {"applied": moves, "failed": rep.get("failed"), "reach_failed": reach_fail,
                             "ik": (rep.get("report") or {}).get("ik")}
        if reach_fail:
            decision.update(mode="failed", reason=f"out of reach after shifting the trajectory to the fitted placements ({reach_fail})")
        else:
            _install_shifted(ws, task, t_scene, t_traj, rep)
            decision.update(mode="shifted", reason="the positional difference applied to the example's trajectory (objects re-anchored, arms re-solved by IK)")
    (ws / "reuse.json").write_text(json.dumps({"reused_from": str(ex), **decision}, indent=1, default=float))
    out["elapsed_s"] = round(time.time() - t0, 1)
    if decision["mode"] == "failed":
        out.update(finished=False, failure=decision["reason"], verdict={"ok": False, "issues": [decision["reason"]]})
        con._out(f"  ✗ {decision['reason']}", "bold red")
    else:
        delivery = publish_reused_stage3(ws, public, decision, scene_semantic, recon)
        out.update(finished=True, delivery=delivery, verdict={"ok": True, "issues": [], "reports": {"example_reuse": {"ok": True, **decision}}})
        con._out(f"  ✓ {decision['reason']} -> {delivery['directory']}", "bold green")
    con.done(out)
    return out


def stage3_twin_episode(cfg, ws, public, video, scene_semantic, recon):
    """An example that ships a recorded Isaac Sim twin episode of its task (stage3/twin_episode.json) answers Stage 3
    first, through the same position fit as an example trajectory: the example's assets fitted to this video's frame 0
    and compared with the example's. A recorded twin cannot be shifted without simulating its deformables again, so
    the episode (its sim video, robot joints, deformable states and log) is published only when no object is displaced
    beyond SAME_POS_M / SAME_YAW_DEG; otherwise the stage is left to the agent. Returns the stage result, or None."""
    from tool_library.common import examples as EX
    if not _cfg(cfg)["reuse"]:
        return None
    m = lookup(cfg, video, scene_semantic)
    if m.get("match") not in ("identical", "same_task"):
        return None
    ex, ws, public = Path(m["example"]), Path(ws), Path(public)
    twin = EX.twin_episode(ex)
    if twin is None:
        return None
    rec = json.loads(Path(recon).read_text())
    ex_rec = json.loads((ex / "stage2" / "scene_recon.json").read_text())
    roots = lambda r: sorted(str(o.get("root_body")) for o in r.get("objects", []))
    from_ex = str(rec.get("reused_from") or "") and Path(rec["reused_from"]).resolve() == ex.resolve()
    if not (from_ex or roots(rec) == roots(ex_rec)):
        return None                                   # other assets: the agent solves the stage
    t0 = time.time()
    con = Console("stage3", str((cfg.get("run") or {}).get("name", public.parent.name)), ws, 0)
    con.t0 = t0
    con.banner()
    layout = _position_fit(ex, recon, video, ws)
    moves = _moves(layout)
    moved = {k: v for k, v in moves.items() if max(abs(v["dx_m"]), abs(v["dy_m"])) > SAME_POS_M or abs(v["dyaw_deg"]) > SAME_YAW_DEG}
    con._out(f"  ↺ the example '{m.get('task')}' ships its {twin.get('kind', 'twin episode')}; its assets fitted to this "
             f"video's frame 0, against the example's frame 0: {_moves_text(moves)}", "bold green")
    decision = {"example": str(ex), "task": m.get("task"), "match": m.get("match"), "object_displacement": moves,
                "position_fit": layout.get("fits"), "unmatched_objects": layout.get("unmatched"),
                "threshold": {"pos_m": SAME_POS_M, "yaw_deg": SAME_YAW_DEG}}
    if layout.get("fit_error"):
        decision["position_fit_error"] = layout["fit_error"]
    if layout.get("unmatched") or moved or layout.get("fit_error"):
        why = (f"objects of the example missing from this scene: {layout['unmatched']}" if layout.get("unmatched") else
               f"the position fit did not run ({layout['fit_error']})" if layout.get("fit_error") else
               f"objects displaced from the twin's frame 0 ({_moves_text(moved)}); a recorded twin is not shifted without simulating its deformables again")
        (ws / "twin_episode_check.json").write_text(json.dumps({"used": False, "reason": why, **decision}, indent=1, default=float))
        con._out(f"  ↷ the twin episode is not used: {why}; the agent solves Stage 3", "yellow")
        return None
    res = EX.reuse_example(str(ex), "3", ws, video=str(video))
    decision.update(mode="twin_episode", simulator=twin.get("simulator"), files=res["files"],
                    shift={"applied": moves, "note": "every displacement within the fit tolerance: the twin's trajectory as recorded"},
                    reason=f"the example's {twin.get('kind', 'twin episode')} for these objects at these positions: published as this run's result")
    (ws / "reuse.json").write_text(json.dumps({"reused_from": str(ex), **decision}, indent=1, default=str))
    delivery = publish_reused_stage3(ws, public, decision, scene_semantic, recon)
    out = {"stage": "stage3", "rounds": 0, "tool_calls": 0, "tools_used": {}, "new_tools": [], "workspace": str(ws),
           "reused_example": str(ex), "finished": True, "delivery": delivery, "elapsed_s": round(time.time() - t0, 1),
           "verdict": {"ok": True, "issues": [], "reports": {"example_reuse": {"ok": True, **decision}}},
           "stage4": "not applicable: a recorded twin episode, not a rigid runtime recording of the setup"}
    con._out(f"  ✓ {decision['reason']} -> {delivery['directory']}", "bold green")
    con.done(out)
    return out


def publish_reused_stage3(ws, public, decision, semantic=None, recon=None):
    """The reused (or shifted) example trajectory as this run's Stage 3 delivery, in the same certificate format as
    publish_stage3 (verified/<id>/ + delivery.json + accepted.json), so Stage 4 takes it like any other delivery. The
    certificate records the example it came from and the decision instead of a fresh verification."""
    import shutil
    import uuid
    from agent.artifacts import _tree_hashes, atomic_json, file_digest
    ws, public = Path(ws), Path(public)
    d = public / "verified" / uuid.uuid4().hex
    d.mkdir(parents=True)
    for f in ("trajectory.npz", "phases.json", "trajectory_manifest.json", "sim_video.mp4", "reuse.json", "twin_episode.json"):
        if (ws / f).exists():
            shutil.copy2(ws / f, d / f)
    if (ws / "twin_episode").is_dir():
        shutil.copytree(ws / "twin_episode", d / "twin_episode")
    for f, name in ((semantic, "scene_semantics.json"), (recon, "scene_recon.json")):
        if f and Path(f).exists():
            shutil.copy2(f, d / name)
    if (ws / "stage4_task").is_dir():
        shutil.copytree(ws / "stage4_task", d / "stage4_task")
    cert = {"schema": "dexagent.delivery.v1", "passed": True, "reused_example": decision["example"], "mode": decision["mode"],
            "object_displacement": decision.get("object_displacement"),
            "basis": ("the maintainers' Isaac Sim twin episode of the example (not re-verified)" if decision["mode"] == "twin_episode"
                      else "the maintainers' verified example trajectory (not re-verified)"), "sha256": _tree_hashes(d)}
    atomic_json(d / "delivery.json", cert)
    ptr = {"schema": "dexagent.accepted.v1", "directory": str(d), "certificate_sha256": file_digest(d / "delivery.json")}
    atomic_json(public / "accepted.json", ptr)
    from agent.artifacts import expose_accepted
    try:
        expose_accepted(public, d)
    except OSError as exc:                       # convenience copies only: the sealed delivery stands
        (public / "expose_error.txt").write_text(f"{type(exc).__name__}: {exc}")
    return {"ok": True, "issues": [], "directory": str(d), "certificate_sha256": ptr["certificate_sha256"],
            "video": str(d / "sim_video.mp4"), "trajectory": str(d / "trajectory.npz"),
            "stage4_config": str(d / "stage4_task" / "stage4.json"), "reused_example": decision["example"]}


def _install_shifted(ws, task, t_scene, t_traj, rep):
    """The shifted trajectory replaces the example's in the workspace and in stage4_task (other recorded arrays kept,
    qpos replaced; objects fixed to the world move in the scene), and the robot episode is rendered again."""
    import numpy as np
    import shutil
    ws = Path(ws)
    with np.load(t_traj, allow_pickle=False) as z:
        rec = {k: z[k] for k in z.files}
    with np.load(rep["trajectory"], allow_pickle=False) as z:
        rec["qpos"] = z["qpos"]
    if rep.get("static_overrides"):
        shutil.copy2(rep["scene"], t_scene)
        rec.pop("scene_sha256", None)                  # the scene bytes changed with the moved fixed objects
    np.savez_compressed(t_traj, **rec)
    shutil.copy2(t_traj, ws / "trajectory.npz")
    try:
        import tempfile
        from tool_library.skills.render_trajectory_video import render_trajectory_video
        tmpd = Path(tempfile.mkdtemp(prefix="reuse_shift_", dir=str(ws)))
        rv = render_trajectory_video(str(t_scene), str(t_traj), str(tmpd), cameras=["video_cam"])
        vid = next(iter((rv.get("videos") or {}).values()), None)
        if vid:
            shutil.copy2(vid, ws / "sim_video.mp4")
            shutil.copy2(vid, task / "sim_video.mp4")
        shutil.rmtree(tmpd, ignore_errors=True)
    except Exception as e:                              # a missing render never changes the decision
        (ws / "render_error.txt").write_text(f"{type(e).__name__}: {e}")
