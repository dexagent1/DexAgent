"""Will Stage 4 accept this task folder?"""
from __future__ import annotations

TOOL = {
    "name": "stage4_handoff_check",
    "title": "Stage 4 handoff check",
    "type": "verifier",
    "category": "Task & Rollout Gates",
    "summary": "Validate a Stage 3 task folder the way Stage 4 will read it: files present, the scene compiles with the "
               "trajectory's qpos width, every segment phase matches frames, anchors change only through blends, carriers "
               "and releases resolve, cameras and table are set, and the video camera matches Stage 2's camera.json.",
    "description": (
        "Loads stage4.json with Stage 4's own config loader and builds its Episode object (the step that re-anchors "
        "frames), so the same assertions Stage 4 makes at the start of `randomize` run here with their messages. On "
        "top it checks that the trajectory's labels follow the 'side: phase' convention, that every segment's phase "
        "patterns match a frame, that every randomized object is a body of the scene, that the human video exists and "
        "decodes (its fps and duration go into the summary; they are not judged), that the cameras and the table are "
        "set, and that the ego camera in the config (pose, field of view, fx, fy, cx, cy) equals Stage 2's camera.json. "
        "Warnings of Stage 4's Episode (such as a hand never anchored to an object it holds) are passed on in "
        "summary.warnings without failing the check. With draws > 0 it also runs "
        "stage4_randomization_gate (that many randomize draws with Stage 4's own code) and fails when fewer than "
        "min_accepted layouts would be kept (default the gate's: two thirds of the draws), so a task Stage 4 cannot "
        "randomize is caught at the handoff. It also reports whether the folder hands off the executable Stage 3 plan "
        "(plan.json; the rigid runtime does not re-execute it) and checks that every entry of it names "
        "bodies of the scene. A task with deformable objects simulated in Isaac Sim (stage4.json `deformables`, written "
        "by execute_isaac_deformables) is checked too: deformables.json, the Isaac recording, its report (which must "
        "have passed) and the mirror exist; the report's sha256 binding matches the recording and the mirror files; the "
        "mirror's qpos width is scene_mirror.xml's nq and its frames the trajectory's; a randomized cord is named as a "
        "cord, not as a body of the scene; and Isaac Sim's Python is available to Stage 4 (DEXAGENT_ISAAC_PYTHON or "
        "stage4.json isaac.python). Returns ok with the list of issues."
    ),
    "inputs": [
        {"name": "task_dir", "type": "str", "description": "Folder with stage4.json (from write_stage4_config)."},
        {"name": "camera_json", "type": "str | None", "description": "Stage 2 camera.json to compare the ego camera with."},
        {"name": "draws", "type": "int", "description": "Randomize draws to try with stage4_randomization_gate (default 0: config checks only)."},
        {"name": "min_accepted", "type": "int | None", "description": "Kept layouts the draws must yield (default: the randomization gate's, two thirds of the draws)."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok, issues [...], summary {frames, fps, sides, objects, segments, labels, randomization, deformables}."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo", "Isaac Sim"],
    "tags": ["stage3", "stage4", "handoff", "config", "deformable"],
    "used_in": ["Open the Bottle", "Giftbox Packing", "Place Toy in Drawer", "Battery Insertion", "Place Cup in Bowl", "Computer Installation"],
    "status": "tested",
    "entrypoint": "stage4_handoff_check",
}

import json
import sys
from pathlib import Path

import mujoco
import numpy as np


def stage4_handoff_check(task_dir, camera_json=None, draws=0, min_accepted=None):
    if int(draws) < 0:
        raise ValueError(f"draws must be >= 0, got {draws}")
    task = Path(task_dir).resolve()
    issues, summary = [], {}
    cfg_p = task / "stage4.json"
    if not cfg_p.exists():
        return {"ok": False, "issues": [f"{cfg_p} is missing"], "summary": {}}
    repo = Path(__file__).resolve().parents[2]
    s4 = repo / "stage4_robot_data_generation"
    if str(s4) not in sys.path:
        sys.path.insert(0, str(s4))
    try:
        from stage4.config import load_config
        from stage4.randomize import Episode
    except Exception as e:                                  # pragma: no cover
        return {"ok": False, "issues": [f"cannot import Stage 4 ({e})"], "summary": {}}
    try:
        cfg = load_config(cfg_p)
    except AssertionError as e:
        return {"ok": False, "issues": [f"config: {e}"], "summary": {}}
    try:
        ep = Episode(cfg)
        summary.update({"frames": ep.T, "fps": ep.fps, "sides": sorted(ep.sides), "objects": {n: o.get("carrier") for n, o in ep.objects.items()},
                        "labels": sorted(set(ep.labels), key=ep.labels.index)[:40],
                        "relations": {n: f"{o['relation'] or 'rests with'} {o['rest']}" for n, o in ep.objects.items() if o.get("rests_from") is not None},
                        "warnings": list(getattr(ep, "warnings", []))})
    except (AssertionError, ValueError) as e:
        issues.append(f"stage4 episode: {e}")
    except Exception as e:
        issues.append(f"stage4 episode: {type(e).__name__}: {e}")
    # label convention and segment coverage
    z = np.load(cfg.path("trajectory"), allow_pickle=True)
    labels = [str(x) for x in z["labels"]] if "labels" in z.files else []
    bad = [l for l in set(labels) if ": " not in l]
    if bad:
        issues.append(f"labels not 'side: phase': {bad[:5]}")
    import fnmatch
    for side, segs in cfg["segments"].items():
        for seg in segs:
            for pat in seg.get("phases", []):
                if not any(fnmatch.fnmatchcase(l, pat) for l in labels):
                    issues.append(f"segments[{side}]: phase pattern {pat!r} matches no frame")
    # deformables simulated in Isaac Sim: the hand-off files, their binding and the mirror
    from stage4 import deformables as DF
    dissues, dfm = DF.verify_task(cfg, cfg.path("trajectory"))
    issues += dissues
    cords = set(dfm["cords"]) if dfm else set()
    if dfm:
        data = dfm["data"]
        summary["deformables"] = {"cords": {n: {"n_links": c.get("n_links"), "preset": c.get("preset")} for n, c in dfm["cords"].items()},
                                  "grasps": len(data.get("grasps") or []), "events": len(data.get("events") or [])}
        try:
            summary["deformables"]["isaac_python"] = DF.isaac_python(cfg)
        except ValueError as e:
            issues.append(str(e))
    # objects must be free or static-on-world (or a cord simulated in Isaac Sim)
    model = mujoco.MjModel.from_xml_path(str(cfg.path("scene")))
    for name in cfg["randomize"]["objects"]:
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if b < 0 and name not in cords:
            issues.append(f"randomize.objects: no body {name!r}")
    # human video
    hv = cfg.path("human_video")
    if not hv.exists():
        issues.append("human_video missing")
    else:
        try:
            import imageio.v2 as iio
            rd = iio.get_reader(str(hv)); meta = rd.get_meta_data(); rd.close()
            summary["human_video"] = {"fps": meta.get("fps"), "duration_s": meta.get("duration")}
        except Exception as exc:
            issues.append(f"human_video cannot be decoded: {type(exc).__name__}: {exc}")
    # camera agreement
    if camera_json:
        cam = json.loads(Path(camera_json).read_text()); ego = cfg["cameras"]["ego"]
        fovy = cam.get("fovy_deg")
        if fovy is None and cam.get("fy") is not None and cam.get("height") is not None:
            fovy = float(np.degrees(2 * np.arctan(0.5 * float(cam["height"]) / float(cam["fy"]))))
        if (np.linalg.norm(np.asarray(cam["pos"]) - np.asarray(ego["pos"])) > 1e-6
                or fovy is None or ego.get("fovy") is None or abs(float(fovy) - float(ego["fovy"])) > 1e-6):
            issues.append("cameras.ego differs from Stage 2 camera.json")
        # the full intrinsics Stage 4 renders with (non-square pixels, off-centre principal point), both at the output
        # size: a quarter pixel apart at most (a config that gives only fovy matches a square, centred camera.json)
        if cam.get("fx") is not None and cam.get("fy") is not None:
            from stage4.config import intrinsics
            W, H = int(cfg["cameras"]["width"]), int(cfg["cameras"]["height"])
            ref = {k: cam[k] for k in ("fx", "fy", "cx", "cy") if cam.get(k) is not None}
            ref["image_size"] = [int(cam.get("width", W)), int(cam.get("height", H))]
            try:
                d_px = max(abs(a - b) for a, b in zip(intrinsics(ego, W, H), intrinsics(ref, W, H)))
            except ValueError as e:
                d_px, why = None, str(e)
            if d_px is None:
                issues.append(f"cameras.ego intrinsics cannot be compared with Stage 2 camera.json: {why}")
            elif d_px > 0.25:
                issues.append(f"cameras.ego intrinsics (fx, fy, cx, cy) differ from Stage 2 camera.json by {d_px:.2f} px "
                              f"at {W} x {H}: rewrite the task folder with write_stage4_config")
        q0, q1 = np.asarray(cam.get("quat_wxyz"), float), np.asarray(ego.get("quat_wxyz"), float)
        if (q0.shape != (4,) or q1.shape != (4,) or not np.isfinite(q0).all() or not np.isfinite(q1).all()
                or min(np.linalg.norm(q0), np.linalg.norm(q1)) < 1e-12
                or 1 - abs(float(np.dot(q0 / np.linalg.norm(q0), q1 / np.linalg.norm(q1)))) > 1e-10):
            issues.append("cameras.ego orientation differs from Stage 2 camera.json")
    for k in ("ego", "views"):
        if k not in cfg["cameras"]:
            issues.append(f"cameras.{k} missing")
    if not cfg.get("table"):
        issues.append("table missing")
    summary["segments"] = {s: len(v) for s, v in cfg["segments"].items()}
    # the executable Stage 3 plan (handed off; the rigid runtime does not re-execute it)
    plan_p = cfg.path_of((cfg.get("reexecute") or {}).get("plan"))
    if plan_p is None:
        summary["plan"] = "not handed off: Stage 4 can only warp this episode (write_stage4_config writes plan.json from the execution folder)"
    elif not Path(plan_p).exists():
        issues.append(f"reexecute.plan {plan_p} does not exist")
    else:
        try:
            pl = json.loads(Path(plan_p).read_text())
            ents = [e for e in pl.get("subgoals", []) if isinstance(e, dict)]
            subs = [h for e in ents for h in ((e.get("hands") or {}).values() if e.get("motion") == "both" else [e])]
            names = {n for h in subs for n in [h.get("object"), h.get("on"), h.get("into"), h.get("over")] if isinstance(n, str)}
            missing = sorted(n for n in names if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) < 0)
            if missing:
                issues.append(f"plan.json names bodies the scene does not have: {missing[:5]}")
            unanchored = sum(1 for h in subs for p in ([h["pose"]] if isinstance(h.get("pose"), dict) else h.get("pose") or [])
                             if isinstance(p, dict) and not p.get("anchor"))
            summary["plan"] = {"entries": len(ents), "poses_without_anchor": unanchored}
        except Exception as e:
            issues.append(f"plan.json unreadable: {type(e).__name__}: {e}")
    if int(draws) > 0 and not issues:
        from tool_library.verifiers.stage4_randomization_gate import stage4_randomization_gate
        g = stage4_randomization_gate(str(task), draws=int(draws), min_accepted=None if min_accepted is None else min(int(min_accepted), int(draws)))
        summary["randomization"] = {k: g.get(k) for k in ("accepted", "draws", "min_accepted", "rejections", "examples", "methods", "seconds_per_draw")}
        issues += [f"stage4 randomization: {i}" for i in g["issues"]]
    return {"ok": not issues, "issues": issues, "summary": summary}
