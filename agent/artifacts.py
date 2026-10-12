"""Publish only verified Stage 3 execution artifacts, with a reproducible video.

The candidate workspace is never a delivery. Publication uses a new directory
and an atomic pointer; a failed attempt cannot replace an earlier accepted run.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

from agent.external import atomic_json
from tool_library.common.evidence import file_digest


def model_digest(model):
    """Compiled physical/rendering content, independent of asset file locations."""
    import numpy as np
    h = hashlib.sha256()
    for name in sorted(dir(model)):
        if name.startswith("_") or name in ("paths", "names_map", "npaths", "nbuffer") or name.endswith("_pathadr"):
            continue
        value = getattr(model, name)
        if isinstance(value, np.ndarray) and value.dtype.kind in "biufc":
            h.update(name.encode())
            h.update(str(value.shape).encode())
            h.update(np.ascontiguousarray(value).tobytes())
        elif isinstance(value, (int, float)):
            h.update(f"{name}:{value}".encode())
    h.update(bytes(model.names))
    for name in sorted(dir(model.opt)):
        if name.startswith("_"):
            continue
        value = getattr(model.opt, name)
        if isinstance(value, np.ndarray):
            h.update(name.encode() + value.tobytes())
        elif isinstance(value, (int, float)):
            h.update(f"opt.{name}:{value}".encode())
    return h.hexdigest()


def bundle_scene(source, destination):
    """Copy expanded runtime MJCF and content-addressed assets without collisions."""
    import mujoco
    source, destination = Path(source).resolve(), Path(destination).resolve()
    root = ET.parse(source).getroot()
    if root.find(".//include") is not None:
        raise ValueError("runtime scene must be expanded before publication (MJCF includes remain)")
    comp = root.find("compiler")
    dirs = dict(comp.attrib) if comp is not None else {}
    if comp is not None:
        for key in ("meshdir", "texturedir", "assetdir"):
            comp.attrib.pop(key, None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    assets = destination.parent / "assets"
    assets.mkdir(exist_ok=True)
    for el in root.iter():
        for attr in ("file", "fileright", "fileleft", "fileup", "filedown", "filefront", "fileback"):
            name = el.get(attr)
            if not name:
                continue
            if el.tag not in ("mesh", "texture", "hfield", "skin"):
                raise ValueError(f"unsupported external runtime dependency: {el.tag}.{attr}")
            path = Path(name)
            directory = dirs.get("meshdir" if el.tag == "mesh" else "texturedir" if el.tag == "texture" else "assetdir", dirs.get("assetdir", ""))
            path = path if path.is_absolute() else source.parent / directory / path
            if not path.is_file():
                raise FileNotFoundError(f"missing runtime asset: {path}")
            digest = file_digest(path)
            copy = assets / (digest + path.suffix.lower())
            if not copy.exists():
                shutil.copy2(path, copy)
            el.set(attr, os.path.relpath(copy, destination.parent))
    ET.indent(root)
    destination.write_text(ET.tostring(root, encoding="unicode"))
    before = model_digest(mujoco.MjModel.from_xml_path(str(source)))
    after = model_digest(mujoco.MjModel.from_xml_path(str(destination)))
    if before != after:
        raise ValueError("packaging changed the compiled runtime model")
    return before


def _resolve(workspace, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (workspace / path).resolve()


def bundle_recording(source, source_scene, bundled_scene, destination):
    """Rebind path-only MJCF packaging while preserving every recorded state.

    XML bytes change when assets become portable. The original recording must
    first name the original XML, and both compiled models must match exactly.
    Only that scene digest may change; this is not a new physical execution.
    """
    import mujoco
    import numpy as np
    source_scene, bundled_scene = Path(source_scene), Path(bundled_scene)
    if model_digest(mujoco.MjModel.from_xml_path(str(source_scene))) != model_digest(
            mujoco.MjModel.from_xml_path(str(bundled_scene))):
        raise ValueError("cannot rebind a recording to a different compiled scene")
    with np.load(source, allow_pickle=False) as recording:
        if "scene_sha256" not in recording or str(recording["scene_sha256"].item()) != file_digest(source_scene):
            raise ValueError("recording does not name its original execution scene")
        data = {key: recording[key] for key in recording.files}
    data["scene_sha256"] = np.asarray(file_digest(bundled_scene))
    np.savez_compressed(destination, **data)


def _tree_hashes(directory):
    return {str(p.relative_to(directory)): file_digest(p) for p in sorted(directory.rglob("*")) if p.is_file()}


def verify_delivery(directory):
    directory = Path(directory).resolve()
    manifest = directory / "delivery.json"
    if not manifest.is_file():
        return {"ok": False, "issues": ["verified delivery certificate is missing"]}
    rec = json.loads(manifest.read_text())
    issues = []
    if rec.get("schema") != "dexagent.delivery.v1" or rec.get("passed") is not True:
        issues.append("not a verified delivery")
    for rel, expected in rec.get("sha256", {}).items():
        path = (directory / rel).resolve()
        if not path.is_relative_to(directory):
            issues.append(f"delivery path escapes bundle: {rel}")
            continue
        if not path.is_file() or file_digest(path) != expected:
            issues.append(f"delivery artifact missing or changed: {rel}")
    if not rec.get("sha256"):
        issues.append("delivery has no bound artifacts")
    return {"ok": not issues, "issues": issues, "directory": str(directory)}


def accepted_delivery(public):
    """Resolve an intact bundle named by the completed current harness run.

    A self-consistent pointer/certificate is only a content-integrity check.
    It cannot turn a failed or still-running attempt into an accepted delivery.
    This is an application lifecycle check, not authentication against a hostile
    process with write access to the entire run directory.
    """
    public = Path(public).resolve()
    run = json.loads((public / "run.json").read_text())
    if not isinstance(run, dict):
        raise ValueError("current Stage 3 run record is invalid")
    verdict = run.get("verdict")
    if run.get("finished") is not True or not isinstance(verdict, dict) or verdict.get("ok") is not True:
        raise ValueError("current Stage 3 run has not finished successful verification")
    pointer = json.loads((public / "accepted.json").read_text())
    directory = Path(pointer["directory"]).resolve()
    reports = verdict.get("reports")
    delivery = run.get("delivery") or (reports.get("verified_delivery") if isinstance(reports, dict) else None)
    if not isinstance(delivery, dict) or delivery.get("ok") is not True or not delivery.get("directory") or Path(delivery["directory"]).resolve() != directory:
        raise ValueError("accepted pointer is not the delivery of the current completed run")
    if delivery.get("certificate_sha256") and delivery["certificate_sha256"] != pointer.get("certificate_sha256"):
        raise ValueError("accepted pointer differs from the completed run's certificate")
    if not directory.is_relative_to(public / "verified"):
        raise ValueError("accepted delivery escapes its versioned output directory")
    if file_digest(directory / "delivery.json") != pointer.get("certificate_sha256"):
        raise ValueError("accepted delivery certificate changed")
    result = verify_delivery(directory)
    if not result["ok"]:
        raise ValueError("; ".join(result["issues"]))
    return directory


# the accepted bundle's episode, copied to the top of the stage folder (runs/<name>/stage3/trajectory.npz, sim_video.mp4,
# stage4_task/, ...) where a user opens it and runs Stage 4 on its own; plus the execution's per-subgoal folders (episode/)
PUBLIC_COPIES = ("trajectory.npz", "phases.json", "sim_video.mp4", "trajectory_manifest.json", "stage4_task", "twin_episode.json",
                 "twin_episode")
PUBLIC_EPISODE = "episode"


def retire_public_copies(public, tag="") -> list:
    """Move a previous run's top-level copies aside (previous-<name>-<tag>), never deleting what Stage 4 may have
    written into them."""
    public, moved = Path(public), []
    for name in PUBLIC_COPIES + (PUBLIC_EPISODE,):
        p = public / name
        if p.exists() or p.is_symlink():
            dst = public / f"previous-{Path(name).stem}-{tag + '-' if tag else ''}{uuid.uuid4().hex[:8]}{Path(name).suffix}"
            p.replace(dst)
            moved.append(str(dst))
    return moved


def expose_accepted(public, accepted) -> dict:
    """Copies of the accepted bundle's episode files at the top of the stage folder, and of the execution's per-subgoal
    folders (keyframes, physics rollout, outcome) as episode/. The sealed bundle under verified/ stays the authority
    (its delivery.json hashes it)."""
    public, accepted = Path(public), Path(accepted)
    retire_public_copies(public, accepted.name[:8])
    out = {}
    for name in PUBLIC_COPIES:
        src = accepted / name
        if src.is_dir():
            shutil.copytree(src, public / name)
        elif src.is_file():
            shutil.copy2(src, public / name)
        else:
            continue
        out[name] = str(public / name)
    try:
        man = json.loads((accepted / "trajectory_manifest.json").read_text())
        execution = Path(str((man.get("episode") or {}).get("execution_dir") or ""))
    except (OSError, ValueError):
        execution = Path("")
    if str(execution) and execution.is_absolute() and execution.is_dir():
        segments = sorted(d for d in execution.glob("subgoal_*") if d.is_dir())
        for d in segments:
            shutil.copytree(d, public / PUBLIC_EPISODE / d.name)
        for f in ("outcomes.json", "execution_report.json", "phases.json"):
            if (execution / f).is_file():
                (public / PUBLIC_EPISODE).mkdir(exist_ok=True)
                shutil.copy2(execution / f, public / PUBLIC_EPISODE / f)
        if segments:
            out[PUBLIC_EPISODE] = str(public / PUBLIC_EPISODE)
    return out


def publish_stage3(workspace, public_dir, verdict, semantic, recon, video=None):
    import mujoco
    import numpy as np
    from tool_library.skills.render_trajectory_video import render_trajectory_video

    if verdict.get("ok") is not True or not verdict.get("reports"):
        raise ValueError("cannot publish without a successful final verification")
    ws, public = Path(workspace).resolve(), Path(public_dir).resolve()
    man_path = ws / "trajectory_manifest.json"
    man = json.loads(man_path.read_text())
    if not man.get("scene_executed"):
        raise ValueError("manifest must identify the actual rigid execution scene")
    scene = _resolve(ws, man["scene_executed"])
    ep = man.get("episode") or {}
    trajectory = _resolve(ws, ep.get("trajectory", "trajectory.npz"))
    phases = _resolve(ws, ep.get("phases", "phases.json"))
    task = _resolve(ws, (man.get("stage4") or {}).get("task_dir", "stage4_task"))
    protected = (man_path, scene, trajectory, phases, Path(semantic), Path(recon))
    source_hashes = {str(p): file_digest(p) for p in protected}
    z = np.load(trajectory, allow_pickle=False)
    if "joint_names" not in z or "ctrl" not in z:
        raise ValueError("delivery requires named runtime joints and actual controls")
    model = mujoco.MjModel.from_xml_path(str(scene))
    from tool_library.common.rigid_robot import is_rigid
    if not is_rigid(model):
        raise ValueError("formal delivery requires the rigid robot execution model")
    from tool_library.common import episode as E
    if list(z["joint_names"].astype(str)) != E.joint_names(model):
        raise ValueError("trajectory joint identity does not match execution scene")
    if z["qpos"].shape[1] != model.nq or not np.isfinite(z["qpos"]).all():
        raise ValueError("invalid runtime trajectory")
    tag = uuid.uuid4().hex
    staged = public / (".publishing-" + tag)
    staged.mkdir(parents=True)
    digest = bundle_scene(scene, staged / "scene.xml")
    bundle_recording(trajectory, scene, staged / "scene.xml", staged / "trajectory.npz")
    shutil.copy2(phases, staged / "phases.json")
    shutil.copy2(semantic, staged / "scene_semantics.json")
    shutil.copy2(recon, staged / "scene_recon.json")
    shutil.copytree(task, staged / "stage4_task")
    task_cfg = json.loads((staged / "stage4_task" / "stage4.json").read_text())
    mechanism_path = (task_cfg.get("physics") or {}).get("scene_recon_json")
    if mechanism_path and file_digest(_resolve(staged / "stage4_task", mechanism_path)) != file_digest(recon):
        raise ValueError("Stage 4 mechanism descriptor differs from frozen Stage 2")
    if video is not None:
        task_video = _resolve(staged / "stage4_task", task_cfg["human_video"])
        if file_digest(task_video) != file_digest(video):
            raise ValueError("Stage 4 source video differs from the actual demonstration input")
    task_scene = _resolve(staged / "stage4_task", task_cfg.get("scene", "scene.xml"))
    if model_digest(mujoco.MjModel.from_xml_path(str(task_scene))) != digest:
        raise ValueError("Stage 4 is packaged with a different execution model")
    task_traj = _resolve(staged / "stage4_task", task_cfg.get("trajectory", "trajectory.npz"))
    with np.load(task_traj, allow_pickle=False) as handed:
        for key in z.files:
            if key == "scene_sha256":
                if str(handed[key].item()) != file_digest(task_scene):
                    raise ValueError("Stage 4 recording does not identify its packaged runtime scene")
                continue
            if key not in handed or not np.array_equal(handed[key], z[key]):
                raise ValueError(f"Stage 4 recording differs from verified execution: {key}")
    # Render again from the snapshot. An agent-supplied movie is never promoted.
    camera_names = [model.camera(i).name for i in range(model.ncam)]
    camera = "video_cam" if "video_cam" in camera_names else "ego" if "ego" in camera_names else None
    rendered = render_trajectory_video(staged / "scene.xml", staged / "trajectory.npz", staged / "views",
                                       cameras=[camera] if camera else None, overlay_text=False)
    video = Path(next(iter(rendered["videos"].values())))
    shutil.copy2(video, staged / "sim_video.mp4")
    shutil.copy2(video, staged / "stage4_task" / "sim_video.mp4")
    man["scene_executed"] = "scene.xml"
    man["scene_semantic"] = "scene_semantics.json"
    man["scene_recon"] = "scene_recon.json"
    man["episode"] = {**ep, "trajectory": "trajectory.npz", "phases": "phases.json", "sim_video": "sim_video.mp4"}
    man["stage4"] = {"task_dir": "stage4_task", "config": "stage4_task/stage4.json"}
    (staged / "trajectory_manifest.json").write_text(json.dumps(man, indent=1))
    (staged / "verification.json").write_text(json.dumps(verdict, indent=1, allow_nan=False))
    if any(file_digest(p) != expected for p, expected in source_hashes.items()):
        raise ValueError("candidate changed during final publication")
    render_record = {**rendered, "videos": {name: os.path.relpath(path, staged) for name, path in rendered["videos"].items()},
                     "preview": os.path.relpath(rendered["preview"], staged) if rendered.get("preview") else None}
    certificate = {"schema": "dexagent.delivery.v1", "passed": True, "compiled_model_sha256": digest,
                   "source_sha256": source_hashes, "sha256": _tree_hashes(staged), "render": render_record}
    atomic_json(staged / "delivery.json", certificate)
    accepted = public / "verified" / tag
    accepted.parent.mkdir(exist_ok=True)
    staged.replace(accepted)
    pointer = {"schema": "dexagent.accepted.v1", "directory": str(accepted),
               "certificate_sha256": file_digest(accepted / "delivery.json")}
    atomic_json(public / "accepted.json", pointer)
    try:
        expose_accepted(public, accepted)
    except OSError as exc:                       # convenience copies only: the sealed delivery stands
        (public / "expose_error.txt").write_text(f"{type(exc).__name__}: {exc}")
    return {"ok": True, "issues": [], "directory": str(accepted), "certificate_sha256": pointer["certificate_sha256"],
            "video": str(accepted / "sim_video.mp4"),
            "trajectory": str(accepted / "trajectory.npz"), "stage4_config": str(accepted / "stage4_task" / "stage4.json")}
