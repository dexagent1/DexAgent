"""Deformable objects of a task, simulated in Isaac Sim.

DexAgent simulates deformable objects (ropes, cables, strings) in Isaac Sim and everything else in MuJoCo. A Stage 3
task folder with deformables carries `deformables.json` (written by the tool library's execute_isaac_deformables):
the cords as Isaac simulates them (geometry, PhysX coefficients, the episode's initial link poses), the grasps and
events of the episode, the Isaac recording of the cords, and the MIRROR: the runtime scene with each cord added as a
MuJoCo cable that MuJoCo never steps (scene_mirror.xml) and the runtime trajectory with the cords' recorded link poses
(trajectory_mirror.npz). scene.xml and trajectory.npz stay the rigid runtime recording (robot and rigid objects).

Stage 4 moves a cord like any randomized object (a planar rigid transform of its initial link poses, drawn from its
ranges in randomize.objects by the cord's name), warps and replays the robot in MuJoCo as for any layout, then runs
the cords again in Isaac Sim, driven by the replayed robot; the Stage 1 goals and the subgoal expectations that name
a cord are judged on the new mirror with the same code as for rigid objects, and the samples render from their
mirror. The original layout's replay reuses the Stage 3 Isaac recording when the replayed robot trajectory is the
recorded one bit for bit (the recording is bound to that trajectory by its sha256); any other trajectory is simulated.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

SCHEMA = "dexagent.isaac_deformables.v1"
KEEP = ("isaac/isaac_recording.npz", "isaac/isaac_events.json", "isaac/job.json", "isaac/isaac_episode.log",
        "isaac_episode_report.json", "deformables.json", "scene_mirror.xml", "trajectory_mirror.npz")


def _library():
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from tool_library.common import isaac_cord as IC
    return IC


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(cfg):
    """The task's deformables (None without any): {path, root, data, cords {name: spec}, recording, report,
    scene_mirror, trajectory_mirror} with absolute paths. Raises ValueError on a malformed hand-off."""
    rel = cfg.get("deformables")
    if not rel:
        return None
    p = cfg.path_of(rel)
    if p is None or not Path(p).is_file():
        raise ValueError(f"deformables: {p} does not exist")
    data = json.loads(Path(p).read_text())
    if data.get("schema") != SCHEMA:
        raise ValueError(f"deformables: {p} is not an Isaac deformables hand-off ({SCHEMA})")
    root = Path(p).parent
    out = {"path": Path(p), "root": root, "data": data, "cords": {}}
    IC = _library()
    for c in data.get("cords") or []:
        c = IC.cord_spec(c) if c.get("schema") != IC.SCHEMA else dict(c)
        if not c.get("link_poses"):                   # laid along its points, as the Isaac episode lays it
            C, Q = IC.lay_poses(c["points"], c["n_links"], c["radius"])
            c["link_poses"] = {"centres": C.tolist(), "quats": Q.tolist()}
        out["cords"][c["name"]] = c
    if not out["cords"]:
        raise ValueError("deformables: no cords")
    for key in ("recording", "report", "scene_mirror", "trajectory_mirror"):
        if not data.get(key):
            raise ValueError(f"deformables: {key} is missing")
        f = root / data[key]
        if not f.is_file():
            raise ValueError(f"deformables: {key} {f} does not exist")
        out[key] = f
    return out


def cord_center(spec):
    return np.asarray(spec["link_poses"]["centres"], float).mean(axis=0)


def first_moved_frame(dfm, name, tol=0.002):
    """The first frame the recorded cord clearly left its start (any link moved more than tol), else None."""
    with np.load(dfm["recording"], allow_pickle=False) as z:
        C = np.asarray(z[f"cord_{name}_centres"], float)
    moved = np.flatnonzero(np.linalg.norm(C - C[0], axis=2).max(axis=1) > tol)
    return int(moved[0]) if len(moved) else None


def moved_spec(spec, D):
    """The cord moved by the rigid world transform D = (R, t): its initial link poses and its centreline points."""
    IC = _library()
    R, t = np.asarray(D[0], float), np.asarray(D[1], float)
    out = copy.deepcopy(spec)
    C = np.asarray(spec["link_poses"]["centres"], float)
    Q = np.asarray(spec["link_poses"]["quats"], float)
    qR = IC.mat_to_quat(R)
    Qn = np.array([IC.qmul(qR, q) for q in Q])
    Qn /= np.linalg.norm(Qn, axis=1, keepdims=True)
    out["link_poses"] = {"centres": (C @ R.T + t).tolist(), "quats": Qn.tolist()}
    out["points"] = (np.asarray(spec["points"], float) @ R.T + t).tolist()
    return out


def retimed(entries, src, keys):
    """Frame fields of grasps / events moved to a retimed episode (src: the source frame of every new frame)."""
    if src is None:
        return copy.deepcopy(entries)
    from . import retime as RT
    out = []
    for e in entries:
        e = copy.deepcopy(e)
        for k in keys:
            if e.get(k) is not None:
                e[k] = int(RT.window(src, int(e[k]), int(e[k]))[0])
        out.append(e)
    return out


def isaac_python(cfg):
    py = (cfg.get("isaac") or {}).get("python") or os.environ.get("DEXAGENT_ISAAC_PYTHON")
    if not py or not Path(py).exists():
        raise ValueError("this task has deformable objects, which Stage 4 simulates in Isaac Sim: give Isaac Sim's Python "
                         "(DEXAGENT_ISAAC_PYTHON, dexagent.yaml environments.isaac, or stage4.json isaac.python)")
    return str(py)


def isaac_home():
    """One HOME for every Isaac run (Isaac writes gigabytes of shader and kit caches there)."""
    home = Path(os.environ.get("DEXAGENT_ISAAC_HOME") or (Path(tempfile.gettempdir()) / "dexagent_isaac_home"))
    home.mkdir(parents=True, exist_ok=True)
    return str(home)


def simulate(dfm, scene_xml, qpos, fps, labels, D=None, src=None, python=None, workdir=None, subgoal=None):
    """Run the cords in Isaac Sim for a replayed robot trajectory. D {cord: (R, t)} moves the cords; src remaps the
    grasp and event frames of a retimed episode. Returns {ok, issues, dir, scene_mirror, trajectory_mirror, report}."""
    _library()
    from tool_library.skills.execute_isaac_deformables import execute_isaac_deformables
    data = dfm["data"]
    cords = [moved_spec(c, (D or {}).get(n, (np.eye(3), np.zeros(3)))) for n, c in dfm["cords"].items()]
    work = Path(tempfile.mkdtemp(prefix="stage4_isaac_", dir=workdir))
    traj = work / "runtime_trajectory.npz"
    arrays = {"qpos": np.asarray(qpos, float), "fps": np.float64(fps), "labels": np.array([str(x) for x in labels])}
    if subgoal is not None and len(subgoal) == len(arrays["qpos"]):
        arrays["subgoal"] = np.asarray(subgoal)
    np.savez_compressed(traj, **arrays)
    try:
        r = execute_isaac_deformables(str(scene_xml), str(traj), out_dir=str(work), cords=cords,
                                      grasps=retimed(data.get("grasps") or [], src, ("weld_frame", "release_frame")),
                                      events=retimed(data.get("events") or [], src, ("frame",)), render=False,
                                      hands=data.get("hands"), robot_collider_bodies=data.get("robot_collider_bodies"),
                                      max_link_speed=float(data.get("max_link_speed", 1.5)), substeps=int(data.get("substeps", 16)),
                                      isaac_python=python, isaac_home=isaac_home())
    except Exception as e:                                # an Isaac run that fails judges the layout, it does not stop Stage 4
        return {"ok": False, "issues": [f"Isaac Sim episode failed: {type(e).__name__}: {str(e)[:400]}"], "dir": str(work)}
    return {"ok": bool(r["ok"]), "issues": list(r["issues"]), "dir": str(work), "scene_mirror": r["scene_mirror"],
            "trajectory_mirror": r["trajectory_mirror"], "report": r["report"], "source": "isaac"}


def stage3_recording(dfm, trajectory_path, qpos_replayed, qpos_recorded):
    """The Stage 3 Isaac recording of the cords, when it belongs to this replay: the replayed robot trajectory is the
    recorded one bit for bit and the recording's report binds that trajectory file and the mirror files. Else None."""
    if not np.array_equal(np.asarray(qpos_replayed), np.asarray(qpos_recorded)):
        return None
    try:
        rep = json.loads(dfm["report"].read_text())
    except (OSError, ValueError):
        return None
    b = rep.get("binding") or {}
    if (rep.get("ok") is not True or b.get("trajectory_sha256") != sha256(trajectory_path)
            or b.get("mirror_scene_sha256") != sha256(dfm["scene_mirror"])
            or b.get("mirror_trajectory_sha256") != sha256(dfm["trajectory_mirror"])
            or b.get("recording_sha256") != sha256(dfm["recording"])):
        return None
    return {"ok": True, "issues": [], "dir": str(dfm["root"]), "scene_mirror": str(dfm["scene_mirror"]),
            "trajectory_mirror": str(dfm["trajectory_mirror"]), "report": str(dfm["report"]), "source": "stage3 recording"}


def mirror(result):
    """(model, qpos [T, nq], labels) of a run's mirror."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(result["scene_mirror"]))
    with np.load(result["trajectory_mirror"], allow_pickle=False) as z:
        Q = np.asarray(z["qpos"], float)
        labels = [str(x) for x in z["labels"]] if "labels" in z.files else None
    if Q.shape[1] != m.nq:
        raise ValueError(f"mirror trajectory has {Q.shape[1]} columns, scene_mirror.xml nq={m.nq}")
    return m, Q, labels


def asset_refs(xml_path):
    """Relative asset files an MJCF names (with its compiler's meshdir / texturedir / assetdir), as paths relative to it."""
    import xml.etree.ElementTree as ET
    root = ET.parse(str(xml_path)).getroot()
    comp = root.find("compiler")
    out = []
    for el in root.iter():
        if el.tag not in ("mesh", "texture", "hfield", "skin"):
            continue
        sub = ((comp.get("meshdir" if el.tag == "mesh" else "texturedir") or comp.get("assetdir")) if comp is not None else None) or ""
        for attr in ("file", "fileright", "fileleft", "fileup", "filedown", "filefront", "fileback"):
            f = el.get(attr)
            if f and not Path(f).is_absolute() and not Path(sub).is_absolute():
                out.append(str(Path(sub) / f))
    return sorted(set(out))


def publish(result, d):
    """Copy a run's cord files into sample folder d: the mirror next to the sample's runtime scene (scene_mirror.xml,
    trajectory_mirror.npz, sharing its assets/ folder), the Isaac recording, events, job, log and report under
    deformables/. Returns what sample.json keeps: the files and their sha256."""
    src, d = Path(result["dir"]), Path(d)
    if result.get("source") == "stage3 recording":
        # the Stage 3 files sit next to the task's deformables.json, under their hand-off names
        data = json.loads((src / "deformables.json").read_text())
        pairs = {"deformables/deformables.json": src / "deformables.json",
                 "deformables/isaac/isaac_recording.npz": src / data["recording"],
                 "deformables/isaac_episode_report.json": src / data["report"]}
    else:
        pairs = {f"deformables/{k}": src / k for k in KEEP if (src / k).is_file() and k not in ("scene_mirror.xml", "trajectory_mirror.npz")}
    scene_m, traj_m = Path(result["scene_mirror"]), Path(result["trajectory_mirror"])
    pairs["scene_mirror.xml"], pairs["trajectory_mirror.npz"] = scene_m, traj_m
    for rel in asset_refs(scene_m):                   # the mirror's meshes, where its relative paths expect them
        pairs.setdefault(rel, scene_m.parent / rel)
    hashes = {}
    for k, f in pairs.items():
        dst = d / k
        if dst.exists() and k.startswith(("assets", "meshes", "textures")):
            if sha256(dst) != sha256(f):
                raise ValueError(f"{dst} exists with other content than the mirror's {f}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
        if not k.startswith(("assets", "meshes", "textures")):
            hashes[k] = sha256(dst)
    return {"dir": "deformables", "source": result.get("source", "isaac"), "sha256": hashes,
            "scene_mirror": "scene_mirror.xml", "trajectory_mirror": "trajectory_mirror.npz"}


def verify_sample(d, info):
    """Issues of a published sample's cord files against the hashes sample.json recorded."""
    issues = []
    meta = info.get("deformables") or {}
    for k, h in (meta.get("sha256") or {}).items():
        f = Path(d) / k
        if not f.is_file():
            issues.append(f"deformables file {k} is missing")
        elif sha256(f) != h:
            issues.append(f"deformables file {k} changed after the replay")
    for k in ("scene_mirror", "trajectory_mirror"):
        if meta.get(k) and meta[k] not in (meta.get("sha256") or {}):
            issues.append(f"deformables: {k} is not covered by the recorded hashes")
    return issues


def verify_task(cfg, trajectory_path=None):
    """Issues of a task folder's deformables hand-off: the files exist, the Isaac report passed and binds the
    recording and the mirror files, the mirror's columns are scene_mirror.xml's nq and its frames the trajectory's."""
    import mujoco
    try:
        dfm = load(cfg)
    except ValueError as e:
        return [str(e)], None
    if dfm is None:
        return [], None
    issues = []
    try:
        rep = json.loads(dfm["report"].read_text())
    except (OSError, ValueError) as e:
        return [f"deformables report unreadable: {e}"], dfm
    if rep.get("ok") is not True:
        issues.append(f"the Stage 3 Isaac episode did not pass: {(rep.get('issues') or ['ok is not true'])[:3]}")
    b = rep.get("binding") or {}
    for key, f in (("recording_sha256", dfm["recording"]), ("mirror_scene_sha256", dfm["scene_mirror"]),
                   ("mirror_trajectory_sha256", dfm["trajectory_mirror"])):
        if b.get(key) != sha256(f):
            issues.append(f"deformables: {Path(f).name} does not match the Isaac report's {key}")
    try:
        m = mujoco.MjModel.from_xml_path(str(dfm["scene_mirror"]))
        with np.load(dfm["trajectory_mirror"], allow_pickle=False) as z:
            Qm = np.asarray(z["qpos"], float)
        if Qm.ndim != 2 or Qm.shape[1] != m.nq:
            issues.append(f"deformables: trajectory_mirror has {Qm.shape} columns, scene_mirror.xml nq={m.nq}")
        if trajectory_path is not None:
            with np.load(trajectory_path, allow_pickle=False) as z:
                T = len(z["qpos"])
            if len(Qm) != T:
                issues.append(f"deformables: the mirror has {len(Qm)} frames, the trajectory {T}")
    except Exception as e:
        issues.append(f"deformables mirror unreadable: {type(e).__name__}: {e}")
    return issues, dfm
