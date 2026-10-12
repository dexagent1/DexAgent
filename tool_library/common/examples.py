"""Verified examples: find the one a new video matches, reuse its outputs, adapt its trajectory to a new layout of the
same task, and save a newly verified run as an example.

Layout of an example folder (examples/default/<task>/ ships with the repository, examples/asset_reuse/<name>/ is filled
by runs with `examples.save_assets: true`):

    manifest.json, fingerprint.json
    stage1/scene_semantics.json, prompt.txt
    stage2/scene.xml, scene_recon.json, camera.json, masks.npz, frame0_signature.npz, objects/<id>/<id>.xml
    stage3/trajectory.npz, phases.json, trajectory_manifest.json, stage4_task/stage4.json

Mesh files live once in examples/_assets (content-addressed); scene files reference them relatively. Examples ship no
image files: the masks are bit-packed in masks.npz, frame 0 is identified by a grey thumbnail (frame0_signature.npz),
and a reuse writes masks/*.png, frame0.jpg (the run's own first frame) and frame0_compare.png into the run.
No video is stored: a run's own video is what Stage 4 inpaints.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "examples"
POOL_NAME = "_assets"
MASKS_NPZ = "masks.npz"                      # stage2/: every mask of the example, bit-packed, by its path in scene_recon
FRAME0_SIGNATURE = "frame0_signature.npz"    # stage2/: the grey frame 0 (320 px wide) that identifies the recording
FRAME0_JPEG_QUALITY = 95


def mask_pixels_digest(mask) -> str:
    """sha256 of a mask's pixels (shape and bits), independent of how the image file was encoded."""
    m = np.asarray(mask) > 0
    return hashlib.sha256(np.asarray(m.shape, np.int64).tobytes() + np.packbits(m.ravel()).tobytes()).hexdigest()


def pack_masks(masks: dict, out_npz):
    """{relative path: bool array} -> one compressed npz (paths, bits, shapes)."""
    rels = sorted(masks)
    arrays = {"paths": np.array(rels)}
    for i, r in enumerate(rels):
        m = np.asarray(masks[r]) > 0
        arrays[f"bits{i}"], arrays[f"shape{i}"] = np.packbits(m.ravel()), np.asarray(m.shape, np.int64)
    np.savez_compressed(out_npz, **arrays)


def example_masks(stage2_dir) -> dict:
    """An example's masks as {relative path: bool array}: from masks.npz, else from the image files under masks/."""
    s2 = Path(stage2_dir)
    out = {}
    if (s2 / MASKS_NPZ).is_file():
        with np.load(s2 / MASKS_NPZ, allow_pickle=False) as z:
            for i, r in enumerate(z["paths"]):
                shape = tuple(int(v) for v in z[f"shape{i}"])
                out[str(r)] = np.unpackbits(z[f"bits{i}"], count=int(np.prod(shape))).reshape(shape).astype(bool)
        return out
    import cv2
    for f in sorted((s2 / "masks").rglob("*.png")) if (s2 / "masks").is_dir() else []:
        a = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if a is not None:
            out[str(f.relative_to(s2))] = (a[..., 0] if a.ndim == 3 else a) > 0
    return out


def write_masks(stage2_dir, workspace) -> list:
    """The example's masks written as 0/255 PNGs at their paths in the workspace (what scene_recon names)."""
    import cv2
    files = []
    for rel, m in example_masks(stage2_dir).items():
        dst = Path(workspace) / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), m.astype(np.uint8) * 255)
        files.append(str(dst))
    return files


def video_frame0(video):
    import cv2
    cap = cv2.VideoCapture(str(video))
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"cannot decode the first frame of {video}")
    return img


def frame0_jpeg_bytes(video) -> bytes:
    """A run's frame0.jpg: the first decoded frame of its video, JPEG quality FRAME0_JPEG_QUALITY."""
    import cv2
    ok, buf = cv2.imencode(".jpg", video_frame0(video), [int(cv2.IMWRITE_JPEG_QUALITY), FRAME0_JPEG_QUALITY])
    if not ok:
        raise ValueError(f"cannot encode the first frame of {video}")
    return buf.tobytes()


def frame0_signature(image):
    """The grey 320 px wide frame agent.reference_evidence compares block by block (uint8)."""
    from agent.reference_evidence import _grey_blocks
    return np.clip(np.rint(_grey_blocks(image)), 0, 255).astype(np.uint8)


def strip_example_images(example_dir) -> dict:
    """An example folder without image files: stage2 masks -> masks.npz, frame0.jpg -> frame0_signature.npz (with its
    resolution), the certificate's file digests -> pixel digests and the signature, every other .png/.jpg/.jpeg under
    the example (comparison renders, Isaac renders) deleted. Returns what was removed."""
    import cv2
    ex = Path(example_dir)
    s2 = ex / "stage2"
    removed = []
    masks = example_masks(s2)
    if masks and not (s2 / MASKS_NPZ).is_file():
        pack_masks(masks, s2 / MASKS_NPZ)
    cert_p = s2 / "example_certificate.json"
    cert = json.loads(cert_p.read_text()) if cert_p.is_file() else None
    f0 = s2 / "frame0.jpg"
    if f0.is_file():
        img = cv2.imread(str(f0))
        np.savez_compressed(s2 / FRAME0_SIGNATURE, grey=frame0_signature(img), resolution=np.asarray(img.shape[:2], np.int64))
    if cert is not None:
        for m in cert.get("masks", []):
            if "sha256" in m and m["path"] in masks:
                m["pixels_sha256"] = mask_pixels_digest(masks[m["path"]])
                m.pop("sha256")
        if "sha256" in cert.get("frame0", {}):
            cert["frame0"] = {"path": cert["frame0"].get("path", "frame0.jpg"), "source": "the first frame of the recording",
                              "signature": FRAME0_SIGNATURE}
        cert_p.write_text(json.dumps(cert, indent=1))
    for f in sorted(ex.rglob("*")):
        if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg"):
            f.unlink()
            removed.append(str(f.relative_to(ex)))
    for d in sorted((p for p in ex.rglob("*") if p.is_dir()), key=lambda p: -len(p.parts)):
        if not any(d.iterdir()):
            d.rmdir()
    return {"removed": removed, "masks": sorted(masks)}

# a re-encode or resize of the same clip scores ~0.999; two takes of one task on the same bench ~0.8-0.92
IDENTICAL = {"mean_ncc": 0.97, "min_ncc": 0.95, "frame0_ncc": 0.97, "duration_ratio": 0.98}


def _semantics(stage1_dir):
    """Stage 1's file in a folder: scene_semantics.json, or scene_semantic.json in examples saved before the release."""
    d = Path(stage1_dir)
    return d / "scene_semantics.json" if (d / "scene_semantics.json").exists() or not (d / "scene_semantic.json").exists() \
        else d / "scene_semantic.json"

def examples_root(examples_dir=None) -> Path:
    return Path(examples_dir) if examples_dir else EXAMPLES


def list_examples(examples_dir=None):
    root = examples_root(examples_dir)
    out = []
    for group in ("default", "asset_reuse"):
        g = root / group
        if not g.is_dir():
            continue
        for d in sorted(g.iterdir()):
            if (d / "manifest.json").exists() and (d / "fingerprint.json").exists():
                out.append(d)
    return out


# ----------------------------------------------------------------------------- matching
def _words(s):
    return set(re.findall(r"[a-z]+", str(s).lower())) - {"the", "a", "an", "of", "and", "to", "in", "on", "with", "into", "then", "from"}


def semantic_similarity(a: dict, b: dict) -> dict:
    """How alike two Stage 1 results are: object sets (by id, name words and type) and the task prompt."""
    oa = {(o.get("id"), o.get("type")) for o in a.get("objects", [])}
    ob = {(o.get("id"), o.get("type")) for o in b.get("objects", [])}
    na = set().union(*[_words(o.get("name", "")) | _words(o.get("id", "")) for o in a.get("objects", [])] or [set()])
    nb = set().union(*[_words(o.get("name", "")) | _words(o.get("id", "")) for o in b.get("objects", [])] or [set()])
    pa, pb = _words(a.get("task_prompt", "")), _words(b.get("task_prompt", ""))
    jac = lambda x, y: len(x & y) / max(len(x | y), 1)
    return {"objects_exact": oa == ob, "object_ids": round(jac({x[0] for x in oa}, {x[0] for x in ob}), 3),
            "object_words": round(jac(na, nb), 3), "prompt": round(jac(pa, pb), 3), "n_objects": [len(oa), len(ob)]}


def find_example(video=None, scene_semantic=None, examples_dir=None):
    """The example a run should reuse. `identical`: the same recording (perceptual fingerprint); `same_task`: the
    same task and objects in another take or layout (Stage 1 result, or a close but not identical video)."""
    from tool_library.common.video import video_fingerprint, fingerprint_similarity
    fp = video_fingerprint(str(video)) if video else None
    sem = json.loads(Path(scene_semantic).read_text()) if isinstance(scene_semantic, (str, Path)) else scene_semantic
    rows = []
    for d in list_examples(examples_dir):
        row = {"example": str(d), "task": json.loads((d / "manifest.json").read_text()).get("task", d.name), "group": d.parent.name}
        if fp is not None:
            row["video"] = fingerprint_similarity(fp, json.loads((d / "fingerprint.json").read_text()))
        if sem is not None and _semantics(d / "stage1").exists():
            row["semantic"] = semantic_similarity(sem, json.loads(_semantics(d / "stage1").read_text()))
        rows.append(row)

    def identical(r):
        v = r.get("video") or {}
        return all(v.get(k, 0) >= t for k, t in IDENTICAL.items())

    def same_task(r):
        s = r.get("semantic") or {}
        v = r.get("video") or {}
        sem_ok = s and (s.get("objects_exact") or (s.get("object_ids", 0) >= 0.99)) and s.get("prompt", 0) >= 0.5
        vid_close = v.get("mean_ncc", 0) >= 0.85 and v.get("duration_ratio", 0) >= 0.5
        return bool(sem_ok or (vid_close and s and s.get("object_words", 0) >= 0.6))

    best = None
    for r in rows:
        r["match"] = "identical" if identical(r) else ("same_task" if same_task(r) else "none")
    ident = [r for r in rows if r["match"] == "identical"]
    same = [r for r in rows if r["match"] == "same_task"]
    if ident:
        best = max(ident, key=lambda r: r["video"]["mean_ncc"])
    elif same:
        best = max(same, key=lambda r: ((r.get("semantic") or {}).get("object_words", 0), (r.get("video") or {}).get("mean_ncc", 0)))
    ranked = sorted(rows, key=lambda r: -((r.get("video") or {}).get("mean_ncc", 0) + (r.get("semantic") or {}).get("object_words", 0)))
    return {"match": best["match"] if best else "none", "example": best["example"] if best else None,
            "task": best["task"] if best else None, "best": best, "candidates": ranked[:5]}


# ----------------------------------------------------------------------------- scene files
def _abs_files(xml_path: Path, out_xml: Path, assets_dir: Path | None = None, absolute: bool = False):
    """Copy an MJCF so that every file it references still resolves from its new location. With `assets_dir` the
    referenced files are copied there (once, by name) and referenced relatively, so the copy is portable; without it
    they are referenced by absolute path."""
    root = ET.parse(xml_path).getroot()
    comp = root.find("compiler")
    base = xml_path.parent
    if comp is not None:
        for a in ("meshdir", "texturedir", "assetdir"):
            comp.attrib.pop(a, None)
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    for el in root.iter():
        for attr in ("file", "fileright", "fileleft", "fileup", "filedown", "filefront", "fileback"):
            f = el.get(attr)
            if not f:
                continue
            src = Path(f) if Path(f).is_absolute() else (base / f).resolve()
            if assets_dir is None:
                el.set(attr, str(src))
                continue
            assets_dir.mkdir(parents=True, exist_ok=True)
            dst = assets_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            # a run's copy references its local assets absolutely: tools write derived scenes (scene_stations.xml) elsewhere
            el.set(attr, str(dst.resolve()) if absolute else os.path.relpath(dst, out_xml.parent))
    ET.indent(root)
    out_xml.write_text(ET.tostring(root, encoding="unicode"))
    return out_xml


def _pool_file(src: Path, pool: Path) -> Path:
    pool.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1(Path(src).read_bytes()).hexdigest()[:20]
    dst = pool / f"{h}{Path(src).suffix.lower()}"
    if not dst.exists():
        shutil.copy2(src, dst)
    return dst


def export_pooled(xml_path, out_xml, pool):
    """Write an MJCF whose mesh/texture/hfield files are moved into a content-addressed pool and referenced relatively."""
    xml_path, out_xml = Path(xml_path), Path(out_xml)
    root = ET.parse(xml_path).getroot()
    comp = root.find("compiler")
    md = td = ad = None
    if comp is not None:
        md, td, ad = comp.get("meshdir"), comp.get("texturedir"), comp.get("assetdir")
        for a in ("meshdir", "texturedir", "assetdir"):
            comp.attrib.pop(a, None)
    base = xml_path.parent
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    for tag, d in (("mesh", md or ad), ("texture", td or ad), ("hfield", ad), ("skin", ad)):
        for el in root.iter(tag):
            for attr in ("file", "fileright", "fileleft", "fileup", "filedown", "filefront", "fileback"):
                f = el.get(attr)
                if not f:
                    continue
                cands = [Path(f)] if Path(f).is_absolute() else [base / (d or "") / f, base / f]
                src = next((c for c in cands if c.exists()), None)
                if src is None:
                    raise FileNotFoundError(f"{tag} {el.get('name')}: {f} not found next to {xml_path}")
                el.set(attr, os.path.relpath(_pool_file(src, pool), out_xml.parent))
    ET.indent(root)
    out_xml.write_text(ET.tostring(root, encoding="unicode"))
    return out_xml


RUNTIME_ASSETS = "runtime_assets.json"   # stage3/: pool files of the byte-exact runtime scenes of a rigid-runtime example
TWIN_EPISODE = "twin_episode.json"        # stage3/: a recorded simulator twin episode (Isaac Sim) shipped as the Stage 3 result


def twin_episode(example):
    """The example's recorded twin episode (stage3/twin_episode.json: its files, simulator and summary), or None."""
    p = Path(example) / "stage3" / TWIN_EPISODE
    return json.loads(p.read_text()) if p.is_file() else None
# a Stage 3 episode with deformables simulated in Isaac Sim: execute_isaac_deformables' files (stage3/ and stage4_task/)
ISAAC_FILES = ("isaac_episode_report.json", "deformables.json", "trajectory_mirror.npz", "isaac/isaac_recording.npz",
               "isaac/isaac_events.json", "isaac/job.json")


def materialize_runtime_scene(example, rel, destination):
    """Copy a rigid-runtime example scene (stage3/<rel>) byte for byte and place each mesh it references, from the pool,
    at the relative path the scene names (checked by sha256): the trajectory's scene_sha256 binds those exact bytes."""
    ex = Path(example)
    s3 = ex / "stage3"
    table = json.loads((s3 / RUNTIME_ASSETS).read_text())
    assets = table["scenes"][rel]
    pool = examples_root(ex.parents[1]) / POOL_NAME
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for name, entry in assets.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"runtime asset path must stay inside the scene folder: {name}")
        source = pool / entry["pool"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"pooled asset {source.name} does not match the runtime scene's {name}")
        target = destination.parent / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shutil.copy2(s3 / rel, destination)
    return destination


# ----------------------------------------------------------------------------- reuse
def reuse_example(example, stage, workspace, video=None):
    """Copy one stage's verified outputs of an example into a run's workspace. Stage 2 scenes are rewritten with
    absolute asset paths; Stage 3's stage4_task gets the run's own video as its human video. Returns the files."""
    ex, ws = Path(example), Path(workspace)
    ws.mkdir(parents=True, exist_ok=True)
    stage = str(stage).replace("stage", "")
    files = []
    note = {"reused_from": str(ex), "example_task": json.loads((ex / "manifest.json").read_text()).get("task")}
    if stage == "1":
        shutil.copy2(_semantics(ex / "stage1"), ws / "scene_semantics.json"); files.append(str(ws / "scene_semantics.json"))
        (ws / "reuse.json").write_text(json.dumps(note, indent=1))
    elif stage == "2":
        s2 = ex / "stage2"
        _abs_files(s2 / "scene.xml", ws / "scene.xml", ws / "assets", absolute=True); files.append(str(ws / "scene.xml"))
        for f in ("camera.json", "frame0.jpg", "frame0_compare.png"):
            if (s2 / f).exists():
                shutil.copy2(s2 / f, ws / f); files.append(str(ws / f))
        if not (s2 / "frame0.jpg").exists() and video:    # frame 0 is the run's own first frame (examples ship no images)
            (ws / "frame0.jpg").write_bytes(frame0_jpeg_bytes(video)); files.append(str(ws / "frame0.jpg"))
        files += write_masks(s2, ws)
        for xml in (s2 / "objects").glob("*/*.xml"):
            _abs_files(xml, ws / "objects" / xml.parent.name / xml.name, ws / "assets", absolute=True)
            files.append(str(ws / "objects" / xml.parent.name / xml.name))
        recon = json.loads((s2 / "scene_recon.json").read_text())
        recon["scene_semantic"] = "../stage1/scene_semantics.json"
        recon.setdefault("notes", []).append(f"reused verbatim from the verified example {note['example_task']} ({ex})")
        recon["reused_from"] = str(ex)
        (ws / "scene_recon.json").write_text(json.dumps(recon, indent=1)); files.append(str(ws / "scene_recon.json"))
        for sub in s2.iterdir():                     # Isaac files: whatever the manifest references
            if sub.is_dir() and sub.name not in ("objects", "masks"):
                shutil.copytree(sub, ws / sub.name, dirs_exist_ok=True)
        for gz in ws.rglob("*.gz"):                   # an Isaac stage shipped gzipped
            import gzip
            with gzip.open(gz, "rb") as fi, open(gz.with_suffix(""), "wb") as fo:
                shutil.copyfileobj(fi, fo)
            gz.unlink()
        if not (ws / "frame0_compare.png").exists() and (ws / "frame0.jpg").exists():
            try:                                       # the comparison render, drawn again for this run
                from tool_library.skills.render_frame_compare import render_frame_compare
                render_frame_compare(str(ws / "scene.xml"), str(ws / "camera.json"), str(ws / "frame0.jpg"), str(ws / "frame0_compare.png"))
                files.append(str(ws / "frame0_compare.png"))
            except Exception as exc:                   # a picture for people: its absence changes nothing
                (ws / "frame0_compare_error.txt").write_text(f"{type(exc).__name__}: {exc}")
    elif stage == "3" and twin_episode(ex) is not None and not (ex / "stage3" / "trajectory.npz").exists():
        # a recorded twin episode: its video is the run's sim_video.mp4, its files travel in twin_episode/
        s3, twin = ex / "stage3", twin_episode(ex)
        shutil.copy2(s3 / TWIN_EPISODE, ws / TWIN_EPISODE); files.append(str(ws / TWIN_EPISODE))
        for rel in twin["files"].values():
            (ws / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s3 / rel, ws / rel); files.append(str(ws / rel))
        shutil.copy2(s3 / twin["files"]["sim_video"], ws / "sim_video.mp4"); files.append(str(ws / "sim_video.mp4"))
    elif stage == "3":
        s3 = ex / "stage3"
        if not (s3 / "trajectory.npz").exists():
            raise ValueError(f"example {ex.name} has no ground-truth trajectory (stage 3); solve this stage")
        for f in ("trajectory.npz", "phases.json", "keyframes.png"):
            if (s3 / f).exists():
                shutil.copy2(s3 / f, ws / f); files.append(str(ws / f))
        man = json.loads((s3 / "trajectory_manifest.json").read_text())
        man.update({"scene_semantic": "../stage1/scene_semantics.json", "scene_recon": "../stage2/scene_recon.json", "reused_from": str(ex)})
        (ws / "trajectory_manifest.json").write_text(json.dumps(man, indent=1)); files.append(str(ws / "trajectory_manifest.json"))
        task = ws / "stage4_task"; task.mkdir(exist_ok=True)
        cfg = json.loads((s3 / "stage4_task" / "stage4.json").read_text())
        executed = s3 / "scene_executed.xml"
        src_task = s3 / "stage4_task"
        runtime = s3 / RUNTIME_ASSETS
        if runtime.exists():
            # a rigid-runtime example: its trajectories record the sha256 of the exact scene bytes they ran in, so the
            # scenes are copied byte for byte and their meshes materialized from the pool where the scenes expect them
            for rel, dst in (("scene_executed.xml", ws / "scene_executed.xml"), ("stage4_task/scene.xml", task / "scene.xml")):
                materialize_runtime_scene(ex, rel, dst); files.append(str(dst))
            if (s3 / "place_arm_stations.json").exists():
                shutil.copy2(s3 / "place_arm_stations.json", ws / "place_arm_stations.json")
        else:
            if executed.exists():                        # moved stations: the trajectory belongs to this scene
                _abs_files(executed, ws / "scene_executed.xml", ws / "assets"); files.append(str(ws / "scene_executed.xml"))
                if (s3 / "place_arm_stations.json").exists():
                    shutil.copy2(s3 / "place_arm_stations.json", ws / "place_arm_stations.json")
            _abs_files(executed if executed.exists() else ex / "stage2" / "scene.xml", task / "scene.xml", task / "assets")
        shutil.copy2(src_task / "trajectory.npz" if (src_task / "trajectory.npz").exists() else s3 / "trajectory.npz", task / "trajectory.npz")
        mirror = None
        if isinstance(man.get("deformables"), dict):
            # deformables simulated in Isaac Sim: the recording, its replay inputs and the MuJoCo mirror (byte for byte:
            # the Isaac report binds their sha256)
            for d_ in (ws, task):
                src_dir = s3 if d_ is ws else src_task
                for f in ISAAC_FILES:
                    if (src_dir / f).is_file():
                        (d_ / f).parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src_dir / f, d_ / f); files.append(str(d_ / f))
                rel = "scene_mirror.xml" if d_ is ws else "stage4_task/scene_mirror.xml"
                if runtime.exists() and rel in json.loads((s3 / RUNTIME_ASSETS).read_text())["scenes"]:
                    materialize_runtime_scene(ex, rel, d_ / "scene_mirror.xml"); files.append(str(d_ / "scene_mirror.xml"))
            mirror = (task / "scene_mirror.xml", task / "trajectory_mirror.npz")
        for f in ("trajectory_manifest.json", "goal_objects.json", "scene_semantics.json", "mechanism_scene_recon.json"):
            if (src_task / f).exists():
                shutil.copy2(src_task / f, task / f); files.append(str(task / f))
        files += [str(task / "scene.xml"), str(task / "trajectory.npz")]
        # examples ship no videos: the robot episode is rendered again from the reused trajectory through the video camera
        try:
            from tool_library.skills.render_trajectory_video import render_trajectory_video
            import tempfile
            tmpd = Path(tempfile.mkdtemp(prefix="reuse_render_", dir=os.environ.get("TMPDIR")))
            sc_, tj_ = mirror if mirror and mirror[0].exists() and mirror[1].exists() else (task / "scene.xml", task / "trajectory.npz")
            rv = render_trajectory_video(str(sc_), str(tj_), str(tmpd), cameras=["video_cam"])
            vid = Path((rv.get("videos") or {}).get("video_cam") or tmpd / "video_cam.mp4")
            if vid.exists():
                shutil.copy2(vid, ws / "sim_video.mp4"); shutil.copy2(vid, task / "sim_video.mp4")
                files += [str(ws / "sim_video.mp4"), str(task / "sim_video.mp4")]
            shutil.rmtree(tmpd, ignore_errors=True)
        except Exception as e:                           # no renderer here: the stage's verifiers will say what is missing
            note_render = f"{type(e).__name__}: {e}"[:200]
            files.append(f"(sim_video not rendered: {note_render})")
        cfg.update({"scene": "scene.xml", "trajectory": "trajectory.npz", "human_video": "human_video.mp4"})
        cfg["sim_video"] = "sim_video.mp4" if (task / "sim_video.mp4").exists() else None
        (task / "stage4.json").write_text(json.dumps(cfg, indent=1))
        if video and Path(video).exists():
            shutil.copy2(video, task / "human_video.mp4"); files.append(str(task / "human_video.mp4"))
        files.append(str(task / "stage4.json"))
    else:
        raise ValueError("stage must be 1, 2 or 3")
    return {"stage": int(stage), "example": str(ex), "files": files}


# ----------------------------------------------------------------------------- same task, new layout
def adapt_example(example, layout, workspace, video=None, checks=None, scene=None, trajectory=None):
    """The example's verified trajectory carried onto a new layout of the same objects: each object's (dx, dy,
    dyaw_deg) from its place in the example, hands re-anchored phase by phase to the objects they work on (the
    example's Stage 4 anchor schedule), arms re-solved by IK, then reach, joint speed, penetration and object
    continuity checked (the Stage 4 randomization machinery). Writes stage3 files into the workspace. `scene` and
    `trajectory` default to the example's Stage 2 scene and Stage 3 trajectory; pass a reused stage4_task's pair when
    the trajectory was recorded in another scene (moved arm stations, rigid runtime)."""
    from tool_library.skills.adapt_trajectory_to_layout import Episode
    ex, ws = Path(example), Path(workspace)
    ws.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ex / "stage3" / "stage4_task" / "stage4.json").read_text())
    scene = Path(scene) if scene else ex / "stage2" / "scene.xml"
    trajectory = Path(trajectory) if trajectory else ex / "stage3" / "trajectory.npz"
    objs = {n: {k: v for k, v in o.items() if k in ("carrier", "release", "attached_to")} for n, o in cfg["randomize"]["objects"].items()}
    lay = {"objects": {n: [0.0, 0.0, 0.0] for n in objs}, "robot": {}}
    for n, v in (layout.get("objects", layout) if isinstance(layout, dict) else {}).items():
        if n not in objs:
            raise ValueError(f"layout object {n!r} is not a movable object of the example ({sorted(objs)})")
        lay["objects"][n] = [float(x) for x in v]
    ep = Episode(str(scene), str(trajectory), cfg["robot"]["end_effectors"], objs, cfg["segments"], checks)
    Q, overrides, report = ep.adapt(lay)
    np.savez_compressed(ws / "trajectory.npz", qpos=Q, fps=ep.fps, labels=np.array(ep.labels))
    # the scene with the objects at their new starts (static objects move in the model; free ones start where qpos says)
    out_scene = ws / "scene_adapted.xml"
    _abs_files(scene, out_scene)
    if overrides:                                    # objects fixed to the world move in the scene itself
        tree = ET.parse(out_scene); root = tree.getroot()
        for b in root.iter("body"):
            ov = overrides.get(b.get("name"))
            if ov:
                b.set("pos", " ".join(f"{v:.6f}" for v in ov["pos"])); b.set("quat", " ".join(f"{v:.6f}" for v in ov["quat_wxyz"]))
                for a in ("euler", "axisangle", "xyaxes", "zaxis"):
                    b.attrib.pop(a, None)
        ET.indent(root); tree.write(out_scene)
    # the audits an adapted trajectory must pass: no worse than the verified example it came from (hand-built twins turn
    # the hand on the flange and touch things by design; the adaptation may not add to it): every check the adapted
    # trajectory fails, the example's own trajectory fails too, and a self-collision is no deeper (2 mm)
    audits = {}
    try:
        from tool_library.verifiers.arm_reach_check import arm_reach_check
        from tool_library.verifiers.robot_self_collision_check import robot_self_collision_check
        for name, fn in (("arm_reach_check", arm_reach_check), ("robot_self_collision_check", robot_self_collision_check)):
            a, b = fn(str(out_scene), str(ws / "trajectory.npz")), fn(str(scene), str(trajectory))
            new = sorted(set(a["failed"]) - set(b["failed"]))
            depth = lambda w: float(w.get("depth_mm", 1000.0 * float(w.get("depth_m", 0.0))))
            deeper = [c for c, w in (a.get("worst") or {}).items()
                      if c in (b.get("worst") or {}) and depth(w) > depth(b["worst"][c]) + 2.0]
            audits[name] = {"ok": bool(a["ok"]) or not (new or deeper), "failed": a["failed"], "example_failed": b["failed"],
                            "new": new, "deeper": deeper}
    except Exception as e:                            # reported as a failed audit, never a crash of the adaptation
        audits = {"audits": {"ok": False, "failed": [f"{type(e).__name__}: {e}"]}}
    ok = bool(report.get("ok")) and all(v.get("ok", False) for v in audits.values() if isinstance(v, dict))
    rep = {"ok": ok, "failed": (report.get("failed") or []) + [k for k, v in audits.items() if isinstance(v, dict) and not v.get("ok", True)],
           "layout": lay, "report": report, "audits_vs_example": audits, "example": str(ex),
           "trajectory": str(ws / "trajectory.npz"), "scene": str(out_scene), "static_overrides": overrides}
    (ws / "adapt_report.json").write_text(json.dumps(rep, indent=1, default=float))
    return rep


# ----------------------------------------------------------------------------- save a verified run
def make_portable(example_dir, run_dir=None):
    """Rewrite every JSON file of an example so it names no path of the machine it was made on: a path inside the run
    becomes relative to the run (stage3/poses/x.json), a path inside the repository relative to it, any other absolute
    path its file name."""
    ex = Path(example_dir)
    run = str(Path(run_dir).resolve()) + os.sep if run_dir else None
    repo = str(REPO.resolve()) + os.sep

    def fix(v):
        if isinstance(v, str) and v.startswith(os.sep):
            if run and v.startswith(run):
                return v[len(run):]
            if v.startswith(repo):
                return v[len(repo):]
            return Path(v).name
        if isinstance(v, list):
            return [fix(x) for x in v]
        if isinstance(v, dict):
            return {k: fix(x) for k, x in v.items()}
        return v

    for f in ex.rglob("*.json"):
        if f.name == "fingerprint.json":
            continue
        try:
            d = json.loads(f.read_text())
        except (ValueError, OSError):
            continue
        d2 = fix(d)
        if d2 != d:
            f.write_text(json.dumps(d2, indent=1))


def save_example(run_dir, name=None, video=None, examples_dir=None, overwrite=False, group="asset_reuse", provenance=None,
                 through_stage=3):
    """Package a run whose stages all passed into examples/<group>/<name> (no video; group 'asset_reuse' for runs saved
    by examples.save_assets, 'default' for the examples the repository ships): stage 1-3 files, the scene with its
    meshes in the shared pool, the video's fingerprint."""
    from tool_library.common.video import video_fingerprint
    run = Path(run_dir)
    root = examples_root(examples_dir)
    out = root / group / (name or run.name)
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"{out} exists")
        shutil.rmtree(out)
    pool = root / POOL_NAME
    s1, s2, s3 = run / "stage1", run / "stage2", run / "stage3"
    for s in ("stage1", "stage2") + (("stage3/stage4_task",) if int(through_stage) >= 3 else ()):
        (out / s).mkdir(parents=True, exist_ok=True)
    shutil.copy2(_semantics(s1), out / "stage1" / "scene_semantics.json")
    recon = json.loads((s2 / "scene_recon.json").read_text())
    export_pooled(s2 / recon.get("scene_xml", "scene.xml"), out / "stage2" / "scene.xml", pool)
    for f in ("camera.json", "frame0_compare.png"):
        if (s2 / f).exists():
            shutil.copy2(s2 / f, out / "stage2" / f)
    frame0 = s2 / recon.get("frame0", "frames/frame00000.jpg")
    if frame0.exists():
        shutil.copy2(frame0, out / "stage2" / "frame0.jpg")
    written = {}                                  # objects that share one asset (a box and its lid) keep sharing one file
    for o in recon.get("objects", []):
        ax = s2 / o.get("asset_xml", "") if o.get("asset_xml") else None
        if ax and ax.exists() and ax.suffix == ".xml":
            key = str(ax.resolve())
            if key in written:
                o["asset_xml"] = written[key]; continue
            dst = out / "stage2" / "objects" / o["id"] / f"{o['id']}.xml"
            try:
                export_pooled(ax, dst, pool)
                o["asset_xml"] = written[key] = str(dst.relative_to(out / "stage2"))
            except FileNotFoundError:
                pass
    # the files Stage 2's manifest check re-reads on reuse: every object's mask (and the hands' mask), and for a
    # deformable its Isaac report and images (the stage file only when small: Isaac stages run to tens of MB)
    for o in recon.get("objects", []):
        if o.get("mask") and (s2 / o["mask"]).exists():
            (out / "stage2" / o["mask"]).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s2 / o["mask"], out / "stage2" / o["mask"])
        isa = o.get("isaac") or {}
        for k, v in isa.items():
            if not (isinstance(v, str) and (s2 / v).is_file()):
                continue
            (out / "stage2" / v).parent.mkdir(parents=True, exist_ok=True)
            if (s2 / v).stat().st_size > 5_000_000:        # an Isaac stage runs to tens of MB of text: shipped gzipped
                import gzip
                with open(s2 / v, "rb") as fi, gzip.open(out / "stage2" / (v + ".gz"), "wb", compresslevel=9) as fo:
                    shutil.copyfileobj(fi, fo)
            else:
                shutil.copy2(s2 / v, out / "stage2" / v)
    if (s2 / "masks" / "hands.png").exists():
        (out / "stage2" / "masks").mkdir(parents=True, exist_ok=True)
        shutil.copy2(s2 / "masks" / "hands.png", out / "stage2" / "masks" / "hands.png")
    recon.update({"scene_xml": "scene.xml", "frame0": "frame0.jpg", "source": "examples/asset_reuse"})
    (out / "stage2" / "scene_recon.json").write_text(json.dumps(recon, indent=1))
    if int(through_stage) < 3:
        # a verified Stage 1 and Stage 2 only: Stage 3 of a run on this video is solved by the agent
        vid = video
        fp = video_fingerprint(str(vid))
        (out / "fingerprint.json").write_text(json.dumps(fp))
        sem = json.loads(_semantics(s1).read_text())
        (out / "manifest.json").write_text(json.dumps({
            "task": name or run.name, "prompt": sem.get("task_prompt"), "fingerprint": "fingerprint.json",
            "video": {k: fp[k] for k in ("duration_s", "fps", "resolution")},
            "objects": [{"id": o.get("id"), "type": o.get("type"), "root_body": o.get("root_body")} for o in recon.get("objects", [])],
            "stages": {"stage1": "stage1/scene_semantics.json", "stage2": "stage2/scene_recon.json"},
            "provenance": provenance or {"run": run.name}, "verified": {"stage_verifiers": "stages 1-2 passed at the end of the run"}}, indent=1))
        make_portable(out, run)
        if vid:
            write_certificate(out, vid)
        strip_example_images(out)                     # examples ship no image files (masks.npz, frame 0 signature)
        return {"example": str(out), "files": sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())}
    man = json.loads((s3 / "trajectory_manifest.json").read_text())
    for f in ("trajectory.npz", "phases.json"):
        if (s3 / f).exists():
            shutil.copy2(s3 / f, out / "stage3" / f)
    task_dir = s3 / man.get("stage4", {}).get("task_dir", "stage4_task")
    cfg = json.loads((task_dir / "stage4.json").read_text())
    scene_rel = "../../stage2/scene.xml"
    if man.get("scene_executed"):
        # the episode ran in a scene with moved stations: that scene travels with the trajectory (and its report)
        se = Path(man["scene_executed"]); se = se if se.is_absolute() else s3 / se
        export_pooled(se, out / "stage3" / "scene_executed.xml", pool)
        man["scene_executed"] = "scene_executed.xml"; scene_rel = "../scene_executed.xml"
        if (s3 / "place_arm_stations.json").exists():
            shutil.copy2(s3 / "place_arm_stations.json", out / "stage3" / "place_arm_stations.json")
    # the hand-off's own files: its trajectory carries the servo commands (reference) the physics replay needs, its
    # manifest the subgoal expectations, and the goal files Stage 1's goals and the object map
    if isinstance(man.get("deformables"), dict):
        for d_src, d_dst in ((s3, out / "stage3"), (task_dir, out / "stage3" / "stage4_task")):
            for f in ISAAC_FILES:
                if (d_src / f).is_file():
                    (d_dst / f).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(d_src / f, d_dst / f)
            if (d_src / "scene_mirror.xml").is_file():
                export_pooled(d_src / "scene_mirror.xml", d_dst / "scene_mirror.xml", pool)
    own_traj = (task_dir / "trajectory.npz").exists()
    for f in ("trajectory.npz", "trajectory_manifest.json", "goal_objects.json", "scene_semantics.json"):
        if (task_dir / f).exists():
            shutil.copy2(task_dir / f, out / "stage3" / "stage4_task" / f)
    cfg.update({"scene": scene_rel, "trajectory": "trajectory.npz" if own_traj else "../trajectory.npz", "human_video": "human_video.mp4"})
    cfg.pop("sim_video", None)                           # examples ship no videos
    (out / "stage3" / "stage4_task" / "stage4.json").write_text(json.dumps(cfg, indent=1))
    man["source"] = "examples/asset_reuse"
    (out / "stage3" / "trajectory_manifest.json").write_text(json.dumps(man, indent=1))
    vid = video or (task_dir / "human_video.mp4")
    fp = video_fingerprint(str(vid))
    (out / "fingerprint.json").write_text(json.dumps(fp))
    sem = json.loads(_semantics(s1).read_text())
    (out / "manifest.json").write_text(json.dumps({
        "task": name or run.name, "prompt": sem.get("task_prompt"), "fingerprint": "fingerprint.json",
        "video": {k: fp[k] for k in ("duration_s", "fps", "resolution")},
        "objects": [{"id": o.get("id"), "type": o.get("type"), "root_body": o.get("root_body")} for o in recon.get("objects", [])],
        "stages": {"stage1": "stage1/scene_semantics.json", "stage2": "stage2/scene_recon.json", "stage3": "stage3/trajectory_manifest.json",
                   "stage4": "stage3/stage4_task/stage4.json"},
        "provenance": provenance or {"run": run.name}, "verified": {"stage_verifiers": "passed at the end of the run"}}, indent=1))
    make_portable(out, run)
    if Path(vid).is_file():
        write_certificate(out, vid)
    strip_example_images(out)                         # examples ship no image files (masks.npz, frame 0 signature)
    return {"example": str(out), "files": sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())}


REPLAY_DIR = "replay"            # stage3/replay/: the executable plan that produced the example's trajectory


def attach_replay(example_dir, source_stage3):
    """Ship the executable Stage 3 plan of an example: plan.json and every JSON file it references (poses, hand scenes),
    copied under stage3/replay/ with machine paths rewritten as ${REPLAY}/..., ${STAGE2}/... (the run's reconstruction)
    and ${SETUP}/... (the configured setup), and the station shift the episode ran with (stations.json). A run on the
    same recording replays it (prepare_replay) under the current runtime and checks instead of copying a trajectory."""
    ex, src = Path(example_dir), Path(source_stage3).resolve()
    run = next((p for p in src.parents if (p / "stage2").is_dir()), src.parent)   # an attempt lives in stage3/.candidates/<id>
    out = ex / "stage3" / REPLAY_DIR
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    if (src / "plan.json").is_file():
        plan = json.loads((src / "plan.json").read_text())
    else:                                         # older runs: the executed plan entries recorded per subgoal
        man = json.loads((src / "trajectory_manifest.json").read_text())
        plan = {"subgoals": [sg["plan"] for sg in man.get("subgoals", []) if isinstance(sg.get("plan"), dict)]}
    copied = {}

    def localize(value, base):
        if isinstance(value, dict):
            return {k: localize(v, base) for k, v in value.items()}
        if isinstance(value, list):
            return [localize(v, base) for v in value]
        if not isinstance(value, str) or "/" not in value and not value.endswith(".json"):
            return value
        if any(ch.isspace() for ch in value.strip()):      # prose (a provenance note): keep file names, drop machine paths
            import re
            return re.sub(r"/[^\s'\",]*/", "", value)
        p = Path(value)
        cand = (p if p.is_absolute() else base / p)
        try:
            cand = cand.resolve()
        except OSError:
            return value
        if cand.suffix == ".json" and cand.is_file() and (cand == src or src in cand.parents):
            rel = str(cand.relative_to(src))
            if rel not in copied:
                copied[rel] = True
                dst = out / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(json.dumps(localize(json.loads(cand.read_text()), cand.parent), indent=1))
            return "${REPLAY}/" + rel
        if (run / "stage2") in cand.parents:
            return "${STAGE2}/" + str(cand.relative_to(run / "stage2"))
        if "/setup/" in value:
            return "${SETUP}/" + value.split("/setup/", 1)[1]
        return value

    plan = localize(plan, src)
    plan.pop("scene_recon_json", None)
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    stations = None
    if (src / "place_arm_stations.json").is_file():
        stations = json.loads((src / "place_arm_stations.json").read_text()).get("shift")
        (out / "stations.json").write_text(json.dumps({"shift": stations}, indent=1))
    return {"replay": str(out), "files": sorted(copied), "stations": stations}


def prepare_replay(example, workspace, stage2_dir, setup_dir):
    """Resolve an example's shipped replay plan into a run: ${REPLAY} -> <workspace>/gt_replay, ${STAGE2} -> the run's
    Stage 2 folder (an object's assets/<id>/<id>.xml -> objects/<id>/<id>.xml as reuse writes it), ${SETUP} -> the
    configured setup. Returns {plan, stations (shift or None), example} or None when the example ships no replay."""
    src = Path(example) / "stage3" / REPLAY_DIR
    if not (src / "plan.json").is_file():
        return None
    ws, s2, setup = Path(workspace).resolve(), Path(stage2_dir).resolve(), Path(setup_dir).resolve()
    out = ws / "gt_replay"
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(src, out)

    def resolve(value):
        if isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v) for v in value]
        if not isinstance(value, str):
            return value
        if value.startswith("${STAGE2}/"):
            rel = value[len("${STAGE2}/"):]
            parts = Path(rel).parts
            if len(parts) == 3 and parts[0] == "assets" and (s2 / "objects" / parts[1] / parts[2]).is_file():
                return str(s2 / "objects" / parts[1] / parts[2])
            return str(s2 / rel)
        return value.replace("${REPLAY}", str(out)).replace("${SETUP}", str(setup))

    for f in out.rglob("*.json"):
        f.write_text(json.dumps(resolve(json.loads(f.read_text())), indent=1))
    plan = json.loads((out / "plan.json").read_text())
    plan["scene_recon_json"] = str(s2 / "scene_recon.json")
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    stations = json.loads((out / "stations.json").read_text()).get("shift") if (out / "stations.json").is_file() else None
    return {"plan": str(out / "plan.json"), "stations": stations, "example": str(example)}


def write_certificate(example_dir, video):
    """stage2/example_certificate.json: the recording this example was made from (file digest) and the digests and
    roles of its frame 0 and masks. On reuse, agent.reference_evidence.certify_example accepts them as the run's
    reference evidence only for that same recording (identical file, or the same decoded first frame)."""
    from tool_library.common.evidence import file_digest
    ex2 = Path(example_dir) / "stage2"
    recon = json.loads((ex2 / "scene_recon.json").read_text())
    masks = [{"path": o["mask"], "role": "object", "object": o.get("id"), "sha256": file_digest(ex2 / o["mask"])}
             for o in recon.get("objects", []) if o.get("mask") and (ex2 / o["mask"]).is_file()]
    if (ex2 / "masks" / "hands.png").is_file():
        masks.append({"path": "masks/hands.png", "role": "hands", "sha256": file_digest(ex2 / "masks" / "hands.png")})
    for o in recon.get("objects", []):                 # an object hidden at frame 0: the mask of the frame that reveals it
        h = o.get("hidden_at_frame0") or {}
        if h.get("reveal_mask") and (ex2 / h["reveal_mask"]).is_file():
            masks.append({"path": h["reveal_mask"], "role": "reveal", "object": o.get("id"), "frame": int(h["reveal_frame"]),
                          "sha256": file_digest(ex2 / h["reveal_mask"])})
    frame0 = recon.get("frame0", "frame0.jpg")
    from agent.reference_evidence import frame0_matches_example
    same, measured = frame0_matches_example(video, ex2 / frame0)
    if not same:
        raise ValueError(f"{example_dir}: its frame 0 is not the first frame of {video} ({measured}); certify only an "
                         f"example reconstructed from that recording")
    cert = {"schema": "dexagent.example_certificate.v1", "task": Path(example_dir).name,
            "source_video_sha256": file_digest(video), "frame0": {"path": frame0, "sha256": file_digest(ex2 / frame0)},
            "masks": masks}
    (ex2 / "example_certificate.json").write_text(json.dumps(cert, indent=1))
    return cert


def layout_from_recon(example, scene_recon_json):
    """Per-object (dx, dy, dyaw_deg) of a run's Stage 2 placements relative to the example's frame-0 placements,
    for the objects the example's Stage 4 config may move (matched by Stage 1 id, then by root body)."""
    ex = Path(example)
    ex_recon = json.loads((ex / "stage2" / "scene_recon.json").read_text())
    cfg = json.loads((ex / "stage3" / "stage4_task" / "stage4.json").read_text())
    movable = set(cfg["randomize"]["objects"])
    new = json.loads(Path(scene_recon_json).read_text())
    new_by_id = {o["id"]: o for o in new.get("objects", [])}
    new_by_root = {o.get("root_body"): o for o in new.get("objects", [])}
    out, unmatched = {}, []
    for o in ex_recon.get("objects", []):
        body = o["root_body"]
        if body not in movable:
            continue
        n = new_by_id.get(o["id"]) or new_by_root.get(body)
        if n is None:
            unmatched.append(o["id"]); continue
        p0, p1 = o["pose"], n["pose"]
        dyaw = (float(p1.get("yaw_deg", 0.0)) - float(p0.get("yaw_deg", 0.0)) + 180.0) % 360.0 - 180.0
        out[body] = [float(p1["x"]) - float(p0["x"]), float(p1["y"]) - float(p0["y"]), dyaw]
    return {"objects": out, "unmatched": unmatched}


FIT_BOUNDS = {"x": 0.05, "y": 0.05, "yaw": 20.0}     # around the start pose: the run's Stage 2 pose, the example's own


def _read_mask(path, size=None):
    import cv2
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path and Path(path).is_file() else None
    if a is None:
        return None
    a = a[..., 0] if a.ndim == 3 else a
    if size and a.shape[::-1] != tuple(size):
        a = cv2.resize(a, tuple(size), interpolation=cv2.INTER_NEAREST)
    return a > 0


def _frame0(video, out_jpg, size):
    import cv2
    cap = cv2.VideoCapture(str(video))
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"cannot read frame 0 of {video}")
    if img.shape[1::-1] != tuple(size):
        img = cv2.resize(img, tuple(size), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_jpg), img)
    return img


def transfer_mask(src_img, dst_img, mask, ignore=None, search=160):
    """An object's mask on one frame carried onto another frame of the same rig: the object's neighbourhood of src_img
    located in dst_img (template match within `search` px), then the Euclidean warp (ECC) from there, applied to the
    mask. Returns (mask, warp 2x3, converged); the identity when ECC does not converge."""
    import cv2
    a = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(dst_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    region = cv2.dilate(mask.astype(np.uint8), np.ones((41, 41), np.uint8))
    if ignore is not None:
        region[ignore] = 0
    warp = np.eye(2, 3, dtype=np.float32)
    ys, xs = np.nonzero(region)
    if len(xs):                                          # coarse: where the patch went (ECC alone is local)
        H, W = mask.shape
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        X0, Y0 = max(0, x0 - search), max(0, y0 - search)
        win = b[Y0:min(H, y1 + search), X0:min(W, x1 + search)]
        if win.shape[0] > y1 - y0 and win.shape[1] > x1 - x0:
            r = cv2.matchTemplate(win, a[y0:y1, x0:x1], cv2.TM_CCOEFF_NORMED, mask=region[y0:y1, x0:x1].astype(np.float32))
            _, _, _, (mx, my) = cv2.minMaxLoc(np.nan_to_num(r, nan=-1.0))
            warp[:, 2] = [X0 + mx - x0, Y0 + my - y0]
    coarse = warp.copy()
    try:
        _, warp = cv2.findTransformECC(a, b, warp, cv2.MOTION_EUCLIDEAN,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), region, 5)
        ok = True
    except cv2.error:
        warp, ok = coarse, False                         # the coarse translation stands
    H, W = mask.shape
    return cv2.warpAffine(mask.astype(np.uint8), warp, (W, H), flags=cv2.INTER_NEAREST) > 0, warp, ok


def fit_layout(example, stage2_dir, video, out_dir, objects=None):
    """Stage 3 reuse's position fit: the example's assets fitted to the objects of the run's own video. Each object's
    pose is fitted to its mask on the run's frame 0 (silhouette IoU through the run's camera, from the run's Stage 2
    pose; a deformable's footprint centre on the table), the same fit runs on the example's frame 0 from the example's
    pose, and the difference of the two fits is the object's (dx, dy, dyaw_deg), so the fitter's own bias cancels. The
    run's masks are its Stage 2 masks; a Stage 2 copied from this example holds the example's masks, which are carried
    onto the run's frame 0 by image alignment (transfer_mask) first. `objects`: the root bodies to fit (default all).
    Returns {"objects": {body: [dx, dy, dyaw_deg]}, "fits": {body: ...}, "unmatched": [ids]}; overlays in out_dir."""
    import cv2
    import tempfile
    from tool_library.skills.fit_object_placement import fit_object_placement
    from tool_library.skills.measure_mask_footprint import measure_mask_footprint
    ex, s2, out = Path(example), Path(stage2_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ex_rec = json.loads((ex / "stage2" / "scene_recon.json").read_text())
    rec = json.loads((s2 / "scene_recon.json").read_text())
    copied = bool(str(rec.get("reused_from") or "")) and Path(rec["reused_from"]).resolve() == ex.resolve()
    ex_masks = example_masks(ex / "stage2")
    ex_img = cv2.imread(str(ex / "stage2" / "frame0.jpg")) if (ex / "stage2" / "frame0.jpg").is_file() else None
    if ex_img is not None:
        H, W = ex_img.shape[:2]
    else:                                             # examples ship no frame image: the size of their masks
        H, W = next(iter(ex_masks.values())).shape if ex_masks else video_frame0(video).shape[:2]
    run_img = _frame0(video, out / "run_frame0.jpg", (W, H))
    ex_frame = str(ex / "stage2" / "frame0.jpg") if ex_img is not None else str(out / "run_frame0.jpg")

    def ex_mask_of(rel):
        m = ex_masks.get(str(rel or ""))
        if m is None:
            return None
        return m if m.shape == (H, W) else cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    ex_hands = ex_mask_of("masks/hands.png")
    run_hands = ex_hands if copied else _read_mask(s2 / "masks" / "hands.png", (W, H))
    by_id = {o["id"]: o for o in rec.get("objects", [])}
    by_root = {o.get("root_body"): o for o in rec.get("objects", [])}
    lay, fits, unmatched = {}, {}, []
    with tempfile.TemporaryDirectory(dir=str(out)) as tmp:
        ref = Path(tmp)
        _abs_files(ex / "stage2" / "scene.xml", ref / "scene.xml", ref / "assets", absolute=True)
        for xml in (ex / "stage2" / "objects").glob("*/*.xml"):
            _abs_files(xml, ref / "objects" / xml.parent.name / xml.name, ref / "assets", absolute=True)
        for o in ex_rec.get("objects", []):
            body = o["root_body"]
            if objects is not None and body not in objects:
                continue
            n = by_id.get(o["id"]) or by_root.get(body)
            ex_mask = ex_mask_of(o.get("mask"))
            if n is None or ex_mask is None:
                unmatched.append(o["id"])
                continue
            own = None if copied else _read_mask(s2 / str(n.get("mask") or ""), (W, H))
            if own is None and ex_img is None:
                # a Stage 2 copied from this example is certified to be its recording: the same frame 0, the same mask
                run_mask, source = ex_mask, {"mask": "the example's mask (this run's recording is the example's, certified)"}
            elif own is None:
                run_mask, warp, conv = transfer_mask(ex_img, run_img, ex_mask, ex_hands)
                source = {"mask": "the example's mask carried onto this frame 0", "ecc_converged": conv,
                          "warp": np.round(warp, 5).tolist()}
            else:
                run_mask, source = own, {"mask": "this run's Stage 2 mask"}
            if o.get("type") == "deformable":           # a cord's shape is a state: its footprint centre on the table
                a = measure_mask_footprint(ex_mask, str(ex / "stage2" / "camera.json"), ex_hands)["contact"]["center_xy"]
                b = measure_mask_footprint(run_mask, str(s2 / "camera.json"), run_hands)["contact"]["center_xy"]
                lay[body] = [float(b[0] - a[0]), float(b[1] - a[1]), 0.0]
                fits[body] = {"method": "mask footprint centre", "example_xy": a, "run_xy": b, **source}
                continue
            kw = dict(init_from_mask=False, bounds=FIT_BOUNDS, seed=0)
            fe = fit_object_placement(str(ref / "scene.xml"), body, ex_mask, str(ex / "stage2" / "camera.json"),
                                      ignore_mask=ex_hands, out_png=str(out / f"{body}_example.png"),
                                      frame=ex_frame, **kw)
            fr = fit_object_placement(str(s2 / "scene.xml"), body, run_mask, str(s2 / "camera.json"),
                                      ignore_mask=run_hands, out_png=str(out / f"{body}_run.png"),
                                      frame=str(out / "run_frame0.jpg"), **kw)
            pe, pr = fe["pose"], fr["pose"]
            lay[body] = [pr["x"] - pe["x"], pr["y"] - pe["y"], (pr["yaw_deg"] - pe["yaw_deg"] + 180.0) % 360.0 - 180.0]
            fits[body] = {"method": "silhouette IoU", "example": {k: round(pe[k], 5) for k in ("x", "y", "yaw_deg")},
                          "run": {k: round(pr[k], 5) for k in ("x", "y", "yaw_deg")},
                          "iou_example": fe["iou"], "iou_run": fr["iou"], **source}
    result = {"objects": lay, "fits": fits, "unmatched": unmatched, "stage2_copied_from_example": copied}
    (out / "position_fit.json").write_text(json.dumps(result, indent=1, default=float))
    return result
