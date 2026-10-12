"""Step 5: composite the rendered robot over the clean plate and encode.

    frame = plate * (1 - alpha) + render * alpha

with the render's own alpha, so anti-aliased edges and the shadows caught on
the table come across in one paste. Frames are streamed: one frame of every
view is in memory at a time. Writes, per sample (only the samples the last
randomize step published):
    <view>.mp4       one video per camera view
    preview.mp4      all views side by side
and for the un-randomized sample, compare.mp4: human video | simulation
video (if given) | generated ego view.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import imageio
import numpy as np


def writer(path, fps, crf):
    return imageio.get_writer(path, fps=fps, macro_block_size=1, codec="libx264",
                              output_params=["-crf", str(crf), "-preset", "slow",
                                             "-pix_fmt", "yuv420p"])


def view_frames(vdir):
    """Composited RGB frames of one view."""
    meta = json.loads((vdir / "scene.json").read_text())
    W, H = meta["width"], meta["height"]
    static = None
    if "plate" in meta:
        static = cv2.cvtColor(cv2.resize(cv2.imread(meta["plate"]), (W, H)),
                              cv2.COLOR_BGR2RGB).astype(np.float32)
    for t in range(meta["n_frames"]):
        rgba = cv2.imread(str(vdir / "frames" / f"f{t:05d}.png"), cv2.IMREAD_UNCHANGED)
        assert rgba is not None and rgba.shape[2] == 4, (
            f"missing or unreadable render {vdir / 'frames' / f'f{t:05d}.png'}")
        if static is not None:
            plate = static
        else:
            plate = cv2.cvtColor(cv2.imread(f"{meta['plates_dir']}/frame{t:05d}.png"),
                                 cv2.COLOR_BGR2RGB).astype(np.float32)
        a = rgba[..., 3:].astype(np.float32) / 255.0
        rgb = cv2.cvtColor(rgba[..., :3], cv2.COLOR_BGR2RGB).astype(np.float32)
        yield (plate * (1 - a) + rgb * a).astype(np.uint8)


def label(img, text):
    img = img.copy()
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(img, (0, 0), (w + 12, h + 12), (0, 0, 0), -1)
    cv2.putText(img, text, (6, h + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                cv2.LINE_AA)
    return img


def iter_video(path, n, size):
    """n frames spread evenly over a video, resized to size (W, H), read one at a time (RGB)."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    want = np.linspace(0, max(0, total - 1), n).round().astype(int)
    k, t, last = 0, 0, None
    try:
        while k < n:
            ok, f = cap.read()
            if not ok:
                break
            while k < n and want[k] == t:
                last = cv2.cvtColor(cv2.resize(f, size), cv2.COLOR_BGR2RGB)
                yield last
                k += 1
            t += 1
    finally:
        cap.release()
    assert last is not None, f"could not read any frame of {path}"
    while k < n:                                   # a frame count the container over-reports
        yield last
        k += 1


def resample(path, n, size):
    """n frames spread evenly over a video, resized (all in memory; encode() streams with iter_video)."""
    return list(iter_video(path, n, size))


def encode(view_dirs, out_dir, crf, references=()):
    """Composite and encode one rendered sample, holding one frame of every view in memory at a time.

    view_dirs  {view: render folder with scene.json and frames/}, ego (or the first view) first
    references [(name, video path)]: when given, also compare.mp4 = those videos | the generated first view
    Writes <view>.mp4 per view and preview.mp4 (all views side by side) into out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    views = list(view_dirs)
    metas = {v: json.loads((Path(view_dirs[v]) / "scene.json").read_text()) for v in views}
    m0 = metas[views[0]]
    fps, n, size = m0["fps"], int(m0["n_frames"]), (int(m0["width"]), int(m0["height"]))
    for v, m in metas.items():
        assert (int(m["n_frames"]), int(m["width"]), int(m["height"])) == (n, *size), (
            f"view {v} has {m['n_frames']} frames of {m['width']} x {m['height']}, {views[0]} {n} of "
            f"{size[0]} x {size[1]}: re-run the export step")
    frames = {v: view_frames(Path(view_dirs[v])) for v in views}
    refs = [(name, iter_video(path, n, size)) for name, path in references]
    writers = {}
    try:
        for v in views:
            writers[v] = writer(out / f"{v}.mp4", fps, crf)
        writers[" preview"] = writer(out / "preview.mp4", fps, crf)
        if refs:
            writers[" compare"] = writer(out / "compare.mp4", fps, crf)
        for _t in range(n):
            now = {v: next(frames[v]) for v in views}
            for v in views:
                writers[v].append_data(now[v])
            writers[" preview"].append_data(np.concatenate([label(now[v], v.replace("_", " ")) for v in views], 1))
            if refs:
                cols = [label(next(it), name) for name, it in refs] + [label(now[views[0]], "generated")]
                writers[" compare"].append_data(np.concatenate(cols, 1))
    finally:
        for w in writers.values():
            w.close()
    return {"videos": {v: str(out / f"{v}.mp4") for v in views}, "preview": str(out / "preview.mp4"),
            "compare": str(out / "compare.mp4") if refs else None, "n_frames": n, "fps": fps}


def run(cfg, log=print):
    from .export import current_samples, stage_views
    crf = int(cfg["render"]["crf"])
    views = stage_views(cfg)
    samples = current_samples(cfg)
    assert samples, "no samples: run the randomize step first"
    root = cfg.out / "render"
    for sdir in samples:
        rdir = root / sdir.name
        missing = [v for v in views if not (rdir / v / "scene.json").exists()]
        assert not missing, (f"sample {sdir.name} has no render bundle for {missing}: run the export and render "
                             "steps after randomize")
        refs = []
        if sdir.name == "original":
            refs = [("human video", cfg.path("human_video"))]
            if cfg.get("sim_video"):
                refs.append(("simulation", cfg.path("sim_video")))
        r = encode({v: rdir / v for v in views}, sdir, crf, refs)
        log(f"[composite] {sdir.name}: {r['n_frames']} frames @ {r['fps']:g} fps -> "
            f"{sdir}/{{{','.join(views)},preview}}.mp4")
