"""Video and mask helpers shared by the perception and rendering tools.

Frames on disk follow one convention everywhere: `frame%05d.jpg` (or .png) in a
folder, numbered from 0 in the order they were extracted. Label maps are
palette PNGs whose pixel value is the object's label (1..K, 0 = background),
which is what SAM 2 tracking writes and what every consumer reads back with
PIL (cv2 would decode the palette to colours).
"""
from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np

PALETTE_COLORS = [(255, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0),
                  (255, 128, 0), (0, 128, 255), (255, 0, 0), (0, 0, 255)]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def video_info(path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {path}")
    info = {"path": str(path), "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
    cap.release()
    info["duration_s"] = round(info["frames"] / info["fps"], 3) if info["fps"] else None
    return info


def read_video(path, start=0, stop=None, stride=1):
    """Decode frames [start, stop) with a stride -> list of RGB uint8 arrays."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {path}")
    out, t = [], 0
    while True:
        ok, f = cap.read()
        if not ok or (stop is not None and t >= stop):
            break
        if t >= start and (t - start) % max(1, stride) == 0:
            out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        t += 1
    cap.release()
    return out


def frame_files(frames_dir) -> list:
    """Sorted frame paths in a folder (jpg or png, `frame*` first)."""
    d = str(frames_dir)
    files = sorted(glob.glob(os.path.join(d, "frame*.jpg")) + glob.glob(os.path.join(d, "frame*.png")))
    if not files:
        files = sorted(glob.glob(os.path.join(d, "*.jpg")) + glob.glob(os.path.join(d, "*.png")))
    return files


def read_frame(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_frames(frames, out_dir, ext="jpg", quality=95, start=0) -> list:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, f in enumerate(frames, start):
        p = out_dir / f"frame{i:05d}.{ext}"
        bgr = cv2.cvtColor(np.asarray(f), cv2.COLOR_RGB2BGR)
        if ext == "jpg":
            cv2.imwrite(str(p), bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        else:
            cv2.imwrite(str(p), bgr)
        paths.append(str(p))
    return paths


class VideoWriter:
    """Incremental RGB encoder: memory use does not grow with episode length."""
    def __init__(self, path, fps=30., crf=18):
        self.path, self.fps, self.crf = Path(path), float(fps), int(crf)
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("video fps must be positive and finite")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer, self.shape, self.ffmpeg = None, None, False

    def __enter__(self):
        return self

    def append_data(self, frame):
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[-1] < 3 or frame.dtype != np.uint8:
            raise ValueError("video frames must be uint8 RGB images")
        frame = np.ascontiguousarray(frame[:frame.shape[0] // 2 * 2, :frame.shape[1] // 2 * 2, :3])
        if min(frame.shape[:2]) < 2 or self.shape is not None and frame.shape != self.shape:
            raise ValueError("video frames must have constant positive dimensions")
        if self.writer is None:
            self.shape = frame.shape
            try:
                import imageio_ffmpeg  # noqa: F401
                import imageio.v2 as imageio
            except ImportError:
                self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps,
                                               (frame.shape[1], frame.shape[0]))
                if not self.writer.isOpened():
                    raise RuntimeError(f"cannot encode video {self.path}")
            else:
                self.ffmpeg = True
                self.writer = imageio.get_writer(str(self.path), fps=self.fps, macro_block_size=1, codec="libx264",
                                                  pixelformat="yuv420p", output_params=["-crf", str(self.crf)])
        if self.ffmpeg:
            self.writer.append_data(frame)
        else:
            self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def __exit__(self, exc_type, exc, tb):
        if self.writer is not None:
            self.writer.close() if self.ffmpeg else self.writer.release()
        elif exc_type is None:
            raise ValueError("cannot encode an empty video")


def write_video(frames, path, fps=30.0, crf=18):
    """Stream RGB frames to H.264, with an OpenCV mp4v fallback."""
    with VideoWriter(path, fps, crf) as writer:
        for frame in frames:
            writer.append_data(frame)
    return str(path)


def save_label_map(labels: np.ndarray, path, n_labels: int | None = None):
    """Write a palette PNG whose pixel values are object labels."""
    from PIL import Image
    n = int(n_labels or max(1, int(labels.max())))
    pal = [0, 0, 0] + [c for k in range(n) for c in PALETTE_COLORS[k % len(PALETTE_COLORS)]]
    pal += [0] * (768 - len(pal))
    im = Image.fromarray(labels.astype(np.uint8), mode="P")
    im.putpalette(pal)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    im.save(str(path))


def load_label_map(path) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(str(path)))


def overlay_labels(rgb: np.ndarray, labels: np.ndarray, names=None, alpha=0.45) -> np.ndarray:
    """Colour each label on the frame and write its name in the corner."""
    vis = rgb.astype(np.float32).copy()
    for k in range(1, int(labels.max()) + 1):
        on = labels == k
        if not on.any():
            continue
        c = np.array(PALETTE_COLORS[(k - 1) % len(PALETTE_COLORS)], np.float32)
        vis[on] = vis[on] * (1 - alpha) + c * alpha
    vis = vis.astype(np.uint8)
    for k, name in enumerate(names or []):
        cv2.putText(vis, str(name), (8, 22 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    PALETTE_COLORS[k % len(PALETTE_COLORS)], 2, cv2.LINE_AA)
    return vis


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=_json_default))
    return str(path)


def _json_default(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, Path):
        return str(v)
    raise TypeError(f"not JSON serialisable: {type(v)}")


# ----------------------------------------------------------------------------- perceptual fingerprint
FP_FRAMES, FP_W, FP_H = 24, 32, 24


def video_fingerprint(path, n=FP_FRAMES, width=FP_W, height=FP_H) -> dict:
    """A small perceptual fingerprint of a video: `n` frames at equal fractions of its length, each grey, resized to
    width x height and normalised to zero mean, unit variance (so re-encoding, resizing and exposure shifts barely
    move it). Stored as rounded lists; compare two with fingerprint_similarity."""
    info = video_info(path)
    total = int(info["frames"]) or len(read_video(path))
    idx = [min(total - 1, int(round(f * (total - 1)))) for f in np.linspace(0, 1, int(n))]
    cap = cv2.VideoCapture(str(path)); frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, i - 2)); ok, f = cap.read()
        g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (int(width), int(height)), interpolation=cv2.INTER_AREA).astype(np.float32) if ok else np.zeros((int(height), int(width)), np.float32)
        g = (g - g.mean()) / (g.std() + 1e-6)
        frames.append(np.round(g, 2).ravel().tolist())
    return {"version": 1, "n": int(n), "size": [int(width), int(height)], "frames": frames,
            "duration_s": info["duration_s"], "fps": info["fps"], "resolution": [info["width"], info["height"]]}


def fingerprint_similarity(a: dict, b: dict) -> dict:
    """Mean normalised cross-correlation of matched fingerprint frames (1 = the same video), the first frame's alone
    (the initial state) and the duration ratio."""
    A, B = np.asarray(a["frames"], np.float32), np.asarray(b["frames"], np.float32)
    if A.shape != B.shape:
        return {"mean_ncc": 0.0, "frame0_ncc": 0.0, "duration_ratio": 0.0}
    ncc = np.clip((A * B).mean(axis=1), -1.0, 1.0)          # rounded fingerprints can overshoot 1 by a hair
    da, db = float(a.get("duration_s") or 0), float(b.get("duration_s") or 0)
    return {"mean_ncc": round(float(ncc.mean()), 4), "min_ncc": round(float(ncc.min()), 4), "frame0_ncc": round(float(ncc[0]), 4),
            "duration_ratio": round(min(da, db) / max(da, db), 4) if da > 0 and db > 0 else 0.0}
