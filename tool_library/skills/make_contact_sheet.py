"""A labelled grid of video frames: how an agent looks at a whole clip in one image."""
from __future__ import annotations

TOOL = {
    "name": "make_contact_sheet",
    "title": "Make contact sheet",
    "type": "skill",
    "category": "Perception & Grounding",
    "summary": "Tile evenly spaced (or chosen) frames of a video or frame folder into one labelled grid image, optionally cropped, so a whole clip or a time window can be read at a glance.",
    "description": (
        "Reads a video or a folder of frame%05d images, picks `count` evenly spaced frames inside "
        "[start, end] (seconds, or exact `frames` indices), optionally crops every frame to a box "
        "given as fractions of the image, resizes each tile to `tile_width`, and tiles them in rows "
        "of `columns` with the frame index and timestamp printed in each tile's corner. Returns the "
        "sheet path and the frame indices and times used, so a frame seen on the sheet can be fetched "
        "at full resolution with extract_frames. Windows and crops are how a vision-language agent "
        "zooms into a moment or a region without decoding the whole clip."
    ),
    "inputs": [
        {"name": "video", "type": "path", "description": "A video file, or a folder of frames written by extract_frames (then fps comes from its meta.json or `fps`)."},
        {"name": "out_path", "type": "path", "description": "Where the sheet image is written (.jpg or .png)."},
        {"name": "count", "type": "int", "description": "Frames to show, evenly spaced over the window. Default 16. Ignored when `frames` is given."},
        {"name": "start", "type": "float, seconds", "description": "Window start. Default 0."},
        {"name": "end", "type": "float, seconds | None", "description": "Window end. Default: the end of the clip."},
        {"name": "frames", "type": "list[int] | None", "description": "Exact frame indices to show instead of an even spacing."},
        {"name": "columns", "type": "int", "description": "Tiles per row. Default 4."},
        {"name": "tile_width", "type": "int, pixels", "description": "Width of each tile; height follows the aspect. Default 480."},
        {"name": "crop", "type": "list[float] | None", "description": "[x0, y0, x1, y1] applied to every tile, as fractions of the frame (0..1) or as pixels of this video's frame (any value > 1; a box running past the frame is refused). Default: the full frame."},
        {"name": "fps", "type": "float | None", "description": "Frame rate when `video` is a frame folder without meta.json."},
    ],
    "outputs": [
        {"name": "sheet", "type": "path", "description": "The grid image."},
        {"name": "frames", "type": "list[int]", "description": "Frame indices shown, in tile order."},
        {"name": "times_s", "type": "list[float]", "description": "Their timestamps."},
        {"name": "fps / n_frames / duration_s", "type": "float / int / float", "description": "Clip facts."},
        {"name": "grid", "type": "[rows, columns]", "description": "Tile layout."},
    ],
    "status": "tested",
    "entrypoint": "make_contact_sheet",
    "requirements": ["opencv-python", "numpy"],
    "tags": ["video", "frames", "contact sheet", "look", "semantic understanding"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope", "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting", "Biology Experiment"],
    "cli": "python -m tool_library.skills.make_contact_sheet human_video.mp4 sheet.jpg --count 24 --columns 6",
}

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from tool_library.common.video import frame_files, read_json, video_info


def _source(video, fps):
    p = Path(video)
    if p.is_dir():
        files = frame_files(p)
        if not files:
            raise FileNotFoundError(f"no frames in {p}")
        meta = p / "meta.json"
        f = fps or (read_json(meta).get("fps") if meta.exists() else None) or 30.0
        def grab(i):
            return cv2.imread(str(files[i]))
        return len(files), float(f), grab, None
    info = video_info(p)
    cap = cv2.VideoCapture(str(p))
    def grab(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        return f if ok else None
    return info["frames"], float(info["fps"] or fps or 30.0), grab, cap


def make_contact_sheet(video, out_path, count=16, start=0.0, end=None, frames=None,
                       columns=4, tile_width=480, crop=None, fps=None):
    n, f, grab, cap = _source(video, fps)
    if n <= 0:
        raise ValueError(f"{video} has no frames")
    if frames is None:
        a = int(max(0, round(float(start) * f)))
        b = int(min(n - 1, round(float(end) * f))) if end is not None else n - 1
        if b < a:
            raise ValueError(f"window [{start}, {end}] s is empty for a clip of {n / f:.1f} s")
        idx = np.unique(np.linspace(a, b, int(max(1, count))).round().astype(int)).tolist()
    else:
        idx = [int(i) for i in frames]
        bad = [i for i in idx if not 0 <= i < n]
        if bad:
            raise ValueError(f"frames {bad} are outside 0..{n - 1}")
    tiles, shown = [], []                      # shown: the frames actually tiled (an undecodable frame is skipped)
    for i in idx:
        img = grab(i)
        if img is None:
            continue
        H, W = img.shape[:2]
        if crop is not None:
            x0, y0, x1, y1 = [float(v) for v in crop]
            if max(x0, y0, x1, y1) <= 1.0:           # fractions of the frame
                x0, x1, y0, y1 = x0 * W, x1 * W, y0 * H, y1 * H
            elif max(x0, x1) > W + 1 or max(y0, y1) > H + 1:
                # pixels of another frame (the human video's 1920 x 1080 on an 854 x 480 sim video) clipped silently to a
                # sliver of the wrong region
                raise ValueError(f"crop {crop} runs past the {W}x{H} frame: pixel crops are in this video's own pixels; "
                                 "give fractions 0..1 (the same box on two videos of one aspect)")
            x0, x1 = sorted((max(0, int(round(x0))), min(W, int(round(x1)))))
            y0, y1 = sorted((max(0, int(round(y0))), min(H, int(round(y1)))))
            img = img[y0:y1, x0:x1]
            if img.size == 0:
                raise ValueError(f"crop {crop} leaves no pixels (frame is {W}x{H}; give fractions 0..1 or pixel coordinates x0, y0, x1, y1)")
        h, w = img.shape[:2]
        tw = int(tile_width)
        th = max(1, int(round(h * tw / w)))
        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        label = f"#{i}  {i / f:.2f}s"
        cv2.rectangle(img, (0, 0), (10 + 11 * len(label), 26), (0, 0, 0), -1)
        cv2.putText(img, label, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img); shown.append(i)
    if cap is not None:
        cap.release()
    if not tiles:
        raise ValueError("no frame could be decoded")
    cols = int(max(1, columns))
    th, tw = tiles[0].shape[:2]
    rows = int(np.ceil(len(tiles) / cols))
    sheet = np.full((rows * (th + 4) + 4, cols * (tw + 4) + 4, 3), 24, np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        y, x = 4 + r * (th + 4), 4 + c * (tw + 4)
        sheet[y:y + th, x:x + tw] = t
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return {"sheet": str(out), "frames": shown, "times_s": [round(i / f, 3) for i in shown],
            "fps": f, "n_frames": int(n), "duration_s": round(n / f, 3), "grid": [rows, cols]}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=TOOL["summary"])
    ap.add_argument("video")
    ap.add_argument("out_path")
    ap.add_argument("--count", type=int, default=16)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float)
    ap.add_argument("--frames", type=int, nargs="*")
    ap.add_argument("--columns", type=int, default=4)
    ap.add_argument("--tile-width", type=int, default=480)
    ap.add_argument("--crop", type=float, nargs=4)
    ap.add_argument("--fps", type=float)
    a = ap.parse_args(argv)
    r = make_contact_sheet(a.video, a.out_path, a.count, a.start, a.end, a.frames or None, a.columns, a.tile_width, a.crop, a.fps)
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    sys.exit(main())
