"""When does something move, and where: a motion-energy timeline from frame differencing."""
from __future__ import annotations

TOOL = {
    "name": "measure_motion_timeline",
    "title": "Measure motion timeline",
    "type": "skill",
    "category": "Measurements",
    "summary": "Frame-difference motion energy over time for the whole frame and for named regions (default: left and right halves), with the active intervals, the peaks and the frame where motion first and last occurs in each region.",
    "description": (
        "Decodes the clip at a stride, converts to grey, blurs, and takes the absolute difference between successive "
        "sampled frames. Motion energy is the fraction of pixels whose change exceeds `pixel_thresh`, per region. Regions "
        "are boxes as fractions of the frame (or pixels when any value > 1); the defaults are the left and right halves, "
        "which on an egocentric table camera separate the two hands' sides. A region is active when its energy is above "
        "`active_thresh`; runs shorter than `min_duration_s` are dropped and gaps shorter than `merge_gap_s` are merged. "
        "Returns per region the active intervals in seconds, the peak time, the first and last active time and the energy "
        "curve (downsampled), plus the whole-frame intervals. Use it to set subgoal time windows, to find when a hand "
        "starts and stops acting, to locate a sudden event (a drop, a snap) as an energy spike, and to confirm which side "
        "of the frame moved. It measures motion, not identity: anything moving in a region counts, hands and objects alike. "
        "On a head-mounted camera the whole frame moves with the head; then compare regions against each other or raise active_thresh."
    ),
    "inputs": [
        {"name": "video", "type": "path", "description": "A video file, or a folder of frames written by extract_frames."},
        {"name": "regions", "type": "dict[str, list[float]] | None", "description": "{name: [x0, y0, x1, y1]} as fractions or pixels. Default {'left': [0, 0, 0.5, 1], 'right': [0.5, 0, 1, 1]}."},
        {"name": "stride", "type": "int", "description": "Sample every k-th frame. Default 2."},
        {"name": "pixel_thresh", "type": "int", "description": "Grey-level change that counts as motion. Default 18."},
        {"name": "active_thresh", "type": "float", "description": "Energy (fraction of changed pixels) above which a region is active. Default 0.004."},
        {"name": "min_duration_s", "type": "float", "description": "Shortest interval kept. Default 0.3."},
        {"name": "merge_gap_s", "type": "float", "description": "Gaps shorter than this are merged. Default 0.5."},
        {"name": "fps", "type": "float | None", "description": "Frame rate for a frame folder without meta.json."},
        {"name": "out_json", "type": "path | None", "description": "Also write the result here."},
    ],
    "outputs": [
        {"name": "fps / n_frames / duration_s", "type": "float / int / float", "description": "Clip facts."},
        {"name": "regions", "type": "dict", "description": "Per region: intervals_s [[start, end], ...], first_active_s, last_active_s, peak_s, peak_energy, energy (downsampled list of [t, e])."},
        {"name": "whole_frame", "type": "dict", "description": "The same for the entire frame."},
        {"name": "events", "type": "list", "description": "Times of the strongest short energy spikes over the whole frame (largest rises between consecutive samples), up to 10."},
    ],
    "status": "tested",
    "entrypoint": "measure_motion_timeline",
    "requirements": ["opencv-python", "numpy"],
    "tags": ["motion", "timeline", "frame difference", "time windows", "hands", "semantic understanding"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope", "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting", "Biology Experiment"],
    "cli": "python -m tool_library.skills.measure_motion_timeline human_video.mp4 --out-json motion.json",
}

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from tool_library.common.video import frame_files, read_json, video_info

DEFAULT_REGIONS = {"left": [0.0, 0.0, 0.5, 1.0], "right": [0.5, 0.0, 1.0, 1.0]}


def _frames(video, stride, fps):
    p = Path(video)
    if p.is_dir():
        files = frame_files(p)
        if not files:
            raise FileNotFoundError(f"no frames in {p}")
        for i in range(0, len(files), max(1, stride)):
            yield i, cv2.imread(str(files[i]), cv2.IMREAD_GRAYSCALE)
        return
    cap = cv2.VideoCapture(str(p))
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % max(1, stride) == 0:
            yield i, cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        i += 1
    cap.release()


def _box(r, W, H):
    x0, y0, x1, y1 = [float(v) for v in r]
    if max(x0, y0, x1, y1) <= 1.0:
        x0, x1, y0, y1 = x0 * W, x1 * W, y0 * H, y1 * H
    x0, x1 = sorted((max(0, int(round(x0))), min(W, int(round(x1)))))
    y0, y1 = sorted((max(0, int(round(y0))), min(H, int(round(y1)))))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"region {r} is empty for a {W}x{H} frame")
    return x0, y0, x1, y1


def _intervals(t, e, thresh, min_dur, gap):
    active = e > thresh
    out, start = [], None
    for k in range(len(t)):
        if active[k] and start is None:
            start = t[k]
        if (not active[k] or k == len(t) - 1) and start is not None:
            end = t[k] if active[k] else t[k - 1]
            out.append([float(start), float(end)]); start = None
    merged = []
    for a, b in out:
        if merged and a - merged[-1][1] <= gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [[round(a, 2), round(b, 2)] for a, b in merged if b - a >= min_dur]


def measure_motion_timeline(video, regions=None, stride=2, pixel_thresh=18, active_thresh=0.004,
                            min_duration_s=0.3, merge_gap_s=0.5, fps=None, out_json=None):
    p = Path(video)
    if p.is_dir():
        meta = p / "meta.json"
        f = float(fps or (read_json(meta).get("fps") if meta.exists() else None) or 30.0)
        n = len(frame_files(p))
    else:
        info = video_info(p); f = float(info["fps"] or fps or 30.0); n = int(info["frames"])
    regions = dict(regions or DEFAULT_REGIONS)
    prev, times, energies, boxes = None, [], {k: [] for k in list(regions) + ["whole_frame"]}, None
    for i, g in _frames(video, int(stride), f):
        g = cv2.GaussianBlur(g, (5, 5), 0)
        if boxes is None:
            H, W = g.shape
            boxes = {k: _box(r, W, H) for k, r in regions.items()}
            boxes["whole_frame"] = (0, 0, W, H)
        if prev is not None:
            d = cv2.absdiff(g, prev) > int(pixel_thresh)
            times.append(i / f)
            for k, (x0, y0, x1, y1) in boxes.items():
                energies[k].append(float(d[y0:y1, x0:x1].mean()))
        prev = g
    if not times:
        raise ValueError(f"{video} has fewer than two sampled frames")
    t = np.asarray(times)
    result = {"fps": f, "n_frames": n, "duration_s": round(n / f, 3), "stride": int(stride), "regions": {}, "events": []}
    for k, vals in energies.items():
        e = np.asarray(vals)
        iv = _intervals(t, e, float(active_thresh), float(min_duration_s), float(merge_gap_s))
        act = t[e > active_thresh]
        step = max(1, len(t) // 200)
        rec = {"intervals_s": iv, "first_active_s": round(float(act[0]), 2) if len(act) else None,
               "last_active_s": round(float(act[-1]), 2) if len(act) else None,
               "peak_s": round(float(t[int(e.argmax())]), 2), "peak_energy": round(float(e.max()), 4),
               "mean_energy": round(float(e.mean()), 5),
               "energy": [[round(float(a), 2), round(float(b), 4)] for a, b in zip(t[::step], e[::step])]}
        if k == "whole_frame":
            result["whole_frame"] = rec
            rise = np.diff(e, prepend=e[0])
            top = np.argsort(-rise)[:10]
            result["events"] = [{"t": round(float(t[j]), 2), "rise": round(float(rise[j]), 4)} for j in sorted(top) if rise[j] > 2 * active_thresh]
        else:
            result["regions"][k] = rec
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(result, indent=1))
        result["out_json"] = str(out_json)
    return result


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=TOOL["summary"])
    ap.add_argument("video")
    ap.add_argument("--regions", help="JSON {name: [x0, y0, x1, y1]}")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--pixel-thresh", type=int, default=18)
    ap.add_argument("--active-thresh", type=float, default=0.004)
    ap.add_argument("--out-json")
    a = ap.parse_args(argv)
    r = measure_motion_timeline(a.video, json.loads(a.regions) if a.regions else None, a.stride, a.pixel_thresh, a.active_thresh, out_json=a.out_json)
    slim = {k: v for k, v in r.items()}
    for k in slim.get("regions", {}):
        slim["regions"][k] = {kk: vv for kk, vv in slim["regions"][k].items() if kk != "energy"}
    slim["whole_frame"] = {kk: vv for kk, vv in slim["whole_frame"].items() if kk != "energy"}
    print(json.dumps(slim, indent=1))


if __name__ == "__main__":
    sys.exit(main())
