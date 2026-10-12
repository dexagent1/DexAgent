"""Extract the frames of a video to numbered image files."""
from __future__ import annotations

TOOL = {
    "name": "extract_frames",
    "title": "Extract frames",
    "type": "skill",
    "category": "Perception & Grounding",
    "summary": "Decode a video into frame%05d.jpg files, optionally at a stride, with a meta.json recording the mapping.",
    "description": (
        "Every perception tool in this library reads frames from a folder rather than from "
        "the video, so that detection, tracking, depth and hand recovery all see exactly the "
        "same pixels and agree on frame numbers. This writes that folder. A stride keeps every "
        "n-th frame (the recorded effective frame rate goes into meta.json), and a frame range "
        "cuts the clip. Frames are numbered from 0 in output order."
    ),
    "inputs": [
        {"name": "video", "type": "path", "description": "The source video."},
        {"name": "out_dir", "type": "path", "description": "Folder that receives frame%05d.jpg and meta.json."},
        {"name": "stride", "type": "int", "description": "Keep every n-th frame. Default 1."},
        {"name": "start, stop", "type": "int | None", "description": "Source frame range [start, stop)."},
        {"name": "max_frames", "type": "int | None", "description": "Stop after this many output frames."},
        {"name": "quality", "type": "int", "description": "JPEG quality. Default 95."},
    ],
    "outputs": [
        {"name": "meta", "type": "dict",
         "description": "frames_dir, n_frames, source fps, effective fps, stride, start, and the list of source frame indices kept."},
    ],
    "status": "tested",
    "entrypoint": "extract_frames",
    "requirements": ["opencv-python", "numpy"],
    "tags": ["video", "frames"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope", "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting", "Biology Experiment"],
    "cli": "python -m tool_library.skills.extract_frames video.mp4 frames/ --stride 2",
}

import json
import sys
from pathlib import Path

import cv2

from tool_library.common.video import video_info, write_json


def extract_frames(video, out_dir, stride=1, start=0, stop=None, max_frames=None, quality=95):
    info = video_info(video)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    kept, t, i = [], 0, 0
    while True:
        ok, f = cap.read()
        if not ok or (stop is not None and t >= stop):
            break
        if t >= start and (t - start) % max(1, int(stride)) == 0:
            cv2.imwrite(str(out / f"frame{i:05d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
            kept.append(t)
            i += 1
            if max_frames and i >= max_frames:
                break
        t += 1
    cap.release()
    meta = {"frames_dir": str(out), "n_frames": len(kept), "source_video": str(video),
            "source_fps": info["fps"], "fps": info["fps"] / max(1, int(stride)),
            "stride": int(stride), "start": int(start), "width": info["width"],
            "height": info["height"], "source_frames": kept}
    write_json(meta, out / "meta.json")
    return meta


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=TOOL["summary"])
    ap.add_argument("video")
    ap.add_argument("out_dir")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int)
    ap.add_argument("--max-frames", type=int)
    a = ap.parse_args(argv)
    m = extract_frames(a.video, a.out_dir, a.stride, a.start, a.stop, a.max_frames)
    print(json.dumps({k: v for k, v in m.items() if k != "source_frames"}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
