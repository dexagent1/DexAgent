"""Probe a video: resolution, frame rate, length, and a few thumbnails."""
from __future__ import annotations

TOOL = {
    "name": "probe_video",
    "title": "Probe video",
    "type": "skill",
    "category": "Perception & Grounding",
    "summary": "Resolution, frame rate, frame count and duration of a video, with optional thumbnails at chosen times.",
    "description": (
        "Reads the container metadata of a video and, when asked, decodes a few frames "
        "spread over the clip and writes them as JPEGs so a person or a vision model can "
        "look at the footage before any heavier step runs. It is the first call of every "
        "pipeline: the frame rate and size it returns decide how frames are extracted and "
        "how far the other tools can trust their timing."
    ),
    "inputs": [
        {"name": "video", "type": "path", "description": "Any video OpenCV can open (mp4, mov, avi)."},
        {"name": "thumbnails", "type": "int", "description": "How many evenly spaced thumbnails to write; 0 for none. Default 0."},
        {"name": "out_dir", "type": "path | None", "description": "Where thumbnails go; required when thumbnails > 0."},
    ],
    "outputs": [
        {"name": "info", "type": "dict",
         "description": "width, height, fps, frames, duration_s, and thumbnails: list of {frame, path}."},
    ],
    "status": "tested",
    "entrypoint": "probe_video",
    "requirements": ["opencv-python", "numpy"],
    "tags": ["video", "metadata"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope", "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting", "Biology Experiment"],
    "cli": "python -m tool_library.skills.probe_video video.mp4 --thumbnails 4 --out-dir probe/",
}

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from tool_library.common.video import video_info


def probe_video(video, thumbnails=0, out_dir=None):
    info = video_info(video)
    info["thumbnails"] = []
    if thumbnails and info["frames"] > 0:
        assert out_dir, "out_dir is needed to write thumbnails"
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video))
        for t in np.linspace(0, info["frames"] - 1, int(thumbnails)).round().astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
            ok, f = cap.read()
            if not ok:
                continue
            p = out / f"thumb{int(t):05d}.jpg"
            cv2.imwrite(str(p), f, [cv2.IMWRITE_JPEG_QUALITY, 90])
            info["thumbnails"].append({"frame": int(t), "path": str(p)})
        cap.release()
    return info


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=TOOL["summary"])
    ap.add_argument("video")
    ap.add_argument("--thumbnails", type=int, default=0)
    ap.add_argument("--out-dir")
    a = ap.parse_args(argv)
    print(json.dumps(probe_video(a.video, a.thumbnails, a.out_dir), indent=1))


if __name__ == "__main__":
    sys.exit(main())
