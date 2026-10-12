"""CPU segmentation computed directly from a source video's decoded pixels."""
from __future__ import annotations

TOOL = {
    "name": "segment_source_frame", "title": "Segment a source video frame on CPU",
    "type": "skill", "category": "Perception & Grounding", "status": "tested",
    "summary": "Fixed OpenCV GrabCut inside a selected object's box on a decoded video frame; CPU fallback for source masks.",
    "description": "Decodes the video frame itself, uses the supplied pixel box as GrabCut initialization, and writes a binary mask and lossless source frame. No proposed mask, synthetic render, or processor can be supplied. Inspect the result; source provenance does not prove correct object identity or segmentation accuracy. Invoke directly through the harness for reference evidence, not inside run_python. Existing masks can be copied without changing their bytes. Multiple separate visible instances can use separate calls; do not union or paint accepted masks after the call.",
    "inputs": [
        {"name": "video", "type": "path", "description": "Original source video, matching the run input."},
        {"name": "frame_index", "type": "int", "description": "Zero-based source frame index."},
        {"name": "box", "type": "list[int]", "description": "Pixel box [x0, y0, x1, y1], with background outside it."},
        {"name": "out_dir", "type": "path", "description": "Receives source.png, mask.png and recipe.json."},
    ],
    "outputs": [{"name": "result", "type": "dict", "description": "mask, source_frame, frame_index, box, shape, foreground_pixels; raises on empty or degenerate segmentation."}],
    "entrypoint": "segment_source_frame", "requirements": ["opencv-python", "numpy"],
    "tags": ["source", "mask", "segmentation", "CPU", "GrabCut", "provenance"],
}

import json
from pathlib import Path

import cv2
import numpy as np


def segment_source_frame(video, frame_index, box, out_dir):
    if isinstance(frame_index, bool) or int(frame_index) != frame_index or int(frame_index) < 0:
        raise ValueError("frame_index must be a nonnegative integer")
    cap = cv2.VideoCapture(str(video))
    try:
        # Sequential decoding gives the same pixels as extract_frames, including
        # videos whose random seek lands at an imprecise keyframe.
        frame = None
        for _ in range(int(frame_index) + 1):
            ok, frame = cap.read()
            if not ok:
                raise ValueError("source frame cannot be decoded")
    finally:
        cap.release()
    h, w = frame.shape[:2]
    if not isinstance(box, (list, tuple)) or len(box) != 4 or any(isinstance(v, bool) or not np.isfinite(v) or int(v) != v for v in box):
        raise ValueError("box must contain four integer pixel coordinates")
    x0, y0, x1, y1 = map(int, box)
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h) or (x1 - x0) * (y1 - y0) >= w * h:
        raise ValueError("box must be inside the source image and leave background pixels")
    mask = np.zeros((h, w), np.uint8)
    bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.setRNGSeed(0)
    cv2.grabCut(frame, mask, (x0, y0, x1 - x0, y1 - y0), bg, fg, 5, cv2.GC_INIT_WITH_RECT)
    binary = np.isin(mask, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8) * 255
    count = int(np.count_nonzero(binary))
    if count == 0 or count == h * w:
        raise ValueError("source segmentation is empty or covers the entire image; inspect the source and revise the box")
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name, pixels in (("source.png", frame), ("mask.png", binary)):
        if not cv2.imwrite(str(out / name), pixels):
            raise OSError(f"cannot write {name}")
    result = {"mask": str(out / "mask.png"), "source_frame": str(out / "source.png"),
              "frame_index": int(frame_index), "box": [x0, y0, x1, y1], "shape": [h, w], "foreground_pixels": count}
    (out / "recipe.json").write_text(json.dumps(result, indent=2))
    return result
