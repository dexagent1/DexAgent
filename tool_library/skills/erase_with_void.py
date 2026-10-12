"""Remove the demonstrator and the manipulated objects from a video with VOID, keeping a clean background plate."""
from __future__ import annotations

TOOL = {
    "name": "erase_with_void",
    "title": "Erase with VOID",
    "type": "skill",
    "category": "Scene Runtime & Rendering",
    "summary": "Video inpainting of the erase masks (hands, arms, objects) with the VOID object-removal model, then a static clean plate: real pixels wherever the background is ever uncovered, VOID only where it never is.",
    "description": (
        "Photoreal robot data is the rendered robot composited onto the real background, so "
        "the human and whatever the simulation re-creates must leave the footage. Masks are "
        "dilated and scaled about their centroids so the faint outline an inpainter leaves "
        "falls inside them; VOID gets a quad-mask (remove / interaction ring where shadows live "
        "/ keep) and a background prompt and runs over windows of 85 frames. Its pixels are "
        "adopted inside the mask always and in the ring only where it changed them. For a static "
        "camera one plate is then assembled as the per-pixel median over uncovered frames, "
        "with frames whose white balance was still settling dropped. Needs the VOID checkout "
        "and the CogVideoX-Fun 5B weights (several minutes on a GPU)."
    ),
    "inputs": [
        {"name": "frames_dir", "type": "path", "description": "frame%05d.png/jpg to clean (sample across the clip, at most ~85)."},
        {"name": "masks_dir", "type": "path", "description": "Binary erase masks frame%05d.png (segment_with_text_prompts)."},
        {"name": "out_dir", "type": "path", "description": "Receives plate.png, clean/frame%05d.png, preview.mp4, meta.json."},
        {"name": "fps", "type": "float", "description": "Frame rate of the sampled frames (for VOID's temporal model)."},
        {"name": "background_prompt", "type": "str", "description": "What the uncovered background is, e.g. 'An empty table surface with nothing on it, static background.'"},
        {"name": "void_dir", "type": "path | None", "description": "VOID checkout; default third_party/void-model under the Stage 4 package or $VOID_DIR."},
        {"name": "params", "type": "dict", "description": "mask_dilate_px 11, mask_scale 1.15, interaction_px 61, ring_diff_threshold 18, color_drift_tol 4.0, gpu_memory_mode."},
        {"name": "static_camera", "type": "bool", "description": "Also build the single clean plate (True)."},
    ],
    "outputs": [
        {"name": "result", "type": "dict", "description": "plate path, clean frames dir, erase_coverage, never_uncovered fraction, n_frames."},
    ],
    "status": "tested",
    "entrypoint": "erase_with_void",
    "requirements": ["torch", "VOID (github.com/netflix/void-model) + alibaba-pai/CogVideoX-Fun-V1.5-5b-InP weights", "huggingface_hub", "opencv-python", "imageio"],
    "simulators": [],
    "tags": ["inpainting", "VOID", "clean plate", "video"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope",
                "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting",
                "Biology Experiment"],
}

import json
import os
import sys
from pathlib import Path

import numpy as np

from tool_library.common.stage4 import stage4_module
from tool_library.common.video import frame_files, write_json

DEFAULT_PARAMS = {"mask_dilate_px": 11, "mask_scale": 1.15, "interaction_px": 61, "ring_diff_threshold": 18,
                  "color_drift_tol": 4.0, "gpu_memory_mode": "model_full_load"}


def erase_with_void(frames_dir, masks_dir, out_dir, fps=10.0, background_prompt="An empty table surface with nothing on it, static background.",
                    void_dir=None, params=None, static_camera=True, log=print):
    import cv2
    if void_dir:
        os.environ["VOID_DIR"] = str(void_dir)
    plate_mod = stage4_module("plate")
    ip = {**DEFAULT_PARAMS, **(params or {}), "background_prompt": background_prompt}
    files = frame_files(frames_dir)
    assert files, f"no frames in {frames_dir}"
    frames = [cv2.imread(str(f)) for f in files]                     # BGR, as plate.py expects
    dil = np.ones((ip["mask_dilate_px"],) * 2, np.uint8)
    masks = []
    for k, f in enumerate(files):
        p = Path(masks_dir) / f"frame{k:05d}.png"
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        assert m is not None, f"missing erase mask {p}"
        if m.shape != frames[k].shape[:2]:
            m = cv2.resize(m, frames[k].shape[1::-1], interpolation=cv2.INTER_NEAREST)
        masks.append(plate_mod.enlarge(cv2.dilate((m > 0).astype(np.uint8) * 255, dil), ip["mask_scale"]))
    work = Path(out_dir)
    work.mkdir(parents=True, exist_ok=True)
    gen = plate_mod.run_void(work, frames, masks, max(1.0, float(fps)), ip, log)
    clean = plate_mod.paste(frames, masks, gen, ip)
    cdir = work / "clean"
    cdir.mkdir(exist_ok=True)
    for k, c in enumerate(clean):
        cv2.imwrite(str(cdir / f"frame{k:05d}.png"), cv2.cvtColor(c, cv2.COLOR_RGB2BGR))
    meta = {"n_frames": len(frames), "fps": float(fps), "clean_dir": str(cdir),
            "erase_coverage": round(float(np.mean([(m > 0).mean() for m in masks])), 4)}
    if static_camera:
        plate, never = plate_mod.static_plate(frames, masks, clean, ip, log)
        cv2.imwrite(str(work / "plate.png"), cv2.cvtColor(plate, cv2.COLOR_RGB2BGR))
        meta.update(plate=str(work / "plate.png"), never_uncovered=round(never, 4))
    from tool_library.common.video import write_video
    prev = []
    for f, m, c in zip(frames, masks, clean):
        ov = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).copy()
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, cnts, -1, (255, 60, 60), 2)
        prev.append(np.concatenate([ov, c], 1))
    meta["preview"] = write_video(prev, work / "preview.mp4", max(1.0, float(fps)))
    write_json(meta, work / "meta.json")
    return meta


if __name__ == "__main__":
    a = sys.argv[1:]
    print(json.dumps(erase_with_void(a[0], a[1], a[2]), indent=1))
