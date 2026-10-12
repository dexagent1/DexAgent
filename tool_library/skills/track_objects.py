"""Track detected objects through a clip with SAM 2, forward and backward."""
from __future__ import annotations

TOOL = {
    "name": "track_objects",
    "title": "Track objects",
    "type": "skill",
    "category": "Perception & Grounding",
    "summary": "SAM 2 video tracking of every detected object from its best seed frame, forward and backward, into per-frame label maps.",
    "description": (
        "Takes a detection scan, picks the seed frame (the sampled "
        "frame that found the most keywords, ties broken by the highest minimum score), seeds "
        "SAM 2's video predictor with each keyword's box there, and propagates through the clip "
        "in both directions, one object at a time so tracks never swap identities. Writes a "
        "palette label map per frame (pixel value = object label, 1..K in keyword order), a "
        "binary mask video per object, an overlay video, and result.json with per-object "
        "coverage. Objects missing at the seed frame are reported, not silently dropped."
    ),
    "inputs": [
        {"name": "frames_dir", "type": "path", "description": "Folder of frame%05d.jpg."},
        {"name": "detect_json", "type": "path", "description": "Output of detect_objects."},
        {"name": "out_dir", "type": "path", "description": "Receives pred/frame%05d.png label maps, videos and result.json."},
        {"name": "checkpoint", "type": "path | None", "description": "SAM 2.1 checkpoint (.pt). None: facebook/sam2.1-hiera-large from Hugging Face."},
        {"name": "model_cfg", "type": "str", "description": "SAM 2 config name. Default configs/sam2.1/sam2.1_hiera_l.yaml."},
        {"name": "fps", "type": "float", "description": "Frame rate of the debug videos. Default 10."},
    ],
    "outputs": [
        {"name": "result", "type": "dict",
         "description": "n_frames, seed_frame, objects: [{prompt, label, score, nonempty_frames}], masks_dir, overlay video path."},
    ],
    "status": "tested",
    "entrypoint": "track_objects",
    "requirements": ["sam2 (pip install git+https://github.com/facebookresearch/sam2.git)", "torch", "SAM 2.1 checkpoint"],
    "tags": ["SAM 2", "tracking", "masks"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope", "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting", "Biology Experiment"],
    "cli": "python -m tool_library.skills.track_objects frames/ scan.json track/ --checkpoint sam2.1_hiera_large.pt",
}

import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from tool_library.common.video import (frame_files, overlay_labels, read_frame,
                                       read_json, save_label_map, slug, write_json, write_video)


def load_sam2_video_predictor(checkpoint=None, model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml"):
    try:
        import torch
        from sam2.build_sam import build_sam2_video_predictor
        from sam2.sam2_video_predictor import SAM2VideoPredictor
    except ImportError as e:
        raise ImportError("SAM 2 is required: pip install git+https://github.com/facebookresearch/sam2.git") from e
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if checkpoint:
        return build_sam2_video_predictor(model_cfg, str(checkpoint), device=device)
    return SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-large", device=device)


def pick_seed_frame(scan):
    """The seed frame: most keywords found, then the highest minimum score."""
    det = {int(k): v for k, v in scan["detections"].items() if v}
    assert det, "the scan found nothing on any frame"
    fi, rec = max(det.items(), key=lambda kv: (len(kv[1]), min(r["score"] for r in kv[1].values())))
    return fi, rec


def track_objects(frames_dir, detect_json, out_dir, checkpoint=None,
                  model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml", fps=10.0, predictor=None):
    import torch
    files = frame_files(frames_dir)
    n = len(files)
    scan = read_json(detect_json)
    seed, rec = pick_seed_frame(scan)
    prompts = [p for p in scan["prompts"] if p in rec]
    missing = [p for p in scan["prompts"] if p not in rec]
    out = Path(out_dir)
    (out / "pred").mkdir(parents=True, exist_ok=True)
    pred = predictor or load_sam2_video_predictor(checkpoint, model_cfg)
    # SAM 2 wants a folder of integer-named JPEGs: link the frames under such names
    tmp = tempfile.mkdtemp(prefix="sam2_frames_", dir=os.environ.get("TMPDIR"))
    for i, f in enumerate(files):
        os.symlink(os.path.abspath(f), os.path.join(tmp, f"{i:05d}.jpg"))
    H, W = read_frame(files[0]).shape[:2]
    tracks = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        state = pred.init_state(video_path=tmp, offload_video_to_cpu=True, offload_state_to_cpu=True)
        for k, p in enumerate(prompts):
            seq = np.zeros((n, H, W), np.uint8)
            for reverse in (False, True):
                if reverse and seed == 0:
                    continue
                pred.reset_state(state)
                pred.add_new_points_or_box(state, frame_idx=seed, obj_id=1,
                                           box=np.asarray(rec[p]["box"], np.float32))
                for fi, _ids, logits in pred.propagate_in_video(state, start_frame_idx=seed, reverse=reverse):
                    m = (logits[0, 0] > 0).cpu().numpy().astype(np.uint8)
                    if m.shape != (H, W):
                        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                    seq[fi] = m
            tracks.append(seq)
    for f in os.listdir(tmp):
        os.unlink(os.path.join(tmp, f))
    os.rmdir(tmp)

    result = {"n_frames": n, "seed_frame": int(seed), "objects": [], "missing_at_seed": missing,
              "masks_dir": str(out / "pred")}
    for k, (p, seq) in enumerate(zip(prompts, tracks)):
        write_video((np.stack([m * 255] * 3, -1) for m in seq), out / f"obj{k + 1}_{slug(p)}_binary.mp4", fps)
        result["objects"].append({"prompt": p, "label": k + 1, "score": rec[p]["score"],
                                  "nonempty_frames": int(sum(m.any() for m in seq))})
    overlay = []
    for i in range(n):
        comb = np.zeros((H, W), np.uint8)
        for k in sorted(range(len(prompts)), key=lambda k: -int(tracks[k][i].sum())):
            comb[tracks[k][i] > 0] = k + 1          # larger masks first, smaller on top
        save_label_map(comb, out / "pred" / f"frame{i:05d}.png", len(prompts))
        overlay.append(overlay_labels(read_frame(files[i]), comb, prompts))
    result["overlay_video"] = write_video(overlay, out / "masks_overlay.mp4", fps)
    write_json(result, out / "result.json")
    return result


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=TOOL["summary"])
    ap.add_argument("frames_dir")
    ap.add_argument("detect_json")
    ap.add_argument("out_dir")
    ap.add_argument("--checkpoint")
    ap.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--fps", type=float, default=10.0)
    a = ap.parse_args(argv)
    r = track_objects(a.frames_dir, a.detect_json, a.out_dir, a.checkpoint, a.model_cfg, a.fps)
    print(json.dumps({k: v for k, v in r.items()}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
