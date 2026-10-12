"""Metric per-frame depth with Video Depth Anything."""
from __future__ import annotations

TOOL = {
    "name": "estimate_depth",
    "title": "Estimate depth",
    "type": "skill",
    "category": "Perception & Grounding",
    "summary": "Metric depth for every frame with Video Depth Anything (metric ViT-L), saved as float16 metres plus an inverse-depth preview video.",
    "description": (
        "Runs the metric Video Depth Anything model over the whole frame folder in one pass, so "
        "the depth is temporally consistent, and writes depths.npz with a (N, H, W) array in "
        "metres. Metric depth is what puts the reconstructed scene in metres: object placement, "
        "scale and the table plane are read from it. The preview video shows inverse depth "
        "(bright = near), the signal the pose fitter consumes. The model repository and its "
        "checkpoint are passed in; nothing is downloaded silently."
    ),
    "inputs": [
        {"name": "frames_dir", "type": "path", "description": "Folder of frame%05d.jpg."},
        {"name": "out_npz", "type": "path", "description": "Output .npz with keys depths (N,H,W float16 m) and fps."},
        {"name": "vda_repo", "type": "path", "description": "Clone of DepthAnything/Video-Depth-Anything (put on sys.path)."},
        {"name": "checkpoint", "type": "path", "description": "metric_video_depth_anything_vitl.pth."},
        {"name": "fps", "type": "float", "description": "Frame rate the frames were extracted at. Default 5."},
        {"name": "input_size", "type": "int", "description": "Network input size. Default 518."},
        {"name": "max_frames", "type": "int | None", "description": "Limit the frames processed."},
        {"name": "preview_mp4", "type": "path | None", "description": "Optional inverse-depth preview video."},
    ],
    "outputs": [
        {"name": "summary", "type": "dict", "description": "out_npz, n_frames, depth range and median in metres."},
    ],
    "status": "tested",
    "entrypoint": "estimate_depth",
    "requirements": ["torch", "Video-Depth-Anything repository (github.com/DepthAnything/Video-Depth-Anything)",
                     "metric_video_depth_anything_vitl.pth", "xformers", "easydict"],
    "tags": ["depth", "metric", "Video Depth Anything"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope", "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting", "Biology Experiment"],
    "cli": "python -m tool_library.skills.estimate_depth frames/ depth.npz --vda-repo Video-Depth-Anything --checkpoint metric_video_depth_anything_vitl.pth",
}

import json
import sys
from pathlib import Path

import numpy as np

from tool_library.common.video import frame_files, read_frame, write_video


def estimate_depth(frames_dir, out_npz, vda_repo, checkpoint, fps=5.0, input_size=518,
                   max_frames=None, preview_mp4=None):
    try:
        import torch
        sys.path.insert(0, str(vda_repo))
        from video_depth_anything.video_depth import VideoDepthAnything
    except ImportError as e:
        raise ImportError("Video Depth Anything is required: clone "
                          "https://github.com/DepthAnything/Video-Depth-Anything and pass vda_repo") from e
    files = frame_files(frames_dir)
    if max_frames:
        files = files[: int(max_frames)]
    assert files, f"no frames in {frames_dir}"
    frames = np.stack([read_frame(f) for f in files])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VideoDepthAnything(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024], metric=True)
    model.load_state_dict(torch.load(str(checkpoint), map_location="cpu"), strict=True)
    model = model.to(device).eval()
    depths, _ = model.infer_video_depth(frames, float(fps), input_size=int(input_size), device=device)
    depths = np.asarray(depths, np.float32)
    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, depths=depths.astype(np.float16), fps=float(fps))
    if preview_mp4:
        inv = 1.0 / np.clip(depths, 0.05, 50.0)
        lo, hi = np.percentile(inv, 2), np.percentile(inv, 98)
        vis = np.clip((inv - lo) / max(hi - lo, 1e-6), 0, 1)
        write_video((np.stack([(v * 255).astype(np.uint8)] * 3, -1) for v in vis), preview_mp4, fps)
    return {"out_npz": str(out_npz), "n_frames": int(len(depths)), "min_m": float(depths.min()),
            "max_m": float(depths.max()), "median_m": float(np.median(depths))}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=TOOL["summary"])
    ap.add_argument("frames_dir")
    ap.add_argument("out_npz")
    ap.add_argument("--vda-repo", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--preview")
    a = ap.parse_args(argv)
    print(json.dumps(estimate_depth(a.frames_dir, a.out_npz, a.vda_repo, a.checkpoint, a.fps,
                                    max_frames=a.max_frames, preview_mp4=a.preview), indent=1))


if __name__ == "__main__":
    sys.exit(main())
