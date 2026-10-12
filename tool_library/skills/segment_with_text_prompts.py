"""Per-frame binary masks of everything matching a list of text prompts (SAM 3)."""
from __future__ import annotations

TOOL = {
    "name": "segment_with_text_prompts",
    "title": "Segment with text prompts",
    "type": "skill",
    "category": "Perception & Grounding",
    "summary": "SAM 3 on every frame: the union of all instances of the given prompts as one binary mask per frame, and optionally one mask per instance.",
    "description": (
        "For each frame, each prompt is segmented with SAM 3 and all instances above the score "
        "floor are merged into a single mask (255 = matched). This is the mask a video eraser "
        "needs: name the hands and arms and every object the simulation re-creates, and the "
        "result marks exactly the pixels to remove. Frames can be downscaled for speed; masks "
        "are returned at full resolution. With `instances` each instance above the floor is also "
        "written as its own mask (instances/<prompt>/frame%05d_<k>.png, k by descending score) with "
        "its score, box and area in meta.json: several same-coloured parts (identical bricks) get "
        "one mask each without a detect-then-track detour. Instance numbers are per frame (ranked "
        "by score), not identities tracked through the video: track_objects does that."
    ),
    "inputs": [
        {"name": "frames_dir", "type": "path", "description": "Folder of frame%05d.jpg or .png."},
        {"name": "prompts", "type": "list[str]", "description": "What to segment, e.g. ['hand', 'arm', 'cup']."},
        {"name": "out_dir", "type": "path", "description": "Receives frame%05d.png masks and meta.json."},
        {"name": "min_score", "type": "float", "description": "Instance confidence floor. Default 0.3."},
        {"name": "downscale", "type": "float", "description": "Detect at this fraction of the resolution. Default 1.0."},
        {"name": "instances", "type": "bool", "description": "Also write one mask per instance (instances/<prompt slug>/frame%05d_<k>.png) and list them in meta.json. Default false (union masks only)."},
        {"name": "version", "type": "str", "description": "Image checkpoint version: 'sam3' (default). SAM 3.1 multiplex video checkpoints are not supported by this image tool."},
    ],
    "outputs": [
        {"name": "meta", "type": "dict", "description": "prompts, n_frames, mean coverage, out_dir, model_identity with actual checkpoint SHA256/runtime; with instances: instances {prompt: [[{mask, score, box [x0, y0, x1, y1] px, area_px}] per frame]}."},
    ],
    "status": "tested",
    "entrypoint": "segment_with_text_prompts",
    "requirements": ["sam3 (pip install git+https://github.com/facebookresearch/sam3.git)", "torch",
                     "Hugging Face access to facebook/sam3 (or its already cached checkpoint)"],
    "tags": ["SAM 3", "segmentation", "masks", "erase"],
    "used_in": ["Giftbox Packing", "Knot the Rope", "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting", "Biology Experiment"],
    "cli": "python -m tool_library.skills.segment_with_text_prompts frames/ masks/ --prompts hand arm cup",
}

import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

from tool_library.common.video import frame_files, write_json
from tool_library.skills.detect_objects import load_sam3


def _np(x):
    """A torch tensor or array as a float/bool numpy array."""
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy() if str(getattr(x, "dtype", "")).startswith("torch.bfloat") else x.detach().cpu().numpy()
    return np.asarray(x)


def _slug(p):
    return re.sub(r"[^a-z0-9]+", "_", p.lower()).strip("_") or "prompt"


def segment_with_text_prompts(frames_dir, prompts, out_dir, min_score=0.3, downscale=1.0,
                              processor=None, instances=False, version="sam3"):
    from PIL import Image
    files = frame_files(frames_dir)
    assert files, f"no frames in {frames_dir}"
    prompts = [p.strip() for p in prompts if p.strip()]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    proc = processor or load_sam3(min_score, version)
    cover = []
    inst = {p: [] for p in prompts} if instances else None
    if instances:
        slugs = {}
        for p in prompts:                               # distinct folders even for prompts that slug alike
            sl = _slug(p); k = 2
            while sl in slugs.values():
                sl = f"{_slug(p)}_{k}"; k += 1
            slugs[p] = sl
            (out / "instances" / sl).mkdir(parents=True, exist_ok=True)
    for t, path in enumerate(files):
        img = Image.open(path).convert("RGB")
        W0, H0 = img.size
        if downscale < 1.0:
            img = img.resize((int(W0 * downscale), int(H0 * downscale)), Image.BILINEAR)
        state = proc.set_image(img)
        union = np.zeros((img.height, img.width), bool)
        for p in prompts:
            res = proc.set_text_prompt(prompt=p, state=state)
            m = res.get("masks")
            if instances:
                inst[p].append([])
            if m is None or m.shape[0] == 0:
                continue
            if instances:                               # every instance as its own mask, best score first
                M = _np(m).astype(bool).reshape(m.shape[0], -1, *m.shape[-2:]).any(axis=1)
                sc = _np(res["scores"]).astype(float).reshape(-1) if res.get("scores") is not None else np.ones(len(M))
                order = np.argsort(-sc)
                for k, i in enumerate(order):
                    mi = M[i]
                    if mi.shape != (H0, W0):
                        mi = cv2.resize(mi.astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST) > 0
                    ys, xs = np.nonzero(mi)
                    path = out / "instances" / slugs[p] / f"frame{t:05d}_{k}.png"
                    cv2.imwrite(str(path), mi.astype(np.uint8) * 255)
                    inst[p][-1].append({"mask": str(path), "score": round(float(sc[i]), 4), "area_px": int(mi.sum()),
                                        "box": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None})
            m = m.any(dim=0).squeeze().detach().cpu().numpy().astype(bool) if hasattr(m, "any") and hasattr(m, "detach") \
                else _np(m).astype(bool).any(axis=0).squeeze()
            if m.shape != union.shape:
                m = cv2.resize(m.astype(np.uint8), union.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
            union |= m
        if union.shape != (H0, W0):
            union = cv2.resize(union.astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST) > 0
        cv2.imwrite(str(out / f"frame{t:05d}.png"), union.astype(np.uint8) * 255)
        cover.append(float(union.mean()))
    meta = {"prompts": prompts, "n_frames": len(files), "min_score": min_score,
            "mean_coverage": round(float(np.mean(cover)), 4), "out_dir": str(out),
            "model_identity": getattr(proc, "model_identity", None)}
    if instances:
        meta["instances"] = inst
        meta["n_instances"] = {p: [len(f) for f in inst[p]] for p in prompts}
    write_json(meta, out / "meta.json")
    return meta


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=TOOL["summary"])
    ap.add_argument("frames_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--min-score", type=float, default=0.3)
    ap.add_argument("--downscale", type=float, default=1.0)
    ap.add_argument("--instances", action="store_true")
    ap.add_argument("--version", default="sam3", choices=["sam3"])
    a = ap.parse_args(argv)
    print(json.dumps(segment_with_text_prompts(a.frames_dir, a.prompts, a.out_dir, a.min_score, a.downscale, instances=a.instances, version=a.version), indent=1))


if __name__ == "__main__":
    sys.exit(main())
