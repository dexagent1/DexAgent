"""Open-vocabulary object detection on sampled frames with SAM 3."""
from __future__ import annotations

TOOL = {
    "name": "detect_objects",
    "title": "Detect objects",
    "type": "skill",
    "category": "Perception & Grounding",
    "summary": "SAM 3 text-prompted detection on a few frames: the best box per keyword and frame, plus every instance found.",
    "description": (
        "Prompts SAM 3 with each keyword on a sparse sample of frames and records, per frame "
        "and keyword, the highest-scoring box and the full list of instances with scores. Two "
        "uses: it validates that the names chosen for the task objects are findable at all, and "
        "it yields the seed boxes that tracking starts from. The number of confident instances "
        "per frame is a count instrument (four 'drawer' boxes is a four-drawer cabinet). Frames "
        "should be spread over the clip, because a part that opens is only visible while it is "
        "open. Without frame_indices it scans a fifth of the frames in frames_dir and never fewer "
        "than 8 (all of them when the folder holds 8 or fewer), evenly spaced from the first frame "
        "to the last. The scan "
        "records which frames it read (frame_indices, frame_selection)."
    ),
    "inputs": [
        {"name": "frames_dir", "type": "path", "description": "Folder of frame%05d.jpg."},
        {"name": "keywords", "type": "list[str]", "description": "Object names to detect, e.g. ['cup', 'bowl']."},
        {"name": "out_json", "type": "path", "description": "Where the scan is written."},
        {"name": "frame_indices", "type": "list[int] | None", "description": "Positions (0-based, in sorted file order) of the frames in frames_dir to scan. Default None: 20% of the frames but at least 8 (or all when there are fewer), evenly spaced including the first and the last."},
        {"name": "min_score", "type": "float", "description": "Detection confidence floor. Default 0.1."},
        {"name": "version", "type": "str", "description": "Image checkpoint version: 'sam3' (default). SAM 3.1 multiplex video checkpoints are not compatible with this image tool."},
    ],
    "outputs": [
        {"name": "scan", "type": "dict",
         "description": "prompts, frame_indices (the frames scanned), frame_selection ('given' or the default rule), n_frames_in_dir, detections[frame][keyword] = {box [x0,y0,x1,y1], score, n_boxes, all_boxes, all_scores}; summary with frames_with_hit and best_score_per_keyword; model_identity records the actual checkpoint SHA256 and runtime."},
    ],
    "status": "tested",
    "entrypoint": "detect_objects",
    "requirements": ["sam3 (pip install git+https://github.com/facebookresearch/sam3.git)", "torch",
                     "Hugging Face access to facebook/sam3 (or its already cached checkpoint)"],
    "tags": ["SAM 3", "detection", "open vocabulary"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope", "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting", "Biology Experiment"],
    "cli": "python -m tool_library.skills.detect_objects frames/ scan.json --keywords cup bowl",
}

import json
from contextlib import nullcontext
import importlib.metadata
import sys
from pathlib import Path

import numpy as np

from tool_library.common.video import frame_files, write_json


class _InferenceProcessor:
    """Keep inference/autocast scoped to each call, including on exceptions."""
    def __init__(self, processor, torch, device, identity):
        self.processor, self.torch, self.device = processor, torch, device
        self.model_identity = identity

    def _call(self, name, *args, **kwargs):
        torch = self.torch
        precision = (torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
                     if self.device == "cuda" else nullcontext())
        with torch.inference_mode(), precision:
            return getattr(self.processor, name)(*args, **kwargs)

    def set_image(self, *args, **kwargs):
        return self._call("set_image", *args, **kwargs)

    def set_text_prompt(self, *args, **kwargs):
        return self._call("set_text_prompt", *args, **kwargs)


def load_sam3(min_score=0.1, version="sam3"):
    """Load the image checkpoint strictly; never use a partial/random model."""
    if version != "sam3":
        raise ValueError("SAM image detection/segmentation requires version='sam3'; "
                         "sam3.1 is a multiplex video checkpoint, not this image model")
    try:
        import torch
        from huggingface_hub import try_to_load_from_cache
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf
    except ImportError as e:
        raise ImportError("SAM 3 image dependencies are missing in this interpreter. Configure environments.sam3 "
                          "and route detect_objects / segment_with_text_prompts to it; install sam3 and torch there "
                          "(https://github.com/facebookresearch/sam3) and provide facebook/sam3 checkpoint access") from e
    checkpoint = try_to_load_from_cache("facebook/sam3", "sam3.pt")
    if not isinstance(checkpoint, str) or not Path(checkpoint).is_file():
        checkpoint = download_ckpt_from_hf(version="sam3")
    from tool_library.common.evidence import file_digest
    identity = {"version": "sam3", "repo_id": "facebook/sam3", "checkpoint_sha256": file_digest(checkpoint),
                "sam3_package_version": importlib.metadata.version("sam3"), "torch_version": torch.__version__}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    identity["device"] = device
    # Upstream's helper currently loads strict=False and prints missing keys.
    # Build without weights and require the complete detector state instead.
    model = build_sam3_image_model(checkpoint_path=None, load_from_HF=False, device=device)
    checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    checkpoint_state = checkpoint_state.get("model", checkpoint_state)
    image_state = {key.removeprefix("detector."): value for key, value in checkpoint_state.items()
                   if key.startswith("detector.")}
    if not image_state:
        raise ValueError("SAM 3 checkpoint has no detector weights; refusing an uninitialized image model")
    expected = set(model.state_dict())
    extra = set(image_state) - expected
    # The shared release also carries the optional SAM2 interactive tracker's
    # neck. This tool constructs no interactive tracker; its unused neck is the
    # only permitted extra state, while every image-model weight is mandatory.
    unsupported = [key for key in extra if not key.startswith("backbone.vision_backbone.sam2_convs.")]
    if unsupported:
        raise ValueError(f"SAM 3 checkpoint has unsupported detector weights: {sorted(unsupported)[:8]}")
    image_state = {key: value for key, value in image_state.items() if key in expected}
    model.load_state_dict(image_state, strict=True)
    model.eval()
    return _InferenceProcessor(Sam3Processor(model, device=device, confidence_threshold=min_score), torch, device, identity)


MIN_FRAMES = 8       # a part that is visible only while open needs more than one look, whatever the folder's size


def sample_indices(n_total, frac=0.2, min_frames=MIN_FRAMES):
    """frac of the frames, at least min_frames (all of them when there are fewer), evenly spaced from first to last."""
    n_total = int(n_total)
    if n_total <= 0:
        return []
    n = min(n_total, max(int(min_frames), int(round(frac * n_total))))
    return sorted({int(round(v)) for v in np.linspace(0, n_total - 1, n)})


def detect_objects(frames_dir, keywords, out_json, frame_indices=None, min_score=0.1,
                   version="sam3", processor=None):
    from PIL import Image
    files = frame_files(frames_dir)
    assert files, f"no frames in {frames_dir}"
    keywords = [k.strip() for k in keywords if k.strip()]
    if frame_indices:
        idx = sorted({int(i) for i in frame_indices if 0 <= int(i) < len(files)})
        if not idx:
            raise ValueError(f"frame_indices {list(frame_indices)[:8]} are all outside the {len(files)} frames of {frames_dir}")
        selection = "given"
    else:
        idx = sample_indices(len(files))
        selection = f"default: {len(idx)} of {len(files)} frames evenly spaced (20%, at least {MIN_FRAMES})"
    proc = processor or load_sam3(min_score, version)
    dets = {}
    for fi in idx:
        state = proc.set_image(Image.open(files[fi]).convert("RGB"))
        rec = {}
        for p in keywords:
            o = proc.set_text_prompt(prompt=p, state=state)
            scores = np.asarray(o["scores"].detach().float().cpu().numpy(), float)
            if scores.size == 0:
                continue
            boxes = np.asarray(o["boxes"].detach().float().cpu().numpy(), float)
            m = int(np.argmax(scores))
            rec[p] = {"box": [float(v) for v in boxes[m]], "score": round(float(scores[m]), 4),
                      "n_boxes": int(scores.size),
                      "all_boxes": [[round(float(v), 1) for v in b] for b in boxes.tolist()],
                      "all_scores": [round(float(s), 3) for s in scores.tolist()]}
        dets[int(fi)] = rec
    hits = {p: sum(1 for r in dets.values() if p in r) for p in keywords}
    best = {p: max([r[p]["score"] for r in dets.values() if p in r] or [0.0]) for p in keywords}
    scan = {"prompts": keywords, "frame_indices": idx, "frame_selection": selection, "n_frames_in_dir": len(files), "min_score": min_score,
            "frames_dir": str(Path(frames_dir).resolve()), "detections": dets,
            "model_identity": getattr(proc, "model_identity", None),
            "summary": {"frames_with_hit": hits, "best_score_per_keyword": best,
                        "not_found": [p for p in keywords if hits[p] == 0]}}
    write_json(scan, out_json)
    return scan


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=TOOL["summary"])
    ap.add_argument("frames_dir")
    ap.add_argument("out_json")
    ap.add_argument("--keywords", nargs="+", required=True)
    ap.add_argument("--frames", nargs="*", type=int)
    ap.add_argument("--min-score", type=float, default=0.1)
    ap.add_argument("--version", default="sam3", choices=["sam3"])
    a = ap.parse_args(argv)
    s = detect_objects(a.frames_dir, a.keywords, a.out_json, a.frames or None, a.min_score, a.version)
    print(json.dumps(s["summary"], indent=1))


if __name__ == "__main__":
    sys.exit(main())
