"""One textured 3D mesh per tracked object with SAM 3D Objects."""
from __future__ import annotations

TOOL = {
    "name": "reconstruct_object_meshes",
    "title": "Reconstruct object meshes",
    "type": "skill",
    "category": "Perception & Grounding",
    "summary": "SAM 3D Objects on the best frame of each tracked object: a textured .glb mesh, its predicted pose, and the anchor frame used.",
    "description": (
        "A rigid object needs one good reconstruction, not one per frame. For each label the "
        "anchor frame is chosen by solidity x temporal mask consistency x relative area, "
        "penalising overlap with the hands, so a frame where the object is whole, still and "
        "unoccluded wins. SAM 3D Objects then reconstructs the masked crop into a textured mesh "
        "with mesh post-processing and texture baking, and the object-to-camera similarity it "
        "predicts is saved beside it. Requires the gated SAM 3D Objects checkpoints."
    ),
    "inputs": [
        {"name": "frames_dir", "type": "path", "description": "Folder of frame%05d.jpg."},
        {"name": "masks_dir", "type": "path", "description": "Palette label maps frame%05d.png."},
        {"name": "out_dir", "type": "path", "description": "Receives <name>.glb, <name>_pose.json, <name>_anchor.png, <name>_mask.png."},
        {"name": "sam3d_repo", "type": "path", "description": "Clone of facebookresearch/sam-3d-objects with checkpoints/hf/pipeline.yaml."},
        {"name": "labels", "type": "dict[int, str] | None", "description": "label -> name; default every label present, named obj<label>."},
        {"name": "hand_masks_dir", "type": "path | None", "description": "Optional per-frame hand masks (RGBA layers or binary) to penalise and cut out."},
        {"name": "anchor_frame", "type": "int | None", "description": "Force this frame instead of picking one."},
        {"name": "seed, min_area", "type": "int", "description": "Sampling seed (42) and minimum mask area in pixels (200)."},
    ],
    "outputs": [
        {"name": "result", "type": "dict", "description": "per object: anchor frame, glb path, pose json path, mask area."},
    ],
    "status": "tested",
    "entrypoint": "reconstruct_object_meshes",
    "requirements": ["torch", "sam-3d-objects repository and gated checkpoints (github.com/facebookresearch/sam-3d-objects)", "trimesh"],
    "tags": ["SAM 3D", "mesh", "reconstruction", "texture"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Open the Bottle"],
    "cli": "python -m tool_library.skills.reconstruct_object_meshes frames/ track/pred meshes/ --sam3d-repo sam-3d-objects",
}

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from tool_library.common.video import frame_files, load_label_map, read_frame, write_json


def _solidity(mask):
    m = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    ha = cv2.contourArea(hull)
    return float(cv2.contourArea(c) / ha) if ha > 0 else 0.0


def _iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def pick_anchor_frame(masks, hand_masks=None, min_area=200):
    """The frame where the object is whole and still: solidity x neighbour IoU x area."""
    areas = np.array([m.sum() for m in masks])
    big = areas[areas >= min_area]
    if not len(big):
        return None
    med = np.median(big)
    best, best_s = None, 0.0
    for i, m in enumerate(masks):
        if areas[i] < min_area:
            continue
        t = [_iou(m, masks[j]) for j in (i - 1, i + 1) if 0 <= j < len(masks)]
        s = _solidity(m) * (np.mean(t) if t else 0.0) * min(1.0, areas[i] / med)
        if hand_masks is not None and hand_masks[i] is not None and m.sum() > 0:
            s *= max(0.0, 1.0 - 3.0 * (m & hand_masks[i]).sum() / m.sum())   # arms score well; punish them
        if s > best_s:
            best, best_s = i, s
    return best


def _hand_mask(hand_masks_dir, i, shape):
    if not hand_masks_dir:
        return None
    for ext in ("png", "jpg"):
        p = Path(hand_masks_dir) / f"frame{i:05d}.{ext}"
        if p.exists():
            im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            m = im[..., 3] > 0 if im.ndim == 3 and im.shape[2] == 4 else (im if im.ndim == 2 else im[..., 0]) > 0
            if m.shape != shape:
                m = cv2.resize(m.astype(np.uint8), shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
            return m
    return None


def reconstruct_object_meshes(frames_dir, masks_dir, out_dir, sam3d_repo, labels=None, hand_masks_dir=None,
                              anchor_frame=None, seed=42, min_area=200):
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    files = frame_files(frames_dir)
    maps = [load_label_map(Path(masks_dir) / f"frame{i:05d}.png") for i in range(len(files))]
    present = sorted(set(int(v) for m in maps for v in np.unique(m) if v > 0))
    labels = {int(k): str(v) for k, v in (labels or {l: f"obj{l}" for l in present}).items()}
    shape = maps[0].shape
    hands = [_hand_mask(hand_masks_dir, i, shape) for i in range(len(files))] if hand_masks_dir else None
    anchors = {}
    for lab in labels:
        masks = [m == lab for m in maps]
        i = anchor_frame if anchor_frame is not None and masks[anchor_frame].sum() >= min_area \
            else pick_anchor_frame(masks, hands, min_area)
        if i is not None:
            anchors[lab] = int(i)
    assert anchors, "no label has a usable mask"
    sys.path.insert(0, str(Path(sam3d_repo) / "notebook"))
    # the SAM 3D notebook helper reads CUDA_HOME from CONDA_PREFIX at import time
    os.environ.setdefault("CONDA_PREFIX", os.environ.get("CUDA_HOME", str(Path(sys.executable).resolve().parents[1])))
    try:
        from inference import Inference
    except ImportError as e:
        raise ImportError("SAM 3D Objects is required: clone https://github.com/facebookresearch/sam-3d-objects, "
                          "download its gated checkpoints to checkpoints/hf and pass sam3d_repo") from e
    inference = Inference(str(Path(sam3d_repo) / "checkpoints" / "hf" / "pipeline.yaml"), compile=False)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {}
    from PIL import Image
    for lab, i in anchors.items():
        name = labels[lab]
        img = read_frame(files[i])
        mask = (maps[i] == lab).astype(np.uint8)
        if hands is not None and hands[i] is not None:
            mask[hands[i]] = 0
        vis = img.copy()
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (255, 0, 255), 2)
        Image.fromarray(vis).save(out / f"{name}_anchor.png")
        Image.fromarray(mask * 255).save(out / f"{name}_mask.png")
        rgba = inference.merge_mask_to_rgba(img, mask)
        o = inference._pipeline.run(rgba, None, int(seed), stage1_only=False, with_mesh_postprocess=True,
                                    with_texture_baking=True, with_layout_postprocess=False, use_vertex_color=False)
        pose = {"anchor_frame": int(i), "quat_order": "wxyz"}
        for k in ("rotation", "translation", "scale"):
            if o.get(k) is not None:
                pose[k] = np.asarray(o[k].detach().cpu()).reshape(-1).tolist()
        write_json(pose, out / f"{name}_pose.json")
        glb = o.get("glb")
        if glb is not None:
            glb.export(str(out / f"{name}.glb"))
        result[name] = {"label": lab, "anchor_frame": int(i), "glb": str(out / f"{name}.glb") if glb is not None else None,
                        "pose_json": str(out / f"{name}_pose.json"), "mask_area_px": int(mask.sum())}
    write_json(result, out / "result.json")
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=TOOL["summary"])
    ap.add_argument("frames_dir"); ap.add_argument("masks_dir"); ap.add_argument("out_dir")
    ap.add_argument("--sam3d-repo", required=True)
    a = ap.parse_args()
    print(json.dumps(reconstruct_object_meshes(a.frames_dir, a.masks_dir, a.out_dir, a.sam3d_repo), indent=1))
