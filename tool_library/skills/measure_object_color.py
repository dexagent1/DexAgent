"""Dominant colours of a masked object, measured from the video."""
from __future__ import annotations

TOOL = {
    "name": "measure_object_color",
    "title": "Measure object color",
    "type": "skill",
    "category": "Perception & Grounding",
    "summary": "Dominant RGB colours of an object from its masked video pixels: border-eroded in proportion to the mask's size, shadow-trimmed, k-means over several frames.",
    "description": (
        "Reads the frames and the object's label maps, keeps the pixels inside the mask "
        "away from its border (mixed edge pixels and a mask that bleeds onto the table would tint the result): "
        "the erosion depth adapts to the mask, `erode_frac` of its inscribed radius, between "
        "`min_erode_px` and `max_erode_px`, so a 28 mm ring a few dozen pixels across keeps only its core instead of a rim of "
        "table-coloured border pixels that outnumber it, while the depth is relaxed when too few pixels "
        "would remain. It then drops the "
        "darkest third of them (self-shadowed interiors are not the surface colour), and runs "
        "a deterministic k-means with luminance-quantile initialisation over up to five frames "
        "around a reference frame. The clusters, sorted by weight, are the colours to paint a "
        "constructed stand-in with; guessing them from a thumbnail is what this replaces."
    ),
    "inputs": [
        {"name": "frames_dir", "type": "path", "description": "Folder of frame%05d.jpg."},
        {"name": "masks_dir", "type": "path", "description": "Folder of palette label maps frame%05d.png (from track_objects), or one binary mask PNG of the object at ref_frame (from segment_with_text_prompts)."},
        {"name": "label", "type": "int | None", "description": "The object's label value in the label maps; None (or a binary mask file) means every non-zero pixel."},
        {"name": "ref_frame", "type": "int", "description": "Frame to centre the sampling on. Default 0."},
        {"name": "k", "type": "int", "description": "Number of colour clusters. Default 3."},
        {"name": "n_frames", "type": "int", "description": "How many frames to pool. Default 5."},
        {"name": "erode_frac", "type": "float", "description": "Erosion depth as a fraction of the mask's inscribed radius (its deepest pixel's distance to the border). Default 0.35."},
        {"name": "min_erode_px", "type": "int", "description": "Smallest erosion depth in pixels (border pixels are mixed on any mask). Default 2."},
        {"name": "max_erode_px", "type": "int | None", "description": "Largest erosion depth in pixels (a big object keeps its outer colours; a mask's spill onto the table is a few pixels at any size). Default 12; None for no cap."},
    ],
    "outputs": [
        {"name": "palette", "type": "list[dict]",
         "description": "The return value is this list itself (not a dict): [{rgb: [r, g, b] in 0..1, weight: fraction}] sorted by weight; clusters under 2% dropped."},
    ],
    "status": "tested",
    "entrypoint": "measure_object_color",
    "requirements": ["numpy", "Pillow"],
    "tags": ["color", "appearance", "masks"],
    "used_in": ["Place Toy in Drawer", "Battery Insertion", "Computer Installation", "Biology Experiment"],
}

import sys
from pathlib import Path

import numpy as np


def _border_depth(m):
    """City-block distance of every mask pixel to the mask's border (1 = a border pixel, 0 outside), by repeated
    4-neighbour erosion; pixels outside the image count as outside the mask."""
    m = np.asarray(m, bool)
    depth = np.zeros(m.shape, np.int32)
    cur = m.copy()
    while cur.any():
        depth[cur] += 1
        nxt = cur.copy()
        nxt[1:, :] &= cur[:-1, :]; nxt[:-1, :] &= cur[1:, :]; nxt[:, 1:] &= cur[:, :-1]; nxt[:, :-1] &= cur[:, 1:]
        nxt[0, :] = nxt[-1, :] = False; nxt[:, 0] = nxt[:, -1] = False
        cur = nxt
    return depth


def core_mask(m, erode_frac=0.35, min_erode_px=2, min_keep=60, max_erode_px=12):
    """The mask without its border band: pixels deeper than erode_frac x the deepest pixel's depth, clipped to
    [min_erode_px, max_erode_px].
    A small object's mask is mostly border (a ring 50 px across: 40% of its pixels within 3 px of the edge, and those
    are where the mask spills onto the table), so a fixed two-pixel erosion leaves the table's colour in the
    average. The depth is relaxed (down to one pixel) when fewer than `min_keep` pixels would remain."""
    depth = _border_depth(m)
    dmax = int(depth.max())
    if dmax == 0:
        return np.zeros_like(np.asarray(m, bool))
    r = max(int(min_erode_px), int(np.ceil(float(erode_frac) * dmax)))
    if max_erode_px is not None:
        r = max(int(min_erode_px), min(r, int(max_erode_px)))
    while r > 0:
        core = depth > r
        if core.sum() >= min_keep or r <= 1:
            return core
        r -= 1
    return depth > 0


def measure_object_color(frames_dir, masks_dir, label=None, ref_frame=0, k=3, n_frames=5, erode_frac=0.35, min_erode_px=2,
                         max_erode_px=12):
    from PIL import Image
    frames_dir, masks_dir = Path(frames_dir), Path(masks_dir)
    binary = label is None or (isinstance(label, str) and not label.strip().lstrip("-").isdigit())
    if isinstance(label, str) and label.strip().lstrip("-").isdigit():
        label = int(label)
    # masks_dir may also be a single binary mask PNG (segment_with_text_prompts output): the frame is ref_frame
    single_mask = masks_dir.is_file()
    cands = [ref_frame] + [ref_frame + d for step in (5, 10, 20, 35) for d in (step, -step)]
    pixels, used = [], 0
    for t in cands:
        if used >= n_frames or t < 0:
            continue
        fp = frames_dir / f"frame{t:05d}.jpg"
        mp = masks_dir if single_mask else masks_dir / f"frame{t:05d}.png"
        if single_mask and t != ref_frame:
            continue
        if not fp.exists() or not mp.exists():
            continue
        arr = np.asarray(Image.open(mp))
        if arr.ndim == 3:
            arr = arr[..., 0]
        m = (arr > 0) if binary else (arr == int(label))
        if not m.any():
            continue
        img = np.asarray(Image.open(fp).convert("RGB"), dtype=np.float32) / 255.0
        if img.shape[:2] != m.shape:                 # erode at the image's resolution, where the colours are read
            m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(
                (img.shape[1], img.shape[0]), Image.NEAREST)) > 127
        if m.sum() < 60:                             # a sliver (the object mostly hidden in this frame); a small object is not
            continue
        m = core_mask(m, erode_frac, min_erode_px, max_erode_px=max_erode_px)
        if m.sum() < 40:
            continue
        pixels.append(img[m])
        used += 1
    if not pixels:
        raise RuntimeError(f"no usable masked pixels for label {label} around frame {ref_frame}")
    px = np.concatenate(pixels)
    lum0 = px @ np.array([0.299, 0.587, 0.114])
    px = px[lum0 >= np.percentile(lum0, 35)]      # the lit surface, not its shadows
    if len(px) > 20000:
        px = px[:: len(px) // 20000 + 1]
    lum = px @ np.array([0.299, 0.587, 0.114])
    order = np.argsort(lum)
    cent = px[order[[int(len(px) * q) for q in np.linspace(0.15, 0.85, k)]]].copy()
    for _ in range(25):
        lab = ((px[:, None, :] - cent[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            if (lab == j).any():
                cent[j] = px[lab == j].mean(0)
    w = np.bincount(lab, minlength=k) / float(len(px))
    return [{"rgb": [round(float(c), 3) for c in cent[j]], "weight": round(float(w[j]), 2)}
            for j in np.argsort(-w) if w[j] > 0.02]


if __name__ == "__main__":
    import json
    a = sys.argv[1:]
    print(json.dumps(measure_object_color(a[0], a[1], int(a[2]), int(a[3]) if len(a) > 3 else 0), indent=1))
