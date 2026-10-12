"""Which way is down: GeoCalib gravity direction for a static-camera video."""
from __future__ import annotations

TOOL = {
    "name": "estimate_gravity",
    "title": "Estimate gravity",
    "type": "skill",
    "category": "Perception & Grounding",
    "summary": "GeoCalib per-frame gravity, aggregated by a confidence-weighted spherical mean with MAD outlier rejection.",
    "description": (
        "A reconstruction needs to know which camera direction is down before it can place a "
        "table plane or stand an object upright. GeoCalib estimates the gravity vector in the "
        "camera frame from a single image; this runs it on a subsample of frames, discards "
        "frames whose estimate sits more than a few median absolute deviations from the median, "
        "and averages the rest weighted by GeoCalib's own confidence. The result is one unit "
        "vector in camera coordinates with its roll and pitch in degrees. Assumes the camera does "
        "not move during the clip."
    ),
    "inputs": [
        {"name": "frames_dir", "type": "path", "description": "Folder of frames."},
        {"name": "out_json", "type": "path | None", "description": "Where to write the result."},
        {"name": "max_frames", "type": "int", "description": "Frames to subsample uniformly. Default 60."},
        {"name": "camera_model", "type": "str", "description": "'pinhole' (default), 'simple_radial' or 'simple_divisional'."},
        {"name": "mad_threshold", "type": "float", "description": "Outlier cut in MADs. Default 3."},
    ],
    "outputs": [
        {"name": "gravity", "type": "dict",
         "description": "up_cam: world-up unit vector in OpenCV camera coordinates (x right, y down, z forward); gravity_cam = -up_cam; roll_deg; pitch_deg (negative = camera looks down); n_frames, n_inliers; per_frame list with frame index, up_cam, roll/pitch, confidence, outlier flag."},
    ],
    "status": "tested",
    "entrypoint": "estimate_gravity",
    "requirements": ["torch", "geocalib (pip install git+https://github.com/cvg/GeoCalib.git)"],
    "tags": ["gravity", "camera", "GeoCalib"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope", "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting", "Biology Experiment"],
    "cli": "python -m tool_library.skills.estimate_gravity frames/ gravity.json",
}

import json
import math
import sys

import numpy as np

from tool_library.common.video import frame_files, write_json


def estimate_gravity(frames_dir, out_json=None, max_frames=60, camera_model="pinhole", mad_threshold=3.0):
    try:
        import torch
        from geocalib import GeoCalib
    except ImportError as e:
        raise ImportError("GeoCalib is required: pip install git+https://github.com/cvg/GeoCalib.git") from e
    files = frame_files(frames_dir)
    assert files, f"no frames in {frames_dir}"
    idx = list(range(len(files)))
    if len(files) > max_frames:
        idx = [int(i) for i in np.round(np.linspace(0, len(files) - 1, int(max_frames)))]
    files = [files[i] for i in idx]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GeoCalib(weights="pinhole" if camera_model == "pinhole" else "distorted").to(device).eval()
    vecs, confs, rows = [], [], []
    with torch.no_grad():
        for fi, f in zip(idx, files):
            r = model.calibrate(model.load_image(str(f)).to(device), camera_model=camera_model)
            g = r["gravity"]
            v = g.vec3d.squeeze(0).cpu().numpy()        # GeoCalib's vec3d is world-UP in camera coords
            c = float((r["up_confidence"].mean() + r["latitude_confidence"].mean()) / 2)
            vecs.append(v)
            confs.append(c)
            rows.append({"frame": int(fi), "file": str(f), "up_cam": v.tolist(), "roll_deg": math.degrees(float(g.roll)),
                         "pitch_deg": math.degrees(float(g.pitch)), "confidence": c})
    V, C = np.asarray(vecs), np.asarray(confs)
    mean = V.mean(0)
    mean /= np.linalg.norm(mean)
    ang = np.arccos(np.clip(V @ mean, -1, 1))
    med = np.median(ang)
    mad = max(np.median(np.abs(ang - med)), 1e-6)
    keep = ang <= med + mad_threshold * mad
    final = (V[keep] * (C[keep] / C[keep].sum())[:, None]).sum(0)
    final /= np.linalg.norm(final)
    for row, k in zip(rows, keep):
        row["outlier"] = not bool(k)
    # GeoCalib's parameterisation: up = [-sin r cos p, -cos r cos p, sin p]
    out = {"up_cam": [float(x) for x in final], "gravity_cam": [float(-x) for x in final],
           "roll_deg": math.degrees(math.atan2(-final[0], -final[1])),
           "pitch_deg": math.degrees(math.asin(float(np.clip(final[2], -1, 1)))),
           "n_frames": int(len(V)), "n_inliers": int(keep.sum()), "per_frame": rows}
    if out_json:
        write_json(out, out_json)
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    print(json.dumps({k: v for k, v in estimate_gravity(a[0], a[1] if len(a) > 1 else None).items() if k != "per_frame"}, indent=1))
