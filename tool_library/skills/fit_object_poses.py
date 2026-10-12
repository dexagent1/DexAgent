"""Per-frame 6-DoF object poses: fit a reconstructed mesh into every frame's mask and depth."""
from __future__ import annotations

TOOL = {
    "name": "fit_object_poses",
    "title": "Fit object poses",
    "type": "skill",
    "category": "Perception & Grounding",
    "summary": "Fits one rigid mesh per object into every frame by mask chamfer, inverse-depth and gravity losses, with occlusion-aware freezing, anchor interpolation and wobble removal.",
    "description": (
        "Takes the single reconstructed mesh of each object and recovers its camera-frame "
        "pose in every frame. The init frame tries all 24 gravity-aligned rotations with "
        "rotation and a global scale free and keeps the best after refinement; every other "
        "frame warm-starts from its neighbour. Under occlusion the rotation is frozen when too "
        "little mask is visible, the pose is carried verbatim when the mask still fits inside "
        "the previous silhouette and a hand or another object explains the missing part, "
        "low-confidence frames are replaced by slerp between anchors, and deviate-and-return "
        "wobbles during an arm sweep are demoted. Output per object: T_seq (N,4,4) camera-frame "
        "poses, the solved scale and mesh centre, per-frame IoU, anchors, frozen flags and a "
        "visibility range. verts_cam(t) = scale * (V - center) @ R_t.T + t_t."
    ),
    "inputs": [
        {"name": "frames_dir", "type": "path", "description": "Folder of frame%05d.jpg."},
        {"name": "masks_dir", "type": "path", "description": "Palette label maps frame%05d.png (track_objects pred/)."},
        {"name": "mesh_dir", "type": "path", "description": "Folder with <object>.glb/.obj per object (reconstruct_object_meshes)."},
        {"name": "depth_npz", "type": "path", "description": "estimate_depth output (depths in metres)."},
        {"name": "objects", "type": "list[str]", "description": "Object names, matching the mesh file stems and mask labels."},
        {"name": "out_dir", "type": "path", "description": "Receives <object>.npz per object."},
        {"name": "focal_px", "type": "float", "description": "Pinhole focal length in pixels (fx = fy)."},
        {"name": "labels", "type": "dict[str,int] | None", "description": "name -> label; default read from result.json beside masks_dir."},
        {"name": "gravity_json", "type": "path | None", "description": "estimate_gravity output enabling the gravity loss."},
        {"name": "hand_layers_dir", "type": "path | None", "description": "extract_hand_meshes layers/ (RGBA) used as occluders."},
        {"name": "options", "type": "dict", "description": "Any fitter flag without dashes, e.g. {'max_frames': 60, 'smooth_win': 5, 'w_grav': 2.0, 'init_frame': 0}."},
        {"name": "python", "type": "str", "description": "Interpreter for the fitter subprocess (default: the current one)."},
    ],
    "outputs": [
        {"name": "result", "type": "dict", "description": "per object: npz path, mean/min IoU, anchors, frozen frames, visible range."},
    ],
    "status": "tested",
    "entrypoint": "fit_object_poses",
    "requirements": ["torch (CUDA recommended)", "trimesh", "scipy", "opencv-python", "pillow"],
    "tags": ["pose", "6-DoF", "tracking", "mesh"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Open the Bottle"],
    "cli": "python -m tool_library.skills.fit_object_poses frames/ track/pred meshes/ depth.npz cup,bowl poses/ --focal 697",
}

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from tool_library.common.video import slug, write_json

SCRIPT = Path(__file__).resolve().parent.parent / "common" / "pose_fit_script.py"


def fit_object_poses(frames_dir, masks_dir, mesh_dir, depth_npz, objects, out_dir, focal_px, labels=None,
                     gravity_json=None, hand_layers_dir=None, options=None, python=None):
    objects = [objects] if isinstance(objects, str) else list(objects)
    cmd = [python or sys.executable, str(SCRIPT), "--frames-dir", str(frames_dir), "--masks-dir", str(masks_dir),
           "--mesh-dir", str(mesh_dir), "--depth-npz", str(depth_npz), "--objects", ",".join(objects),
           "--out-dir", str(out_dir), "--focal", str(float(focal_px))]
    if labels:
        cmd += ["--labels", json.dumps({k: int(v) for k, v in labels.items()})]
    if gravity_json:
        cmd += ["--gravity", str(gravity_json)]
    if hand_layers_dir:
        cmd += ["--hand-layers", str(hand_layers_dir)]
    for k, v in (options or {}).items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    log = Path(out_dir) / "fit_log.txt"
    with open(log, "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    if rc != 0:
        raise RuntimeError(f"pose fitter failed (rc={rc}); see {log}\n" + log.read_text()[-3000:])
    result = {}
    for obj in objects:
        p = Path(out_dir) / f"{slug(obj)}.npz"
        d = np.load(p)
        ious = d["ious"]
        v = ious[~np.isnan(ious)]
        vis = np.nonzero(d["visible"])[0]
        result[obj] = {"npz": str(p), "n_frames": int(len(d["T_seq"])), "scale": float(d["scale"]),
                       "iou_mean": float(v.mean()) if len(v) else None, "iou_min": float(v.min()) if len(v) else None,
                       "n_anchors": int(len(d["anchors"])), "n_frozen": int(d["frozen"].sum()),
                       "visible_range": [int(vis[0]), int(vis[-1])] if len(vis) else None}
    write_json(result, Path(out_dir) / "result.json")
    return result


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=TOOL["summary"])
    for a in ("frames_dir", "masks_dir", "mesh_dir", "depth_npz", "objects", "out_dir"):
        ap.add_argument(a)
    ap.add_argument("--focal", type=float, required=True)
    ap.add_argument("--gravity"); ap.add_argument("--hand-layers"); ap.add_argument("--max-frames", type=int)
    a = ap.parse_args(argv)
    opts = {"max_frames": a.max_frames} if a.max_frames else {}
    print(json.dumps(fit_object_poses(a.frames_dir, a.masks_dir, a.mesh_dir, a.depth_npz, a.objects.split(","), a.out_dir,
                                      a.focal, gravity_json=a.gravity, hand_layers_dir=a.hand_layers, options=opts), indent=1))


if __name__ == "__main__":
    sys.exit(main())
