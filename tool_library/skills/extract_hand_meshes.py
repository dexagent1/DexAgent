"""The demonstrator's hands as MANO meshes per frame (ViTPose keypoints -> HaMeR)."""
from __future__ import annotations

TOOL = {
    "name": "extract_hand_meshes",
    "title": "Extract hand meshes",
    "type": "skill",
    "category": "Perception & Grounding",
    "summary": "MANO hand meshes for every frame: ViTPose finds the hands and their handedness, HaMeR recovers vertices, camera translation and MANO parameters.",
    "description": (
        "Runs ViTPose whole-body pose on the full frame (egocentric footage rarely shows a "
        "person, so a person detector would find nothing), turns the wrist and finger keypoints "
        "into hand boxes with correct left/right labels, and passes each box to HaMeR. Per frame "
        "it records the MANO vertices, the weak-perspective camera translation lifted to the "
        "full image, handedness, boxes and MANO parameters, and renders an overlay video plus "
        "per-frame RGBA hand layers for compositing. Detection runs on the raw frame and falls "
        "back to a contrast/brightness-boosted copy only when nothing is found: the boost raised "
        "recall on dark egocentric clips and lowered it on bright bench footage. The output "
        "npz is what hand retargeting and the robot's initial pose are read from."
    ),
    "inputs": [
        {"name": "frames_dir", "type": "path", "description": "Folder of frame%05d.jpg."},
        {"name": "out_dir", "type": "path", "description": "Receives hands.npz, overlay.mp4, layers/frame%05d.png, summary.json."},
        {"name": "hamer_root", "type": "path | None", "description": "HaMeR checkout holding _DATA/ (MANO, HaMeR and ViTPose checkpoints), vitpose_model.py and third-party/ViTPose; the tool runs with this as the working directory because HaMeR resolves its data relatively."},
        {"name": "checkpoint", "type": "path | None", "description": "HaMeR checkpoint; the package default when None."},
        {"name": "fps", "type": "float", "description": "Frame rate of the overlay video. Default 10."},
        {"name": "enhance", "type": "str | tuple | None", "description": "'auto' (default): detect on the raw frame and, when no hand is found, retry on a contrast x2 + 50 brightened copy (the setting a sweep picked for dark egocentric clips; measured to HURT on well-lit bench footage). A (contrast, brightness) tuple forces it; None disables."},
        {"name": "kp_thresh", "type": "float", "description": "Keypoint confidence for a hand box. Default 0.5."},
        {"name": "rescale_factor", "type": "float", "description": "HaMeR crop padding. Default 2.0."},
        {"name": "max_frames", "type": "int | None", "description": "Limit the frames processed."},
    ],
    "outputs": [
        {"name": "summary", "type": "dict",
         "description": "frames processed, frames_with_hands, detection_rate, hands_left, hands_right, paths. hands.npz holds per-frame object arrays verts (n_hands,778,3), cam_t, is_right, bbox, global_orient, hand_pose, betas and the MANO faces."},
    ],
    "status": "tested",
    "entrypoint": "extract_hand_meshes",
    "requirements": ["torch", "hamer (github.com/geopavlakos/hamer, with its ViTPose third-party checkout)", "pyrender", "imageio"],
    "tags": ["hands", "MANO", "HaMeR", "ViTPose"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope", "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting", "Biology Experiment"],
    "cli": "python -m tool_library.skills.extract_hand_meshes frames/ hands/",
}

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from tool_library.common.video import frame_files, read_frame, write_json, write_video

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def _enhance(rgb, contrast, brightness):
    if contrast == 1.0 and brightness == 0.0:
        return rgb
    return np.clip(rgb.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)


def _hand_boxes(vitposes_out, kp_thresh):
    """Boxes and handedness from the 42 hand keypoints of each detected body."""
    boxes, is_right = [], []
    for vp in vitposes_out:
        left, right = vp["keypoints"][-42:-21], vp["keypoints"][-21:]
        for keyp, r in ((left, 0), (right, 1)):
            valid = keyp[:, 2] > kp_thresh
            if valid.sum() > 3:
                boxes.append([keyp[valid, 0].min(), keyp[valid, 1].min(),
                              keyp[valid, 0].max(), keyp[valid, 1].max()])
                is_right.append(r)
    return boxes, is_right


def extract_hand_meshes(frames_dir, out_dir, hamer_root=None, checkpoint=None, fps=10.0, enhance="auto",
                        kp_thresh=0.5, rescale_factor=2.0, max_frames=None):
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    frames_dir, out_dir = Path(frames_dir).resolve(), Path(out_dir).resolve()
    cwd = os.getcwd()
    if hamer_root:
        root = Path(hamer_root).resolve()
        for extra in (root / "third-party" / "ViTPose", root):   # mmpose lives in the ViTPose checkout
            if extra.exists():
                sys.path.insert(0, str(extra))
        os.chdir(root)                            # HaMeR/ViTPose resolve _DATA relative to cwd
    try:
        return _run(frames_dir, out_dir, checkpoint, fps, enhance, kp_thresh, rescale_factor, max_frames)
    finally:
        os.chdir(cwd)


def _run(frames_dir, out_dir, checkpoint, fps, enhance, kp_thresh, rescale_factor, max_frames):
    try:
        import torch
        from hamer.datasets.vitdet_dataset import ViTDetDataset
        from hamer.models import DEFAULT_CHECKPOINT, load_hamer
        from hamer.utils import recursive_to
        from hamer.utils.renderer import Renderer, cam_crop_to_full
        from vitpose_model import ViTPoseModel
    except ImportError as e:
        raise ImportError("HaMeR is required: install https://github.com/geopavlakos/hamer with its "
                          "ViTPose checkout (vitpose_model.py on the path) and the demo checkpoints") from e
    files = frame_files(frames_dir)
    if max_frames:
        files = files[: int(max_frames)]
    assert files, f"no frames in {frames_dir}"
    out = Path(out_dir)
    (out / "layers").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_hamer(checkpoint or DEFAULT_CHECKPOINT)
    model = model.to(device).eval()
    renderer = Renderer(cfg, faces=model.mano.faces)
    cpm = ViTPoseModel(device)
    rec = {k: [] for k in ("frame_idx", "n_hands", "verts", "cam_t", "is_right", "bbox",
                           "global_orient", "hand_pose", "betas")}
    overlay, n_with, n_left, n_right = [], 0, 0, 0
    t0 = time.time()
    if enhance == "auto":
        variants = [(1.0, 0.0), (2.0, 50.0)]
    elif enhance:
        variants = [tuple(float(v) for v in enhance)]
    else:
        variants = [(1.0, 0.0)]
    for fi, path in enumerate(files):
        raw = read_frame(path)
        H, W = raw.shape[:2]
        for contrast, brightness in variants:                 # raw first; boosted copy only if nothing found
            rgb = _enhance(raw, contrast, brightness)
            bb, right = _hand_boxes(cpm.predict_pose(rgb, [np.array([[0, 0, W, H, 1.0]], np.float32)]), kp_thresh)
            if bb:
                break
        if not bb:
            rec["frame_idx"].append(fi)
            rec["n_hands"].append(0)
            for k in ("verts", "cam_t", "is_right", "bbox", "global_orient", "hand_pose", "betas"):
                rec[k].append(np.empty((0,)))
            overlay.append(rgb)
            continue
        boxes, rt = np.stack(bb), np.stack(right)
        ds = ViTDetDataset(cfg, rgb[:, :, ::-1].copy(), boxes, rt, rescale_factor=rescale_factor)
        loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
        v_list, t_list, r_list, go, hp, bt = [], [], [], [], [], []
        for batch in loader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                o = model(batch)
            bc, bs, isz = batch["box_center"].float(), batch["box_size"].float(), batch["img_size"].float()
            sf = cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * isz.max()
            pred_cam = o["pred_cam"].clone()
            pred_cam[:, 1] = (2 * batch["right"].float() - 1) * pred_cam[:, 1]   # un-flip left hands
            ctf = cam_crop_to_full(pred_cam, bc, bs, isz, sf).cpu().numpy()
            mp = o["pred_mano_params"]
            for n in range(batch["img"].shape[0]):
                r = int(batch["right"][n].cpu().numpy())
                v = o["pred_vertices"][n].cpu().numpy()
                v[:, 0] = (2 * r - 1) * v[:, 0]
                v_list.append(v); t_list.append(ctf[n]); r_list.append(r)
                go.append(mp["global_orient"][n].cpu().numpy()); hp.append(mp["hand_pose"][n].cpu().numpy())
                bt.append(mp["betas"][n].cpu().numpy())
                n_left += r == 0; n_right += r == 1
        rec["frame_idx"].append(fi); rec["n_hands"].append(len(v_list))
        rec["verts"].append(np.stack(v_list)); rec["cam_t"].append(np.stack(t_list))
        rec["is_right"].append(np.array(r_list)); rec["bbox"].append(boxes)
        rec["global_orient"].append(np.stack(go)); rec["hand_pose"].append(np.stack(hp)); rec["betas"].append(np.stack(bt))
        n_with += 1
        cam_view = renderer.render_rgba_multiple(v_list, cam_t=t_list, render_res=isz[-1].cpu().numpy(),
                                                 is_right=r_list, mesh_base_color=LIGHT_BLUE,
                                                 scene_bg_color=(1, 1, 1), focal_length=float(sf))
        import imageio.v2 as imageio
        imageio.imwrite(str(out / "layers" / f"frame{fi:05d}.png"), (cam_view * 255).astype(np.uint8))
        a = cam_view[:, :, 3:]
        overlay.append(((rgb / 255.0 * (1 - a) + cam_view[:, :, :3] * a) * 255).astype(np.uint8))
    obj = lambda k: np.array(rec[k], dtype=object)
    np.savez(str(out / "hands.npz"), frame_idx=np.array(rec["frame_idx"]), n_hands=np.array(rec["n_hands"]),
             verts=obj("verts"), cam_t=obj("cam_t"), is_right=obj("is_right"), bbox=obj("bbox"),
             global_orient=obj("global_orient"), hand_pose=obj("hand_pose"), betas=obj("betas"),
             faces=model.mano.faces, fps=float(fps))
    write_video(overlay, out / "overlay.mp4", fps)
    summary = {"processed_frames": len(files), "frames_with_hands": n_with,
               "detection_rate": round(n_with / max(1, len(files)), 3), "hands_left": int(n_left),
               "hands_right": int(n_right), "runtime_s": round(time.time() - t0, 1),
               "hands_npz": str(out / "hands.npz"), "overlay_video": str(out / "overlay.mp4")}
    write_json(summary, out / "summary.json")
    return summary


if __name__ == "__main__":
    a = sys.argv[1:]
    print(json.dumps(extract_hand_meshes(a[0], a[1], a[2] if len(a) > 2 else None), indent=1))
