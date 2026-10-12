"""Step 2: remove the demonstrator and the manipulated objects from the human video.

    human video --SAM 3--> erase masks --VOID--> clean background plate(s)

SAM 3 segments everything named in inpaint.erase_prompts (the hands and arms,
plus the objects the simulation re-creates). VOID then deletes those regions
and the shadows they cast. The result is the background the rendered robot is
composited onto.

The ego camera is static, so the background is one plate: a pixel that is
ever seen uncovered keeps its real color (the median over the uncovered
frames), and only pixels that stay covered for the whole clip come from VOID.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import imageio
import numpy as np

PKG_ROOT = Path(__file__).resolve().parent.parent
VOID_WINDOW, VOID_STRIDE = 85, 56          # VOID's temporal window and slide


def void_dir():
    d = Path(os.environ.get("VOID_DIR", PKG_ROOT / "third_party" / "void-model"))
    assert (d / "inference/cogvideox_fun/predict_v2v.py").exists(), (
        f"VOID not found at {d}: run setup.sh or set VOID_DIR")
    return d


def read_video(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    assert frames, f"could not read {path}"
    return frames, float(fps)


def enlarge(mask, scale):
    """Scale every connected component about its centroid, so the faint outline
    an inpainter leaves along a silhouette falls inside the mask."""
    if scale <= 1.0 or not mask.any():
        return mask
    out = mask.copy()
    n, lab, _stats, cents = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
    for i in range(1, n):
        cx, cy = cents[i]
        M = np.float32([[scale, 0, cx * (1 - scale)], [0, scale, cy * (1 - scale)]])
        out |= cv2.warpAffine((lab == i).astype(np.uint8) * 255, M, mask.shape[::-1],
                              flags=cv2.INTER_NEAREST)
    return out


def run_void(work, frames, masks, fps, ip, log):
    """VOID pass 1 on (frames, erase masks). Returns RGB frames at the input size."""
    from huggingface_hub import hf_hub_download, snapshot_download
    n = len(frames)
    H, W = frames[0].shape[:2]
    seq, out = work / "void_in" / "clip", work / "void_out"
    shutil.rmtree(seq, ignore_errors=True)
    shutil.rmtree(out, ignore_errors=True)
    seq.mkdir(parents=True)
    # the quadmask values {0, 127, 255} must survive the codec: near-lossless writes
    vw = imageio.get_writer(seq / "input_video.mp4", fps=fps, macro_block_size=1,
                            output_params=["-crf", "10"])
    qw = imageio.get_writer(seq / "quadmask_0.mp4", fps=fps, macro_block_size=1,
                            output_params=["-crf", "0"])
    ring_k = np.ones((ip["interaction_px"],) * 2, np.uint8)
    for f, m in zip(frames, masks):
        vw.append_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        quad = np.full((H, W), 255, np.uint8)                 # background: keep
        quad[cv2.dilate(m, ring_k) > 0] = 127                 # interactions: shadows live here
        quad[m > 0] = 0                                       # object: remove
        qw.append_data(np.stack([quad] * 3, -1))
    vw.close()
    qw.close()
    (seq / "prompt.json").write_text(json.dumps({"bg": ip["background_prompt"]}))

    # VOID denoises max_video_length frames whatever the input length, in
    # 85-frame windows slid by 56: only 85 + 56k is a valid total.
    L = ((n - 2) // 4 + 1) * 4 + 1
    vlen = VOID_WINDOW if L <= VOID_WINDOW else VOID_WINDOW + VOID_STRIDE * (
        (L - VOID_WINDOW + VOID_STRIDE - 1) // VOID_STRIDE)
    cmd = [sys.executable, "inference/cogvideox_fun/predict_v2v.py",
           "--config", "config/quadmask_cogvideox.py",
           f"--config.data.max_video_length={vlen}",
           f"--config.data.data_rootdir={seq.parent}",
           "--config.experiment.run_seqs=clip",
           f"--config.experiment.save_path={out}",
           "--config.experiment.skip_if_exists=False",
           f"--config.video_model.model_name={snapshot_download('alibaba-pai/CogVideoX-Fun-V1.5-5b-InP')}",
           f"--config.video_model.transformer_path={hf_hub_download('netflix/void-model', 'void_pass1.safetensors')}",
           f"--config.system.gpu_memory_mode={ip['gpu_memory_mode']}"]
    log(f"[plate] VOID on {n} frames (loads a 5B video model; several minutes)")
    t0 = time.time()
    with open(work / "void.log", "w") as lf:
        rc = subprocess.run(cmd, cwd=void_dir(), stdout=lf, stderr=subprocess.STDOUT).returncode
    outs = [o for o in sorted(glob.glob(str(out / "clip-fg=*[0-9].mp4")))
            if not o.endswith("_tuple.mp4")]
    assert rc == 0 and outs, f"VOID failed (exit {rc}); see {work / 'void.log'}"
    log(f"[plate] VOID done in {time.time() - t0:.0f}s")
    rd = imageio.get_reader(outs[-1])
    gen = [cv2.resize(f, (W, H), interpolation=cv2.INTER_CUBIC) for f in rd]
    rd.close()
    assert len(gen) >= n, f"VOID returned {len(gen)} frames for {n} inputs"
    return gen[:n]                                            # the tail is window padding


def paste(frames, masks, gen, ip):
    """Adopt VOID's pixels inside the erase mask always, and in the interaction
    ring only where VOID changed them (its shadow removal)."""
    ring_k = np.ones((ip["interaction_px"],) * 2, np.uint8)
    out = []
    for f, m, g in zip(frames, masks, gen):
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        ring = (cv2.dilate(m, ring_k) > 0) & (m == 0)
        diff = cv2.absdiff(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),
                           cv2.cvtColor(g, cv2.COLOR_RGB2GRAY))
        adopt = (m > 0) | (ring & (cv2.GaussianBlur(diff, (9, 9), 0) > ip["ring_diff_threshold"]))
        a = cv2.GaussianBlur(adopt.astype(np.float32), (7, 7), 0)[..., None]
        out.append((rgb * (1 - a) + g.astype(np.float32) * a).astype(np.uint8))
    return out


def static_plate(frames, masks, clean, ip, log=print):
    """One plate for a camera that does not move: real pixels wherever the
    background is ever uncovered, VOID only where it never is."""
    ring_k = np.ones((ip["interaction_px"],) * 2, np.uint8)
    raw = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]).astype(np.float32)
    free = np.stack([cv2.dilate(m, ring_k) == 0 for m in masks])          # (n, H, W)
    # a camera's auto white balance settles over the first frames: drop frames
    # whose color, on pixels that are uncovered in every frame, is off the clip's
    always = free.all(0)
    if always.sum() > 1000:
        means = np.array([f[always].mean(0) for f in raw])
        drift = np.abs(means - np.median(means, 0)).max(1)
        keep = drift <= float(ip["color_drift_tol"])
        if keep.sum() >= 3 and not keep.all():
            log(f"[plate] {int((~keep).sum())} frames dropped for color drift "
                f"(up to {drift.max():.0f} levels): white balance still settling")
            raw, free = raw[keep], free[keep]
            clean = [c for c, k in zip(clean, keep) if k]
    count = free.sum(0)
    raw[~free] = np.nan
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            seen = np.nanmedian(raw, axis=0)
    filled = np.median(np.stack(clean).astype(np.float32), axis=0)
    w = np.clip((count.astype(np.float32) - 1) / 4.0, 0.0, 1.0)           # need a few real samples
    w = cv2.GaussianBlur(w, (21, 21), 0)[..., None]
    plate = np.where(np.isnan(seen), filled, seen) * w + filled * (1 - w)
    return np.clip(plate, 0, 255).astype(np.uint8), float((count == 0).mean())


def check_aspect(w, h, W, H, what):
    """The plate is resized to the output size W x H, and the renders use the camera's intrinsics at that size:
    an image of another aspect ratio would be stretched and never line up with them."""
    from .config import ASPECT_TOL, aspect_mismatch
    assert aspect_mismatch((w, h), (W, H)) <= ASPECT_TOL, (
        f"{what} is {w} x {h} but cameras.width x height is {W} x {H}: set cameras.width / height to the "
        "video's aspect ratio (and give the intrinsics for that image: cameras.ego fx, fy, cx, cy, image_size)")


def run(cfg, log=print):
    ip, work = cfg["inpaint"], cfg.out / "plate"
    work.mkdir(parents=True, exist_ok=True)
    W, H = cfg["cameras"]["width"], cfg["cameras"]["height"]
    meta_f = work / "meta.json"

    if ip.get("clean_plate"):
        plate = cv2.imread(str(cfg.path_of(ip["clean_plate"])))
        assert plate is not None, f"cannot read inpaint.clean_plate {ip['clean_plate']}"
        check_aspect(plate.shape[1], plate.shape[0], W, H, "inpaint.clean_plate")
        cv2.imwrite(str(work / "plate.png"), cv2.resize(plate, (W, H)))
        meta_f.write_text(json.dumps({"source": "user clean_plate"}))
        log(f"[plate] using the provided clean plate -> {work / 'plate.png'}")
        return

    frames, fps = read_video(cfg.path("human_video"))
    check_aspect(frames[0].shape[1], frames[0].shape[0], W, H, "the human video")
    # a static background needs samples across the clip, not every frame
    idx = np.unique(np.linspace(0, len(frames) - 1,
                                min(len(frames), int(ip["max_frames"]))).round().astype(int))
    sel = [frames[i] for i in idx]
    sel_fps = fps * len(sel) / len(frames)
    log(f"[plate] {cfg.path('human_video').name}: {len(frames)} frames @ {fps:.1f} fps, "
        f"{len(sel)} frames sampled for the clean plate")

    fdir, mdir = work / "frames", work / "masks"
    shutil.rmtree(fdir, ignore_errors=True)
    fdir.mkdir()
    for k, f in enumerate(sel):
        cv2.imwrite(str(fdir / f"frame{k:05d}.png"), f)
    if ip.get("masks_dir"):
        mdir = cfg.path_of(ip["masks_dir"])
        log(f"[plate] using the provided erase masks in {mdir}")
    else:
        shutil.rmtree(mdir, ignore_errors=True)
        t0 = time.time()
        subprocess.run([sys.executable, "-m", "stage4.masks", str(fdir), str(mdir),
                        ",".join(ip["erase_prompts"]), str(ip["min_score"])],
                       cwd=PKG_ROOT, check=True)
        log(f"[plate] SAM 3 masks in {time.time() - t0:.0f}s")
    dil = np.ones((ip["mask_dilate_px"],) * 2, np.uint8)
    masks = []
    for k in range(len(sel)):
        m = cv2.imread(str(Path(mdir) / f"frame{k:05d}.png"), cv2.IMREAD_GRAYSCALE)
        assert m is not None, f"missing erase mask for frame {k} in {mdir}"
        if m.shape != sel[k].shape[:2]:
            m = cv2.resize(m, sel[k].shape[1::-1], interpolation=cv2.INTER_NEAREST)
        masks.append(enlarge(cv2.dilate((m > 0).astype(np.uint8) * 255, dil), ip["mask_scale"]))

    gen = run_void(work, sel, masks, max(1.0, sel_fps), ip, log)
    clean = paste(sel, masks, gen, ip)

    meta = {"n_source_frames": len(frames), "source_fps": fps,
            "indices": [int(i) for i in idx],
            "erase_coverage": round(float(np.mean([(m > 0).mean() for m in masks])), 4)}
    cdir = work / "clean"
    shutil.rmtree(cdir, ignore_errors=True)
    cdir.mkdir()
    plate, never = static_plate(sel, masks, clean, ip, log)
    meta["never_uncovered"] = round(never, 4)
    cv2.imwrite(str(work / "plate.png"),
                cv2.resize(cv2.cvtColor(plate, cv2.COLOR_RGB2BGR), (W, H)))
    log(f"[plate] clean plate -> {work / 'plate.png'} "
        f"({never:.0%} of the frame is never uncovered and comes from VOID)")
    for k, c in enumerate(clean):
        cv2.imwrite(str(cdir / f"frame{k:05d}.png"),
                    cv2.resize(cv2.cvtColor(c, cv2.COLOR_RGB2BGR), (W, H)))

    vw = imageio.get_writer(work / "preview.mp4", fps=max(1.0, sel_fps), macro_block_size=1)
    for f, m, c in zip(sel, masks, clean):                    # [raw + erase outline | clean]
        ov = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).copy()
        cont, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, cont, -1, (255, 220, 0), 2)
        vw.append_data(np.concatenate([ov, c], axis=1))
    vw.close()
    meta_f.write_text(json.dumps(meta, indent=1))
    log(f"[plate] per-frame VOID results -> {cdir}; preview {work / 'preview.mp4'}")
