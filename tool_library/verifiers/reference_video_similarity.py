"""Does a generated simulation video reproduce a reference video of the same task?"""
from __future__ import annotations

TOOL = {
    "name": "reference_video_similarity",
    "title": "Reference video similarity",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "Compares a generated simulation video with a reference video of the same task frame by frame at matched "
               "normalized times (SSIM and absolute difference on downscaled grey frames, a motion-energy curve correlation), "
               "writes a side-by-side contact sheet, and passes when the similarity is above a threshold.",
    "description": (
        "A reproducibility check: the same video through the pipeline again, or a hand-built twin against the pipeline's "
        "output, should give the same simulation video up to rendering noise. The two videos may differ in length, so "
        "frames are compared at the same fraction of the duration (`n_samples` pairs), after resizing both to "
        "`width` x `height` grey. Per pair it measures SSIM and the mean absolute difference; over the whole video it "
        "correlates the two motion-energy curves (mean absolute frame-to-frame change) to see whether things move at the "
        "same times. The sheet puts the reference above the candidate for every sampled time, with the per-pair SSIM. "
        "Passes when the mean SSIM is at least `min_ssim` (0.9 is 'the same scene and motion', 0.75 'the same task "
        "from a similar layout', below 0.6 'something else is happening') and every sampled frame of both videos "
        "decodes (an undecodable frame would be compared as black). It judges pixels, not physics: use the "
        "subgoal and episode checks for what the robot did, and this one for whether the output looks like the reference."
    ),
    "inputs": [
        {"name": "video", "type": "str", "description": "The generated video (mp4)."},
        {"name": "reference", "type": "str", "description": "The reference video (mp4)."},
        {"name": "n_samples", "type": "int", "description": "Frame pairs compared. Default 12."},
        {"name": "width", "type": "int", "description": "Comparison width in pixels. Default 160."},
        {"name": "height", "type": "int", "description": "Default 120."},
        {"name": "min_ssim", "type": "float", "description": "Pass threshold on the mean SSIM. Default 0.75."},
        {"name": "crop", "type": "list | None", "description": "[x0, y0, x1, y1] as fractions of the frame to compare only a region (both videos)."},
        {"name": "reference_crop", "type": "list | None", "description": "A crop for the reference only (a side-by-side reference whose simulation is one half)."},
        {"name": "reference_window_s", "type": "list | None", "description": "[start, end] seconds of the reference that correspond to the whole generated video (a twin of a longer recording)."},
        {"name": "out_png", "type": "str | None", "description": "Side-by-side sheet."},
        {"name": "out_json", "type": "str | None", "description": "Report."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok, issues, verdict, mean_ssim, min_ssim_pair, mean_abs_diff, motion_correlation, durations_s, per_pair [{t_frac, ssim, abs_diff}], threshold, unreadable_frames {video, reference}, out_png."},
    ],
    "requirements": ["numpy", "opencv-python", "scikit-image"],
    "simulators": [],
    "tags": ["reproducibility", "video", "similarity", "ssim", "audit"],
    "used_in": ["Battery Insertion", "Open the Bottle", "Giftbox Packing", "Computer Installation", "Drawing and Wiping", "Knot the Rope", "Meal Box (extra sample)", "Place Cup in Bowl", "Place Toy in Drawer", "Scissor Cutting", "Biology Experiment"],
    "status": "tested",
    "entrypoint": "reference_video_similarity",
}

import json
from pathlib import Path

import numpy as np


def _grey_frames(path, n, width, height, crop=None, window_s=None):
    """`n` frames at equal fractions of the video's length, grey and resized, plus their colour versions, the frame
    count, the fps and how many of the `n` frames could not be decoded (they are compared as black)."""
    from skimage.transform import resize
    from tool_library.common.video import video_info, read_video
    info = video_info(path)
    total, fps = int(info["frames"]), float(info["fps"] or 30.0)
    if total <= 0:
        allf = read_video(path); total = len(allf)
    else:
        allf = None
    lo, hi = 0, total - 1
    if window_s:                                  # only this part of the video is the task the other one shows
        lo = int(max(0, min(total - 1, round(float(window_s[0]) * fps)))); hi = int(max(lo, min(total - 1, round(float(window_s[1]) * fps))))
    idx = [min(total - 1, int(round(lo + f * (hi - lo)))) for f in np.linspace(0, 1, n)]
    unreadable = 0
    if allf is None:
        import cv2
        cap = cv2.VideoCapture(str(path)); frames = []
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, fr = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, i - 1)); ok, fr = cap.read()
            unreadable += 0 if ok else 1
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) if ok else np.zeros((info["height"], info["width"], 3), np.uint8))
        cap.release()
    else:
        frames = [allf[i] for i in idx]
    out, color = [], []
    for fr in frames:
        if crop:
            h, w = fr.shape[:2]; x0, y0, x1, y1 = crop
            fr = fr[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
        g = fr.astype(float).mean(axis=2) / 255.0 if fr.ndim == 3 else fr.astype(float) / 255.0
        out.append(resize(g, (height, width), anti_aliasing=True))
        color.append(resize(fr, (height, width), anti_aliasing=True, preserve_range=True).astype(np.uint8) if fr.ndim == 3 else None)
    return out, color, total, fps, unreadable


def _motion_curve(path, n=48, width=80, height=60, crop=None, window_s=None):
    g = _grey_frames(path, n, width, height, crop, window_s)[0]
    return np.array([np.abs(g[i + 1] - g[i]).mean() for i in range(len(g) - 1)])


def reference_video_similarity(video, reference, n_samples=12, width=160, height=120, min_ssim=0.75, crop=None, reference_crop=None,
                               reference_window_s=None, out_png=None, out_json=None):
    from skimage.metrics import structural_similarity as ssim
    a, ac, na, fa, ua = _grey_frames(video, int(n_samples), width, height, crop)
    b, bc, nb, fb, ub = _grey_frames(reference, int(n_samples), width, height, reference_crop or crop, reference_window_s)
    pairs = []
    for k, (ga, gb) in enumerate(zip(a, b)):
        s = float(ssim(ga, gb, data_range=1.0))
        pairs.append({"t_frac": round(k / max(len(a) - 1, 1), 3), "ssim": round(s, 4), "abs_diff": round(float(np.abs(ga - gb).mean()), 4)})
    ma = _motion_curve(video, crop=crop); mb = _motion_curve(reference, crop=reference_crop or crop, window_s=reference_window_s)
    corr = float(np.corrcoef(ma, mb)[0, 1]) if ma.std() > 1e-9 and mb.std() > 1e-9 else 0.0
    mean_ssim = float(np.mean([p["ssim"] for p in pairs])); worst = min(pairs, key=lambda p: p["ssim"])
    # a frame that cannot be decoded is compared as black: two undecodable videos would otherwise "match" perfectly
    issues = [f"{nm}: {k} of {len(pairs)} sampled frames could not be decoded ({path}); re-encode it (H.264 mp4) and re-run"
              for nm, k, path in (("video", ua, video), ("reference", ub, reference)) if k]
    ok = mean_ssim >= float(min_ssim) and not issues
    verdict = (f"{'MATCHES' if ok else 'DIFFERS FROM'} the reference: mean SSIM {mean_ssim:.3f} over {len(pairs)} matched times "
               f"(worst {worst['ssim']:.3f} at t={worst['t_frac']}), motion correlation {corr:.2f}; durations {na / fa:.1f} s vs {nb / fb:.1f} s"
               + (f"; UNREADABLE frames: {ua} in the video, {ub} in the reference" if issues else ""))
    rep = {"ok": bool(ok), "issues": issues, "verdict": verdict, "mean_ssim": round(mean_ssim, 4), "min_ssim_pair": worst, "mean_abs_diff": round(float(np.mean([p["abs_diff"] for p in pairs])), 4),
           "motion_correlation": round(corr, 3), "durations_s": {"video": round(na / fa, 2), "reference": round(nb / fb, 2)}, "per_pair": pairs, "threshold": float(min_ssim),
           "unreadable_frames": {"video": int(ua), "reference": int(ub)}}
    if out_png:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        n = len(pairs); fig, axs = plt.subplots(2, n, figsize=(1.6 * n, 3.2), squeeze=False)
        for k in range(n):
            axs[0, k].imshow(bc[k] if bc[k] is not None else b[k], cmap=None if bc[k] is not None else "gray"); axs[0, k].set_title(f"t={pairs[k]['t_frac']}", fontsize=7)
            axs[1, k].imshow(ac[k] if ac[k] is not None else a[k], cmap=None if ac[k] is not None else "gray"); axs[1, k].set_title(f"ssim {pairs[k]['ssim']:.2f}", fontsize=7)
            for r in (0, 1):
                axs[r, k].axis("off")
        axs[0, 0].text(-0.1, 0.5, "reference", transform=axs[0, 0].transAxes, rotation=90, va="center", ha="right", fontsize=8)
        axs[1, 0].text(-0.1, 0.5, "generated", transform=axs[1, 0].transAxes, rotation=90, va="center", ha="right", fontsize=8)
        fig.suptitle(verdict, fontsize=8); fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig); rep["out_png"] = str(out_png)
    if out_json:
        Path(out_json).write_text(json.dumps(rep, indent=1)); rep["out_json"] = str(out_json)
    return rep
