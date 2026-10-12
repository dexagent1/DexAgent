"""Text-prompted erase masks with SAM 3 (run as a subprocess by plate.py).

    python -m stage4.masks <frames_dir> <out_dir> "hand,arm,cup" [min_score]

For every frame, each prompt is segmented and all instances above the score
floor are merged into one binary mask: 255 = erase this pixel.
"""
from __future__ import annotations

import glob
import os
import sys

import cv2
import numpy as np


def main():
    frames_dir, out_dir, prompts = sys.argv[1], sys.argv[2], sys.argv[3]
    min_score = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
    prompts = [p.strip() for p in prompts.split(",") if p.strip()]
    files = sorted(glob.glob(os.path.join(frames_dir, "frame*.png")))
    assert files and prompts, f"nothing to segment ({len(files)} frames, {len(prompts)} prompts)"
    os.makedirs(out_dir, exist_ok=True)

    import torch
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf

    with torch.autocast("cuda", dtype=torch.bfloat16):   # bf16 inference, as SAM 3 runs it
        model = build_sam3_image_model(checkpoint_path=download_ckpt_from_hf(version="sam3.1"))
        processor = Sam3Processor(model, confidence_threshold=min_score)
        print(f"[masks] SAM 3 loaded; prompts {prompts} on {len(files)} frames", flush=True)

        cover = []
        for t, path in enumerate(files):
            img = Image.open(path).convert("RGB")
            state = processor.set_image(img)
            union = np.zeros((img.height, img.width), bool)
            for p in prompts:
                m = processor.set_text_prompt(prompt=p, state=state).get("masks")
                if m is None or m.shape[0] == 0:
                    continue
                m = m.any(dim=0).squeeze().detach().cpu().numpy().astype(bool)
                if m.shape != union.shape:
                    m = cv2.resize(m.astype(np.uint8), union.shape[::-1],
                                   interpolation=cv2.INTER_NEAREST) > 0
                union |= m
            cv2.imwrite(os.path.join(out_dir, os.path.basename(path)), union.astype(np.uint8) * 255)
            cover.append(float(union.mean()))
            if t % 20 == 0 or t == len(files) - 1:
                print(f"[masks] {t + 1}/{len(files)}", flush=True)
    print(f"[masks] mean erase coverage {np.mean(cover):.1%} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
