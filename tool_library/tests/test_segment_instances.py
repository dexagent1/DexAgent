"""segment_with_text_prompts(instances=True): one mask per instance besides the union mask.

Several same-coloured bricks came back as one union mask per prompt, and the agent needed a detect-then-track detour
to get one mask per brick. These cases drive the skill with a stand-in for the SAM 3 processor (three bricks and a
hand at known places), so they run without the gated model; the default output must stay exactly the union masks.

    python tool_library/tests/test_segment_instances.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.skills.segment_with_text_prompts import segment_with_text_prompts  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dexagent_seg_", dir=os.environ.get("TMPDIR")))
FAILS = 0
H, W = 48, 64
BOXES = {"red brick": [(4, 4, 14, 12, 0.61), (30, 6, 40, 14, 0.93), (50, 30, 60, 40, 0.77)], "hand": [(20, 24, 44, 46, 0.88)]}


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


class FakeSam3:
    """set_image / set_text_prompt like sam3's Sam3Processor: masks [N, 1, h, w], scores [N], boxes [N, 4]."""
    def __init__(self, as_torch=False):
        self.as_torch = as_torch

    def set_image(self, img):
        return {"size": img.size}

    def set_text_prompt(self, prompt, state):
        w, h = state["size"]; sx, sy = w / W, h / H
        boxes = BOXES.get(prompt, [])
        m = np.zeros((len(boxes), 1, h, w), bool)
        for i, (x0, y0, x1, y1, _) in enumerate(boxes):
            m[i, 0, int(y0 * sy):int(y1 * sy), int(x0 * sx):int(x1 * sx)] = True
        out = {"masks": m, "scores": np.array([b[4] for b in boxes]), "boxes": np.array([b[:4] for b in boxes], float).reshape(-1, 4)}
        if self.as_torch:
            import torch
            out = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in out.items()}
        return out


frames = TMP / "frames"; frames.mkdir()
for t in range(2):
    cv2.imwrite(str(frames / f"frame{t:05d}.jpg"), np.full((H, W, 3), 128, np.uint8))
prompts = ["red brick", "hand"]
union_truth = np.zeros((H, W), bool)
for bl in BOXES.values():
    for x0, y0, x1, y1, _ in bl:
        union_truth[y0:y1, x0:x1] = True

# 1. default: union masks only, as before
meta = segment_with_text_prompts(frames, prompts, TMP / "union", processor=FakeSam3())
u = cv2.imread(str(TMP / "union" / "frame00000.png"), 0) > 0
check("default output unchanged: union masks, no instances", "instances" not in meta and not (TMP / "union" / "instances").exists()
      and np.array_equal(u, union_truth), f"meta keys {sorted(meta)}")

# 2. instances: one file per instance, best score first, the union file identical
meta = segment_with_text_prompts(frames, prompts, TMP / "inst", processor=FakeSam3(), instances=True)
rows = meta["instances"]["red brick"][0]
files_ok = all(Path(r["mask"]).exists() for p in prompts for f in meta["instances"][p] for r in f)
order_ok = [r["score"] for r in rows] == sorted([b[4] for b in BOXES["red brick"]], reverse=True)
first = cv2.imread(rows[0]["mask"], 0) > 0
x0, y0, x1, y1, _ = BOXES["red brick"][1]                                  # the 0.93 brick
check("instances: three bricks and a hand per frame, ranked by score, each its own mask",
      meta["n_instances"] == {"red brick": [3, 3], "hand": [1, 1]} and files_ok and order_ok and first.sum() == (x1 - x0) * (y1 - y0)
      and first[y0:y1, x0:x1].all() and rows[0]["box"] == [x0, y0, x1 - 1, y1 - 1] and rows[0]["area_px"] == int(first.sum()),
      json.dumps(meta["n_instances"]))
check("instances: the union mask is the same as without them",
      np.array_equal(cv2.imread(str(TMP / "inst" / "frame00001.png"), 0) > 0, union_truth))

# 3. detection at half resolution (and torch tensors, when torch is importable): instance masks at full resolution
try:
    import torch  # noqa: F401
    as_torch = True
except ImportError:
    as_torch = False
meta = segment_with_text_prompts(frames, prompts, TMP / "half", downscale=0.5, processor=FakeSam3(as_torch=as_torch), instances=True)
m0 = cv2.imread(meta["instances"]["hand"][0][0]["mask"], 0)
check(f"downscaled detection{' with torch tensors' if as_torch else ''}: instance masks come back at full resolution",
      m0.shape == (H, W) and abs(int((m0 > 0).sum()) - 24 * 22) <= 60, f"shape {m0.shape} area {(m0 > 0).sum()}")

print("ALL PASS" if not FAILS else f"{FAILS} FAILED")
sys.exit(1 if FAILS else 0)
