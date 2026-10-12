"""measure_object_color on small objects: the mask is eroded in proportion to its size before sampling.

A 28 mm ring is a few dozen pixels across; its SAM mask spills a few pixels onto the table and the edge pixels are
mixed. With a fixed two-pixel erosion those border pixels dominated (and the darkest-third trim then dropped the
object's own dark pixels), so the palette came out table-tinted. These cases draw synthetic frames (a dark object on a
light table, blurred, with a mask that spills 3 px) and compare the palette with the object's true colour.

    python tool_library/tests/test_measure_object_color.py
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.skills.measure_object_color import core_mask, measure_object_color  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dexagent_color_", dir=os.environ.get("TMPDIR")))
FAILS = 0
TABLE = np.array([0.66, 0.52, 0.36])


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def scene(tag, radius, colour, spill=3, size=(480, 640), mask_scale=1.0, blur=2.0):
    """frame00000.jpg with a disc of `colour` (radius px) on the table, blurred; mask = the disc grown by `spill`."""
    d = TMP / tag; d.mkdir()
    H, W = size
    yy, xx = np.mgrid[:H, :W]
    r = np.hypot(yy - H / 2, xx - W / 2)
    img = np.ones((H, W, 3)) * TABLE
    img[r <= radius] = colour
    img = cv2.GaussianBlur(img, (0, 0), blur)
    cv2.imwrite(str(d / "frame00000.jpg"), (img[..., ::-1] * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95])
    m = (r <= radius + spill).astype(np.uint8) * 255
    if mask_scale != 1.0:
        m = cv2.resize(m, (int(W * mask_scale), int(H * mask_scale)), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(d / "mask.png"), m)
    return d


def dominant(pal):
    return np.asarray(pal[0]["rgb"], float)


def mean_rgb(pal):
    return sum(np.asarray(p["rgb"]) * p["weight"] for p in pal) / sum(p["weight"] for p in pal)


# 1. a dark ring 32 px across at 640x480 (the EgoDex case): the old fixed erosion is table-tinted, the adaptive one is not
dark = np.array([0.16, 0.13, 0.11])
d = scene("ring", 16, dark)
new = measure_object_color(d, d / "mask.png")
old = measure_object_color(d, d / "mask.png", erode_frac=0.0, min_erode_px=2, max_erode_px=2)      # the former fixed 2-px erosion
e_new, e_old = np.linalg.norm(mean_rgb(new) - dark), np.linalg.norm(mean_rgb(old) - dark)
check("small dark object: adaptive erosion reads the object, the fixed one the table", e_new < 0.06 and e_old > 0.12,
      f"error {e_new:.3f} (adaptive) vs {e_old:.3f} (fixed 2 px); adaptive {mean_rgb(new).round(3)}")

# 2. a red part 24 px across with its mask at half resolution (eroded at the image's resolution)
red = np.array([0.72, 0.14, 0.12])
d = scene("red_half", 12, red, mask_scale=0.5)
pal = measure_object_color(d, d / "mask.png")
check("half-resolution mask of a small red part", np.linalg.norm(mean_rgb(pal) - red) < 0.08, f"{mean_rgb(pal).round(3)}")

# 3. a big object (240 px across): unchanged within a few hundredths of the former result
green = np.array([0.20, 0.55, 0.25])
d = scene("big", 120, green)
big_new, big_old = measure_object_color(d, d / "mask.png"), measure_object_color(d, d / "mask.png", erode_frac=0.0, min_erode_px=2, max_erode_px=2)
check("big object: same colour as before", np.linalg.norm(dominant(big_new) - green) < 0.05 and np.linalg.norm(mean_rgb(big_new) - mean_rgb(big_old)) < 0.05,
      f"{dominant(big_new).round(3)} vs former {dominant(big_old).round(3)}")

# 4. a tiny object (16 px across): the erosion relaxes instead of leaving nothing
d = scene("tiny", 8, dark, spill=2)
try:
    pal = measure_object_color(d, d / "mask.png"); ok = np.linalg.norm(mean_rgb(pal) - dark) < 0.15; det = f"{mean_rgb(pal).round(3)}"
except RuntimeError as e:
    ok, det = False, str(e)
check("tiny object: still measured, close to its colour", ok, det)

# 5. core_mask itself: depth adapts and is capped
disc = np.hypot(*np.mgrid[-60:61, -60:61]) <= 50
ring = disc & ~(np.hypot(*np.mgrid[-60:61, -60:61]) <= 30)
c_disc, c_ring = core_mask(disc), core_mask(ring)
check("core_mask: a 50 px disc loses its 12 px rim (capped), a 20 px-thick annulus a proportional rim",
      abs(np.sqrt(c_disc.sum() / np.pi) - 38) <= 2 and 0 < c_ring.sum() < 0.6 * ring.sum(), f"disc core r={np.sqrt(c_disc.sum() / np.pi):.1f}, ring kept {c_ring.sum() / ring.sum():.2f}")

print("ALL PASS" if not FAILS else f"{FAILS} FAILED")
sys.exit(1 if FAILS else 0)
