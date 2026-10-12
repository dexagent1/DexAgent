"""measure_rope on fragmented, coiled and crossing cords: join the pieces, flag what cannot be measured.

    python tool_library/tests/test_measure_rope.py

From the runs: a thin cord whose SAM mask came in 3 pieces measured 0.074 m (one piece), and a 2.5-turn coil
reported 0.48 m (real ~1.9 m) as if it were right. Synthetic masks of known cords on a table camera.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.common import tablecam as TC  # noqa: E402
from tool_library.skills.measure_rope import measure_rope  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dexagent_rope_meas_", dir=os.environ.get("TMPDIR")))
FAILS = 0
cam = TC.camera(focal_px=870, width=640, height=480, h_above_table_m=0.684, pitch_deg=36.0)


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def draw(poly, r, extra_px=0):
    uv = TC.project(poly, cam)
    mask = np.zeros((480, 640), np.uint8)
    for (u0, v0), (u1, v1), p in zip(uv[:-1], uv[1:], poly[:-1]):
        r_px = r * cam["focal_px"] / float(TC.optical_depth(p[None], cam)[0])
        cv2.line(mask, (int(round(u0)), int(round(v0))), (int(round(u1)), int(round(v1))), 1, max(2, int(round(2 * r_px))) + extra_px)
    return mask


def length(P):
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


# 1. a clean S-curve: measured as before and marked reliable
s = np.linspace(0, 1, 200)
sine = np.stack([0.70 + 0.25 * s, -0.10 + 0.15 * np.sin(2 * np.pi * s), np.full_like(s, 0.006)], axis=1)
r = measure_rope(draw(sine, 0.006) > 0, cam)
check("clean S-curve: reliable, length within 4%, radius within 2 mm, no crossings",
      r["reliable"] and r["ok"] and abs(r["length_m"] - length(sine)) / length(sine) < 0.04 and abs(r["radius_m"] - 0.006) < 0.002
      and r["self_crossings"] == 0, f"{r['length_m']} vs {length(sine):.4f}, r {r['radius_m']}, issues {r['issues']}")

# 2. a thin cord whose mask is in three pieces (two 18 px gaps): joined, bridged, length within 3%
s2 = np.linspace(0, 1, 300)
thin = np.stack([0.65 + 0.30 * s2, -0.15 + 0.10 * np.sin(3 * np.pi * s2), np.full_like(s2, 0.00125)], axis=1)
mt = draw(thin, 0.00125)
uv = TC.project(thin, cam)
for frac in (0.33, 0.66):
    c = uv[int(frac * len(uv))]
    cv2.circle(mt, (int(c[0]), int(c[1])), 9, 0, -1)
png = TMP / "thin_fragments.png"
cv2.imwrite(str(png), mt * 255)
r = measure_rope(str(png), cam)
check("3-piece mask (PNG path): all pieces joined across 2 bridged gaps, length within 3% (was one piece, 32%)",
      r["fragments"]["joined"] == 3 and len(r["bridged_gaps_px"]) == 2 and abs(r["length_m"] - length(thin)) / length(thin) < 0.03
      and r["reliable"], f"{r['length_m']} vs {length(thin):.4f}, fragments {r['fragments']}, gaps {r['bridged_gaps_px']}, {r['issues']}")

# 3. a 2.5-turn coil whose turns touch (blurred mask): flagged, not returned as a confident length
t = np.linspace(0, 1, 800)
rad = 0.035 + 0.0026 * 2.5 * t
coil = np.stack([0.80 + rad * np.cos(2 * np.pi * 2.5 * t), rad * np.sin(2 * np.pi * 2.5 * t), np.full_like(t, 0.00125)], 1)
mc = cv2.dilate(draw(coil, 0.0016), np.ones((3, 3), np.uint8))
r = measure_rope(mc > 0, cam)
check("coil with touching turns: reliable False, ok False, the issue says the length is a lower bound",
      not r["reliable"] and not r["ok"] and any("lower bound" in i for i in r["issues"]) and r["length_m"] < 0.5 * length(coil),
      f"length {r['length_m']} (true {length(coil):.3f}), coverage {r['coverage']}, issues {r['issues']}")

# 4. a cord crossing itself once (the curve (t^2, t^3 - t) crosses at t = +-1) is flagged with the crossing; a sharp
#    kink that does not cross (a '2' drawn with a cusp, whose thick mask has a pinhole) is not
t = np.linspace(-1.5, 1.5, 500)
fish = np.stack([0.70 + 0.1 * t ** 2, 0.06 * (t ** 3 - t), np.full_like(t, 0.003)], 1)
r = measure_rope(draw(fish, 0.003) > 0, cam)
check("a cord that crosses itself is flagged (self_crossings >= 1, reliable False, length a lower bound)",
      r["self_crossings"] >= 1 and not r["reliable"] and r["length_m"] < length(fish), f"{r['self_crossings']} {r['issues']}")
u = np.linspace(0, 1, 400)
two = np.stack([0.72 + 0.22 * u - 0.09 * np.sin(2 * np.pi * u), -0.05 + 0.07 * np.sin(2 * np.pi * u) * (1 - u) + 0.04 * u,
                np.full_like(u, 0.003)], 1)
r = measure_rope(draw(two, 0.003) > 0, cam)
check("a sharp kink without a crossing is not a crossing (reliable, length within 5%)",
      r["self_crossings"] == 0 and r["reliable"] and abs(r["length_m"] - length(two)) / length(two) < 0.05,
      f"{r['self_crossings']} {r['length_m']} vs {length(two):.4f} {r['issues']}")

# 5. a separate piece far off the cord's line is not glued on; the result says so
mx = draw(sine, 0.006)
cv2.line(mx, (40, 40), (40, 140), 1, 8)
r = measure_rope(mx > 0, cam)
check("an unrelated piece is not joined and is reported", r["fragments"]["joined"] == 1 and not r["reliable"]
      and any("could not be joined" in i for i in r["issues"]) and abs(r["length_m"] - length(sine)) / length(sine) < 0.04,
      f"{r['fragments']} {r['issues']}")

# 6. the end to start from, and a known radius
end = TC.project(thin[-1:], cam)[0]
r = measure_rope(mt > 0, cam, start_near_px=end.tolist(), radius_m=0.00125)
check("start_near_px puts tip a at that end; radius_m is used and the mask's estimate kept beside it",
      np.linalg.norm(np.array(r["tip_px"]["a"]) - end) < 15 and r["radius_m"] == 0.00125 and r["radius_measured_m"] > 0.00125,
      f"tip a {r['tip_px']['a']} vs {end.round(1).tolist()}, radius {r['radius_m']} measured {r['radius_measured_m']}")

print("ALL PASS" if not FAILS else f"{FAILS} FAILED")
sys.exit(1 if FAILS else 0)
