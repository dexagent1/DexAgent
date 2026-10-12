"""Segmentation renders of scenes with hundreds of geoms (common.render.segmentation_renderer).

    MUJOCO_GL=egl python tool_library/tests/test_segmentation_render.py

A segmentation image encodes geom ids as colours; MuJoCo's default multisampled offscreen buffer averages them at
every silhouette edge, which above ~255 geoms produces ids past the end of the renderer's lookup table
(IndexError in segid2output, reached from render_body_masks in fit_object_placement and silhouette_iou_check) and
below it silently paints a third geom's id along the edges. The scene here has 441 small rotated boxes in a random id
order seen from above, so neighbouring pixels belong to geoms with very different ids.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.common import scene as S  # noqa: E402
from tool_library.common.render import Offscreen, segmentation_renderer  # noqa: E402

FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def many_box_scene(n_side=21, seed=0):
    rng = np.random.default_rng(seed)
    cells = [(i, j) for i in range(n_side) for j in range(n_side)]
    bodies, centres = [], {}
    for k in rng.permutation(len(cells)):
        i, j = cells[k]
        x, y = -0.25 + 0.025 * i, -0.25 + 0.025 * j
        centres[f"b{i:02d}_{j:02d}"] = (x, y)
        bodies.append(f'<body name="b{i:02d}_{j:02d}" pos="{x:.4f} {y:.4f} 0.02"><geom type="box" size="0.0125 0.0125 0.01" '
                      f'euler="0 0 {int(k) * 7}" rgba="0.5 0.5 0.5 1"/></body>')
    xml = (f'<mujoco><worldbody><light pos="0 0 2"/><geom name="floor" type="plane" size="1 1 0.1"/>{"".join(bodies)}'
           f'<camera name="top" pos="0 0 0.9"/></worldbody></mujoco>')
    return mujoco.MjModel.from_xml_string(xml), centres


m, centres = many_box_scene()
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
check("scene has more than 400 geoms", m.ngeom > 400, f"ngeom {m.ngeom}")

# the failure this guards against: MuJoCo's own multisampled render yields ids beyond the scene's geom count
with mujoco.Renderer(m, 480, 640) as r:
    r.update_scene(d, camera="top")
    r.scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = True
    r.scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = True
    raw = r.render().astype(np.int64)
    n_scene = int(r.scene.ngeom)
ids = raw[..., 0] + 256 * raw[..., 1] + 65536 * raw[..., 2]
print(f"  (multisampled render: {int((ids > n_scene).sum())} pixels with an id past the {n_scene} scene geoms)")

# 1. render_body_masks no longer crashes and every box's mask sits where the box is
try:
    masks = S.render_body_masks(m, d, 640, 480, camera="top")
    err = None
except Exception as e:                                   # noqa: BLE001 - the regression is the exception
    masks, err = {}, repr(e)
check("render_body_masks on a 441-box scene returns masks instead of raising", err is None, err or "")
f = 240.0 / np.tan(np.radians(45.0) / 2)                # MuJoCo's default fovy 45, camera 0.9 m above, boxes 3 cm tall
off = []
for name, (x, y) in centres.items():
    mk = masks.get(name)
    if mk is None or not mk.any():
        off.append((name, "missing"))
        continue
    vs, us = np.nonzero(mk)
    u0, v0 = 320 + f * x / 0.87, 240 - f * y / 0.87
    if abs(us.mean() - u0) > 4 or abs(vs.mean() - v0) > 4 or us.max() - us.min() > 0.04 * f / 0.87:
        off.append((name, round(float(us.mean() - u0), 1), round(float(vs.mean() - v0), 1)))
check("every one of the 441 boxes has its own mask centred on it, none smeared by blended ids", not off, str(off[:5]))
claimed = sum(int(v.sum()) for v in masks.values())
check("masks partition the image (no pixel claimed twice or dropped)", claimed == 480 * 640, f"{claimed} of {480 * 640}")

# 2. the renderer helper leaves the model's anti-aliasing as it was
before = int(m.vis.quality.offsamples)
with segmentation_renderer(m, 240, 320) as r:
    r.update_scene(d, camera="top")
    seg = r.render()
check("segmentation_renderer restores the model's offsamples", int(m.vis.quality.offsamples) == before and before > 0,
      f"{before} -> {int(m.vis.quality.offsamples)}")
check("segmentation_renderer ids are all valid geoms", int(seg[..., 0].max()) < m.ngeom and int(seg[..., 0].min()) >= -1)

# 3. Offscreen.segmentation (used by the stage tools) gives valid body ids and RGB still renders
off_r = Offscreen(m, 640, 480)
try:
    body = off_r.segmentation(d, camera="top")
    rgb = off_r.rgb(d, camera="top")
    ok = int(body.max()) < m.nbody and rgb.shape == (480, 640, 3) and len(np.unique(body)) == len(centres) + 1
    detail = f"{len(np.unique(body))} body ids"
except Exception as e:                                   # noqa: BLE001
    ok, detail = False, repr(e)
finally:
    off_r.close()
check("Offscreen.segmentation on the 441-box scene: one id per box plus the floor", ok, detail)

print("ALL PASS" if not FAILS else f"{FAILS} FAILED")
sys.exit(1 if FAILS else 0)
