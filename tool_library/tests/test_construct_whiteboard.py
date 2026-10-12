"""construct_whiteboard's texture on a UV-mapped face mesh, with the panel box kept for InkSurface.

    MUJOCO_GL=egl python tool_library/tests/test_construct_whiteboard.py

Every whiteboard scene failed appearance_texture_check: the textured panel was a box, and Stage 4 drops a texture on
a box. The box now stays (same name, size, pose, contacts) collision-only in group 3, and a visual-only mesh
<name>_face with UVs carries the texture. The ink must still land where InkSurface writes it: the rendered marks of
the new board are compared with those of the old textured box.
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.common.ink import InkSurface  # noqa: E402
from tool_library.skills.construct_whiteboard import construct_whiteboard  # noqa: E402
from tool_library.verifiers.appearance_texture_check import appearance_texture_check  # noqa: E402
from tool_library.verifiers.ink_round_trip_test import test_ink_round_trip  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dexagent_board_", dir=os.environ.get("TMPDIR")))
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


for i, kw in enumerate([{}, dict(surface_w=0.6, surface_d=0.4, yaw_deg=25.0, pos=[0.1, 0.05, 0.0])]):
    b = construct_whiteboard(TMP / f"board{i}", **kw)
    m = mujoco.MjModel.from_xml_path(b["xml"])
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    pg, fg = m.geom(b["surface_geom"]), m.geom(b["face_geom"])
    tag = f"[{i}] {kw or 'defaults'}"

    # 1. the panel box is unchanged as a collision face; the face mesh is visual only and carries the texture
    check(f"{tag}: panel stays a colliding box, hidden in group 3, without a material",
          int(m.geom_type[pg.id]) == int(mujoco.mjtGeom.mjGEOM_BOX) and int(m.geom_group[pg.id]) == 3
          and int(m.geom_matid[pg.id]) < 0 and int(m.geom_contype[pg.id]) and int(m.geom_conaffinity[pg.id]))
    mid = int(m.geom_dataid[fg.id])
    check(f"{tag}: face is a visible, non-colliding mesh with UVs and the ink material, covering the panel",
          int(m.geom_type[fg.id]) == int(mujoco.mjtGeom.mjGEOM_MESH) and int(m.geom_group[fg.id]) == 0
          and int(m.geom_contype[fg.id]) == 0 and int(m.geom_conaffinity[fg.id]) == 0
          and int(m.mesh_texcoordadr[mid]) >= 0 and m.mat(int(m.geom_matid[fg.id])).name.endswith("_inkmat")
          and np.allclose(d.geom_xpos[fg.id], d.geom_xpos[pg.id], atol=1e-6))

    # 2. appearance_texture_check: the texture survives (it was dropped on the box)
    r = appearance_texture_check(b["xml"])
    check(f"{tag}: appearance_texture_check passes with no dropped texture", r["ok"] and not r["dropped_textures"],
          str(r["issues"])[:200])

    # 3. ink and wipe still round-trip, the rendered frame included
    r = test_ink_round_trip(b)
    check(f"{tag}: ink_round_trip_test passes (render check included)", r["ok"], r["verdict"][:110])

# 4. the face shows the ink where InkSurface wrote it: an L in the +x+y corner renders where the old textured box drew it
b = construct_whiteboard(TMP / "uv")
old_xml = Path(b["xml"]).with_name("whiteboard_oldbox.xml")
txt = Path(b["xml"]).read_text()
txt = re.sub(r'\s*<geom name="whiteboard_face"[^>]*/>', "", txt, flags=re.S)
txt = txt.replace('group="3"\n            rgba=', 'material="whiteboard_inkmat"\n            rgba=', 1)
old_xml.write_text(txt)


def render_L(xml):
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    ink = InkSurface(m, b)
    c = d.geom_xpos[ink.gid]
    z = float(c[2]) + ink.half_t
    for f in np.linspace(0.0, 1.0, 60):
        ink.stroke(d, [0.60 * ink.hx, (0.60 - 0.55 * f) * ink.hy, z], radius_m=0.004, key="a")
    for f in np.linspace(0.0, 1.0, 60):
        ink.stroke(d, [(0.60 - 0.45 * f) * ink.hx, 0.05 * ink.hy, z], radius_m=0.004, key="b")
    cam = mujoco.MjvCamera()
    cam.lookat[:] = c
    cam.distance, cam.azimuth, cam.elevation = 0.9, 90.0, -89.9
    with mujoco.Renderer(m, 360, 480) as r:
        ink.upload(r)
        r.update_scene(d, camera=cam)
        return r.render().mean(2) < 90


new_dark, old_dark = render_L(b["xml"]), render_L(old_xml)
iou = float((new_dark & old_dark).sum() / max((new_dark | old_dark).sum(), 1))


def bbox_centroid(a):
    ys, xs = np.nonzero(a)
    return np.array([xs.min(), xs.max(), ys.min(), ys.max()]), np.array([xs.mean(), ys.mean()])


(bn, cn), (bo, co) = bbox_centroid(new_dark), bbox_centroid(old_dark)
# texture filtering differs by a texel between a box face and a mesh, so a stroke edge can gain a pixel row; a
# mirrored or shifted mapping would move the L's corner and arms by tens of pixels
check("the face mesh draws the L where the textured box did (same extent within 1 px, centroid within 4 px, IoU > 0.8)",
      new_dark.sum() > 200 and np.abs(bn - bo).max() <= 1 and np.linalg.norm(cn - co) < 4.0 and iou > 0.8,
      f"IoU {iou:.3f}, bbox {bn.tolist()} vs {bo.tolist()}, centroid offset {np.linalg.norm(cn - co):.2f} px")

print("ALL PASS" if not FAILS else f"{FAILS} FAILED")
sys.exit(1 if FAILS else 0)
