"""texture_material_resolution_check: present, decodable, reasonably sized textures pass; missing or corrupt files, oversized
textures and a 2D texture on a mesh without UVs fail.

    python tool_library/tests/test_texture_material_resolution_check.py
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.verifiers.texture_material_resolution_check import texture_material_resolution_check  # noqa: E402

fails = 0


def check(name, cond, msg=""):
    global fails
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + ("" if cond else f"  {msg}"))
    fails += 0 if cond else 1


tmp = Path(tempfile.mkdtemp(prefix="textures_"))
img = (np.indices((64, 64)).sum(0) % 16 < 8).astype(np.uint8) * 200
cv2.imwrite(str(tmp / "label.png"), np.dstack([img, 255 - img, img]))
(tmp / "broken.png").write_bytes(b"not a png at all")
V = "v 0 0 0\nv .04 0 0\nv .04 .04 0\nv 0 .04 0\nv 0 0 .04\nv .04 0 .04\nv .04 .04 .04\nv 0 .04 .04\n"
F = "f 1 3 2\nf 1 4 3\nf 5 6 7\nf 5 7 8\nf 1 2 6\nf 1 6 5\nf 2 3 7\nf 2 7 6\nf 3 4 8\nf 3 8 7\nf 4 1 5\nf 4 5 8\n"
(tmp / "box_nouv.obj").write_text(V + F)
VT = "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
FT = "".join(f"f {a}/{1 + i % 4} {b}/{1 + (i + 1) % 4} {c}/{1 + (i + 2) % 4}\n"
             for i, (a, b, c) in enumerate([tuple(map(int, l.split()[1:])) for l in F.strip().splitlines()]))
(tmp / "box_uv.obj").write_text(V + VT + FT)


def scene(name, tex="label.png", geom='type="box" size=".03 .03 .03"', mesh=None):
    m = f'<mesh name="m" file="{mesh}"/>' if mesh else ""
    xml = (f'<mujoco><asset><texture name="t" type="2d" file="{tex}"/><material name="mat" texture="t"/>{m}</asset>'
           f'<worldbody><body><freejoint/><geom {geom} material="mat"/></body></worldbody></mujoco>')
    p = tmp / f"{name}.xml"
    p.write_text(xml)
    return str(p)


# 1. the setup (builtin textures) and a box with a 64 px PNG label pass
r0 = texture_material_resolution_check(str(ROOT / "setup" / "scene.xml"))
r1 = texture_material_resolution_check(scene("good"))
check("setup and a box with a decodable 64 px texture pass", r0["ok"] and r1["ok"] and r1["textures"], f"{r0['issues']} {r1['issues']}")

# 2. a missing texture file, and a file that is not an image
r2 = texture_material_resolution_check(scene("missing", tex="nope.png"))
r3 = texture_material_resolution_check(scene("broken", tex="broken.png"))
check("a missing and a corrupt texture file fail", not r2["ok"] and not r3["ok"] and "does not exist" in " ".join(r2["issues"]),
      f"{r2['issues']} {r3['issues']}")

# 3. a texture over the size budget
r4 = texture_material_resolution_check(scene("big"), max_texture_px=32)
check("a 64 px texture over a 32 px budget fails", not r4["ok"] and "exceeds" in r4["issues"][0], str(r4["issues"]))

# 4. a 2D-textured material on a mesh without UVs fails (Stage 4 drops the texture); the mesh with UVs passes
r5 = texture_material_resolution_check(scene("nouv", geom='type="mesh" mesh="m"', mesh="box_nouv.obj"))
r6 = texture_material_resolution_check(scene("uv", geom='type="mesh" mesh="m"', mesh="box_uv.obj"))
check("a textured mesh without texture coordinates fails; with coordinates it passes",
      not r5["ok"] and r5["untextured_meshes"] and r6["ok"], f"{r5['issues']} {r6['issues']}")

sys.exit(1 if fails else 0)
