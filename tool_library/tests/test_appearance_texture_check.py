"""Smoke test for appearance_texture_check: coloured and UV-textured objects pass; default grey, invisible objects, dropped textures and missing or unreadable files fail."""
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SETUP = ROOT / "setup"
TMP = Path(tempfile.mkdtemp())
fails = 0


def check(name, cond, msg=""):
    global fails
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + ("" if cond else f"  {str(msg)[:600]}"))
    fails += 0 if cond else 1


def bench(name, objects="", key_extra="", edit=None):
    """The setup's bench (setup/scene.xml) with objects added to the worldbody and their free-joint qpos appended to
    keyframe 'home', written to a temp file (meshes resolved from setup/)."""
    xml = (SETUP / "scene.xml").read_text()
    xml = xml.replace('<compiler angle="radian" />', f'<compiler angle="radian" meshdir="{SETUP}" texturedir="{SETUP}" />', 1)
    xml = xml.replace("</worldbody>", objects + "\n  </worldbody>", 1)
    if key_extra:
        xml = re.sub(r'(<key name="home" qpos=")([^"]*)', lambda m: m.group(1) + m.group(2) + " " + key_extra, xml, count=1)
    if edit:
        xml = edit(xml)
    p = TMP / f"{name}.xml"
    p.write_text(xml)
    return str(p)


CUBE = ('<body name="cube" pos="0 0.5 0.025"><freejoint name="cube_free"/>'
        '<geom name="cube_geom" type="box" size="0.025 0.025 0.025" mass="0.1" rgba="0.8 0.2 0.2 1"/></body>')
CUBE_Q = "0 0.5 0.025 1 0 0 0"


import numpy as np
from PIL import Image

from tool_library.verifiers.appearance_texture_check import appearance_texture_check as check_app

# a textured quad mesh with UVs and a PNG next to the scene
Image.fromarray(np.full((8, 8, 3), (200, 40, 40), np.uint8)).save(TMP / "red.png")
(TMP / "card.obj").write_text("v -0.03 -0.02 0\nv 0.03 -0.02 0\nv 0.03 0.02 0\nv -0.03 0.02 0\nv 0 0 0.002\n"
                              "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvt 0.5 0.5\nf 1/1 2/2 3/3\nf 1/1 3/3 4/4\nf 1/1 2/2 5/5\nf 2/2 3/3 5/5\n")
(TMP / "card_nouv.obj").write_text("v -0.03 -0.02 0\nv 0.03 -0.02 0\nv 0.03 0.02 0\nv -0.03 0.02 0\nv 0 0 0.002\n"
                                   "f 1 2 3\nf 1 3 4\nf 1 2 5\nf 2 3 5\n")


def textured(name, mesh_file, tex_file="red.png", prim=False):
    asset = (f'<asset><texture name="{name}_tex" type="2d" file="{TMP / tex_file}"/><material name="{name}_mat" texture="{name}_tex"/>'
             f'<mesh name="{name}_mesh" file="{TMP / mesh_file}"/></asset>')
    geom = (f'<geom name="{name}_g" type="box" size="0.03 0.02 0.002" material="{name}_mat" mass="0.01"/>' if prim else
            f'<geom name="{name}_g" type="mesh" mesh="{name}_mesh" material="{name}_mat" mass="0.01"/>')
    body = f'<body name="{name}" pos="0 0.5 0.01"><freejoint/>{geom}</body>'
    return bench(name, body, "0 0.5 0.01 1 0 0 0", lambda x: x.replace("</mujoco>", asset + "\n</mujoco>"))


r = check_app(bench("cube", CUBE, CUBE_Q))
check("a coloured cube passes", r["ok"] and r["objects"][0]["colours"], r["issues"])
r = check_app(bench("grey", CUBE.replace(' rgba="0.8 0.2 0.2 1"', ''), CUBE_Q))
check("a cube left in MuJoCo's default grey is caught", (not r["ok"]) and r["objects"][0]["verdict"] == "default grey", r["issues"])
r = check_app(textured("card", "card.obj"))
check("a textured mesh with UV coordinates passes", r["ok"] and r["objects"][0]["textured"] and r["files"]["textures"] == 1, r["issues"])
r = check_app(textured("card_nouv", "card_nouv.obj"))
check("a texture on a mesh without UVs (dropped by Stage 4) is caught", (not r["ok"]) and "texture coordinates" in r["dropped_textures"][0]["why"], r["issues"])
r = check_app(textured("card_box", "card.obj", prim=True))
check("a texture on a box primitive (dropped by Stage 4) is caught", (not r["ok"]) and "primitive" in r["dropped_textures"][0]["why"], r["issues"])
r = check_app(textured("card_missing", "card.obj", tex_file="missing.png"))
check("a missing texture file is reported, not raised", (not r["ok"]) and r["files"]["missing"], r["issues"][:2])
(TMP / "broken.png").write_bytes(b"not a png")
r = check_app(textured("card_broken", "card.obj", tex_file="broken.png"))
check("an unreadable texture file is reported", (not r["ok"]) and r["files"]["unreadable"], r["issues"][:2])
r = check_app(bench("invisible", CUBE.replace(' rgba="0.8 0.2 0.2 1"', ' rgba="0.8 0.2 0.2 0"'), CUBE_Q))
check("an invisible object is caught", (not r["ok"]) and r["objects"][0]["verdict"] == "invisible", r["issues"])

print(f"{'OK' if fails == 0 else 'FAILED'}: appearance_texture_check ({fails} failure(s))")
sys.exit(1 if fails else 0)
