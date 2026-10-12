"""Smoke test for collision_visual_alignment_check: covered visuals pass; colliders smaller than their visuals fail; thin wires and enclosed decorations do not."""
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


from tool_library.verifiers.collision_visual_alignment_check import collision_visual_alignment_check as check_cva


def obj(vis_half, col_half, extra=""):
    return (f'<body name="block" pos="0 0.5 0.03"><freejoint/>'
            f'<geom name="block_col" type="box" size="{col_half} {col_half} {col_half}" group="3" mass="0.1"/>'
            f'<geom name="block_vis" type="box" size="{vis_half} {vis_half} {vis_half}" contype="0" conaffinity="0" mass="0" rgba="0.2 0.6 0.2 1"/>'
            f'{extra}</body>'), "0 0.5 0.03 1 0 0 0"


r = check_cva(bench("same", *obj(0.03, 0.03)))
check("a visual box covered by its collision box passes", r["ok"] and r["geoms"][0]["max_out_m"] <= 0.005, r["geoms"])
r = check_cva(bench("small_col", *obj(0.03, 0.013)))
check("a collision box 17 mm smaller than the visual is caught", (not r["ok"]) and r["geoms"][0]["verdict"] == "exposed" and r["worst"] > 0.015, r["geoms"])
r = check_cva(bench("slightly", *obj(0.03, 0.028)))
check("a 2 mm thinner collider is within tolerance", r["ok"], r["geoms"])
wire = '<geom type="capsule" fromto="0.03 0 0 0.12 0 0" size="0.0008" contype="0" conaffinity="0" mass="0" rgba="0 0 0 1"/>'
plug = '<geom type="box" pos="0.15 0 0" size="0.004 0.003 0.003" contype="0" conaffinity="0" mass="0" rgba="1 1 1 1"/>'
inner = '<geom type="box" size="0.01 0.01 0.01" contype="0" conaffinity="0" mass="0" rgba="0 0 1 1"/>'
r = check_cva(bench("decor", *obj(0.03, 0.03, wire + plug + inner)))
check("a thin wire is skipped, a small loose plug is only warned about, an enclosed part passes",
      r["ok"] and len(r["warnings"]) == 1 and any("thin" in s["why"] for s in r["skipped"]), (r["issues"], r["warnings"], r["skipped"]))
mesh_cup = ('<body name="cup" pos="0.2 0.5 0.0"><freejoint/>'
            '<geom name="cup_col" type="cylinder" size="0.02 0.03" pos="0 0 0.03" group="3" mass="0.05"/>'
            '<geom name="cup_vis" type="cylinder" size="0.03 0.03" pos="0 0 0.03" contype="0" conaffinity="0" mass="0" rgba="0.1 0.1 0.8 1"/></body>')
r = check_cva(bench("cup", mesh_cup, "0.2 0.5 0 1 0 0 0"))
check("a cup whose collider is 10 mm narrower than its visual is caught", (not r["ok"]) and r["geoms"][0]["object"] == "cup", r["issues"])

print(f"{'OK' if fails == 0 else 'FAILED'}: collision_visual_alignment_check ({fails} failure(s))")
sys.exit(1 if fails else 0)
