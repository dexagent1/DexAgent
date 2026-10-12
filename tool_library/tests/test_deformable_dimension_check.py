"""Smoke test for deformable_dimension_check: a rope at its measured length, radius and centreline passes; short, thick and misplaced ropes fail."""
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


import json

from tool_library.verifiers.deformable_dimension_check import deformable_dimension_check as check_dim


def rope(n=30, r=0.006, h=0.01, y=0.5):
    xml, close = "", ""
    for i in range(n):
        pos = f"-0.3 {y} {r}" if i == 0 else f"{2 * h} 0 0"
        j = '<freejoint/>' if i == 0 else '<joint type="ball"/>'
        xml += f'<body name="ropeB_{i}" pos="{pos}">{j}<geom type="capsule" size="{r} {h}" quat="0.7071068 0 0.7071068 0" mass="0.002"/>'
        close += "</body>"
    p = TMP / f"rope_{n}_{r}_{y}.xml"
    p.write_text(f'<mujoco><worldbody><geom name="floor" type="plane" size="1 1 0.1"/>{xml}{close}</worldbody></mujoco>')
    return str(p)


L = 30 * 0.02
meas = {"length_m": L, "radius_m": 0.006, "centreline_rig": [[-0.3 + 0.02 * i, 0.5, 0.006] for i in range(31)]}
mp = TMP / "measure.json"
mp.write_text(json.dumps(meas))
r = check_dim(rope(), str(mp))
check("a rope built from its measured centreline passes (length, radius, placement)", r["ok"] and abs(r["length_m"]["rel_err"]) < 0.02
      and r["centreline_mean_dist_m"] < 0.005, r)
r = check_dim(rope(n=20), str(mp))
check("a rope a third too short is caught", (not r["ok"]) and r["length_m"]["rel_err"] < -0.3, r["issues"])
r = check_dim(rope(r=0.012), {"length_m": L, "radius_m": 0.006})
check("a rope twice as thick is caught", (not r["ok"]) and r["radius_m"]["rel_err"] > 0.9, r["issues"])
r = check_dim(rope(y=0.65), str(mp))
check("a rope laid 15 cm off the measured centreline is caught", (not r["ok"]) and r["centreline_mean_dist_m"] > 0.1, r["issues"])
r = check_dim(rope(n=33), {"length_m": L})
check("a 10% longer rope (no radius given) is within tolerance", r["ok"], r)
r = check_dim(rope(), dict(meas, reliable=False, issues=["the centreline explains only 50% of the mask"]))
check("a measurement measure_rope marked unreliable fails the check with that reason", (not r["ok"])
      and any("unreliable" in i and "50%" in i for i in r["issues"]), r["issues"])
try:
    check_dim(rope(), {"radius_m": 0.006})
    check("a measurement without length_m raises ValueError", False)
except ValueError as e:
    check("a measurement without length_m raises ValueError naming it", "measurement" in str(e), e)

print(f"{'OK' if fails == 0 else 'FAILED'}: deformable_dimension_check ({fails} failure(s))")
sys.exit(1 if fails else 0)
