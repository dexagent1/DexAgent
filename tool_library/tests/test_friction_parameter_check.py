"""Smoke test for friction_parameter_check: plausible friction passes; sticky, torsional, slippery, over-cap pairs, damped free joints and a changed table fail."""
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


from tool_library.verifiers.friction_parameter_check import friction_parameter_check as check_fr

r = check_fr(bench("cube", CUBE, CUBE_Q))
check("a cube with default friction on the setup's table passes", r["ok"] and r["geoms_checked"] >= 2, r["issues"])
r = check_fr(bench("sticky", CUBE.replace('mass="0.1"', 'mass="0.1" friction="3.0 0.005 0.0001"'), CUBE_Q))
check("a 3.0 sliding friction is caught", (not r["ok"]) and r["bad_geoms"] and r["worst"] == 3.0, r["bad_geoms"])
r = check_fr(bench("torsion", CUBE.replace('mass="0.1"', 'mass="0.1" friction="1.0 0.5 0.0001"'), CUBE_Q))
check("a 0.5 m torsional friction is caught", (not r["ok"]) and "torsional" in r["bad_geoms"][0]["why"][0], r["bad_geoms"])
r = check_fr(bench("ice", CUBE.replace('mass="0.1"', 'mass="0.1" friction="0.001 0.005 0.0001"'), CUBE_Q))
check("an ice-like 0.001 sliding friction is caught", not r["ok"], r["bad_geoms"])
exec_pair = '<pair geom1="robot_right_collision_hand_right_index_2" geom2="cube_geom" condim="6" friction="1.2 1.2 0.02 0.001 0.001"/>'
r = check_fr(bench("execpair", CUBE, CUBE_Q, lambda x: x.replace("</contact>", exec_pair + "\n</contact>", 1)))
check("a hand-object pair with the executor's own friction passes", r["ok"] and r["pairs_checked"] == 1, r["issues"])
r = check_fr(bench("stickypair", CUBE, CUBE_Q, lambda x: x.replace("</contact>", exec_pair.replace('1.2 1.2 0.02', '4 4 0.02') + "\n</contact>", 1)))
check("a hand-object pair with friction 4 is caught", (not r["ok"]) and r["bad_pairs"], r["bad_pairs"])
r = check_fr(bench("spinpair", CUBE, CUBE_Q, lambda x: x.replace("</contact>", exec_pair.replace('0.02 0.001 0.001', '1.0 0.1 0.1') + "\n</contact>", 1)))
check("a hand-object pair with 1.0 m torsional friction is caught", (not r["ok"]) and r["bad_pairs"], r["bad_pairs"])
r = check_fr(bench("damped", CUBE.replace('<freejoint name="cube_free"/>', '<joint name="cube_free" type="free" damping="5"/>'), CUBE_Q))
check("a damped object free joint is caught", (not r["ok"]) and r["bad_joints"] and r["bad_joints"][0]["damping"] == 5.0, r["bad_joints"])
r = check_fr(bench("table", CUBE, CUBE_Q, lambda x: x.replace('<geom name="g_table" size="1.8 1.075 0.02" type="box"', '<geom name="g_table" size="1.8 1.075 0.02" type="box" friction="2 0.005 0"')))
check("a table with a changed friction is caught", (not r["ok"]) and "table geom g_table" in r["issues"][0], r["issues"])

print(f"{'OK' if fails == 0 else 'FAILED'}: friction_parameter_check ({fails} failure(s))")
sys.exit(1 if fails else 0)
