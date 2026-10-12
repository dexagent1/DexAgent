"""Smoke test for initial_keyframe_state_check: the setup home and resting objects pass; missing keyframes, off-home robots, objects or lids not at rest and welded parts off their mates fail."""
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


import math

from tool_library.verifiers.initial_keyframe_state_check import initial_keyframe_state_check as check_key

r = check_key(str(SETUP / "scene.xml"))
check("the setup's home keyframe passes", r["ok"] and r["home"]["compared"] >= 60 and not r["warnings"], r["issues"])
r = check_key(bench("cube", CUBE, CUBE_Q))
check("a cube resting on the table at keyframe 0 passes", r["ok"] and r["settle"]["cube"]["drift_m"] < 0.002, r["settle"])

# the hand-built cup example: the object starts tilted and settles tens of millimetres
c, s = math.cos(math.radians(10)), math.sin(math.radians(10))
tilted = CUBE.replace('pos="0 0.5 0.025"', 'pos="0 0.5 0.06"')
r = check_key(bench("tilted", tilted, f"0 0.5 0.06 {c:.6f} {s:.6f} 0 0"))
check("an object starting tilted in the air is caught as not at rest", (not r["ok"]) and r["settle"]["cube"]["drift_m"] > 0.02, r["settle"])
r = check_key(bench("nokey", CUBE, CUBE_Q, lambda x: re.sub(r"<keyframe>.*?</keyframe>", "", x, flags=re.S).replace('<size nkey="1" />', '')))
check("a scene without a keyframe is caught", (not r["ok"]) and r["keyframe"]["n"] == 0, r["issues"])


def finger_off(x):
    m = re.search(r'<key name="home" qpos="([^"]*)"', x)
    v = m.group(1).split()
    v[10] = "0.5"                                       # a right-hand finger joint off its home value
    return x.replace(m.group(1), " ".join(v), 1)


r = check_key(bench("finger", CUBE, CUBE_Q, finger_off))
check("a robot joint off the setup's home is caught", (not r["ok"]) and r["home"]["off"], r["issues"])


def stale_ctrl(x):
    m = re.search(r'ctrl="([^"]*)"', x)
    v = m.group(1).split()
    v[1] = str(float(v[1]) - 0.3)                       # the right wrist's y target from an old root frame
    return x.replace(m.group(1), " ".join(v), 1)


p = bench("stale", CUBE, CUBE_Q, stale_ctrl)
r = check_key(p)
check("a stale servo target in key ctrl is a warning, not a failure", r["ok"] and r["warnings"] and r["ctrl_mismatch"], r)
r = check_key(p, ctrl_must_match=True)
check("with ctrl_must_match it fails", not r["ok"], r["issues"])
LID = ('<body name="box" pos="0.1 0.6 0.02"><freejoint/><geom name="box_base" type="box" size="0.05 0.05 0.02" mass="0.2"/>'
       '<body name="box_lid" pos="0 0.05 0.02"><joint name="box_hinge" type="hinge" axis="-1 0 0" range="0 2.5" damping="0.001"/>'
       '<geom name="box_lid_g" type="box" size="0.05 0.05 0.003" pos="0 -0.05 0.003" mass="0.05"/></body></body>')
r = check_key(bench("lid_closed", LID, "0.1 0.6 0.02 1 0 0 0 0"))
check("a closed lid resting on its box passes", r["ok"], r["issues"])
r = check_key(bench("lid_up", LID, "0.1 0.6 0.02 1 0 0 0 1.2"))
check("a lid held half open in the keyframe that falls shut is caught", (not r["ok"]) and abs(r["joints"]["box_hinge"]) > 0.25, r["joints"])
WELD = ('<body name="part" pos="0.2 0.6 0.05"><freejoint/><geom name="part_g" type="box" size="0.01 0.01 0.01" mass="0.01"/></body>')
r = check_key(bench("snap", CUBE + WELD, CUBE_Q + " 0.2 0.6 0.05 1 0 0 0",
                    lambda x: x.replace("</equality>", '<weld body1="cube" body2="part" relpose="0 0 0.04 1 0 0 0" solref="0.0005"/>\n</equality>', 1)))
check("a part far from the weld that holds it (a snap at the first step) is caught", not r["ok"], r["issues"])

print(f"{'OK' if fails == 0 else 'FAILED'}: initial_keyframe_state_check ({fails} failure(s))")
sys.exit(1 if fails else 0)
