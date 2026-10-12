"""Smoke test for articulation_axis_semantics_check: joints of the described kind and direction pass; wrong types, directions, hinge sides and screw axes fail."""
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

import yaml

from tool_library.common import rigcam as RC
from tool_library.verifiers.articulation_axis_semantics_check import articulation_axis_semantics_check as check_art, parse_cues


def scene(name, body, key=""):
    p = TMP / f"{name}.xml"
    p.write_text(f'<mujoco><worldbody><geom name="floor" type="plane" size="1 1 0.1"/>{body}</worldbody>'
                 + (f'<keyframe><key qpos="{key}"/></keyframe>' if key else "") + "</mujoco>")
    return str(p)


def sem(oid, jt, axis):
    p = TMP / f"sem_{oid}_{abs(hash(axis))}.json"
    p.write_text(json.dumps({"objects": [{"id": oid, "name": oid, "type": "articulated", "role": "container", "properties": ["x"], "evidence": "x",
                                          "articulation": {"moving_part": "tray", "fixed_part": "housing", "joint_type": jt, "axis": axis, "state_over_video": "closed -> open"}}]}))
    return str(p)


def drawer(axis="0 -1 0"):
    return ('<body name="drawer" pos="0 0.6 0.05"><freejoint/><geom name="housing" type="box" size="0.08 0.15 0.05" mass="0.5"/>'
            f'<body name="drawer_tray" pos="0 0 0.0"><joint name="drawer_slide" type="slide" axis="{axis}" range="0 0.2"/>'
            '<geom name="tray" type="box" size="0.07 0.14 0.03" mass="0.1" contype="0" conaffinity="0"/></body></body>')


check("cue parsing reads the Stage 1 words", parse_cues("along the housing's long axis, toward the operator (down and left in the image)")
      == ["image_down", "image_left", "long_axis", "open_toward_operator"] and parse_cues("the bottle's long axis (vertical when standing; left-right while held)") == ["long_axis"],
      parse_cues("along the housing's long axis, toward the operator (down and left in the image)"))
cfg = yaml.safe_load((SETUP / "setup.yaml").read_text())
e = cfg["cameras"]["ego"]
cam = RC.camera_from_mujoco(640, 480, e["fovy_deg"], e["pos"], e["quat_wxyz"])
s_ok = sem("drawer", "prismatic", "along the housing's long axis, toward the operator (down in the image)")
r = check_art(scene("drawer_ok", drawer()), s_ok, camera=cam)
check("a drawer sliding out toward the operator along its long axis passes", r["ok"] and r["articulations"][0]["verdict"] == "ok", r["articulations"])
r = check_art(scene("drawer_away", drawer("0 1 0")), s_ok, camera=cam)
check("a drawer that opens away from the operator is caught", (not r["ok"]) and r["articulations"][0]["verdict"] == "wrong direction", r["issues"])
r = check_art(scene("drawer_side", drawer("1 0 0")), s_ok, camera=cam)
check("a drawer sliding sideways (across its long axis) is caught", not r["ok"], r["issues"])
HINGED = ('<body name="box" pos="0 0.6 0.03"><freejoint/><geom name="base" type="box" size="0.1 0.08 0.03" mass="0.3"/>'
          '<body name="box_lid" pos="0 {y} 0.03"><joint name="lid_hinge" type="hinge" axis="{ax}" range="0 2.5"/>'
          '<geom name="lid" type="box" size="0.1 0.08 0.004" pos="0 {gy} 0.004" mass="0.05"/></body></body>')
s_lid = sem("box", "revolute", "horizontal, left-right along the top of the base's back wall")
r = check_art(scene("lid_back", HINGED.format(y=0.08, ax="-1 0 0", gy=-0.08)), s_lid)
check("a lid hinged along the back wall, left-right, passes", r["ok"], r["articulations"])
r = check_art(scene("lid_front", HINGED.format(y=-0.08, ax="1 0 0", gy=0.08)), s_lid)
check("a lid hinged on the front wall is caught", (not r["ok"]) and "front" in r["issues"][0], r["issues"])
r = check_art(scene("lid_vertical", HINGED.format(y=0.08, ax="0 0 1", gy=-0.08)), s_lid)
check("a lid on a vertical hinge is caught", not r["ok"], r["issues"])
r = check_art(scene("wrong_type", HINGED.format(y=0.08, ax="-1 0 0", gy=-0.08)), sem("box", "prismatic", "toward the operator"))
check("a hinge where Stage 1 says prismatic is caught", (not r["ok"]) and r["articulations"][0]["verdict"] == "wrong joint type", r["issues"])
SCREW = ('<body name="bottle" pos="0 0.6 0.08"><freejoint/><geom name="bottle_g" type="cylinder" size="0.03 0.08" mass="0.3"/>'
         '<body name="bottle_carrier" pos="0 0 0.09"><joint name="yaw" type="hinge" axis="{a}"/><joint name="lift" type="slide" axis="{a}" range="0 0.02"/>'
         '<geom name="cap" type="cylinder" size="0.015 0.01" mass="0.01"/></body></body>')
s_screw = sem("bottle", "screw", "the bottle's vertical axis")
r = check_art(scene("screw_ok", SCREW.format(a="0 0 1")), s_screw)
check("a screw cap on a vertical hinge + slide passes", r["ok"], r["articulations"])
r = check_art(scene("screw_bad", SCREW.format(a="1 0 0")), s_screw)
check("a screw cap on a horizontal axis is caught", not r["ok"], r["issues"])

print(f"{'OK' if fails == 0 else 'FAILED'}: articulation_axis_semantics_check ({fails} failure(s))")
sys.exit(1 if fails else 0)
