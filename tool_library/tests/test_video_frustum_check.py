"""Smoke test for video_frustum_check: objects on the work area pass; objects behind the camera, off-screen or missing fail."""
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
from tool_library.verifiers.video_frustum_check import video_frustum_check as check_fr

cfg = yaml.safe_load((SETUP / "setup.yaml").read_text())
e = cfg["cameras"]["ego"]
cam = RC.camera_from_mujoco(640, 480, e["fovy_deg"], e["pos"], e["quat_wxyz"], extra={"source": "setup"})
cam_p = TMP / "camera.json"
RC.save(cam, cam_p)
sem = TMP / "sem.json"
sem.write_text(json.dumps({"objects": [{"id": "cube", "name": "red cube", "role": "manipulated"}]}))

r = check_fr(bench("ok", CUBE, CUBE_Q), camera=str(cam_p), semantic=str(sem))
check("a cube on the work area is inside the video frame", r["ok"] and r["objects"][0]["inside_frac"] == 1.0, r["objects"])
r = check_fr(bench("ok_name", CUBE, CUBE_Q), camera="ego", width=640, height=480, semantic=str(sem))
check("the scene's own camera by name gives the same verdict", r["ok"], r["objects"])
behind = CUBE.replace('pos="0 0.5 0.025"', 'pos="0 -0.6 0.025"')
r = check_fr(bench("behind", behind, "0 -0.6 0.025 1 0 0 0"), camera=str(cam_p), semantic=str(sem))
check("a cube behind the camera is caught", (not r["ok"]) and r["objects"][0]["verdict"] != "ok", r["objects"])
side = CUBE.replace('pos="0 0.5 0.025"', 'pos="1.2 0.5 0.025"')
r = check_fr(bench("side", side, "1.2 0.5 0.025 1 0 0 0"), camera=str(cam_p), semantic=str(sem))
check("a cube 1.2 m to the side (off-screen) is caught", (not r["ok"]) and r["objects"][0]["verdict"] == "outside the frame", r["objects"])
sem2 = TMP / "sem2.json"
sem2.write_text(json.dumps({"objects": [{"id": "cube", "role": "manipulated"}, {"id": "bowl", "role": "container"}]}))
r = check_fr(bench("ok", CUBE, CUBE_Q), camera=str(cam_p), semantic=str(sem2))
check("a Stage 1 object without a body in the scene is reported", (not r["ok"]) and any(o.get("verdict") == "missing" for o in r["objects"]), r["issues"])

print(f"{'OK' if fails == 0 else 'FAILED'}: video_frustum_check ({fails} failure(s))")
sys.exit(1 if fails else 0)
