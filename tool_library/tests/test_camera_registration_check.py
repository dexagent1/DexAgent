"""Smoke test for camera_registration_check: the setup camera passes; inconsistent intrinsics, a scene camera out of sync, implausible poses and masks off the table plane fail."""
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


import copy

import mujoco
import numpy as np
import yaml
from PIL import Image

from tool_library.common import rigcam as RC
from tool_library.common import scene_assert as SA
from tool_library.verifiers.camera_registration_check import camera_registration_check as check_cam

cfg = yaml.safe_load((SETUP / "setup.yaml").read_text())
e = cfg["cameras"]["ego"]
cam = RC.camera_from_mujoco(640, 480, e["fovy_deg"], e["pos"], e["quat_wxyz"], extra={"source": "setup"})
scene = bench("cube", CUBE, CUBE_Q)

# a mask of the cube rendered through the camera: its contact lies on the table
m = mujoco.MjModel.from_xml_path(scene)
d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
seg = SA.render_segmentation(m, d, cam)
cube = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
Image.fromarray(((seg == cube) * 255).astype(np.uint8)).save(TMP / "cube.png")
low = RC.camera_dict(640, 480, cam["fx"], cam["fy"], cam["cx"], cam["cy"], RC.R_cv_from_angles(90, 10, 0), [0, -0.2, 0.3], {"source": "given"})
hv = SA.horizon_v(low, 320)                                             # the table's horizon is inside this camera's image
sky = np.zeros((480, 640), np.uint8); sky[int(hv) - 60:int(hv) - 20, 300:340] = 255
Image.fromarray(sky).save(TMP / "sky.png")
far = np.zeros((480, 640), np.uint8); far[int(hv) - 20:int(hv) + 3, 300:340] = 255
Image.fromarray(far).save(TMP / "far.png")
r = check_cam(cam, scene=scene, scene_camera="ego", masks={"cube": str(TMP / "cube.png")})
check("the setup camera, its scene copy and an object standing on the table pass", r["ok"] and r["masks"][0]["on_table"], r["issues"])
r = check_cam(low, masks={"sky": str(TMP / "sky.png"), "far": str(TMP / "far.png")})
v = {x["id"]: x["verdict"] for x in r["masks"]}
check("a mask standing above the horizon and one whose contact lands far beyond the table are caught",
      (not r["ok"]) and v["sky"] == "above the horizon" and v["far"] == "off the table plane", r["masks"])
c2 = copy.deepcopy(cam); c2["fovy_deg"] = 30.8                          # the stale second ego camera
r = check_cam(c2)
check("a fovy that disagrees with the focal length is caught", (not r["ok"]) and "fovy_deg" in r["issues"][0], r["issues"])
c3 = RC.camera_from_mujoco(640, 480, 43.0, e["pos"], e["quat_wxyz"], extra={"source": "setup"})
r = check_cam(c3, scene=scene, scene_camera="ego")
check("a camera.json that differs from the scene's camera (fovy 43 vs 38) is caught", (not r["ok"]) and r["scene_camera"]["fovy_diff_deg"] < -4, r["scene_camera"])
c4 = RC.camera_dict(640, 480, cam["fx"], cam["fy"], cam["cx"], cam["cy"], RC.R_cv_from_angles(90, -10, 0), [0, -0.2, 0.5])
r = check_cam(c4)
check("a camera looking up (pitch -10 deg) is caught", (not r["ok"]) and r["camera"]["pitch_deg"] < 0, r["issues"])
c5 = RC.camera_dict(640, 480, cam["fx"], cam["fy"], cam["cx"], cam["cy"], RC.R_cv_from_angles(90, 35, 0), [0, -0.2, 3.5])
r = check_cam(c5)
check("a camera 3.5 m above the table is caught", (not r["ok"]) and any("above the tabletop" in i for i in r["issues"]), r["issues"])
c6 = copy.deepcopy(cam); c6["quat_wxyz"] = [1.0, 0.0, 0.0, 0.0]
r = check_cam(c6)
check("quat_wxyz disagreeing with R_cv is caught", (not r["ok"]) and r["consistency"]["quat_vs_R_deg"] > 1, r["issues"])

print(f"{'OK' if fails == 0 else 'FAILED'}: camera_registration_check ({fails} failure(s))")
sys.exit(1 if fails else 0)
