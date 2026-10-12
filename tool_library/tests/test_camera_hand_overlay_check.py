"""Smoke test for camera_hand_overlay_check: home hands through the right camera pass; pitch and focal errors and an empty hands mask fail; footage that does not start at home is not judged."""
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
from tool_library.verifiers.camera_hand_overlay_check import camera_hand_overlay_check as check_ov

cfg = yaml.safe_load((SETUP / "setup.yaml").read_text())
e = cfg["cameras"]["ego"]
cam = RC.camera_from_mujoco(640, 480, e["fovy_deg"], e["pos"], e["quat_wxyz"], extra={"source": "setup"})
# frame 0 of a bench recording: the hands at home, seen by the ego camera (rendered here as the hand mask)
m = mujoco.MjModel.from_xml_path(str(SETUP / "scene.xml"))
d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
seg = SA.render_segmentation(m, d, cam)
hands = SA.hand_body_ids(m, SA.robot_body_ids(m))
gb = np.where(seg >= 0, m.geom_bodyid[np.clip(seg, 0, m.ngeom - 1)], -1)
mask = np.isin(gb, list(hands["left"] | hands["right"]))
Image.fromarray((mask * 255).astype(np.uint8)).save(TMP / "hands.png")

r = check_ov(cam, str(TMP / "hands.png"))
check("the home hands through the registered camera cover the hands of frame 0", r["ok"] and r["applicable"] and r["iou"] > 0.95, (r["issues"], r["iou"]))
c2 = RC.camera_dict(640, 480, cam["fx"], cam["fy"], cam["cx"], cam["cy"], RC.R_cv_from_angles(e["yaw_deg"], e["pitch_deg"] + 4, e["roll_deg"]),
                    cam["pos"], {"source": "setup"})
r = check_ov(c2, str(TMP / "hands.png"))
check("a camera pitched 4 deg off is caught", (not r["ok"]) and r["iou"] < 0.3, (r["iou"], r["hands"]))
c3 = RC.camera_dict(640, 480, cam["fx"] * 1.25, cam["fy"] * 1.25, cam["cx"], cam["cy"], np.asarray(cam["R_cv"]), cam["pos"], {"source": "setup"})
r = check_ov(c3, str(TMP / "hands.png"))
check("a wrong focal length (x1.25) is caught", not r["ok"], (r["iou"], r["hands"]))
c4 = copy.deepcopy(cam); c4["source"] = "egodex"
shifted = np.roll(mask, 120, axis=1)
Image.fromarray((shifted * 255).astype(np.uint8)).save(TMP / "hands_elsewhere.png")
r = check_ov(c4, str(TMP / "hands_elsewhere.png"))
check("footage from another session (source egodex) is not judged", r["ok"] and not r["applicable"], r)
r = check_ov(c4, str(TMP / "hands_elsewhere.png"), start_at_home=True)
check("forcing start_at_home on hands elsewhere fails", not r["ok"], (r["iou"], r["hands"]))
Image.fromarray(np.zeros((480, 640), np.uint8)).save(TMP / "nohands.png")
r = check_ov(cam, str(TMP / "nohands.png"))
check("an empty hands mask fails when frame 0 should show the hands at home", not r["ok"] and r["applicable"] and r["issues"], r)
r = check_ov(c4, str(TMP / "nohands.png"))
check("an empty hands mask on footage that is not judged passes with a note", r["ok"] and "no hands" in r["note"], r)

print(f"{'OK' if fails == 0 else 'FAILED'}: camera_hand_overlay_check ({fails} failure(s))")
sys.exit(1 if fails else 0)
