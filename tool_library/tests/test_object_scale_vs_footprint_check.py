"""Smoke test for object_scale_vs_footprint_check: objects at the size of their silhouette pass; 0.7x-0.75x and 1.4x objects fail, from masks and from footprint bboxes."""
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

import mujoco
import numpy as np
import yaml
from PIL import Image

from tool_library.common import rigcam as RC
from tool_library.common import scene_assert as SA
from tool_library.verifiers.object_scale_vs_footprint_check import object_scale_vs_footprint_check as check_scale

cfg = yaml.safe_load((SETUP / "setup.yaml").read_text())
e = cfg["cameras"]["ego"]
cam = RC.camera_from_mujoco(640, 480, e["fovy_deg"], e["pos"], e["quat_wxyz"])
cam_p = TMP / "camera.json"
RC.save(cam, cam_p)


def mug(half, name="mug"):
    return (f'<body name="{name}" pos="0.05 0.45 {half}"><freejoint/><geom name="{name}_g" type="cylinder" size="{half * 0.8} {half}" '
            f'mass="0.2" rgba="0.3 0.3 0.8 1"/></body>'), f"0.05 0.45 {half} 1 0 0 0"


# the "video" mask: the true object (5 cm tall mug) rendered through the camera
ref = bench("truth", *mug(0.05))
m = mujoco.MjModel.from_xml_path(ref)
d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
seg = SA.render_segmentation(m, d, cam)
g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "mug_g")
mask = (seg == g)
(TMP / "masks").mkdir(exist_ok=True)
Image.fromarray((mask * 255).astype(np.uint8)).save(TMP / "masks" / "mug.png")
ys, xs = np.nonzero(mask)

r = check_scale(ref, str(cam_p), masks=str(TMP / "masks"))
check("an object built at the size of its silhouette passes", r["ok"] and abs(r["objects"][0]["scale"] - 1) < 0.05, r["objects"])
r = check_scale(bench("small", *mug(0.05 * 0.75)), str(cam_p), masks={"mug": str(TMP / "masks" / "mug.png")})
check("the same object built at 0.75x is caught as too small", (not r["ok"]) and r["objects"][0]["verdict"] == "too small", r["objects"])
r = check_scale(bench("large", *mug(0.05 * 1.4)), str(cam_p), masks={"mug": str(TMP / "masks" / "mug.png")})
check("built at 1.4x it is caught as too large", (not r["ok"]) and r["objects"][0]["verdict"] == "too large", r["objects"])
r = check_scale(bench("near", *mug(0.05 * 1.1)), str(cam_p), masks={"mug": str(TMP / "masks" / "mug.png")})
check("a 10% size difference is within the tolerance", r["ok"], r["objects"])
fp = {"contact": {}, "bbox_px": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]}
(TMP / "fp.json").write_text(json.dumps(fp))
r = check_scale(bench("small2", *mug(0.05 * 0.7)), str(cam_p), footprints={"mug": str(TMP / "fp.json")})
check("a measure_mask_footprint result (bbox) catches the 0.7x object too", (not r["ok"]) and r["objects"][0]["verdict"] == "too small", r["objects"])
r = check_scale(ref, str(cam_p), footprints={"mug": str(TMP / "fp.json")})
check("and passes the right-sized one", r["ok"], r["objects"])

Image.fromarray((np.ones_like(mask) * 255).astype(np.uint8)).save(TMP / "hands_all.png")
r = check_scale(bench("small3", *mug(0.05 * 0.7)), str(cam_p), masks={"mug": str(TMP / "masks" / "mug.png")}, ignore_mask=str(TMP / "hands_all.png"))
check("an object the hands of frame 0 cover is not judged (held in the hand)", r["ok"] and r["objects"][0]["verdict"] == "occluded by the hands", r["objects"])

# a deformable simulated in Isaac Sim has no MuJoCo body: its settled cord is drawn as a tube along its link centres
from tool_library.common import isaac_cord as IC
R = 0.01386
t = np.linspace(0, np.pi, 200)
U = np.stack([0.02 + 0.14 * np.cos(t), 0.50 + 0.16 * np.sin(t), np.full_like(t, R)], 1)
spec = IC.cord_spec({"name": "rope", "points": U.tolist(), "radius": R, "preset": "segmented_rope", "n_links": 24})
C, Q = IC.lay_poses(spec["points"], 24, R)
spec["link_poses"] = {"centres": C.tolist(), "quats": Q.tolist()}
IC.save_spec(spec, TMP / "rope_cord.json")
rope_rec = {"objects": [{"id": "rope", "type": "deformable", "simulator": "isaac", "isaac": {"cord_json": str(TMP / "rope_cord.json")}}]}
Image.fromarray((SA.cord_tube_mask(cam, C, R, spec["segment_m"], quats=Q) * 255).astype(np.uint8)).save(TMP / "rope_true.png")
big = C.copy(); big[:, :2] = C[:, :2].mean(0) + 1.5 * (C[:, :2] - C[:, :2].mean(0))
Image.fromarray((SA.cord_tube_mask(cam, big, 1.5 * R, 1.5 * spec["segment_m"]) * 255).astype(np.uint8)).save(TMP / "rope_big.png")
empty = bench("empty")
r = check_scale(empty, str(cam_p), masks={"rope": str(TMP / "rope_true.png")}, recon=rope_rec)
check("an Isaac Sim cord drawn along its settled centreline matches its own silhouette", r["ok"] and abs(r["objects"][0]["scale"] - 1) < 0.05, r["objects"])
r = check_scale(empty, str(cam_p), masks={"rope": str(TMP / "rope_big.png")}, recon=rope_rec)
check("an Isaac Sim cord 1.5x smaller than its mask is caught as too small", (not r["ok"]) and r["objects"][0]["verdict"] == "too small", r["objects"])

print(f"{'OK' if fails == 0 else 'FAILED'}: object_scale_vs_footprint_check ({fails} failure(s))")
sys.exit(1 if fails else 0)
