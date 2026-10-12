"""stage_on_bench orientations beyond yaw, and fit_object_placement keeping to its bounds.

    MUJOCO_GL=egl python tool_library/tests/test_stage_on_bench.py

stage_on_bench took x, y, z, yaw only, so an object that lies flipped (a phone screen up whose body frame has its
back up) needed scene.xml patched by hand; fit_object_placement scanned yaw over the full circle after its bounded
search, so bounds {yaw: 20} still returned a yaw 30 degrees away.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.skills.stage_on_bench import stage_on_bench  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dexagent_stage_", dir=os.environ.get("TMPDIR")))
FAILS = 0
RIG = {"origin": [0, 0, 0], "R": np.eye(3).tolist()}
WORLD = str(ROOT / "setup" / "scene.xml")


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


# a phone whose body frame has its back up: the screen is the geom on the local -z face
phone = TMP / "phone.xml"
phone.write_text("""<mujoco model="phone"><worldbody><body name="phone"><freejoint/>
<geom name="phone_body" type="box" size="0.035 0.075 0.004" pos="0 0 0.004" rgba="0.2 0.2 0.2 1" mass="0.18"/>
<geom name="phone_screen" type="box" size="0.032 0.07 0.0005" pos="0 0 -0.0005" rgba="0.05 0.05 0.3 1" mass="0.001"/>
</body></worldbody></mujoco>""")


def staged(asset, tag):
    info = stage_on_bench(WORLD, [asset], TMP / f"{tag}.xml", rig=RIG)
    m = mujoco.MjModel.from_xml_path(info["out_xml"])
    d = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    return info, m, d


# 1. roll 180: the screen ends up on top
info, m, d = staged({"name": "ph", "xml": str(phone), "x": 0.05, "y": 0.4, "z": 0.009, "roll_deg": 180.0, "yaw_deg": 30.0}, "flip")
zb = d.geom_xpos[m.geom("ph_phone_body").id][2]
zs = d.geom_xpos[m.geom("ph_phone_screen").id][2]
R = d.xmat[m.body("ph_phone").id].reshape(3, 3)
check("roll_deg 180 lays the phone screen up (screen above the body, body z axis pointing down)", zs > zb and R[2, 2] < -0.999,
      f"screen z {zs:.4f} body z {zb:.4f} R_zz {R[2, 2]:.3f}")
yaw_long = np.degrees(np.arctan2(R[1, 1], R[0, 1]))                 # the phone's long (y) axis in the rig
check("... with its yaw kept (long axis along 30 + 90 deg in the rig, as a line)", abs(((yaw_long - 120.0) + 90) % 180 - 90) < 0.5, f"{yaw_long:.2f}")

# 2. the same orientation as a quaternion gives the same pose; pitch tilts about rig y
q = info["assets"][0]["quat_rig"]
info_q, mq, dq = staged({"name": "ph", "xml": str(phone), "x": 0.05, "y": 0.4, "z": 0.009, "quat": q}, "flip_q")
check("quat [w, x, y, z] reproduces the roll/yaw pose", np.allclose(dq.xmat[mq.body("ph_phone").id], d.xmat[m.body("ph_phone").id], atol=1e-6))
info_p, mp, dp = staged({"name": "ph", "xml": str(phone), "x": 0.05, "y": 0.4, "z": 0.05, "pitch_deg": 90.0}, "pitch")
Rp = dp.xmat[mp.body("ph_phone").id].reshape(3, 3)
check("pitch_deg 90 turns the body's x axis to point down (rotation about rig y)", np.allclose(Rp[:, 0], [0, 0, -1], atol=1e-6), str(Rp[:, 0].round(3)))

# 3. yaw-only assets are placed exactly as before
info_y, my, dy = staged({"name": "ph", "xml": str(phone), "x": 0.05, "y": 0.4, "yaw_deg": 45.0}, "yaw")
c, s = np.cos(np.radians(45)), np.sin(np.radians(45))
check("yaw_deg alone is a rotation about z as before", np.allclose(dy.xmat[my.body("ph_phone").id].reshape(3, 3),
                                                                 [[c, -s, 0], [s, c, 0], [0, 0, 1]], atol=1e-6))

# 4. misuse is refused before anything is built
for bad, word in (({"quat": [1, 0, 0, 0], "roll_deg": 180.0}, "not both"), ({"rot": [0, 0, 1]}, "unknown key"),
                  ({"quat": [0, 0, 0]}, "four finite numbers")):
    try:
        stage_on_bench(WORLD, [dict({"name": "ph", "xml": str(phone), "x": 0.0, "y": 0.4}, **bad)], TMP / "bad.xml", rig=RIG)
        check(f"refused: {word}", False)
    except ValueError as e:
        check(f"refused: {word}", word in str(e), str(e)[:100])

# ---------------------------------------------------------------- fit_object_placement honours its yaw bounds
from tool_library.common import rigcam as RC  # noqa: E402
from tool_library.common.render import segmentation_renderer  # noqa: E402
from tool_library.skills.fit_object_placement import fit_object_placement  # noqa: E402

cam = RC.camera_from_fovy(640, 480, 45.0, [0.0, -0.2, 0.6], RC.R_cv_from_angles(90.0, 40.0))
BOX = '<body name="box" pos="{x} {y} 0.03" euler="0 0 {yaw}"><freejoint/><geom type="box" size="0.07 0.03 0.03" rgba="0.8 0.1 0.1 1" mass="0.2"/></body>'
SCENE = """<mujoco model="t"><compiler angle="degree"/><visual><global offwidth="1280" offheight="960"/></visual>
<worldbody><light pos="0 0 2" dir="0 0 -1"/><geom name="g_table" type="box" size="1 1 0.02" pos="0 0.6 -0.02" rgba="0.4 0.3 0.2 1"/>
{box}</worldbody></mujoco>"""
truth = TMP / "truth.xml"
truth.write_text(SCENE.format(box=BOX.format(x=0.05, y=0.42, yaw=30)))
spec = mujoco.MjSpec.from_file(str(truth))
cm = spec.worldbody.add_camera(); cm.name = "v"; cm.pos = cam["pos"]; cm.quat = cam["quat_wxyz"]; cm.fovy = cam["fovy_deg"]
mt = spec.compile(); dt = mujoco.MjData(mt); mujoco.mj_forward(mt, dt)
with segmentation_renderer(mt, 480, 640) as r:
    r.update_scene(dt, camera="v")
    seg = r.render()
mask = np.zeros(seg.shape[:2], bool)
ok_px = seg[..., 0] >= 0
mask[ok_px] = mt.geom_bodyid[seg[..., 0][ok_px]] == mt.body("box").id
start = TMP / "start.xml"
start.write_text(SCENE.format(box=BOX.format(x=0.05, y=0.42, yaw=0)))
fb = fit_object_placement(start, "box", mask, cam, params=["x", "y", "yaw"], bounds={"yaw": 20.0}, init_from_mask=False, render_scale=0.5)
check("bounds {yaw: 20} from a start at 0 deg: the fitted yaw stays within +-20 (the true 30 is outside)",
      abs(fb["pose"]["yaw_deg"]) <= 20.0 + 1e-6, f"yaw {fb['pose']['yaw_deg']:.2f}")
fb2 = fit_object_placement(start, "box", mask, cam, params=["x", "y", "yaw"], bounds={"yaw": 20.0, "x": 0.01, "y": 0.01},
                           init_from_mask=False, render_scale=0.5)
check("translation bounds are kept too", abs(fb2["pose"]["x"] - 0.05) <= 0.01 + 1e-6 and abs(fb2["pose"]["y"] - 0.42) <= 0.01 + 1e-6,
      f"x {fb2['pose']['x']:.4f} y {fb2['pose']['y']:.4f}")
ff = fit_object_placement(start, "box", mask, cam, params=["x", "y", "yaw"], init_from_mask=False, render_scale=0.5)
yerr = abs(((ff["pose"]["yaw_deg"] - 30.0) + 90.0) % 180.0 - 90.0)
check("default bounds (full circle) still recover the true yaw of 30 deg (mod 180)", yerr < 4.0 and ff["iou"] > 0.9,
      f"yaw {ff['pose']['yaw_deg']:.2f} iou {ff['iou']:.3f}")

print("ALL PASS" if not FAILS else f"{FAILS} FAILED")
sys.exit(1 if FAILS else 0)
