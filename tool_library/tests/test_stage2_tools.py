"""Smoke tests for the Stage 2 tools: camera registration, EgoDex reading, mask footprints, placement fitting,
frame comparison, the Isaac export and the manifest check. Synthetic scenes and files; no GPU.

    MUJOCO_GL=egl python tool_library/tests/test_stage2_tools.py
Set DEXAGENT_ISAAC_PYTHON to also run one Isaac Sim settle (about a minute).
"""
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.common import rigcam as RC  # noqa: E402
from tool_library.skills.build_isaac_scene import export_scene  # noqa: E402
from tool_library.skills.fit_object_placement import fit_object_placement  # noqa: E402
from tool_library.skills.measure_mask_footprint import measure_mask_footprint  # noqa: E402
from tool_library.skills.read_egodex_episode import read_egodex_episode  # noqa: E402
from tool_library.skills.register_table_camera import register_table_camera  # noqa: E402
from tool_library.skills.render_frame_compare import render_frame_compare, render_with_camera  # noqa: E402
from tool_library.verifiers.scene_recon_manifest_check import scene_recon_manifest_check  # noqa: E402
from tool_library.verifiers.box_enclosure_check import box_enclosure_check  # noqa: E402
from tool_library.skills.construct_fold_box import construct_fold_box  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dexagent_stage2_", dir=os.environ.get("TMPDIR")))
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


# ---------------------------------------------------------------- a small table world with a box and a cylinder
WORLD = """<mujoco model="t"><option timestep="0.002"/><visual><global offwidth="1280" offheight="960"/></visual>
<worldbody><light pos="0 0 2" dir="0 0 -1"/>
<geom name="g_table" type="box" size="1 1 0.02" pos="0 0.6 -0.02" rgba="0.4 0.3 0.2 1"/>
<body name="box" pos="0.10 0.45 0.03" quat="0.9659 0 0 0.2588"><freejoint/><geom type="box" size="0.06 0.04 0.03" rgba="0.8 0.1 0.1 1" mass="0.2"/></body>
<body name="can" pos="-0.12 0.55 0.05"><geom type="cylinder" size="0.03 0.05" rgba="0.1 0.1 0.8 1" mass="0.1"/></body>
</worldbody><keyframe><key name="home" qpos="0.10 0.45 0.03 0.9659 0 0 0.2588"/></keyframe></mujoco>"""
world = TMP / "world.xml"; world.write_text(WORLD)

# ---------------------------------------------------------------- 1. camera registration (three sources)
r1 = register_table_camera(TMP / "cam_setup.json", "setup", 640, 480)
c1 = r1["camera"]
check("register: setup camera rescales to the video size", c1["width"] == 640 and abs(c1["fovy_deg"] - 38.0) < 1e-3)
r1b = register_table_camera(TMP / "cam_setup2.json", "setup", 1280, 960)
check("register: same camera at double resolution keeps fovy and doubles focal", abs(r1b["camera"]["fovy_deg"] - c1["fovy_deg"]) < 1e-6 and abs(r1b["camera"]["fx"] - 2 * c1["fx"]) < 1e-6)
r2 = register_table_camera(TMP / "cam_est.json", "estimate", 640, 480, fovy_deg=45.0, pitch_deg=40.0, roll_deg=0.0, height_m=0.6, yaw_deg=90.0, foot_xy=[0.0, -0.2])
c2 = r2["camera"]
check("register: estimate builds the pose from angles", abs(c2["pitch_deg"] - 40) < 1e-6 and abs(c2["yaw_deg"] - 90) < 1e-6 and abs(c2["pos"][2] - 0.6) < 1e-9, f"pitch {c2['pitch_deg']:.2f}")
r3 = register_table_camera(TMP / "cam_given.json", "given", 640, 480, fovy_deg=45.0, pos=c2["pos"], quat_wxyz=c2["quat_wxyz"])
check("register: given pos+quat reproduces the estimate", np.allclose(r3["camera"]["R_cv"], c2["R_cv"], atol=1e-6))
try:
    register_table_camera(TMP / "cam_bad.json", "estimate", 640, 480, fovy_deg=45.0)
    check("register: estimate without pitch/height is refused", False)
except ValueError:
    check("register: estimate without pitch/height is refused", True)
# pixel <-> plane round trip
P = np.array([[0.1, 0.5, 0.0], [-0.2, 0.9, 0.0]])
uv, z = RC.project(c2, P)
check("rigcam: project then back-project returns the points", np.allclose(RC.pixel_on_plane(c2, uv), P, atol=1e-6))

# ---------------------------------------------------------------- 2. footprint from a rendered mask
cam = c2
import cv2  # noqa: E402
seg = render_with_camera(world, cam, None, 640, 480)      # RGB render just for a frame image
frame = TMP / "frame.jpg"; cv2.imwrite(str(frame), cv2.cvtColor(seg, cv2.COLOR_RGB2BGR))
# segmentation masks of each body
spec = mujoco.MjSpec.from_file(str(world)); c = spec.worldbody.add_camera(); c.name = "v"; c.pos = cam["pos"]; c.quat = cam["quat_wxyz"]; c.fovy = cam["fovy_deg"]
m = spec.compile(); d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
with mujoco.Renderer(m, 480, 640) as rr:
    rr.enable_segmentation_rendering(); rr.update_scene(d, camera="v"); S = rr.render()
gid = S[..., 0]
def body_mask(name):
    bid = m.body(name).id
    out = np.zeros(gid.shape, bool); ok = gid >= 0
    out[ok] = (m.geom_bodyid[gid[ok]] == bid)
    return out
mb, mc = body_mask("box"), body_mask("can")
cv2.imwrite(str(TMP / "mask_box.png"), mb.astype(np.uint8) * 255); cv2.imwrite(str(TMP / "mask_can.png"), mc.astype(np.uint8) * 255)
fp = measure_mask_footprint(TMP / "mask_can.png", cam, out_png=TMP / "fp.png", frame=frame)
cx, cy = fp["contact"]["center_xy"]
check("footprint: can's contact centre lies on the can's near edge", abs(cx + 0.12) < 0.02 and 0.49 < cy < 0.56, f"({cx:.3f}, {cy:.3f})")
check("footprint: can width about its diameter", abs(fp["contact"]["width_m"] - 0.06) < 0.015, f"{fp['contact']['width_m']:.3f}")
check("footprint: can apparent height near 0.10", 0.07 < fp["apparent_height_m"] < 0.16, f"{fp['apparent_height_m']:.3f}")
fpb = measure_mask_footprint(TMP / "mask_box.png", cam, height_guess_m=0.03)
check("footprint: box yaw hint from the flat footprint within 35 deg of 30 (a hint; the fit resolves it)", min(abs(fpb["footprint"]["yaw_deg"] - 30), abs(fpb["footprint"]["yaw_deg"] + 150)) < 35, f"{fpb['footprint']['yaw_deg']:.1f}")

# ---------------------------------------------------------------- 3. placement fit recovers a perturbed pose (free body, fixed body, with scale)
pert = WORLD.replace('pos="0.10 0.45 0.03" quat="0.9659 0 0 0.2588"', 'pos="0.16 0.52 0.03" quat="1 0 0 0"').replace('qpos="0.10 0.45 0.03 0.9659 0 0 0.2588"', 'qpos="0.16 0.52 0.03 1 0 0 0"')
pw = TMP / "world_pert.xml"; pw.write_text(pert)
f = fit_object_placement(pw, "box", TMP / "mask_box.png", cam, params=["x", "y", "yaw"], init_from_mask=False, out_xml=TMP / "world_fit.xml", out_png=TMP / "fit_box.png", frame=frame, render_scale=0.5, seed=1)
p = f["pose"]
yaw_err = min(abs(((p["yaw_deg"] - 30) + 90) % 180 - 90), 180)
check("fit: free box pose recovered (x, y within 1 cm, yaw within 5 deg mod 180)", abs(p["x"] - 0.10) < 0.01 and abs(p["y"] - 0.45) < 0.01 and yaw_err < 5 and f["iou"] > 0.9, f"x {p['x']:.3f} y {p['y']:.3f} yaw {p['yaw_deg']:.1f} iou {f['iou']:.2f}")
mf = mujoco.MjModel.from_xml_path(str(TMP / "world_fit.xml")); df = mujoco.MjData(mf); mujoco.mj_resetDataKeyframe(mf, df, 0); mujoco.mj_forward(mf, df)
check("fit: written scene starts at the fitted pose (keyframe too)", abs(df.xpos[mf.body('box').id][0] - p["x"]) < 1e-5 and abs(df.xpos[mf.body('box').id][1] - p["y"]) < 1e-5)
pert2 = WORLD.replace('<body name="can" pos="-0.12 0.55 0.05">', '<body name="can" pos="-0.05 0.62 0.05">')
pw2 = TMP / "world_pert2.xml"; pw2.write_text(pert2)
f2 = fit_object_placement(pw2, "can", TMP / "mask_can.png", cam, params=["x", "y"], init_from_mask=True, render_scale=0.5)
check("fit: fixed can recovered from a mask-initialised start", abs(f2["pose"]["x"] + 0.12) < 0.01 and abs(f2["pose"]["y"] - 0.55) < 0.015 and f2["iou"] > 0.85, f"x {f2['pose']['x']:.3f} y {f2['pose']['y']:.3f} iou {f2['iou']:.2f}")
small = WORLD.replace('size="0.03 0.05" rgba="0.1 0.1 0.8 1"', 'size="0.021 0.035" rgba="0.1 0.1 0.8 1"').replace('<body name="can" pos="-0.12 0.55 0.05">', '<body name="can" pos="-0.12 0.55 0.035">')
pw3 = TMP / "world_small.xml"; pw3.write_text(small)
f3 = fit_object_placement(pw3, "can", TMP / "mask_can.png", cam, params=["x", "y", "z", "scale"], init_from_mask=False, bounds={"scale": 0.6, "z": 0.03}, render_scale=0.5, n_coarse=60)
check("fit: scale parameter grows an undersized can toward the mask", f3["pose"]["scale"] > 1.2 and f3["iou"] > 0.8, f"scale {f3['pose']['scale']:.2f} iou {f3['iou']:.2f}")
try:
    fit_object_placement(pw, "nobody", TMP / "mask_box.png", cam)
    check("fit: unknown body is refused", False)
except ValueError:
    check("fit: unknown body is refused", True)

# ---------------------------------------------------------------- 4. frame comparison
rc = render_frame_compare(TMP / "world_fit.xml", cam, frame, TMP / "compare.png", mask=TMP / "mask_box.png")
check("compare: composite written with three panels and a difference number", Path(rc["out_png"]).exists() and rc["panels"] == ["video", "simulation", "blend"] and rc["mean_abs_diff"] is not None)
rc2 = render_frame_compare(TMP / "world_fit.xml", cam, frame, TMP / "compare5.png", isaac_png=frame)
check("compare: isaac panels added", len(rc2["panels"]) == 5)

# ---------------------------------------------------------------- 5. isaac export (+ optional real run)
geoms = export_scene(world)
names = {g["name"] for g in geoms}
check("isaac export: table flagged and bodies exported", "g_table" in names and any(g["is_table"] for g in geoms) and len(geoms) == 3)
isaac_py = os.environ.get("DEXAGENT_ISAAC_PYTHON")
if isaac_py and Path(isaac_py).exists():
    from tool_library.skills.build_isaac_scene import build_isaac_scene
    pts = [[-0.1 + 0.2 * t, 0.3 + 0.05 * np.sin(4 * t), 0.0] for t in np.linspace(0, 1, 30)]
    try:
        ri = build_isaac_scene(world, cam, [{"name": "cord", "points": pts, "radius": 0.006, "rgb": [0.8, 0.1, 0.1]}], TMP / "isaac", settle_s=0.8, isaac_python=isaac_py)
        check("isaac: cord settles on the table with millimetre drift", ri["report"]["cord"]["mean_drift_m"] < 0.01 and Path(ri["sim_png"]).exists(), f"drift {ri['report']['cord']['mean_drift_m']*1000:.1f} mm")
    except Exception as exc:
        check("isaac: cord settles on the table with millimetre drift", False, repr(exc)[:300])
else:
    print("[SKIP] isaac run: set DEXAGENT_ISAAC_PYTHON")

# ---------------------------------------------------------------- 6. EgoDex reader on a synthetic episode
import h5py  # noqa: E402
h5 = TMP / "ep.hdf5"; mp4 = TMP / "ep.mp4"
n = 12
with h5py.File(h5, "w") as f:
    f.attrs["task"] = "synthetic"; f.attrs["llm_description"] = "a test episode"
    f.create_dataset("camera/intrinsic", data=np.array([[736.6, 0, 960], [0, 736.6, 540], [0, 0, 1]]))
    T = np.tile(np.eye(4), (n, 1, 1)); T[:, :3, 3] = [0, 1.2, 0.0]            # camera 1.2 m above an ARKit floor (y up)
    R = np.array([[1, 0, 0], [0, np.cos(np.radians(-30)), -np.sin(np.radians(-30))], [0, np.sin(np.radians(-30)), np.cos(np.radians(-30))]])
    # OpenCV camera looking along +z world (ARKit forward is -z, but any consistent choice works here) pitched down 30 deg
    T[:, :3, :3] = R @ np.diag([1, -1, -1])
    f.create_dataset("transforms/camera", data=T)
    for side in ("left", "right"):
        for tip in ("ThumbTip", "IndexFingerTip", "MiddleFingerTip", "RingFingerTip", "LittleFingerTip"):
            J = np.tile(np.eye(4), (n, 1, 1)); J[:, 1, 3] = 0.76 + 0.02 * np.random.rand(n); J[:, 0, 3] = 0.1; J[:, 2, 3] = -0.4
            f.create_dataset(f"transforms/{side}{tip}", data=J)
            f.create_dataset(f"confidences/{side}{tip}", data=np.ones(n))
from tool_library.common.video import write_video  # noqa: E402
write_video([np.full((1080, 1920, 3), 120, np.uint8)] * n, mp4, fps=30.0)
re_ = read_egodex_episode(h5, TMP / "ego", stride=5)
check("egodex: frames, camera.json and episode.json written", Path(re_["episode"]["camera_json"]).exists() and len(re_["frames"]) >= 2 and re_["episode"]["task"] == "synthetic")
check("egodex: table plane from fingertips puts the camera about 0.45 m above the table", abs(re_["camera"]["height_above_table_m"] - (1.2 - 0.76 + 0.008)) < 0.03, f"{re_['camera']['height_above_table_m']:.3f}")
check("egodex: fovy from the intrinsics", abs(re_["camera"]["fovy_deg"] - 72.5) < 0.5, f"{re_['camera']['fovy_deg']:.2f}")

# ---------------------------------------------------------------- 6b. box enclosure
tray = construct_fold_box(TMP / "tray", style="tray", length=0.2, width=0.14, height=0.06)
mt = mujoco.MjModel.from_xml_path(tray["xml"]); qt = np.array(mt.qpos0)
for j in range(mt.njnt):                                   # fold the four wall creases up: the shipped-flat tray is not a box yet
    if mt.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
        qt[mt.jnt_qposadr[j]] = np.pi / 2
rt_flat = box_enclosure_check(tray["xml"], "box")
rt = box_enclosure_check(tray["xml"], "box", expect_top="open", qpos=qt)
check("box enclosure: a folded tray has a bottom, four walls and an open top; shipped flat it does not", rt["ok"] and rt["top_state"] == "open" and not rt_flat["ok"], str(rt["issues"][:1]))
lidbox = construct_fold_box(TMP / "lidbox", style="lid", length=0.2, width=0.14, height=0.06)
rl = box_enclosure_check(lidbox["xml"], "box")
check("box enclosure: a lid box is closed and enclosed", rl["ok"] and rl["top_state"] == "closed", str(rl["issues"][:1]))
slab = '<mujoco><worldbody><body name="slab" pos="0 0 0.002"><geom type="box" size="0.1 0.06 0.002"/><body name="slab_lid" pos="0 -0.06 0.002"><joint type="hinge" axis="1 0 0" range="0 2"/><geom type="box" size="0.1 0.06 0.002" pos="0 0.06 0.004"/></body></body></worldbody></mujoco>'
check("box enclosure: a slab with a hinged panel is not a box", not box_enclosure_check(slab, "slab")["ok"])
three = '<mujoco><worldbody><body name="t"><geom type="box" size="0.1 0.06 0.002" pos="0 0 0.002"/><geom type="box" size="0.002 0.06 0.03" pos="0.098 0 0.03"/><geom type="box" size="0.002 0.06 0.03" pos="-0.098 0 0.03"/><geom type="box" size="0.1 0.002 0.03" pos="0 0.058 0.03"/></body></worldbody></mujoco>'
r3 = box_enclosure_check(three, "t")
check("box enclosure: a missing wall is named", not r3["ok"] and "y_neg" in r3["issues"][0])
solid = '<mujoco><worldbody><body name="blk" pos="0 0 0.03"><geom type="box" size="0.1 0.06 0.03"/></body></worldbody></mujoco>'
check("box enclosure: a solid block fails", not box_enclosure_check(solid, "blk")["ok"])

# ---------------------------------------------------------------- 7. manifest check: a complete manifest passes, a broken one is itemised
sem = {"task_prompt": "x", "task_summary": "x", "environment": {"support_surface": "t"}, "hands": {"used": ["right"]},
       "objects": [{"id": "box", "name": "red box", "type": "rigid", "role": "manipulated", "properties": ["p"], "evidence": "0 s"},
                   {"id": "can", "name": "blue can", "type": "rigid", "role": "static", "properties": ["p"], "evidence": "0 s"}],
       "subgoals": [{"index": 1, "acting_hand": "right", "action": "a", "object_refs": ["box"], "motion": "grasp_lift", "hand_roles": {"left": "r", "right": "g"}, "goal": "g"}],
       "task_success": "s"}
(TMP / "scene_semantics.json").write_text(json.dumps(sem))
RC.save(cam, TMP / "camera.json")
man = {"scene_semantic": "scene_semantics.json", "scene_xml": "world_fit.xml", "camera_json": "camera.json", "frame0_compare_png": "compare.png", "simulator": "mujoco",
       "objects": [{"id": "box", "type": "rigid", "skill": "construct_cubes", "skill_origin": "library", "asset_xml": "world.xml", "root_body": "box",
                    "pose": {"x": 0.1, "y": 0.45, "yaw_deg": 30}, "mask": "mask_box.png", "iou": 0.9, "verification_plan": {"settle_stability_check": "rests on the table"}, "properties_covered": {"p": "settle_stability_check"}, "verifiers": {"settle_stability_check": {"ok": True}, "box_enclosure_check": {"ok": True}}}]}
(TMP / "scene_recon.json").write_text(json.dumps(man))
rm = scene_recon_manifest_check(TMP / "scene_recon.json")
check("manifest: complete manifest passes (static object may be omitted)", rm["ok"], str(rm["issues"][:3]))
bad = dict(man); bad["objects"] = [dict(man["objects"][0], iou=0.2, verifiers={"settle_stability_check": {"ok": False}}, skill="no_such_skill")]
(TMP / "scene_recon_bad.json").write_text(json.dumps(bad))
rb = scene_recon_manifest_check(TMP / "scene_recon_bad.json")
check("manifest: low IoU, failed verifier and unknown skill are itemised", not rb["ok"] and sum(any(w in i for w in ("iou", "verifiers", "skill")) for i in rb["issues"]) >= 3, str(len(rb["issues"])))
sem2 = dict(sem); sem2["objects"] = sem["objects"] + [{"id": "rope", "name": "rope", "type": "deformable", "role": "manipulated", "properties": ["p"], "evidence": "0 s", "deformation": {"kind": "rope", "how_the_task_deforms_it": "x"}}]
(TMP / "scene_semantic2.json").write_text(json.dumps(sem2))
rd = scene_recon_manifest_check(TMP / "scene_recon.json", scene_semantic_json=TMP / "scene_semantic2.json")
check("manifest: a deformable object without Isaac files fails", not rd["ok"] and any("isaac" in i or "rope" in i for i in rd["issues"]))

print(f"\n{'all passed' if not FAILS else str(FAILS) + ' FAILED'}; outputs in {TMP}")
sys.exit(1 if FAILS else 0)
