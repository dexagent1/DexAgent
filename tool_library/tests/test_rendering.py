"""Smoke tests for the Scene Runtime & Rendering skills.

    MUJOCO_GL=egl python tool_library/tests/test_rendering.py

Everything runs on synthetic scenes; nothing is downloaded. render_cycles (bpy)
and erase_with_void (VOID weights) are exercised only when their dependencies
are importable, otherwise reported as skipped.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import mujoco  # noqa: E402

from tool_library.common.render import camera_dict, camera_from_table_lens, focal_px, project  # noqa: E402
from tool_library.skills.add_wrist_cameras import add_wrist_cameras  # noqa: E402
from tool_library.skills.build_tabletop_world import build_tabletop_world  # noqa: E402
from tool_library.skills.composite_renders import composite_renders  # noqa: E402
from tool_library.skills.export_blender_bundle import export_blender_bundle  # noqa: E402
from tool_library.skills.identify_moving_joint import identify_moving_joint  # noqa: E402
from tool_library.skills.render_scene import render_scene  # noqa: E402
from tool_library.skills.render_trajectory_video import render_trajectory_video  # noqa: E402
from tool_library.skills.stage_on_bench import stage_on_bench  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dexagent_render_", dir=os.environ.get("TMPDIR")))
RESULTS = []


def ok(name, cond, note=""):
    RESULTS.append((name, bool(cond), note))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {note}")


CUP = """<mujoco model="cup"><compiler angle="radian"/>
<worldbody><light pos="0 0 1" dir="0 0 -1"/>
<body name="cup" pos="0 0 0.05"><freejoint/><geom name="cup_body" type="cylinder" size=".035 .05" rgba=".8 .2 .2 1" mass="0.2"/></body>
</worldbody></mujoco>"""
CABINET = """<mujoco model="cabinet"><compiler angle="radian"/>
<worldbody>
<body name="carcass" pos="0 0 0.16"><geom name="box" type="box" size=".12 .14 .16" rgba=".6 .5 .4 1"/>
 <body name="drawer_top" pos="0 -0.14 0.08"><joint name="drawer_top_slide" type="slide" axis="0 -1 0" range="0 0.2"/>
  <geom name="front_top" type="box" size=".11 .01 .06" rgba=".3 .3 .3 1"/><geom type="box" size=".10 .12 .01" pos="0 0.13 -0.05" rgba=".5 .5 .5 1"/></body>
 <body name="drawer_bottom" pos="0 -0.14 -0.08"><joint name="drawer_bottom_slide" type="slide" axis="0 -1 0" range="0 0.2"/>
  <geom name="front_bottom" type="box" size=".11 .01 .06" rgba=".3 .3 .3 1"/><geom type="box" size=".10 .12 .01" pos="0 0.13 -0.05" rgba=".5 .5 .5 1"/></body>
</body></worldbody></mujoco>"""
HAND = """<mujoco model="hand"><compiler angle="radian"/>
<worldbody>
<body name="hand_root" pos="0 0 0.3" quat="1 0 0 0">
 <joint name="hand_free" type="free"/>
 <geom name="palm" type="box" size=".04 .01 .05" pos="0.045 0 0.06" rgba=".9 .7 .6 1" mass="0.3"/>
 <site name="tip_thumb" pos="0.101 0 0.12" size="0.005"/>
 <site name="tip_index" pos="0.052 0.03 0.17" size="0.005"/>
 <site name="tip_middle" pos="0.048 0.01 0.176" size="0.005"/>
 <site name="tip_ring" pos="0.044 -0.01 0.17" size="0.005"/>
 <site name="tip_pinky" pos="0.040 -0.03 0.16" size="0.005"/>
</body></worldbody></mujoco>"""
BENCH = """<mujoco model="bench"><compiler angle="radian"/>
<worldbody><light pos="0 0 2" dir="0 0 -1"/>
<geom name="table_top" type="box" size=".6 .4 .02" pos="0 0 -0.02" rgba=".7 .7 .7 1"/>
<body name="robot_left_arm_root" pos="-0.25 -0.3 0"><geom type="cylinder" size=".04 .05" pos="0 0 0.05" rgba=".2 .2 .2 1"/>
 <body name="robot_left_link1" pos="0 0 0.1"><joint name="l_j1" type="hinge" axis="0 0 1" range="-3 3"/><geom type="capsule" size=".02" fromto="0 0 0 0 0.2 0.2" rgba=".3 .3 .9 1"/></body></body>
<body name="robot_right_arm_root" pos="0.25 -0.3 0"><geom type="cylinder" size=".04 .05" pos="0 0 0.05" rgba=".2 .2 .2 1"/>
 <body name="robot_right_link1" pos="0 0 0.1"><joint name="r_j1" type="hinge" axis="0 0 1" range="-3 3"/><geom type="capsule" size=".02" fromto="0 0 0 0 0.2 0.2" rgba=".9 .3 .3 1"/></body></body>
</worldbody></mujoco>"""
for n, x in (("cup", CUP), ("cabinet", CABINET), ("hand", HAND), ("bench", BENCH)):
    (TMP / f"{n}.xml").write_text(x)


def test_camera_maths():
    pos, quat = camera_from_table_lens(fovy_deg=45, height_m=0.6, pitch_deg=50, yaw_deg=90, xy=(0, -0.4), table_top_z=0.0)
    uv, d = project(np.array([[0.0, 0.0, 0.0]]), pos, quat, 45, 640, 480)
    ok("camera_from_table_lens: table point in front and roughly centred",
       d[0] > 0 and 200 < uv[0, 0] < 440 and 100 < uv[0, 1] < 400, f"uv={uv[0].round(1)} d={d[0]:.2f}")
    ok("focal_px(45, 480) ~ 579", abs(focal_px(45, 480) - 579.4) < 1.0)


def test_build_world():
    cams = {"ego": {"pos": [0, -0.9, 0.7], "target": [0, 0, 0.0], "fovy": 42},
            "side": {"pos": [1.0, 0, 0.5], "quat": [0.5, 0.5, 0.5, 0.5], "fovy": 45}}
    info = build_tabletop_world(TMP / "world.xml", table={"size": [0.5, 0.35], "top_z": 0.0},
                                assets=[{"name": "cup", "xml": TMP / "cup.xml", "pos": [0.1, 0.0], "yaw_deg": 30},
                                        {"name": "cab", "xml": TMP / "cabinet.xml", "pos": [-0.3, 0.1, 0.0], "yaw_deg": 0}],
                                cameras=cams)
    m = mujoco.MjModel.from_xml_path(info["out_xml"])
    ok("build_tabletop_world compiles with both assets", m.nbody >= 5 and m.ncam == 2, f"nq={m.nq} joints={info['joints']}")
    ok("asset bodies are prefixed", "cup_cup" in info["assets"]["cup"]["bodies"] and "cab_drawer_top" in info["assets"]["cab"]["bodies"])
    ok("joints kept from assets", "cab_drawer_top_slide" in info["joints"] and m.nq == 7 + 2)
    return info["out_xml"]


def test_render(world):
    img = render_scene(world, "ego", TMP / "ego.png", width=320, height=240)
    ok("render_scene rgb", img.shape == (240, 320, 3) and img.std() > 5)
    seg = render_scene(world, "ego", TMP / "seg.png", width=320, height=240, mode="segmentation")
    m = mujoco.MjModel.from_xml_path(world)
    cup_b = m.body("cup_cup").id
    ok("segmentation holds the cup body id", (seg == cup_b).sum() > 20, f"{int((seg == cup_b).sum())} px")
    dep = render_scene(world, "ego", None, width=320, height=240, mode="depth")
    ok("depth is metric (cup ~1.1 m away)", 0.6 < float(np.median(dep[seg == cup_b])) < 1.6, f"{float(np.median(dep[seg == cup_b])):.2f} m")
    free = render_scene(world, {"pos": [0, -0.9, 0.7], "target": [0, 0, 0], "fovy": 42}, None, width=320, height=240)
    ok("free camera dict (pos+target) renders like the named ego camera", np.abs(free.astype(int) - img.astype(int)).mean() < 2.0,
       f"mean abs diff {np.abs(free.astype(int) - img.astype(int)).mean():.2f}")


def test_trajectory_video(world):
    m = mujoco.MjModel.from_xml_path(world)
    Q = np.tile(mujoco.MjData(m).qpos.copy(), (12, 1))
    Q[:, 2] += np.linspace(0, 0.1, 12)                           # lift the cup
    np.savez(TMP / "traj.npz", qpos=Q, fps=10.0)
    r = render_trajectory_video(world, TMP / "traj.npz", TMP / "vid", ["ego", "side"], width=320, height=240)
    ok("render_trajectory_video wrote per-camera mp4 + preview",
       all(Path(v).stat().st_size > 1000 for v in r["videos"].values()) and Path(r["preview"]).exists() and r["n_frames"] == 12)


def test_stage_on_bench():
    # an EgoDex video camera (1920 x 1080, as camera.json writes it) and a 640 x 480 bench webcam
    cams = {"video_cam": {"width": 1920, "height": 1080, "fovy_deg": 72.5, "pos": [0.0, -0.6, 0.5], "quat_wxyz": [0.9239, 0.3827, 0.0, 0.0]},
            "webcam": {"width": 640, "height": 480, "fovy_deg": 38.0, "pos": [0.0, -0.7, 0.6], "quat_wxyz": [0.9239, 0.3827, 0.0, 0.0]}}
    info = stage_on_bench(TMP / "bench.xml", [{"name": "cup", "xml": TMP / "cup.xml", "x": 0.0, "y": 0.35, "yaw_deg": 0}], TMP / "bench_staged.xml",
                          cameras=cams)
    m = mujoco.MjModel.from_xml_path(info["out_xml"])
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    p = d.xpos[m.body("cup_cup").id]
    ok("rig frame: origin at base midpoint, +y into the work", np.allclose(info["origin"], [0, -0.3, 0], atol=1e-6) and np.allclose(info["R"], np.eye(3), atol=1e-6))
    ok("cup placed 0.35 m into the work from the baseline", np.allclose(p[:2], [0.0, 0.05], atol=1e-6) and abs(p[2] - 0.05) < 1e-6, f"cup at {p.round(3)}")
    ok("bench qpos layout unchanged before the new body", info["nq_before"] == 2 and info["nq_after"] == 9)
    res = {n: m.cam_resolution[m.camera(n).id].tolist() for n in cams}
    ok("stage_on_bench writes each camera's image size from camera.json", res == {"video_cam": [1920, 1080], "webcam": [640, 480]}, str(res))
    # the sim video keeps each camera's aspect (it rendered the EgoDex camera at 640 x 480, squashed, and a contact
    # sheet then refused crops given in the human video's frame); an explicit size still wins
    Q = np.tile(d.qpos.copy(), (6, 1))
    np.savez(TMP / "staged.npz", qpos=Q, fps=10.0)
    r = render_trajectory_video(info["out_xml"], TMP / "staged.npz", TMP / "vid_staged", ["video_cam", "webcam"])
    import imageio.v2 as iio
    shp = {k: iio.get_reader(v).get_data(0).shape[:2] for k, v in r["videos"].items()}
    ok("render_trajectory_video renders each camera at its aspect (16:9 at 854 x 480, 4:3 at 640 x 480)",
       r["sizes"] == {"video_cam": [854, 480], "webcam": [640, 480]} and shp == {"video_cam": (480, 854), "webcam": (480, 640)}, f"{r['sizes']} {shp}")
    r2 = render_trajectory_video(info["out_xml"], TMP / "staged.npz", TMP / "vid_staged2", ["video_cam"], width=320, height=240)
    r3 = render_trajectory_video(info["out_xml"], TMP / "staged.npz", TMP / "vid_staged3", ["video_cam"], width=1280)
    ok("an explicit size is kept; one given size scales the camera's aspect", r2["sizes"]["video_cam"] == [320, 240] and r3["sizes"]["video_cam"] == [1280, 720],
       f"{r2['sizes']} {r3['sizes']}")


def test_wrist_cameras():
    hands = {"right": {"body": "hand_root", "fingertips": {k: f"tip_{k}" for k in ("thumb", "index", "middle", "ring", "pinky")}}}
    info = add_wrist_cameras(TMP / "hand.xml", TMP / "hand_cam.xml", hands, mode="overhand",
                             fixed={"ego": {"pos": [0, -1, 1], "target": [0, 0, 0.3], "fovy": 45}})
    m = mujoco.MjModel.from_xml_path(info["out_xml"])
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    cam = camera_dict(m, d, "right_wrist")
    tips = np.array([d.site_xpos[m.site(f"tip_{k}").id] for k in ("index", "middle", "ring", "pinky")])
    uv, depth = project(tips, cam["pos"], cam["quat"], cam["fovy"], 640, 480)
    ok("overhand wrist camera sees all four fingertips in front of it", (depth > 0).all() and (uv[:, 0] > 0).all() and (uv[:, 0] < 640).all(),
       f"uv={uv.round(0).tolist()}")
    ok("nq unchanged and fixed camera added", info["nq"] == 7 and m.ncam == 2)
    info2 = add_wrist_cameras(TMP / "hand.xml", TMP / "hand_cam2.xml", hands, mode="behind")
    m2 = mujoco.MjModel.from_xml_path(info2["out_xml"])
    d2 = mujoco.MjData(m2)
    mujoco.mj_forward(m2, d2)
    cam2 = camera_dict(m2, d2, "right_wrist")
    uv2, dep2 = project(tips, cam2["pos"], cam2["quat"], cam2["fovy"], 640, 480)
    ok("behind-the-palm wrist camera also frames the fingertips", (dep2 > 0).all() and (np.abs(uv2[:, 0] - 320) < 320).all())


def test_identify_joint():
    world = build_tabletop_world(TMP / "cab_world.xml", assets=[{"name": "cab", "xml": TMP / "cabinet.xml", "pos": [0.0, 0.1, 0.0]}],
                                 cameras={"video": {"pos": [0, -1.1, 0.8], "target": [0, 0.0, 0.1], "fovy": 45}})["out_xml"]
    m = mujoco.MjModel.from_xml_path(world)
    d = mujoco.MjData(m)
    before = render_scene(m, "video", None, width=640, height=480)
    d.qpos[m.joint("cab_drawer_bottom_slide").qposadr[0]] = 0.18       # the demonstrator opened the BOTTOM drawer
    during = render_scene(m, "video", None, qpos=d.qpos, width=640, height=480)
    r = identify_moving_joint(world, before, during, "video", joint_prefix="cab_", out_dir=TMP / "joint_id")
    ok("identify_moving_joint picks the drawer that moved", r["most_changed"] == "cab_drawer_bottom_slide",
       f"ranked={[(c['joint'], c['edge_match']) for c in r['ranked']]}")
    ok("annotated frames saved", len(r["images"]) == 2 and all(Path(p).exists() for p in r["images"]))


def test_export_and_composite(world):
    import cv2
    m = mujoco.MjModel.from_xml_path(world)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    ego = camera_dict(m, d, "ego")
    plate = np.full((240, 320, 3), (200, 190, 170), np.uint8)
    cv2.imwrite(str(TMP / "plate.png"), plate)
    Q = np.tile(d.qpos.copy(), (4, 1))
    Q[:, 2] += np.linspace(0, 0.05, 4)
    b = export_blender_bundle(world, Q, TMP / "bundle", ego={"pos": ego["pos"].tolist(), "quat_wxyz": ego["quat"].tolist(), "fovy": ego["fovy"]},
                              table={"top_z": 0.0, "center": [0, 0], "half_size": [0.5, 0.35]}, plate_png=TMP / "plate.png",
                              views={"cup_cam": {"parent_body": "cup_cup", "pos": [0, -0.3, 0.2], "quat_wxyz": [1, 0, 0, 0], "fovy": 60}},
                              width=320, height=240, fps=10)
    meta = json.loads((TMP / "bundle" / "ego" / "scene.json").read_text())
    T = np.load(TMP / "bundle" / "transforms.npz")["T"]
    ok("export_blender_bundle: geoms, transforms and both views", b["n_geoms"] >= 3 and T.shape[:2] == (4, b["n_geoms"]) and set(b["views"]) == {"ego", "cup_cam"},
       f"{b['n_geoms']} geoms, robot={b['n_robot_geoms']}")
    ok("tracked camera moves with the cup", np.linalg.norm(np.array(meta["camera"]["pos"])) > 0 and
       abs(json.loads((TMP / 'bundle' / 'cup_cam' / 'scene.json').read_text())["camera_track"][3]["pos"][2]
           - json.loads((TMP / 'bundle' / 'cup_cam' / 'scene.json').read_text())["camera_track"][0]["pos"][2] - 0.05) < 1e-6)
    ok("per-frame plates for the tracked view", len(list((TMP / "bundle" / "cup_cam" / "plates").glob("frame*.png"))) == 4)
    # fake Cycles output: an RGBA disc so compositing can be checked without bpy
    for v in ("ego", "cup_cam"):
        fd = TMP / "bundle" / v / "frames"
        fd.mkdir(exist_ok=True)
        for t in range(4):
            rgba = np.zeros((240, 320, 4), np.uint8)
            cv2.circle(rgba, (160 + 10 * t, 120), 30, (40, 60, 220, 255), -1)
            cv2.imwrite(str(fd / f"f{t:05d}.png"), rgba)
    r = composite_renders(TMP / "bundle", TMP / "final")
    cap = cv2.VideoCapture(r["videos"]["ego"])
    okr, f0 = cap.read()
    cap.release()
    ok("composite_renders: plate outside the disc, render inside", okr and np.abs(f0[20, 20].astype(int) - plate[20, 20]).max() < 8
       and f0[120, 160, 2] > 150, f"corner={f0[20, 20].tolist() if okr else None} centre={f0[120, 160].tolist() if okr else None}")
    ok("preview with both views", Path(r["preview"]).exists() and r["n_frames"] == 4)


def test_optional_cycles():
    try:
        import bpy  # noqa: F401
    except ImportError:
        print("[SKIP] render_cycles: bpy not importable in this interpreter")
        return
    from tool_library.skills.render_cycles import render_cycles
    for f in (TMP / "bundle" / "ego" / "frames").glob("*.png"):
        f.unlink()
    r = render_cycles(TMP / "bundle" / "ego", samples=4)
    ok("render_cycles rendered the ego view", r["n_rendered"] == 4, f"{r['runtime_s']} s")


if __name__ == "__main__":
    test_camera_maths()
    world = test_build_world()
    test_render(world)
    test_trajectory_video(world)
    test_stage_on_bench()
    test_wrist_cameras()
    test_identify_joint()
    test_export_and_composite(world)
    test_optional_cycles()
    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed; outputs in {TMP}")
    sys.exit(1 if failed else 0)
