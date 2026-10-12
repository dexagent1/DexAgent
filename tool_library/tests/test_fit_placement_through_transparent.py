import os, sys, tempfile, math
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, mujoco
from PIL import Image
from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.skills.fit_placement_through_transparent import fit_placement_through_transparent

SCENE = """<mujoco><visual><global offwidth="320" offheight="240"/></visual><worldbody>
<geom type="plane" size="2 2 0.01" rgba="0.5 0.5 0.5 1"/>
<camera name="cam" pos="0 -0.45 0.35" xyaxes="1 0 0 0 0.6 0.8" fovy="50"/>
<geom name="panel" type="box" size="0.12 0.003 0.06" pos="0 -0.08 0.06" rgba="0.9 0.9 0.9 {alpha}"/>
<body name="obj" pos="{x0} {y0} 0.025" quat="{qw} 0 0 {qz}"><freejoint/><geom type="{gt}" size="{gs}" rgba="1 0 0 1"/></body>
</worldbody></mujoco>"""

def make(td, name, alpha, truth, start, gt="box", gs="0.04 0.02 0.025", hide_panel=None):
    hide_panel = (alpha < 0.6) if hide_panel is None else hide_panel
    a = math.radians(truth[2])
    p = os.path.join(td, name + "_truth.xml")
    open(p, "w").write(SCENE.format(alpha=alpha, x0=truth[0], y0=truth[1], qw=math.cos(a/2), qz=math.sin(a/2), gt=gt, gs=gs))
    m = mujoco.MjModel.from_xml_path(p); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    if hide_panel:
        m.geom_group[m.geom("panel").id] = 5
    r = mujoco.Renderer(m, 240, 320); r.enable_segmentation_rendering()
    opt = mujoco.MjvOption(); opt.geomgroup[5] = 0
    r.update_scene(d, camera="cam", scene_option=opt); seg = r.render()
    mk = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(seg[..., 0], [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("obj").id])
    mp = os.path.join(td, name + "_mask.png"); Image.fromarray(mk.astype(np.uint8) * 255).save(mp)
    a = math.radians(start[2]); s = os.path.join(td, name + "_start.xml")
    open(s, "w").write(SCENE.format(alpha=alpha, x0=start[0], y0=start[1], qw=math.cos(a/2), qz=math.sin(a/2), gt=gt, gs=gs))
    return s, mp

fails = 0
cases = [
    ("box_behind_clear_panel", 0.3, (0.02, 0.0, 15), (0.0, 0.015, 0), "box", "0.04 0.02 0.025", True),
    ("ellipsoid_behind_clear_panel", 0.2, (-0.015, 0.01, -20), (0.0, -0.005, 0), "ellipsoid", "0.045 0.03 0.025", True),
    ("open_view_no_panel_alpha_one_far", 1.0, (0.0, 0.10, 10), (0.01, 0.09, 0), "box", "0.04 0.02 0.025", True),
]
with tempfile.TemporaryDirectory() as td:
    for name, alpha, truth, start, gt, gs, expect in cases:
        try:
            s, mp = make(td, name, alpha, truth, start, gt, gs)
            r = fit_placement_through_transparent(s, "obj", mp, "cam", half_range={"x": 0.03, "y": 0.03, "yaw": 30},
                                                  render_scale=1.0, steps=5, n_refine=3, out_xml=os.path.join(td, name + "_fit.xml"))
            err = math.hypot(r["pose"]["x"] - truth[0], r["pose"]["y"] - truth[1])
            ok = r["iou"] > 0.85 and err < 0.008 and r["iou"] >= r["iou_start"]
            mujoco.MjModel.from_xml_path(r["out_xml"])
            print(("[PASS] " if ok else "[FAIL] ") + f"{name} iou={r['iou']} start={r['iou_start']} err={err:.4f} hidden={r['n_transparent_geoms']}")
            fails += 0 if ok else 1
        except Exception as e:
            fails += 1; print(f"[FAIL] {name}: {e!r}")
    # an opaque panel must still occlude: IoU stays low against a mask of the hidden object
    s, mp = make(td, "opaque", 1.0, (0.0, 0.0, 0), (0.0, 0.0, 0), hide_panel=True)
    r = fit_placement_through_transparent(s, "obj", mp, "cam", half_range={"x": 0.01, "y": 0.01, "yaw": 5}, steps=3, n_refine=0, render_scale=1.0)
    ok = r["n_transparent_geoms"] == 0 and r["iou"] < 0.5
    print(("[PASS] " if ok else "[FAIL] ") + f"opaque_panel_still_occludes iou={r['iou']}"); fails += 0 if ok else 1
    try:
        fit_placement_through_transparent(s, "obj", mp, "cam", half_range={"roll": 3}); print("[FAIL] bad key accepted"); fails += 1
    except ValueError:
        print("[PASS] unknown_half_range_key_rejected")
sys.exit(1 if fails else 0)
