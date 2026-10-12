"""Test construct_conical_tube_rack_lowpoly on three racks: a micropipette leaning in a catch-hole well stands still
where the same pose on the plain rack slides out; a flat-bottomed tube stands upright over a catch hole; a small ball
dropped into a catch hole reaches the floor (the opening is real); misuse raises."""
import json
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np

from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.common.mjcf import vec
from tool_library.common.pipettekit import build_micropipette
from tool_library.skills.construct_conical_tube_rack_lowpoly import construct_conical_tube_rack_lowpoly, hole_xy

FAIL = []


def check(case, cond, msg):
    if not cond:
        FAIL.append(f"{case}: {msg}")
        print(f"[FAIL] {case}: {msg}")
    return cond


def world(rack_xml, extra):
    x = Path(rack_xml).read_text()
    return x.replace("  </worldbody>", f"{extra}\n  </worldbody>")


def quat(axis, ang):
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.array(axis, float), np.radians(ang))
    return q


def pipette_stand(td, rack_params, col, lean_deg, off_m, seconds=6.0):
    """Drift (m) of a micropipette standing leaning in well (0, col): nose offset off_m against the lean."""
    b = construct_conical_tube_rack_lowpoly(Path(td) / "r", **rack_params)
    p = json.loads(Path(b["params_json"]).read_text())
    pip = build_micropipette({"name": "mpip", "layout": "upright", "pos": [0, 0, 0], "include_floor": False, "total_len": 0.23,
                              "handle_len": 0.085}, Path(td) / "pip")
    inner = Path(pip["xml"]).read_text().split("<worldbody>")[1].split("</worldbody>")[0]
    inner = "\n".join(ln for ln in inner.splitlines() if "<light" not in ln)
    xp = Path(td) / "w.xml"
    xp.write_text(world(b["xml"], inner))
    m = mujoco.MjModel.from_xml_path(str(xp))
    d = mujoco.MjData(m)
    pb = m.body("mpip").id
    qa = m.jnt_qposadr[m.body_jntadr[pb]]
    hx, hy = hole_xy(p, 0, col)
    dv = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)          # lean toward a corner of the square well
    q = quat(np.cross([0, 0, 1], dv), lean_deg)
    d.qpos[qa:qa + 3] = [hx - off_m * dv[0], hy - off_m * dv[1], 0.0003]
    d.qpos[qa + 3:qa + 7] = q
    mujoco.mj_forward(m, d)
    for _ in range(int(1.5 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    p0 = d.xpos[pb].copy()
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
    return float(np.linalg.norm(d.xpos[pb] - p0)), b


base = {"name": "rk", "length": 0.16, "width": 0.058, "height": 0.068, "rows": 1, "cols": 4, "pitch_x": 0.036, "hole_r": 0.0128,
        "friction": [1.2, 0.01, 0.0002], "timestep": 0.001}

with tempfile.TemporaryDirectory() as td:
    # 1. the leaning pipette: plain rack slides, catch hole holds
    case = "pipette_leaning_in_catch_hole"
    plain, _ = pipette_stand(Path(td) / "a", dict(base), 3, 9.0, 0.0)
    held, b = pipette_stand(Path(td) / "b", dict(base, catch_holes=[[0, 3, 0.0085]]), 3, 9.0, 0.005)
    m = mujoco.MjModel.from_xml_path(b["xml"])
    n_bottom = sum(1 for g in range(m.ngeom) if m.geom(g).name.startswith("rk_bottom"))
    check(case, held < 0.001, f"the pipette moved {1000 * held:.2f} mm standing in the catch hole")
    check(case, plain > 0.003, f"control: on the plain rack the same lean moved only {1000 * plain:.2f} mm (the test would prove nothing)")
    check(case, n_bottom == 4, f"bottom plate around one hole should be 4 boxes, got {n_bottom}")
    n_top = sum(1 for g in range(m.ngeom) if m.geom(g).name.startswith("rk_top"))
    check(case, n_top == 2 + 4 + 1, f"top plate of a 4-well rack should be 7 boxes, got {n_top}")
    if not any(f.startswith(case) for f in FAIL):
        print(f"[PASS] {case}: plain rack {1000 * plain:.1f} mm in 6 s, catch hole {1000 * held:.2f} mm, bottom boxes {n_bottom}, top boxes {n_top}")

    # 2. a flat-bottomed tube stands over a catch hole of a 2 x 3 rack
    case = "tube_over_catch_hole_1x3"
    prm = dict(base, cols=3, width=0.07, length=0.14, pitch_x=0.042, hole_r=0.0125,
               catch_holes=[[0, 1, 0.005], [0, 2, 0.006]])
    b = construct_conical_tube_rack_lowpoly(Path(td) / "c", **prm)
    p = json.loads(Path(b["params_json"]).read_text())
    hx, hy = hole_xy(p, 0, 2)
    tube = (f'<body name="tube" pos="{vec((hx, hy, 0.0045))}"><freejoint/>'
            f'<geom type="cylinder" size="0.0105 0.04" pos="0 0 0.04" density="300"/></body>')
    xp = Path(td) / "c" / "w.xml"
    xp.write_text(world(b["xml"], tube))
    m = mujoco.MjModel.from_xml_path(str(xp))
    d = mujoco.MjData(m)
    for _ in range(int(2.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    tb = m.body("tube").id
    z = float(d.xpos[tb][2])
    tilt = float(np.degrees(np.arccos(np.clip(d.xmat[tb].reshape(3, 3)[2, 2], -1, 1))))
    n_bottom = sum(1 for g in range(m.ngeom) if m.geom(g).name.startswith("rk_bottom"))
    check(case, abs(z - float(p["plate_t"])) < 0.0015, f"the tube base sits at z {1000 * z:.1f} mm, not on the plate ({1000 * p['plate_t']:.0f} mm)")
    check(case, tilt < 2.0, f"the tube tilted {tilt:.1f} deg over the catch hole")
    if not any(f.startswith(case) for f in FAIL):
        print(f"[PASS] {case}: tube on the plate at {1000 * z:.1f} mm, tilt {tilt:.2f} deg, bottom boxes {n_bottom}")

    # 3. the opening is real: a 2 mm ball dropped into a small catch hole reaches the floor
    case = "ball_through_small_catch_hole"
    b = construct_conical_tube_rack_lowpoly(Path(td) / "e", **dict(base, cols=2, length=0.09, pitch_x=0.04, catch_holes=[[0, 0, 0.003]]))
    p = json.loads(Path(b["params_json"]).read_text())
    hx, hy = hole_xy(p, 0, 0)
    ball = f'<body name="ball" pos="{vec((hx, hy, 0.03))}"><freejoint/><geom type="sphere" size="0.002" density="1000"/></body>'
    xp = Path(td) / "e" / "w.xml"
    xp.write_text(world(b["xml"], ball))
    m = mujoco.MjModel.from_xml_path(str(xp))
    d = mujoco.MjData(m)
    for _ in range(int(1.5 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    z = float(d.xpos[m.body("ball").id][2])
    check(case, z < float(p["plate_t"]), f"the ball stopped at z {1000 * z:.1f} mm, on top of the plate instead of in the hole")
    if not any(f.startswith(case) for f in FAIL):
        print(f"[PASS] {case}: ball rests at z {1000 * z:.2f} mm inside the {2 * 3:.0f} mm opening")

    # misuse
    bad = 0
    for kw in ({"catch_holes": [[0, 9, 0.004]]}, {"catch_holes": [[0, 0, 0.03]]}, {"catch_holes": [[0, 0]]}, {"rows": 2}):
        try:
            construct_conical_tube_rack_lowpoly(Path(td) / "bad", **dict(base, **kw))
        except ValueError:
            bad += 1
    if check("misuse", bad == 4, f"{4 - bad} malformed inputs were accepted"):
        print("[PASS] misuse: a well outside the grid, a hole wider than the pitch, a malformed entry and two rows raise ValueError")

sys.exit(1 if FAIL else 0)
