import sys, tempfile, math
import numpy as np
import mujoco
from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.skills.construct_stud_brick import construct_stud_brick

fails = 0
def case(name, fn):
    global fails
    try:
        fn(); print(f"[PASS] {name}")
    except Exception as e:
        fails += 1; print(f"[FAIL] {name}: {e!r}")

def rest_on_floor(xml_path, body):
    # load with a floor appended, drop 5 mm, check it comes to rest flat on z = 0
    s = open(xml_path).read().replace("<worldbody>", '<worldbody>\n    <geom name="floor" type="plane" size="1 1 0.01"/>', 1)
    m = mujoco.MjModel.from_xml_string(s); d = mujoco.MjData(m)
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    d.qpos[2] += 0.005
    for _ in range(1500): mujoco.mj_step(m, d)
    z, tilt = d.xpos[b][2], math.degrees(math.acos(min(1.0, d.xmat[b][8])))
    assert abs(z) < 0.0015 and tilt < 1.0, f"rest z {z:.4f} tilt {tilt:.2f}"
    return m

def duplo():
    out = tempfile.mkdtemp()
    b = construct_stud_brick(out, name="duplo")
    assert b["studs"] == [2, 2] and len(b["stud_sites"]) == 4 and b["n_tubes"] == 1
    assert abs(b["size_m"][0] - 0.0318) < 1e-4 and b["wall_thickness_m"] > 0.0005
    m = rest_on_floor(b["xml"], "duplo")
    assert 0.003 < m.body_mass[1] < 0.02, m.body_mass[1]

def lego_2x4():
    out = tempfile.mkdtemp()
    b = construct_stud_brick(out, name="lego", studs_x=4, studs_y=2, pitch=0.008, body_height=0.0096, stud_radius=0.0024,
                             stud_height=0.0017, top_thickness=0.001, clearance=0.0002, clutch_margin=0.00025, mass=0.0023,
                             rgba=[0.1, 0.3, 0.8, 1])
    assert len(b["stud_sites"]) == 8 and b["n_tubes"] == 3, b
    m = rest_on_floor(b["xml"], "lego")
    assert abs(m.body_mass[1] - 0.0023) < 2e-4, m.body_mass[1]

def one_by_three_large():
    out = tempfile.mkdtemp()
    b = construct_stud_brick(out, name="big13", studs_x=1, studs_y=3, pitch=0.03, body_height=0.027, stud_radius=0.0085,
                             stud_height=0.007, top_thickness=0.0025, clearance=0.0006, clutch_margin=0.0007, mass=0.02,
                             yaw_deg=30, lead_in_frac=0.7)
    assert len(b["stud_sites"]) == 3 and b["n_tubes"] == 2 and b["tube_radius_m"] > 0.0005, b
    rest_on_floor(b["xml"], "big13")

def misuse():
    out = tempfile.mkdtemp()
    for bad in ({"colour": [1, 0, 0, 1]}, {"stud_radius": 0.009, "pitch": 0.016}, {"studs_x": 0}, {"stud_height": 0.03}):
        try:
            construct_stud_brick(out, **bad)
        except ValueError:
            continue
        raise AssertionError(f"no ValueError for {bad}")

case("duplo 2x2 default compiles, rests flat, plausible mass", duplo)
case("lego 2x4 small pitch, 3 tubes, mass override", lego_2x4)
case("1x3 large brick with posts, yawed", one_by_three_large)
case("unknown or impossible parameters raise ValueError", misuse)
sys.exit(1 if fails else 0)
