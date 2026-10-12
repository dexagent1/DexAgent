
import sys, tempfile, mujoco, numpy as np
from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.skills.construct_bottle_straw import construct_bottle_straw
fails = 0
def case(tag, fn):
    global fails
    try:
        fn(); print(f"[PASS] {tag}")
    except Exception as e:
        fails += 1; print(f"[FAIL] {tag}: {e}")
def check_straw(params, L):
    with tempfile.TemporaryDirectory() as td:
        b = construct_bottle_straw(td, **params)
        m = mujoco.MjModel.from_xml_path(b["xml"]); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        if L == 0:
            assert b["straw"] is None; return
        g = m.geom(b["straw"]["geom"]); assert m.geom_bodyid[g.id] == m.body(b["bodies"]["cap"]).id
        assert m.geom_contype[g.id] == 0 and m.geom_conaffinity[g.id] == 0
        assert abs(2 * m.geom_size[g.id][1] - L) < 1e-5
        zb = d.geom_xpos[g.id][2] - m.geom_size[g.id][1]
        assert abs(zb - b["straw"]["bottom_z_seated_m"]) < 1e-3, (zb, b["straw"])
        assert b["straw"]["straw_clear_lift_m"] > L * 0.5
        mujoco.mj_step(m, d, 200); assert np.all(np.isfinite(d.qpos))
case("default bottle with 0.12 m straw", lambda: check_straw({"straw_length": 0.12, "include_floor": True}, 0.12))
case("wide sports bottle, free body, short straw", lambda: check_straw({"body_radius": 0.033, "body_height": 0.11, "neck_radius": 0.0255, "cap_radius": 0.0285, "cap_height": 0.04, "cap_top_wall": 0.024, "neck_height": 0.018, "shoulder_height": 0.012, "thread_pitch": 0.0055, "thread_turns": 3.0, "bottle_fixed": False, "straw_length": 0.06}, 0.06))
case("no straw = plain bottle", lambda: check_straw({"straw_length": 0.0}, 0))
def too_long():
    with tempfile.TemporaryDirectory() as td:
        try:
            construct_bottle_straw(td, straw_length=0.5)
        except ValueError as e:
            assert "straw_length" in str(e); return
        raise AssertionError("no ValueError for an over-long straw")
case("over-long straw refused", too_long)
def unknown():
    with tempfile.TemporaryDirectory() as td:
        try:
            construct_bottle_straw(td, straw_lenght=0.1)
        except (ValueError, KeyError, TypeError):
            return
        raise AssertionError("unknown parameter accepted")
case("unknown parameter refused", unknown)
sys.exit(1 if fails else 0)
