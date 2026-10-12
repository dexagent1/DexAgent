"""Tests for construct_plug_cable: coil and centreline layouts, both plug ends, two connector sizes,
physical insertion of the built plug into a matching receptacle (and blocking off-axis), and misuse errors."""
import sys
import tempfile

import numpy as np
import mujoco

from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.skills.construct_plug_cable import construct_plug_cable

FAILS = []


def case(name):
    def deco(fn):
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception as e:                       # report and continue with the next case
            FAILS.append(name)
            print(f"[FAIL] {name}: {e!r}")
        return fn
    return deco


def _site(m, d, name):
    return d.site_xpos[m.site(name).id].copy()


@case("coil layout with tail, USB-C plug on end b, lies flat and settles on a floor")
def _():
    out = tempfile.mkdtemp()
    tail_end = [0.30, -0.02]
    b = construct_plug_cable(out, name="cc", segments=70, include_floor=True,
                             coil={"center": [0.15, 0.10], "rx": 0.10, "ry": 0.08, "yaw_deg": 20, "turns": 2.0,
                                   "pitch_m": 0.012, "winding": "cw", "tail_end": tail_end, "tail_dir": [0, -1]})
    P = b["plug"]
    assert abs(P["tip_w"] - 0.0082) < 1e-9 and abs(P["tip_h"] - 0.0024) < 1e-9, P
    assert b["min_self_gap_m"] >= 2 * b["radius_m"], b["min_self_gap_m"]
    assert abs(P["wide_axis"][2]) < 0.02, P["wide_axis"]
    assert P["min_z_m"] >= -1e-4, P["min_z_m"]
    end = np.array(b["centreline"][-1])
    exp = end + np.array(P["insert_dir"]) * (P["housing"][0] + P["tip_len"])
    assert np.linalg.norm(exp - np.array(P["tip_pos"])) < 1e-3, (exp, P["tip_pos"])
    assert np.linalg.norm(end[:2] - tail_end) < 2e-3, (end, tail_end)
    m = mujoco.MjModel.from_xml_path(b["xml"]); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    x0 = d.xpos.copy()
    for _ in range(int(0.6 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    assert np.isfinite(d.qpos).all()
    drift = float(np.abs(d.xpos - x0).max())
    assert drift < 0.003, drift


@case("straight centreline, micro-USB sized plug on end a points outward from the first vertex")
def _():
    out = tempfile.mkdtemp()
    b = construct_plug_cable(out, name="mu", segments=30, plug_end="a", port_w=0.0069, port_h=0.0018,
                             housing_w=0.0105, housing_h=0.005, housing_len=0.015, tip_len=0.006,
                             centreline=[[0.0, 0.0, 0.0016], [0.25, 0.05, 0.0016], [0.5, 0.0, 0.0016]])
    P = b["plug"]
    assert abs(P["tip_w"] - 0.0067) < 1e-9 and abs(P["tip_h"] - 0.0016) < 1e-9, P
    first, second = np.array(b["centreline"][0]), np.array(b["centreline"][1])
    out_dir = (first - second) / np.linalg.norm(first - second)
    assert float(np.dot(out_dir, P["insert_dir"])) > 0.98, (out_dir, P["insert_dir"])
    exp = first + np.array(P["insert_dir"]) * (0.015 + 0.006)
    assert np.linalg.norm(exp - np.array(P["tip_pos"])) < 1e-3, (exp, P["tip_pos"])
    assert abs(b["length_m"] - 0.5099) < 0.01, b["length_m"]


def _insertion_trial(offset_frac):
    out = tempfile.mkdtemp()
    b = construct_plug_cable(out, name="ins", segments=24, plug_flat=False, plug_solref=[0.001, 1.0],
                             centreline=[[0.0, 0.0, 0.05], [0.3, 0.0, 0.05]])
    P = b["plug"]
    spec = mujoco.MjSpec.from_file(b["xml"])
    spec.option.gravity = [0, 0, 0]
    spec.option.timestep = 0.0002
    dirv, wide = np.array(P["insert_dir"]), np.array(P["wide_axis"])
    thin = np.cross(dirv, wide)
    pw, ph, D, wall = 0.0084, 0.0026, 0.0068, 0.003
    mouth = np.array(P["tip_pos"]) + 0.003 * dirv + offset_frac * pw * wide
    Rm = np.column_stack([dirv, wide, thin])
    q = np.zeros(4); mujoco.mju_mat2Quat(q, Rm.ravel())
    rb = spec.worldbody.add_body(); rb.name = "receptacle"; rb.pos = mouth.tolist(); rb.quat = q.tolist()
    E = 0.02                                       # the receptacle face extends 2 cm around the opening
    Wy, Wz = E - pw / 2, E - ph / 2
    boxes = [((D / 2, pw / 2 + Wy / 2, 0), (D / 2, Wy / 2, E)),
             ((D / 2, -pw / 2 - Wy / 2, 0), (D / 2, Wy / 2, E)),
             ((D / 2, 0, ph / 2 + Wz / 2), (D / 2, pw / 2, Wz / 2)),
             ((D / 2, 0, -ph / 2 - Wz / 2), (D / 2, pw / 2, Wz / 2)),
             ((D + wall / 2, 0, 0), (wall / 2, E, E))]
    for i, (c, h) in enumerate(boxes):
        g = rb.add_geom(); g.name = f"rec{i}"; g.type = mujoco.mjtGeom.mjGEOM_BOX; g.pos = list(c); g.size = list(h)
        g.solref = [0.001, 1.0]
    m = spec.compile(); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    pid = m.body(P["body"]).id
    for _ in range(int(0.5 / m.opt.timestep)):
        d.xfrc_applied[pid, :3] = 5.0 * dirv
        mujoco.mj_step(m, d)
    tip = _site(m, d, P["tip_site"])
    depth = float(np.dot(tip - (np.array(P["tip_pos"]) + 0.003 * dirv), dirv))
    return depth, D, np.isfinite(d.qpos).all()


@case("built plug inserts into a matching receptacle under a 5 N push and the housing stops at the face")
def _():
    depth, D, ok = _insertion_trial(0.0)
    assert ok
    assert 0.85 * D <= depth <= D + 0.0005, (depth, D)


@case("plug offset by one port width is blocked at the receptacle face")
def _():
    depth, D, ok = _insertion_trial(1.0)
    assert ok
    assert depth < 0.0005, depth


@case("misuse raises ValueError naming the parameter")
def _():
    out = tempfile.mkdtemp()
    bad = [dict(colour=[1, 0, 0, 1]),
           dict(tip_clearance_m=0.002),
           dict(housing_w=0.008),
           dict(coil={"center": [0, 0], "rx": 0.1, "ry": 0.1, "pitch_m": 0.002}),
           dict(coil={"center": [0, 0], "rx": 0.1, "ry": 0.1}, centreline=[[0, 0, 0], [1, 0, 0]]),
           dict(plug_end="c")]
    for kw in bad:
        try:
            construct_plug_cable(out, **kw)
        except ValueError:
            continue
        raise AssertionError(f"no ValueError for {kw}")


@case("plug geoms ride on the end link: no extra body, every cord body after the first has a joint")
def _():
    out = tempfile.mkdtemp()
    b = construct_plug_cable(out, name="el", segments=20, centreline=[[0.0, 0.0, 0.0016], [0.4, 0.0, 0.0016]])
    m = mujoco.MjModel.from_xml_path(b["xml"])
    P = b["plug"]
    assert P["body"] == b["bodies"]["last"], P["body"]
    assert m.geom_bodyid[m.geom(P["tip_geom"]).id] == m.body(P["body"]).id
    assert m.geom_bodyid[m.geom(P["housing_geom"]).id] == m.body(P["body"]).id
    cord = [m.body(i).name for i in range(m.nbody) if m.body(i).name.startswith("el") and int(m.body_mocapid[i]) < 0]
    jointless = [n for n in cord if n != b["bodies"]["first"] and int(m.body_jntnum[m.body(n).id]) == 0]
    assert not jointless, jointless
    assert len(cord) == 19, len(cord)


@case("deformable_topology_check accepts the cable as one jointed strand (the child plug body was a fused rigid link)")
def _():
    from tool_library.verifiers.deformable_topology_check import deformable_topology_check
    out = tempfile.mkdtemp()
    b = construct_plug_cable(out, name="tp", segments=24, centreline=[[0.0, 0.0, 0.0016], [0.3, 0.05, 0.0016]])
    r = deformable_topology_check(b["xml"])
    assert r["ok"] and r["ropes"][0]["fused_links"] == [] and r["ropes"][0]["strands"] == 1, r["issues"]


if FAILS:
    print("FAILED:", FAILS)
    sys.exit(1)
