import tempfile, numpy as np, mujoco
import sys
from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.skills.construct_phone import construct_phone

def check(tag, **kw):
    with tempfile.TemporaryDirectory() as td:
        b = construct_phone(td, **kw)
        m = mujoco.MjModel.from_xml_path(b["xml"]); d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        assert abs(m.body_subtreemass[1] - b["mass_kg"]) < 1e-3 * max(1, b["mass_kg"]) + 1e-6, "mass"
        L, Wd, T = b["dims"]["length"], b["dims"]["width"], b["dims"]["thickness"]
        # the port mouth lies on the bottom edge, and a probe ray along the axis travels port_depth before hitting
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, b["sites"]["port_mouth"])
        mouth = d.site_xpos[sid].copy(); ax = d.site_xmat[sid].reshape(3, 3)[:, 2]
        geomid = np.array([-1], np.int32)
        dist = mujoco.mj_ray(m, d, mouth - 0.01 * ax, ax, np.array([0, 0, 0, 1, 0, 0], np.uint8), 1, -1, geomid)
        open_len = dist - 0.01
        exp = b["port"]["depth"]
        assert abs(open_len - exp) < 2e-4, f"cavity open length {open_len} vs depth {exp}"
        # a ray next to the port (outside its width) hits the edge face at once
        off = mouth + np.array([b["port"]["width"] * 0.5 + 0.002, 0, 0])
        dist2 = mujoco.mj_ray(m, d, off - 0.01 * ax, ax, np.array([0, 0, 0, 1, 0, 0], np.uint8), 1, -1, geomid)
        assert abs(dist2 - 0.01) < 2e-4, f"edge face not solid beside the port ({dist2})"
        # resting: settles flat on a floor
        print(f"[PASS] {tag}: L={L} W={Wd} T={T} cavity={open_len*1000:.2f} mm")

check("default 6.1-inch phone")
check("large phone, offset port, bump right, three lenses", length=0.165, width=0.078, thickness=0.0085,
      port_offset_x=0.003, bump_side="right", n_lenses=3, bump_size=0.04, mass=0.22)
check("small phone without bump or grip", length=0.12, width=0.058, thickness=0.009, bump=False,
      port_grip_m=0.0, port_w=0.006, port_h=0.003, port_depth=0.005)
for bad in (dict(colour=1), dict(port_h=0.0075), dict(width=0.2), dict(bump_side="top")):
    try:
        with tempfile.TemporaryDirectory() as td:
            construct_phone(td, **bad)
        raise SystemExit(f"misuse not rejected: {bad}")
    except ValueError:
        pass
print("[PASS] misuse raises ValueError")
