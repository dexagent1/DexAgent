import sys, tempfile, math
import numpy as np
import mujoco
from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.skills.construct_tape_dispenser import construct_tape_dispenser

fails = 0


def check(case, cond, msg=""):
    global fails
    if cond:
        print(f"[PASS] {case}")
    else:
        fails += 1
        print(f"[FAIL] {case} {msg}")


def extent(m, d):
    lo, hi = np.full(3, 1e9), np.full(3, -1e9)
    for g in range(m.ngeom):
        if m.geom_contype[g] == 0 or m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        R = d.geom_xmat[g].reshape(3, 3); c = d.geom_xpos[g]; s = m.geom_size[g]
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * s
        else:   # cylinder: sample its rim
            a = np.linspace(0, 2 * np.pi, 24)
            corners = np.concatenate([np.stack([s[0] * np.cos(a), s[0] * np.sin(a), np.full_like(a, h)], 1) for h in (-s[1], s[1])])
        P = corners @ R.T + c
        lo, hi = np.minimum(lo, P.min(0)), np.maximum(hi, P.max(0))
    return lo, hi


def under_clamp_clear(m, d, b):
    """Ray straight up from the clamp site must stay clear of the dispenser up to clamp_gap."""
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, b["sites"]["clamp"])
    p = d.site_xpos[sid].copy()
    gid = np.array([-1], dtype=np.int32)
    dist = mujoco.mj_ray(m, d, p, np.array([0, 0, 1.0]), None, 1, -1, gid)
    return dist, p


with tempfile.TemporaryDirectory() as tmp:
    # 1) default dispenser, fixed: extents equal the requested size, clamp gap free under the arm
    b = construct_tape_dispenser(f"{tmp}/a")
    m = mujoco.MjModel.from_xml_path(b["xml"]); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    lo, hi = extent(m, d)
    size = hi - lo
    check("default size matches length/width/height",
          np.allclose(size, [0.15, 0.06, 0.075], atol=0.0015) and abs(lo[2]) < 1e-6 and m.njnt == 0, f"size {size} lo {lo}")
    dist, p = under_clamp_clear(m, d, b)
    check("default clamp gap is free of geometry", dist < 0 or dist >= 0.012 - 0.0015, f"ray hit at {dist}")

    # 2) narrow tall variant, free body on a floor: settles upright and stays put
    b2 = construct_tape_dispenser(f"{tmp}/b", length=0.12, width=0.045, height=0.07, nose_frac=0.25, clamp_gap=0.008,
                                  fixed=False, include_floor=True, mass=0.4, yaw_deg=30)
    m2 = mujoco.MjModel.from_xml_path(b2["xml"]); d2 = mujoco.MjData(m2)
    mujoco.mj_forward(m2, d2)
    p0 = d2.qpos[:3].copy()
    for _ in range(int(1.5 / m2.opt.timestep)):
        mujoco.mj_step(m2, d2)
    q = d2.qpos[3:7]
    tilt = math.degrees(2 * math.acos(min(1.0, math.sqrt(q[0] ** 2 + q[3] ** 2))))
    check("free narrow variant rests upright on a floor",
          np.linalg.norm(d2.qpos[:2] - p0[:2]) < 0.003 and tilt < 2.0 and np.isfinite(d2.qpos).all() and b2["joint"] is not None,
          f"drift {np.linalg.norm(d2.qpos[:2] - p0[:2])} tilt {tilt}")
    mujoco.mj_forward(m2, d2)
    lo2, hi2 = extent(m2, d2)
    check("narrow variant height and mass", abs((hi2 - lo2)[2] - 0.07) < 0.0015 and abs(m2.body_subtreemass[1] - 0.4) < 1e-6,
          f"h {(hi2 - lo2)[2]} mass {m2.body_subtreemass[1]}")

    # 3) long dispenser with a small explicit roll and a cord lying in the clamp gap: no contact with the cord
    b3 = construct_tape_dispenser(f"{tmp}/c", length=0.2, width=0.07, height=0.09, roll_radius=0.025, roll_width=0.03,
                                  nose_frac=0.4, clamp_gap=0.01)
    xml = open(b3["xml"]).read()
    cx = b3["clamp_local"][0]
    cord = (f'<body name="cord" pos="{cx} 0 0.0015"><freejoint/><geom type="capsule" fromto="0 -0.05 0 0 0.05 0" '
            f'size="0.0015" mass="0.001"/></body>')
    xml = xml.replace("  </worldbody>", f"    {cord}\n    <geom name='floor' type='plane' size='1 1 0.01'/>\n  </worldbody>")
    m3 = mujoco.MjModel.from_xml_string(xml); d3 = mujoco.MjData(m3); mujoco.mj_forward(m3, d3)
    disp = mujoco.mj_name2id(m3, mujoco.mjtObj.mjOBJ_BODY, b3["body"]); cb = mujoco.mj_name2id(m3, mujoco.mjtObj.mjOBJ_BODY, "cord")
    touching = any({m3.geom_bodyid[c.geom1], m3.geom_bodyid[c.geom2]} == {disp, cb} for c in d3.contact[:d3.ncon])
    check("a cord fits in the clamp gap of a long variant", not touching and b3["roll_radius_m"] == 0.025)

    # 4) misuse is refused with a message naming the parameter
    for case, kw, word in (("unknown parameter refused", {"colour": [1, 0, 0, 1]}, "unknown"),
                           ("oversized roll refused", {"roll_radius": 0.05}, "roll_radius"),
                           ("overlong nose refused", {"nose_frac": 0.6, "length": 0.08}, "nose_frac")):
        try:
            construct_tape_dispenser(f"{tmp}/x", **kw)
            check(case, False, "no error")
        except ValueError as e:
            check(case, word in str(e), str(e))

sys.exit(1 if fails else 0)
