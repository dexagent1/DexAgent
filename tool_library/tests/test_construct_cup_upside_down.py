import sys, tempfile
import numpy as np
import mujoco
from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.skills.construct_cup_upside_down import construct_cup_upside_down

def check(case, **p):
    d = tempfile.mkdtemp()
    b = construct_cup_upside_down(d, include_floor=True, **p)
    m = mujoco.MjModel.from_xml_path(b["xml"]); data = mujoco.MjData(m)
    mujoco.mj_forward(m, data)
    base = data.geom(b["body"] + "_base")
    assert abs(base.xpos[2] - (b["top_z_m"] - 0.5 * b.get("bottom_inner_z_m", 0.003))) < 0.004, (case, base.xpos)
    zs = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == m.body(b["body"]).id and m.geom_contype[g]:
            zs.append(data.geom_xpos[g][2])
    assert min(zs) >= -1e-3, (case, "geom below z=0")
    q0 = data.qpos[:7].copy()
    data.qpos[2] += 0.003
    for _ in range(1000):
        mujoco.mj_step(m, data)
    drift = np.linalg.norm(data.qpos[:2] - q0[:2])
    up = np.zeros(9); mujoco.mju_quat2Mat(up, data.qpos[3:7]); tilt = np.degrees(np.arccos(np.clip(up[8], -1, 1)))
    assert drift < 0.003 and tilt < 3 and abs(data.qpos[2]) < 0.004, (case, drift, tilt, data.qpos[2])
    print(f"[PASS] {case}: top {b['top_z_m']:.3f} m, settled drift {drift*1000:.2f} mm tilt {tilt:.2f} deg")

check("small stacking cup", base_radius=0.014, rim_radius=0.018, height=0.04)
check("tall tumbler", base_radius=0.025, rim_radius=0.035, height=0.11)
check("wide short cup", base_radius=0.03, rim_radius=0.04, height=0.05, mass=0.03)
try:
    construct_cup_upside_down(tempfile.mkdtemp(), bogus=1)
    print("[FAIL] unknown parameter accepted"); sys.exit(1)
except ValueError:
    print("[PASS] unknown parameter refused")
