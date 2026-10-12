import os, sys, tempfile
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco, numpy as np
from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.skills.construct_plush_toy import construct_plush_toy

def settle(xml, secs=1.5):
    m = mujoco.MjModel.from_xml_path(xml); d = mujoco.MjData(m)
    z0 = None
    for _ in range(int(secs / m.opt.timestep)):
        mujoco.mj_step(m, d)
    return m, d

fails = 0
cases = [
    ("fist_size_default", dict()),
    ("flattened_in_drawer", dict(length=0.075, width=0.06, height=0.03, n_ears=2, mass=0.025)),
    ("large_no_head_one_ear", dict(length=0.25, width=0.15, height=0.12, head_frac=0.0, n_ears=0, mass=0.2, accent_rgba=None)),
    ("one_ear_yawed", dict(length=0.1, width=0.05, height=0.06, n_ears=1, yaw_deg=40, pos=[0.1, -0.05, 0.0])),
]
with tempfile.TemporaryDirectory() as td:
    for name, kw in cases:
        try:
            out = construct_plush_toy(os.path.join(td, name), include_floor=True, **kw)
            p = dict(length=0.08, width=0.06, height=0.05); p.update({k: v for k, v in kw.items() if k in p})
            sz = out["size_m"]
            assert abs(sz[2] - p["height"]) < 0.15 * p["height"] + 1e-3, f"height {sz[2]} vs {p['height']}"
            assert max(sz[0], sz[1]) <= p["length"] * 1.1 + 1e-3, f"length {sz}"
            assert abs(out["mass_kg"] - kw.get("mass", 0.03)) < 1e-6
            m, d = settle(out["xml"])
            assert np.all(np.isfinite(d.qpos)), "non-finite state"
            z = d.qpos[2]; assert -0.012 < z < 0.01, f"settled base z {z}"
            assert np.linalg.norm(d.qvel) < 0.05, "not at rest"
            print(f"[PASS] {name} size={sz} mass={out['mass_kg']:.3f} z={z:.4f}")
        except Exception as e:
            fails += 1; print(f"[FAIL] {name}: {e}")
    try:
        construct_plush_toy(td, colour=[1, 0, 0, 1]); fails += 1; print("[FAIL] unknown param accepted")
    except ValueError:
        print("[PASS] unknown_parameter_rejected")
sys.exit(1 if fails else 0)
