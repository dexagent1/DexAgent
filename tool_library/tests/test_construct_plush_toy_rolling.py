import os, sys, tempfile, math
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco, numpy as np
from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.skills.construct_plush_toy_rolling import construct_plush_toy_rolling


def drop_rocks(xml, rpy=(35.0, 15.0, 30.0)):
    """Drop the toy tilted onto the floor; peak rotation (deg) during 1 s after 1.5 s of settling."""
    m = mujoco.MjModel.from_xml_path(xml); d = mujoco.MjData(m)
    q = np.zeros(4); mujoco.mju_euler2Quat(q, np.radians(rpy), "XYZ")
    d.qpos[2] = 0.08; d.qpos[3:7] = q
    for _ in range(int(1.5 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    q0 = d.qpos[3:7].copy(); pk = 0.0
    for _ in range(int(1.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
        pk = max(pk, math.degrees(2 * math.acos(min(1.0, abs(float(np.dot(q0, d.qpos[3:7])))))))
    return pk, d


fails = 0
cases = [
    ("toy_in_drawer_size_explicit_roll", dict(length=0.064, width=0.0425, height=0.038, mass=0.03, rolling_friction=0.005)),
    ("flattened_in_drawer", dict(length=0.075, width=0.06, height=0.03, n_ears=2, mass=0.025)),
    ("large_no_head", dict(length=0.25, width=0.15, height=0.12, head_frac=0.0, n_ears=0, mass=0.2, accent_rgba=None)),
    ("one_ear_yawed_tall", dict(length=0.1, width=0.05, height=0.06, n_ears=1, yaw_deg=40)),
]
with tempfile.TemporaryDirectory() as td:
    for name, kw in cases:
        try:
            out = construct_plush_toy_rolling(os.path.join(td, name), include_floor=True, **kw)
            m = mujoco.MjModel.from_xml_path(out["xml"])
            bid = m.body(out["body"]).id
            col = [g for g in range(m.ngeom) if m.geom_contype[g] and m.geom_bodyid[g] == bid]
            assert col, "no collision geoms on the toy"
            assert all(int(m.geom_condim[g]) == 6 for g in col), "collision geoms must be condim 6"
            roll = kw.get("rolling_friction", round(min(0.03, 0.2 * min(kw["width"], kw["height"])), 5))
            assert all(abs(m.geom_friction[g][2] - roll) < 1e-9 for g in col), "rolling friction not written"
            assert abs(out["mass_kg"] - kw.get("mass", 0.03)) < 1e-6
            pk, d = drop_rocks(out["xml"])
            assert np.all(np.isfinite(d.qpos)), "non-finite state"
            assert pk < 2.0, f"still rocking after the drop: {pk:.1f} deg in 1 s"
            assert np.linalg.norm(d.qvel) < 0.05, "not at rest"
            print(f"[PASS] {name} size={out['size_m']} rock_after_drop={pk:.2f} deg")
        except Exception as e:
            fails += 1; print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    try:     # the reason for the fork: without rolling resistance the same toy keeps rocking
        out = construct_plush_toy_rolling(os.path.join(td, "noroll"), include_floor=True, length=0.064, width=0.0425, height=0.038, rolling_friction=0.0)
        pk, _ = drop_rocks(out["xml"])
        assert pk > 2.0, f"expected rocking without rolling friction, got {pk:.2f} deg"
        print(f"[PASS] zero_rolling_friction_rocks ({pk:.1f} deg)")
    except Exception as e:
        fails += 1; print(f"[FAIL] zero_rolling_friction_rocks: {e}")
    for name, kw in (("unknown_parameter_rejected", dict(colour=[1, 0, 0, 1])), ("rolling_out_of_range_rejected", dict(rolling_friction=0.5))):
        try:
            construct_plush_toy_rolling(td, **kw); fails += 1; print(f"[FAIL] {name}: accepted")
        except ValueError:
            print(f"[PASS] {name}")
sys.exit(1 if fails else 0)
