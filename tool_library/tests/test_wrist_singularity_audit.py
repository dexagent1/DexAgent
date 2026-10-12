"""wrist_singularity_audit: fast motion away from singularities and slow motion through one pass; fast joint motion
at the wrist singularity (joint 5 near 86 deg) fails."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures" / "episode_audits"))
import synth as S
import mujoco
import numpy as np
from tool_library.verifiers.wrist_singularity_audit import wrist_singularity_audit

tmp = S.tmpdir()
scene = str(S.SETUP)
m = mujoco.MjModel.from_xml_path(scene)
q0 = S.home(m)
arm = [S.jadr(m, f"robot_right_arm_dof_joint{i}") for i in range(1, 7)]
qs = q0.copy(); qs[arm[4]] = 1.5; qs[arm[2]] = 0.5            # sigma_min about 0.0002 m


def sweep(base, n, d4=0.0, d6=0.0, total=60):
    Q = np.repeat(base[None], total, 0)
    a = 10
    Q[a:a + n, arm[3]] = base[arm[3]] + np.linspace(0, d4, n)
    Q[a:a + n, arm[5]] = base[arm[5]] + np.linspace(0, d6, n)
    Q[a + n:, arm[3]] = base[arm[3]] + d4; Q[a + n:, arm[5]] = base[arm[5]] + d6
    return Q


C = S.Cases()


def fast_away_from_singularity_passes():
    r = wrist_singularity_audit(scene, S.save(tmp / "a.npz", sweep(q0, 12, d6=1.0), "right: turn #1"))
    assert r["ok"] and r["arms"]["right"]["min_sigma_m"] > 0.01, r


def slow_through_singularity_passes():
    r = wrist_singularity_audit(scene, S.save(tmp / "b.npz", sweep(qs, 45, d4=0.3, d6=-0.3), "right: turn #1"))
    assert r["ok"] and r["arms"]["right"]["frames_near_singular"] > 30, r


def fast_at_singularity_fails():
    r = wrist_singularity_audit(scene, S.save(tmp / "c.npz", sweep(qs, 8, d4=0.6, d6=-0.6), "right: turn #1"))
    assert not r["ok"], r["arms"]
    w = r["worst"]
    assert w["side"] == "right" and w["sigma_min_m"] < 0.01 and (w["joint_rate_deg_s"] > 90 or w["amplification"] > 60), w


def misuse():
    S.expect_value_error(lambda: wrist_singularity_audit(scene, S.save(tmp / "d.npz", sweep(q0, 10), "x: y"), smin_m=0), "smin_m")


C.run("a fast wrist turn far from any singularity passes", fast_away_from_singularity_passes)
C.run("a slow counter-rotation at the wrist singularity passes", slow_through_singularity_passes)
C.run("the same counter-rotation at 130 deg/s fails", fast_at_singularity_fails)
C.run("a zero singularity threshold is refused", misuse)
C.exit()
