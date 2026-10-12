"""joint_acceleration_audit: a 2 s arm move passes; the same move in 0.25 s fails on the joints and on the flange;
per-joint limits and misuse."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures" / "episode_audits"))
import synth as S
import mujoco
import numpy as np
from tool_library.verifiers.joint_acceleration_audit import joint_acceleration_audit

tmp = S.tmpdir()
scene = str(S.SETUP)
m = mujoco.MjModel.from_xml_path(scene)
q0 = S.home(m)
arm = [S.jadr(m, f"robot_right_arm_dof_joint{i}") for i in range(1, 7)]


def move(n_move, total=120):
    Q = np.repeat(q0[None], total, 0)
    a = 20
    Q[a:a + n_move, arm] = S.smooth(q0[arm], q0[arm] + np.array([0.5, 0.4, 0.4, 0, 0, 0]), n_move)
    Q[a + n_move:, arm] = Q[a + n_move - 1, arm]
    return Q


slow = S.save(tmp / "slow.npz", move(60), "right: reach #1")
fast = S.save(tmp / "fast.npz", move(8), "right: reach #1")
C = S.Cases()


def slow_passes():
    r = joint_acceleration_audit(scene, slow)
    assert r["ok"], r["issues"]
    assert max(v["peak_rad_s2"] for k, v in r["arms"]["right"].items() if isinstance(v, dict)) < 5, r["arms"]["right"]


def fast_fails():
    r = joint_acceleration_audit(scene, fast)
    assert not r["ok"] and any("dof_joint1" in s for s in r["issues"]), r["issues"]
    assert r["worst"]["side"] == "right" and r["worst"]["value"] > 15, r["worst"]
    r = joint_acceleration_audit(scene, fast, limit_rad_s2=500, flange_limit_m_s2=1.0)     # the payload rule alone
    assert not r["ok"] and r["worst"]["joint"] == "flange", r["worst"]


def per_joint_limits():
    r = joint_acceleration_audit(scene, fast, limit_rad_s2=[200, 200, 200, 200, 200, 200], flange_limit_m_s2=None)
    assert r["ok"], r["issues"]
    r = joint_acceleration_audit(scene, slow, limit_rad_s2=[0.5] * 6)
    assert not r["ok"], r["arms"]


def misuse():
    S.expect_value_error(lambda: joint_acceleration_audit(scene, slow, limit_rad_s2=[1, 2]), "limit_rad_s2")
    S.expect_value_error(lambda: joint_acceleration_audit(scene, slow, window_s=0), "window_s")


C.run("a 2 s arm move passes", slow_passes)
C.run("the same move in 0.27 s fails on joint 1; with a 1 m/s^2 payload limit the flange fails too", fast_fails)
C.run("per-joint limits move the verdict both ways", per_joint_limits)
C.run("a wrong-length limit list and a zero window are refused", misuse)
C.exit()
