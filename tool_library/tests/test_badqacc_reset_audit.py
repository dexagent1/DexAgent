"""badqacc_reset_audit: an ordinary episode passes and so does fingers interpolating onto qpos0; a frame snapped to
qpos0, a frame snapped back to the keyframe and a frozen stretch across phases fail."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures" / "episode_audits"))
import synth as S
import numpy as np
from tool_library.verifiers.badqacc_reset_audit import badqacc_reset_audit

tmp = S.tmpdir()
scene, m = S.bench(tmp, [dict(name="box", size=[0.03, 0.03, 0.03], pos=[0.0, 0.45, 0.03])])
q0 = S.home(m)
arm = [S.jadr(m, f"robot_right_arm_dof_joint{i}") for i in range(1, 7)]
wr = S.wrist_adr(m, "right")
fing = [S.jadr(m, n) for n in ("robot_right_right_index_MCP_FE", "robot_right_right_middle_MCP_FE", "robot_right_right_ring_MCP_FE",
                               "robot_right_right_pinky_MCP_FE", "robot_right_right_index_PIP", "robot_right_right_middle_PIP")]
T = 100
Q = np.repeat(q0[None], T, 0)
Q[:, arm] = S.smooth(q0[arm], q0[arm] + 0.2, T)
Q[:, wr[:3]] = S.smooth(q0[wr[:3]], q0[wr[:3]] + [0.05, 0.1, 0.0], T)
Q[:50, fing] = S.smooth(q0[fing], q0[fing] + 0.6, 50); Q[50:, fing] = q0[fing] + 0.6
Q += 1e-7 * np.random.default_rng(0).standard_normal(Q.shape)        # a physics run never repeats exactly
L = ["right: reach #1"] * 50 + ["right: grip box #1"] * 50
good = S.save(tmp / "good.npz", Q, L)
C = S.Cases()


def ordinary_passes():
    r = badqacc_reset_audit(scene, good)
    assert r["ok"], r["issues"]
    Qi = Q.copy(); Qi[60:80, fing] = S.smooth(q0[fing] + 0.6, np.zeros(len(fing)), 20); Qi[80:, fing] = 0.0   # opened onto 0
    assert badqacc_reset_audit(scene, S.save(tmp / "open.npz", Qi, L))["ok"]


def snap_to_qpos0_fails():
    Qb = Q.copy(); Qb[70] = m.qpos0
    r = badqacc_reset_audit(scene, S.save(tmp / "q0.npz", Qb, L))
    assert not r["ok"] and r["worst"]["frame"] == 70 and r["worst"]["reference"] == "qpos0" and r["worst"]["robot_snapped"] > 6, r["worst"]


def snap_to_keyframe_fails():
    Qb = Q.copy(); Qb[80:] = q0                                      # back to the keyframe at once, and stuck there
    Qb[80:] += 1e-9 * np.random.default_rng(1).standard_normal(Qb[80:].shape)
    Qb[80] = q0
    r = badqacc_reset_audit(scene, S.save(tmp / "key.npz", Qb, L))
    assert not r["ok"] and r["worst"]["reference"].startswith("keyframe"), r["worst"]


def frozen_stretch_fails():
    Qf = Q.copy(); Qf[45:60] = Qf[45]                                # the guard copied the last state across a phase change
    r = badqacc_reset_audit(scene, S.save(tmp / "frozen.npz", Qf, L))
    assert not r["ok"] and r["frozen_runs"][0]["first"] == 45 and r["frozen_runs"][0]["crosses_phases"], r["frozen_runs"]


def misuse():
    S.expect_value_error(lambda: badqacc_reset_audit(scene, good, min_frac=2), "min_frac")


C.run("an ordinary episode, and fingers interpolated onto qpos0, pass", ordinary_passes)
C.run("a frame snapped to qpos0 fails", snap_to_qpos0_fails)
C.run("a frame snapped back to the keyframe fails", snap_to_keyframe_fails)
C.run("a bit-identical stretch across a phase change fails", frozen_stretch_fails)
C.run("a share above 1 is refused", misuse)
C.exit()
