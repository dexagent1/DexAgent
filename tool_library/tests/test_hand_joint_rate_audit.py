"""hand_joint_rate_audit: smooth finger closing passes; a chattering finger and a full-speed reversal fail."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures" / "episode_audits"))
import synth as S
import mujoco
import numpy as np
from tool_library.verifiers.hand_joint_rate_audit import hand_joint_rate_audit

tmp = S.tmpdir()
scene = str(S.SETUP)
m = mujoco.MjModel.from_xml_path(scene)
q0 = S.home(m)
f = [S.jadr(m, n) for n in ("robot_left_left_index_MCP_FE", "robot_left_left_middle_MCP_FE", "robot_left_left_ring_MCP_FE")]
T = 120
Q = np.repeat(q0[None], T, 0)
Q[20:80, f] = S.smooth(q0[f], q0[f] + 1.0, 60)
Q[80:, f] = Q[79, f]
good = S.save(tmp / "good.npz", Q, "left: close on the box #1")
C = S.Cases()


def smooth_passes():
    r = hand_joint_rate_audit(scene, good)
    assert r["ok"], r["issues"]
    assert r["hands"]["left"]["max_reversals_per_window"]["value"] == 0, r["hands"]["left"]


def chatter_fails():
    Qc = Q.copy()
    Qc[85:115, f[0]] += 0.03 * (-1) ** np.arange(30)               # a finger buzzing against its contact
    r = hand_joint_rate_audit(scene, S.save(tmp / "chatter.npz", Qc, "left: grip the box #1"))
    assert not r["ok"] and r["worst"]["kind"] == "chatter" and r["worst"]["joint"] == "robot_left_left_index_MCP_FE", r["worst"]


def reversal_fails():
    Qr = Q.copy()
    Qr[90, f[1]] += 0.25                                           # out at full speed and straight back
    r = hand_joint_rate_audit(scene, S.save(tmp / "rev.npz", Qr, "left: grip the box #1"))
    assert not r["ok"] and any("rad/s^2" in s for s in r["issues"]), r["issues"]
    assert hand_joint_rate_audit(scene, S.save(tmp / "rev2.npz", Qr, "left: grip"), max_acc_rad_s2=1000)["ok"]


def misuse():
    S.expect_value_error(lambda: hand_joint_rate_audit(scene, good, max_reversals=0), "max_reversals")
    S.expect_value_error(lambda: hand_joint_rate_audit(scene, good, frames=[0, 500]), "frames")


C.run("a smooth finger close passes", smooth_passes)
C.run("a finger oscillating 0.03 rad every frame fails as chatter", chatter_fails)
C.run("a 0.25 rad out-and-back in one frame fails on acceleration (passes with a 1000 rad/s^2 limit)", reversal_fails)
C.run("zero reversal budget and an out-of-range window are refused", misuse)
C.exit()
