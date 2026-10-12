"""joint_limit_margin_audit against the setup's real limits: in-range episodes pass; an arm past its real stop in a
widened scene fails on both counts; a widened range alone fails; a finger a degree past its stop passes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures" / "episode_audits"))
import synth as S
import mujoco
import numpy as np
from tool_library.verifiers.joint_limit_margin_audit import joint_limit_margin_audit

tmp = S.tmpdir()
scene = str(S.SETUP)
m = mujoco.MjModel.from_xml_path(scene)
q0 = S.home(m)
j4 = S.jadr(m, "robot_right_arm_dof_joint4")
thumb = S.jadr(m, "robot_right_right_thumb_IP")


def widen(spec):
    for j in spec.joints:
        if j.name in ("robot_right_arm_dof_joint4", "robot_left_arm_dof_joint4"):
            j.range = [-2.2, 2.2]


wide, mw = S.bench(tmp / "wide", [], edit=widen)
T = 60
Q = np.repeat(q0[None], T, 0)
Q[:, j4] = S.smooth([q0[j4]], [1.2], T)[:, 0]
good = S.save(tmp / "good.npz", Q, "right: reach #1")
Qp = Q.copy(); Qp[:, j4] = S.smooth([q0[j4]], [1.75], T)[:, 0]                 # 10 deg past the real 90 deg stop
past = S.save(tmp / "past.npz", Qp, "right: reach #1")
C = S.Cases()


def in_range_passes():
    r = joint_limit_margin_audit(scene, good)
    assert r["ok"], r["issues"]
    row = r["joints"]["robot_right_arm_dof_joint4"]
    assert row["reference_joint"] == "robot_right_arm_dof_joint4" and abs(row["reference_range"][1] - 90.0) < 0.01, row


def past_the_stop_in_widened_scene_fails():
    r = joint_limit_margin_audit(wide, past)
    assert not r["ok"]
    v = r["worst"]
    assert v["joint"] == "robot_right_arm_dof_joint4" and v["which"] == "upper" and 9.5 < v["excess"] < 10.5, v
    assert "robot_right_arm_dof_joint4" in r["widened"], r["widened"]
    assert any("widened" in s for s in r["issues"]), r["issues"]


def widened_range_alone_fails():
    r = joint_limit_margin_audit(wide, good)
    assert not r["ok"] and not r["violations"] and r["widened"], r
    r2 = joint_limit_margin_audit(wide, good, check_scene_ranges=False)
    assert r2["ok"], r2["issues"]


def finger_soft_limit_tolerated():
    Qf = Q.copy(); Qf[:, thumb] = -np.radians(1.0)                               # a contact presses the thumb 1 deg past 0
    r = joint_limit_margin_audit(scene, S.save(tmp / "finger.npz", Qf, "right: grip #1"))
    assert r["ok"], r["issues"]
    Qf[:, thumb] = -np.radians(3.0)
    r = joint_limit_margin_audit(scene, S.save(tmp / "finger3.npz", Qf, "right: grip #1"))
    assert not r["ok"] and r["worst"]["joint"] == "robot_right_right_thumb_IP", r["worst"]


def misuse():
    S.expect_value_error(lambda: joint_limit_margin_audit(scene, good, reference=str(tmp / "nope.xml")), "reference")
    S.expect_value_error(lambda: joint_limit_margin_audit(scene, good, tol_deg=0), "tol_deg")


C.run("an arm inside the real range passes and is matched to the setup joint", in_range_passes)
C.run("j4 10 deg past its real stop in a scene widened to +-126 deg fails twice", past_the_stop_in_widened_scene_fails)
C.run("a widened scene range alone fails (and passes with check_scene_ranges=False)", widened_range_alone_fails)
C.run("a finger 1 deg past its stop passes, 3 deg fails", finger_soft_limit_tolerated)
C.run("a missing reference and a zero tolerance are refused", misuse)
C.exit()
