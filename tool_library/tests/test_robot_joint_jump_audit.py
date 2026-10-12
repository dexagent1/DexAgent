"""robot_joint_jump_audit on the bench: smooth arm and finger motion passes; a seam step, a glitch frame and a
floating-wrist glitch fail."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures" / "episode_audits"))
import synth as S
import numpy as np
from tool_library.verifiers.robot_joint_jump_audit import robot_joint_jump_audit

tmp = S.tmpdir()
scene = str(S.SETUP)
import mujoco
m = mujoco.MjModel.from_xml_path(scene)
q0 = S.home(m)
arm = [S.jadr(m, f"robot_right_arm_dof_joint{i}") for i in range(1, 7)]
fing = [S.jadr(m, n) for n in ("robot_right_right_index_MCP_FE", "robot_right_right_middle_MCP_FE", "robot_right_right_thumb_IP")]
T = 150
Q = np.repeat(q0[None], T, 0)
Q[:90, arm] = S.smooth(q0[arm], q0[arm] + 0.3, 90)
Q[90:, arm] = Q[89, arm]
Q[30:120, fing] = S.smooth(q0[fing], q0[fing] + 0.8, 90)
Q[120:, fing] = Q[119, fing]
good = S.save(tmp / "good.npz", Q, "right: reach for the box #1")
C = S.Cases()


def smooth_passes():
    r = robot_joint_jump_audit(scene, good)
    assert r["ok"], r["issues"]
    assert r["groups"]["right_arm"]["max_isolated_ratio"] < 10, r["groups"]


def seam_step_fails():
    Qb = Q.copy(); Qb[100:, arm[1]] += np.radians(3.0)            # a concatenation seam on a still arm
    r = robot_joint_jump_audit(scene, S.save(tmp / "seam.npz", Qb, "right: hold #2"))
    assert not r["ok"] and r["worst"]["kind"] == "isolated_step" and r["worst"]["frame"] == 100, r["worst"]
    assert "right_arm" in r["issues"][0], r["issues"]


def finger_glitch_fails():
    Qb = Q.copy(); Qb[60, fing[0]] += np.radians(12.0)             # one frame displaced and put back
    r = robot_joint_jump_audit(scene, S.save(tmp / "glitch.npz", Qb, "right: grip #1"))
    assert not r["ok"] and any(e["kind"] == "glitch" and e["frame"] == 60 for e in r["events"]), r["events"][:3]


def wrist_slide_glitch_fails():
    Qb = Q.copy(); Qb[70, S.wrist_adr(m, "right")[0]] += 0.006
    r = robot_joint_jump_audit(scene, S.save(tmp / "wrist.npz", Qb, "right: carry #1"))
    assert not r["ok"] and r["worst"]["group"] == "right_wrist" and r["worst"]["unit"] == "mm", r["worst"]
    r2 = robot_joint_jump_audit(scene, S.save(tmp / "wrist2.npz", Qb, "right: carry #1"), min_step={"wrist_mm": 8.0})
    assert r2["ok"], r2["issues"]


def misuse():
    S.expect_value_error(lambda: robot_joint_jump_audit(scene, good, min_step={"elbow": 1}), "min_step")
    S.expect_value_error(lambda: robot_joint_jump_audit(scene, good, ratio=0), "ratio")
    S.expect_value_error(lambda: robot_joint_jump_audit(scene, S.save(tmp / "w.npz", Q[:, :10], "x: y")), "qpos columns")


C.run("smooth arm and finger motion passes", smooth_passes)
C.run("a 3 deg seam step on a still arm fails at its frame", seam_step_fails)
C.run("a finger glitch frame (12 deg out and back) fails", finger_glitch_fails)
C.run("a 6 mm floating-wrist glitch fails, passes with an 8 mm floor", wrist_slide_glitch_fails)
C.run("unknown floor key, zero ratio and a mismatched trajectory are refused", misuse)
C.exit()
