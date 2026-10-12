"""object_teleport_audit on a box sliding across the bench: a smooth slide passes; a one-frame glitch, an orientation
flip, a snap back to the start pose, a non-unit quaternion and a jumping drawer joint fail."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures" / "episode_audits"))
import synth as S
import mujoco
import numpy as np
from tool_library.verifiers.object_teleport_audit import object_teleport_audit

tmp = S.tmpdir()


def add_drawer(spec):
    b = spec.worldbody.add_body(); b.name = "drawer"; b.pos = [0.25, 0.7, 0.04]
    j = b.add_joint(); j.name = "drawer_slide"; j.type = mujoco.mjtJoint.mjJNT_SLIDE; j.axis = [0, 1, 0]; j.range = [-0.2, 0.0]
    g = b.add_geom(); g.type = mujoco.mjtGeom.mjGEOM_BOX; g.size = [0.05, 0.05, 0.02]; g.mass = 0.3


scene, m = S.bench(tmp, [dict(name="box", size=[0.03, 0.03, 0.03], pos=[-0.05, 0.45, 0.03])], edit=add_drawer)
q0 = S.home(m)
a = S.free_adr(m, "box")
dj = S.jadr(m, "drawer_slide")
T = 90
Q = np.repeat(q0[None], T, 0)
Q[:, a] = q0[a] + S.smooth([0], [0.10], T)[:, 0]
Q[:, dj] = S.smooth([0], [-0.12], T)[:, 0]
good = S.save(tmp / "good.npz", Q, "right: push box #1")
C = S.Cases()


def smooth_passes():
    r = object_teleport_audit(scene, good)
    assert r["ok"], r["issues"]
    assert r["objects"]["box"]["max_glitch_mm"] < 1 and r["joints"]["drawer_slide"]["max_step"] < 5, r


def glitch_and_flip_fail():
    Qg = Q.copy(); Qg[40, a + 1] += 0.012                           # 12 mm off for one frame
    r = object_teleport_audit(scene, S.save(tmp / "glitch.npz", Qg, "right: push box #1"))
    assert not r["ok"] and r["worst"]["kind"] == "glitch" and r["worst"]["frame"] == 40, r["worst"]
    Qf = Q.copy(); Qf[50:, a + 3:a + 7] = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]   # 90 deg about z from frame 50 on
    r = object_teleport_audit(scene, S.save(tmp / "flip.npz", Qf, "right: push box #1"))
    assert not r["ok"] and r["worst"]["kind"] == "flip" and r["worst"]["frame"] == 50, r["worst"]


def reset_and_quaternion_fail():
    Qr = Q.copy(); Qr[70:, a:a + 7] = q0[a:a + 7]                   # back to where it started, in one frame
    r = object_teleport_audit(scene, S.save(tmp / "reset.npz", Qr, "right: push box #1"))
    assert not r["ok"] and r["worst"]["kind"] == "reset" and r["worst"]["frame"] == 70, r["worst"]
    Qn = Q.copy(); Qn[20, a + 3:a + 7] *= 1.05
    r = object_teleport_audit(scene, S.save(tmp / "norm.npz", Qn, "right: push box #1"))
    assert not r["ok"] and any(e["kind"] == "quat_norm" for e in r["events"]), r["events"]


def drawer_joint_jump_fails():
    Qd = Q.copy(); Qd[60:, dj] += 0.05                              # 50 mm in one frame
    r = object_teleport_audit(scene, S.save(tmp / "drawer.npz", Qd, "left: open drawer #2"))
    assert not r["ok"] and r["worst"]["body"] == "drawer_slide" and r["worst"]["kind"] == "joint_step", r["worst"]


def misuse():
    S.expect_value_error(lambda: object_teleport_audit(scene, good, glitch_m=-1), "glitch_m")
    S.expect_value_error(lambda: object_teleport_audit(str(S.SETUP), good), "qpos columns")


C.run("a box sliding 10 cm and a drawer opening smoothly pass", smooth_passes)
C.run("a 12 mm one-frame glitch and a 90 deg orientation flip fail at their frames", glitch_and_flip_fail)
C.run("a snap back to the frame-0 pose and a non-unit quaternion fail", reset_and_quaternion_fail)
C.run("a drawer slide jumping 50 mm in one frame fails", drawer_joint_jump_fails)
C.run("a negative tolerance and the wrong scene are refused", misuse)
C.exit()
