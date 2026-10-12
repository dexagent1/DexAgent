"""object_spin_rate_audit: a box turned 90 deg in a second passes; one spinning at 1500 deg/s for half a second, one
flying at 2 m/s and a lid swinging at 1000 deg/s fail; a three-frame spin is not sustained."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures" / "episode_audits"))
import synth as S
import mujoco
import numpy as np
from tool_library.verifiers.object_spin_rate_audit import object_spin_rate_audit

tmp = S.tmpdir()


def add_lid(spec):
    b = spec.worldbody.add_body(); b.name = "lid"; b.pos = [0.25, 0.7, 0.04]
    j = b.add_joint(); j.name = "lid_hinge"; j.type = mujoco.mjtJoint.mjJNT_HINGE; j.axis = [1, 0, 0]
    g = b.add_geom(); g.type = mujoco.mjtGeom.mjGEOM_BOX; g.size = [0.05, 0.05, 0.003]; g.mass = 0.05


scene, m = S.bench(tmp, [dict(name="box", size=[0.03, 0.03, 0.03], pos=[0.0, 0.45, 0.03])], edit=add_lid)
q0 = S.home(m)
a = S.free_adr(m, "box")
h = S.jadr(m, "lid_hinge")
T = 60


def yaw(deg):
    return np.c_[np.cos(np.radians(deg) / 2), np.zeros((len(deg), 2)), np.sin(np.radians(deg) / 2)]


def with_yaw(deg):
    Q = np.repeat(q0[None], T, 0)
    Q[:, a + 3:a + 7] = yaw(deg)
    return Q


C = S.Cases()


def slow_turn_passes():
    r = object_spin_rate_audit(scene, S.save(tmp / "slow.npz", with_yaw(np.linspace(0, 90, T)), "right: turn box #1"))
    assert r["ok"] and 40 < r["objects"]["box"]["sustained_deg_s"] < 50, r


def spin_and_flight_fail():
    deg = np.r_[np.zeros(20), np.arange(1, 16) * 50.0, np.full(25, 750.0)]     # 50 deg per frame = 1500 deg/s for 15 frames
    r = object_spin_rate_audit(scene, S.save(tmp / "spin.npz", with_yaw(deg), "right: let go of box #1"))
    assert not r["ok"] and r["worst"]["kind"] == "spin" and r["worst"]["sustained"] > 1400, r["worst"]
    Q = np.repeat(q0[None], T, 0)
    Q[20:, a] = q0[a] + 2.0 / 30 * np.arange(1, T - 19)                     # 2 m/s
    r = object_spin_rate_audit(scene, S.save(tmp / "fly.npz", Q, "right: let go of box #1"))
    assert not r["ok"] and r["worst"]["kind"] == "flight", r["worst"]


def short_spin_not_sustained():
    deg = np.r_[np.zeros(20), np.arange(1, 4) * 50.0, np.full(37, 150.0)]       # only three fast frames
    r = object_spin_rate_audit(scene, S.save(tmp / "short.npz", with_yaw(deg), "right: let go of box #1"))
    assert r["ok"], r["issues"]
    assert not object_spin_rate_audit(scene, S.save(tmp / "short2.npz", with_yaw(deg), "x: y"), min_frames=2)["ok"]


def lid_spin_fails():
    Q = np.repeat(q0[None], T, 0)
    Q[:, h] = np.radians(np.r_[np.zeros(20), np.arange(1, 41) * 35.0])           # 1050 deg/s
    r = object_spin_rate_audit(scene, S.save(tmp / "lid.npz", Q, "left: flip lid #1"))
    assert not r["ok"] and r["worst"]["body"] == "lid_hinge", r["worst"]


def misuse():
    S.expect_value_error(lambda: object_spin_rate_audit(scene, S.save(tmp / "m.npz", with_yaw(np.zeros(T)), "x: y"), max_deg_s=0), "max_deg_s")


C.run("a 90 deg turn in two seconds passes", slow_turn_passes)
C.run("1500 deg/s for half a second and 2 m/s flight fail", spin_and_flight_fail)
C.run("three fast frames are not sustained (but are with min_frames=2)", short_spin_not_sustained)
C.run("a hinged lid swinging at 1050 deg/s fails", lid_spin_fails)
C.run("a zero speed limit is refused", misuse)
C.exit()
