"""redundant_motion_audit on the floating hand's orientation: a geodesic turn and a turn split over two phases pass;
turning 40 deg away and back inside one phase fails, also in a two-hand label."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures" / "episode_audits"))
import synth as S
import mujoco
import numpy as np
from tool_library.verifiers.redundant_motion_audit import redundant_motion_audit

tmp = S.tmpdir()
scene = str(S.SETUP)
m = mujoco.MjModel.from_xml_path(scene)
q0 = S.home(m)
w = S.wrist_adr(m, "right")
rz, rx = w[5], w[3]
T = 90
C = S.Cases()


def ep(profile, labels, axis=rz):
    Q = np.repeat(q0[None], T, 0)
    Q[:, axis] = q0[axis] + np.radians(profile)
    return Q, labels


def geodesic_turn_passes():
    Q, L = ep(np.r_[np.linspace(0, 90, 60), np.full(30, 90)], ["right: turn the cap #1"] * T)
    r = redundant_motion_audit(scene, S.save(tmp / "g.npz", Q, L))
    assert r["ok"], r["issues"]
    assert r["phases"][0]["net_deg"] > 85, r["phases"]


def away_and_back_in_one_phase_fails():
    prof = np.r_[np.linspace(0, 40, 30), np.linspace(40, 0, 30), np.zeros(30)]
    Q, L = ep(prof, ["right: carry the box #2"] * T, axis=rx)
    r = redundant_motion_audit(scene, S.save(tmp / "b.npz", Q, L))
    assert not r["ok"] and 38 < r["worst"]["excursion_deg"] < 42 and r["worst"]["side"] == "right", r["worst"]


def turn_split_over_phases_passes():
    prof = np.r_[np.linspace(0, 40, 30), np.linspace(40, 0, 30), np.zeros(30)]
    L = ["right: twist #3"] * 30 + ["right: turn back #3"] * 60                # out in one phase, back in the next
    Q, _ = ep(prof, L)
    r = redundant_motion_audit(scene, S.save(tmp / "s.npz", Q, L))
    assert r["ok"], r["issues"]


def two_hand_label():
    prof = np.r_[np.linspace(0, 40, 30), np.linspace(40, 0, 30), np.zeros(30)]
    Q, _ = ep(prof, None)
    L = ["left: hold the box #2 | right: press the lid #2"] * T
    r = redundant_motion_audit(scene, S.save(tmp / "t.npz", Q, L))
    assert not r["ok"] and r["worst"]["side"] == "right", r["worst"]
    assert redundant_motion_audit(scene, S.save(tmp / "t2.npz", Q, L), excursion_deg=50)["ok"]


def misuse():
    Q, L = ep(np.zeros(T), ["right: x"] * T)
    S.expect_value_error(lambda: redundant_motion_audit(scene, S.save(tmp / "m.npz", Q, L), excursion_deg=0), "excursion_deg")


C.run("a 90 deg geodesic turn passes", geodesic_turn_passes)
C.run("a 40 deg turn away and back inside one carry phase fails", away_and_back_in_one_phase_fails)
C.run("a twist and its turn back in separate phases pass", turn_split_over_phases_passes)
C.run("the right hand's half of a 'left: X | right: Y' label is judged (threshold moves it)", two_hand_label)
C.run("a zero excursion threshold is refused", misuse)
C.exit()
