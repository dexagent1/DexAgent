"""held_object_slip_audit: a box carried rigidly passes; one sliding 30 mm or turning 25 deg in the hand during the carry
fails; a lift phase seats freely by default; two-hand labels are read per hand. A strap of box links on ball joints
(a cable) is judged at the link and point the hand touches, not at its root: its end link held under the fingertip
while the rest of the strap swings 60 deg about it passes (the root junction read a false 6 cm before), the held link
sliding 30 mm along the hand fails."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures" / "episode_audits"))
import synth as S
import numpy as np
from tool_library.verifiers.held_object_slip_audit import held_object_slip_audit

tmp = S.tmpdir()
scene, m = S.bench(tmp, [dict(name="box", size=[0.02, 0.02, 0.02], pos=[0.2, 0.4, 0.02])])
wr = S.wrist_adr(m, "right")
q0 = S.home(m)
T = 90
Q = np.repeat(q0[None], T, 0)
Q[:, wr[1]] = q0[wr[1]] + S.smooth([0], [0.2], T)[:, 0]
Q[:, wr[5]] = q0[wr[5]] + S.smooth([0], [0.5], T)[:, 0]
L = ["right: lift box #1"] * 30 + ["right: carry box (up) #2"] * 30 + ["right: carry box #2"] * 30


def carried(drift=(0, 0, 0), turn=0.0, start=30):
    rel = np.zeros((T, 3)); rel[:] = (0.0, 0.12, -0.06)
    rel[start:] += np.outer(S.smooth([0], [1], T - start)[:, 0], drift)
    Qc = Q.copy()
    for t in range(T):
        Qc[t] = S.carry_with_hand(m, Q[t:t + 1], "right", "box", rel_pos=rel[t])[0]
    if turn:
        a = S.free_adr(m, "box")
        from scipy.spatial.transform import Rotation as Rot
        for t in range(start, T):
            r0 = Rot.from_quat(np.r_[Qc[t, a + 4:a + 7], Qc[t, a + 3]])
            r1 = r0 * Rot.from_euler("z", turn * (t - start) / (T - start - 1), degrees=True)
            x = r1.as_quat(); Qc[t, a + 3:a + 7] = [x[3], x[0], x[1], x[2]]
    return Qc


C = S.Cases()


def rigid_passes():
    r = held_object_slip_audit(scene, S.save(tmp / "ok.npz", carried(), L))
    assert r["ok"], r["issues"]
    assert r["n_windows"] == 1 and r["windows"][0]["frames"] == [30, 89] and r["windows"][0]["drift_mm"] < 0.5, r["windows"]


def slide_and_turn_fail():
    r = held_object_slip_audit(scene, S.save(tmp / "slide.npz", carried(drift=(0.0, 0.03, 0.0)), L))
    assert not r["ok"] and 29 < r["worst"]["drift_mm"] < 31, r["worst"]
    r = held_object_slip_audit(scene, S.save(tmp / "turn.npz", carried(turn=25.0), L))
    assert not r["ok"] and r["worst"]["drift_deg"] > 24, r["worst"]


def lift_seating_allowed():
    rel = np.array([0.0, 0.12, -0.06])
    seat = np.minimum(np.arange(T), 29) / 29.0 * 0.03                # 30 mm of seating during the lift, then rigid
    Ql = np.vstack([S.carry_with_hand(m, Q[t:t + 1], "right", "box", rel_pos=rel + [0, seat[t], 0]) for t in range(T)])
    r = held_object_slip_audit(scene, S.save(tmp / "lift.npz", Ql, L))
    assert r["ok"], r["issues"]
    r = held_object_slip_audit(scene, S.save(tmp / "lift2.npz", Ql, L), words=["lift"])
    assert not r["ok"] and r["worst"]["frames"] == [0, 29], r["windows"]


def two_hand_label():
    L2 = [f"left: hold the bowl #2 | {x}" for x in L]
    r = held_object_slip_audit(scene, S.save(tmp / "two.npz", carried(drift=(0.0, 0.03, 0.0)), L2))
    assert not r["ok"] and r["worst"]["side"] == "right", r["worst"]


def misuse():
    S.expect_value_error(lambda: held_object_slip_audit(scene, S.save(tmp / "m.npz", carried(), L), words=[]), "words")


# a strap: a free junction box with six 12 mm box links on ball joints, carried by its end link under the index tip
NL = 6


def add_strap(spec):
    b = spec.worldbody.add_body(); b.name = "strap"; b.pos = [0.2, 0.4, 0.15]
    j = b.add_freejoint(); j.name = "strap_free"
    g = b.add_geom(); g.name = "strap_geom"; g.type = mujoco.mjtGeom.mjGEOM_BOX; g.size = [0.006, 0.01, 0.003]; g.mass = 0.002
    parent = b
    for i in range(NL):
        c = parent.add_body(); c.name = f"strap_l{i}"; c.pos = [0.006, 0, 0] if i == 0 else [0.012, 0, 0]
        jj = c.add_joint(); jj.name = f"strap_j{i}"; jj.type = mujoco.mjtJoint.mjJNT_BALL
        gg = c.add_geom(); gg.name = f"strap_g{i}"; gg.type = mujoco.mjtGeom.mjGEOM_BOX; gg.size = [0.006, 0.004, 0.0006]
        gg.pos = [0.006, 0, 0]; gg.mass = 0.0005
        parent = c


import mujoco  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402
from tool_library.common import episode_audit as EA  # noqa: E402

scene_s, ms = S.bench(tmp, [], edit=add_strap, name="strap.xml")
wrs = S.wrist_adr(ms, "right")
Qs = np.repeat(S.home(ms)[None], T, 0)
Qs[:, wrs[1]] += S.smooth([0], [0.2], T)[:, 0]
Qs[:, wrs[5]] += S.smooth([0], [0.5], T)[:, 0]
LS = ["right: lift strap #1"] * 30 + ["right: carry strap (across) #2"] * 60
ds = mujoco.MjData(ms)
fa = int(ms.jnt_qposadr[mujoco.mj_name2id(ms, mujoco.mjtObj.mjOBJ_JOINT, "strap_free")])
ja = int(ms.jnt_qposadr[mujoco.mj_name2id(ms, mujoco.mjtObj.mjOBJ_JOINT, f"strap_j{NL - 1}")])
end = mujoco.mj_name2id(ms, mujoco.mjtObj.mjOBJ_BODY, f"strap_l{NL - 1}")
hand_root = mujoco.mj_name2id(ms, mujoco.mjtObj.mjOBJ_BODY, "robot_right_right_base_yaw")


def pose_of(q, b):
    ds.qpos[:] = q; mujoco.mj_kinematics(ms, ds)
    return ds.xpos[b].copy(), ds.xmat[b].reshape(3, 3).copy()


def strap_held(swing_deg=0.0, slide=(0.0, 0.0, 0.0)):
    """The strap's end link riding with the wrist (sliding by `slide` in the wrist frame over the carry), its last joint
    turning by swing_deg about the vertical, so the junction and the other links swing about the held link."""
    # the end link right under the index tip at frame 0 (its centre 8 mm under the tip site: the pad rests on its top)
    q = Qs[0].copy()
    tip = S.site_pos(ms, q, "robot_right_right_index_tip")
    ph, Rh = pose_of(q, hand_root)
    Q = Qs.copy()
    u = np.minimum(np.maximum(np.arange(T) - 30, 0) / 59.0, 1.0)
    s_ = u * u * (3 - 2 * u)
    for t in range(T):
        q = Q[t].copy()
        q[ja:ja + 4] = np.r_[Rot.from_euler("z", swing_deg * s_[t], degrees=True).as_quat()[[3, 0, 1, 2]]]
        q[fa:fa + 7] = [0, 0, 0, 1, 0, 0, 0]
        pk, Rk = pose_of(q, end)                         # the end link in the junction's frame
        pht, Rht = pose_of(q, hand_root)
        rel = Rh.T @ (tip + np.array([0, 0, -0.008]) - ph) + np.asarray(slide) * s_[t]     # in the wrist frame
        target_c = pht + Rht @ rel                       # the link centre rides with the wrist
        target_R = Rht @ Rh.T                            # and turns with it (flat at frame 0)
        Rj = target_R @ Rk.T                             # junction orientation that puts the end link there
        pj = target_c - Rj @ (pk + Rk @ np.array([0.006, 0, 0]))
        q[fa:fa + 3] = pj; q[fa + 3:fa + 7] = Rot.from_matrix(Rj).as_quat()[[3, 0, 1, 2]]
        Q[t] = q
    return Q


def touching_end(Q):
    ep = EA.Episode(scene_s, {"qpos": Q[:1], "fps": 30.0, "labels": [""]})
    return EA.ContactScan(ep, 0.003).touching(0, ep.hands["right"]["bodies"], {end}) < np.inf


def strap_swing_passes():
    Q = strap_held(swing_deg=60.0)
    assert touching_end(Q), "the end link must touch the hand at the carry's start"
    r = held_object_slip_audit(scene_s, S.save(tmp / "strap.npz", Q, LS))
    w = [x for x in r["windows"] if x["frames"][0] == 30]
    assert r["ok"] and w and w[0]["body"] == f"strap_l{NL - 1}" and w[0]["drift_mm"] < 1.0 and not w[0]["turn_judged"], r
    pj = [pose_of(q, mujoco.mj_name2id(ms, mujoco.mjtObj.mjOBJ_BODY, "strap"))[0] for q in Q]
    ph = [pose_of(q, hand_root) for q in Q]
    rel = [R.T @ (p - h) for p, (h, R) in zip(pj, ph)]
    assert np.linalg.norm(rel[-1] - rel[30]) > 0.05, "the junction should swing several cm about the held link"


def strap_slide_fails():
    Q = strap_held(slide=(0.0, 0.03, 0.0))
    r = held_object_slip_audit(scene_s, S.save(tmp / "strap_slide.npz", Q, LS))
    assert not r["ok"] and r["worst"]["body"] == f"strap_l{NL - 1}" and 29 < r["worst"]["drift_mm"] < 31, r["worst"]
    assert "held at strap_l5" in r["issues"][0], r["issues"]


C.run("a box carried rigidly passes", rigid_passes)
C.run("30 mm of sliding and 25 deg of turning in the hand fail", slide_and_turn_fail)
C.run("seating during the lift is allowed by default, judged when lift is a carry word", lift_seating_allowed)
C.run("the right hand's half of a two-hand label is judged", two_hand_label)
C.run("an empty word list is refused", misuse)
C.run("a strap held by its end link while the rest swings 60 deg about it passes (judged at the held link)", strap_swing_passes)
C.run("the held link sliding 30 mm along the hand fails, naming the link", strap_slide_fails)
C.exit()
