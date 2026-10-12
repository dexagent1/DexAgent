"""contactless_motion_audit: a box pushed by the hand in physics passes; a box sliding on its own long after contact,
a box carried a centimetre clear of the hand, and a settling motion beyond its allowance fail."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures" / "episode_audits"))
import synth as S
import numpy as np
from tool_library.verifiers.contactless_motion_audit import contactless_motion_audit

tmp = S.tmpdir()
scene, m = S.bench(tmp, [dict(name="box", size=[0.025, 0.025, 0.025], pos=[0.17, 0.30, 0.025], mass=0.3)])
q0 = S.home(m)
a = S.free_adr(m, "box")
wr = S.wrist_adr(m, "right")
q0[wr[2]] -= 0.08                                                     # lower the hand so its fingers reach the box
start = S.home(m); start[wr[2]] -= 0.08
x0 = start[wr[0]]


def drive(t, q):
    q = q.copy()
    q[wr[2]] = start[wr[2]]
    q[wr[0]] = x0 - min(t, 90) * 0.0015 + max(0, t - 100) * 0.003   # sweep 13.5 cm toward -x at 4.5 cm/s, pause, back off
    return q


Qp = S.rollout(scene, start, 130, drive=drive)
labels = ["right: push box #1"] * 100 + ["right: back off #1"] * 30
push = S.save(tmp / "push.npz", Qp, labels)
moved = float(np.linalg.norm(Qp[-1, a:a + 3] - Qp[0, a:a + 3]))
C = S.Cases()


def physics_push_passes():
    assert moved > 0.02, f"the hand did not push the box ({moved:.3f} m): fixture"
    r = contactless_motion_audit(scene, push)
    assert r["ok"], r["issues"]
    assert r["objects"]["box"]["robot_connected_frames"] > 5, r["objects"]


def slides_on_its_own_fails():
    Q = np.vstack([Qp, np.repeat(Qp[-1:], 90, 0)])
    Q[190:, a] = Qp[-1, a] + np.linspace(0, 0.05, 30)               # 5 cm, 2 s after the hand left
    r = contactless_motion_audit(scene, S.save(tmp / "slide.npz", Q, labels + ["right: return home #2"] * 90))
    assert not r["ok"] and not r["worst"]["settling"] and r["worst"]["travel_mm"] > 40, r["worst"]


def air_carry_fails():
    Q = np.repeat(start[None], 60, 0)
    Q[:, wr[1]] = start[wr[1]] + np.linspace(0, 0.15, 60)
    Q = S.carry_with_hand(m, Q, "right", "box", rel_pos=(0.0, 0.10, -0.12))   # rides along well clear of the fingers
    r = contactless_motion_audit(scene, S.save(tmp / "air.npz", Q, "right: carry box #1"))
    assert not r["ok"] and r["worst"]["since_robot_frames"] is None, r["worst"]


def settling_allowance():
    from tool_library.common import episode_audit as EA
    ep = EA.Episode(scene, push)
    tl = EA.object_contact_timeline(ep, margin=0.003)
    last = max(t for t in range(ep.T) if tl[t]["box"]["robot"])        # the hand's last touch
    Q = Qp.copy()
    k = min(last + 2, len(Q) - 20)
    Q[k:k + 14, a + 1] = Qp[k, a + 1] + np.linspace(0, 0.015, 14)     # 1.5 cm more right after it: settling
    Q[k + 14:, a + 1] = Qp[k, a + 1] + 0.015
    Q[k:, a] = Qp[k, a]
    r0 = contactless_motion_audit(scene, S.save(tmp / "set.npz", Q, labels))
    assert r0["ok"], (r0["issues"], last)
    r = contactless_motion_audit(scene, S.save(tmp / "set2.npz", Q, labels), settle_m=0.012)
    assert not r["ok"] and r["worst"]["settling"], r["worst"]


def misuse():
    S.expect_value_error(lambda: contactless_motion_audit(scene, push, settle_s=0), "settle_s")


C.run("a box pushed by the hand in physics passes", physics_push_passes)
C.run("a box sliding 5 cm two seconds after the hand left fails", slides_on_its_own_fails)
C.run("a box carried 12 cm clear of the hand fails (never touched)", air_carry_fails)
C.run("a box sliding on just after the last touch is settling; with a 12 mm allowance it fails", settling_allowance)
C.run("a zero settle time is refused", misuse)
C.exit()
