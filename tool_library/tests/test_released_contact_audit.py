"""released_contact_audit: a hand that lets go and lifts away passes; a retreat that drags the box, a hand that never
leaves it and fingers buried in it while releasing fail; two-hand labels are read per hand."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures" / "episode_audits"))
import synth as S
import numpy as np
from tool_library.verifiers.released_contact_audit import released_contact_audit

tmp = S.tmpdir()
scene, m = S.bench(tmp, [dict(name="box", size=[0.025, 0.025, 0.025], pos=[0.2, 0.3, 0.025], mass=0.2)])
wr = S.wrist_adr(m, "right")
a = S.free_adr(m, "box")
q = S.home(m); q[wr[2]] -= 0.06
q = S.touch_from_above(scene, m, q, "right", "box")                  # a fingertip resting on the box's top
T = 60
L = ["right: let go of box #1"] * 20 + ["right: retreat #1"] * 40


def episode(lift=0.08, drag=0.0, dz_box=0.0):
    Q = np.repeat(q[None], T, 0)
    Q[20:, wr[2]] = q[wr[2]] + S.smooth([0], [lift], 40)[:, 0]
    if drag:
        Q[20:40, wr[0]] = q[wr[0]] + np.linspace(0, drag, 20); Q[40:, wr[0]] = q[wr[0]] + drag
        Q[20:40, wr[2]] = q[wr[2]]
        Q[20:40, a] = q[a] + np.linspace(0, drag, 20); Q[40:, a] = q[a] + drag    # the box goes along with the finger
    Q[:20, a + 2] += dz_box
    return Q


C = S.Cases()


def lift_away_passes():
    r = released_contact_audit(scene, S.save(tmp / "ok.npz", episode(), L))
    assert r["ok"], r["issues"]
    rel = r["releases"][0]
    assert rel["object"] == "box" and rel["frames_touching"] > 0 and rel["last_touch_frame"] < 30, rel


def drag_fails():
    r = released_contact_audit(scene, S.save(tmp / "drag.npz", episode(lift=0.0, drag=0.025), L))
    assert not r["ok"] and r["worst"]["kind"] == "drag" and r["worst"]["value"] > 20, r["worst"]


def linger_and_sink_fail():
    r = released_contact_audit(scene, S.save(tmp / "stay.npz", episode(lift=0.0), L))
    assert not r["ok"] and any(e for e in [r["worst"]] if e["kind"] == "linger"), r["worst"]
    r = released_contact_audit(scene, S.save(tmp / "sink.npz", episode(dz_box=0.012), L))
    assert not r["ok"] and r["worst"]["kind"] == "sink" and r["worst"]["value"] > 10, r["worst"]


def two_hand_labels():
    L2 = ["left: hold the lid #1 | right: let go of box #1"] * 20 + ["left: hold the lid #1 | right: retreat #1"] * 40
    assert released_contact_audit(scene, S.save(tmp / "two.npz", episode(), L2))["ok"]
    r = released_contact_audit(scene, S.save(tmp / "two_drag.npz", episode(lift=0.0, drag=0.025), L2))
    assert not r["ok"] and r["worst"]["side"] == "right", r["worst"]


def misuse():
    S.expect_value_error(lambda: released_contact_audit(scene, S.save(tmp / "m.npz", episode(), L), grace_frames=-1), "grace_frames")


C.run("letting go and lifting 8 cm away passes", lift_away_passes)
C.run("a retreat that slides the box 25 mm along fails as a drag", drag_fails)
C.run("a hand that stays on the box fails (linger); fingers 12 mm inside it fail (sink)", linger_and_sink_fail)
C.run("the right hand's half of 'left: X | right: Y' labels is read", two_hand_labels)
C.run("negative grace frames are refused", misuse)
C.exit()
