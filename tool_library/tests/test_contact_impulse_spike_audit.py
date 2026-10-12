"""contact_impulse_spike_audit: a gentle physics drop passes; a box slammed onto the table, a hand slamming into a
box and a logged wrist-force spike fail."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures" / "episode_audits"))
import synth as S
import numpy as np
from tool_library.verifiers.contact_impulse_spike_audit import contact_impulse_spike_audit

tmp = S.tmpdir()
scene, m = S.bench(tmp, [dict(name="box", size=[0.025, 0.025, 0.025], pos=[0.17, 0.30, 0.027], mass=0.3)])
q0 = S.home(m)
a = S.free_adr(m, "box")
wr = S.wrist_adr(m, "right")
gentle = S.save(tmp / "gentle.npz", S.rollout(scene, q0, 30), "right: set down box #1")
rest = S.rollout(scene, q0, 20)[-1]
C = S.Cases()


def gentle_passes():
    r = contact_impulse_spike_audit(scene, gentle)
    assert r["ok"], r["issues"]
    assert r["objects"]["box"]["peak_force_ratio"] < 2, r["objects"]


def slam_fails():
    Q = np.repeat(rest[None], 40, 0)
    z = rest[a + 2]
    Q[:20, a + 2] = z + 1.5 / 30 * np.arange(20, 0, -1)               # coming down at 1.5 m/s, stopped in one frame
    r = contact_impulse_spike_audit(scene, S.save(tmp / "slam.npz", Q, "right: set down box #1"))
    assert not r["ok"] and r["worst"]["body"] == "box" and r["worst"]["value"] > 4, r["worst"]


def hand_hit_fails():
    q = rest.copy(); q[wr[2]] -= 0.06
    q = S.touch_from_above(scene, m, q, "right", "box")              # the box under the index fingertip, just touching
    Q = np.repeat(q[None], 40, 0)
    Q[:20, wr[2]] = q[wr[2]] + 0.6 / 30 * np.arange(20, 0, -1)       # the hand comes down at 0.6 m/s and stops dead on it
    r = contact_impulse_spike_audit(scene, S.save(tmp / "hit.npz", Q, "right: press box #1"))
    assert not r["ok"] and r["worst"]["body"] == "right hand" and r["worst"]["value"] > 8, (r["worst"], r["hands"])
    Q[:20, wr[2]] = q[wr[2]] + 0.05 * (1 - S.smooth([0], [1], 20)[:, 0])                # a slow, eased descent
    assert contact_impulse_spike_audit(scene, S.save(tmp / "soft.npz", Q, "right: press box #1"))["ok"]


def logged_spike_fails():
    w = np.full(30, 3.0); w[15] = 60.0
    r = contact_impulse_spike_audit(scene, S.save(tmp / "log.npz", np.repeat(rest[None], 30, 0), "right: press #1", wrist_force_n=w))
    assert not r["ok"] and r["worst"]["kind"] == "log" and r["worst"]["frame"] == 15, r["worst"]


def misuse():
    S.expect_value_error(lambda: contact_impulse_spike_audit(scene, gentle, max_force_ratio=0), "max_force_ratio")


C.run("a box settling onto the table in physics passes", gentle_passes)
C.run("a box stopped from 1.5 m/s in one frame on the table fails", slam_fails)
C.run("a hand stopping dead on a box from 0.6 m/s fails; an eased descent passes", hand_hit_fails)
C.run("a 60 N spike in a logged 3 N wrist force fails", logged_spike_fails)
C.run("a zero force ratio is refused", misuse)
C.exit()
