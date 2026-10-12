"""hand_object_relative_jitter_audit: a box carried rigidly, and one slipping slowly, pass; a box vibrating 4 mm or 6 deg
relative to the hand fails."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures" / "episode_audits"))
import synth as S
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from tool_library.verifiers.hand_object_relative_jitter_audit import hand_object_relative_jitter_audit

tmp = S.tmpdir()
scene, m = S.bench(tmp, [dict(name="box", size=[0.02, 0.02, 0.02], pos=[0.2, 0.4, 0.02])])
wr = S.wrist_adr(m, "right")
a = S.free_adr(m, "box")
q0 = S.home(m)
T = 90
Q = np.repeat(q0[None], T, 0)
Q[:, wr[1]] = q0[wr[1]] + S.smooth([0], [0.2], T)[:, 0]
L = "right: carry box #1"


def carried(rel_fn, rot_fn=None):
    Qc = Q.copy()
    for t in range(T):
        Qc[t] = S.carry_with_hand(m, Q[t:t + 1], "right", "box", rel_pos=rel_fn(t))[0]
        if rot_fn is not None:
            r = Rot.from_quat(np.r_[Qc[t, a + 4:a + 7], Qc[t, a + 3]]) * Rot.from_euler("x", rot_fn(t), degrees=True)
            x = r.as_quat(); Qc[t, a + 3:a + 7] = [x[3], x[0], x[1], x[2]]
    return Qc


base = np.array([0.0, 0.12, -0.06])
C = S.Cases()


def rigid_and_slow_slip_pass():
    r = hand_object_relative_jitter_audit(scene, S.save(tmp / "ok.npz", carried(lambda t: base), L))
    assert r["ok"] and r["windows"][0]["peak_mm"] < 0.1, r
    r = hand_object_relative_jitter_audit(scene, S.save(tmp / "slip.npz", carried(lambda t: base + [0, 0.02 * t / T, 0]), L))
    assert r["ok"], r["issues"]                                      # slow drift is held_object_slip_audit's


def positional_jitter_fails():
    r = hand_object_relative_jitter_audit(scene, S.save(tmp / "jit.npz", carried(lambda t: base + [0.004 * (-1) ** t, 0, 0]), L))
    assert not r["ok"] and r["worst"]["peak_mm"] > 3 and r["worst"]["object"] == "box", r["worst"]


def rotational_jitter_fails():
    r = hand_object_relative_jitter_audit(scene, S.save(tmp / "rot.npz", carried(lambda t: base, lambda t: 6.0 * (-1) ** t), L))
    assert not r["ok"] and r["worst"]["peak_deg"] > 4, r["worst"]
    assert hand_object_relative_jitter_audit(scene, S.save(tmp / "rot2.npz", carried(lambda t: base, lambda t: 6.0 * (-1) ** t), L),
                                             peak_deg=20, rms_deg=20)["ok"]


def misuse():
    S.expect_value_error(lambda: hand_object_relative_jitter_audit(scene, S.save(tmp / "m.npz", Q, L), smooth_frames=0), "smooth_frames")


C.run("a rigid carry and a slow 20 mm slip pass", rigid_and_slow_slip_pass)
C.run("a box vibrating 4 mm frame to frame in the hand fails", positional_jitter_fails)
C.run("a box vibrating 6 deg fails (passes with 20 deg tolerances)", rotational_jitter_fails)
C.run("a zero smoothing window is refused", misuse)
C.exit()
