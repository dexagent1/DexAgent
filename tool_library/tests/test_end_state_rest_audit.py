"""end_state_rest_audit: a box at rest on the table passes; a box still sliding at the end, a box left balanced on its
edge and a box propped by the hand fail."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures" / "episode_audits"))
import synth as S
import numpy as np
from tool_library.verifiers.end_state_rest_audit import end_state_rest_audit

tmp = S.tmpdir()
scene, m = S.bench(tmp, [dict(name="box", size=[0.025, 0.025, 0.025], pos=[0.0, 0.45, 0.025], mass=0.2)])
q0 = S.home(m)
a = S.free_adr(m, "box")
rest = S.rollout(scene, q0, 20)[-1]
T = 40
C = S.Cases()


def at_rest_passes():
    r = end_state_rest_audit(scene, S.save(tmp / "ok.npz", np.repeat(rest[None], T, 0), "right: retreat #1"))
    assert r["ok"], r["issues"]
    assert r["objects"]["box"]["settle_peak_mm"] < 1 and not r["objects"]["box"]["touched_by_robot"], r["objects"]


def still_moving_fails():
    Q = np.repeat(rest[None], T, 0)
    Q[:, a] = rest[a] + 0.05 / 30 * np.arange(T)                     # sliding at 5 cm/s when the episode ends
    r = end_state_rest_audit(scene, S.save(tmp / "slide.npz", Q, "right: retreat #1"))
    assert not r["ok"] and r["worst"]["kind"] == "moving", r["worst"]


def balanced_on_edge_fails():
    Q = np.repeat(rest[None], T, 0)
    c, s = np.cos(np.radians(20) / 2), np.sin(np.radians(20) / 2)
    Q[:, a + 3:a + 7] = [c, s, 0, 0]                                 # tilted 20 deg about x, resting on one edge
    Q[:, a + 2] = 0.025 * (np.cos(np.radians(20)) + np.sin(np.radians(20)))
    r = end_state_rest_audit(scene, S.save(tmp / "edge.npz", Q, "right: retreat #1"))
    assert not r["ok"] and r["objects"]["box"]["settle_peak_deg"] > 10, r["objects"]


def propped_by_hand_fails():
    wr = S.wrist_adr(m, "right")
    q = rest.copy(); q[wr[2]] -= 0.05
    q = S.touch_from_above(scene, m, q, "right", "box")              # the box lifted until the fingertip rests on it ...
    a2 = S.free_adr(m, "box")
    q[a2 + 2] += 0.0                                                  # ... floating under the finger: only the hand holds it
    Q = np.repeat(q[None], T, 0)
    r = end_state_rest_audit(scene, S.save(tmp / "prop.npz", Q, "right: hold box #1"))
    assert r["objects"]["box"]["touched_by_robot"], r["objects"]
    assert not r["ok"], r["objects"]
    assert end_state_rest_audit(scene, S.save(tmp / "prop2.npz", Q, "right: hold box #1"), allow_held=["box"])["objects"]["box"]["touched_by_robot"]


def misuse():
    S.expect_value_error(lambda: end_state_rest_audit(scene, S.save(tmp / "m.npz", np.repeat(rest[None], T, 0), "x: y"), settle_s=0), "settle_s")


C.run("a box at rest on the table passes", at_rest_passes)
C.run("a box still sliding at 5 cm/s at the end fails", still_moving_fails)
C.run("a box left balanced 20 deg on its edge fails when simulated on", balanced_on_edge_fails)
C.run("a box held up only by a fingertip fails as unsupported", propped_by_hand_fails)
C.run("a zero settle time is refused", misuse)
C.exit()
