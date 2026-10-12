"""start_state_audit: an episode starting from a settled keyframe passes; a keyframe with the box hovering, a frame 0
that differs from the keyframe and a scene without a keyframe fail."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures" / "episode_audits"))
import synth as S
import numpy as np
from tool_library.verifiers.start_state_audit import start_state_audit

tmp = S.tmpdir()
box = dict(name="box", size=[0.025, 0.025, 0.025], pos=[0.0, 0.45, 0.025], mass=0.2)
scene, m = S.bench(tmp / "ok", [box])
a = S.free_adr(m, "box")
T = 30
C = S.Cases()


def settled_start_passes():
    r = start_state_audit(scene, S.save(tmp / "ok.npz", np.repeat(S.home(m)[None], T, 0), "right: reach #1"))
    assert r["ok"], r["issues"]
    assert r["settle"]["frame 0"]["box"]["peak_moved_mm"] < 1, r["settle"]


def hovering_keyframe_fails():
    sc, mh = S.bench(tmp / "hover", [dict(box, pos=[0.0, 0.45, 0.045])])            # 2 cm above the table
    r = start_state_audit(sc, S.save(tmp / "hover.npz", np.repeat(S.home(mh)[None], T, 0), "right: reach #1"))
    assert not r["ok"] and r["worst"]["body"] == "box" and r["worst"]["peak_moved_m"] > 0.015, r["worst"]


def frame0_differs_fails():
    Q = np.repeat(S.home(m)[None], T, 0)
    Q[:, a] += 0.02
    r = start_state_audit(scene, S.save(tmp / "moved.npz", Q, "right: reach #1"))
    assert not r["ok"] and r["keyframe"]["objects"]["box"]["diff_mm"] > 19, r["keyframe"]
    Qr = np.repeat(S.home(m)[None], T, 0)
    Qr[:, S.jadr(m, "robot_right_arm_dof_joint2")] += 0.05
    r = start_state_audit(scene, S.save(tmp / "robot.npz", Qr, "right: reach #1"))
    assert not r["ok"] and r["keyframe"]["robot_worst_joint"] == "robot_right_arm_dof_joint2", r["keyframe"]


def no_keyframe_fails():
    sc, mk = S.bench(tmp / "nokey", [dict(box, pos=[0.0, 0.45, 0.06], quat=[0.966, 0.259, 0, 0])], keyframe=False)
    Q = np.repeat(mk.qpos0[None], T, 0)
    Q[:, S.free_adr(mk, "box") + 2] = 0.025; Q[:, S.free_adr(mk, "box") + 3:S.free_adr(mk, "box") + 7] = [1, 0, 0, 0]
    r = start_state_audit(sc, S.save(tmp / "nokey.npz", Q, "right: reach #1"))
    assert not r["ok"] and "no keyframe" in r["issues"][0], r["issues"]
    assert any("qpos0" in s and "box" in s for s in r["issues"][1:]), r["issues"]   # the tilted box in qpos0 settles


def misuse():
    t = S.save(tmp / "m.npz", np.repeat(S.home(m)[None], T, 0), "x: y")
    S.expect_value_error(lambda: start_state_audit(scene, t, key=3), "key")
    S.expect_value_error(lambda: start_state_audit(scene, t, key="nope"), "key")


C.run("an episode starting from a settled keyframe passes", settled_start_passes)
C.run("a keyframe with the box 2 cm above the table fails (it drops)", hovering_keyframe_fails)
C.run("a frame 0 with the box 2 cm off, or the arm 0.05 rad off, the keyframe fails", frame0_differs_fails)
C.run("a scene without a keyframe fails, and the tilted box of its qpos0 is reported", no_keyframe_fails)
C.run("a keyframe index or name the scene lacks is refused", misuse)
C.exit()
