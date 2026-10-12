"""free_fall_plausibility_audit: a box dropped in physics passes, and so does one held by a world weld; a box
hovering with nothing touching it and one sinking at constant speed fail."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures" / "episode_audits"))
import synth as S
import mujoco
import numpy as np
from tool_library.verifiers.free_fall_plausibility_audit import free_fall_plausibility_audit

tmp = S.tmpdir()
scene, m = S.bench(tmp, [dict(name="box", size=[0.03, 0.03, 0.03], pos=[0.0, 0.45, 0.20])])
q0 = S.home(m)
a = S.free_adr(m, "box")
drop = S.save(tmp / "drop.npz", S.rollout(scene, q0, 45), "right: let go of box #1")
C = S.Cases()


def physics_drop_passes():
    r = free_fall_plausibility_audit(scene, drop)
    assert r["ok"], r["issues"]
    assert r["objects"]["box"]["free_frames"] >= 3 and r["objects"]["box"]["worst_acc_error"] < 1.0, r["objects"]


def hover_fails():
    Q = np.repeat(q0[None], 40, 0)                                   # held 17 cm up by nothing
    r = free_fall_plausibility_audit(scene, S.save(tmp / "hover.npz", Q, "right: let go of box #1"))
    assert not r["ok"] and r["worst"]["kind"] == "hovers" and r["worst"]["n"] > 30, r["worst"]


def slow_sink_fails():
    Q = np.repeat(q0[None], 40, 0)
    Q[:, a + 2] = 0.20 - 0.2 / 30 * np.arange(40)                    # 0.2 m/s down, no acceleration
    r = free_fall_plausibility_audit(scene, S.save(tmp / "sink.npz", Q, "right: let go of box #1"))
    assert not r["ok"] and r["worst"]["kind"] == "falls too slowly", r["worst"]


def world_weld_supports():
    def weld(spec):
        e = spec.add_equality(); e.type = mujoco.mjtEq.mjEQ_WELD; e.name1 = "box"; e.name2 = "world"
        e.objtype = mujoco.mjtObj.mjOBJ_BODY
    sc2, m2 = S.bench(tmp / "weld", [dict(name="box", size=[0.03, 0.03, 0.03], pos=[0.0, 0.45, 0.20])], edit=weld)
    Q = np.repeat(S.home(m2)[None], 40, 0)
    r = free_fall_plausibility_audit(sc2, S.save(tmp / "weld.npz", Q, "right: let go of box #1"))
    assert r["ok"] and r["objects"]["box"]["free_frames"] == 0, r


def misuse():
    S.expect_value_error(lambda: free_fall_plausibility_audit(scene, drop, acc_tol=0), "acc_tol")


C.run("a box dropped from 17 cm in physics passes", physics_drop_passes)
C.run("a box hovering with nothing touching it fails", hover_fails)
C.run("a box sinking at a constant 0.2 m/s fails", slow_sink_fails)
C.run("a box held by a weld to the world is supported", world_weld_supports)
C.run("a zero tolerance is refused", misuse)
C.exit()
