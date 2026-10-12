"""swept_collision_audit: a ball moving beside a 2 mm wall passes; a ball crossing the wall in one frame fails (no
penetration on either frame); a ball pushed slowly through the wall also fails on its recorded states."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures" / "episode_audits"))
import synth as S
import mujoco
import numpy as np
from tool_library.verifiers.swept_collision_audit import swept_collision_audit

tmp = S.tmpdir()


def wall(spec):
    b = spec.worldbody.add_body(); b.name = "wall"; b.pos = [0.0, 0.6, 0.1]
    g = b.add_geom(); g.name = "wall_geom"; g.type = mujoco.mjtGeom.mjGEOM_BOX; g.size = [0.001, 0.1, 0.1]


scene, m = S.bench(tmp, [dict(name="ball", type="sphere", size=[0.01], pos=[-0.03, 0.6, 0.1], mass=0.05)], edit=wall)
q0 = S.home(m)
a = S.free_adr(m, "ball")
T = 40


def path(xs, ys=None):
    Q = np.repeat(q0[None], T, 0)
    Q[:, a] = xs
    if ys is not None:
        Q[:, a + 1] = ys
    return Q


C = S.Cases()


def beside_the_wall_passes():
    r = swept_collision_audit(scene, S.save(tmp / "ok.npz", path(np.full(T, -0.03), 0.5 + np.linspace(0, 0.2, T)), "right: x #1"))
    assert r["ok"] and r["frame_pairs_subdivided"] > 0, r


def one_frame_crossing_fails():
    xs = np.r_[np.full(20, -0.016), np.full(20, 0.016)]               # 32 mm across a 2 mm wall between frames 19 and 20
    r = swept_collision_audit(scene, S.save(tmp / "cross.npz", path(xs), "right: x #1"))
    assert not r["ok"], r
    w = r["worst"]
    assert w["frames"] == [19, 20] and "wall" in w["bodies"] and w["frame_depth_mm"] == 0 and w["depth_mm"] > 8, w


def slow_push_through_fails_on_recorded_states():
    xs = np.linspace(-0.03, 0.03, T)                                   # 1.5 mm per frame: inside the wall on the frames too
    r = swept_collision_audit(scene, S.save(tmp / "slow.npz", path(xs), "right: x #1"))
    assert not r["ok"], r
    w = r["worst"]
    assert set(w["bodies"]) == {"ball", "wall"} and w["frame_depth_mm"] > 8, w
    assert w["depth_mm"] >= w["frame_depth_mm"] and w["fraction"] in (0.0, 1.0), w


def tolerance():
    xs = np.r_[np.full(20, -0.016), np.full(20, 0.016)]
    assert swept_collision_audit(scene, S.save(tmp / "cross2.npz", path(xs), "right: x #1"), tol_m=0.02)["ok"]


def misuse():
    S.expect_value_error(lambda: swept_collision_audit(scene, S.save(tmp / "m.npz", path(np.full(T, -0.03)), "x: y"), sub_m=0), "sub_m")


C.run("a ball moving beside a thin wall passes", beside_the_wall_passes)
C.run("a ball crossing a 2 mm wall between two frames fails", one_frame_crossing_fails)
C.run("a ball pushed slowly through the wall fails on recorded-state penetration", slow_push_through_fails_on_recorded_states)
C.run("a 20 mm tolerance lets the crossing pass", tolerance)
C.run("a zero sub-step size is refused", misuse)
C.exit()
