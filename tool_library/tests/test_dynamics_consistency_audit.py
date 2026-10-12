"""dynamics_consistency_audit: a physics drop, a physics push and a tag hanging on a magnet pass; a box shaken sideways
on the table faster than friction allows, a box riding on a weld to the hand, a tag held where no magnet pulls and a
tag too heavy for its magnet fail."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures" / "episode_audits"))
import synth as S
import mujoco
import numpy as np
from tool_library.verifiers.dynamics_consistency_audit import dynamics_consistency_audit

tmp = S.tmpdir()
scene, m = S.bench(tmp, [dict(name="box", size=[0.025, 0.025, 0.025], pos=[0.17, 0.30, 0.10], mass=0.3)])
q0 = S.home(m)
a = S.free_adr(m, "box")
wr = S.wrist_adr(m, "right")
drop = S.save(tmp / "drop.npz", S.rollout(scene, q0, 40), "right: let go of box #1")
rest = S.rollout(scene, q0, 30)[-1]
start = rest.copy(); start[wr[2]] -= 0.08
x0 = start[wr[0]]
Qp = S.rollout(scene, start, 80, drive=lambda t, q: np.where(np.arange(len(q)) == wr[0], x0 - min(t, 60) * 0.0015, start))
push = S.save(tmp / "push.npz", Qp, "right: push box #1")
C = S.Cases()


def physics_passes():
    for t in (drop, push):
        r = dynamics_consistency_audit(scene, t)
        assert r["ok"], r["issues"]
        assert r["objects"]["box"]["frames_judged"] > 10 and r["objects"]["box"]["worst_unexplained_m_s2"] < 4, r["objects"]


def shaken_on_the_table_fails():
    Q = np.repeat(rest[None], 60, 0)
    Q[10:52, a] = rest[a] + 0.03 * np.sin(2 * np.pi * 5 * np.arange(42) / 30)   # up to 30 m/s^2 sideways, only the table touches it
    r = dynamics_consistency_audit(scene, S.save(tmp / "shake.npz", Q, "right: hold #1"))
    assert not r["ok"] and r["worst"]["body"] == "box" and r["worst"]["contacts"] == ["static"], r["worst"]


def weld_attach_fails():
    def weld(spec):
        e = spec.add_equality(); e.type = mujoco.mjtEq.mjEQ_WELD; e.objtype = mujoco.mjtObj.mjOBJ_BODY
        e.name1 = "robot_right_right_base_yaw"; e.name2 = "box"
    sc2, m2 = S.bench(tmp / "weld", [dict(name="box", size=[0.025, 0.025, 0.025], pos=[0.17, 0.30, 0.10], mass=0.3)], edit=weld)
    q = S.home(m2)
    Q = np.repeat(q[None], 40, 0)
    Q[:, S.wrist_adr(m2, "right")[2]] += np.linspace(0, 0.1, 40)
    d0 = mujoco.MjData(m2); mujoco.mj_resetData(m2, d0); mujoco.mj_kinematics(m2, d0)
    hb = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_BODY, "robot_right_right_base_yaw")
    bb = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_BODY, "box")
    rel = d0.xmat[hb].reshape(3, 3).T @ (d0.xpos[bb] - d0.xpos[hb])               # the weld's relative pose (qpos0)
    Q = S.carry_with_hand(m2, Q, "right", "box", rel_pos=rel)
    r = dynamics_consistency_audit(sc2, S.save(tmp / "weld.npz", Q, "right: carry box #1"))
    assert not r["ok"] and r["attaches"] and r["attaches"][0]["n"] == 40, r


def magnet_scene(sub, mass, force_n):
    """A tag hanging under a fixed plate; with force_n > 0 it carries a constant-force adhesion actuator (a magnet,
    as construct_magnetic_ring builds it)."""
    def magnet(spec):
        g = next(g for g in spec.geoms if g.name == "tag_geom")
        g.margin, g.gap = 0.0005, 0.0005                         # adhesion acts through contacts: the face must be in contact range
        if force_n > 0:
            act = spec.add_actuator(); act.name = "tag_magnet"; act.trntype = mujoco.mjtTrn.mjTRN_BODY; act.target = "tag"
            act.gainprm[0] = 0.0; act.biastype = mujoco.mjtBias.mjBIAS_AFFINE; act.biasprm[0] = float(force_n)
            act.ctrllimited = 1; act.ctrlrange = [0.0, 1.0]
            for k in spec.keys:
                if len(k.ctrl):
                    k.ctrl = list(k.ctrl) + [0.0]
    return S.bench(tmp / sub, [dict(name="plate", size=[0.04, 0.04, 0.005], pos=[0.0, 0.45, 0.30], free=False),
                               dict(name="tag", size=[0.01, 0.01, 0.01], pos=[0.0, 0.45, 0.2850], mass=mass)], edit=magnet)


def magnet_holds():
    sc, mm = magnet_scene("mag", 0.05, 2.0)
    Q = S.rollout(sc, S.home(mm), 45)
    tz = Q[:, S.free_adr(mm, "tag") + 2]
    assert abs(tz[-1] - tz[0]) < 0.002, f"the magnet did not hold the tag in physics: z {tz[0]:.4f} -> {tz[-1]:.4f}"
    tr = S.save(tmp / "mag.npz", Q, "right: wait #1")
    r = dynamics_consistency_audit(sc, tr)
    assert r["ok"] and r["objects"]["tag"]["frames_judged"] > 10, (r["issues"], r["objects"].get("tag"))
    sc0, _ = magnet_scene("mag_none", 0.05, 0.0)                    # the same motion where nothing can pull: unexplained
    r0 = dynamics_consistency_audit(sc0, tr)
    assert not r0["ok"] and r0["worst"]["body"] == "tag", r0["objects"].get("tag")


def magnet_too_weak_fails():
    sc, mm = magnet_scene("mag_heavy", 0.5, 2.0)                    # 4.9 N of weight on a 2 N magnet, held kinematically
    q = S.home(mm)
    r = dynamics_consistency_audit(sc, S.save(tmp / "mag_heavy.npz", np.repeat(q[None], 40, 0), "right: wait #1"))
    assert not r["ok"] and r["worst"]["body"] == "tag" and r["worst"]["unexplained_m_s2"] > 4, r["objects"].get("tag")


def misuse():
    S.expect_value_error(lambda: dynamics_consistency_audit(scene, drop, mu_cap=0), "mu_cap")


C.run("a box dropped and a box pushed in physics pass", physics_passes)
C.run("a box shaken at up to 30 m/s^2 on the table with only the table touching it fails", shaken_on_the_table_fails)
C.run("a box carried by a weld to the hand fails as an attach", weld_attach_fails)
C.run("a tag held under a plate by a 2 N magnet in physics passes; the same motion with no magnet fails", magnet_holds)
C.run("a 0.5 kg tag held by a 2 N magnet fails: the pull is capped at the magnet's force", magnet_too_weak_fails)
C.run("a zero friction cap is refused", misuse)
C.exit()
