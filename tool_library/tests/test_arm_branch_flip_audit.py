"""arm_branch_flip_audit: an ordinary reach passes; an IK branch switch (joints swing 50 deg while the flange stays
within 26 mm and 1.4 deg) fails; thresholds move the verdict."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures" / "episode_audits"))
import synth as S
import mujoco
import numpy as np
from tool_library.verifiers.arm_branch_flip_audit import arm_branch_flip_audit

tmp = S.tmpdir()
scene = str(S.SETUP)
m = mujoco.MjModel.from_xml_path(scene); d = mujoco.MjData(m)
q0 = S.home(m)
arm = np.array([S.jadr(m, f"robot_right_arm_dof_joint{i}") for i in range(1, 7)])
dadr = np.array([int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"robot_right_arm_dof_joint{i}")]) for i in range(1, 7)])
fl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot_right_arm_gripper")


def fk(q):
    d.qpos[:] = q; mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
    return d.xpos[fl].copy(), d.xmat[fl].reshape(3, 3).copy()


# a reach: the flange travels about 15 cm
T = 90
Qr = np.repeat(q0[None], T, 0)
Qr[:, arm] = S.smooth(q0[arm], q0[arm] + np.array([0.4, 0.3, 0.3, 0.0, 0.0, 0.2]), T)
reach = S.save(tmp / "reach.npz", Qr, "right: reach for the box #1")

# a branch switch: j4 sweeps 50 deg while the other joints are re-solved to hold the flange (near the wrist singularity)
q = q0.copy(); q[arm[4]] = 1.0; q[arm[2]] = 0.8
p0, R0 = fk(q)
free = [0, 1, 2, 4, 5]
frames = []
for k in range(31):
    q[arm[3]] = q0[arm[3]] + np.radians(50) * k / 30
    for _ in range(60):
        p, R = fk(q)
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv)); mujoco.mj_jacBody(m, d, jp, jr, fl)
        e = np.r_[p0 - p, 0.15 * (np.cross(R[:, 0], R0[:, 0]) + np.cross(R[:, 1], R0[:, 1]) + np.cross(R[:, 2], R0[:, 2]))]
        Jm = np.vstack([jp[:, dadr[free]], 0.3 * jr[:, dadr[free]]])
        q[arm[free]] += Jm.T @ np.linalg.solve(Jm @ Jm.T + 1e-4 * np.eye(6), e)
    frames.append(q.copy())
Qf = np.vstack([np.repeat(frames[:1], 15, 0), np.array(frames), np.repeat(frames[-1:], 15, 0)])
flip = S.save(tmp / "flip.npz", Qf, "right: carry the box #2")
C = S.Cases()


def reach_passes():
    r = arm_branch_flip_audit(scene, reach)
    assert r["ok"], r["issues"]
    assert r["arms"]["right"]["flange_mm"] > 30 or r["arms"]["right"]["joint_excursion_deg"] < 45, r["arms"]["right"]


def branch_switch_fails():
    r = arm_branch_flip_audit(scene, flip)
    assert not r["ok"], r["arms"]
    w = r["worst"]
    assert w["side"] == "right" and w["joint_excursion_deg"] >= 45 and w["flange_mm"] <= 30, w
    assert "branch" in r["issues"][0], r["issues"]


def thresholds_move_the_verdict():
    assert arm_branch_flip_audit(scene, flip, min_joint_deg=80)["ok"]
    assert arm_branch_flip_audit(scene, flip, max_flange_mm=5)["ok"]
    assert not arm_branch_flip_audit(scene, flip, window_s=[2.0])["ok"]


def misuse():
    S.expect_value_error(lambda: arm_branch_flip_audit(scene, reach, window_s=[]), "window_s")
    S.expect_value_error(lambda: arm_branch_flip_audit(scene, reach, min_joint_deg=-1), "min_joint_deg")
    S.expect_value_error(lambda: arm_branch_flip_audit(scene, reach, frames=[50, 10]), "frames")


C.run("a 15 cm reach passes", reach_passes)
C.run("j4 swinging 50 deg with the flange held within 26 mm fails as a branch switch", branch_switch_fails)
C.run("a higher joint threshold or a tighter flange tolerance lets it pass; a 2 s window alone still catches it", thresholds_move_the_verdict)
C.run("empty windows, negative thresholds and a reversed frame window are refused", misuse)
C.exit()
