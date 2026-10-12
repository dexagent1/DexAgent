"""MuJoCo kinematics helpers: body poses, kinematic chains, pose IK."""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


# plain ints: MuJoCo's enum objects do not compare equal to numpy integers
FREE, BALL = int(mujoco.mjtJoint.mjJNT_FREE), int(mujoco.mjtJoint.mjJNT_BALL)
SLIDE, HINGE = int(mujoco.mjtJoint.mjJNT_SLIDE), int(mujoco.mjtJoint.mjJNT_HINGE)
WELD, CONNECT = int(mujoco.mjtEq.mjEQ_WELD), int(mujoco.mjtEq.mjEQ_CONNECT)


def load_model(path):
    path = Path(path)
    if path.suffix == ".mjb":
        return mujoco.MjModel.from_binary_path(str(path))
    return mujoco.MjModel.from_xml_path(str(path))


def body_id(model, name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert bid >= 0, f"body '{name}' is not in the scene"
    return bid


def body_pose(data, bid):
    return data.xpos[bid].copy(), data.xmat[bid].reshape(3, 3).copy()


def rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotvec(R):
    """Rotation matrix -> axis * angle."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).flatten())
    if q[0] < 0:
        q = -q
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


def slerp(Ra, Rb, w):
    """Rotation Ra turned a fraction w of the way to Rb."""
    v = rotvec(Rb @ Ra.T)
    ang = np.linalg.norm(v)
    if ang < 1e-12:
        return Ra.copy()
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, v / ang, ang * w)
    M = np.zeros(9)
    mujoco.mju_quat2Mat(M, q)
    return M.reshape(3, 3) @ Ra


def mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).flatten())
    return q


def quat2mat(q):
    M = np.zeros(9)
    mujoco.mju_quat2Mat(M, np.asarray(q, float))
    return M.reshape(3, 3)


def chain_joints(model, bid):
    """Hinge and slide joints on the path from the world to a body."""
    out = []
    b = bid
    while b != 0:
        for j in range(model.body_jntadr[b], model.body_jntadr[b] + model.body_jntnum[b]):
            if int(model.jnt_type[j]) in (HINGE, SLIDE):
                out.append(j)
        b = model.body_parentid[b]
    return sorted(out)


class Chain:
    """The joints that position one body, with a damped least squares pose IK."""

    def __init__(self, model, bid):
        self.model, self.bid = model, bid
        self.joints = chain_joints(model, bid)
        assert self.joints, f"body '{model.body(bid).name}' has no joints above it"
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.joints])
        self.dadr = np.array([model.jnt_dofadr[j] for j in self.joints])
        self.lo = np.array([model.jnt_range[j][0] for j in self.joints])
        self.hi = np.array([model.jnt_range[j][1] for j in self.joints])
        # a hinge whose range spans a full turn or more is a free rotation
        # with a bookkeeping bound, not a stop: never clamp it
        self.limited = np.array([bool(model.jnt_limited[j]) for j in self.joints]) & (
            (self.hi - self.lo) < 2 * np.pi - 1e-6)
        self._jp = np.zeros((3, model.nv))
        self._jr = np.zeros((3, model.nv))

    def solve(self, data, p_target, R_target=None, point=None, rest=None, iters=80,
              tol_p=1e-5, tol_r=1e-5, damping=1e-3, rot_weight=0.2, max_step=0.3):
        """Move this chain's joints in data.qpos so the body reaches a pose.

        R_target=None asks for position only: `point` (in the body frame,
        default its origin) must land on p_target and the orientation is free.
        rest gives joint values to stay close to: whatever freedom the target
        leaves is spent on returning to them, so a redundant chain comes back
        to its recorded posture as the target comes back to the recorded pose.
        Returns (position error m, rotation error rad)."""
        m = self.model
        local = np.zeros(3) if point is None else np.asarray(point, float)
        n = len(self.qadr)

        def errors():
            p, R = body_pose(data, self.bid)
            p = p + R @ local
            dr = np.zeros(3) if R_target is None else rotvec(R_target @ R.T)
            return p, p_target - p, dr

        for _ in range(iters):
            mujoco.mj_kinematics(m, data)
            mujoco.mj_comPos(m, data)
            p, dp, dr = errors()
            mujoco.mj_jac(m, data, self._jp, self._jr, p, self.bid)
            if R_target is None:
                J, e = self._jp[:, self.dadr], dp
            else:
                J = np.vstack([self._jp[:, self.dadr], rot_weight * self._jr[:, self.dadr]])
                e = np.concatenate([dp, rot_weight * dr])
            q = data.qpos[self.qadr]
            # a joint sitting on its stop and asked to go further is frozen for
            # this step, so the others take over instead of the clamp tearing
            # the solution to another branch
            for _pass in range(2):
                Jinv = J.T @ np.linalg.inv(J @ J.T + damping ** 2 * np.eye(len(e)))
                dq = Jinv @ e
                null = np.zeros(n)
                if rest is not None:
                    null = (np.eye(n) - Jinv @ J) @ (rest - q)
                stuck = self.limited & (((q <= self.lo + 1e-9) & (dq + null < 0))
                                        | ((q >= self.hi - 1e-9) & (dq + null > 0)))
                if not stuck.any():
                    break
                J = J.copy()
                J[:, stuck] = 0.0
            if (np.linalg.norm(dp) < tol_p and np.linalg.norm(dr) < tol_r
                    and np.abs(null).max() < 1e-6):
                break
            dq = dq + null
            dq[stuck] = 0.0
            big = np.abs(dq).max()
            if big > max_step:
                dq *= max_step / big
            q = q + dq
            data.qpos[self.qadr] = np.where(self.limited, np.clip(q, self.lo, self.hi), q)
        mujoco.mj_kinematics(m, data)
        _p, dp, dr = errors()
        return float(np.linalg.norm(dp)), float(np.linalg.norm(dr))
