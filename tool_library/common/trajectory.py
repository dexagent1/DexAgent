"""Shared helpers for trajectory skills and audits.

Everything here works on a MuJoCo model and a qpos trajectory `Q[T, nq]`:
replaying it kinematically, reading body poses, solving an arm chain by
inverse kinematics, fitting an arm to a wrist path, and the measurements the
Episode Audits and Task & Rollout Gates are built on. Numbers come out in
metres, radians (degrees where the name says so) and frames.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

# MuJoCo enum values as plain ints: the enum objects do not compare equal to
# numpy integers in recent versions.
FREE = int(mujoco.mjtJoint.mjJNT_FREE)
BALL = int(mujoco.mjtJoint.mjJNT_BALL)
SLIDE = int(mujoco.mjtJoint.mjJNT_SLIDE)
HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
WELD = int(mujoco.mjtEq.mjEQ_WELD)
CONNECT = int(mujoco.mjtEq.mjEQ_CONNECT)


# ----------------------------------------------------------------- model --
def load_model(scene):
    """An MjModel from a path (.xml or .mjb) or the model itself."""
    if isinstance(scene, mujoco.MjModel):
        return scene
    p = Path(scene)
    if p.suffix == ".mjb":
        return mujoco.MjModel.from_binary_path(str(p))
    return mujoco.MjModel.from_xml_path(str(p))


def body_id(model, name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise ValueError(f"body '{name}' is not in the scene")
    return bid


def joint_id(model, name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise ValueError(f"joint '{name}' is not in the scene")
    return jid


def joint_qadr(model, names):
    """qpos column of each named scalar joint (hinge or slide)."""
    out = []
    for n in names:
        j = joint_id(model, n)
        if int(model.jnt_type[j]) not in (HINGE, SLIDE):
            raise ValueError(f"joint '{n}' is not a hinge or slide")
        out.append(int(model.jnt_qposadr[j]))
    return np.asarray(out, int)


def free_joint_qadr(model, body):
    """qpos address of a body's free joint (pos xyz, quat wxyz), or None."""
    bid = body_id(model, body) if isinstance(body, str) else int(body)
    for j in range(model.body_jntadr[bid], model.body_jntadr[bid] + model.body_jntnum[bid]):
        if int(model.jnt_type[j]) == FREE:
            return int(model.jnt_qposadr[j])
    return None


def chain_joints(model, bid):
    """Hinge and slide joints on the path from the world to a body."""
    out, b = [], bid
    while b != 0:
        for j in range(model.body_jntadr[b], model.body_jntadr[b] + model.body_jntnum[b]):
            if int(model.jnt_type[j]) in (HINGE, SLIDE):
                out.append(j)
        b = model.body_parentid[b]
    return sorted(out)


def subtree(model, bid):
    out, stack = set(), [int(bid)]
    while stack:
        b = stack.pop()
        out.add(b)
        stack += [c for c in range(model.nbody) if model.body_parentid[c] == b and c != b]
    return out


def body_geoms(model, bid):
    return [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid]


def geom_world_aabb(model, data, gids):
    """World axis-aligned bounds of a set of geoms (from their sizes, not
    their meshes: good to a few millimetres for boxes and cylinders)."""
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in gids:
        R = np.asarray(data.geom_xmat[g]).reshape(3, 3)
        half = np.abs(R) @ np.asarray(model.geom_size[g], float)
        c = np.asarray(data.geom_xpos[g], float)
        lo, hi = np.minimum(lo, c - half), np.maximum(hi, c + half)
    return lo, hi


# ------------------------------------------------------------- rotations --
def mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).flatten())
    return q


def quat2mat(q):
    M = np.zeros(9)
    mujoco.mju_quat2Mat(M, np.asarray(q, float))
    return M.reshape(3, 3)


def rotvec(R):
    """Rotation matrix -> axis * angle (radians)."""
    q = mat2quat(R)
    if q[0] < 0:
        q = -q
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def euler_to_mat(e):
    """Intrinsic XYZ Euler angles (the usual 3-hinge wrist) -> matrix."""
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_euler("XYZ", np.asarray(e, float)).as_matrix()


def mat_to_euler(R):
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_matrix(np.asarray(R, float)).as_euler("XYZ")


def angle_between(Ra, Rb):
    """Geodesic angle between two rotations, radians."""
    return float(np.linalg.norm(rotvec(np.asarray(Rb) @ np.asarray(Ra).T)))


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5


def quintic_leg(q0, v0, q1, v1, n, fps):
    """C2 joint leg over n frames with end velocities (per second) and zero
    end acceleration, so a transit hands over without a jerk."""
    Tt = n / fps
    t = np.linspace(0.0, 1.0, n)[:, None]
    h00 = 1 - 10 * t ** 3 + 15 * t ** 4 - 6 * t ** 5
    h10 = t - 6 * t ** 3 + 8 * t ** 4 - 3 * t ** 5
    h01 = 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5
    h11 = -4 * t ** 3 + 7 * t ** 4 - 3 * t ** 5
    return (h00 * np.asarray(q0) + h10 * (np.asarray(v0) * Tt)
            + h01 * np.asarray(q1) + h11 * (np.asarray(v1) * Tt))


# ------------------------------------------------------------ trajectory --
def replay_poses(model, Q, bodies):
    """Kinematic replay: world position [T, B, 3] and rotation [T, B, 3, 3]
    of the named bodies along a qpos trajectory."""
    Q = np.atleast_2d(np.asarray(Q, float))
    if Q.shape[1] != model.nq:
        raise ValueError(f"Q has {Q.shape[1]} columns, the scene has nq={model.nq}")
    ids = [body_id(model, b) if isinstance(b, str) else int(b) for b in bodies]
    data = mujoco.MjData(model)
    P = np.zeros((len(Q), len(ids), 3))
    R = np.zeros((len(Q), len(ids), 3, 3))
    for t in range(len(Q)):
        data.qpos[:] = Q[t]
        mujoco.mj_kinematics(model, data)
        P[t] = data.xpos[ids]
        R[t] = data.xmat[ids].reshape(-1, 3, 3)
    return P, R


def motion_window(positions, thresh_m=0.06, min_frames=4, pad=3):
    """[first, last] frame of sustained displacement from the start pose, or
    None when the object never really moves (fit noise stays below thresh)."""
    P = np.asarray(positions, float)
    d = np.linalg.norm(P - P[0], axis=1)
    idx = np.flatnonzero(d > thresh_m)
    if len(idx) < min_frames:
        return None
    return (max(0, int(idx[0]) - pad), min(len(P) - 1, int(idx[-1]) + pad))


def phase_runs(labels):
    """[(label, first frame, last frame)] for consecutive identical labels."""
    runs = []
    for i, lab in enumerate(labels):
        if runs and runs[-1][0] == lab:
            runs[-1][2] = i
        else:
            runs.append([lab, i, i])
    return [tuple(r) for r in runs]


def per_frame_steps(X):
    """Largest absolute per-frame change of each row of X[T, k] -> [T-1]."""
    X = np.atleast_2d(np.asarray(X, float))
    if X.ndim == 1:
        X = X[:, None]
    return np.abs(np.diff(X, axis=0)).max(axis=1) if len(X) > 1 else np.zeros(0)


def rotation_steps_deg(R):
    """Attitude change between consecutive frames of R[T, 3, 3], degrees."""
    if len(R) < 2:
        return np.zeros(0)
    trc = np.einsum("nij,nij->n", R[:-1], R[1:])
    return np.degrees(np.arccos(np.clip((trc - 1.0) / 2.0, -1.0, 1.0)))


def savgol(Q, window=9, poly=3):
    from scipy.signal import savgol_filter
    Q = np.asarray(Q, float)
    if len(Q) <= window:
        return Q.copy()
    return savgol_filter(Q, window, poly, axis=0)


def rate_limit(Q, dq_max):
    """Clamp per-frame joint steps to dq_max, forward and backward, so the
    result is continuous from both ends."""
    Q = np.asarray(Q, float).copy()
    for i in range(1, len(Q)):
        Q[i] = np.clip(Q[i], Q[i - 1] - dq_max, Q[i - 1] + dq_max)
    for i in range(len(Q) - 2, -1, -1):
        Q[i] = np.clip(Q[i], Q[i + 1] - dq_max, Q[i + 1] + dq_max)
    return Q


# ------------------------------------------------------------- IK chain --
class Chain:
    """The joints that position one body, with a damped least squares IK.

    A hinge whose range spans a full turn is treated as a free rotation.
    Joints sitting on a stop and asked to go further are frozen for the
    step so the others take over instead of the clamp tearing the solution
    onto another branch."""

    def __init__(self, model, bid, joints=None):
        self.model, self.bid = model, int(bid)
        self.joints = list(joints) if joints is not None else chain_joints(model, self.bid)
        if not self.joints:
            raise ValueError(f"body '{model.body(self.bid).name}' has no joints above it")
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.joints])
        self.dadr = np.array([model.jnt_dofadr[j] for j in self.joints])
        self.lo = np.array([model.jnt_range[j][0] for j in self.joints], float)
        self.hi = np.array([model.jnt_range[j][1] for j in self.joints], float)
        self.limited = np.array([bool(model.jnt_limited[j]) for j in self.joints]) & (
            (self.hi - self.lo) < 2 * np.pi - 1e-6)
        self._jp = np.zeros((3, model.nv))
        self._jr = np.zeros((3, model.nv))

    def set_limits(self, lo, hi):
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        self.limited = np.ones(len(self.qadr), bool)

    def pose(self, data, point=None):
        p = np.asarray(data.xpos[self.bid], float)
        R = np.asarray(data.xmat[self.bid], float).reshape(3, 3)
        if point is not None:
            p = p + R @ np.asarray(point, float)
        return p, R

    def solve(self, data, p_target, R_target=None, point=None, rest=None, iters=80,
              tol_p=1e-5, tol_r=1e-5, damping=1e-3, rot_weight=0.2, max_step=0.3):
        """Move this chain's joints in data.qpos so the body (or `point` in
        its frame) reaches p_target and, unless R_target is None, R_target.
        `rest` pulls the redundant directions toward those joint values.
        Returns (position error m, rotation error rad)."""
        m, n = self.model, len(self.qadr)
        local = np.zeros(3) if point is None else np.asarray(point, float)
        for _ in range(iters):
            mujoco.mj_kinematics(m, data)
            mujoco.mj_comPos(m, data)
            p, R = self.pose(data, local)
            dp = np.asarray(p_target, float) - p
            dr = np.zeros(3) if R_target is None else rotvec(np.asarray(R_target) @ R.T)
            mujoco.mj_jac(m, data, self._jp, self._jr, p, self.bid)
            if R_target is None:
                J, e = self._jp[:, self.dadr], dp
            else:
                J = np.vstack([self._jp[:, self.dadr], rot_weight * self._jr[:, self.dadr]])
                e = np.concatenate([dp, rot_weight * dr])
            q = data.qpos[self.qadr]
            for _pass in range(2):
                Jinv = J.T @ np.linalg.inv(J @ J.T + damping ** 2 * np.eye(len(e)))
                dq = Jinv @ e
                null = np.zeros(n)
                if rest is not None:
                    null = (np.eye(n) - Jinv @ J) @ (np.asarray(rest, float) - q)
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
        p, R = self.pose(data, local)
        er = 0.0 if R_target is None else float(np.linalg.norm(rotvec(np.asarray(R_target) @ R.T)))
        return float(np.linalg.norm(np.asarray(p_target, float) - p)), er


# -------------------------------------------------- arm fitted to a path --
# The DLS settings a real arm was driven with: damping and step size that
# converge from a warm start in a dozen iterations, a rotation weight that
# keeps position the primary goal, and a per-axis rotation weight that makes
# the roll axis (the real wrist stop) the soft one.
ARM_FIT = dict(damp=0.08, rot_w=0.35, step=0.25, iters=12, iters_first=200,
               rot_w_ax=(1.0, 1.0, 0.75), frame_dq_max=0.35,
               reset_pos_mm=30.0, reset_tilt_deg=20.0, reset_blend=12, reset_cooldown=30)


def _arm_step(model, data, ch, q, pt, Rt, iters, cfg):
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    w_ax = np.asarray(cfg["rot_w_ax"], float)
    for _ in range(iters):
        data.qpos[ch.qadr] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        p = data.xpos[ch.bid]
        Rf = data.xmat[ch.bid].reshape(3, 3)
        ep = pt - p
        er = w_ax * (Rt.T @ rotvec(Rt @ Rf.T))
        if np.linalg.norm(ep) < 3e-4 and (cfg["rot_w"] == 0.0 or np.linalg.norm(er) < 5e-3):
            break
        mujoco.mj_jacBody(model, data, jacp, jacr, ch.bid)
        Jr = w_ax[:, None] * (Rt.T @ jacr[:, ch.dadr])
        J = np.vstack([jacp[:, ch.dadr], cfg["rot_w"] * Jr])
        e = np.concatenate([ep, cfg["rot_w"] * er])
        dq = J.T @ np.linalg.solve(J @ J.T + cfg["damp"] ** 2 * np.eye(6), e)
        n = np.linalg.norm(dq)
        if n > cfg["step"]:
            dq *= cfg["step"] / n
        q = np.clip(q + dq, ch.lo, ch.hi)
    return q


def _arm_residual(model, data, ch, q, pt, Rt):
    data.qpos[ch.qadr] = q
    mujoco.mj_kinematics(model, data)
    Rf = data.xmat[ch.bid].reshape(3, 3)
    er = Rt.T @ rotvec(Rt @ Rf.T)
    return (1000.0 * float(np.linalg.norm(pt - data.xpos[ch.bid])),
            float(np.degrees(np.linalg.norm(er[:2]))), float(np.degrees(abs(er[2]))))


def _arm_multistart(model, data, ch, pt, Rt, seeds, rng, cfg, n_random=6, iters=150):
    cands = [np.clip(np.asarray(c, float), ch.lo, ch.hi) for c in seeds]
    cands += [rng.uniform(ch.lo, ch.hi) for _ in range(n_random)]
    best = None
    for c in cands:
        qc = _arm_step(model, data, ch, c.copy(), pt, Rt, iters, cfg)
        pe, ti, ro = _arm_residual(model, data, ch, qc, pt, Rt)
        score = pe + (0.0 if cfg["rot_w"] == 0.0 else 3 * ti + 0.5 * ro)
        if best is None or score < best[0]:
            best = (score, qc)
    return best


def fit_arm_to_poses(model, chain, pos, rot, seed=None, rng=None, cfg=None):
    """Fit an arm chain to a sequence of flange poses, frame by frame.

    pos [T, 3], rot [T, 3, 3]: where the chain's body must be. Each frame
    is warm-started from the previous one inside a trust region (no branch
    jumps); a frame the warm chain leaves badly off-pose triggers one
    multi-start branch reset, blended over the preceding frames so the
    command stays continuous. Returns (Q [T, n], pos_mm [T], tilt_deg [T],
    roll_deg [T], reset frames)."""
    cfg = {**ARM_FIT, **(cfg or {})}
    data = mujoco.MjData(model)
    rng = rng or np.random.default_rng(0)
    T = len(pos)
    seed = np.zeros(len(chain.qadr)) if seed is None else np.asarray(seed, float)
    q = _arm_multistart(model, data, chain, pos[0], rot[0], [seed, np.zeros_like(seed)],
                        rng, cfg, n_random=8, iters=cfg["iters_first"])[1]
    Q = np.zeros((T, len(chain.qadr)))
    rp, rt, rr = np.zeros(T), np.zeros(T), np.zeros(T)
    resets, last_reset = [], -10 ** 9
    for t in range(T):
        if t:
            q_prev = Q[t - 1]
            q = _arm_step(model, data, chain, q.copy(), pos[t], rot[t], cfg["iters"], cfg)
            pe, ti, _ro = _arm_residual(model, data, chain, q, pos[t], rot[t])
            if ((pe > cfg["reset_pos_mm"] or ti > cfg["reset_tilt_deg"])
                    and t - last_reset >= cfg["reset_cooldown"]):
                sc = _arm_multistart(model, data, chain, pos[t], rot[t],
                                     [np.zeros_like(seed), seed, q], rng, cfg)
                pe_r, ti_r, _ = _arm_residual(model, data, chain, sc[1], pos[t], rot[t])
                wt = 0.0 if cfg["rot_w"] == 0.0 else 3.0
                if (pe_r + wt * ti_r) < 0.5 * (pe + wt * ti):
                    q, last_reset = sc[1], t
                    resets.append(t)
                    k = min(cfg["reset_blend"], t)
                    for j in range(1, k):
                        u = smoothstep(j / k)
                        Q[t - k + j] = (1 - u) * Q[t - k] + u * q
                        rp[t - k + j], rt[t - k + j], rr[t - k + j] = _arm_residual(
                            model, data, chain, Q[t - k + j], pos[t - k + j], rot[t - k + j])
                else:
                    q = np.clip(q, q_prev - cfg["frame_dq_max"], q_prev + cfg["frame_dq_max"])
                    q = np.clip(q, chain.lo, chain.hi)
            else:
                q = np.clip(q, q_prev - cfg["frame_dq_max"], q_prev + cfg["frame_dq_max"])
                q = np.clip(q, chain.lo, chain.hi)
        Q[t] = q
        rp[t], rt[t], rr[t] = _arm_residual(model, data, chain, q, pos[t], rot[t])
    return Q, rp, rt, rr, resets


def pose_track(model, Q, bid):
    """World position and rotation of one body along Q."""
    P, R = replay_poses(model, Q, [bid])
    return P[:, 0], R[:, 0]


# -------------------------------------------------------------- contacts --
def classify_body(name, robot_prefix="robot_", table_names=("table", "floor", "ground")):
    """Coarse class of a body for contact bookkeeping: robot_left / robot_right
    / robot, table, or the body's own first name token."""
    if name.startswith(robot_prefix):
        side = "left" if "left" in name else ("right" if "right" in name else "")
        return f"robot_{side}" if side else "robot"
    low = name.lower()
    if any(t in low for t in table_names):
        return "table"
    return name


def contact_profile(model, Q, stride=1):
    """Deepest penetration per geom pair over Q:
    {(geom a, geom b): (depth m, frame, (body a, body b), (geom name a, geom name b))}.
    Uses the model's own collision geometry (see collidable_model for all-visible)."""
    data = mujoco.MjData(model)
    out = {}
    for t in range(0, len(Q), max(1, stride)):
        data.qpos[:] = Q[t]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)
        for i in range(data.ncon):
            c = data.contact[i]
            if c.dist >= 0:
                continue
            g1, g2 = sorted((int(c.geom1), int(c.geom2)))
            depth = -float(c.dist)
            if depth > out.get((g1, g2), (0.0,))[0]:
                b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
                out[(g1, g2)] = (depth, t, (min(b1, b2), max(b1, b2)),
                                 (model.geom(g1).name or f"geom{g1}", model.geom(g2).name or f"geom{g2}"))
    return out


def welded_pairs(model):
    """(body a, body b) of active weld/connect constraints."""
    out = []
    for e in range(model.neq):
        if model.eq_active0[e] and int(model.eq_type[e]) in (WELD, CONNECT):
            out.append((int(model.eq_obj1id[e]), int(model.eq_obj2id[e])))
    return out


# ------------------------------------------------------------- gates ----
# Thresholds the paper's rollouts were gated on. Each gate function takes
# measured arrays and returns {"ok", ...numbers...}; the thresholds are
# arguments with these defaults so a task can tighten them.
GATES = dict(
    final_pos_err_m=0.10,       # object end pose vs the demo's end pose
    lift_ratio_min=0.5,         # rise vs the demo's own rise ...
    lift_ref_floor_m=0.04,      # ... only judged when the demo rose this much
    carry_contact_frac=0.5,     # hand touching the object while it travels
    penetration_p95_mm=12.0,    # object penetration while carried
    end_penetration_mm=5.0,
    track_mean_m=0.06,          # per-frame object error vs the demo path
    enclosure_min=0.25,         # multi-finger contact fraction over the hold window
    gentle_max_mps=1.5,
    tilt_deg_max=45.0,
    table_clearance_min_mm=-12.0,
    q_tol=0.06,
)
# Per-frame step budgets at 30 fps for a replayable episode; above these a
# replay looks like a jump. arm 3.5 deg/frame is an 85 deg/s command cap.
STEP_LIMITS = dict(arm_deg=3.5, wrist_mm=15.0, wrist_deg=8.0, finger=0.30, arm_jerk_deg=3.0)
PEN_OK_MM = 1.5   # sub-millimetre interpenetration is how soft contacts carry load


def worst_step(values, limit, labels=None, offset=1):
    v = np.asarray(values, float)
    if len(v) == 0:
        return {"max": 0.0, "limit": limit, "p99": 0.0, "at": None, "over_limit_frames": 0, "ok": True}
    i = int(np.argmax(v))
    return {"max": round(float(v[i]), 3), "limit": limit, "p99": round(float(np.percentile(v, 99)), 3),
            "at": (labels[min(i + offset, len(labels) - 1)] if labels is not None else i + offset),
            "over_limit_frames": int((v > limit).sum()), "ok": bool(v.max() <= limit)}


# ------------------------------------------------------- episode loading --
def load_trajectory(path):
    """trajectory.npz -> (Q [T, nq], labels [T] str, fps). Accepts `qpos` or
    `Q`; labels and fps are optional (empty labels, 30 fps)."""
    z = np.load(str(path), allow_pickle=True)
    key = "qpos" if "qpos" in z.files else ("Q" if "Q" in z.files else None)
    if key is None:
        raise ValueError(f"{path} has no 'qpos' array (keys: {z.files})")
    Q = np.asarray(z[key], float)
    labels = [str(x) for x in z["labels"]] if "labels" in z.files else [""] * len(Q)
    fps = float(z["fps"]) if "fps" in z.files else 30.0
    return Q, labels, fps


def slerp(Ra, Rb, w):
    """Rotation Ra turned a fraction w of the way to Rb."""
    v = rotvec(np.asarray(Rb) @ np.asarray(Ra).T)
    ang = np.linalg.norm(v)
    if ang < 1e-12:
        return np.array(Ra, float)
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, v / ang, ang * w)
    return quat2mat(q) @ Ra


def apply_rigid(D, p, R):
    """Rigid world transform D = (R, t) applied to a pose (p, R)."""
    return D[0] @ p + D[1], D[0] @ R


def joined_bodies(model):
    """Body pairs bolted together by an equality constraint (hand on arm)."""
    return {(min(a, b), max(a, b)) for a, b in welded_pairs(model)}


def discover_sides(model, end_effectors):
    """Per side: the end-effector chain and the arm bolted to it, if any.

    end_effectors: {"right": body name, ...}. An arm is any chain whose tip
    body is joined to the end effector by a weld or connect constraint and
    that shares no joint with it. A weld with torquescale 0 or a connect is
    position-only: the arm's orientation is then free and only its attachment
    point has to follow."""
    sides = {}
    for side, body in end_effectors.items():
        bid = body_id(model, body)
        # the weld may hold an ancestor of the named body (the palm named, the hand root welded): walk up to it
        b_w, welded = bid, None
        while b_w > 0 and welded is None:
            for e in range(model.neq):
                if model.eq_active0[e] and int(model.eq_type[e]) in (WELD, CONNECT) and b_w in (int(model.eq_obj1id[e]), int(model.eq_obj2id[e])):
                    welded = b_w
                    break
            b_w = int(model.body_parentid[b_w])
        if welded is not None and welded != bid:
            bid = welded
        ee = Chain(model, bid)
        follower = None
        for e in range(model.neq):
            kind = int(model.eq_type[e])
            if not model.eq_active0[e] or kind not in (WELD, CONNECT):
                continue
            b1, b2 = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
            if bid not in (b1, b2):
                continue
            tip = b1 if b2 == bid else b2
            try:
                ch = Chain(model, tip)
            except ValueError:
                continue
            if set(ch.joints) & set(ee.joints):
                continue
            d = model.eq_data[e]
            if kind == CONNECT:
                pos_only, point = True, (d[0:3] if tip == b1 else d[3:6])
            else:
                pos_only = float(d[10]) == 0.0
                point = d[0:3] if tip == b2 else d[3:6] + quat2mat(d[6:10]) @ d[0:3]
            follower = {"chain": ch, "point": np.array(point, float), "pos_only": pos_only}
        sides[side] = {"ee": ee, "follower": follower}
    return sides


def limb_bodies(model, sides):
    """Bodies of each arm-plus-hand assembly: the end effector's subtree, the
    chain above it, and the arm chain above the follower tip."""
    groups = []
    for s in sides.values():
        bodies = subtree(model, s["ee"].bid)
        for start in [s["ee"].bid] + ([s["follower"]["chain"].bid] if s["follower"] else []):
            b = start
            while b != 0:
                bodies.add(b)
                b = model.body_parentid[b]
        groups.append(bodies)
    return groups


def robot_chains(model, sides):
    """{"<side>_ee": qpos columns, "<side>_arm": qpos columns, "<side>_fingers": ...}
    for the joint-speed and jerk audits."""
    out = {}
    for side, s in sides.items():
        out[f"{side}_ee"] = np.asarray(s["ee"].qadr, int)
        if s["follower"]:
            out[f"{side}_arm"] = np.asarray(s["follower"]["chain"].qadr, int)
        fingers = []
        for b in subtree(model, s["ee"].bid) - {s["ee"].bid}:
            for j in range(model.body_jntadr[b], model.body_jntadr[b] + model.body_jntnum[b]):
                if int(model.jnt_type[j]) in (HINGE, SLIDE):
                    fingers.append(int(model.jnt_qposadr[j]))
        if fingers:
            out[f"{side}_fingers"] = np.asarray(sorted(fingers), int)
    return out


def collidable_model(scene_path, visual_groups=(0, 1, 2), keep_own_collision=True):
    """The MJCF recompiled so every visible geom collides. Returns (model,
    number of geoms switched on). Needs the .xml: convex hulls are built at
    compile time, so a compiled model cannot be patched afterwards.

    keep_own_collision: a body that already has collision geoms of its own (contype/conaffinity set: a bowl built
    as a ring of wall pieces) is measured by them, not by the convex hull of its visual mesh, which would fill the
    bowl and report a cup resting inside it as 5 cm deep. Bodies without (arm links, the hand's visual shells) get
    their visible meshes switched on."""
    p = Path(scene_path)
    if p.suffix.lower() == ".mjb":
        raise ValueError("the collision copy needs the scene as MJCF (.xml), not .mjb")
    spec = mujoco.MjSpec.from_file(str(p))
    has_col = set()
    if keep_own_collision:
        for g in spec.geoms:
            if (g.contype or g.conaffinity) and g.parent is not None and not (g.parent.name or "").startswith("robot_"):
                has_col.add(g.parent.name)
    on = 0
    plane = int(mujoco.mjtGeom.mjGEOM_PLANE)
    for g in spec.geoms:
        if g.parent is not None and g.parent.name in has_col:
            continue
        if (int(g.group) in visual_groups and g.rgba[3] > 0.0 and g.contype == 0
                and g.conaffinity == 0 and int(g.type) != plane):
            g.contype = g.conaffinity = 1
            on += 1
    return spec.compile(), on


def free_objects(model, robot_prefix="robot_"):
    """{body name: free-joint qpos address} of every free body that is not
    part of the robot: the things an episode can move."""
    out = {}
    for b in range(1, model.nbody):
        name = model.body(b).name
        if name.startswith(robot_prefix):
            continue
        adr = free_joint_qadr(model, b)
        if adr is not None:
            out[name] = adr
    return out


def frames_matching(labels, patterns):
    """Frames whose phase label matches any of the fnmatch patterns."""
    import fnmatch
    if isinstance(patterns, str):
        patterns = [patterns]
    return [t for t, lab in enumerate(labels)
            if any(fnmatch.fnmatchcase(lab, p) for p in patterns)]


def as_trajectory(trajectory, labels=None, fps=None):
    """An npz path, an array, or a (Q, labels, fps) tuple -> (Q, labels, fps)."""
    if isinstance(trajectory, (str, Path)):
        Q, lab, f = load_trajectory(trajectory)
    elif isinstance(trajectory, tuple) and len(trajectory) == 3:
        Q, lab, f = np.asarray(trajectory[0], float), list(trajectory[1]), float(trajectory[2])
    else:
        Q = np.atleast_2d(np.asarray(trajectory, float))
        lab, f = [""] * len(Q), 30.0
    if labels is not None:
        lab = [str(x) for x in labels]
    if fps is not None:
        f = float(fps)
    if len(lab) != len(Q):
        raise ValueError(f"{len(lab)} labels for {len(Q)} frames")
    return Q, lab, f


def resolve_chains(model, chains=None, end_effectors=None, robot_prefix="robot_"):
    """Named qpos column groups to audit. Explicit {name: [joint names]} wins;
    then end effectors (ee / arm / fingers per side); else every scalar joint
    of bodies with the robot prefix as one group."""
    if chains:
        return {k: joint_qadr(model, v) for k, v in chains.items()}
    if end_effectors:
        return robot_chains(model, discover_sides(model, end_effectors))
    cols = []
    for j in range(model.njnt):
        if int(model.jnt_type[j]) in (HINGE, SLIDE) and \
                model.body(model.jnt_bodyid[j]).name.startswith(robot_prefix):
            cols.append(int(model.jnt_qposadr[j]))
    if not cols:
        raise ValueError(f"no scalar joints on bodies starting with '{robot_prefix}'; "
                         "pass chains or end_effectors")
    return {"robot": np.asarray(cols, int)}


# ------------------------------------------------------------ bounds ----
def mesh_bounds_cache(model):
    cache = {}
    for g in range(model.ngeom):
        if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH):
            mid = int(model.geom_dataid[g])
            v0, nv = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
            cache[g] = np.asarray(model.mesh_vert[v0:v0 + nv], float)
    return cache


def _geom_half_extent(model, g, R):
    """World-axis half extents of a primitive geom about its centre. geom_size means a different thing per type
    (a cylinder's is [radius, half-height, 0]), so each type gets its own exact bound."""
    T = mujoco.mjtGeom
    t, s = int(model.geom_type[g]), np.asarray(model.geom_size[g], float)
    if t == int(T.mjGEOM_SPHERE):
        return np.full(3, s[0])
    if t in (int(T.mjGEOM_CAPSULE), int(T.mjGEOM_CYLINDER)):
        a = np.abs(R[:, 2])                                   # the geom's axis in the world
        rim = s[0] if t == int(T.mjGEOM_CAPSULE) else s[0] * np.sqrt(np.clip(1.0 - a * a, 0.0, 1.0))
        return a * s[1] + rim
    if t == int(T.mjGEOM_ELLIPSOID):
        return np.sqrt((R * R) @ (s * s))
    if t in (int(T.mjGEOM_BOX), int(T.mjGEOM_PLANE)):
        return np.abs(R) @ s
    return np.abs(R) @ np.asarray(model.geom_aabb[g][3:], float)   # height fields and others: their local box


def body_bounds(model, data, bids, mesh_cache, geom_ok=None):
    """World AABB of the geoms of a set of bodies. Meshes use their vertices, primitives their exact extent."""
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) not in bids or (geom_ok and not geom_ok(g)):
            continue
        R = np.asarray(data.geom_xmat[g]).reshape(3, 3)
        c = np.asarray(data.geom_xpos[g], float)
        if g in mesh_cache:
            V = mesh_cache[g] @ R.T + c
            lo, hi = np.minimum(lo, V.min(axis=0)), np.maximum(hi, V.max(axis=0))
        else:
            if int(model.geom_type[g]) not in (int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_PLANE)):
                c = c + R @ np.asarray(model.geom_aabb[g][:3], float)
            half = _geom_half_extent(model, g, R)
            lo, hi = np.minimum(lo, c - half), np.maximum(hi, c + half)
    return lo, hi


def table_top_from_scene(model, name="table"):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise ValueError("no body named 'table'; pass table_top_z")
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    return float(body_bounds(model, data, {bid}, mesh_bounds_cache(model))[1][2])
