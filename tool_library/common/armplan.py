"""Arm motion planning: collision-aware joint-space legs, an idle arm parked out of the way, and retiming.

The executor fills each arm by IK along the hand reference and joins keyframes with straight joint-space legs
(armhand.couple). Nothing looked at the way between two keyframes, and measured failures came from exactly that: a
steadying arm's forearm in the acting hand's path, the parked right thumb hit by the left hand, an arm swept through a
part on a straight swing, and legs too fast for the arm rejected by the speed audits instead of slowed. This module
gives couple those capabilities; each one is a no-op when the straight plan is already clean.

  ArmPlanner(scene)             a collision copy of the bench: the arms' links as their convex hulls, the hands'
                                collision geoms (palm and thumb hulls included), the objects and the table.
                                clearance(q) is the smallest signed distance between what moves (one arm, its hand and
                                what the hand carries) and everything else, including its own arm against its own hand
                                (the flange and the palm it is bolted to excepted).
  ArmPlanner.plan_leg(...)      RRT-Connect in the arm's joint box between two arm configurations, then shortcut and
                                corner-smoothed, keeping `margin` from everything. Near the two ends, where the plan
                                puts the hand close to a part on purpose, the required clearance shrinks to what that
                                end has (a cone that widens by slope_m_rad per radian away from it).
  ArmPlanner.park(...)          a clearance configuration for an arm whose hand has nothing to do in an entry: out of
                                the active arm's and hand's swept volume, collision free and close to where the arm is,
                                plus the collision-free way there.
  retime(X, fps, vmax, amax)    a time map that slows the trajectory only where a joint velocity or a smoothed
                                acceleration limit is exceeded (measured the way joint_acceleration_audit measures it);
                                frames in `keep` (contact phases) keep their pace unless their own limit forces it.
  retime_arms(model, Q, fps)    retime on a full qpos trajectory: the arm joints and the flanges' linear acceleration of
                                a bench scene, every column resampled, free-body quaternions kept unit. It reads nothing
                                from Stage 3 and can be called on any trajectory (a Stage 4 layout warped from one).

mj_geomDistance returns exact zeros for some separated hull pairs (hand.geom_distance); the planner reads distances
from mj_collision with a contact margin and re-measures an exact zero the same way.
"""
from __future__ import annotations

import time

import mujoco
import numpy as np
from scipy.signal import savgol_filter

from tool_library.common import trajectory as TJ

DEFAULTS = dict(margins_m=(0.02, 0.01, 0.005),    # the clearance a planned leg keeps, tried in this order
                trigger_m=0.0,                    # a straight leg is replanned when it comes closer than this (it penetrates)
                slope_m_rad=0.15,                 # near an end the required clearance is the end's own + this per radian away
                res_rad=0.02,                     # largest joint change between two checked configurations
                step_rad=0.25,                    # RRT extension step
                iters=3000, time_s=3.0,           # RRT budget per margin
                shortcut_iters=150,
                park_margin_m=0.03,               # an idle arm is parked when the active side comes closer than this
                engage_m=0.012,                   # an idle hand this close to an object is holding or steadying it: not parked
                vmax_deg_s=90.0, amax_rad_s2=12.0,  # under joint_acceleration_audit's 15 rad/s^2 (smoothed over 0.3 s)
                flange_amax_m_s2=4.8,             # under its 6 m/s^2 flange limit
                window_s=0.3, seed=0)

# collision roles (contype bits); the conaffinity of each role lists the roles it is checked against
_A, _AM, _H, _HM, _O = 1, 2, 4, 8, 16
_AFFINITY = {_A: _H | _HM | _O, _AM: _H | _O, _H: _A | _AM | _O, _HM: _A | _O, _O: _A | _AM | _H | _HM}


def _euler_xyz(R):
    """Intrinsic XYZ angles of R (R = Rx(a) Ry(b) Rz(c)), closed form (scipy's per-call overhead made the hand
    placement the planner's bottleneck)."""
    sb = float(np.clip(R[0, 2], -1.0, 1.0))
    b = np.arcsin(sb)
    if abs(sb) < 1.0 - 1e-9:
        return np.array([np.arctan2(-R[1, 2], R[2, 2]), b, np.arctan2(-R[0, 1], R[0, 0])])
    return np.array([np.arctan2(R[2, 1], R[1, 1]), b, 0.0])     # gimbal lock: the roll absorbs the yaw


def _quat_cols(model):
    """qpos addresses of every free-joint and ball-joint quaternion (4 columns each)."""
    out = []
    for j in range(model.njnt):
        t = int(model.jnt_type[j])
        if t == int(mujoco.mjtJoint.mjJNT_FREE):
            out.append(int(model.jnt_qposadr[j]) + 3)
        elif t == int(mujoco.mjtJoint.mjJNT_BALL):
            out.append(int(model.jnt_qposadr[j]))
    return out


def resample(X, src, quat_cols=()):
    """X [T, n] sampled at fractional source indices src [T'] (linear interpolation; the 4-column quaternion blocks
    starting at `quat_cols` are renormalized)."""
    X = np.asarray(X, float)
    src = np.clip(np.asarray(src, float), 0.0, len(X) - 1.0)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, len(X) - 1)
    w = (src - i0)[:, None]
    Y = (1 - w) * X[i0] + w * X[i1]
    for a in quat_cols:
        Y[:, a:a + 4] /= np.maximum(np.linalg.norm(Y[:, a:a + 4], axis=1, keepdims=True), 1e-12)
    return Y


def smoothed_acceleration(X, fps, window_s=0.3):
    """Second derivative per column, Savitzky-Golay (cubic, window_s), in units/s^2: joint_acceleration_audit's measure."""
    X = np.asarray(X, float)
    W = max(5, int(round(window_s * fps)) | 1)
    if len(X) < W + 2:
        return np.zeros_like(X)
    return savgol_filter(X, W, 3, deriv=2, axis=0) * float(fps) ** 2


def _limit(v, n):
    a = np.atleast_1d(np.asarray(np.inf if v is None else v, float))
    return np.full(n, float(a[0])) if len(a) == 1 else a


def _profile(req, beta):
    """Slow-down per source step: the largest path speed (source frames per frame, 1/req) that each step allows, with
    its square changing by at most 2 beta between neighbouring steps (beta: the path acceleration the joint
    acceleration limit leaves, unbounded where nothing moves), by a forward and a backward pass: the arm decelerates
    into a slow stretch and accelerates out of it as fast as the limit lets it, and a still stretch costs no time."""
    x = (1.0 / np.asarray(req, float)) ** 2
    b = np.minimum(beta[:-1], beta[1:]) if len(beta) > 1 else beta
    for i in range(1, len(x)):
        x[i] = min(x[i], x[i - 1] + 2.0 * b[i - 1])
    for i in range(len(x) - 2, -1, -1):
        x[i] = min(x[i], x[i + 1] + 2.0 * b[i])
    return 1.0 / np.sqrt(np.maximum(x, 1e-12))


def _integrate(r, T, max_frames=None):
    taus = [0.0]
    while taus[-1] < T - 1:
        i = min(int(taus[-1]), T - 2)
        taus.append(taus[-1] + 1.0 / r[i])
        if max_frames and len(taus) > int(max_frames):
            break
    taus[-1] = min(taus[-1], T - 1.0)
    return np.asarray(taus)


def retime(X, fps, vmax=None, amax=None, keep=None, window_s=0.3, iters=16, max_frames=None, ignore_step=None):
    """Time map that keeps every column of X [T, n] within its velocity limit vmax (units/s, one value or one per
    column) and its smoothed acceleration limit amax (units/s^2; None skips), slowing down only where needed.

    The slow-down r (new frames per source frame) of each source step is at least max(1, v/vmax, sqrt(a/amax)) (a time
    scale r divides a velocity by r and an acceleration by r^2); the path speed 1/r then changes no faster than the
    acceleration limit allows (_profile: a forward and a backward pass, so a still stretch costs no time), and r is
    raised wherever the resampled trajectory still exceeds a limit, measured as joint_acceleration_audit measures it,
    until it does not. Steps inside `keep` (bool [T] on source frames: contact phases) are not slowed for their
    neighbours' sake; only their own motion, or a neighbour that arrives at speed, slows them. A step larger than
    `ignore_step` in some column (a branch jump: a failure to report, not to slow down for) is left as it is. Returns
    {src [T'] fractional source index per new frame, added_frames, ok, vel_ratio, acc_ratio, iterations};
    resample(X, src) gives the retimed trajectory."""
    X = np.asarray(X, float)
    T, n = X.shape if X.ndim == 2 else (len(X), 1)
    X = X.reshape(T, n)
    vm, am = _limit(vmax, n), (None if amax is None else _limit(amax, n))
    ident = {"src": np.arange(T, dtype=float), "added_frames": 0, "ok": True, "vel_ratio": 0.0, "acc_ratio": 0.0, "iterations": 0}
    if T < 3:
        return ident
    keep_step = None
    if keep is not None:
        kp = np.asarray(keep, bool)
        keep_step = kp[:-1] & kp[1:]
    jump = np.zeros(T - 1, bool)
    if ignore_step is not None:
        jump = np.abs(np.diff(X, axis=0)).max(axis=1) > float(ignore_step)
    W = max(5, int(round(window_s * fps)) | 1)

    def ratios(Y):
        st = np.abs(np.diff(Y, axis=0)) * float(fps)
        rv = (st / vm).max(axis=1)
        ra = np.zeros(len(Y))
        if am is not None:
            ra = np.sqrt((np.abs(smoothed_acceleration(Y, fps, window_s)) / am).max(axis=1))
        return rv, ra

    rv0, ra0 = ratios(X)
    req = np.maximum(1.0, np.maximum(rv0, np.maximum(ra0[:-1], ra0[1:])))
    # the path acceleration (source frames per frame^2) the joint acceleration limits leave on each step, at 70 %
    # (the rest is for the path's curvature)
    dX = np.abs(np.diff(X, axis=0))
    beta = np.full(T - 1, np.inf)
    if am is not None:
        with np.errstate(divide="ignore"):
            beta = np.min(np.where(dX > 1e-12, 0.7 * am / float(fps) ** 2 / np.maximum(dX, 1e-300), np.inf), axis=1)
    if jump.any():                                    # nothing is slowed for a jump, nor for the acceleration it causes
        near = np.convolve(jump.astype(float), np.ones(W), mode="same") > 0
        req[near] = 1.0
    src, r = ident["src"], np.ones(T - 1)
    ok, it, rv, ra = True, 0, rv0, ra0
    deferred = {}
    for it in range(1, int(iters) + 1):
        if (req <= 1.0 + 1e-9).all():
            break
        r = _profile(req, beta)
        src = _integrate(r, T, max_frames)
        Y = resample(X, src)
        rv, ra = ratios(Y)
        tol = 1.0 + 1e-3
        bad_v, bad_a = np.where(rv > tol)[0], np.where(ra > tol)[0]
        new_jump = np.zeros(len(rv), bool)
        if jump.any():                                # a new step that is a source jump is not judged
            new_jump = jump[np.minimum(np.floor(src[:-1]).astype(int), T - 2)]
            near = np.convolve(new_jump.astype(float), np.ones(W), mode="same") > 0
            bad_v = bad_v[~new_jump[bad_v]]
            bad_a = bad_a[~near[np.minimum(bad_a, len(near) - 1)]]
        if not len(bad_v) and not len(bad_a):
            ok = True
            break
        ok = False
        grow = np.ones(T - 1)
        for k in bad_v:
            for i in range(int(np.floor(src[k])), min(int(np.ceil(src[k + 1])), T - 1)):
                grow[i] = max(grow[i], 1.05 * rv[k])
        for k in bad_a:
            c = src[min(k, len(src) - 1)]
            cand = list(range(max(0, int(np.floor(c)) - 1), min(int(np.ceil(c)) + 1, T - 1)))
            if keep_step is not None and all(keep_step[i] for i in cand):
                # inside a contact phase: slow the free motion next to it first (the smoothed acceleration there is
                # mostly the neighbour's); the phase itself only when that did not do it
                key = tuple(cand)
                deferred[key] = deferred.get(key, 0) + 1
                if deferred[key] == 1:
                    near_ = [i for i in range(max(0, cand[0] - W), min(T - 1, cand[-1] + W + 1)) if not keep_step[i] and dX[i].max() > 1e-12]
                    cand = near_ or cand
            for i in cand:
                grow[i] = max(grow[i], 1.05 * np.sqrt(ra[k]))
        # raise the request only where a limit is still exceeded (raising it everywhere would widen the slow-down by
        # the spread on every pass)
        up = grow > 1.0
        req[up] = np.maximum(req[up], r[up] * grow[up])
        if max_frames and len(src) > int(max_frames):
            break
    return {"src": src, "added_frames": int(len(src) - T), "ok": bool(ok), "vel_ratio": round(float(np.max(rv, initial=0.0)), 3),
            "acc_ratio": round(float(np.max(ra, initial=0.0)) ** 2, 3), "iterations": int(it)}


def flange_positions(model, Q, parts=None):
    """{side: [T, 3]} world position of each arm's flange (gripper body) over Q."""
    from tool_library.common import episode as E
    parts = parts or E.robot_parts(model)
    d = mujoco.MjData(model)
    out = {}
    for s, pt in parts.items():
        if pt["gripper"] < 0 or not pt["arm_joints"]:
            continue
        P = np.zeros((len(Q), 3))
        for t in range(len(Q)):
            d.qpos[:] = Q[t]
            mujoco.mj_kinematics(model, d)
            P[t] = d.xpos[pt["gripper"]]
        out[s] = P
    return out


def arm_limits(cfg=None):
    c = {**DEFAULTS, **(cfg or {})}
    return np.radians(float(c["vmax_deg_s"])), float(c["amax_rad_s2"]), float(c["flange_amax_m_s2"])


def retime_arms(model, Q, fps, vmax_rad_s=None, amax_rad_s2=None, flange_amax_m_s2=None, keep=None, sides=None, parts=None,
                window_s=0.3, max_frames=None, ignore_step=None):
    """retime() on a full qpos trajectory Q [T, nq] of a bench scene: the arm joints of `sides` (default both) against
    vmax (rad/s, one value or {side: value}; default 90 deg/s) and amax (default 12 rad/s^2), each flange's linear
    acceleration against flange_amax (default 4.8 m/s^2; 0 skips). Returns {qpos [T', nq] (every column resampled,
    quaternions unit), src [T'] (fractional source frame of each new frame, to carry labels, phases or object tracks
    over), added_s, ok, vel_ratio, acc_ratio}. Stage 4 can pass its own joint-speed bound, e.g. vmax={side: 1.5 x the
    original peak + 0.5}, and keep= the contact frames."""
    from tool_library.common import episode as E
    parts = parts or E.robot_parts(model)
    v0, a0, f0 = arm_limits()
    vmax = v0 if vmax_rad_s is None else vmax_rad_s
    amax = a0 if amax_rad_s2 is None else amax_rad_s2
    fam = f0 if flange_amax_m_s2 is None else flange_amax_m_s2
    Q = np.asarray(Q, float)
    cols, vm, am = [], [], []
    for s, pt in parts.items():
        if (sides and s not in sides) or pt["gripper"] < 0 or not pt["arm_joints"]:
            continue
        v_s = vmax.get(s, v0) if isinstance(vmax, dict) else vmax      # per station (Stage 4 bounds each chain by its original)
        cols.append(Q[:, pt["arm_qadr"]]); vm += [float(v_s)] * len(pt["arm_qadr"]); am += [amax] * len(pt["arm_qadr"])
    if not cols:
        return {"qpos": Q, "src": np.arange(len(Q), dtype=float), "added_s": 0.0, "ok": True}
    if fam:
        for s, P in flange_positions(model, Q, parts).items():
            if sides and s not in sides:
                continue
            cols.append(P); vm += [np.inf] * 3; am += [fam] * 3
    r = retime(np.hstack(cols), fps, vmax=vm, amax=am, keep=keep, window_s=window_s, max_frames=max_frames, ignore_step=ignore_step)
    out = dict(r)
    out["qpos"] = resample(Q, r["src"], _quat_cols(model)) if r["added_frames"] else Q.copy()
    out["added_s"] = round(r["added_frames"] / float(fps), 3)
    return out


# ------------------------------------------------------------------------------------------------- path search
def _edge_points(a, b, res):
    n = int(np.ceil(np.abs(b - a).max() / res))
    if n <= 1:
        return []
    # bisection order: the middle first, so a blocked edge is found after few checks
    out, seen = [], set()
    lvl = [(0.0, 1.0)]
    while lvl and len(out) < n - 1:
        nxt = []
        for lo, hi in lvl:
            mid = 0.5 * (lo + hi)
            k = int(round(mid * n))
            if 0 < k < n and k not in seen:
                seen.add(k); out.append(k / n)
            if hi - lo > 1.0 / n:
                nxt += [(lo, mid), (mid, hi)]
        lvl = nxt
    for k in range(1, n):
        if k not in seen:
            out.append(k / n)
    return [a + u * (b - a) for u in out]


def rrt_connect(valid, q0, q1, lo, hi, rng, step=0.25, res=0.02, iters=3000, deadline=None):
    """Bidirectional RRT (RRT-Connect) in a joint box. valid(q) -> bool. Returns a waypoint list from q0 to q1, or None."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)

    def edge(a, b):
        return all(valid(p) for p in _edge_points(a, b, res))

    if edge(q0, q1):
        return [q0, q1]
    trees = [[q0], [q1]]
    parents = [[-1], [-1]]

    def nearest(tree, q):
        A = np.asarray(tree)
        return int(np.argmin(np.linalg.norm(A - q, axis=1)))

    def steer(a, b):
        d = b - a
        L = float(np.linalg.norm(d))
        return b.copy() if L <= step else a + d * (step / L)

    def extend(k, q):
        i = nearest(trees[k], q)
        qn = steer(trees[k][i], q)
        if valid(qn) and edge(trees[k][i], qn):
            trees[k].append(qn); parents[k].append(i)
            return len(trees[k]) - 1, np.allclose(qn, q)
        return None, False

    def trace(k, i):
        out = []
        while i >= 0:
            out.append(trees[k][i]); i = parents[k][i]
        return out

    for it in range(int(iters)):
        if deadline is not None and time.time() > deadline:
            break
        a, b = (0, 1) if it % 2 == 0 else (1, 0)
        qr = rng.uniform(lo, hi)
        ia, _ = extend(a, qr)
        if ia is None:
            continue
        target = trees[a][ia]
        while True:                                   # connect: the other tree steps toward the new node until blocked
            ib, reached = extend(b, target)
            if ib is None:
                break
            if reached:
                pa, pb = trace(a, ia), trace(b, ib)
                path = pa[::-1] + pb[1:] if a == 0 else pb[::-1] + pa[1:]
                return path
    return None


def shortcut(path, valid, rng, res=0.02, iters=150):
    """Random-point shortcutting (two points on the path joined straight when the segment is free), then a greedy pass
    that joins each kept waypoint to the farthest one it sees."""
    P = [np.asarray(p, float) for p in path]

    def edge(a, b):
        return all(valid(p) for p in _edge_points(a, b, res))

    for _ in range(int(iters)):
        if len(P) < 3:
            break
        seg = np.array([np.linalg.norm(P[i + 1] - P[i]) for i in range(len(P) - 1)])
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        if cum[-1] <= 1e-9:
            break
        u1, u2 = np.sort(rng.uniform(0.0, cum[-1], 2))
        i, j = int(np.searchsorted(cum, u1, "right") - 1), int(np.searchsorted(cum, u2, "right") - 1)
        i, j = min(i, len(P) - 2), min(j, len(P) - 2)
        if i == j:
            continue
        a = P[i] + (u1 - cum[i]) / max(seg[i], 1e-12) * (P[i + 1] - P[i])
        b = P[j] + (u2 - cum[j]) / max(seg[j], 1e-12) * (P[j + 1] - P[j])
        if edge(a, b):
            P = P[:i + 1] + [a, b] + P[j + 1:]
    out, i = [P[0]], 0
    while i < len(P) - 1:
        j = len(P) - 1
        while j > i + 1 and not edge(P[i], P[j]):
            j -= 1
        out.append(P[j]); i = j
    return out


def densify(path, step=0.01):
    """Waypoints -> samples at most `step` apart (max joint change)."""
    out = [np.asarray(path[0], float)]
    for a, b in zip(path[:-1], path[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(1, int(np.ceil(np.abs(b - a).max() / step)))
        out += [a + (k / n) * (b - a) for k in range(1, n + 1)]
    return np.asarray(out)


def smooth_corners(D, valid, max_window=31):
    """The densified path with its corners rounded by a moving average (ends pinned), as wide as stays valid."""
    w = int(max_window) | 1
    while w >= 3:
        k = w // 2
        S = D.copy()
        for _ in range(2):
            pad = np.vstack([np.repeat(S[:1], k, 0), S, np.repeat(S[-1:], k, 0)])
            S = np.array([pad[i:i + w].mean(axis=0) for i in range(len(S))])
            S[0], S[-1] = D[0], D[-1]
        if all(valid(q) for q in S):
            return S, w
        w = (w // 2) | 1 if w > 3 else 1
    return D, 1


def quintic_frames(D, n):
    """n frames along the dense path D (from just after D[0] to D[-1]) with a minimum-jerk profile on arc length (the
    leg starts and ends at rest)."""
    seg = np.linalg.norm(np.diff(D, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    L = cum[-1]
    tau = np.arange(1, int(n) + 1) / float(n)
    s = L * (10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5)
    out = np.zeros((len(s), D.shape[1]))
    for j in range(D.shape[1]):
        out[:, j] = np.interp(s, cum, D[:, j])
    return out


def path_duration_s(D, vmax, amax):
    """Shortest minimum-jerk duration of the dense path D under a joint speed and acceleration limit (curvature
    neglected; retime enforces the exact limits afterwards)."""
    if len(D) < 2:
        return 0.0
    Lm = float(np.abs(np.diff(D, axis=0)).max(axis=1).sum())
    return max(1.875 * Lm / vmax, np.sqrt(5.77 * Lm / amax))


# ------------------------------------------------------------------------------------------------- the planner
class ArmPlanner:
    """Collision queries and paths for the bench's two arm+hand stations (see the module docstring)."""

    def __init__(self, scene_xml, cfg=None):
        from tool_library.common import episode as E
        from tool_library.common.handgeom import add_hand_hulls
        self.cfg = {**DEFAULTS, **(cfg or {})}
        spec = mujoco.MjSpec.from_file(str(scene_xml))
        add_hand_hulls(spec)
        plane = mujoco.mjtGeom.mjGEOM_PLANE
        for g in spec.geoms:
            body = g.parent.name if g.parent is not None else ""
            if body.startswith("robot_"):
                is_arm = "_arm_" in body
                on = (is_arm and int(g.group) <= 2 and g.type != plane and g.rgba[3] > 0) or (not is_arm and "_collision_hand_" in (g.name or ""))
                g.contype, g.conaffinity = (1, 1) if on else (0, 0)
            g.margin, g.gap = 0.0, 0.0
        for p in list(spec.pairs):
            spec.delete(p)
        for e in list(spec.excludes):
            spec.delete(e)
        self.model = m = spec.compile()
        m.opt.disableflags = int(m.opt.disableflags) & ~int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        self.data = mujoco.MjData(m)
        self.parts = E.robot_parts(m)
        self.quat_cols = _quat_cols(m)
        col = np.array([g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]], int)
        self.col = col
        self.role0 = {}                                # geom -> (side or None, kind): kind arm | armmount | hand | handmount | world
        grip = {s: pt["gripper"] for s, pt in self.parts.items()}
        for g in col:
            b = int(m.geom_bodyid[g])
            n = m.body(b).name or ""
            if n.startswith("robot_"):
                side = "left" if n.startswith("robot_left") else "right"
                if "_arm_" in n:
                    mount = b == grip.get(side) or any(int(m.body_parentid[c]) == b for c in [grip.get(side, -1)] if c >= 0)
                    self.role0[int(g)] = (side, "armmount" if mount else "arm")
                else:
                    mount = "_base_" in n or n.endswith("_hand_C_MC") or n.endswith("_palm")
                    self.role0[int(g)] = (side, "handmount" if mount else "hand")
            else:
                self.role0[int(g)] = (None, "world")
        # an arm's base and first link (moved by at most its first joint, about the vertical) keep their distance to the
        # table whatever the arm does (the right station's link 1 sat 18 mm over the shifted bench): for them only
        # penetration counts, not the margin
        self.base_geom = np.zeros(m.ngeom, bool)
        for g in col:
            side, kind = self.role0[int(g)]
            if side is not None and kind == "arm":
                b, n_j = int(m.geom_bodyid[g]), 0
                while b > 0:
                    n_j += int(m.body_jntnum[b])
                    b = int(m.body_parentid[b])
                self.base_geom[g] = n_j <= 1
        # static scene bodies (the table, the world) versus objects: an idle hand near an object is engaged
        self.static_geoms = {int(g) for g in col if self.role0[int(g)][1] == "world"
                             and (int(m.geom_bodyid[g]) == 0 or "table" in (m.body(int(m.geom_bodyid[g])).name or ""))}
        self.static_geom = np.zeros(m.ngeom, bool)
        self.static_geom[list(self.static_geoms)] = True
        self.free_body = {}                           # free-joint qpos address -> geoms of the subtree it moves
        for j in range(m.njnt):
            if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
                b = int(m.jnt_bodyid[j])
                if (m.body(b).name or "").startswith("robot_"):
                    continue
                sub = set(E.subtree_bodies(m, b))
                self.free_body[int(m.jnt_qposadr[j])] = [int(g) for g in col if int(m.geom_bodyid[g]) in sub]
        from tool_library.common import armhand as AH
        self.arm = {s: AH.ArmSide(m, s, parts=self.parts) for s, pt in self.parts.items() if pt["gripper"] >= 0 and pt["arm_joints"]}
        self.margin = 0.0
        self._key = None
        self._roots = {}
        self.n_checks = 0

    # ---------------------------------------------------------------- collision filter
    def setup(self, moving, held=(), obstacles="all", margin=0.02, own=True, extra=(), check=("arm", "hand")):
        """Which geoms are checked against which for the next queries. `moving`: the side whose arm and hand move;
        `held`: geoms carried by that hand (they move with it). `obstacles`: 'all' (the table, the objects, the other
        station), 'robot' (the other station only), 'world' (the table and objects only), 'objects' (objects only) or
        'none';
        `extra`: more obstacle geoms (what the other hand carries). `own`: also check the moving arm against its own
        hand and what it carries. `check`: which moving parts meet the obstacles ('arm', 'hand'; the arm's links alone in
        a contact phase, where the hand touches the part on purpose; the hand alone to ask what it touches). Obstacles
        carry the contact margin."""
        key = (moving, tuple(sorted(int(g) for g in held)), obstacles, round(float(margin), 6), own, tuple(sorted(int(g) for g in extra)), tuple(check))
        if key == self._key:
            return
        m = self.model
        ct = np.zeros(m.ngeom, np.int32)
        held, extra = {int(g) for g in held}, {int(g) for g in extra}
        for g in self.col:
            side, kind = self.role0[int(g)]
            if int(g) in held:
                r = _H if own else _A
            elif side == moving:
                r = {"arm": _A, "armmount": _AM, "hand": _H, "handmount": _HM}[kind] if own else _A
            elif int(g) in extra:
                r = _O
            else:
                is_robot = side is not None
                if obstacles == "all":
                    r = _O
                elif obstacles == "robot":
                    r = _O if is_robot else 0
                elif obstacles == "world":
                    r = _O if not is_robot else 0
                elif obstacles == "objects":          # no table, no robot
                    r = _O if (not is_robot and int(g) not in self.static_geoms) else 0
                else:                                 # 'none': only the `extra` geoms
                    r = 0
            ct[g] = r
        aff = dict(_AFFINITY)
        if "hand" not in check:                       # the moving hand does not meet the obstacles
            aff[_H], aff[_HM], aff[_O] = aff[_H] & ~_O, aff[_HM] & ~_O, aff[_O] & ~(_H | _HM)
        if "arm" not in check:                        # the moving arm does not meet the obstacles
            aff[_A], aff[_AM], aff[_O] = aff[_A] & ~_O, aff[_AM] & ~_O, aff[_O] & ~(_A | _AM)
        ca = np.array([aff.get(int(r), 0) if r else 0 for r in ct], np.int32)
        if not own:
            ca[ct == _A] = _O
        m.geom_contype[:] = ct
        m.geom_conaffinity[:] = ca
        m.geom_margin[:] = np.where(ct == _O, float(margin), 0.0)
        self.margin = float(margin)
        self._key = key

    def clearance(self, q):
        """(smallest signed distance m capped at the margin, (geom1, geom2) or None) under the current setup()."""
        from tool_library.common import hand as H
        m, d = self.model, self.data
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        best, pair = self.margin, None
        n = int(d.ncon)
        if n:
            dist = np.array(d.contact.dist[:n], float)
            for i in np.argsort(dist):
                v = float(dist[i])
                if v >= best:
                    break
                g1, g2 = int(d.contact.geom1[i]), int(d.contact.geom2[i])
                if v >= 0.0 and ((self.base_geom[g1] and self.static_geom[g2]) or (self.base_geom[g2] and self.static_geom[g1])):
                    continue                          # a base link near the table: only penetration counts
                if v == 0.0:                          # a false zero of the convex collider: re-measured
                    v = H.geom_distance(m, d, g1, g2, max(self.margin, 1e-4) + 0.01)
                    if v >= best:
                        continue
                best, pair = v, (g1, g2)
                break
        return best, pair

    def pair_names(self, pair):
        if not pair:
            return None
        m = self.model
        return [m.geom(g).name or m.body(int(m.geom_bodyid[g])).name for g in pair]

    # ---------------------------------------------------------------- configurations
    def _root(self, side):
        """World frame of the hand's wrist joints (constant: the wrist chain hangs from the world)."""
        if side not in self._roots:
            from tool_library.common import plan as P
            d = mujoco.MjData(self.model)
            mujoco.mj_kinematics(self.model, d)
            rig = self.arm[side].pt["rig"]
            p0, R0 = P.root_frame(self.model, d, rig)
            self._roots[side] = (np.asarray(p0, float), np.asarray(R0, float), [rig.wrist_qpos[k] for k in ("x", "y", "z", "rx", "ry", "rz")])
        return self._roots[side]

    def place(self, q_ctx, side, qa, held=()):
        """Full qpos: q_ctx with `side`'s arm at qa, its hand root put on the flange (exact, analytic) and the carried
        free bodies `held` [(qpos adr, p_rel, R_rel)] moved with the hand. The wrist angles are one exact intrinsic-XYZ
        decomposition (a collision query needs the pose, not the branch nearest the reference)."""
        m, d = self.model, self.data
        S = self.arm[side]
        q = np.array(q_ctx, float)
        q[S.arm.qadr] = qa
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        Rg, pg = d.xmat[S.pt["gripper"]].reshape(3, 3), d.xpos[S.pt["gripper"]]
        Ry = Rg @ S.R_m.T
        py = pg - Ry @ S.p_m
        p0, R0, wa = self._root(side)
        q[wa[:3]] = R0.T @ (py - p0)
        q[wa[3:]] = _euler_xyz(R0.T @ Ry)
        for adr, p_rel, R_rel in held:
            q[adr:adr + 3] = py + Ry @ p_rel
            q[adr + 3:adr + 7] = TJ.mat2quat(Ry @ R_rel)
        return q

    def hand_pose(self, q, side):
        d = self.data
        d.qpos[:] = q
        mujoco.mj_kinematics(self.model, d)
        b = self.arm[side].pt["ee_body"]
        return d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()

    def carried(self, q_start, q_end, side, tol_m=0.001):
        """Free bodies the plan moves between q_start and q_end and the hand of `side` carries (it touches them at the
        start, or is the nearest hand): [(qpos adr, p_rel, R_rel)] relative to that hand at q_start, and their geoms."""
        out, geoms = [], []
        moved = [a for a in self.free_body if np.linalg.norm(np.asarray(q_end)[a:a + 3] - np.asarray(q_start)[a:a + 3]) > tol_m
                 or abs(abs(float(np.dot(q_end[a + 3:a + 7], q_start[a + 3:a + 7]))) - 1.0) > 1e-5]
        if not moved:
            return out, geoms
        hands = {s: self.hand_pose(q_start, s) for s in self.arm}
        for a in moved:
            p = np.asarray(q_start[a:a + 3], float)
            near = min(hands, key=lambda s: np.linalg.norm(hands[s][0] - p))
            if near != side and not self.touches(q_start, side, self.free_body[a]):
                continue
            py, Ry = hands[side]
            R = TJ.quat2mat(np.asarray(q_start[a + 3:a + 7], float))
            out.append((a, Ry.T @ (p - py), Ry.T @ R))
            geoms += self.free_body[a]
        return out, geoms

    def touches(self, q, side, geoms, within=0.005):
        """True when the hand of `side` is within `within` of any of `geoms` at q (it holds or presses that part)."""
        if not len(geoms):
            return False
        self.setup(side, obstacles="none", extra=geoms, margin=within + 0.002, check=("hand",))
        return self.clearance(q)[0] < within

    # ---------------------------------------------------------------- legs
    def _checker(self, side, contexts, qa0, qa1, held, margin, c_ends=None, q_ends=None):
        """valid(qa): every context (full qpos of the rest of the scene) clear of the moving side by the required
        clearance (the margin, shrunk near the two ends to what they have)."""
        slope = float(self.cfg["slope_m_rad"])
        if c_ends is None:
            c_ends = (self.clearance(self.place(contexts[0], side, qa0, held))[0],
                      self.clearance(self.place(contexts[-1], side, qa1, held))[0])
        c0, c1 = c_ends
        a0, a1 = (qa0, qa1) if q_ends is None else q_ends
        lo, hi = self.arm[side].arm.lo, self.arm[side].arm.hi

        def req(qa):
            return min(margin, c0 - 1e-3 + slope * float(np.abs(qa - a0).max()), c1 - 1e-3 + slope * float(np.abs(qa - a1).max()))

        def valid(qa):
            if (qa < lo - 1e-9).any() or (qa > hi + 1e-9).any():
                return False
            need = req(qa)
            for ctx in contexts:
                self.n_checks += 1
                if self.clearance(self.place(ctx, side, qa, held))[0] < need:
                    return False
            return True
        return valid, req, (c0, c1)

    def leg_violation(self, side, A, contexts, held=(), held_geoms=(), level=0.0):
        """How a timed arm path A [n, 6] (row 0 the configuration it starts from) fares against per-frame contexts [n]
        (each frame against the scene at its own time, sub-steps between frames): the worst of clearance minus the
        requirement, which is `level` shrunk near the two ends to the clearance those ends have (the plan may put a
        hand near a part on purpose there). Returns (worst m, negative when violated; frame; pair names; min
        clearance m)."""
        res = float(self.cfg["res_rad"])
        slope = float(self.cfg["slope_m_rad"])
        self.setup(side, held=held_geoms, margin=max(float(level), 0.0) + 0.02)
        A = np.asarray(A, float)
        c0 = self.clearance(self.place(contexts[0], side, A[0], held))[0]
        c1 = self.clearance(self.place(contexts[-1], side, A[-1], held))[0]
        worst, at, pair, cmin = np.inf, None, None, np.inf
        for k in range(1, len(A)):
            for p in _edge_points(A[k - 1], A[k], res) + [A[k]]:
                c, pr = self.clearance(self.place(contexts[k], side, p, held))
                need = min(float(level), c0 - 1e-3 + slope * float(np.abs(p - A[0]).max()), c1 - 1e-3 + slope * float(np.abs(p - A[-1]).max()))
                cmin = min(cmin, c)
                if c - need < worst:
                    worst, at, pair = c - need, k, pr
        if at is None:
            return 0.0, None, None, c0
        return float(worst), at, self.pair_names(pair), float(cmin)

    def plan_leg(self, side, qa0, qa1, contexts, held=(), held_geoms=(), margins=None, seed=None, time_s=None):
        """A collision-free arm path from qa0 to qa1 against every context (full qpos of the rest of the scene: the
        other station, the objects), for the largest margin of `margins` that works. Returns {ok, path (dense, from
        qa0 to qa1), waypoints, margin, smoothing, checks, seconds, reason}."""
        t0 = time.time()
        self.n_checks = 0
        margins = list(margins or self.cfg["margins_m"])
        rng = np.random.default_rng(self.cfg["seed"] if seed is None else seed)
        S = self.arm[side]
        qa0, qa1 = np.asarray(qa0, float), np.asarray(qa1, float)
        res = float(self.cfg["res_rad"])
        reason = None
        for margin in margins:
            self.setup(side, held=held_geoms, margin=margin)
            valid, req, ends = self._checker(side, contexts, qa0, qa1, held, margin)
            budget = float(time_s or self.cfg["time_s"])
            path = rrt_connect(valid, qa0, qa1, S.arm.lo, S.arm.hi, rng, step=float(self.cfg["step_rad"]), res=res,
                               iters=int(self.cfg["iters"]), deadline=time.time() + budget)
            if path is None:
                reason = f"no path at {1000 * margin:.0f} mm clearance within {budget:.0f} s (ends clear by {1000 * ends[0]:.1f} / {1000 * ends[1]:.1f} mm)"
                continue
            path = shortcut(path, valid, rng, res=res, iters=int(self.cfg["shortcut_iters"]))
            D = densify(path, step=res / 2)
            # shortcut segments graze the margin at the corners: rounding them may give up to 40 % of it (the timed leg
            # is verified at half the margin)
            valid_s = self._checker(side, contexts, qa0, qa1, held, 0.6 * margin, c_ends=ends)[0]
            D, w = smooth_corners(D, valid_s)
            return {"ok": True, "path": D, "waypoints": [p.tolist() for p in path], "margin": margin, "smoothing": int(w),
                    "checks": int(self.n_checks), "seconds": round(time.time() - t0, 2), "ends_clear_mm": [round(1000 * e, 1) for e in ends]}
        return {"ok": False, "reason": reason, "seconds": round(time.time() - t0, 2)}

    def time_along(self, side, D, contexts, held=(), held_geoms=(), level=0.0, vmax_step=None):
        """Timing of a geometric path D [N, n] (dense, from the leg's start to its target) against a moving scene
        (path-velocity decomposition): contexts[k] is the scene at frame k (the other station where it is then);
        returns the sample index s[k] the arm is at on each frame (s[0] = 0, s[-1] = N-1, non-decreasing, at most
        vmax_step samples per frame), every (frame, sample) clear by `level` (shrunk near the path's ends), or None.
        The search is a breadth-first sweep over frames: the arm may wait anywhere the other station leaves room."""
        D = np.asarray(D, float)
        N, K = len(D), len(contexts)
        if K < 2:
            return None
        step = int(vmax_step or max(1, np.floor(np.radians(self.cfg["vmax_deg_s"]) / 30.0 / max(np.abs(np.diff(D, axis=0)).max(), 1e-9))))
        slope = float(self.cfg["slope_m_rad"])
        self.setup(side, held=held_geoms, margin=max(float(level), 0.0) + 0.02)
        c0 = self.clearance(self.place(contexts[0], side, D[0], held))[0]
        c1 = self.clearance(self.place(contexts[-1], side, D[-1], held))[0]
        need = np.minimum(float(level), np.minimum(c0 - 1e-3 + slope * np.abs(D - D[0]).max(axis=1), c1 - 1e-3 + slope * np.abs(D - D[-1]).max(axis=1)))
        memo = {}

        def ok(k, j):
            if (k, j) not in memo:
                memo[(k, j)] = self.clearance(self.place(contexts[k], side, D[j], held))[0] >= need[j]
            return memo[(k, j)]
        reach = [np.zeros(N, bool) for _ in range(K)]
        back = [np.full(N, -1) for _ in range(K)]
        reach[0][0] = True                            # the start is where the arm is
        for k in range(1, K):
            idx = np.nonzero(reach[k - 1])[0]
            if not len(idx):
                return None
            lo_ = int(idx.min())
            hi_ = int(min(N - 1, idx.max() + step))
            remain = K - 1 - k                        # the target must still be reachable in the frames left
            for j in range(hi_, lo_ - 1, -1):
                if j + remain * step < N - 1:
                    break
                prev = idx[(idx <= j) & (idx >= j - step)]
                if not len(prev) or not ok(k, j):
                    continue
                reach[k][j] = True
                back[k][j] = int(prev.max())
        if not reach[K - 1][N - 1]:
            return None
        s = [N - 1]
        for k in range(K - 1, 0, -1):
            s.append(int(back[k][s[-1]]))
        return np.asarray(s[::-1])

    # ---------------------------------------------------------------- parking
    def engaged(self, q, side, within=None):
        """True when the hand of `side` touches or nearly touches an object (it holds or steadies it): such a hand is
        not idle, whatever its plan says."""
        within = float(self.cfg["engage_m"] if within is None else within)
        self.setup(side, obstacles="objects", margin=within + 0.002, check=("hand",))
        c, pair = self.clearance(q)
        if pair is None or c >= within:
            return False, None
        names = self.pair_names(pair)
        return True, names

    def side_gap(self, Qa, idle, active, held_geoms=(), margin=0.05, stride=2):
        """Smallest distance between the active station (arm, hand, carried geoms) and the idle station over the frames
        of Qa (full qpos, both arms set). Returns (min m, frame, pair names)."""
        self.setup(active, held=held_geoms, obstacles="robot", margin=margin, own=False)
        worst, at, pair = margin, None, None
        for t in range(0, len(Qa), max(1, int(stride))):
            c, pr = self.clearance(Qa[t])
            if c < worst:
                worst, at, pair = c, t, pr
        return worst, at, self.pair_names(pair)

    def park(self, side, q_now, active_frames, active_side, active_held=(), n_candidates=400, margin=None):
        """A clearance configuration for the idle arm of `side`: within its joint limits, clear of the table and the
        objects by the first planning margin, and at least park_margin_m from the active station at every frame of
        active_frames (full qpos, the active arm set), the one nearest the current arm among those (joint distance,
        then clearance). Candidates: the current configuration perturbed locally and the arm's seed library.
        Returns {ok, qa, joint_dist_rad, clear_world_mm, clear_active_mm, reason}."""
        S = self.arm[side]
        rng = np.random.default_rng(self.cfg["seed"] + 7)
        qa_now = np.asarray(q_now, float)[S.arm.qadr]
        pm = float(self.cfg["park_margin_m"])
        wm = float(self.cfg["margins_m"][0] if margin is None else margin)
        lib, _, _ = S.seed_library()
        local = [np.clip(qa_now + rng.normal(0.0, sd, len(qa_now)), S.arm.lo, S.arm.hi) for sd in (0.25, 0.5, 0.9) for _ in range(150)]
        C = np.vstack([np.asarray(local), lib])
        dist = np.linalg.norm(C - qa_now, axis=1)
        order = np.argsort(dist)[:int(n_candidates)]
        frames = list(active_frames)[::max(1, len(active_frames) // 30)] + [active_frames[-1]]
        best, bonus = None, 2.0 * (pm + 0.03)
        for i in order:
            if best is not None and float(dist[i]) - bonus >= best[0]:
                break                                 # candidates come nearest first: none further on can score better
            qa = C[i]
            self.setup(side, obstacles="world", margin=wm)
            cw, _ = self.clearance(self.place(q_now, side, qa))
            if cw < wm:
                continue
            # against the active station at every frame (its arm, hand and carried parts): the idle side checked as moving
            self.setup(side, held=(), obstacles="robot", margin=pm + 0.03, own=False, extra=active_held)
            ca = pm + 0.03
            for f in frames:
                ca = min(ca, self.clearance(self.place(f, side, qa))[0])
                if ca < pm:
                    break
            if ca < pm:
                continue
            score = float(dist[i]) - 2.0 * min(ca, pm + 0.03)
            if best is None or score < best[0]:
                best = (score, qa, float(dist[i]), cw, ca)
        if best is None:
            return {"ok": False, "reason": f"no arm configuration of {len(order)} tried is clear of the table/objects by {1000 * wm:.0f} mm "
                                            f"and of the active station by {1000 * pm:.0f} mm"}
        return {"ok": True, "qa": best[1], "joint_dist_rad": round(best[2], 3), "clear_world_mm": round(1000 * best[3], 1),
                "clear_active_mm": round(1000 * best[4], 1)}


# ------------------------------------------------------------------------------------------------- couple's planning step
def _moving_bodies(planner, Qref):
    """Free bodies the hand reference moves (carried by a hand), with the frame they start moving."""
    out = {}
    for a in planner.free_body:
        d = np.linalg.norm(Qref[:, a:a + 3] - Qref[0, a:a + 3], axis=1)
        dq = np.abs(np.abs(Qref[:, a + 3:a + 7] @ Qref[0, a + 3:a + 7]) - 1.0)
        mv = np.where((d > 1e-3) | (dq > 1e-5))[0]
        if len(mv):
            out[a] = int(mv[0])
    return out


def _timed_leg(planner, side, D, per, held, held_geoms, fps, vmax, levels):
    """A geometric path D timed over the leg's frames against the moving scene `per` (per[0] the frame before the
    leg): planner.time_along at the largest clearance level that has a timing, the timing smoothed (it stays
    non-decreasing). Returns (A [len(per) - 1, n], level) or (None, None)."""
    spacing = max(float(np.abs(np.diff(D, axis=0)).max()), 1e-9) if len(D) > 1 else 1.0
    step = max(1, int(np.floor(vmax / float(fps) / spacing)))
    for lvl in levels:
        s = planner.time_along(side, D, per, held, held_geoms, level=lvl, vmax_step=step)
        if s is None:
            continue
        k = 3
        sm = np.convolve(np.pad(s.astype(float), k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
        sm[0], sm[-1] = 0.0, len(D) - 1.0
        out = np.zeros((len(per) - 1, D.shape[1]))
        for j in range(D.shape[1]):
            out[:, j] = np.interp(sm[1:], np.arange(len(D)), D[:, j])
        return out, float(lvl)
    return None, None


def plan_arms(model, planner, Qref, arms, legs, starts, labels, fps, park=True, cfg=None):
    """The planning step of armhand.couple (only when it is given a planner).

    arms {side: A [T, n]}: couple's fill (IK along the hand reference, straight joint-space legs); legs {side: {first
    frame: target frame}} its free-space legs; starts {side: arm configuration before frame 0}; Qref the hand reference.
      1. An idle station (its hand reference does not move and its hand touches no object) that comes within
         park_margin_m of the active station (arm, hand, carried part) anywhere in the entry is parked: moved to a
         clearance configuration (ArmPlanner.park) along a planned path, concurrently with the active hand when the
         timed path stays clear of it, else before the active hand starts (frames are prepended).
      2. Every free-space leg whose straight joint path comes into anything (the table, an object, the other station
         where it is at that time, its own hand) is replaced by a planned path with the same ends and the same frames
         (retime slows it afterwards when it is longer); a leg that cannot be planned keeps its straight path and is
         reported. While the other station moves (a two-hand entry) it is an obstacle at its own time: a leg first
         tries its straight path timed to wait where the other passes (time_along), then a planned path timed the same
         way; every leg is checked once more against both stations' final paths.
    Returns {Qref, arms, labels, legs, prefix (frames prepended), report}."""
    from tool_library.common import episode as E
    c = {**planner.cfg, **(cfg or {})}
    rep = {"planned_legs": [], "failed_legs": [], "parked": [], "checked_legs": 0}
    if planner.model.nq != model.nq:
        rep["note"] = f"the planner's scene has nq={planner.model.nq}, the execution model {model.nq}: not planned"
        return {"Qref": Qref, "arms": arms, "labels": labels, "legs": legs, "prefix": 0, "report": rep}
    t_start = time.time()
    parts = E.robot_parts(model)
    Qref = np.asarray(Qref, float).copy()
    labels = list(labels)
    arms = {s: np.asarray(A, float).copy() for s, A in arms.items()}
    legs = {s: dict(v) for s, v in legs.items()}
    sides = [s for s in arms if s in planner.arm]
    T = len(Qref)
    vmax, amax, _ = arm_limits(c)
    active = []
    for s in sides:
        hq = E.hand_qadr(model, parts[s]["rig"])
        if legs.get(s) or (T > 1 and np.abs(np.diff(Qref[:, hq], axis=0)).max() > 1e-9):
            active.append(s)
    idle = [s for s in sides if s not in active]
    cache = {}

    def ctx(f, me):
        """Qref[f] with every other station's arm at its path and its hand on the flange."""
        key = (int(f), me)
        if key not in cache:
            q = Qref[int(f)].copy()
            for s in sides:
                if s != me:
                    q = planner.place(q, s, arms[s][int(f)])
            cache[key] = q
        return cache[key]

    def full(f, me):
        return planner.place(ctx(f, me), me, arms[me][int(f)])

    # what each hand carries during the entry (free bodies the reference moves, nearest hand when they start moving)
    carried_geoms = {s: [] for s in sides}
    for a, f0 in _moving_bodies(planner, Qref).items():
        p = Qref[f0, a:a + 3]
        q = full(f0, sides[0]) if sides else Qref[f0]
        near = min(sides, key=lambda s: np.linalg.norm(planner.hand_pose(q, s)[0] - p)) if sides else None
        if near is not None:
            carried_geoms[near] += planner.free_body[a]
    prefix = 0
    # ---- 1. an idle station out of the active one's way
    if park and active and idle and T > 1:
        a = active[0]
        for i in idle:
            q0 = full(0, i)
            eng, what = planner.engaged(q0, i)
            if eng:
                rep["parked"].append({"side": i, "parked": False, "why": f"its hand touches {'/'.join(what)}: it holds or steadies something"})
                continue
            frames_a = [full(f, a) for f in range(T)]
            pm = float(c["park_margin_m"])
            gap, at, pair = planner.side_gap(frames_a, idle=i, active=a, held_geoms=carried_geoms[a], margin=pm + 0.03)
            if gap >= pm:
                continue
            tgt = planner.park(i, q0, frames_a, a, active_held=carried_geoms[a])
            rec = {"side": i, "gap_mm": round(1000 * gap, 1), "at_frame": at, "at_label": str(labels[at]) if at is not None else None,
                   "between": pair}
            if not tgt["ok"]:
                rep["parked"].append(dict(rec, parked=False, why=tgt["reason"]))
                continue
            qa0, qa1 = arms[i][0].copy(), tgt["qa"]
            done = None
            n_est = max(int(round(0.8 * fps)), int(np.ceil(1.5 * fps * path_duration_s(np.vstack([qa0, qa1]), vmax, amax))))
            for mode in ("concurrent", "before"):
                if mode == "concurrent":
                    # the active station where it is while the arm parks (about the time the way takes)
                    win = sorted({int(x) for x in np.linspace(0, min(T - 1, n_est), 6)})
                    contexts = [frames_a[f] for f in win]
                else:
                    contexts = [frames_a[0]]
                pr = planner.plan_leg(i, qa0, qa1, contexts)
                if not pr["ok"]:
                    continue
                n = max(int(round(0.8 * fps)), int(np.ceil(1.15 * fps * path_duration_s(pr["path"], vmax, amax))))
                Ap = quintic_frames(pr["path"], n)
                if mode == "concurrent":
                    if n >= T:
                        continue
                    per = [frames_a[0]] + [frames_a[f] for f in range(n)]
                    v, fr, pn, cm = planner.leg_violation(i, np.vstack([qa0, Ap]), per, level=0.5 * pr["margin"])
                    if v < 0:
                        continue
                    arms[i][:n] = Ap
                    arms[i][n:] = qa1
                else:
                    per = [frames_a[0]] * (n + 1)
                    v, fr, pn, cm = planner.leg_violation(i, np.vstack([qa0, Ap]), per, level=0.5 * pr["margin"])
                    if v < 0:
                        continue
                    Qref = np.vstack([np.repeat(Qref[:1], n, axis=0), Qref])
                    labels = [labels[0]] * n + labels
                    for s in sides:
                        arms[s] = np.vstack([Ap, np.repeat(qa1[None], T, axis=0)]) if s == i else np.vstack([np.repeat(arms[s][:1], n, axis=0), arms[s]])
                    legs = {s: {t_ + n: e_ + n for t_, e_ in v_.items()} for s, v_ in legs.items()}
                    prefix += n; T += n
                done = dict(rec, parked=True, mode=mode, frames=n, joint_dist_rad=tgt["joint_dist_rad"], clear_active_mm=tgt["clear_active_mm"],
                            margin_mm=round(1000 * pr["margin"], 1), path_waypoints=len(pr["waypoints"]))
                cache.clear()
                break
            rep["parked"].append(done or dict(rec, parked=False, why="no collision-free way to a clearance configuration"))
    # ---- 2. free-space legs: planned where the straight joint path runs into something. First every leg that collides
    # only with the other station's motion is retimed along its own straight path (cheap, and the order of the two arms
    # does not matter); then the legs still colliding get a planned path; finally every leg is checked again
    trig = float(c["trigger_m"])

    def leg_state(a, t, e):
        t0_ = max(t - 1, 0)
        q_prev = arms[a][t - 1] if t > 0 else np.asarray(starts.get(a, arms[a][0]), float)
        held, held_geoms = planner.carried(Qref[t0_], Qref[e], a)
        per = [ctx(t0_, a)] + [ctx(f, a) for f in range(t, e + 1)]
        v, fr, pn, cm = planner.leg_violation(a, np.vstack([q_prev, arms[a][t:e + 1]]), per, held, held_geoms, level=trig)
        return t0_, q_prev, held, held_geoms, per, v, fr, pn, cm

    def apply(a, t, e, A_new, held, rec):
        arms[a][t:e + 1] = A_new
        for k, f in enumerate(range(t, e + 1)):   # what the hand carries follows the new hand path in the reference
            if held:
                qf = planner.place(Qref[f], a, A_new[k], held)
                for adr, _, _ in held:
                    Qref[f, adr:adr + 7] = qf[adr:adr + 7]
        cache.clear()
        rep["planned_legs"] = [p_ for p_ in rep["planned_legs"] if not (p_["side"] == a and p_["frames"] == rec["frames"])]
        rep["planned_legs"].append(rec)

    tried = {}
    for geometric in (False, True):
        for a in active:
            for t, e in sorted(legs.get(a, {}).items()):
                t0_, q_prev, held, held_geoms, per, v, fr, pn, cm = leg_state(a, t, e)
                if v >= 0:
                    continue
                if any(planner.touches(per[0], o, held_geoms) for o in sides if o != a):
                    continue                          # both hands carry the part (a coupled carry): the leg is not one arm's
                rec = {"side": a, "frames": [int(t - prefix), int(e - prefix)], "label": str(labels[t]),
                       "straight": {"clearance_mm": round(1000 * cm, 1), "between": pn, "frame": None if fr is None else int(t + fr - 1 - prefix)}}
                tried.setdefault((a, t), rec)
                # the other station moves during this leg (a two-hand entry, an arm parking): an obstacle that moves
                other_moves = any(np.abs(np.diff(arms[o][t0_:e + 1], axis=0)).max() > 1e-6 for o in sides if o != a and e > t0_)
                A_new, how, why = None, None, None
                if not geometric:
                    if not other_moves:
                        continue
                    # the straight path itself, timed to wait where the other station passes
                    A_new, lvl = _timed_leg(planner, a, densify([q_prev, arms[a][e]], float(c["res_rad"]) / 2), per, held, held_geoms, fps, vmax, c["margins_m"])
                    how = "straight path, timed around the other station"
                else:
                    # a path around it: against the scene where the other station starts and ends this leg (when it
                    # moves, then timed against its motion) or everywhere it is during it (when it stands)
                    frames = [t0_, e] if other_moves else sorted({int(x) for x in np.linspace(t0_, e, min(8, e - t + 2))})
                    pr = planner.plan_leg(a, q_prev, arms[a][e], [ctx(f, a) for f in frames], held, held_geoms)
                    if not pr["ok"]:
                        tried[(a, t)]["why"] = pr["reason"]
                        continue
                    if other_moves:
                        A_new, lvl = _timed_leg(planner, a, pr["path"], per, held, held_geoms, fps, vmax, [pr["margin"]] + [m_ for m_ in c["margins_m"] if m_ < pr["margin"]])
                        how = f"planned path ({len(pr['waypoints'])} waypoints), timed around the other station"
                        why = "the planned path finds no timing that lets the other station pass"
                    else:
                        A_new, lvl = quintic_frames(pr["path"], e - t + 1), pr["margin"]
                        how = f"planned path ({len(pr['waypoints'])} waypoints, smoothing {pr['smoothing']})"
                if A_new is None:
                    if why:
                        tried[(a, t)]["why"] = why
                    continue
                v2, fr2, pn2, cm2 = planner.leg_violation(a, np.vstack([q_prev, A_new]), per, held, held_geoms, level=0.5 * lvl)
                if v2 < 0:
                    tried[(a, t)]["why"] = f"{how}: timed, it comes {1000 * cm2:.1f} mm to {pn2}"
                    continue
                apply(a, t, e, A_new, held, dict(rec, how=how, margin_mm=round(1000 * lvl, 1), clearance_mm=round(1000 * cm2, 1)))
    # every leg once more against the final paths of both stations: what still collides is reported
    for a in active:
        for t, e in sorted(legs.get(a, {}).items()):
            rep["checked_legs"] += 1
            st = leg_state(a, t, e)
            if st[5] < 0 and not any(planner.touches(st[4][0], o, st[3]) for o in sides if o != a):
                rec = tried.get((a, t)) or {"side": a, "frames": [int(t - prefix), int(e - prefix)], "label": str(labels[t])}
                rep["failed_legs"].append(dict(rec, now={"clearance_mm": round(1000 * st[8], 1), "between": st[7]},
                                               why=rec.get("why") or "no planned path or timing clears it"))
    rep["seconds"] = round(time.time() - t_start, 2)
    return {"Qref": Qref, "arms": arms, "labels": labels, "legs": legs, "prefix": prefix, "report": rep}

