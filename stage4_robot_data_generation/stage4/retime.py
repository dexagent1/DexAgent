"""Retiming: slow down the frames where a joint moves faster than its limit.

A layout can ask an arm to cover more ground in the same blend than the
original did (the object moved away from the arm's base, so the reach is
longer). Instead of rejecting such a layout, the trajectory is stretched in
time exactly where it is too fast: every frame step gets a slow-down ratio
(the fastest joint's speed over its limit, at least 1), the ratios are
widened and smoothed over a few frames so the slow-down ramps in and out, and
the trajectory is resampled at the stretched times. Frames that were fast
enough keep their timing.

This is a small, self-contained velocity-only retime with the interface of the
arm planner's (tool_library.common.armplan.retime: X [T, n], fps, vmax per
column, keep, ignore_step -> {src, added_frames, ok, vel_ratio, ...}, and
resample(X, src)). Its window is the one the Stage 3 executor uses when it
couples an arm to its hand (a 7-frame running maximum, then a 7-frame box
average, which never lowers a frame's own ratio). Stage 4 passes the limits its
joint-speed check applies, so a retimed trajectory passes that check by
construction. It stays separate from armplan.retime on purpose: that one also
limits acceleration and ramps over a time window, while the Stage 4 check it
must satisfy is on velocity only.
"""
from __future__ import annotations

import numpy as np

SMOOTH_FRAMES = 7          # half width of the ramp, as in the executor's arm planner
MARGIN = 1.02              # aim 2 % under the limit, so interpolation round-off never crosses it


def _limit(v, n):
    v = np.asarray(v, float).reshape(-1)
    return np.full(n, float(v[0])) if v.size == 1 else v


def retime(X, fps, vmax=None, amax=None, keep=None, max_frames=None, ignore_step=None, smooth=SMOOTH_FRAMES):
    """Time map that keeps every column of X [T, n] within its velocity limit vmax (units/s, one value or one per
    column; a column with an infinite limit is free), slowing down only where needed. amax is accepted for the
    armplan interface and not used (velocity only). Steps inside `keep` (bool [T] on source frames) keep their own
    ratio but are not widened by their neighbours'. A step larger than `ignore_step` in some column is left as it is.
    A retime that would need more than max_frames frames returns the identity map with ok False. Returns {src [T']
    (fractional source frame of each new frame), added_frames, ok, vel_ratio, acc_ratio, iterations}; resample(X, src)
    gives the retimed trajectory."""
    X = np.asarray(X, float)
    T = len(X)
    X = X.reshape(T, -1)
    ident = {"src": np.arange(T, dtype=float), "added_frames": 0, "ok": True, "vel_ratio": 0.0, "acc_ratio": 0.0,
             "iterations": 0}
    if T < 2 or vmax is None:
        return ident
    vm = _limit(vmax, X.shape[1])
    st = np.abs(np.diff(X, axis=0)) * float(fps)
    rr = np.max(np.where(np.isfinite(vm) & (vm > 0), MARGIN * st / np.where(vm > 0, vm, 1.0), 0.0), axis=1)
    if ignore_step is not None:
        rr[np.abs(np.diff(X, axis=0)).max(axis=1) > float(ignore_step)] = 0.0    # a jump is reported, not slowed for
    own = np.maximum(rr, 1.0)
    if not (own > 1.0 + 1e-9).any():
        return dict(ident, vel_ratio=round(float(rr.max(initial=0.0)) / MARGIN, 4))
    k = int(smooth)
    pad = np.pad(own, k, mode="edge")
    wide = np.array([pad[i:i + 2 * k + 1].max() for i in range(len(own))])
    wide = np.convolve(np.pad(wide, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
    ratio = np.maximum(wide, own)
    if keep is not None:
        kp = np.asarray(keep, bool)
        ks = kp[:-1] & kp[1:]
        ratio[ks] = own[ks]                       # a kept step slows only for its own motion
    taus = [0.0]
    while taus[-1] < T - 1:
        i = min(int(taus[-1]), T - 2)
        taus.append(taus[-1] + 1.0 / ratio[i])
        if max_frames and len(taus) > int(max_frames):
            # slowing down this much is no retime: leave the timing as it is and say so
            return dict(ident, ok=False, vel_ratio=round(float(rr.max()) / MARGIN, 4))
    taus[-1] = min(taus[-1], T - 1.0)
    src = np.asarray(taus)
    return {"src": src, "added_frames": int(len(src) - T), "ok": True, "vel_ratio": round(float(rr.max()) / MARGIN, 4),
            "acc_ratio": 0.0, "iterations": 1}


def quaternion_slices(model):
    """qpos slices holding unit quaternions (free joints' rotation, ball joints)."""
    import mujoco
    out = []
    for j in range(model.njnt):
        kind = int(model.jnt_type[j])
        a = int(model.jnt_qposadr[j])
        if kind == int(mujoco.mjtJoint.mjJNT_FREE):
            out.append(slice(a + 3, a + 7))
        elif kind == int(mujoco.mjtJoint.mjJNT_BALL):
            out.append(slice(a, a + 4))
    return out


def resample(X, src, quats=()):
    """X at the (float) source frames src: linear between neighbours, unit quaternions renormalized (and kept on the
    hemisphere of the earlier frame, so the blend never passes through the long way round)."""
    X = np.asarray(X, float)
    src = np.asarray(src, float)
    T = len(X)
    if len(src) == T and np.allclose(src, np.arange(T)):
        return X.copy()
    i0 = np.clip(np.floor(src).astype(int), 0, T - 1)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (src - i0)[:, None]
    A, B = X[i0], X[i1].copy()
    for sl in quats:
        flip = np.sum(A[:, sl] * B[:, sl], axis=1) < 0
        B[flip, sl] = -B[flip, sl]
    out = (1 - w) * A + w * B
    for sl in quats:
        out[:, sl] /= np.maximum(np.linalg.norm(out[:, sl], axis=1, keepdims=True), 1e-12)
    return out


def resample_labels(labels, src):
    """Each retimed frame keeps the label of the source frame it starts in."""
    T = len(labels)
    return [labels[min(int(np.floor(s)), T - 1)] for s in src]


def window(src, a, b):
    """The retimed frames [a', b'] that start in the source frames [a, b] (inclusive): consecutive windows stay
    consecutive, every retimed frame belongs to the window of the source frame it starts in."""
    src = np.asarray(src, float)
    lo = int(np.searchsorted(src, float(a) - 1e-9, side="left"))
    hi = int(np.searchsorted(src, float(b) + 1.0 - 1e-9, side="left")) - 1
    lo = min(max(lo, 0), len(src) - 1)
    return lo, max(lo, min(hi, len(src) - 1))
