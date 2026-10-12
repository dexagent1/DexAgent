"""Arm and hand solved together: the hand is bolted to its arm's flange.

At every frame the hand root (the `ee_body` of `episode.robot_parts`) must sit exactly where the arm's flange puts it,
hand = FK(arm) x mount^-1. Planning the hand alone and fitting the arm afterwards produced hands that hovered up to
centimetres off the flange or turned against it (the arm fitter accepted 30 mm / 20 deg, lagged behind a 95 deg/s rate
limit, and the plan never knew a pose was out of reach). Here the arm is part of the plan:

  couple(model, Qref, ...)   dense IK of the arm along the hand reference (every frame, full 6-DoF, warm-started and
                             branch-continuous); free-space legs (reach for, return home) are planned in ARM joint
                             space between reachable ends; the hand is then put on the flange (moved by at most the
                             projection tolerance); legs the arm would take faster than its speed limit or accelerate
                             harder than the acceleration audits allow are slowed down (retime_arms). With a planner
                             (armplan.ArmPlanner) a leg whose straight joint path runs into something is planned around
                             it and an idle arm in the active one's way is parked (armplan.plan_arms). Frames the arm cannot hold the hand at (residual above the projection
                             tolerance) are reported: the caller re-solves the pose (a different grasp, a re-oriented
                             pose, a via).
  fit_executed(model, Q, A)  after physics: the arm under the EXECUTED hand (contact pushes the hand a few millimetres
                             off its command), warm-started on the planned arm path, strict.
  weld_residuals(model, Q)   per frame, per side: how far the hand root is from where the flange holds it (mm, deg).
"""
from __future__ import annotations

import numpy as np
import mujoco

from tool_library.common import trajectory as TJ

FREE_WORDS = (": reach for", ": reach to hold", ": return home")
FREE_MARK = "(free)"        # a carry planned with arm_path 'joint': the transfer is a joint-space leg (the arm may change
                            # configuration on the way); its set-down '(down)' still follows the planned straight path
# phases where nothing touches anything new and the exact hand path does not matter (the hand holds what it holds or
# nothing): there the hand may follow the arm farther from the planned straight line
TRANSIT_WORDS = (": carry", ": lift", ": retreat", ": back off", ": reach for", ": reach to hold", ": return home", ": move away")
CONTACT_WORDS = ("(down)", "set down", "descend", "close on", "grip", "let go", "insert", "press", "push", "pull", "turn",
                 "screw", "pour", "tilt", "hold", "approach")
DEFAULTS = dict(tol_pos_m=0.0005, tol_rot_deg=0.3,          # the arm holds the hand: IK converged
                project_pos_m=0.006, project_rot_deg=4.0,    # the hand may be moved onto the flange by this much
                transit_pos_m=0.03, transit_rot_deg=10.0,    # ... and by this much in a transit phase
                max_speed_deg_s=90.0,                         # arm joint speed ceiling (slower legs are retimed; joint_speed_check allows 3.5 deg/frame)
                jump_rad=0.35,                                # a joint step above this in one frame is a branch jump
                n_seeds=10, ik_iters=60)


def _parts(model):
    from tool_library.common import episode as E
    return E.robot_parts(model)


def _rel(model, data, pt, q):
    data.qpos[:] = q
    mujoco.mj_kinematics(model, data)
    Ry, py = data.xmat[pt["ee_body"]].reshape(3, 3).copy(), data.xpos[pt["ee_body"]].copy()
    Rg, pg = data.xmat[pt["gripper"]].reshape(3, 3).copy(), data.xpos[pt["gripper"]].copy()
    return Ry.T @ (pg - py), Ry.T @ Rg


def _chordal_mean(Rs):
    U, _, Vt = np.linalg.svd(np.mean(Rs, axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def mount(model, side, Q=None, parts=None):
    """The bolt: the flange (arm gripper body) pose in the hand-root frame -> (p [3], R [3, 3], source).

    A DexAgent bench scene's home keyframe has the hand on the flange. A scene without one (a hand-built twin) gives
    the relative pose most frames of its trajectory share: the median position and the mean rotation of the frames
    within 5 deg of the overall mean."""
    parts = parts or _parts(model)
    pt = parts[side]
    d = mujoco.MjData(model)
    if model.nkey:
        p, R = _rel(model, d, pt, model.key_qpos[0])
        if np.linalg.norm(p) < 0.05:
            return p, R, "home keyframe"
    if Q is not None and len(Q):
        idx = np.unique(np.linspace(0, len(Q) - 1, min(len(Q), 300)).astype(int))
        rel = [_rel(model, d, pt, Q[t]) for t in idx]
        P = np.array([r[0] for r in rel]); Rs = np.array([r[1] for r in rel])
        R = _chordal_mean(Rs)
        ang = np.array([TJ.angle_between(R, r) for r in Rs])
        close = Rs[ang < np.radians(5.0)]
        if len(close) >= 3:
            R = _chordal_mean(close)
        return np.median(P, axis=0), R, "trajectory mode"
    p, R = _rel(model, d, pt, model.qpos0)
    return p, R, "qpos0"


class ArmSide:
    """One station: the arm chain, the hand root's own (virtual) joints and the bolt between them."""

    def __init__(self, model, side, parts=None, bolt=None, cfg=None, Q_for_mount=None):
        self.model, self.side = model, side
        self.cfg = {**DEFAULTS, **(cfg or {})}
        parts = parts or _parts(model)
        self.pt = pt = parts[side]
        if pt["gripper"] < 0 or not pt["arm_joints"]:
            raise ValueError(f"the {side} station has no arm")
        self.arm = TJ.Chain(model, pt["gripper"], pt["arm_joints"])
        self.hand = TJ.Chain(model, pt["ee_body"])
        self.p_m, self.R_m, self.mount_source = bolt or mount(model, side, Q=Q_for_mount, parts=parts)
        self.data = mujoco.MjData(model)
        q0 = model.key_qpos[0] if model.nkey else model.qpos0
        self.home = np.asarray(q0, float)[self.arm.qadr].copy()
        self.rng = np.random.default_rng(0)

    # --- poses
    def flange_target(self, q):
        """Where the flange must be for the hand of full qpos q."""
        d = self.data
        d.qpos[:] = q
        mujoco.mj_kinematics(self.model, d)
        Ry = d.xmat[self.pt["ee_body"]].reshape(3, 3)
        return d.xpos[self.pt["ee_body"]] + Ry @ self.p_m, Ry @ self.R_m

    def hand_from_arm(self, q, qa):
        """The hand-root pose the arm configuration qa holds the hand at."""
        d = self.data
        d.qpos[:] = q
        d.qpos[self.arm.qadr] = qa
        mujoco.mj_kinematics(self.model, d)
        Rg, pg = d.xmat[self.pt["gripper"]].reshape(3, 3), d.xpos[self.pt["gripper"]]
        Ry = Rg @ self.R_m.T
        return pg - Ry @ self.p_m, Ry

    def residual(self, q):
        """(mm, deg): the hand root of qpos q against where q's own arm holds it."""
        p, R = self.hand_from_arm(q, np.asarray(q, float)[self.arm.qadr])
        d = self.data
        d.qpos[:] = q
        mujoco.mj_kinematics(self.model, d)
        py, Ry = d.xpos[self.pt["ee_body"]], d.xmat[self.pt["ee_body"]].reshape(3, 3)
        return 1000.0 * float(np.linalg.norm(py - p)), float(np.degrees(TJ.angle_between(R, Ry)))

    # --- IK
    _SEEDS = {}

    def seed_library(self, n=4000):
        """Arm configurations spread over the joint box with their flange poses (cached per model and side): the
        nearest ones in pose space are the multistart seeds, which finds solutions near the joint limits that random
        restarts miss."""
        b0 = int(self.model.jnt_bodyid[self.pt["arm_joints"][0]])
        dk = mujoco.MjData(self.model)
        mujoco.mj_kinematics(self.model, dk)                  # the station's WORLD placement (a moved station invalidates the cache)
        key = (id(self.model), self.side, self.model.nq, tuple(np.round(dk.xpos[b0], 4)), tuple(np.round(dk.xquat[b0], 4)),
               tuple(np.round(self.arm.lo, 3)))
        if key not in ArmSide._SEEDS:
            rng = np.random.default_rng(12345)
            Qs = rng.uniform(self.arm.lo, self.arm.hi, size=(int(n), len(self.arm.qadr)))
            d = mujoco.MjData(self.model)
            d.qpos[:] = self.model.key_qpos[0] if self.model.nkey else self.model.qpos0
            P = np.zeros((len(Qs), 3)); Rq = np.zeros((len(Qs), 4))
            for i, qa in enumerate(Qs):
                d.qpos[self.arm.qadr] = qa
                mujoco.mj_kinematics(self.model, d)
                P[i] = d.xpos[self.pt["gripper"]]
                Rq[i] = TJ.mat2quat(d.xmat[self.pt["gripper"]].reshape(3, 3))
            ArmSide._SEEDS[key] = (Qs, P, Rq)
        return ArmSide._SEEDS[key]

    def nearest_seeds(self, p, R, k=6):
        Qs, P, Rq = self.seed_library()
        qt = TJ.mat2quat(R)
        ang = 2.0 * np.arccos(np.clip(np.abs(Rq @ qt), 0.0, 1.0))
        score = np.linalg.norm(P - np.asarray(p), axis=1) + 0.08 * ang
        return [Qs[i] for i in np.argsort(score)[:int(k)]]

    def _ik(self, q, p, R, seed, iters):
        d = self.data
        d.qpos[:] = q
        d.qpos[self.arm.qadr] = seed
        ep, er = self.arm.solve(d, p, R, iters=iters, rot_weight=0.5, damping=1e-3, max_step=0.3)
        return float(ep), float(er), d.qpos[self.arm.qadr].copy()

    def ok(self, ep, er):
        return ep <= self.cfg["tol_pos_m"] and er <= np.radians(self.cfg["tol_rot_deg"])

    def within_projection(self, ep, er):
        return ep <= self.cfg["project_pos_m"] and er <= np.radians(self.cfg["project_rot_deg"])

    def solve(self, q, p, R, prev, more_seeds=(), multistart=True):
        """Arm configuration that puts the flange at (p, R): the solution nearest `prev` among those that converge.
        A warm-started solution within the projection tolerance and without a branch jump is taken as it is; other
        branches are searched only when it is not (or when `multistart` is forced on by the caller)."""
        c = self._ik(q, p, R, prev, self.cfg["ik_iters"])
        cont = np.abs(c[2] - prev).max() <= self.cfg["jump_rad"]
        if cont and (self.ok(c[0], c[1]) or (self.within_projection(c[0], c[1]) and multistart != "force")):
            return c
        if not multistart:
            return c
        seeds = [self.home] + [np.asarray(s, float) for s in more_seeds]
        seeds += [np.clip(prev + self.rng.normal(0.0, 0.3, len(prev)), self.arm.lo, self.arm.hi) for _ in range(self.cfg["n_seeds"] // 2)]
        seeds += self.nearest_seeds(p, R, k=self.cfg["n_seeds"] - self.cfg["n_seeds"] // 2)
        cands = [c] + [self._ik(q, p, R, s, 2 * self.cfg["ik_iters"]) for s in seeds]
        good = [x for x in cands if self.ok(x[0], x[1])]
        if not good:                                  # a last, wider net before calling the pose unreachable
            more = self.nearest_seeds(p, R, k=24)[self.cfg["n_seeds"] // 2:] + [self.rng.uniform(self.arm.lo, self.arm.hi) for _ in range(6)]
            cands += [self._ik(q, p, R, s, 3 * self.cfg["ik_iters"]) for s in more]
            good = [x for x in cands if self.ok(x[0], x[1])]
        if good:
            return min(good, key=lambda x: float(np.abs(x[2] - prev).max()))
        return min(cands, key=lambda x: x[0] + 0.05 * x[1] + 0.002 * float(np.abs(x[2] - prev).max()))

    def put_hand(self, q, qa):
        """q with the hand root moved onto the flange of arm configuration qa (and the arm set to qa)."""
        p, R = self.hand_from_arm(q, qa)
        d = self.data
        d.qpos[:] = q
        d.qpos[self.arm.qadr] = qa
        self.hand.solve(d, p, R, iters=40, rot_weight=1.0, damping=1e-4, max_step=0.5)
        return d.qpos.copy()


def is_transit(label):
    lab = str(label).lower()
    return any(w in lab for w in TRANSIT_WORDS) and not any(w in lab for w in CONTACT_WORDS)


def free_legs(labels, keyframe_frames=None):
    """[(first frame, target frame)] of the free-space legs: maximal runs of a 'reach for' / 'return home' label
    whose hand moves; the target is where the label's keyframe is reached (its start frame when known, else the
    last frame of the run)."""
    runs, t, T = [], 0, len(labels)
    while t < T:
        lab = str(labels[t])
        s = t
        while t + 1 < T and str(labels[t + 1]) == lab:
            t += 1
        if any(w in lab for w in FREE_WORDS) or (FREE_MARK in lab and "(down)" not in lab):
            runs.append((s, t))
        t += 1
    out = []
    kfs = sorted(int(k) for k in (keyframe_frames or []))
    for s, e in runs:
        inside = [k for k in kfs if s < k <= e]       # the leg's destination keyframe (a run can start on its own first keyframe)
        tgt = inside[-1] if inside else e
        if tgt > s:
            out.append((s, tgt))
    return out


def couple(model, Qref, fps, arm_start=None, labels=None, keyframe_frames=None, sides=None, cfg=None, parts=None,
           retime=True, max_frames=None, planner=None, park=True):
    """Make the hand reference one the arms execute exactly. Returns dict:
        qpos [T', nq]   the reference with the arm columns filled and every hand on its flange,
        arm {side: [T', n]}, labels [T'], src_index [T'] (fractional index into Qref, for retimed frames),
        report {side: {...}, ok, infeasible [...], retimed_s, plan (with a planner)}.
    `arm_start` {side: arm joints at the first frame} (default: Qref[0]'s own arm columns). With `planner` (an
    armplan.ArmPlanner of the same scene) the free-space legs whose straight joint path runs into something are planned
    around it and an idle arm in the active station's way is parked (`park`), see armplan.plan_arms; without one the
    legs stay straight. `legs` and `arm_start_used` in the result let a caller plan the arms itself (_couple_two_hands)."""
    cfg = {**DEFAULTS, **(cfg or {})}
    parts = parts or _parts(model)
    Qref = np.asarray(Qref, float)
    T = len(Qref)
    labels = list(labels) if labels is not None else [""] * T
    sides = [s for s in (sides or parts) if parts[s]["gripper"] >= 0 and parts[s]["arm_joints"]]
    legs = free_legs(labels, keyframe_frames)
    Q = Qref.copy()
    arms, report = {}, {"ok": True, "infeasible": []}
    side_legs, starts = {}, {}
    for side in sides:
        S = ArmSide(model, side, parts=parts, cfg=cfg)
        a0 = np.asarray((arm_start or {}).get(side, Qref[0][S.arm.qadr]), float)
        A = np.zeros((T, len(S.arm.qadr)))
        ep, er = np.zeros(T), np.zeros(T)
        moved = np.zeros(T, bool)                  # the hand reference moves on this side (a still hand is skipped)
        hq = S.hand.qadr
        moved[1:] = np.abs(np.diff(Qref[:, hq], axis=0)).max(axis=1) > 1e-9
        leg_of = {}
        for s, e in legs:
            if moved[s:e + 1].any() and str(labels[s]).startswith(f"{side}:"):
                leg_of[s] = e
        q_prev = a0.copy()
        free = np.zeros(T, bool)
        t, cool = 0, 0
        while t < T:
            if t in leg_of:                         # a free-space leg: arm joint space between reachable ends
                e = leg_of[t]
                p, R = S.flange_target(Qref[e])
                ep_e, er_e, qe = S.solve(Qref[e], p, R, q_prev)
                n = e - t + 1
                u = TJ.smoothstep(np.arange(1, n + 1) / n)[:, None]
                A[t:e + 1] = q_prev + u * (qe - q_prev)
                ep[t:e], er[t:e] = 0.0, 0.0
                ep[e], er[e] = ep_e, er_e
                free[t:e + 1] = True
                q_prev = qe.copy()
                t = e + 1
                continue
            p, R = S.flange_target(Qref[t])
            ep[t], er[t], A[t] = S.solve(Qref[t], p, R, q_prev, multistart=(t >= cool))
            if not S.within_projection(ep[t], er[t]) and t >= cool:
                cool = t + 10                   # the other branches could not hold it either: no new search for a while
            q_prev = A[t].copy()
            t += 1
        arms[side] = A
        side_legs[side], starts[side] = dict(leg_of), a0.copy()
        transit = np.array([is_transit(l) for l in labels])
        # where a subgoal ends (a carry's destination) and where a joint-space carry arrives, the hand must be on its
        # target: the transit slack (30 mm) is for the way there, not for the destination
        if T:
            transit[-1] = False
        for s_, e_ in leg_of.items():
            if FREE_MARK in str(labels[e_]):
                transit[e_] = False
        tol_p = np.where(transit, cfg["transit_pos_m"], cfg["project_pos_m"])
        tol_r = np.radians(np.where(transit, cfg["transit_rot_deg"], cfg["project_rot_deg"]))
        bad = np.where((ep > tol_p) | (er > tol_r))[0]
        projected = np.where(((ep > cfg["tol_pos_m"]) | (er > np.radians(cfg["tol_rot_deg"]))) & ~((ep > tol_p) | (er > tol_r)))[0]
        jumps = np.where(np.abs(np.diff(A, axis=0)).max(axis=1) > cfg["jump_rad"])[0] + 1 if T > 1 else np.array([], int)
        spans = []
        for f in bad:
            if spans and f <= spans[-1][1] + 3:
                spans[-1][1] = int(f)
            else:
                spans.append([int(f), int(f)])
        inf = [{"side": side, "frames": [a, b], "label": str(labels[a]), "worst_mm": round(1000 * float(ep[a:b + 1].max()), 1),
                "worst_deg": round(float(np.degrees(er[a:b + 1].max())), 1)} for a, b in spans]
        report[side] = {"mount": S.mount_source, "frames": T, "free_leg_frames": int(free.sum()),
                        "ik_residual_mm_max": round(1000 * float(ep.max()), 2), "ik_residual_deg_max": round(float(np.degrees(er.max())), 2),
                        "projected_frames": int(len(projected)),
                        "projected_mm_max": round(1000 * float(ep[projected].max()), 2) if len(projected) else 0.0,
                        "infeasible_frames": int(len(bad)), "branch_jumps": [int(j) for j in jumps[:8]],
                        "branch_jump_labels": sorted({str(labels[j]) for j in jumps})[:4]}
        report["infeasible"] += inf
        if len(bad) or len(jumps):
            report["ok"] = False
    # --- around obstacles and out of each other's way (with a planner only): may prepend frames for an arm to park
    base = np.arange(T, dtype=float)                  # source index into Qref of every frame
    if planner is not None and arms and T > 1:
        from tool_library.common import armplan as AP
        pl = AP.plan_arms(model, planner, Q, arms, side_legs, starts, labels, fps, park=park)
        report["plan"] = pl["report"]
        if pl["prefix"]:
            base = np.concatenate([np.zeros(pl["prefix"]), base])
        Q, arms, labels, side_legs = pl["Qref"], pl["arms"], pl["labels"], pl["legs"]
    # --- retime where an arm would be too fast or accelerate too hard
    Q, arms, labels, src, report["retimed_s"] = retime_arms(model, Q, arms, labels, fps, cfg, parts, max_frames) if retime else (Q, arms, labels, np.arange(len(Q), dtype=float), 0.0)
    src = np.interp(src, np.arange(len(base)), base)
    # --- every hand on its flange
    for side, A in arms.items():
        S = ArmSide(model, side, parts=parts, cfg=cfg)
        for t in range(len(Q)):
            Q[t] = S.put_hand(Q[t], A[t])
        sp = np.degrees(np.abs(np.diff(A, axis=0)).max()) * float(fps) if len(A) > 1 else 0.0
        report[side]["peak_speed_deg_s"] = round(float(sp), 1)
    return {"qpos": Q, "arm": arms, "labels": labels, "src_index": src, "report": report, "legs": side_legs, "arm_start_used": starts}


def retime_arms(model, Q, arms, labels, fps, cfg=None, parts=None, max_frames=None):
    """Slow the reference where an arm would move faster than max_speed_deg_s or accelerate harder than the
    acceleration audits allow (armplan.retime: joint accelerations and the flanges' linear acceleration, smoothed as
    joint_acceleration_audit smooths them); contact phases keep their pace unless their own motion needs slowing, a
    branch jump is reported, not slowed for. Every column of Q is resampled (free-body quaternions kept unit), the arms
    and labels with it. Returns (Q, arms, labels, src [T'] fractional index, added seconds)."""
    from tool_library.common import armplan as AP
    cfg = {**DEFAULTS, **(cfg or {})}
    T = len(Q)
    if not arms or T < 3:
        return Q, arms, labels, np.arange(T, dtype=float), 0.0
    _, amax, flange_amax = AP.arm_limits()
    cols, vm, am = [], [], []
    d = mujoco.MjData(model)
    parts = parts or _parts(model)
    for s, A in arms.items():
        cols.append(np.asarray(A, float)); vm += [np.radians(cfg["max_speed_deg_s"])] * A.shape[1]; am += [amax] * A.shape[1]
        P = np.zeros((T, 3))                          # the flange (what a held part feels), FK of the arm alone
        d.qpos[:] = Q[0]
        for t in range(T):
            d.qpos[parts[s]["arm_qadr"]] = A[t]
            mujoco.mj_kinematics(model, d)
            P[t] = d.xpos[parts[s]["gripper"]]
        cols.append(P); vm += [np.inf] * 3; am += [flange_amax] * 3
    keep = np.array([not is_transit(l) for l in labels])
    r = AP.retime(np.hstack(cols), fps, vmax=vm, amax=am, keep=keep, max_frames=max_frames, ignore_step=cfg["jump_rad"])
    src = r["src"]
    if not r["added_frames"]:
        return Q, arms, labels, np.arange(T, dtype=float), 0.0
    Q = AP.resample(Q, src, AP._quat_cols(model))
    arms = {s: AP.resample(A, src) for s, A in arms.items()}
    labels = [labels[i] for i in np.floor(src).astype(int)]
    return Q, arms, labels, src, round((len(src) - T) / float(fps), 2)


def fit_executed(model, Q, arm_plan=None, sides=None, cfg=None, parts=None, bolt=None, fps=30.0, weld_tol_mm=2.0, weld_tol_deg=2.0):
    """The arms under the EXECUTED hands. Q [T, nq] from physics; arm_plan {side: [T, n]} the planned arm path
    (warm starts; default Q's own arm columns). Returns (Q with the arm columns set, report per side).

    The planned arm where it already holds the executed hand within 0.5 mm / 1 deg; elsewhere exact where that keeps
    the arm under its speed limit. Near a singularity (a folded elbow) the millimetre the hand
    lags its servo command can ask the joints for a 10-degree jump in one frame; then the arm takes the speed-limited
    step instead, as long as the hand stays on the flange within the weld tolerance (else the exact jump is kept and
    shows as a speed failure)."""
    cfg = {**DEFAULTS, **(cfg or {})}
    parts = parts or _parts(model)
    Q = np.array(Q, float)
    rep = {}
    for side in (sides or parts):
        pt = parts[side]
        if pt["gripper"] < 0 or not pt["arm_joints"]:
            continue
        S = ArmSide(model, side, parts=parts, cfg=cfg, bolt=(bolt or {}).get(side))
        plan = None if arm_plan is None else arm_plan.get(side)
        T = len(Q)
        A = np.zeros((T, len(S.arm.qadr)))
        ep, er = np.zeros(T), np.zeros(T)
        prev = (plan[0] if plan is not None else Q[0][S.arm.qadr]).copy()
        step_max = np.radians(min(1.15 * cfg["max_speed_deg_s"], 103.0)) / float(fps)     # under the audit's 105 deg/s
        limited = 0
        for t in range(T):
            p, R = S.flange_target(Q[t])
            seed = plan[t] if plan is not None else prev
            if plan is not None and (t == 0 or np.abs(plan[t] - prev).max() <= step_max):
                # the planned arm already holds the executed hand (within 0.5 mm / 1 deg): keep it. Near a folded elbow
                # an exact re-fit of a sub-degree hand deviation lands on solutions degrees apart (300 deg/s steps)
                d = S.data
                d.qpos[:] = Q[t]; d.qpos[S.arm.qadr] = plan[t]; mujoco.mj_kinematics(model, d)
                e_p0 = float(np.linalg.norm(d.xpos[S.pt["gripper"]] - p))
                e_r0 = float(TJ.angle_between(R, d.xmat[S.pt["gripper"]].reshape(3, 3)))
                if 1000 * e_p0 <= min(0.5, weld_tol_mm) and np.degrees(e_r0) <= min(1.0, weld_tol_deg):
                    ep[t], er[t], A[t] = e_p0, e_r0, np.asarray(plan[t], float).copy()
                    prev = A[t].copy()
                    continue
            c = S._ik(Q[t], p, R, seed, cfg["ik_iters"])
            if not (S.ok(c[0], c[1]) and np.abs(c[2] - prev).max() <= cfg["jump_rad"]):
                c2 = S.solve(Q[t], p, R, prev, more_seeds=[seed])
                if c2[0] + 0.05 * c2[1] < c[0] + 0.05 * c[1] or np.abs(c[2] - prev).max() > cfg["jump_rad"]:
                    c = c2
            if t > 0 and np.abs(c[2] - prev).max() > step_max and plan is not None and np.abs(plan[t] - prev).max() <= step_max:
                # the planned arm, when it still holds the executed hand within the weld tolerance: near a folded elbow
                # the exact fit chases a degree of hand deviation with a 300 deg/s joint jump the planned path avoids
                d = S.data
                d.qpos[:] = Q[t]; d.qpos[S.arm.qadr] = plan[t]; mujoco.mj_kinematics(model, d)
                pg, Rg = d.xpos[S.pt["gripper"]], d.xmat[S.pt["gripper"]].reshape(3, 3)
                e_p, e_r = float(np.linalg.norm(pg - p)), float(TJ.angle_between(R, Rg))
                if 1000 * e_p <= 0.9 * weld_tol_mm and np.degrees(e_r) <= 0.9 * weld_tol_deg:
                    c = (e_p, e_r, np.asarray(plan[t], float).copy()); limited += 1
            if t > 0 and np.abs(c[2] - prev).max() > step_max:
                # the speed-limited step toward the exact solution, refined with the step bound kept
                lim = prev + np.clip(c[2] - prev, -step_max, step_max)
                d = S.data
                d.qpos[:] = Q[t]; d.qpos[S.arm.qadr] = lim
                e_l = S.arm.solve(d, p, R, iters=20, rot_weight=0.5, damping=0.05, max_step=step_max)
                q_l = np.clip(d.qpos[S.arm.qadr], prev - step_max, prev + step_max)
                d.qpos[S.arm.qadr] = q_l
                mujoco.mj_kinematics(model, d)
                pg, Rg = d.xpos[S.pt["gripper"]], d.xmat[S.pt["gripper"]].reshape(3, 3)
                e_p, e_r = float(np.linalg.norm(pg - p)), float(TJ.angle_between(R, Rg))
                if 1000 * e_p <= weld_tol_mm and np.degrees(e_r) <= weld_tol_deg:
                    c = (e_p, e_r, q_l.copy()); limited += 1
            ep[t], er[t], A[t] = c
            prev = A[t].copy()
        Q[:, S.arm.qadr] = A
        sp = np.degrees(np.abs(np.diff(A, axis=0)).max(axis=1)) * float(fps) if T > 1 else np.zeros(1)
        rep[side] = {"mount": S.mount_source, "residual_mm_max": round(1000 * float(ep.max()), 2),
                     "residual_deg_max": round(float(np.degrees(er.max())), 2),
                     "frames_over_2mm_2deg": int(((ep > 0.002) | (er > np.radians(2.0))).sum()),
                     "worst_frame": int(np.argmax(ep + 0.05 * er)), "peak_step_deg_per_frame": round(float(sp.max()) / float(fps), 2),
                     "peak_speed_deg_s": round(float(sp.max()), 1), "speed_limited_frames": limited}
    return Q, rep


def weld_residuals(model, Q, sides=None, parts=None, bolt=None):
    """{side: (pos mm [T], rot deg [T], mount source)}: the hand root against where its own arm holds it."""
    parts = parts or _parts(model)
    out = {}
    for side in (sides or parts):
        pt = parts[side]
        if pt["gripper"] < 0 or not pt["arm_joints"]:
            continue
        S = ArmSide(model, side, parts=parts, bolt=(bolt or {}).get(side), Q_for_mount=Q)
        pe, re = np.zeros(len(Q)), np.zeros(len(Q))
        for t in range(len(Q)):
            pe[t], re[t] = S.residual(Q[t])
        out[side] = (pe, re, S.mount_source)
    return out


class ArmProbe:
    """Kinematic collision check of the arms: every visible arm link (as its convex hull) against the table, the
    objects' collision geoms, the other station's arm and both hands' collision geoms (palm and thumb hulls
    included). The flange and the hand bolted to it overlap by construction and are skipped, as are links of one
    arm with each other."""

    def __init__(self, scene_xml, tol_m=0.003):
        from tool_library.common.handgeom import add_hand_hulls
        spec = mujoco.MjSpec.from_file(str(scene_xml))
        add_hand_hulls(spec)
        plane = mujoco.mjtGeom.mjGEOM_PLANE
        for g in spec.geoms:
            body = g.parent.name if g.parent is not None else ""
            if not body.startswith("robot_"):
                continue
            is_arm = "_arm_" in body
            if (is_arm and int(g.group) <= 2 and g.type != plane and g.rgba[3] > 0) or (not is_arm and "_collision_hand_" in (g.name or "")):
                g.contype, g.conaffinity, g.condim, g.margin = 1, 1, 1, 0.0
            else:
                g.contype, g.conaffinity = 0, 0
        for p in list(spec.pairs):
            spec.delete(p)
        for e in list(spec.excludes):
            spec.delete(e)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.tol = float(tol_m)
        m = self.model
        self.kind, self.side = {}, {}
        for b in range(m.nbody):
            n = m.body(b).name or ""
            if n.startswith("robot_"):
                self.kind[b] = "arm" if "_arm_" in n else "hand"
                self.side[b] = "left" if n.startswith("robot_left") else "right"
            else:
                self.kind[b] = "table" if (n in ("table", "world") or "table" in n) else "object"
                self.side[b] = None

    def _mount_part(self, b):
        m, n = self.model, self.model.body(b).name or ""
        if "_arm_" in n:
            return n.endswith("gripper") or any((m.body(c).name or "").endswith("gripper") for c in range(m.nbody) if int(m.body_parentid[c]) == b)
        return "_base_" in n or n.endswith("_hand_C_MC") or n.endswith("_palm")

    def check(self, Q, labels=None, stride=2, frames=None, max_hits=6):
        m, d = self.model, self.data
        Q = np.asarray(Q, float)
        frames = list(range(0, len(Q), max(1, int(stride)))) if frames is None else list(frames)
        worst = {}
        for t in frames:
            d.qpos[:] = Q[t]
            mujoco.mj_forward(m, d)
            for i in range(d.ncon):
                c = d.contact[i]
                if c.dist >= -self.tol:
                    continue
                b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
                k1, k2 = self.kind[b1], self.kind[b2]
                if "arm" not in (k1, k2):
                    continue
                if k1 == k2 == "arm" and self.side[b1] == self.side[b2]:
                    continue
                if {k1, k2} == {"arm", "hand"} and self.side[b1] == self.side[b2] and self._mount_part(b1) and self._mount_part(b2):
                    continue
                other = k2 if k1 == "arm" else k1
                key = f"arm-{other}" if other != "arm" else "arm-arm"
                depth = -float(c.dist)
                if key not in worst or depth > worst[key]["depth_mm"] / 1000.0:
                    worst[key] = {"frame": int(t), "label": str(labels[t]) if labels is not None else "", "depth_mm": round(1000 * depth, 1),
                                  "geoms": [m.geom(c.geom1).name or m.body(b1).name, m.geom(c.geom2).name or m.body(b2).name]}
        return sorted(worst.values(), key=lambda h: -h["depth_mm"])[:max_hits]


def _held_waypoint_states(model, rig, q_grasp, waypoints):
    """Kinematic search states with a free object and hand moved together.

    These are hypothetical reach checks, never executable object trajectories.
    Object targets use the world pose of its free-joint body, not its bottom or
    collision-box centre. Intermediate orientations follow the shortest arc.
    """
    from scipy.spatial.transform import Rotation, Slerp
    from tool_library.common import plan as P
    data = mujoco.MjData(model)
    data.qpos[:] = q_grasp
    mujoco.mj_kinematics(model, data)
    wrist_pos, wrist_rot = P.wrist_world(model, data, rig)
    previous = {}
    states = []
    for index, waypoint in enumerate(waypoints or []):
        if not isinstance(waypoint, dict):
            offset = np.asarray(waypoint, float)
            if offset.shape != (3,) or not np.isfinite(offset).all():
                raise ValueError("arm_waypoints offsets must be three finite metres")
            data.qpos[:] = q_grasp
            P.set_wrist(model, data, rig, wrist_pos + offset, wrist_rot)
            states.append((f"waypoint {index}", data.qpos.copy()))
            continue
        unknown = set(waypoint) - {"object", "target_pos", "target_euler_deg", "steps"}
        if unknown or not {"object", "target_pos"} <= set(waypoint):
            raise ValueError("held-object waypoint requires object and target_pos; optional target_euler_deg and steps")
        name = waypoint["object"]
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(name))
        if bid < 1:
            raise ValueError(f"unknown held-object body: {name}")
        j = int(model.body_jntadr[bid])
        if (int(model.body_parentid[bid]) != 0 or int(model.body_jntnum[bid]) != 1
                or int(model.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_FREE)):
            raise ValueError("held-object waypoints require a world-root free body; articulated parts cannot be teleported")
        adr = int(model.jnt_qposadr[j])
        target_pos = np.asarray(waypoint["target_pos"], float)
        angles = np.asarray(waypoint.get("target_euler_deg", [0., 0., 0.]), float)
        steps = waypoint.get("steps", 8)
        if (target_pos.shape != (3,) or angles.shape != (3,) or not np.isfinite(target_pos).all()
                or not np.isfinite(angles).all() or isinstance(steps, bool) or not isinstance(steps, int)
                or not 1 <= steps <= 64):
            raise ValueError("held-object target poses must be finite 3-vectors and steps an integer in [1, 64]")
        data.qpos[:] = q_grasp
        mujoco.mj_kinematics(model, data)
        obj_pos = data.xpos[bid].copy()
        obj_rot = data.xmat[bid].reshape(3, 3).copy()
        local_pos = obj_rot.T @ (wrist_pos - obj_pos)
        local_rot = obj_rot.T @ wrist_rot
        prev_pos, prev_rot = previous.get(name, (obj_pos, obj_rot))
        target_rot = Rotation.from_euler("XYZ", angles, degrees=True).as_matrix() if "target_euler_deg" in waypoint else prev_rot
        rotation = Slerp([0., 1.], Rotation.from_matrix([prev_rot, target_rot]))
        for fraction in np.linspace(0., 1., steps + 1)[1:]:
            pos = prev_pos + fraction * (target_pos - prev_pos)
            rot = rotation(fraction).as_matrix()
            data.qpos[:] = q_grasp
            data.qpos[adr:adr + 3] = pos
            data.qpos[adr + 3:adr + 7] = TJ.mat2quat(rot)
            P.set_wrist(model, data, rig, pos + rot @ local_pos, rot @ local_rot)
            states.append((f"waypoint {index} ({fraction:.3f})", data.qpos.copy()))
        previous[name] = (target_pos, target_rot)
    return states


def screen_poses(model, side, poses, q_base, probe=None, arm_seed=None, cfg=None, parts=None, approach_m=0.08, waypoints=None):
    """Rank hand poses by whether the station's arm can hold them: for each pose (pos, euler, finger_targets; the
    grasp tools' format) the arm is solved at the grasp and at the pre-grasp (`approach_m` out along the pose's
    approach), and the arm (probe) is checked against the scene at both. Returns [{index, ok, residual_mm,
    residual_deg, arm_hits, reason}] in the input order; duplicates (same wrist pose within 1 mm / 1 deg) are marked.
    Waypoints accept legacy wrist translation offsets, or held-object target
    dictionaries consumed by _held_waypoint_states. This is a necessary reach
    screen; execution still checks continuous motion, contact and stability."""
    from tool_library.common import plan as P
    from tool_library.common.subgoal import load_pose
    cfg = {**DEFAULTS, **(cfg or {})}
    parts = parts or _parts(model)
    S = ArmSide(model, side, parts=parts, cfg=cfg)
    rig = P.scene_rigs(model)[side]
    seed = np.asarray(arm_seed if arm_seed is not None else S.home, float)
    out, seen = [], []
    for i, pose in enumerate(poses):
        try:
            pz = load_pose(pose)
            pos, eul = np.asarray(pz["pos"], float), np.asarray(pz["euler"], float)
        except Exception as e:
            out.append({"index": i, "ok": False, "reason": f"unreadable pose: {type(e).__name__}"})
            continue
        ap = P.retreat_axis(pz, TJ.euler_to_mat(eul))
        approach = float(approach_m if np.ndim(approach_m) == 0 else approach_m[i])
        shape = (pz.get("finger_targets") or {}, pz.get("finger_targets_open") or {})
        dup = next((j for j, (p2, e2, a2, dist2, shape2) in seen
                    if np.linalg.norm(p2 - pos) < 1e-3 and np.abs(e2 - eul).max() < np.radians(1.0)
                    and np.linalg.norm(a2 - ap) < 1e-6 and abs(dist2 - approach) < 1e-6 and shape2 == shape), None)
        seen.append((i, (pos, eul, ap, approach, shape)))
        if dup is not None:
            out.append({"index": i, "ok": False, "duplicate_of": dup, "reason": "duplicate wrist pose"})
            continue
        q = np.asarray(q_base, float).copy()
        d = S.data
        d.qpos[:] = q
        P.set_wrist(model, d, rig, pos, TJ.euler_to_mat(eul))
        rig.set_fingers(d, pz.get("finger_targets_open") or pz.get("finger_targets") or {})
        q_open = d.qpos.copy()
        rig.set_fingers(d, pz.get("finger_targets") or {})
        q_grasp = d.qpos.copy()
        q_pre = q_open.copy()
        d.qpos[:] = q_pre
        P.set_wrist(model, d, rig, pos + approach * ap, TJ.euler_to_mat(eul))
        q_pre = d.qpos.copy()
        worst, hits, q_arm = (0.0, 0.0), [], seed
        checks = [("pre-grasp", q_pre), ("grasp", q_grasp)]
        checks.extend(_held_waypoint_states(model, rig, q_grasp, waypoints))
        failed_at = None
        for label, qq in checks:
            p, R = S.flange_target(qq)
            ep, er, qa = S.solve(qq, p, R, q_arm, multistart="force")
            q_arm = qa
            worst = (max(worst[0], 1000 * ep), max(worst[1], float(np.degrees(er))))
            if probe is not None:
                qq2 = qq.copy(); qq2[S.arm.qadr] = qa
                hits += probe.check(qq2[None], frames=[0])
            if ep > cfg["project_pos_m"] or er > np.radians(cfg["project_rot_deg"]) or hits:
                failed_at = label
                break
        ok = worst[0] <= 1000 * cfg["project_pos_m"] and worst[1] <= cfg["project_rot_deg"] and not hits
        reach_ok = worst[0] <= 1000 * cfg["project_pos_m"] and worst[1] <= cfg["project_rot_deg"]
        reason = None if ok else (f"arm cannot hold it ({worst[0]:.1f} mm / {worst[1]:.1f} deg off)" if not reach_ok
                                  else "arm runs into " + "/".join(hits[0]["geoms"]) + f" ({hits[0]['depth_mm']} mm)")
        out.append({"index": i, "ok": bool(ok), "residual_mm": round(worst[0], 2), "residual_deg": round(worst[1], 2),
                    "arm_hits": hits[:2], "reason": reason, "family": pz.get("family"), "failed_at": failed_at,
                    "approach_m": approach})
    return out
