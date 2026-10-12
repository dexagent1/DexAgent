"""Hinge drive check: did a hand (or finger) DRIVE a hinged part through its turn in an executed episode?"""
from __future__ import annotations

TOOL = {
    "name": "hinge_drive_check",
    "title": "Hinge drive check",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "On an executed episode, a hinged part (a lid, a flap, a door) turned through a required angle while a "
               "holder (a hand) was on it, without falling back, so the turn was driven by the hand rather than flung "
               "or dropped.",
    "description": (
        "The physics proof for a part moved about its hinge by a hand that does not grasp it: fingers hooked under a "
        "lid, a flat hand lifting a flap, a fingertip pushing a door. Such contact is one-sided and intermittent, so "
        "a hold-force window (contact_force_check) under-reads it and a floating-hand turn test cannot reproduce the "
        "arm-carried hand. The check replays the window frame by frame in the scene as the bench executor runs it "
        "(hand-object contact pairs added; a scene without robot hands uses its own collision geometry), counts a "
        "frame as touching when any holder collision geom is within `contact_tol_m` of the part (contact margins "
        "raised to the tolerance for the replay), and measures the part's joint: its travel in the driven direction "
        "(inferred from the net change unless `direction` is given), the largest fall-back below its best progress, "
        "the share of the forward travel made on frames with the holder on the part or within `lag_frames` after "
        "(the driven fraction), the share of touching frames and the longest stretch without contact. Passes when "
        "the travel reaches `min_travel_deg`, the fall-back stays under `max_backslide_deg` and the driven fraction "
        "reaches `min_driven_frac`. Issues name what failed and what to change."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF scene the trajectory belongs to (a bench scene, or any scene with the holder and the part)."},
        {"name": "trajectory", "type": "str | ndarray [T, nq]", "description": "trajectory.npz (qpos, labels, fps) or a qpos array."},
        {"name": "joint", "type": "str", "description": "The hinge (or slide) the part moves on."},
        {"name": "holder", "type": "str", "description": "Root body of the holder (its subtree counts: a hand with its fingers)."},
        {"name": "frames", "type": "list[int] | list[str] | None", "description": "Window: frame indices, or phase-label patterns. Default all frames."},
        {"name": "direction", "type": "int | None", "description": "+1 or -1: the sign of the joint change the hand drives. Default: the sign of the net change."},
        {"name": "min_travel_deg", "type": "float", "description": "Required travel in the driven direction (degrees, or mm for a slide). Default 20."},
        {"name": "max_backslide_deg", "type": "float", "description": "Largest fall-back below the best progress (degrees / mm). Default 5."},
        {"name": "min_driven_frac", "type": "float", "description": "Required share of the forward travel made while the holder touched the part. Default 0.8."},
        {"name": "contact_tol_m", "type": "float", "description": "Gap below which the holder counts as on the part. Default 0.003."},
        {"name": "lag_frames", "type": "int", "description": "Frames after a touching frame whose travel still counts as driven (the part's momentum). Default 2."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; measured {frames, direction, joint_start, joint_end, travel, best, backslide and its frame, driven_frac, contact_frac, longest_gap_frames, unit}."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["hinge", "lid", "flap", "door", "non-prehensile", "push", "hook", "physics", "audit"],
    "used_in": ["Giftbox Packing"],
    "status": "tested",
    "entrypoint": "hinge_drive_check",
}

import copy

import mujoco
import numpy as np

from tool_library.common import trajectory as TJ


def _load(scene):
    if isinstance(scene, mujoco.MjModel):
        return copy.copy(scene)                         # the margins are raised below: never on the caller's model
    try:
        from tool_library.common import episode as E
        return E.build_execution_model(str(scene))[0]
    except ValueError:                                  # no robot hands in it: the scene's own collision geometry
        return TJ.load_model(scene)


def hinge_drive_check(scene, trajectory, joint, holder, frames=None, direction=None, min_travel_deg=20.0,
                      max_backslide_deg=5.0, min_driven_frac=0.8, contact_tol_m=0.003, lag_frames=2):
    if direction not in (None, 1, -1):
        raise ValueError("direction must be +1, -1 or None (inferred)")
    if not 0.0 < float(min_driven_frac) <= 1.0:
        raise ValueError("min_driven_frac must lie in (0, 1]")
    if float(contact_tol_m) < 0 or float(contact_tol_m) > 0.02:
        raise ValueError("contact_tol_m must lie in [0, 0.02]")
    if int(lag_frames) < 0:
        raise ValueError("lag_frames must be >= 0")
    model = _load(scene)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    if jid < 0:
        raise ValueError(f"joint {joint!r} is not in the scene")
    jt = int(model.jnt_type[jid])
    if jt not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
        raise ValueError(f"joint {joint!r} is neither a hinge nor a slide")
    hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, holder)
    if hid < 0:
        raise ValueError(f"holder body {holder!r} is not in the scene")
    Q, labels, fps = TJ.as_trajectory(trajectory)
    if Q.shape[1] != model.nq:
        raise ValueError(f"trajectory width {Q.shape[1]} != scene nq {model.nq}: give the scene the episode was executed in")
    if frames is not None and len(frames) and isinstance(next(iter(frames)), str):
        frames = TJ.frames_matching(labels, list(frames))
    frames = list(range(len(Q))) if frames is None else [int(t) for t in frames]
    if len(frames) < 2:
        raise ValueError("frames: the window needs at least two frames")
    if min(frames) < 0 or max(frames) >= len(Q):
        raise ValueError(f"frames outside the trajectory (0..{len(Q) - 1})")
    is_hinge = jt == int(mujoco.mjtJoint.mjJNT_HINGE)
    scale = np.degrees(1.0) if is_hinge else 1000.0
    unit = "deg" if is_hinge else "mm"
    qa = int(model.jnt_qposadr[jid]); part = int(model.jnt_bodyid[jid])
    hb = TJ.subtree(model, hid); pb = TJ.subtree(model, part)
    # near contacts count: raise the margins of every pair / geom that can touch holder-part to the tolerance
    tol = float(contact_tol_m)
    for i in range(model.npair):
        b1, b2 = int(model.geom_bodyid[model.pair_geom1[i]]), int(model.geom_bodyid[model.pair_geom2[i]])
        if (b1 in hb and b2 in pb) or (b2 in hb and b1 in pb):
            model.pair_margin[i] = max(float(model.pair_margin[i]), tol)
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) in hb or int(model.geom_bodyid[g]) in pb:
            model.geom_margin[g] = max(float(model.geom_margin[g]), tol)
    data = mujoco.MjData(model)
    q = np.zeros(len(frames)); touch = np.zeros(len(frames), bool); gap = np.full(len(frames), np.inf)
    for k, t in enumerate(frames):
        data.qpos[:] = Q[t]; data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        q[k] = float(data.qpos[qa])
        for i in range(data.ncon):
            c = data.contact[i]
            b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            if (b1 in hb and b2 in pb) or (b2 in hb and b1 in pb):
                gap[k] = min(gap[k], float(c.dist))
                if c.dist <= tol:
                    touch[k] = True
    net = q[-1] - q[0]
    sgn = int(direction) if direction is not None else (1 if net >= 0 else -1)
    prog = sgn * (q - q[0]) * scale                      # progress in the driven direction
    best = np.maximum.accumulate(prog)
    back = best - prog
    kb = int(np.argmax(back))
    dp = np.diff(prog)
    near = np.array([touch[max(0, k - int(lag_frames)):k + 2].any() for k in range(len(dp))])
    fwd = np.clip(dp, 0.0, None)
    driven = float(fwd[near].sum() / fwd.sum()) if fwd.sum() > 1e-9 else 0.0
    gaps, run = [], 0
    for v in touch:
        run = 0 if v else run + 1
        gaps.append(run)
    measured = {"frames": [int(frames[0]), int(frames[-1])], "n_frames": len(frames), "direction": sgn, "unit": unit,
                "joint_start": round(float(q[0] * scale), 2), "joint_end": round(float(q[-1] * scale), 2),
                "travel": round(float(prog[-1]), 2), "best": round(float(best[-1]), 2),
                "backslide": round(float(back[kb]), 2), "backslide_frame": int(frames[kb]),
                "backslide_label": str(labels[frames[kb]]) if labels is not None and len(labels) > frames[kb] else None,
                "driven_frac": round(driven, 3), "contact_frac": round(float(touch.mean()), 3),
                "longest_gap_frames": int(max(gaps) if gaps else 0),
                "min_gap_mm": round(float(np.min(gap)) * 1000, 2) if np.isfinite(gap).any() else None,
                "contact_tol_m": tol, "lag_frames": int(lag_frames)}
    issues = []
    if prog[-1] < float(min_travel_deg):
        issues.append(f"the part travelled {prog[-1]:.1f} {unit} in the driven direction (need {min_travel_deg}): the hand did not "
                      f"move it far enough; check the window, the direction, or drive the contact farther along the hinge arc")
    if back[kb] > float(max_backslide_deg):
        issues.append(f"the part fell back {back[kb]:.1f} {unit} below its best progress at frame {frames[kb]}: it was flung ahead "
                      f"of the hand or dropped off it; move the contact farther from the free edge or slow the sweep")
    if driven < float(min_driven_frac):
        issues.append(f"only {100 * driven:.0f}% of the travel happened with the holder on the part (need {100 * float(min_driven_frac):.0f}%): "
                      f"the part moved on its own (batted or falling); keep the hand on it through the sweep")
    return {"ok": not issues, "issues": issues, "measured": measured}
