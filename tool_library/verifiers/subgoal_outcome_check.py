"""Did the executed episode achieve what a subgoal required?"""
from __future__ import annotations

TOOL = {
    "name": "subgoal_outcome_check",
    "title": "Subgoal outcome check",
    "type": "verifier",
    "category": "Task & Rollout Gates",
    "summary": "Measure a subgoal's outcome on the executed (physics) trajectory against its expectations: rise, in hand, "
               "no slip, at target, released, at rest, supported, upright, joint reached, turned, tilted, displaced, "
               "a rope severed, a press-fit part seated, untouched bystanders, penetration and servo tracking; per hand for "
               "a two-hand entry.",
    "description": (
        "The simulation-verifiable subgoal of the paper. Expectations come from plan_subgoal_keyframes (`expect`) or "
        "are written by hand: object (body), hand (side), rise_m, in_hand (+min_links), no_slip (+slip_from frame), "
        "target_pos (+target_axes; the tolerance is half the subgoal's planned displacement, start to target over all "
        "three axes, at least 3 mm and at most pos_tol_m, default 30 mm, so a short move must arrive; target_frac and "
        "pos_tol_min_m override the half and the 3 mm; measured target_err_m, planned_move_m, target_tol_m), "
        "target_quat_wxyz (a finite unit object orientation, checked within 15 degrees), released, at_rest, supported (+on), upright, tilt_deg, stays_put, joint "
        "(+joint_target | turned_rad | joint_min_travel), displacement {dir, min_m}, yaw_turned_deg, knot (a cable's "
        "centreline is knotted at the end, true = trefoil or a knot name; see rope_knot_check), severed (a rope cut by "
        "scissors: a join equality name, a list of them or true for any rope join; it passes when the episode's recorded "
        "eq_active shows the join whole at the subgoal's start and cut by its end -- `severed` -- and the two cut ends are "
        "at least sever_gap_m (default 3 mm) apart at the end -- `cut_ends_apart`; measured.severed gives each join's "
        "cut_frame and end_gap_mm; an episode without eq_active fails it with a hint), seated (press_seat: {axis, target, "
        "depth_m, tol_m, phase}; the part's root must end within tol_m of the target along the press axis; measured "
        "seat_travel_mm, seat_remaining_mm, peak_press_force_n), untouched [bodies], and for a two-hand entry hands "
        "{side: expect} (each hand's checks named '<side>:<check>', slip and pad frames read from that hand's part of "
        "the 'left: ... | right: ...' labels) with hands_apart (no hand-hand contact on any frame). Every "
        "expectation is measured by forward kinematics and contacts on the frames of the subgoal (all frames, a "
        "frame range, or the frames whose `subgoal` index matches), and the rollout diagnostics stored in the episode "
        "(hand penetration, object penetration, wrist tracking) are judged when present. The report names each "
        "failed expectation with the number behind it."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "Bench scene MJCF (the physics scene the episode was executed in, or the Stage 2 scene)."},
        {"name": "trajectory", "type": "str", "description": "Episode npz from rollout_keyframes_physics / execute_subgoal_plan."},
        {"name": "expect", "type": "dict", "description": "Expectations (see description)."},
        {"name": "subgoal", "type": "int | None", "description": "Evaluate the frames whose subgoal index equals this (needs the npz `subgoal` array)."},
        {"name": "frames", "type": "list[int] | None", "description": "[start, end] frame range instead."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok, failed [names], checks {name: bool}, measured {...}, frames."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "subgoal", "outcome", "gate"],
    "used_in": ["Open the Bottle", "Giftbox Packing", "Place Toy in Drawer", "Battery Insertion", "Place Cup in Bowl", "Computer Installation"],
    "status": "tested",
    "entrypoint": "subgoal_outcome_check",
}

import numpy as np

from tool_library.common import episode as E
from tool_library.common import outcome as O


def _resolve(expect, labels, idx):
    """Absolute frames for the phase-named references of one hand's expectation (the executor resolves the same labels
    on its segment): slip from the first lift / carry / pour / tilt / set-down frame, the pads where the plan grips."""
    if expect.get("no_slip") and "slip_from" not in expect and "slip_from_rel" not in expect:
        starts = [t for t in idx if any(w in str(labels[t]) for w in O.SLIP_WORDS)]
        expect["slip_from"] = starts[0] if starts else idx[0]
    if expect.get("pads_at_label") and "pads_at" not in expect and "pads_at_rel" not in expect:
        lab = str(expect["pads_at_label"])
        hits = [t for t in idx if str(labels[t]).split(" #")[0] == lab]
        expect["pads_at"] = hits[-1] if hits else idx[-1]
    st = expect.get("seated")
    if isinstance(st, dict) and st.get("phase") and "frames_rel" not in st:
        hits = [i for i, t in enumerate(idx) if str(labels[t]).split(" #")[0] == str(st["phase"])]
        if hits:
            expect["seated"] = dict(st, frames_rel=[hits[0], hits[-1]])
    return expect


def subgoal_outcome_check(scene, trajectory, expect, subgoal=None, frames=None):
    model, _ = E.build_execution_model(scene)
    ep = E.load_episode(trajectory)
    Q = ep["qpos"]
    if frames is not None:
        idx = list(range(int(frames[0]), int(frames[-1]) + 1))
    elif subgoal is not None and "subgoal" in ep:
        idx = [t for t, s in enumerate(np.asarray(ep["subgoal"])) if int(s) == int(subgoal)]
        if not idx:
            raise ValueError(f"no frames with subgoal index {subgoal}")
    else:
        idx = list(range(len(Q)))
    expect = dict(expect)
    if expect.get("hands"):
        # a two-hand entry: each hand's phase-named frames on its own part of the 'left: ... | right: ...' labels
        expect["hands"] = {s: _resolve(dict(e), E.side_view(ep["labels"], s), idx) for s, e in expect["hands"].items()}
    else:
        _resolve(expect, ep["labels"], idx)
    z = np.load(str(trajectory), allow_pickle=True)
    diag = None
    if all(k in z.files for k in ("hand_pen_m", "object_pen_m", "track_pos_m", "track_rot_deg")):
        diag = {k: np.asarray(z[k]) for k in ("hand_pen_m", "object_pen_m", "track_pos_m", "track_rot_deg", "wrist_force_n") if k in z.files}
    if "eq_active" in z.files and "eq_names" in z.files:
        # the equality state per frame (a scene with cut joins): what `severed` is judged on
        diag = dict(diag or {})
        diag.update({k: np.asarray(z[k]) for k in ("eq_active", "eq_names", "eq_active_start") if k in z.files})
    return O.evaluate(model, Q, ep["fps"], idx, expect, diagnostics=diag)
