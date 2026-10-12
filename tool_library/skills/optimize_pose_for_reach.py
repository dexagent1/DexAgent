"""Turn a hand pose about its own contact so the station's arm can reach it, keeping the contact."""
from __future__ import annotations

TOOL = {
    "name": "optimize_pose_for_reach",
    "title": "Optimize pose for reach",
    "type": "skill",
    "category": "Grasp & Poses",
    "summary": "Search rotations of a hand pose about an axis that preserves its contact (the pinch line between the pads, "
               "the surface normal of a push, or the object's vertical) for the orientation the station's arm can reach "
               "with the smallest residual, optionally along a planned joint sweep or a set of waypoints, and return the "
               "re-oriented pose with its reach margins.",
    "description": (
        "A grasp is a contact set, not a wrist orientation: a pinch may be turned about the line through its pads and a "
        "fingertip push about its surface normal without changing what touches what, and a wrap about the object's axis "
        "within limits. The arm, however, reaches only some orientations at a given place. This skill takes a "
        "pose from any grasp tool, the bench scene and side, picks the contact axis (`axis`: 'pinch' uses the two grip "
        "pads, 'normal' the palm normal (the pads' contact normal for a push), 'vertical' the world z, or an explicit [x, y, z] through "
        "`pivot`), samples both signs and every enabled axis in `step_deg` steps before larger turns within their ranges, then descends on every enabled axis (the primary, "
        "the tilt about the pads' line, an opt-in roll) until the worst residual stops improving, and for each candidate solves the arm "
        "(position and the home-pose mount rotation, multi-start) at the pose and, when `sweep` is given, at the pose "
        "carried through a joint sweep (joint name, from, to, in the bench scene, with the hand following rigidly or "
        "level) or through `waypoints` (world offsets). Candidates whose turned hand penetrates the scene (a pinky into a box wall), or whose arm holds it only by running into the table, an object or its own hand (the executor's arm screen, armhand.ArmProbe: every arm link's hull at the solved arm), at the pose, along the path or on the way in (the pre-grasp `approach_m` back along the pose's approach, where the executor's reach ends, and half way down), are infeasible, and so is one whose pre-grasp the arm cannot reach within the transit tolerance. Objects are where `state` puts them (default the hand scene's recorded bench_state, else the keyframe); rebuild the hand scene from the episode so far once parts have moved. On the sweep, among the "
        "feasible candidates that reach, the smallest turn wins of those whose worst-case residual is within 1 mm (2 deg of "
        "tilt) of the best, so a pose that reaches unturned stays unturned (when none reaches, the best worst-case residual "
        "wins); the descent from there moves only for a candidate that reaches where it does not, or for a residual lower "
        "by more than that. Its finger "
        "targets are kept, its approach vector is rotated (kept along the surface normal, or in the surface, for a one-sided "
        "contact). Pose anchors follow the current materialized part as in the executor; returned poses carry their new "
        "anchor to prevent a second transform. A hand scene's recorded bench_state is the default state, and an explicitly "
        "inconsistent object state is rejected. co_sides requires actual contact with the moving part in this state; a "
        "parked hand or a hand holding its stationary base cannot be attached to the moving part. "
        "Screen the complete intended manipulation, not just the initial contact: use a joint sweep for mechanisms "
        "or held-object waypoint dictionaries for subsequent translation AND rotation. A rigid grasp follows the part's "
        "orientation; level following keeps wrist orientation and follows the contact centroid, so contact must be proved "
        "again during physical execution. This finite sampled screen is necessary, not a continuous-motion or task-success "
        "certificate. Re-verify the result with the stress tests: a rotated pinch may seat differently."
        " With axis='normal' and parallel coplanar recorded contacts, the pivot/tilt line uses the surface points, "
        "the supporting plane normals stay fixed, and rotations that move recorded points off the plane are rejected. "
        "When the target part is identified by sweep or anchor, every named grip finger must also remain within "
        "max_contact_gap_m of its collision geometry at every nominal path state; metadata alone cannot prove contact."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "Bench scene MJCF (reach is a property of the station)."},
        {"name": "side", "type": "str", "description": "'left' or 'right'."},
        {"name": "pose", "type": "dict | str", "description": "Pose dict or json from a grasp tool (pos, euler, finger_targets, grip_fingers, approach)."},
        {"name": "hand_scene", "type": "dict | str | None", "description": "The hand scene the pose was solved in (to locate the pads for the 'pinch' axis); default: pads from the bench scene at the pose."},
        {"name": "axis", "type": "str | list", "description": "'pinch' | 'normal' | 'vertical' | [x, y, z]. Default 'pinch' when the pose has two or more grip fingers, else 'normal'."},
        {"name": "pivot", "type": "list | None", "description": "Point the axis passes through (default: the grip pads' centroid, or the wrist)."},
        {"name": "range_deg", "type": "float", "description": "Half-range of the search. Default 60."},
        {"name": "step_deg", "type": "float", "description": "Default 10."},
        {"name": "sweep", "type": "dict | None", "description": "{joint, to, hand_follows: 'rigid'|'level', steps: 6, co_sides: []}: evaluate both moving-part geometry and wrist motion. At least one sample per 10 degrees / 1 cm. Optional from must match state. co_sides names hands already holding this moving part in state; their wrists and arms are screened too."},
        {"name": "waypoints", "type": "list | None", "description": "World wrist offsets, or {object, target_pos, target_euler_deg, steps} for held free-body transforms (same convention as optimize_grasp_pose arm_waypoints). Include downstream orientation/placement targets when choosing a grasp."},
        {"name": "tilt_axis", "type": "str | list | None", "description": "Second search axis through the pivot: 'pads_line' (the line through the pads; a push keeps its pads on the surface when the hand tips about it), 'fingers', [x, y, z] or None. Default 'auto' = pads_line for a push, none for a pinch."},
        {"name": "tilt_range_deg", "type": "float", "description": "Default 30."},
        {"name": "roll_axis", "type": "str | list | None", "description": "Third axis, opt-in ('fingers' or [x, y, z]): a roll lifts the outer pads, keep it small. Default None."},
        {"name": "roll_range_deg", "type": "float", "description": "Default 15."},
        {"name": "pos_tol_m", "type": "float", "description": "Reach tolerance. Default 0.004 (the executor holds the hand on the flange within 6 mm / 4 deg and fails beyond)."},
        {"name": "max_overlap_m", "type": "float", "description": "Maximum hand-object penetration at every sampled state, including pads and allowed contact bodies. Default 0.003."},
        {"name": "max_contact_gap_m", "type": "float", "description": "For recorded planar contacts (axis='normal') against an identified part, maximum grip-finger collision-geometry gap at every nominal state. Default 0.003. Does not establish contact force."},
        {"name": "tilt_tol_deg", "type": "float", "description": "Default 3."},
        {"name": "out_json", "type": "str | None", "description": "Write the chosen pose here."},
        {"name": "state", "type": "str | list | None", "description": "The scene state candidates are checked against: a trajectory.npz (its last frame) or a qpos. Defaults to hand_scene.bench_state when recorded, otherwise the scene keyframe. Rebuild the hand scene from the episode so far when a part has moved; an explicit contradictory object state is rejected."},
        {"name": "approach_m", "type": "float", "description": "Also screen the arm at the pre-grasp this far back along the pose's approach (and half way): the executor reaches in from there. Default 0.08 (the executor's grasp_lift); 0 disables."},
        {"name": "time_budget_s", "type": "float | None", "description": "Search budget in seconds (120). Finishes the current candidate's complete path, then returns the best completed candidate with budget_reached; no witness means inconclusive. None permits an unbounded search."},
    ],
    "outputs": [
        {"name": "result", "type": "dict", "description": "pose (re-oriented, same finger targets), rotation_deg {primary, tilt, roll} applied, axes, pivot, reach {worst_pos_err_mm, worst_rot_err_deg, ok, overlap_mm, overlap_with, arm_hit, pregrasp_pos_err_mm}, original_reach, candidates of the axis seed sweep [{angles, worst_pos_err_mm, worst_rot_err_deg, ok, arm_hit}], n_evaluated, note (when no state was given)."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "reach", "arm", "pose", "optimization"],
    "used_in": ["Giftbox Packing", "Place Toy in Drawer", "Open the Bottle", "Computer Installation"],
    "status": "tested",
    "entrypoint": "optimize_pose_for_reach",
}

import json
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from tool_library.common import episode as E
from tool_library.common import hand as H
from tool_library.common import plan as P


def _load(x):
    if isinstance(x, (str, Path)):
        return json.loads(Path(x).read_text())
    return x


def _reach_solver(model, side):
    """Exact arm IK of the flange for a hand pose (armhand: 6-DoF, warm-started, seed library over the joint box), the
    same solver execute_subgoal_plan couples the arm with, so a pose ranked reachable here is one the executor holds."""
    from tool_library.common import armhand as AH
    S = AH.ArmSide(model, side)
    q0 = np.array(model.key_qpos[0] if model.nkey else model.qpos0, float)

    def solve(p, R, q_prev=None, pos_tol=0.005, rot_tol=np.radians(3.0)):
        R = np.asarray(R, float)
        pt_, Rt = np.asarray(p, float) + R @ S.p_m, R @ S.R_m
        prev = S.home if q_prev is None else np.asarray(q_prev, float)
        # Try the continuous local solution first. A failed full multistart
        # followed by another forced multistart repeated almost the entire
        # expensive search at every sampled contact and sweep waypoint.
        ep, er, qa = S.solve(q0, pt_, Rt, prev, multistart=False)
        if ep > pos_tol or er > rot_tol:
            ep2, er2, qa2 = S.solve(q0, pt_, Rt, prev, multistart="force")
            if ep2 + 0.05 * er2 < ep + 0.05 * er:
                ep, er, qa = ep2, er2, qa2
        return (float(ep), float(er), qa)
    solve.arm_qadr = S.arm.qadr
    return solve


EQUAL_SCORE = 0.001        # reachable candidates whose scores (residual in m + 0.5 mm per degree) differ less are equally
                           # good: the smallest turn among them is kept


def _unit(v):
    v = np.asarray(v, float); return v / max(np.linalg.norm(v), 1e-9)


def _rotation_seeds(axes, step_deg):
    """Visit both signs and every enabled axis before larger turns.

    A primary grid ordered from -range to +range can spend a finite budget
    entirely on one sign without ever trying a useful tilt. All candidates
    still receive the complete nominal and commanded motion screen.
    """
    seeds = []
    for k, (_, _, extent) in enumerate(axes):
        magnitudes = list(np.arange(float(step_deg), float(extent), float(step_deg)))
        if extent > 0:
            magnitudes.append(float(extent))
        for magnitude in magnitudes:
            for sign in (1., -1.):
                angles = [0.] * len(axes)
                angles[k] = sign * magnitude
                seeds.append((magnitude, k, sign < 0, angles))
    return [entry[-1] for entry in sorted(seeds, key=lambda x: x[:3])]


def _anchor_to_state(model, data, pose, hand_scene=None):
    """Use the planner's materialized-part anchor, then stamp the new frame.

    Without the new anchor a returned pose would move twice when the executor
    subsequently reanchors it using the original hand scene.
    """
    from tool_library.common.subgoal import pose_anchor, reanchor_pose
    anchor = pose_anchor({"hand_scene": hand_scene}, pose)
    if not anchor:
        return dict(pose)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(anchor.get("body", "")))
    if body < 1:
        raise ValueError("pose anchor body is not present in the supplied bench scene")
    mujoco.mj_forward(model, data)
    out = dict(reanchor_pose(model, data, pose, anchor))
    quat = Rot.from_matrix(data.xmat[body].reshape(3, 3)).as_quat()
    out["anchor"] = {"body": model.body(body).name, "pos": data.xpos[body].tolist(),
                     "quat_wxyz": [float(quat[3]), *map(float, quat[:3])]}
    return out


def _check_co_holders(model, data, sweep):
    """A following hand must already contact this moving part in the state.

    co_sides is not an instruction to transport an idle hand from home on a
    fictitious rigid attachment. This is an input precondition, not a reach
    verdict or proof that a contact can sustain a physical load.
    """
    if not sweep or not sweep.get("co_sides"):
        return {}
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(sweep.get("joint", "")))
    if jid < 0:
        raise ValueError("co_sides requires an existing mechanism joint")
    bodies = E.subtree_bodies(model, int(model.jnt_bodyid[jid]))
    objects = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies
               and (model.geom_contype[g] or model.geom_conaffinity[g])]
    mujoco.mj_forward(model, data)
    gaps = {}
    for side in sweep["co_sides"]:
        hand = [g for g in range(model.ngeom) if f"_collision_hand_{side}_" in (model.geom(g).name or "")]
        gap = min((H.geom_distance(model, data, h, o, .05) for h in hand for o in objects), default=float("inf"))
        if not np.isfinite(gap) or gap > .01:
            raise ValueError(f"co_sides {side!r} is not in contact with the moving part in state "
                             f"(gap {1000 * gap:.1f} mm). Supply the current state with the co-holder seated; "
                             "a parked hand or a hand steadying the stationary base must not follow the moving part")
        gaps[side] = round(1000 * float(gap), 3)
    return gaps


def _mechanism_command_states(model, rigs, side, q_touch, pose, sweep):
    """The planner's commanded clamp and joint sweep (not a contact simulation).

    Finger servo overtravel is a force command. Its collision depth is not a
    measured physical penetration; nominal touch geometry is screened separately.
    The rigid wrist must nevertheless reach its actual commanded clamp path.
    """
    rig = rigs[side]
    clamp = H.executor_clamp(rig, pose)
    d = mujoco.MjData(model); d.qpos[:] = q_touch; mujoco.mj_forward(model, d)
    p, R = P.wrist_world(model, d, rig)
    if clamp.get("wrist_press"):
        P.set_wrist(model, d, rig, p + np.asarray(clamp["wrist_press"]), R)
    rig.set_fingers(d, clamp["press"]); mujoco.mj_forward(model, d)
    if pose.get("points"):
        follow = np.mean(list(pose["points"].values()), axis=0)
    else:
        fingers = pose.get("grip_fingers") or rig.fingertip_sites
        follow = np.mean([d.site_xpos[rig.fingertip_sites[f]] for f in fingers if f in rig.fingertip_sites], axis=0)
    q = d.qpos.copy()
    return [("grip command", q)] + [("command " + label, state) for label, state in
                                    _joint_sweep_states(model, rigs, side, q, sweep, follow)]


def _joint_sweep_states(model, rigs, side, q_pose, sweep, follow_point):
    """Hypothetical contact-following states, never an executable trajectory.

    Keep the moved joint and every following wrist in the *same* state, so
    collision checks cannot accidentally use the mechanism's initial pose.
    The follow transforms match Planner.move_joint, including co-holders.
    """
    unknown = set(sweep) - {"joint", "from", "to", "steps", "hand_follows", "co_sides"}
    if unknown or not {"joint", "to"} <= set(sweep):
        raise ValueError("sweep requires joint and to; optional from, steps, hand_follows, co_sides")
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(sweep["joint"]))
    if jid < 0 or int(model.jnt_type[jid]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
        raise ValueError("sweep joint must be an existing hinge or slide")
    mode = sweep.get("hand_follows", "rigid")
    if mode not in ("rigid", "level"):
        raise ValueError("hand_follows must be 'rigid' or 'level'; search each mode explicitly")
    co = sweep.get("co_sides", [])
    if not isinstance(co, (list, tuple)) or any(s not in rigs or s == side for s in co) or len(set(co)) != len(co):
        raise ValueError("co_sides must name distinct other hands already holding the moving part")
    adr, part = int(model.jnt_qposadr[jid]), int(model.jnt_bodyid[jid])
    q0, target = float(q_pose[adr]), float(sweep["to"])
    if not np.isfinite(target) or ("from" in sweep and (not np.isfinite(float(sweep["from"])) or abs(float(sweep["from"]) - q0) > 1e-6)):
        raise ValueError("sweep target must be finite and from must match the supplied scene state")
    if model.jnt_limited[jid] and not model.jnt_range[jid, 0] - 1e-9 <= target <= model.jnt_range[jid, 1] + 1e-9:
        raise ValueError("sweep target is outside the mechanism's joint limits")
    n = sweep.get("steps", 6)
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or not 1 <= n <= 512:
        raise ValueError("sweep steps must be an integer in [1, 512]")
    spacing = np.radians(10.) if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_HINGE) else .01
    n = max(int(n), int(np.ceil(abs(target - q0) / spacing)), 1)
    if n > 512:
        raise ValueError("sweep needs more than 512 samples; split the physical motion")
    d = mujoco.MjData(model); d.qpos[:] = q_pose; mujoco.mj_forward(model, d)
    pp0, Rp0 = d.xpos[part].copy(), d.xmat[part].reshape(3, 3).copy()
    followers = {}
    for s in [side, *co]:
        p, R = P.wrist_world(model, d, rigs[s])
        if mode == "level":
            c = np.asarray(follow_point, float) if s == side else np.mean([d.site_xpos[i] for i in rigs[s].fingertip_sites.values()], axis=0)
            rel_p, rel_R = P.relative_transform(pp0, Rp0, c, R)
            followers[s] = (rel_p, rel_R, c - p, R)
        else:
            rel_p, rel_R = P.relative_transform(pp0, Rp0, p, R)
            followers[s] = (rel_p, rel_R, None, R)
    out = []
    for i in range(1, n + 1):
        d.qpos[:] = q_pose
        d.qpos[adr] = q0 + (target - q0) * i / n
        mujoco.mj_forward(model, d)
        pp, Rp = d.xpos[part].copy(), d.xmat[part].reshape(3, 3).copy()
        for s, (rel_p, rel_R, off, R0) in followers.items():
            p, R = P.compose(pp, Rp, rel_p, rel_R)
            P.set_wrist(model, d, rigs[s], p - off if off is not None else p, R0 if off is not None else R)
        out.append((f"sweep {i}/{n} ({mode})", d.qpos.copy()))
    return out


def optimize_pose_for_reach(scene, side, pose, hand_scene=None, axis=None, pivot=None, range_deg=60.0, step_deg=10.0, sweep=None, waypoints=None,
                            tilt_axis="auto", tilt_range_deg=30.0, roll_axis=None, roll_range_deg=15.0,
                            pos_tol_m=0.004, tilt_tol_deg=3.0, max_overlap_m=0.003, out_json=None, state=None, approach_m=0.08,
                            time_budget_s=120.0, max_contact_gap_m=0.003):
    started = time.monotonic()
    if time_budget_s is not None and (not np.isfinite(time_budget_s) or time_budget_s <= 0):
        raise ValueError("time_budget_s must be positive and finite, or None")
    if not np.isfinite(step_deg) or step_deg <= 0 or not np.isfinite(range_deg) or range_deg < 0:
        raise ValueError("step_deg must be positive and range_deg nonnegative, both finite")
    if not np.isfinite(max_contact_gap_m) or max_contact_gap_m < 0:
        raise ValueError("max_contact_gap_m must be finite and nonnegative")
    deadline = float('inf') if time_budget_s is None else started + float(time_budget_s)
    budget_reached = False
    pose = _load(pose)
    model = E.build_execution_model(str(scene))[0]       # hand-object contacts on, so a candidate's overlap is measurable
    rigs = P.scene_rigs(model)
    if side not in rigs:
        raise ValueError(f"side must be one of {list(rigs)}")
    rig = rigs[side]
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    from tool_library.skills.hand_scene_from_recon import resolve_bench_state
    q = resolve_bench_state(hand_scene, model, state)
    if q is not None:                         # the exact state used to construct the hand scene, or explicit state
        data.qpos[:] = q
    pose = _anchor_to_state(model, data, pose, hand_scene)
    co_holder_gaps = _check_co_holders(model, data, sweep)
    p0, R0 = P.pose_from_dict(pose)
    # the pads at the pose (on the bench, the same frame as the hand scene)
    P.set_wrist(model, data, rig, p0, R0); rig.set_fingers(data, {k: v for k, v in pose["finger_targets"].items() if k in rig.finger_qpos}); mujoco.mj_forward(model, data)
    grip = list(pose.get("grip_fingers") or [])
    pads = {f: data.site_xpos[rig.fingertip_sites[f]].copy() for f in grip if f in rig.fingertip_sites}
    pts = np.array(list(pads.values())) if pads else None
    cen = pts.mean(axis=0) if pads else p0.copy()
    # what rides the part in a level sweep: the contact points when the pose records them, else the pads
    follow0 = np.mean([np.asarray(v, float) for v in pose["points"].values()], axis=0) if pose.get("points") else cen.copy()
    if axis is None:
        axis = "pinch" if ("thumb" in pads and len(pads) >= 2) else "normal"
    if isinstance(axis, str):
        if axis == "pinch":
            if len(pads) < 2:
                raise ValueError("axis 'pinch' needs at least two grip fingers in the pose")
            if "thumb" in pads:
                others = np.array([v for f, v in pads.items() if f != "thumb"]).mean(axis=0); ax = others - pads["thumb"]
            else:
                ax = pts[0] - pts[-1]
        elif axis == "normal":
            if pose.get("normals"):                   # the pads' contact normal (a push keeps its pads on the surface when it turns about it)
                ax = -np.mean([_unit(v) for v in pose["normals"].values()], axis=0)
                if np.linalg.norm(ax) < 0.5:
                    raise ValueError("axis 'normal' needs one-sided contacts (parallel normals); use 'pinch' for an opposed grasp")
            else:
                ax = R0[:, 0].copy()                  # local x = palm normal, the best guess without recorded normals
        elif axis == "vertical":
            ax = np.array([0, 0, 1.0])
        else:
            raise ValueError("axis must be 'pinch', 'normal', 'vertical' or [x, y, z]")
    else:
        ax = np.asarray(axis, float)
    ax = _unit(ax)
    # In the planar contact family, turning/tilting the hand does not turn
    # the object's plane. These normals drive the executor's wrist press;
    # rotating them with a tilted hand commands pressure off the real normal.
    fixed_surface_normals, surface_pts, plane_origin, plane_normal = False, None, None, None
    if isinstance(axis, str) and axis == "normal" and pose.get("normals") and pose.get("points"):
        ns = np.array([_unit(v) for v in pose["normals"].values()])
        ps = np.array(list(pose["points"].values()), float)
        nn, origin = _unit(ns.mean(axis=0)), ps.mean(axis=0)
        if np.max(np.linalg.norm(ns - nn, axis=1)) < 1e-5 and np.max(np.abs((ps - origin) @ nn)) < 1e-5:
            fixed_surface_normals, surface_pts, plane_origin, plane_normal = True, ps, origin, nn
    piv = np.asarray(pivot, float) if pivot is not None else plane_origin if fixed_surface_normals else cen
    tilt_points = surface_pts if fixed_surface_normals else pts
    # the secondary axes: the line through the pads (a push keeps its pads on the surface when the hand tips about
    # it) and the fingers' direction (a roll, which lifts the outer pads: opt-in, small)
    axes = [("primary", ax, float(range_deg))]
    if tilt_axis == "auto":
        tilt_axis = "pads_line" if (tilt_points is not None and len(tilt_points) >= 2 and axis != "pinch") else None
    if tilt_axis is not None:
        if isinstance(tilt_axis, str):
            if tilt_axis == "pads_line":
                i, j = np.unravel_index(np.argmax(np.linalg.norm(tilt_points[:, None] - tilt_points[None], axis=2)), (len(tilt_points), len(tilt_points)))
                t_ax = tilt_points[i] - tilt_points[j]
            elif tilt_axis == "fingers":
                t_ax = R0[:, 2].copy()
            else:
                raise ValueError("tilt_axis must be 'pads_line', 'fingers', [x, y, z] or None")
        else:
            t_ax = np.asarray(tilt_axis, float)
        t_ax = _unit(t_ax); t_ax = _unit(t_ax - ax * (ax @ t_ax)) if abs(ax @ t_ax) < 0.99 else None
        if t_ax is not None:
            axes.append(("tilt", t_ax, float(tilt_range_deg)))
    if roll_axis is not None:
        r_ax = _unit(R0[:, 2].copy() if roll_axis == "fingers" else np.asarray(roll_axis, float))
        axes.append(("roll", r_ax, float(roll_range_deg)))
    from tool_library.common import armhand as AH
    solvers = {side: _reach_solver(model, side)}
    # Validate before allocating additional arm solvers (and before searching).
    if sweep:
        _joint_sweep_states(model, rigs, side, data.qpos, sweep, follow0)
        for s in sweep.get("co_sides", []):
            solvers[s] = _reach_solver(model, s)

    def path_states(p, R, Rd):
        d = mujoco.MjData(model); d.qpos[:] = data.qpos
        P.set_wrist(model, d, rig, p, R)
        q = d.qpos.copy()
        out = [("pose", q)]
        out.extend(AH._held_waypoint_states(model, rig, q, waypoints))
        if sweep:
            out.extend(_joint_sweep_states(model, rigs, side, q, sweep, piv + Rd @ (follow0 - piv)))
        return out

    def command_states(p, R, Rd):
        if not sweep:
            return []
        candidate = dict(pose, pos=p.tolist(), euler=H.mat_to_euler(R).tolist())
        if pose.get("points"):
            candidate["points"] = {f: (piv + Rd @ (np.asarray(v) - piv)).tolist() for f, v in pose["points"].items()}
        if pose.get("normals"):
            candidate["normals"] = {f: (np.asarray(v) if fixed_surface_normals else Rd @ np.asarray(v)).tolist()
                                    for f, v in pose["normals"].items()}
        d = mujoco.MjData(model); d.qpos[:] = data.qpos
        P.set_wrist(model, d, rig, p, R)
        return _mechanism_command_states(model, rigs, side, d.qpos, candidate, sweep)

    cache = {}
    class SearchExpired(Exception):
        pass
    hand_side = {g: s for g in range(model.ngeom) for s in rigs if f"_collision_hand_{s}_" in (model.geom(g).name or "")}
    hand_geoms = {g for g, s in hand_side.items() if s in solvers}
    d_ov = mujoco.MjData(model)
    contact_objects, contact_fingers = [], {}
    if fixed_surface_normals:
        part = (int(model.jnt_bodyid[model.joint(sweep["joint"]).id]) if sweep else
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str((pose.get("anchor") or {}).get("body", ""))))
        if part > 0:
            bodies = E.subtree_bodies(model, part)
            paired = set(map(int, model.pair_geom1)) | set(map(int, model.pair_geom2))
            contact_objects = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies
                               and (model.geom_contype[g] or model.geom_conaffinity[g] or g in paired)]
            contact_fingers = {f: [g for g in rig.collision_gids
                                  if f"_{f}_" in (model.body(int(model.geom_bodyid[g])).name or "")]
                                  or list(rig.tip_geoms(f))
                               for f in (grip or pose["points"])}
            if contact_objects and any(not geoms for geoms in contact_fingers.values()):
                raise ValueError("planar grip fingers must name hand collision geometry")
    probe = AH.ArmProbe(str(scene))                   # the executor's arm screen: links vs table, objects, both hands
    d_arm = mujoco.MjData(model)

    def approach_of(Rd):
        """The retreat direction (pads to wrist) the turned pose carries, as the result below writes it."""
        a0 = P.retreat_axis(pose, R0)
        a = Rd @ a0
        if pose.get("approach") is not None and pose.get("normals"):
            nbar = np.mean([_unit(v) for v in pose["normals"].values()], axis=0)
            if np.linalg.norm(nbar) > 0.5:
                nbar = _unit(nbar); nbar_new = nbar if fixed_surface_normals else _unit(Rd @ nbar)
                a = nbar_new * np.sign(a0 @ nbar) if abs(a0 @ nbar) > 0.9 else a - nbar_new * (nbar_new @ a)
        return _unit(a)

    def overlap(q):
        """Contact permissions do not exempt pads or held bodies from the depth bound."""
        d_ov.qpos[:] = q; mujoco.mj_forward(model, d_ov)
        worst, what = 0.0, None
        for i in range(d_ov.ncon):
            c = d_ov.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 not in hand_geoms and g2 not in hand_geoms:
                continue
            if g1 in hand_side and g2 in hand_side and hand_side[g1] == hand_side[g2]:
                continue
            hg, og = (g1, g2) if g1 in hand_geoms else (g2, g1)
            ob = int(model.geom_bodyid[og])
            if (model.body(ob).name or "").startswith("robot_") and og not in hand_side:
                continue
            if -c.dist > worst:
                worst, what = -float(c.dist), (model.geom(hg).name, model.geom(og).name)
        return worst, what

    transit_pos, transit_rot = AH.DEFAULTS["transit_pos_m"], np.radians(AH.DEFAULTS["transit_rot_deg"])

    def finger_gap(geoms):
        # Bounding spheres only discard pairs provably farther away than an
        # already measured gap. Every surviving near pair uses real geometry;
        # a bounding-sphere overlap is never counted as contact.
        cap = max(.05, max_contact_gap_m + .01)
        delta = d_ov.geom_xpos[geoms, None, :] - d_ov.geom_xpos[None, contact_objects, :]
        lower = np.linalg.norm(delta, axis=2) - model.geom_rbound[geoms, None] - model.geom_rbound[None, contact_objects]
        lower[:, model.geom_type[contact_objects] == int(mujoco.mjtGeom.mjGEOM_PLANE)] = -np.inf
        distance = cap
        for flat in np.argsort(lower, axis=None):
            i, j = np.unravel_index(flat, lower.shape)
            if lower[i, j] >= distance:
                break
            distance = min(distance, H.geom_distance(model, d_ov, geoms[i], contact_objects[j], cap))
        return float(distance)

    def evaluate(angles):
        key = tuple(round(a, 3) for a in angles)
        if key in cache:
            return cache[key]
        # Finish each candidate's complete path before reporting a witness.
        # Never turn a partially checked sweep into a passing reach result.
        if cache and time.monotonic() >= deadline:
            raise SearchExpired
        Rd = np.eye(3)
        for (name, a_ax, _), deg in zip(axes, angles):
            # A planar tilt is about the contact line after the primary
            # heading turn, not the old line elsewhere on the surface.
            actual_axis = Rd @ a_ax if fixed_surface_normals and name == "tilt" else a_ax
            Rd = Rot.from_rotvec(np.radians(deg) * actual_axis).as_matrix() @ Rd
        p, R = piv + Rd @ (p0 - piv), Rd @ R0
        plane_error = (float(np.max(np.abs(((surface_pts - piv) @ Rd.T + piv - plane_origin) @ plane_normal)))
                       if fixed_surface_normals else 0.)
        worst = (0.0, 0.0); qs = {}; hits = []; q_grasp = None
        ov, what, overlap_at = 0.0, None, None
        gap, gap_at, gap_finger = 0., None, None
        states = path_states(p, R, Rd)
        for label, qq in states:
            d_arm.qpos[:] = qq; mujoco.mj_forward(model, d_arm)
            reachable = True
            for s, solve in solvers.items():
                pp, RR = P.wrist_world(model, d_arm, rigs[s])
                e = solve(pp, RR, qs.get(s)); qs[s] = e[2]
                d_arm.qpos[solve.arm_qadr] = e[2]
                worst = (max(worst[0], e[0]), max(worst[1], e[1]))
                reachable &= e[0] <= pos_tol_m and e[1] <= np.radians(tilt_tol_deg)
            q_grasp = dict(qs) if q_grasp is None else q_grasp
            if not hits and reachable:
                hits = [dict(h, where=label) for h in probe.check(d_arm.qpos[None].copy(), frames=[0])]
            depth, pair = overlap(qq)
            if depth > ov:
                ov, what, overlap_at = depth, pair, label
            if contact_objects:
                # overlap() has already forwarded this exact nominal state.
                # Check all declared fingers, not a nearby palm/idle thumb.
                for f, geoms in contact_fingers.items():
                    distance = finger_gap(geoms)
                    if distance > gap:
                        gap, gap_at, gap_finger = float(distance), label, f
        # Match the executor's actual wrist commands as well as the nominal
        # contact geometry. These are separate scopes: servo overtravel is not
        # physical finger penetration, and only execution can prove the contact.
        commands = command_states(p, R, Rd)
        qs_command = dict(q_grasp)
        for label, qq in commands:
            d_arm.qpos[:] = qq; mujoco.mj_forward(model, d_arm)
            reachable = True
            for s, solve in solvers.items():
                pp, RR = P.wrist_world(model, d_arm, rigs[s])
                e = solve(pp, RR, qs_command.get(s)); qs_command[s] = e[2]
                d_arm.qpos[solve.arm_qadr] = e[2]
                worst = (max(worst[0], e[0]), max(worst[1], e[1]))
                reachable &= e[0] <= pos_tol_m and e[1] <= np.radians(tilt_tol_deg)
            if not hits and reachable:
                hits = [dict(h, where=label) for h in probe.check(d_arm.qpos[None].copy(), frames=[0])]
        # the way in: the executor's reach ends at the pre-grasp (approach_m back along the pose's approach, a transit
        # pose) and descends from there; an arm link that sweeps through a neighbour on the way fails in the executor
        pre_err = None
        reach_ok = worst[0] <= pos_tol_m and worst[1] <= np.radians(tilt_tol_deg)
        if approach_m and float(approach_m) > 0 and reach_ok:       # an unreachable pose is ranked by its residual alone
            a_w = approach_of(Rd)
            q2 = q_grasp[side]
            solve = solvers[side]
            for frac in np.linspace(1., 0., max(2, int(np.ceil(float(approach_m) / .02))) + 1)[:-1]:
                pp = p + a_w * float(approach_m) * frac
                e = solve(pp, R, q2); q2 = e[2]
                pre_err = (max((pre_err or (0., 0.))[0], e[0]), max((pre_err or (0., 0.))[1], e[1]))
                d_arm.qpos[:] = states[0][1]
                P.set_wrist(model, d_arm, rig, pp, R)
                opened = pose.get("finger_targets_open") or H.open_targets(rig, pose["finger_targets"], float(pose.get("open_frac", .55)))
                rig.set_fingers(d_arm, opened)
                for s, solver in solvers.items():
                    d_arm.qpos[solver.arm_qadr] = e[2] if s == side else q_grasp[s]
                if e[0] <= transit_pos and e[1] <= transit_rot and not hits:
                    hits = [dict(h, where="pre-grasp" if frac == 1.0 else "approach") for h in probe.check(d_arm.qpos[None].copy(), frames=[0])]
                depth, pair = overlap(d_arm.qpos)
                if depth > ov:
                    ov, what, overlap_at = depth, pair, "approach"
        pre_bad = pre_err is not None and (pre_err[0] > transit_pos or pre_err[1] > transit_rot)
        if hits and ov <= max_overlap_m:
            what = ("arm", "/".join(hits[0]["geoms"]) + f" {hits[0]['depth_mm']} mm" + (f" at the {hits[0]['where']}" if hits[0].get("where") else ""))
        elif pre_bad and ov <= max_overlap_m:
            what = ("arm", f"pre-grasp {1000 * pre_err[0]:.0f} mm / {np.degrees(pre_err[1]):.0f} deg out of reach")
        feasible = ov <= max_overlap_m and not hits and not pre_bad and plane_error <= 1e-5 and gap <= max_contact_gap_m
        rec = {"angles": list(key), "worst_pos_err_mm": round(1000 * worst[0], 1), "worst_rot_err_deg": round(float(np.degrees(worst[1])), 1),
               "overlap_mm": round(1000 * ov, 1), "overlap_with": what, "overlap_at": overlap_at, "n_path_states": len(states),
               "n_command_states": len(commands),
               "surface_point_error_mm": round(1000 * plane_error, 3),
               "contact_gap_mm": round(1000 * gap, 3) if contact_objects else None,
               "contact_gap_at": gap_at, "contact_gap_finger": gap_finger,
               "arm_hit": ({"geoms": hits[0]["geoms"], "depth_mm": hits[0]["depth_mm"], "where": hits[0].get("where", "pose")} if hits else None),
               "pregrasp_pos_err_mm": None if pre_err is None else round(1000 * pre_err[0], 1),
               "ok": bool(feasible and worst[0] <= pos_tol_m and worst[1] <= np.radians(tilt_tol_deg)), "pos": p.tolist(), "R": R,
               "score": worst[0] + 0.5 * worst[1] / np.radians(1) / 1000 + (10.0 if not feasible else 0.0)}
        cache[key] = rec
        return rec

    def turn(c):
        return float(np.sum(np.abs(c["angles"])))

    def pick(cs):
        """The grid candidate to start from: among those that reach (ok), the smallest turn of the ones whose score is
        within EQUAL_SCORE of the best (a pose reached unturned with 0 mm residual is not turned 30 or 60 deg for a few
        micrometres less); when none reaches, the lowest score, then the smallest turn."""
        oks = [c for c in cs if c["ok"]]
        if oks:
            s0 = min(c["score"] for c in oks)
            return min((c for c in oks if c["score"] <= s0 + EQUAL_SCORE), key=lambda c: (turn(c), c["score"]))
        return min(cs, key=lambda c: (c["score"], turn(c)))

    def improves(r, b):
        """The descent moves for a candidate that reaches where the current one does not, or for a clearly lower score
        (by EQUAL_SCORE once both reach): never for a few micrometres, and never toward the reach boundary just to turn
        less (a turn shaved from 10 to 5 deg next to an unreachable 0 deg seated the grasp 0.7 mm deeper)."""
        if r["ok"] != b["ok"]:
            return r["ok"]
        return r["score"] < b["score"] - (EQUAL_SCORE if r["ok"] else 0.0)

    # Symmetric seeds on every enabled axis, then coordinate descent from
    # the best completed seed. This remains useful when the budget is short.
    zeros = [0.0] * len(axes)
    orig = evaluate(zeros)
    cands = [orig]
    # Scores are nonnegative. At zero turn and within EQUAL_SCORE of zero,
    # no other candidate can win either the score or smallest-turn rule.
    already_optimal = orig["ok"] and orig["score"] <= EQUAL_SCORE
    if not already_optimal:
        for a in _rotation_seeds(axes, step_deg):
            try:
                cands.append(evaluate(a))
            except SearchExpired:
                budget_reached = True
                break
    best = pick(cands)
    cur = list(best["angles"]); step = float(step_deg)
    while not already_optimal and not budget_reached and step >= 2.0:
        improved = False
        for k, (name, _, rng_k) in enumerate(axes):
            for sgn in (+1, -1):
                trial = list(cur); trial[k] = float(np.clip(trial[k] + sgn * step, -rng_k, rng_k))
                if trial == cur:
                    continue
                try:
                    r = evaluate(trial)
                except SearchExpired:
                    budget_reached = True
                    break
                if improves(r, best):
                    best, cur, improved = r, trial, True
            if budget_reached:
                break
        if not improved:
            step /= 2
    orig = evaluate(zeros)
    Rd = best["R"] @ R0.T
    new = dict(pose); new["pos"] = best["pos"]; new["euler"] = H.mat_to_euler(best["R"]).tolist()
    # Recorded points turn about the pivot; a planar contact's surface normals
    # belong to its stationary supporting plane, not the tilted hand.
    if pose.get("points"):
        new["points"] = {f: (piv + Rd @ (np.asarray(v, float) - piv)).tolist() for f, v in pose["points"].items()}
    if pose.get("normals"):
        new["normals"] = {f: (np.asarray(v, float) if fixed_surface_normals else Rd @ np.asarray(v, float)).tolist()
                          for f, v in pose["normals"].items()}
    if pose.get("targets"):
        new["targets"] = {f: (piv + Rd @ (np.asarray(v, float) - piv)).tolist() for f, v in pose["targets"].items()}
    if pose.get("approach") is not None:
        a0 = _unit(pose["approach"]); a = Rd @ a0
        if pose.get("normals"):
            # the contact surface does not turn with the hand: an approach that came along the surface normal stays
            # along it, one that came along the surface stays in it
            nbar = np.mean([_unit(v) for v in pose["normals"].values()], axis=0)
            if np.linalg.norm(nbar) > 0.5:
                nbar = _unit(nbar); nbar_new = nbar if fixed_surface_normals else _unit(Rd @ nbar)
                if abs(a0 @ nbar) > 0.9:
                    a = nbar_new * np.sign(a0 @ nbar)
                else:
                    a = a - nbar_new * (nbar_new @ a)
        new["approach"] = _unit(a).tolist()
    rot = {name: round(a, 2) for (name, _, _), a in zip(axes, best["angles"])}
    checked_motion = {"sweep": dict(sweep, hand_follows=sweep.get("hand_follows", "rigid")) if sweep else None,
                      "waypoints": waypoints, "scope": "sampled_kinematics", "execution_required": True,
                      "collision_geometry": "nominal contact", "mechanism_reach": "nominal and commanded clamp",
                      "planar_contact_geometry": "plane and finger-to-part gaps" if contact_objects else "plane only" if fixed_surface_normals else "not assessed",
                      "max_contact_gap_m": float(max_contact_gap_m) if contact_objects else None,
                      "contact_gap_distance_cap_m": max(.05, float(max_contact_gap_m) + .01) if contact_objects else None,
                      "co_holder_contact_gaps_mm": co_holder_gaps}
    new["reach_optimized"] = {"rotation_deg": rot, "axes": {name: a_ax.round(4).tolist() for name, a_ax, _ in axes}, "pivot": piv.round(4).tolist(),
                              "ok": bool(best["ok"]), "checked_motion": checked_motion}
    strip = lambda c: {k: c[k] for k in ("angles", "worst_pos_err_mm", "worst_rot_err_deg", "overlap_mm", "overlap_with", "overlap_at", "arm_hit", "surface_point_error_mm", "contact_gap_mm", "ok")}
    keys = ("ok", "worst_pos_err_mm", "worst_rot_err_deg", "overlap_mm", "overlap_with", "overlap_at", "arm_hit", "pregrasp_pos_err_mm", "n_path_states", "n_command_states", "surface_point_error_mm", "contact_gap_mm", "contact_gap_at", "contact_gap_finger")
    res = {"pose": new, "rotation_deg": rot, "axes": {name: a_ax.round(4).tolist() for name, a_ax, _ in axes}, "pivot": piv.round(4).tolist(),
           "reach": {k: best[k] for k in keys}, "original_reach": {k: orig[k] for k in keys},
           "candidates": [strip(c) for c in cands], "n_evaluated": len(cache),
           "seconds": round(time.monotonic() - started, 3), "budget_reached": budget_reached,
           "search_status": "feasible_witness" if best["ok"] else "inconclusive", "checked_motion": checked_motion}
    if q is None:
        res["note"] = ("arm links were screened against the scene's keyframe; if objects have moved since (a stack built, a lid "
                       "opened), pass state= the episode so far (trajectory.npz) or the arm may still run into them in the executor")
    if out_json:
        Path(out_json).write_text(json.dumps(new, indent=1, default=float)); res["out_json"] = str(out_json)
    return res
