"""Execute keyframes under contact physics with the hands' servos."""
from __future__ import annotations

TOOL = {
    "name": "rollout_keyframes_physics",
    "title": "Roll out keyframes in physics",
    "type": "skill",
    "category": "Trajectory & Execution",
    "summary": "Interpolate keyframes, couple the proposal to arm IK and execute a rigid flange-hand robot with bounded "
               "arm motor torques and finger servos. Record actual qpos, qvel and controls at every integration step.",
    "description": (
        "The physics counterpart of chain_subgoals. The scene is compiled with explicit contact pairs between every hand "
        "collision geom and every object geom (and between the two hands). From the start state, every frame's robot "
        "reference becomes bounded motor and finger commands. The simulation steps at its own timestep, and all "
        "integration states are kept; object columns of the reference are ignored. Hands are fixed children of the arm "
        "flanges, so no soft weld or post-simulation arm fitting is used. Per frame it records tracking error and the deepest "
        "hand-object/table interpenetration, the deepest object-object interpenetration and the hand-object contact "
        "set. A reference may be given instead of keyframes (a trajectory.npz from chain_subgoals)."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "Bench scene MJCF."},
        {"name": "keyframes", "type": "list | str | None", "description": "Keyframes (or plan json from plan_subgoal_keyframes). Either this or reference."},
        {"name": "reference", "type": "str | dict | None", "description": "A kinematic trajectory.npz to track instead of keyframes, or {qpos, labels} in memory."},
        {"name": "qpos_start", "type": "str | list | None", "description": "Physics start state (npz last frame or list); default the first keyframe / reference frame."},
        {"name": "fps", "type": "float", "description": "Default 30."},
        {"name": "max_rate_deg_s", "type": "float", "description": "Joint-rate ceiling for stretching transits. Default 90."},
        {"name": "max_rate_m_s", "type": "float", "description": "Slide-rate ceiling (the wrist). Default 0.4."},
        {"name": "wrist_kp", "type": "float", "description": "Wrist servo stiffness. Default 10000."},
        {"name": "arm", "type": "str | dict", "description": "'coupled' (default): solve arm IK for the proposal. A dictionary {side: [T,6]} supplies joint references to real torque motors; hand targets must agree with flange FK."},
        {"name": "out_npz", "type": "str | None", "description": "Write proposal diagnostics here and authoritative actual robot states/controls to the sibling *_runtime.npz. execute_subgoal_plan assembles the full runtime scene/trajectory pair."},
        {"name": "subgoal_index", "type": "int | None", "description": "Written per frame into the npz for bookkeeping."},
        {"name": "max_frames", "type": "int", "description": "Refuse references longer than this (a stretched leg from a bad start state). Default 3000."},
        {"name": "allowed_bodies", "type": "list[str] | None", "description": "Bodies the hands may touch in the plan (the handled part, held objects); contact with any other object or the other hand deeper than 4 mm in the kinematic plan is reported as plan_collisions."},
        {"name": "finger_force_nm", "type": "float | None", "description": "Finger servo torque cap for this rollout (default the library's 0.8)."},
        {"name": "wrist_force_n", "type": "float | list | dict | None", "description": "Wrist force cap, scalar or [fx, fy, fz] (default 40 N; a light lateral cap such as [10, 10, 40] for fine work like unscrewing), or per hand {side: cap | {cap, phases}}: {cap, phases} applies cap only on the frames whose label (that hand's part of a two-hand label) is one of `phases` (press_seat's press), the library's 40 N elsewhere; a hand not named keeps 40 N."},
        {"name": "wrist_torque_nm", "type": "float | None", "description": "Wrist torque cap (default 3 N m)."},
        {"name": "eq_inactive", "type": "list[str] | None", "description": "Equality constraints already released by earlier segments (kept off; a connector latch named here starts disarmed until its plug has left the port)."},
        {"name": "latched", "type": "dict | None", "description": "{latch weld name: relpose [7]}: connector latches the previous segment left engaged (its result's `latched`), switched on at the start with that relpose."},
        {"name": "rules", "type": "list[dict] | None", "description": "Mechanism rules applied during the rollout: [{engaged: equality name, lift_joint: joint, travel_m: thread length, frac: 0.96}] releases a screw cap's engagement weld once its lift joint has travelled the thread; {kind: 'cut', cutter: scissors cutter with its scene prefix (scissors.cutter_from_recon), joins: [{eq, body, anchor_local, diameter_m}]} switches a rope's join off when scissors.try_sever measures the blades closing on it (checked every physics step; cord-scaled gates); {kind: 'latch', eq: weld, site_a: plug tip site, site_b: port floor site, seat_m: 0.0015, release_N: 15} switches a connector latch weld on (at the plug's current pose, no snap) when the plug tip has bottomed out within seat_m of the port floor and no longer presses on it, and off when the pull-out load exceeds release_N (add_connector_latch). A scene with equalities also gets eq_active [T, neq] (the state after each frame), eq_active_start and eq_names in the npz; the result lists `cuts` [{equality, frame, cut}] and `latch_events` [{equality, frame, event: engage | release}]."},
    ],
    "outputs": [
        {"name": "result", "type": "dict", "description": "n_frames, duration_s, final_qpos, max_hand_pen_mm, max_object_pen_mm, max_track_pos_mm, max_track_rot_deg, stretched legs, contacts_end, eq_inactive, released, latched (the engaged latches for the next segment), latch_events, out_npz, scene_physics_xml."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "physics", "execution", "servo"],
    "used_in": ["Open the Bottle", "Giftbox Packing", "Place Toy in Drawer", "Battery Insertion", "Place Cup in Bowl", "Computer Installation"],
    "status": "tested",
    "entrypoint": "rollout_keyframes_physics",
}

import json
from pathlib import Path

import numpy as np

from tool_library.common import episode as E
from tool_library.skills.chain_subgoals import chain_subgoals


def _pregrasp_hits(model, Qref, labels):
    """Frames before the hand closes (reach, approach, descend) where the plan already drives a hand into ANY object,
    the target included: a reach that knocks the pipette over before the fingers close is a plan error the physics
    then shows as a tipped object."""
    pre = [i for i, l in enumerate(labels) if any(w in str(l) for w in (": reach", ": approach", ": descend"))]
    if not pre:
        return []
    import numpy as _np
    hits = E.plan_collision_sweep(model, _np.asarray(Qref)[pre], allowed_bodies=[], tol_m=0.006)
    for h in hits:
        h["frame"] = int(pre[h["frame"]]); h["kind"] = "pre-grasp " + h["kind"]
    return hits


def _force_schedule(wrist_force_n, labels, T, default):
    """{side: {cap, phases}} -> {side: [T, 3]} per-frame caps (cap on the frames of those phases, `default` elsewhere);
    other forms pass through unchanged."""
    if not isinstance(wrist_force_n, dict) or not any(isinstance(v, dict) for v in wrist_force_n.values()):
        return wrist_force_n
    out = {}
    for side, v in wrist_force_n.items():
        if not isinstance(v, dict):
            out[side] = v
            continue
        cap = np.broadcast_to(np.asarray(v.get("cap", default), float), (3,))
        phases = {str(p).split(" #")[0] for p in (v.get("phases") or [])}
        view = E.side_view(labels, side)
        F = np.full((int(T), 3), float(default))
        for t in range(min(int(T), len(view))):
            if str(view[t]).split(" #")[0] in phases:
                F[t] = cap
        out[side] = F
    return out


def rollout_keyframes_physics(scene, keyframes=None, reference=None, qpos_start=None, fps=30.0, max_rate_deg_s=90.0,
                              max_rate_m_s=0.4, wrist_kp=10000.0, arm="coupled", out_npz=None, subgoal_index=None, model=None, return_diag=False,
                              eq_inactive=(), rules=(), max_frames=3000, finger_force_nm=None, wrist_force_n=None, wrist_torque_nm=None,
                              allowed_bodies=None, latched=None, qvel_start=None, state_start=None, mechanism_state_start=None):
    if model is None:
        model, _ = E.build_execution_model(scene)
    if isinstance(keyframes, (str, Path)):
        kf = json.loads(Path(keyframes).read_text())
        keyframes = kf["keyframes"] if isinstance(kf, dict) else kf
    if keyframes is not None:
        ref = chain_subgoals(model, keyframes, fps=fps, max_rate_deg_s=max_rate_deg_s, max_rate_m_s=max_rate_m_s)
        Qref, labels, stretched = ref["qpos"], ref["labels"], ref["stretched"]
    elif isinstance(reference, dict):            # an in-memory reference (armhand.couple's output)
        Qref, labels, stretched = np.asarray(reference["qpos"], float), list(reference["labels"]), list(reference.get("stretched", []))
    elif reference is not None:
        ep = E.load_episode(reference); Qref, labels, stretched = ep["qpos"], ep["labels"], []
    else:
        raise ValueError("give keyframes or reference")
    if arm == "coupled":
        from tool_library.common import armhand as AH, rigid_robot as RR
        if not RR.is_rigid(model):
            coupled = AH.couple(model, Qref, fps, labels=labels, max_frames=max_frames)
            if not coupled["report"]["ok"]:
                raise ValueError(f"hand reference is not reachable by the rigid robot: {coupled['report']['infeasible'][:3]}")
            Qref, labels, arm = coupled["qpos"], coupled["labels"], coupled["arm"]
    if len(Qref) > int(max_frames):
        raise ValueError(f"the reference is {len(Qref)} frames ({len(Qref) / fps:.0f} s) long, above max_frames={max_frames}: a leg was stretched "
                         f"to respect the joint-rate ceiling (stretched legs: {stretched[:3]}); the start state or a keyframe is far off, "
                         "or the plan asks for a very long motion")
    if isinstance(qpos_start, (str, Path)):
        qpos_start = E.state_qpos(model, qpos_start)
    from tool_library.common import hand as H
    wrist_force_n = _force_schedule(wrist_force_n, labels, len(Qref), H.WRIST_FORCE_N)
    r = E.servo_rollout(model, Qref, fps, qpos_start=None if qpos_start is None else np.asarray(qpos_start, float), wrist_kp=wrist_kp, arm=arm,
                        eq_inactive=eq_inactive, rules=rules, latched=latched,
                        finger_force_nm=H.FINGER_FORCE_NM if finger_force_nm is None else finger_force_nm,
                        wrist_force_n=H.WRIST_FORCE_N if wrist_force_n is None else wrist_force_n,
                        wrist_torque_nm=H.WRIST_TORQUE_NM if wrist_torque_nm is None else wrist_torque_nm, qvel_start=qvel_start,
                        state_start=state_start, mechanism_state_start=mechanism_state_start)
    Q = r["qpos"]
    labels = labels[:len(Q)]
    res = {"n_frames": int(len(Q)), "duration_s": round(len(Q) / float(fps), 2), "final_qpos": Q[-1].tolist(),
           "max_hand_pen_mm": round(1000 * float(r["hand_pen_m"].max()), 2), "max_object_pen_mm": round(1000 * float(r["object_pen_m"].max()), 2),
           "max_track_pos_mm": round(1000 * float(r["track_pos_m"].max()), 1), "max_track_rot_deg": round(float(r["track_rot_deg"].max()), 1),
           "peak_wrist_force_n": round(float(r["wrist_force_n"].max()), 1), "peak_wrist_torque_nm": round(float(r["wrist_torque_nm"].max()), 2),
           "stretched": stretched, "contacts_end": r["contacts"][-1] if r["contacts"] else [], "labels": sorted(set(labels), key=labels.index),
           "eq_inactive": r["eq_inactive"], "released": r["released"], "unstable": r["unstable"],
           "latched": r.get("latched", {}), "latch_events": r.get("latch_events", []),
           "mechanism_final": r.get("mechanism_final"),
           "plan_collisions": E.plan_collision_sweep(model, Qref, allowed_bodies=allowed_bodies if allowed_bodies is not None else [])
                              + _pregrasp_hits(model, Qref, labels)}
    if "cuts" in r:
        res["cuts"] = r["cuts"]
    if out_npz:
        Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
        # a scene with cut joins also carries the equality state per frame (eq_active [T, neq] after each frame, the
        # start state and the column names): the cut is run-time state no qpos holds, and a replay must apply it
        eq = ({"eq_active": r["eq_active"], "eq_active_start": r["eq_active_start"], "eq_names": np.array(r["eq_names"])}
              if "eq_active" in r else {})
        E.save_episode(out_npz, Q, labels, fps, reference=Qref, subgoal=np.full(len(Q), -1 if subgoal_index is None else int(subgoal_index)),
                       hand_pen_m=r["hand_pen_m"], object_pen_m=r["object_pen_m"], track_pos_m=r["track_pos_m"], track_rot_deg=r["track_rot_deg"],
                       wrist_force_n=r["wrist_force_n"], wrist_torque_nm=r["wrist_torque_nm"], hand_object_force_n=r["hand_object_force_n"],
                       joint_names=np.array(E.joint_names(model)), **eq)
        res["out_npz"] = str(out_npz)
        if "robot_schema" in r:
            from tool_library.common import rigid_robot as RR
            arrays = RR.episode_arrays(model, r, Qref)
            arrays.update(labels=np.array(labels), fps=float(fps), subgoal=np.full(len(Q), -1 if subgoal_index is None else int(subgoal_index)))
            runtime_path = Path(out_npz).with_name(Path(out_npz).stem + "_runtime.npz")
            np.savez_compressed(runtime_path, **arrays)
            res["runtime_npz"] = str(runtime_path)
            res["robot_schema"] = RR.SCHEMA
    if return_diag:
        res["_diag"] = r
        res["_labels"] = labels
    return res
