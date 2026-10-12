"""Plan one subgoal as keyframes on the bench scene."""
from __future__ import annotations

TOOL = {
    "name": "plan_subgoal_keyframes",
    "title": "Plan subgoal keyframes",
    "type": "skill",
    "category": "Trajectory & Execution",
    "summary": "Turn one Stage 1 subgoal (motion, hand, object, a verified hand pose, targets) into keyframes on the bench "
               "scene: reach, descend, close, lift, carry, set down, let go, joint sweeps, screw bites, pours, wrist presses "
               "that seat press-fit parts, two hands acting on two parts at once, one hand pinning a part while the other "
               "acts, two hands lifting one cable, a second object in the same hand, a cable link laid over or dragged across "
               "other strands; with the phases they form and what "
               "the simulation must show afterwards (`expect`).",
    "description": (
        "The planner starts from a state of the bench scene (its keyframe, or a frame of an executed trajectory so a "
        "subgoal continues from where the previous one really ended) and writes full-qpos keyframes with hold and "
        "transit durations. Motions follow Stage 1's vocabulary: grasp_lift (open hand to the pre-grasp along the pose's "
        "approach, down to the pose, close, lift), hold (the same without the lift), carry / place / insert (the carried "
        "object is moved rigidly with the wrist to target_pos, or onto / over another body; place adds a set-down, a "
        "let-go and a retreat), release, open_joint / close_joint / pull / push / press with a joint (the hand follows the "
        "moving part while the joint sweeps to its target: `hand_follows` 'rigid' = the wrist rides the part like a pinch "
        "that turns with a lid; 'level' = the wrist keeps its orientation and the pads of the grip fingers, or a world "
        "`follow_point`, ride the part, for fingertips that lift a lid from below or hook a drawer, since an arm cannot "
        "roll its hand 100 degrees with a lid and a level hand stays in reach), push / pull / press without a joint (a straight "
        "stroke from a pose along a direction), unscrew / screw (bites of bite_deg about the yaw joint with the lift "
        "joint following the pitch, opening and turning back between bites), pour / tilt (rotation about an axis "
        "through a pivot, by default the object's top over the target; `keep` leaves the object turned), cut (with scissors "
        "already held: the open blades are carried, keeping their orientation, so a rope's join (`join`: its equality name "
        "from Stage 2 cut_joins, or `cut_point`) comes in from beyond the tips and sits on the blades' bisector at "
        "`cut_s_frac` of the blade length, then the gripping fingers close the blades -- pressed `close_depth_rad` past the "
        "holding grip, or to explicit `finger_targets_close`; expect: severed + the scissors still in hand; without a join "
        "it just shuts them, expect joint_target closed. The rope parts only through the executor's cut rule), press_seat (a "
        "press-fit part pushed home by the WRIST: a palm or flat-finger pose on the part, the wrist reference driven travel_m "
        "past the contact while the wrist force cap on that axis is force_n, at most the library's 40 N; expect seated = the "
        "part's root at its seat height within seat_tol_m), the thin-part moves (common/nonprehensile.py; thin_grasp_diagnosis "
        "says which one a part needs): slide (upright fingertips behind the part push it along the table, mode 'push', or "
        "pads pressed on its top drag it with a force-capped wrist press, mode 'drag'; a slide_contact_pose; how far: "
        "`overhang_m` past the table's edge along `direction` (refused when its centre of mass would come within 12 mm of "
        "the edge), `distance_m` or `target_pos`; expect displaced, overhang, upright, released), edge_pinch / "
        "overhang_grasp (an overhang_pinch_pose: grip fingers under the overhanging edge deeper than the thumb on top, "
        "the hand coming down beside the part and in under it; refused, with the slide that fixes it, when the part "
        "overhangs less than the pose needs; then a grasp_lift), tilt_up / pivot (a press pose on the overhanging end: "
        "pressed down far enough that the far end rises `lift_m` about the table's edge, the press force capped at "
        "`force_n`; with `hold` (default) the hand keeps pressing into the next entry, where the other hand closes on the "
        "raised end (`hold` with an overhang_pinch_pose solved with raised true, grip 'sides': the raised end's side faces "
        "stand a pad radius above the table, under it there is room for fingers but not a palm), then this hand's "
        "`release` and the other hand's grasp_lift (no pose: it holds); expect end_raised. A side pinch of a raised end "
        "holds the part's weight only by the pads' torsional friction: on parts under about 1 cm it rarely passes the "
        "stress ladder, and overhang_pinch_pose says so; slide + edge_pinch is the move that holds), hook (a hook_grasp_pose: fingers down through an opening, the thumb on "
        "the wall's outside, straight down and lift; without the thumb it pulls along the pose's pull_dir while it "
        "rises), strip_pinch (a pinch across an arch of a flat strip: the thumb on the strip where it arches over "
        "something, a fingertip slid in under the arch, then lift; no pose tool holds one yet, so the pose comes "
        "from handle_pinch_pose or pose_from_points, stress-tested); each of them expects no finger more than 2 mm into the table "
        "(table_clear), regrasp, home, wait, and both (two hands on two different parts "
        "in one rollout: {motion: 'both', hands: {left: entry, right: entry}, sync: 'start'|'end'|'keyframes'}; each hand's "
        "entry is planned from the same start, the plans are merged on one time line, labels join the two phases "
        "'left: ... | right: ...', expect = each hand's own plus untouched bystanders and hands_apart), pin (one hand steadies "
        "a part, or one link or end of a cable, while the OTHER hand acts: a press pose (flat_hand_pose pads down, "
        "push_pose_from_point, pin_pose) comes down on it and the WRIST presses force_n (default 20 N, never above the 40 N "
        "cap) from where the posed hand first meets it, with the wrist force on that axis capped at force_n; a grip pose "
        "(a pinch, a wrap) closes on it instead. The pin lasts through the following entries (the executor keeps the "
        "pinning hand's commands, press reference and force cap) until a release of that hand, and every one of those "
        "entries expects `pinned`: the part within tol_m (default 2 mm) and rot_tol_deg (3) of where the pin left it on "
        "EVERY frame, the pinning hand still on it; a pinning hand may only release, go home or wait), lift_together "
        "({hands: {left: {object, pose, hand_scene}, right: {...}}, lift_m}: both hands close on two links of ONE object, a "
        "heavy cable, in the same keyframes and lift it by the same offsets, so a SPAN rises (the end can pass under it), "
        "the load is shared and the spacing kept; expect per hand rise, the link in hand (its pads on the link or its "
        "neighbours) and no_slip, plus `spacing` within spacing_tol_m, 2 cm). Each pose's approach_waypoints are visited; "
        "a hand with fewer waypoints waits open at its last point. t_reach / t_descend / t_close / t_grip may be set "
        "on lift_together or per hand; each synchronized stage uses the slower hand's duration. approach_controls "
        "reports the applied waypoint offsets and timings so repairs can be checked before physics. "
        "carry_together (both hands holding links of one object move by the same `offset`, or so their links' midpoint "
        "reaches `target_pos`; expect each link at its own target, in hand, the spacing kept), grasp_more (a hand that "
        "holds one object takes a second with its FREE fingers: the pose's grip_fingers, e.g. second_grasp_pose's middle, "
        "ring and pinky inside a cup, must not overlap the fingers holding the first, which keep their press; the hand "
        "carries the first object to the new pose and closes only the free fingers; `state.in_hand` records which fingers "
        "hold what, later carries, places and lifts move every object the hand holds and expect `also_in_hand`, and a "
        "release or place naming one of them opens only its fingers), lay (one over-move of a cable lying on the table, "
        "the moves plan_knot_moves plans to tie a knot: a pinch on one link, the link lifted just clear of the strands "
        "under its path (its underside 12 mm above the highest, at least 3 cm up, or `lift_m`), carried through `via` "
        "[[x, y], ...] to `target_pos` [x, y] (where the link's body origin lands), SET DOWN there, the fingers opened "
        "with the wrist still and the hand backed out along the pose's approach (a hand that moves while it opens drags "
        "the rope); expect the link at its target in x-y, released, at rest, `expect_order` (the crossing order the move "
        "must leave, knot.crossing_order's code, kinks dropped) and `knot` on the last move), drag (the same with the "
        "link held 8 mm off the table, so the rope slides: a pull-through or tuck of a tail along itself). Carry/place "
        "heights from on / over / into come from the bodies' real collision geometry (the support's highest surface "
        "under the carried part's footprint, studs included, and the part's own lowest point), never bounding spheres. A hand pose is a dict or "
        "a JSON path from the grasp skills; it may carry `finger_targets_open` (its own open shape, used instead of the 55% "
        "opening), `open_frac`, `press_depth_rad` (how far past the closed targets the grip fingers' flexion is driven; a "
        "negative value drives them that far toward extension instead, spreading what they hold open: scissor rings; the "
        "pose's finger_targets_squeezed still win where they flex deeper; a one-sided grip, no thumb opposing the grip fingers, "
        "presses by 4 mm of pad travel, and when that flexion would lift its pads off their surface, a fingers-down fingertip "
        "on a strip, the fingers stay as posed and the wrist presses 4 mm along the contact normal instead, and the hand reaches "
        "and descends with those grip fingers as posed rather than opened, since opening them pushes the pads into the "
        "surface) and `approach_waypoints` (world offsets from the pose visited between the "
        "pre-grasp and the pose, for a hook that goes over a bar and down behind it); the grip memory (closed, pressed and opened finger targets, what each hand "
        "holds) is carried in `state` between calls. The keyframes go to chain_subgoals / rollout_keyframes_physics; "
        "`expect` goes to subgoal_outcome_check. Labels are '<side>: <phase>' as Stage 4 wants them."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "Bench scene MJCF (Stage 2 scene.xml)."},
        {"name": "subgoal", "type": "dict", "description": "{index, motion, side, object (body name), pose (dict|json path), lift_m, approach_m, t_reach / t_descend / t_close / t_grip (positive seconds for grasp_lift, hold, articulated grasp approaches and grip-mode pin; defaults 2.0 / 0.8 / 0.5 / 0.3, so contact can be slowed after impulse feedback), t_open / t_retreat (positive release seconds, defaults 0.5 / 0.8; duration_s scales their total unless either is explicitly set), settle_s (nonnegative stationary physics after release, default 0.5; included in the release outcome and recorded episode), target_pos|on|over|into, offset, height_m, via_height, into (a container body: the carried object's lowest point lands on the container's highest surface under the object's footprint at its centre -- the inner floor, or the sloped wall where the object reaches it -- + `offset`, `clearance_m` 3 mm), on (a body: the carried object's lowest point lands 2 mm above the highest collision surface of that body under the object's footprint, studs and rims included, measured from the real geometry, plus `offset` [dx, dy, dz] in metres (no compensation offset is needed); `height_m` is the travel height of the carry, not the landing height; `set_down_s` (default 1.0) controls a place's final descent time, `placement_force_n` bounds its contact and release effort (default at least 10 N, enough for object weight plus 5 N, always within the configured wrist limit); `carry_rise_m` how far a place lifts the wrist above the higher of itself and the target on the way, default 0.05; `lift_dir` [x, y, z] (a grasp_lift's way out: the part drawn out from under an overhang toward the operator while it rises) or `lift_path` [[dx, dy, dz], ...] (offsets from the grip visited in order, e.g. out first, then up; the rise expected is half the last offset's height); `via_rise_m` a carry's rise above the higher of its two ends on the way (instead of an absolute via_height); `arm_path: 'joint'` plans a carry's or place's transfer (not its set-down) in arm joint space, so an arm parked in a configuration that cannot follow the straight carry changes configuration in the air) | over (the same target without the set-down), target_pos (a world position of the carried object's root), target_euler_deg or turn {axis, deg} (a carried object's orientation at the destination; carries and places keep it by default), keep (a tilt that stays turned instead of returning to level), joint, target, hand_follows ('rigid'|'level'), follow_point, yaw_joint, lift_joint, angle_deg and pivot (rotate: the part is turned on the table about the vertical through pivot, default its centre, by pushing the pose's face around it; positive = counterclockwise from above), turns, pitch_m, bite_deg, regrasp_lift_m (an unscrew's open hand rises this far off the cap before turning back and comes down again after: fingers left on the cap's top edge drag it back; default 0.006), direction, distance_m, angle_deg, axis, pivot, release, duration_s, expect (overrides), with (a body, or a list of bodies, that rides along with the part this subgoal moves and is not a disturbed bystander: the toy inside the drawer being shut), open_frac / finger_targets_open / press_depth_rad / press_travel_m (grip settings that override the pose's own; a negative press_depth_rad extends the grip fingers from their closed targets instead of clamping, e.g. to spread scissor rings), reanchor ('full' default: the pose follows its part's whole motion since it was solved; 'yaw': translation and turn about the vertical only, for a cylinder that rolled about its own axis; 'translate'; 'none'), and for `cut`: join (a rope join's equality name) | cut_point [x, y, z], cut_s_frac (0 pivot .. 1 tip, default 0.5), approach_m (0.03), position (true), via [[x, y, z], ...], close_depth_rad (0.6) | finger_targets_close {joint: rad}, close_s (1.2), hold_s (0.6), joint (the scissors' hinge); for `press_seat`: pose (palm or flat fingers on the part), seat_depth_m | seat_z | seat_on + seat_offset_m, force_n (default and ceiling 40 N), travel_m (default: from where the hand first meets the part, seat depth + force/10000 N/m + 2 mm, at most 0.05), settle (default true: the pose is rocked up to 15 deg about the line across its fingers and shifted up to 12 mm in the press plane until its first contacts surround the part's centre, as a compliant hand settles; a point contact off the centre tips a press-fit part and jams it; expect.press.settle records it), direction ([0, 0, -1]), press_s (1.0), hold_s (0.8), approach_m (0.05), seat_tol_m (0.0015); for `both`: hands {left: entry, right: entry} (each a full entry with its own motion, object, pose ...), sync ('start' default | 'end' | 'keyframes'), hand_clearance_m (0.005); for `slide`: direction [dx, dy], overhang_m | distance_m | target_pos, mode ('push' | 'drag', default the pose's slide_mode), force_n (drag press, default 2.5x what the friction needs, at least 15 N, at most 40), extra_m (0.004); for `edge_pinch`: lift_m (0.08); for `tilt_up`: lift_m (the far end's rise, default a fingertip thickness + 8 mm), force_n, hold (true), edge_dir, press_point; for `hook`: lift_m, pull_m (a pure hook's pull, 0.012); for `strip_pinch`: lift_m (0.06); any of them max_table_pen_m (0.002); for `pin`: object (part, link or cable end), pose, mode ('press' | 'grip'; default press for a pose whose grip fingers have no thumb), force_n (20), direction (default minus the pose's contact normal), tol_m (0.002), rot_tol_deg (3), stay_tol_m (0.01: how far placing the pin may shift the part); for `lift_together`: hands {left: {object, pose, hand_scene, approach_m}, right: {...}}, lift_m (0.08), lift_dir | lift_path, spacing_tol_m (0.02), duration_s; for `carry_together`: offset [dx, dy, dz] | target_pos (the two held links' midpoint), via_rise_m, duration_s; for `lay` / `drag`: object (a cable link), pose (a pinch on it), target_pos [x, y], via, lift_m, expect_order, knot, approach_m (0.06), lift_s (0.6), carry_s (1.6), set_s (0.6), open_s (0.6), retreat_m (0.06); for `grasp_more`: object (the second object), pose (grip_fingers = the free fingers; finger_targets_open = their insertion shape), lift_m (0), approach_m (0.05)}."},
        {"name": "qpos0", "type": "str | list | None", "description": "Start state: a trajectory.npz (frame `frame`), a qpos list, or None for the scene keyframe."},
        {"name": "frame", "type": "int", "description": "Frame of qpos0 when it is an npz. Default -1."},
        {"name": "state", "type": "dict | str | None", "description": "Grip memory from a previous call (dict or state.json): {held: {side: body}, grip: {side: {closed, press, opened}}, pinned: {side: {body, mode, tol_m, caps, pos, quat}} (the pins in force), in_hand: {side: [{body, grip_fingers}]} (a hand holding several objects)}."},
        {"name": "command_qpos", "type": "str | list | None", "description": "The last commanded hand qpos (reference frame) so a holding hand keeps its press; default qpos0."},
        {"name": "fps", "type": "float", "description": "Default 30."},
        {"name": "max_rate_deg_s, max_rate_m_s", "type": "float", "description": "The executor's joint-rate ceilings, used to interpolate each hand's plan of a two-hand entry before the two are merged frame by frame. Defaults 90, 0.4."},
        {"name": "auto_via", "type": "bool", "description": "Route a transit the arm cannot follow (its midpoint out of reach) through an inserted via keyframe: the midpoint raised, or with either end's orientation, whichever the arm follows best. Default true."},
        {"name": "out_json", "type": "str | None", "description": "Write {keyframes, phases, expect, state} here."},
    ],
    "outputs": [
        {"name": "plan", "type": "dict", "description": "keyframes [{qpos, transit, hold, label}], phases [{label, hand, objects, anchor}] (a two-hand phase is listed once per hand), expect (a two-hand entry: {hands: {side: expect}, untouched, hands_apart, hand_clearance_m}; on/over/into targets carry `placement` {support, support_top_z, method}), state, rollout (physics settings the motion needs, e.g. press_seat's wrist_force_n), n_keyframes, duration_s, arm_reach {ok, worst_pos_err_mm, worst_rot_err_deg, unreachable [...], transit_warnings [...]} (per hand under `sides` for a two-hand entry): whether the station's arm can follow the hand through the plan (checked before anything is executed); reach_vias lists the via keyframes inserted for transits the arm could not follow."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "planning", "keyframes", "subgoal"],
    "used_in": ["Open the Bottle", "Giftbox Packing", "Place Toy in Drawer", "Battery Insertion", "Place Cup in Bowl", "Computer Installation"],
    "status": "tested",
    "entrypoint": "plan_subgoal_keyframes",
}

import json
from pathlib import Path

import mujoco
import numpy as np

from tool_library.common import episode as E
from tool_library.common import plan as P
from tool_library.common.subgoal import SubgoalPlanner


def _qpos(model, q, frame):
    if q is None:
        d = mujoco.MjData(model)
        if model.nkey:
            mujoco.mj_resetDataKeyframe(model, d, 0)
        return d.qpos.copy()
    if isinstance(q, (str, Path)):
        return E.state_qpos(model, q, frame)
    return np.asarray(q, float).copy()


def _strip_poses(sg):
    """The subgoal as recorded in the plan json: pose dicts left out (paths kept), also inside a two-hand entry."""
    out = {k: v for k, v in sg.items() if k not in ("pose", "hands")}
    if isinstance(sg.get("pose"), str):
        out["pose"] = sg["pose"]
    if isinstance(sg.get("hands"), dict):
        out["hands"] = {s: _strip_poses(e) for s, e in sg["hands"].items()}
    return out


def _two_hand_reach(model, side_keyframes):
    per = {s: E.keyframe_reach(model, k, s) for s, k in side_keyframes.items()}
    per = {s: r for s, r in per.items() if r}
    if not per:
        return None
    return {"ok": all(r["ok"] for r in per.values()), "worst_pos_err_mm": max(r["worst_pos_err_mm"] for r in per.values()),
            "worst_rot_err_deg": max(r["worst_rot_err_deg"] for r in per.values()),
            "unreachable": [dict(u, side=s) for s, r in per.items() for u in r["unreachable"]][:8],
            "transit_warnings": [dict(u, side=s) for s, r in per.items() for u in r["transit_warnings"]][:8], "sides": per}


def plan_subgoal_keyframes(scene, subgoal, qpos0=None, frame=-1, state=None, command_qpos=None, fps=30.0, out_json=None, auto_via=True,
                           max_rate_deg_s=90.0, max_rate_m_s=0.4):
    model = mujoco.MjModel.from_xml_path(str(scene)) if isinstance(scene, (str, Path)) else scene
    q = _qpos(model, qpos0, frame)
    if command_qpos is not None:
        qc = _qpos(model, command_qpos, frame)
        for rig in P.scene_rigs(model).values():
            for a in E.hand_qadr(model, rig):
                q[a] = qc[a]
    if isinstance(state, (str, Path)):
        state = json.loads(Path(state).read_text())
    sp = SubgoalPlanner(model, q, state=state, fps=fps, rates=(max_rate_deg_s, max_rate_m_s))
    res = sp.run(subgoal)
    res["n_keyframes"] = len(res["keyframes"])
    side = subgoal.get("side") or subgoal.get("acting_hand")
    side_kf = res.pop("side_keyframes", None)
    try:
        if side_kf:                              # a two-hand entry: each arm's reach along its own hand's plan
            res["arm_reach"] = _two_hand_reach(model, side_kf)
        else:
            if side and auto_via:
                res["keyframes"], vias = E.insert_reach_vias(model, res["keyframes"], side)
                if vias:
                    res["reach_vias"] = vias
            res["arm_reach"] = E.keyframe_reach(model, res["keyframes"], side) if side else None
    except Exception as e:                      # reach is advice, never a crash of the planner
        res["arm_reach"] = {"error": str(e)}
    if side_kf:                                  # one keyframe per frame: count the frames chain_subgoals lays out
        n = sum((max(int(k["transit"]) - 1, 0) if i else 0) + max(1, int(k["hold"])) for i, k in enumerate(res["keyframes"]))
        res["duration_s"] = round(n / float(fps), 2)
    else:
        res["duration_s"] = round(sum(k["transit"] + k["hold"] for k in res["keyframes"]) / float(fps), 2)
    res["subgoal"] = _strip_poses(subgoal)
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(res, indent=1, default=float))
        res["out_json"] = str(out_json)
    return res
