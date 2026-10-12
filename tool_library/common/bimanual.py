"""Bimanual coordination for Stage 3, beyond two hands acting on two parts at once (`both`):

- pin: one hand steadies a part, or one link or end of a cable, while the OTHER hand acts. A press pin pushes the part
  onto its support with the wrist (fingertips, flat fingers or the palm; the wrist force on the press axis is capped at
  force_n, so the press is a known force, not a position); a grip pin holds it in a pinch or a wrap. The pin lasts
  through the following entries (the executor carries the pinning hand's commands, its press reference and its force
  cap over) until an entry of that hand releases it, and every one of those entries is judged on the pinned part
  staying within tol_m of where it was when the entry started, on EVERY frame (`pinned`), with the pinning hand still
  on it.
- lift_together / carry_together: both hands on two links (two bodies) of ONE free object, a heavy cable, moved together
  by the same displacement, so the load is shared and the spacing along the object kept (`spacing`): a span of the cable
  is lifted, not one point; the expectations are checked for both hands (a link held by its pads or its neighbours').
- grasp_more: a hand that already holds an object with some fingers takes a second one with its FREE fingers (finger
  allocation: the first cup between thumb and index, the second one on middle, ring and pinky). The held fingers keep
  their press; only the free fingers are driven; both objects ride with the wrist (`also_in_hand`).

SubgoalPlanner's motions call into this module; outcome.evaluate calls span_checks for the expectations above."""
from __future__ import annotations

import copy

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from tool_library.common import episode as E
from tool_library.common import hand as H
from tool_library.common import plan as P
from tool_library.common import held_contact as HC

SIDES = ("left", "right")
TWO_HAND_MOTIONS = ("both", "lift_together", "carry_together")
# a pinned part may move at most this far (and turn at most this much) from where the pin left it, on every frame of
# every entry while the pin lasts: a steadied part that creeps 2 mm per bite has moved a centimetre after five bites
PIN_TOL_M = 0.002
PIN_ROT_TOL_DEG = 3.0
# the press of a press pin, N: half the wrist's 40 N cap. Measured on a free cable end pressed by a fingertip while the
# other hand dragged a link 30 cm along it 5 cm: 10 N let the end creep 2.2 mm, 20 N held it within 1.1 mm
PIN_FORCE_N = 20.0
WRIST_KP_REF = 10000.0       # the executor's default wrist servo stiffness (subgoal.WRIST_KP_REF)
# the press reference goes this far past force / kp beyond the first contact, so the servo stays saturated at the cap
# when the part settles a millimetre into its support (a reference exactly at force / kp pressed with 4 N after 0.6 mm)
PIN_EXTRA_M = 0.003
PIN_PLACE_TOL_M = 0.01       # placing the pin itself may shift the part this much (stays_put of the pin entry)
# what a pinning hand may do while it pins: let go, go home, wait, or pin again
PINNING_OK = ("release", "home", "wait", "other", "pin")
SPACING_TOL_M = 0.02         # two hands carrying one cable keep the distance between their links within this
LINK_SPAN = 2                # a pinch on a cable link holds it with the pads on that link or its neighbours
SPAN_KEYS = ("pinned", "spacing", "also_in_hand", "link_in_hand")   # the expectations span_checks measures


# ----------------------------------------------------------------------------- shared helpers
def is_two_hand(sg):
    return str((sg or {}).get("motion", "")) in TWO_HAND_MOTIONS


def bid(model, name):
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(name)) if name else -1
    if b < 0:
        raise ValueError(f"no body {name!r} in the scene")
    return int(b)


def rigid_bodies(model, b):
    """The body and the children welded to it (no joint between them): the rigid piece a hand touches. A cable link's
    subtree is every link after it, which is not the link."""
    out, stack = {int(b)}, [int(b)]
    while stack:
        x = stack.pop()
        for c in range(model.nbody):
            if int(model.body_parentid[c]) == x and c != x and c not in out and int(model.body_jntnum[c]) == 0:
                out.add(c); stack.append(c)
    return out


def chain_neighbours(model, b, span):
    """A link of a chain (a cable) and the links within `span` of it: up the parents while they are links of the same
    chain (stop at the free root's parent), down the children that hang on a joint."""
    out = {int(b)}
    x = int(b)
    for _ in range(int(span)):
        par = int(model.body_parentid[x])
        if par <= 0 or int(model.body_jntnum[x]) == 0 or int(model.jnt_type[int(model.body_jntadr[x])]) == int(mujoco.mjtJoint.mjJNT_FREE):
            break
        out.add(par); x = par
    front = [int(b)]
    for _ in range(int(span)):
        nxt = [c for f in front for c in range(model.nbody) if int(model.body_parentid[c]) == f and c != f and int(model.body_jntnum[c]) > 0]
        out |= set(nxt); front = nxt
    return out


def collision_geoms(model, bodies):
    return [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies and (model.geom_contype[g] or model.geom_conaffinity[g])]


def free_root(model, name):
    """The body carrying the free joint the named body moves with (a cable link: the cable's root), or None."""
    j = P.free_joint_of(model, name)
    return None if j is None else int(model.jnt_bodyid[j])


def press_caps(model, data, rig, dvec, force):
    """Per-axis wrist force caps that press with `force` along dvec: the wrist slide axis nearest the press direction gets
    force, the others keep the library's cap and hold the hand on line; a direction between axes caps every axis
    (press_seat's rule)."""
    R0 = P.root_frame(model, data, rig)[1]
    loc = R0.T @ (np.asarray(dvec, float) / max(np.linalg.norm(dvec), 1e-12))
    k = int(np.argmax(np.abs(loc)))
    if abs(loc[k]) >= np.cos(np.radians(25.0)):
        caps = [float(H.WRIST_FORCE_N)] * 3
        caps[k] = float(force)
    else:
        caps = [float(force)] * 3
    return caps


def _quat_wxyz(R):
    q = Rot.from_matrix(np.asarray(R, float)).as_quat()
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def _push2(sp, labels, transit_s, hold_s, objects, anchors):
    """One keyframe of a two-hand motion planned in ONE planner: the label joins both hands' phases and a phase is
    recorded per hand (what run_both writes for a merged plan)."""
    lab = E.combine_label(labels)
    kf = sp.pl.push(lab, transit_s, hold_s, "left", objects.get("left", []), anchor=anchors.get("left"))
    sp.pl.phases[-1]["hand"] = "left"
    sp.pl.phases.append({"label": lab, "hand": "right", "objects": list(objects.get("right", [])), "anchor": anchors.get("right"),
                         "keyframe": len(sp.pl.keyframes) - 1})
    return kf


# ----------------------------------------------------------------------------- pin
def pin_press_geometry(model, data, rig, pose, part_gids, direction=None, approach_m=0.05,
                       max_overlap_m=0.003, max_contact_offset_m=0.01):
    """Validate a press's first contact and descent before creating any commands.

    The supplied pose may sit just above or below first contact. Small offsets
    are corrected to that contact; a large correction, incidental thumb/palm
    contact, or a colliding approach is an invalid pin proposal. This is a
    geometric screen, not proof that contact physics holds the part still.
    """
    from tool_library.common import subgoal as SG
    if not 0 <= float(max_overlap_m) <= .003 or not 0 < float(max_contact_offset_m) <= .01:
        raise ValueError("pin geometry limits must be no greater than 3 mm overlap and 10 mm contact correction")
    p, R = P.pose_from_dict(pose)
    a = P.retreat_axis(pose, R)
    normals = list((pose.get("normals") or {}).values())
    if direction is None:
        if normals:
            direction = -np.mean([np.asarray(n, float) / max(np.linalg.norm(n), 1e-12) for n in normals], axis=0)
        elif pose.get("normal") is not None:
            direction = -np.asarray(pose["normal"], float)
        else:
            direction = -a
    direction = np.asarray(direction, float)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(direction).all() or norm < 1e-9:
        return {"ok": False, "reason": "press_direction_undefined"}
    direction /= norm
    clamp = H.executor_clamp(rig, pose)
    posed = dict(clamp["closed"])
    opened = dict(posed)
    opened.update({k: float(v) for k, v in (pose.get("finger_targets_open") or {}).items() if k in rig.finger_range})
    d = mujoco.MjData(model)
    d.qpos[:] = data.qpos
    mujoco.mj_forward(model, d)
    off, contacts = SG.press_contacts(model, d, rig, p, R, posed, part_gids, direction)
    report = {"ok": False, "reason": None, "contact_offset_m": None if off is None else float(off),
              "direction": direction.tolist(), "approach_m": float(approach_m)}
    if off is None or not contacts:
        return dict(report, reason="press_misses_part")
    if abs(float(off)) > float(max_contact_offset_m):
        return dict(report, reason="press_contact_offset_too_large")
    contact_pos = p + direction * float(off)
    P.set_wrist(model, d, rig, contact_pos, R)
    rig.set_fingers(d, posed)
    mujoco.mj_kinematics(model, d)
    distances = [(float(H.geom_distance(model, d, h, g, .045)), h, g) for _, h, g in contacts]
    nearest = min(x[0] for x in distances)
    report["first_contact_gap_m"] = nearest
    report["first_contact_geoms"] = [[model.geom(h).name, model.geom(g).name] for dist, h, g in distances if dist <= nearest + .001]
    # press_contacts is an iterative geometry query. A final iterate alone
    # does not establish that the hand ever reached the requested surface.
    if abs(nearest) > .0003:
        return dict(report, reason="press_contact_not_converged")
    pressing = list(pose.get("grip_fingers") or [])
    intended = set().union(*(rig.tip_geoms(f) for f in pressing)) if pressing else set()
    intended |= {g for g in rig.collision_gids if any(f"_{f}_" in (model.geom(g).name or "") for f in pressing)}
    if pressing and not any(h in intended and dist <= .001 for dist, h, _ in distances):
        return dict(report, reason="press_wrong_first_contact")
    own = set(rig.geom_ids)
    # This free-wrist proposal has not solved the active arm's IK yet. Its
    # starting arm pose is not an obstacle that stays fixed during the descent;
    # the coupled planner checks its actual arm/hand geometry after solving IK.
    active_arm = {g for g in range(model.ngeom)
                  if (model.body(int(model.geom_bodyid[g])).name or "").startswith(f"robot_{rig.side}_arm_")
                  and int(model.body_jntnum[int(model.body_weldid[int(model.geom_bodyid[g])])]) > 0}
    obstacles = {g for g in range(model.ngeom) if g not in own | active_arm and (model.geom_contype[g] or model.geom_conaffinity[g])}
    for g, h in zip(model.pair_geom1, model.pair_geom2):
        if int(g) in own and int(h) not in own | active_arm:
            obstacles.add(int(h))
        elif int(h) in own and int(g) not in own | active_arm:
            obstacles.add(int(g))
    floors = [g for g in obstacles if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE)]
    ids = {"obj_gids": obstacles - set(floors), "floor_gid": floors[0] if floors else -1}
    peak, phase, samples = 0., None, 0
    # One millimetre translation samples include all collision geometry,
    # including the pressing pads. Explicit opening shapes are swept too.
    n = max(2, int(np.ceil(abs(float(approach_m)) / .001)) + 1)
    poses = [(contact_pos + a * float(approach_m) * t, opened, "approach") for t in np.linspace(1., 0., n)]
    delta = max((abs(opened[k] - posed[k]) for k in posed), default=0.)
    if delta > 1e-12:
        poses += [(contact_pos, {k: (1-t)*opened[k] + t*posed[k] for k in posed}, "close")
                  for t in np.linspace(0., 1., max(2, int(np.ceil(delta / .01)) + 1))]
    for pp, fingers, label in poses:
        P.set_wrist(model, d, rig, pp, R)
        rig.set_fingers(d, fingers)
        mujoco.mj_forward(model, d)
        overlap = H.deepest_overlap(model, d, rig, ids, keep_tips=(), visual_floor=True)
        samples += 1
        if overlap > peak:
            peak, phase = overlap, label
        if overlap > float(max_overlap_m):
            return dict(report, reason="press_approach_collision", max_overlap_m=float(peak), collision_phase=phase, samples=samples)
    return dict(report, ok=True, contact_pos=contact_pos.tolist(), posed=posed, opened=opened,
                max_overlap_m=float(peak), samples=samples)


def check_pinning_side(sp, sg, side, motion):
    """A pinning hand only lets go, goes home, waits or pins again: any other motion would drag the part it steadies."""
    pin = (sp.state.get("pinned") or {}).get(side)
    if pin and motion not in PINNING_OK:
        raise ValueError(f"subgoal {sg.get('index')}: the {side} hand pins {pin['body']!r}; release it first (a release entry of the {side} "
                         f"hand) before it does a {motion}")


def plan_pin(sp, sg, side, obj):
    """Plan a pin (see the module docstring). Entry: object (the part, link or cable end), pose (a press pose: flat_hand_pose
    pads down, push_pose_from_point, pin_pose; or a grip: any grasp pose), mode ('press' | 'grip'; default: press for a
    pose whose grip fingers have no thumb), force_n (press, default PIN_FORCE_N, never above the wrist cap), direction
    (press, default minus the pose's mean contact normal, else into its approach), tol_m / rot_tol_deg (how still the part
    must stay while the pin lasts), approach_m, press_s, hold_s."""
    m, d = sp.model, sp.pl.data
    if not obj:
        raise ValueError("pin: give `object` (the part, link or cable end to steady)")
    ob = bid(m, obj)
    held = sp.state["held"].get(side)
    if held and held != obj:
        raise ValueError(f"pin: the {side} hand holds {held!r}; let go first")
    pose = sp._pose(sg)
    rig = sp.pl.rigs[side]
    grip = list(pose.get("grip_fingers") or [])
    mode = sg.get("mode") or pose.get("pin_mode") or ("press" if grip and "thumb" not in grip else "grip")
    if mode not in ("press", "grip"):
        raise ValueError(f"pin: mode must be 'press' or 'grip', got {mode!r}")
    # the phase is named as the steadying it is: a press pin presses the part, a grip pin holds it (Stage 1's goal gate
    # finds a steadying hold's window by these words)
    label = f"{side}: {'press' if mode == 'press' else 'hold'} {obj} (pinned)"
    rec = {"body": obj, "mode": mode, "tol_m": float(sg.get("tol_m", PIN_TOL_M)), "rot_tol_deg": float(sg.get("rot_tol_deg", PIN_ROT_TOL_DEG)),
           "caps": None}
    if mode == "grip":
        sp._approach_pose(side, pose, obj, approach_m=float(sg.get("approach_m", 0.06)), label=f"{side}: reach to pin {obj}", **sp._approach_timing(sg))
        sp.pl.phases[-1]["label"] = sp.pl.keyframes[-1]["label"] = label
        if sp._pads:
            sp._pads["pads_at_label"] = label
        exp = {"in_hand": True, "min_links": 2, "stays_put": True}
    else:
        asked = float(sg.get("force_n", PIN_FORCE_N))
        if not asked > 0:
            raise ValueError("pin: force_n must be > 0")
        force = min(asked, float(H.WRIST_FORCE_N))
        p, R = P.pose_from_dict(pose)
        if sg.get("direction") is not None:
            dvec = np.asarray(sg["direction"], float)
        elif pose.get("normals"):
            dvec = -np.mean([np.asarray(n, float) / max(np.linalg.norm(n), 1e-12) for n in pose["normals"].values()], axis=0)
        elif pose.get("normal") is not None:
            dvec = -np.asarray(pose["normal"], float)
        else:
            dvec = -P.retreat_axis(pose, R)
        if np.linalg.norm(dvec) < 1e-9:
            raise ValueError("pin: the press direction is undefined (give `direction`)")
        dvec = dvec / np.linalg.norm(dvec)
        g = sp._grip_from_pose(side, pose)
        posed = dict(g["closed"])
        part_g = collision_geoms(m, rigid_bodies(m, ob))
        mujoco.mj_forward(m, d)
        geometry = pin_press_geometry(m, d, rig, pose, part_g, dvec, approach_m=float(sg.get("approach_m", 0.05)))
        if not geometry["ok"]:
            raise ValueError(f"pin: {geometry['reason']} for {obj}: {geometry}; generate a valid surface contact and clear approach")
        off = float(geometry["contact_offset_m"])
        p = np.asarray(geometry["contact_pos"], float)
        posed = geometry["posed"]
        travel = force / WRIST_KP_REF + PIN_EXTRA_M
        g.update({"press": posed, "wrist_press": None})        # the fingers stay as posed: the wrist presses
        g["opened"] = geometry["opened"]
        sp.state["grip"][side] = g
        a = P.retreat_axis(pose, R)
        am = float(sg.get("approach_m", 0.05))
        sp.pl.fingers(side, g["opened"]); sp.pl.move_wrist(side, p + a * am, R)
        sp.pl.push(f"{side}: reach to pin {obj}", float(sg.get("t_reach", 2.0)), 0.1, side, [obj], anchor=obj)
        sp.pl.move_wrist(side, p, R); sp.pl.push(f"{side}: descend on {obj}", 0.8, 0.1, side, [obj], anchor=obj)
        if g["opened"] != posed:
            sp.pl.fingers(side, posed); sp.pl.push(f"{side}: close on {obj}", 0.4, 0.1, side, [obj], anchor=obj)
        sp.pl.move_wrist(side, p + dvec * travel, R)
        sp.pl.push(label, float(sg.get("press_s", 0.8)), float(sg.get("hold_s", 0.6)), side, [obj], anchor=obj)
        caps = press_caps(m, d, rig, dvec, force)
        # the cap holds from the press on, and through every later entry while the pin lasts (execute_subgoal_plan)
        sp._rollout["wrist_force_n"] = {side: {"cap": caps, "phases": [label]}}
        rec.update({"caps": caps, "press_dir": dvec.round(6).tolist(), "travel_m": round(travel, 5), "force_n": force,
                    "contact_offset_m": round(float(off), 5)})
        exp = {"in_hand": True, "min_links": 1, "stays_put": True,
               "pin_press": {"force_n": force, "travel_m": round(travel, 5), "contact_offset_m": round(off, 5),
                             "geometry": {k: v for k, v in geometry.items() if k not in ("posed", "opened")},
                             **({"force_clamped_from_n": asked} if asked > force else {})}}
    exp["stay_tol_m"] = float(sg.get("stay_tol_m", PIN_PLACE_TOL_M))
    sp.state["held"][side] = obj
    sp.state.setdefault("pinned", {})[side] = rec
    return exp


def pins_for(sp, acting):
    """The pins in force through an entry whose acting sides are `acting`: every pin of a hand that does not act, with
    its reference pose (recorded the first time an entry starts after the pin: where the pin left the part)."""
    out = []
    m, d = sp.model, sp.pl.data
    mujoco.mj_forward(m, d)
    for s_, pin in sorted((sp.state.get("pinned") or {}).items()):
        if not pin or s_ in acting:
            continue
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, pin["body"])
        if b < 0:
            continue
        if pin.get("pos") is None:
            pin["pos"] = d.xpos[b].round(6).tolist(); pin["quat"] = _quat_wxyz(d.xmat[b].reshape(3, 3))
        out.append({"side": s_, "body": pin["body"], "pos": pin["pos"], "quat": pin["quat"], "tol_m": pin["tol_m"],
                    "rot_tol_deg": pin["rot_tol_deg"]})
    return out


def add_pin_expect(sp, sg, expect, acting, pins=None):
    """Put the pins in force (pins_for, read BEFORE the entry's motion is planned: the planner moves the parts it carries,
    a cable rigidly with the link a hand lifts) into the entry's expectations."""
    pins = pins_for(sp, acting) if pins is None else pins
    if pins:
        expect["pinned"] = pins
        expect["involved"] = list(dict.fromkeys(list(expect.get("involved") or []) + [p_["body"] for p_ in pins]))
    return expect


def acting_sides(sg):
    """The hands an entry moves (a two-hand entry: each hand whose own entry is not a wait)."""
    mo = str(sg.get("motion", ""))
    if mo in ("lift_together", "carry_together"):
        return set(SIDES)
    if mo == "both":
        return {s_ for s_, e in (sg.get("hands") or {}).items() if str((e or {}).get("motion")) not in ("wait", "other")}
    if mo in ("wait", "other"):
        return set()
    s_ = sg.get("side") or sg.get("acting_hand")
    return {s_} if s_ else set()


def ease_off_pin(sp, side):
    """Before a pressing hand lets go: the wrist backs out of its press to where it met the part, so the opening is not
    done at the force cap; the pin ends."""
    pin = (sp.state.get("pinned") or {}).pop(side, None)
    if not pin:
        return
    if pin.get("mode") == "press" and pin.get("press_dir") and pin.get("travel_m"):
        pw, Rw = sp.pl.wrist(side)
        back = np.asarray(pin["press_dir"], float) * float(pin["travel_m"])
        sp.pl.move_wrist(side, pw - back, Rw)
        sp.pl.push(f"{side}: ease off {pin['body']}", 0.4, 0.1, side, [pin["body"]], anchor=pin["body"])


def with_pin_caps(lateral, state):
    """The wrist force caps of an entry with the pins in force: each pressing hand keeps its press cap for the whole
    entry (its reference still lies past the contact), the other hand keeps what the entry asks for."""
    caps = {s_: p_["caps"] for s_, p_ in ((state or {}).get("pinned") or {}).items() if p_ and p_.get("caps")}
    if not caps:
        return lateral
    out = {}
    for s_ in SIDES:
        if s_ in caps:
            out[s_] = list(caps[s_])
        elif isinstance(lateral, dict):
            if s_ in lateral:
                out[s_] = lateral[s_]
        elif lateral is not None:
            out[s_] = lateral
    return out


# ----------------------------------------------------------------------------- two hands, one object
def _sub(sg, s_):
    e = dict((sg.get("hands") or {}).get(s_) or {})
    e["side"] = s_
    e.setdefault("index", sg.get("index"))
    return e


def _same_tree(model, a, b):
    ra, rb = free_root(model, a), free_root(model, b)
    if ra is None or rb is None or ra != rb:
        raise ValueError(f"{a!r} and {b!r} are not parts of one free object (a lift_together carries ONE object with both hands; two "
                         "objects at once are a 'both' entry)")
    return ra


def _attach_shared(sp, objs):
    """Both hands on one object: the object rides the left wrist and the right hand moves with it (a pure translation keeps
    both grips where they are on it)."""
    sp.pl.attach("left", objs["left"])
    sp.pl.coupled["left"] = ["right"]
    sp.pl.coupled["right"] = []


def _per_hand_hold(sp, s_, link):
    """What a hand holding a link of a cable must show: the standard in_hand for a rigid part; for a link of a chain, the
    pads on that link or its LINK_SPAN neighbours (a pinch straddles two links as soon as the cable bends, and the
    standard check counts only the link and the links AFTER it)."""
    m = sp.model
    b = bid(m, link)
    chain = int(m.body_jntnum[b]) > 0 and int(m.jnt_type[int(m.body_jntadr[b])]) != int(mujoco.mjtJoint.mjJNT_FREE)
    if chain or any(int(m.body_parentid[c]) == b and int(m.body_jntnum[c]) > 0 for c in range(m.nbody)):
        return {"link_in_hand": [{"body": link, "hand": s_, "span": LINK_SPAN, "min_links": 2}]}
    return {"in_hand": True}


def _two_hand_expect(sp, sg, objs, per_hand, spacing):
    root = free_root(sp.model, objs["left"])
    # the whole object is handled (both hands on it): its every link is the hands' to touch, none is a bystander
    involved = [objs["left"], objs["right"]] + ([sp.model.body(root).name] if root is not None else [])
    exp = {"hands": per_hand, "objects": dict(objs), "involved": involved, "untouched": sp._untouched(involved),
           "hands_apart": True, "hand_clearance_m": float(sg.get("hand_clearance_m", 0.005)), "spacing": spacing}
    for k in ("pos_tol_m", "max_hand_pen_m", "max_obj_pen_m", "max_track_m", "fraction", "slip_tol_m"):
        if k in sg:
            exp[k] = sg[k]
    exp = HC.merge_motion_expectations(exp, sg.get("expect"))
    return exp


def _spacing(sp, objs, sg):
    m, d = sp.model, sp.pl.data
    mujoco.mj_forward(m, d)
    pa, pb = d.xpos[bid(m, objs["left"])], d.xpos[bid(m, objs["right"])]
    return {"bodies": [objs["left"], objs["right"]], "d0_m": round(float(np.linalg.norm(pa - pb)), 5),
            "tol_m": float(sg.get("spacing_tol_m", SPACING_TOL_M))}


def plan_lift_together(sp, sg):
    """{motion: 'lift_together', hands: {left: {object, pose, hand_scene?, approach_m?}, right: {...}}, lift_m (0.08),
    lift_dir | lift_path, duration_s}: both hands reach, descend and close on their own link of ONE free object in the
    same keyframes, then lift it together: both wrists move by the same offsets, so the spacing between the grips and the
    share of the load stay as they were (a heavy cable lifted along a span, so its end can pass under it). A hand that
    already holds its link
    (held in the state, no pose given) skips its approach."""
    m = sp.model
    subs = {s_: _sub(sg, s_) for s_ in SIDES}
    objs = {s_: subs[s_].get("object") or sp.state["held"].get(s_) for s_ in SIDES}
    if not all(objs.values()):
        raise ValueError("lift_together: each hand's entry needs its `object` (a link or body of the one object both hands lift)")
    for s_ in SIDES:
        bid(m, objs[s_])
        check_pinning_side(sp, sg, s_, "lift_together")
    _same_tree(m, objs["left"], objs["right"])
    lift = float(sg.get("lift_m", 0.08))
    offsets = sp._lift_offsets(sg, lift)
    duration = float(sg.get("duration_s", max(1.2, 0.6 * len(offsets))))
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("lift_together duration_s must be finite positive seconds")
    default_times = {"t_reach": 2., "t_descend": .8, "t_close": .5, "t_grip": .3}
    shared_times = sp._approach_timing(sg)
    times = {s_: {**default_times, **shared_times, **sp._approach_timing(subs[s_])} for s_ in SIDES}
    need = {s_: not (sp.state["held"].get(s_) == objs[s_] and subs[s_].get("pose") is None) for s_ in SIDES}
    if any(need.values()):
        geo, waypoints = {}, {}
        for s_ in SIDES:
            if not need[s_]:
                continue
            pose = sp._pose(subs[s_])
            g = sp._grip_from_pose(s_, pose)
            if not g.get("own_open"):
                g["opened"] = sp._floor_clear_open(s_, pose, g)
            sp.state["grip"][s_] = g
            p, R = P.pose_from_dict(pose)
            geo[s_] = (p, R, P.retreat_axis(pose, R), float(subs[s_].get("approach_m", sg.get("approach_m", 0.08))), g)
            waypoints[s_] = [np.asarray(off, float) for off in (pose.get("approach_waypoints") or [])]
            if any(off.shape != (3,) or not np.all(np.isfinite(off)) for off in waypoints[s_]):
                raise ValueError("lift_together approach_waypoints must contain finite xyz offsets")
        objs_l = {s_: [objs[s_]] for s_ in SIDES}
        anch = dict(objs)

        def step(word, set_hand, transit_s, hold_s):
            for s_ in SIDES:
                if s_ in geo:
                    set_hand(s_, *geo[s_])
            _push2(sp, {s_: (f"{s_}: {word} {objs[s_]}" if s_ in geo else f"{s_}: hold {objs[s_]}") for s_ in SIDES}, transit_s, hold_s, objs_l, anch)

        # Contact-speed and obstacle-avoidance repairs must affect the reference
        # just as they do for a one-hand grasp. Synchronize each stage at the
        # slower requested duration; a hand with fewer waypoints waits at its
        # last point while the other completes its approach.
        def transit(key):
            return max(times[s_][key] for s_ in geo)

        sp._approach_controls = {
            "scope": "planned reference; physical execution still requires its outcome checks",
            "synchronization": "each stage uses the slowest requested hand duration; shorter waypoint lists wait open",
            "transit_s": {key: transit(key) for key in default_times},
            "hands": {s_: {"approach_waypoints": [off.tolist() for off in waypoints[s_]],
                             "approach_m": geo[s_][3]} for s_ in geo},
        }
        step("reach for", lambda s_, p, R, a, am, g: (sp.pl.fingers(s_, g["opened"]), sp.pl.move_wrist(s_, p + a * am, R)), transit("t_reach"), 0.1)
        for i in range(max(map(len, waypoints.values()), default=0)):
            def approach(s_, p, R, a, am, g):
                if i < len(waypoints[s_]):
                    sp.pl.move_wrist(s_, p + waypoints[s_][i], R)
            step(f"approach ({i + 1})", approach, .6, 0.)
        step("descend on", lambda s_, p, R, a, am, g: sp.pl.move_wrist(s_, p, R), transit("t_descend"), 0.1)
        step("close on", lambda s_, p, R, a, am, g: sp.pl.fingers(s_, g["closed"]), transit("t_close"), 0.2)

        def grip_(s_, p, R, a, am, g):
            if g.get("wrist_press"):
                sp.pl.move_wrist(s_, p + np.asarray(g["wrist_press"], float), R)
            sp.pl.fingers(s_, g["press"])
        step("grip", grip_, transit("t_grip"), 0.3)
    _attach_shared(sp, objs)
    for s_ in SIDES:
        sp.pl.fingers(s_, sp._grip(s_)["press"])
    starts = {s_: sp.pl.wrist(s_) for s_ in SIDES}
    t_each = duration / len(offsets)
    for i, off in enumerate(offsets):
        sp.pl.move_wrist("left", starts["left"][0] + off, starts["left"][1])        # the right hand rides the object
        _push2(sp, {s_: f"{s_}: lift {objs[s_]}" for s_ in SIDES}, t_each, 0.3 if i == len(offsets) - 1 else 0.05,
               {s_: [objs[s_]] for s_ in SIDES}, dict(objs))
    for s_ in SIDES:
        sp.state["held"][s_] = objs[s_]
    sp.state["together"] = {"bodies": dict(objs)}
    rise = float(offsets[-1][2])
    per = {s_: {"object": objs[s_], "hand": s_, "rise_m": max(0.005, 0.5 * rise), "no_slip": True, **_per_hand_hold(sp, s_, objs[s_])} for s_ in SIDES}
    return _two_hand_expect(sp, sg, objs, per, _spacing(sp, objs, sg))


def plan_carry_together(sp, sg):
    """{motion: 'carry_together', offset: [dx, dy, dz] | target_pos (where the midpoint of the two held links goes),
    via_rise_m, duration_s}: both hands, each holding a link of the same object (lift_together first), move by the same
    displacement; each link is expected at its own start plus that displacement, both still in hand, the spacing kept."""
    m, d = sp.model, sp.pl.data
    objs = {s_: sp.state["held"].get(s_) for s_ in SIDES}
    if not all(objs.values()):
        raise ValueError(f"carry_together: both hands must hold a link of one object (lift_together first); held: {sp.state['held']}")
    _same_tree(m, objs["left"], objs["right"])
    mujoco.mj_forward(m, d)
    pos0 = {s_: d.xpos[bid(m, objs[s_])].copy() for s_ in SIDES}
    if sg.get("offset") is not None:
        disp = np.asarray(sg["offset"], float)
    elif sg.get("target_pos") is not None:
        disp = np.asarray(sg["target_pos"], float) - 0.5 * (pos0["left"] + pos0["right"])
    else:
        raise ValueError("carry_together needs `offset` [dx, dy, dz] or `target_pos` (the midpoint of the two held links)")
    _attach_shared(sp, objs)
    for s_ in SIDES:
        sp.pl.fingers(s_, sp._grip(s_)["press"])
    pw, Rw = sp.pl.wrist("left")
    t = float(sg.get("duration_s", 1.5))
    labs = {s_: f"{s_}: carry {objs[s_]}" for s_ in SIDES}
    if sg.get("via_rise_m"):
        up = np.array([0.0, 0.0, float(sg["via_rise_m"])])
        sp.pl.move_wrist("left", pw + up, Rw); _push2(sp, labs, 0.3 * t, 0.0, {s_: [objs[s_]] for s_ in SIDES}, dict(objs))
        sp.pl.move_wrist("left", pw + disp + up, Rw); _push2(sp, labs, 0.5 * t, 0.0, {s_: [objs[s_]] for s_ in SIDES}, dict(objs))
        t = 0.3 * t
    sp.pl.move_wrist("left", pw + disp, Rw)
    _push2(sp, labs, t, 0.2, {s_: [objs[s_]] for s_ in SIDES}, dict(objs))
    per = {s_: {"object": objs[s_], "hand": s_, "target_pos": (pos0[s_] + disp).round(6).tolist(), "target_axes": sg.get("target_axes", "xyz"),
                "no_slip": True, **_per_hand_hold(sp, s_, objs[s_])} for s_ in SIDES}
    return _two_hand_expect(sp, sg, objs, per, _spacing(sp, objs, sg))


def run_together(sp, sg):
    """lift_together / carry_together planned in one planner (both hands in every keyframe); same output as run_both."""
    sp._rollout = {}
    sp._target_info = None
    sp._approach_controls = None
    for s_ in SIDES:
        if s_ not in sp.pl.rigs:
            raise ValueError(f"subgoal {sg.get('index')}: {sg.get('motion')} needs a {s_} hand in the scene")
    before = len(sp.pl.keyframes)
    sp.pl.push("start", 0.0, 1 / sp.pl.fps, None, [])
    sp.pl.phases.pop()
    pins = pins_for(sp, set(SIDES))
    exp = plan_lift_together(sp, sg) if sg["motion"] == "lift_together" else plan_carry_together(sp, sg)
    add_pin_expect(sp, sg, exp, set(SIDES), pins)
    if len(sp.pl.keyframes) > before + 1:
        sp.pl.keyframes[before]["label"] = sp.pl.keyframes[before + 1]["label"]
    out = {"keyframes": sp.pl.keyframes[before:], "phases": sp.pl.phases[before:], "expect": exp, "state": sp.state}
    if sp._approach_controls is not None:
        out["approach_controls"] = sp._approach_controls
    return out


# ----------------------------------------------------------------------------- a second object in the same hand
def in_hand_list(state, side):
    """What a hand holds, in grasp order, with the fingers each object is held by: state['in_hand'][side], or the one
    object of `held` with the grip memory's fingers."""
    lst = list(((state.get("in_hand") or {}).get(side)) or [])
    held = (state.get("held") or {}).get(side)
    if held and not any(e["body"] == held for e in lst):
        lst = [{"body": held, "grip_fingers": list(((state.get("grip") or {}).get(side) or {}).get("grip_fingers") or [])}]
    return lst


def attach_extras(sp):
    """At the start of a planner: every object a hand holds besides its primary rides the wrist as well."""
    for s_ in SIDES:
        for e in in_hand_list(sp.state, s_):
            if e["body"] != sp.state["held"].get(s_):
                try:
                    sp.pl.attach_extra(s_, e["body"])
                except ValueError:
                    pass


def make_primary(sp, side, obj):
    """A carry, place or release naming another of the hand's objects makes it the primary (the one the motion's target
    is for); the rest ride along."""
    lst = in_hand_list(sp.state, side)
    if len(lst) < 2 or sp.state["held"].get(side) == obj or not any(e["body"] == obj for e in lst):
        return
    sp.state["held"][side] = obj
    sp.pl.detach(side)
    sp.pl.attach(side, obj)
    for e in lst:
        if e["body"] != obj:
            sp.pl.attach_extra(side, e["body"])


def _finger_joints(rig, fingers):
    fs = set(fingers)
    return [k for k in rig.finger_range if H.finger_of_joint(k) in fs]


def plan_grasp_more(sp, sg, side, obj):
    """{motion: 'grasp_more', side, object, pose (its grip_fingers = the free fingers that take the new object;
    finger_targets of those fingers; finger_targets_open = their insertion shape), lift_m (default 0), approach_m (0.05),
    approach_waypoints}: the hand keeps what it holds with the fingers that hold it (their press is not touched), moves
    with it to the new pose, and closes only the free fingers on the new object. Refused when the pose's grip fingers
    overlap the fingers already holding something."""
    m = sp.model
    if not obj:
        raise ValueError("grasp_more: give `object` (the second object)")
    bid(m, obj)
    lst = in_hand_list(sp.state, side)
    if not lst:
        raise ValueError(f"grasp_more: the {side} hand holds nothing yet; take the first object with grasp_lift")
    held_f = set()
    for e in lst:
        if not e.get("grip_fingers"):
            raise ValueError(f"grasp_more: the grasp of {e['body']!r} names no grip_fingers, so which fingers are free is unknown")
        held_f |= set(e["grip_fingers"])
    if any(e["body"] == obj for e in lst):
        raise ValueError(f"grasp_more: the {side} hand already holds {obj!r}")
    pose = sp._pose(sg)
    free = set(pose.get("grip_fingers") or [])
    if not free:
        raise ValueError("grasp_more: the pose must name its grip_fingers (the free fingers that take the new object)")
    clash = free & held_f
    if clash:
        raise ValueError(f"grasp_more: fingers {sorted(clash)} already hold {[e['body'] for e in lst]}; the new grasp may only use the "
                         f"free fingers {sorted(set(H.FINGERS) - held_f)}")
    rig = sp.pl.rigs[side]
    c2 = H.executor_clamp(rig, pose)
    keep_j = set(_finger_joints(rig, held_f))
    new_j = set(_finger_joints(rig, free))
    cur = {k: float(sp.pl.data.qpos[a]) for k, a in rig.finger_qpos.items()}      # the holding press, as commanded

    def merge(new):
        out = dict(cur)
        for k, v in (new or {}).items():
            if k in new_j:
                out[k] = float(v)
        return out
    own_open = {k: float(v) for k, v in (pose.get("finger_targets_open") or {}).items() if k in rig.finger_range}
    opened = merge(own_open or H.open_targets(rig, c2["closed"], float(pose.get("open_frac", 0.55))))
    closed, press = merge(c2["closed"]), merge(c2["press"])
    for k in keep_j:                                   # the held fingers never move
        opened[k] = closed[k] = press[k] = cur[k]
    p, R = P.pose_from_dict(pose)
    a = P.retreat_axis(pose, R)
    am = float(sg.get("approach_m", 0.05))
    first = sp.state["held"].get(side)
    sp.pl.attach(side, first)
    held_names = [e["body"] for e in lst]
    sp.pl.fingers(side, opened); sp.pl.move_wrist(side, p + a * am, R)
    sp.pl.push(f"{side}: reach for {obj}", float(sg.get("t_reach", 2.0)), 0.1, side, held_names + [obj], anchor=obj)
    for i, off in enumerate(pose.get("approach_waypoints") or []):
        sp.pl.move_wrist(side, p + np.asarray(off, float), R)
        sp.pl.push(f"{side}: approach {obj} ({i + 1})", 0.6, 0.0, side, held_names + [obj], anchor=obj)
    sp.pl.move_wrist(side, p, R); sp.pl.push(f"{side}: descend on {obj}", 0.8, 0.1, side, held_names + [obj], anchor=obj)
    sp.pl.fingers(side, closed); sp.pl.push(f"{side}: close on {obj}", 0.5, 0.2, side, held_names + [obj], anchor=obj)
    sp.pl.fingers(side, press); sp.pl.push(f"{side}: grip {obj}", 0.3, 0.3, side, held_names + [obj], anchor=obj)
    # the new object is the primary from here; what the hand held before rides along
    sp.pl.detach(side)
    sp.pl.attach(side, obj)
    for b in held_names:
        sp.pl.attach_extra(side, b)
    lift = float(sg.get("lift_m", 0.0))
    exp = {"object": obj, "in_hand": True, "min_links": int(sg.get("min_links", min(2, len(free))))}
    if lift > 0:
        offsets = sp._lift_offsets(sg, lift)
        sp._lift(side, obj, offsets, max(0.8, 0.4 * len(offsets)) / len(offsets))
        exp.update({"rise_m": max(0.005, 0.5 * float(offsets[-1][2])), "no_slip": True})
    # letting go opens every finger: flexion backed off, and the new grasp's fingers back to their own open shape (an
    # inside spread lets go only when its abduction comes back; backing off flexion leaves it pressing the wall)
    opened_all = H.open_targets(rig, press, 0.55)
    for k in new_j:
        opened_all[k] = float(opened[k])
    for e in lst:
        for k, v in (e.get("opened") or {}).items():
            opened_all[k] = float(v)
    g_all = {"closed": dict(press), "press": dict(press), "opened": opened_all, "own_open": False, "wrist_press": None,
             "grip_fingers": sorted(held_f | free), "grip_posed": False}
    sp.state["grip"][side] = g_all
    sp.state["held"][side] = obj
    sp.state.setdefault("in_hand", {})[side] = lst + [{"body": obj, "grip_fingers": sorted(free), "opened": {k: float(opened[k]) for k in new_j}}]
    exp["also_in_hand"] = [{"body": e["body"], "hand": side, "fingers": e["grip_fingers"]} for e in lst]
    exp["involved"] = held_names
    return exp


def also_in_hand_expect(sp, side, expect):
    """A motion of a hand that holds more than one object: each other object must stay in that hand without slipping."""
    lst = in_hand_list(sp.state, side)
    prim = sp.state["held"].get(side)
    extra = [e for e in lst if e["body"] != prim]
    if len(lst) > 1 and extra and "also_in_hand" not in expect:
        expect["also_in_hand"] = [{"body": e["body"], "hand": side, "fingers": e["grip_fingers"]} for e in extra]
        expect["involved"] = list(dict.fromkeys(list(expect.get("involved") or []) + [e["body"] for e in extra]))
    return expect


def release_one(sp, side, obj, sg, anchor=None):
    """Let go of ONE of the objects a hand holds: only its fingers open (to the open shape of their grasp), the others keep
    their press; the hand backs off along retreat_dir. Returns None when the hand holds only that object (the whole
    release applies)."""
    lst = in_hand_list(sp.state, side)
    if len(lst) < 2 or not any(e["body"] == obj for e in lst):
        return None
    rig = sp.pl.rigs[side]
    ent = next(e for e in lst if e["body"] == obj)
    js = set(_finger_joints(rig, ent["grip_fingers"]))
    cur = {k: float(sp.pl.data.qpos[a]) for k, a in rig.finger_qpos.items()}
    opened = dict(cur)
    for k, v in H.open_targets(rig, cur, 0.55).items():
        if k in js:
            opened[k] = v
    for k, v in (ent.get("opened") or {}).items():      # its grasp's own open shape (an inside spread un-spreads)
        if k in js:
            opened[k] = float(v)
    make_primary(sp, side, obj)
    sp.pl.fingers(side, opened)
    sp.pl.detach_body(side, obj)
    sp.pl.push(f"{side}: let go of {obj}", float(sg.get("t_open", .5)), 0.3, side, [obj], anchor=anchor)
    pw, Rw = sp.pl.wrist(side)
    dv = np.asarray(sg.get("retreat_dir", (0, 0, 1)), float); dv /= max(np.linalg.norm(dv), 1e-9)
    sp.pl.move_wrist(side, pw + dv * float(sg.get("retreat_m", 0.08)), Rw)
    sp.pl.push(f"{side}: retreat", float(sg.get("t_retreat", .8)), 0.2, side, [obj], anchor=anchor)
    rest = [e for e in lst if e["body"] != obj]
    sp.state["in_hand"][side] = rest
    sp.state["held"][side] = rest[-1]["body"]
    sp.pl.attach(side, rest[-1]["body"])
    g = dict(sp._grip(side))
    keep = set().union(*[set(e["grip_fingers"]) for e in rest])
    g.update({"closed": dict(opened), "press": dict(opened), "grip_fingers": sorted(keep)})
    sp.state["grip"][side] = g
    return {"released": True, "at_rest": True, "also_in_hand": [{"body": e["body"], "hand": side, "fingers": e["grip_fingers"]} for e in rest]}


# ----------------------------------------------------------------------------- outcome: checks over the whole span
def _touching(model, data, rig, bodies):
    return len(E.contacts_at(model, data, set(rig.body_ids), bodies)) > 0


def span_checks(model, data, Q, t0, t1, expect, parts, checks, meas, fps=30.):
    """`pinned` (every frame: the pinned part within tol_m / rot_tol_deg of where it was when the entry started, the
    pinning hand still on it at the end), `spacing` (two links of one carried object: their distance within tol_m of where it started, on every frame)
    and `also_in_hand` (the other objects a hand holds: touching that hand at the end and not slipping in it)."""
    parts = parts or E.robot_parts(model)
    stride = max(1, (t1 - t0) // 240)
    frames = list(range(t0, t1 + 1, stride))
    if frames[-1] != t1:
        frames.append(t1)
    for pin in expect.get("pinned") or []:
        try:
            b = bid(model, pin["body"])
        except ValueError:
            checks[f"pinned:{pin.get('body')}"] = False
            continue
        # judged against where the part is when the entry starts (a replay, or a layout Stage 4 moved, puts the part
        # elsewhere than the Stage 3 run did); the drift from where the pin first left it is recorded as well
        data.qpos[:] = Q[t0]; mujoco.mj_kinematics(model, data)
        p_ref, R_ref = data.xpos[b].copy(), data.xmat[b].reshape(3, 3).copy()
        dev, rot, worst = 0.0, 0.0, t0
        for t in frames:
            data.qpos[:] = Q[t]; mujoco.mj_kinematics(model, data)
            dv = float(np.linalg.norm(data.xpos[b] - p_ref))
            rv = float(np.degrees(np.linalg.norm(Rot.from_matrix(R_ref.T @ data.xmat[b].reshape(3, 3)).as_rotvec())))
            if dv > dev:
                dev, worst = dv, t
            rot = max(rot, rv)
        rig = parts[pin["side"]]["rig"] if pin.get("side") in parts else None
        touch = None
        if rig is not None:
            data.qpos[:] = Q[t1]; mujoco.mj_forward(model, data)
            touch = _touching(model, data, rig, rigid_bodies(model, b))
        since_pin = None if pin.get("pos") is None else round(1000 * float(np.linalg.norm(data.xpos[b] - np.asarray(pin["pos"], float))), 2)
        meas.setdefault("pinned", {})[pin["body"]] = {"side": pin.get("side"), "max_dev_mm": round(1000 * dev, 2), "max_rot_deg": round(rot, 2),
                                                      "end_drift_since_pin_mm": since_pin,
                                                      "worst_frame": int(worst), "tol_mm": round(1000 * float(pin["tol_m"]), 2),
                                                      "pinning_hand_touching_end": touch}
        ok = dev <= float(pin["tol_m"]) and rot <= float(pin.get("rot_tol_deg", PIN_ROT_TOL_DEG)) and touch is not False
        checks[f"pinned:{pin['body']}"] = bool(ok)
        if not ok:
            meas["hint"] = (meas.get("hint", "") + " " if meas.get("hint") else "") + (
                f"the part the {pin.get('side')} hand pins ({pin['body']}) moved {1000 * dev:.1f} mm / {rot:.1f} deg (tolerance "
                f"{1000 * float(pin['tol_m']):.1f} mm) at frame {worst}" + ("; the pinning hand lost it" if touch is False else "") +
                ": press harder (force_n, at most the wrist cap), pin nearer the part's centre or the pulled side, or grip it")
    sp_ = expect.get("spacing")
    if sp_:
        a, b = (bid(model, n) for n in sp_["bodies"])
        dmin, dmax = np.inf, -np.inf
        for t in frames:
            data.qpos[:] = Q[t]; mujoco.mj_kinematics(model, data)
            dd = float(np.linalg.norm(data.xpos[a] - data.xpos[b]))
            dmin, dmax = min(dmin, dd), max(dmax, dd)
        d0 = float(sp_["d0_m"])
        err = max(abs(dmin - d0), abs(dmax - d0))
        meas["spacing"] = {"d0_mm": round(1000 * d0, 1), "min_mm": round(1000 * dmin, 1), "max_mm": round(1000 * dmax, 1), "tol_mm": round(1000 * float(sp_["tol_m"]), 1)}
        checks["spacing"] = err <= float(sp_["tol_m"])
    for e in expect.get("link_in_hand") or []:
        try:
            b = bid(model, e["body"])
        except ValueError:
            checks[f"link_in_hand:{e.get('body')}"] = False
            continue
        side = e.get("hand")
        if side not in parts:
            continue
        near = chain_neighbours(model, b, int(e.get("span", LINK_SPAN)))
        bodies = set().union(*[rigid_bodies(model, x) for x in near])
        data.qpos[:] = Q[t1]; mujoco.mj_forward(model, data)
        rig = parts[side]["rig"]
        cs = E.contacts_at(model, data, set(rig.body_ids), bodies)
        links = set()
        for i, _ in cs:
            c = data.contact[i]
            for g in (int(c.geom1), int(c.geom2)):
                if int(model.geom_bodyid[g]) in rig.body_ids:
                    links.add(model.body(int(model.geom_bodyid[g])).name)
        meas.setdefault("link_in_hand", {})[e["body"]] = {"hand_links": sorted(links), "links": sorted(model.body(x).name for x in near)}
        checks[f"link_in_hand:{e['body']}"] = len(links) >= int(e.get("min_links", 2))
    for e in expect.get("also_in_hand") or []:
        try:
            b = bid(model, e["body"])
        except ValueError:
            checks[f"also_in_hand:{e.get('body')}"] = False
            continue
        side = e.get("hand")
        if side not in parts:
            checks[f"also_in_hand:{e['body']}"] = False
            continue
        rig, ee = parts[side]["rig"], parts[side]["ee_body"]
        held = HC.measure_slip(model, Q, fps, t0, t1, b, ee, set(rig.body_ids), E.held_bodies(model, b), data=data)
        slip = held["drift_m"]
        data.qpos[:] = Q[t1]; mujoco.mj_forward(model, data)
        touch = _touching(model, data, rig, rigid_bodies(model, b))
        tol = float(expect.get("slip_tol_m", 0.02))
        if not np.isfinite(tol) or tol < 0:
            raise ValueError("slip_tol_m must be finite and nonnegative")
        tol = min(tol, .02)
        meas.setdefault("also_in_hand", {})[e["body"]] = {
            "touching_end": touch, "slip_in_hand_m": round(slip, 4),
            "slip_in_hand_deg": round(held["turn_deg"], 3), "turn_judged": held["turn_judged"],
            "translation_peak_frame": held["translation_peak_frame"], "rotation_peak_frame": held["rotation_peak_frame"],
            "held_body": held["body"]}
        checks[f"also_in_hand:{e['body']}"] = bool(touch and slip <= tol
                                                 and (not held["turn_judged"] or held["turn_deg"] <= HC.RIGID_SLIP_DEG))
    return checks
