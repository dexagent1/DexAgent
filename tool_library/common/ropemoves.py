"""Rope moves for Stage 3: `lay`, one over-move of a cable lying on the table (plan_knot_moves plans them), and the
crossing-order expectation that judges it.

A lay picks one link of the cable with a pinch, lifts it just clear of the strands its path passes over, carries it
(through optional via points) to its landing point, SETS IT DOWN there and only then opens the fingers, without moving
the wrist, and backs the open hand out along the pose's approach. Measured on a hand-built knot twin on this bench: a
link dropped from the air lands off the strand it was meant to cross, and a hand that moves while its fingers open drags
the rope (11 cm), so the hand stays where the fingers opened and leaves the way it came. The rope behind the link
follows it (follow-the-leader) and lies over the strands it was drawn across; what the move did to the rope's crossings
is checked on the executed rope by its crossing ORDER (knot.crossing_order: 'a+ b- a- b+' ...), not by the count, and
the last move of a tie by the knot itself (expect knot, rope_knot_check's classification)."""
from __future__ import annotations

import mujoco
import numpy as np

from tool_library.common import episode as E
from tool_library.common import hand as H
from tool_library.common import knot as K
from tool_library.common import plan as P

LAY_CLEAR_M = 0.012          # the carried link's underside passes this far above the highest strand under its path
LAY_MIN_LIFT_M = 0.03        # and is lifted at least this much off where it lay
DRAG_LIFT_M = 0.008          # a drag holds the link just off the table: the rope behind it slides on the cloth
RELEASE_OPEN_FRAC = 0.85     # the fingers open this far (toward extension) to let go: the pads come off the rope
MOTIONS = ("lay", "drag")


def install(cls):
    """Add the rope moves to a planner class as m_<motion> methods (subgoal.SubgoalPlanner); returns their names."""
    if not hasattr(cls, "m_lay"):
        cls.m_lay = lambda self, sg, side, obj: plan_lay(self, sg, side, obj)
    if not hasattr(cls, "m_drag"):
        cls.m_drag = lambda self, sg, side, obj: plan_drag(self, sg, side, obj)
    return MOTIONS


def cable_of(model, body):
    """(prefix, ordered body ids) of the construct_rope cable a link belongs to (B_first, B_1 .., B_last)."""
    from tool_library.verifiers.rope_knot_check import _cable_bodies
    import re
    name = model.body(int(body)).name if not isinstance(body, str) else body
    pre = re.sub(r"B_(first|last|\d+)$", "", name)
    return _cable_bodies(model, pre or None)


def centreline(model, data, bodies, pre):
    from tool_library.verifiers.rope_knot_check import _centreline
    return _centreline(model, data, bodies, pre)


def cable_radius(model, bodies):
    rs = [float(model.geom_size[g][0]) for g in range(model.ngeom) if int(model.geom_bodyid[g]) in set(bodies)
          and int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_CAPSULE)]
    return float(np.median(rs)) if rs else 0.005


def strand_top_along(model, data, bodies, skip, pts, radius, reach_m=None):
    """The highest top (centre + radius) of the cable's links lying within `reach_m` (default 2.5 radii) of the xy path
    `pts`, the carried link and its neighbours (`skip`) left out; the table top (0) when none."""
    reach_m = float(reach_m if reach_m is not None else 2.5 * radius)
    top = 0.0
    P_ = np.asarray(pts, float)[:, :2]
    for b in bodies:
        if b in skip:
            continue
        p = data.xpos[b]
        for a, c in zip(P_[:-1], P_[1:]):
            d = c - a
            t = float(np.clip(np.dot(p[:2] - a, d) / max(np.dot(d, d), 1e-12), 0.0, 1.0))
            if np.linalg.norm(a + t * d - p[:2]) <= reach_m:
                top = max(top, float(p[2]) + radius)
                break
    return top


def _release_open(sp, side, g):
    """The let-go shape: flexion backed off RELEASE_OPEN_FRAC of the way (idle fingers stay curled), narrowed while any
    finger would reach into the table at the hand's current pose."""
    m, d = sp.model, sp.pl.data
    rig = sp.pl.rigs[side]
    table = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "g_table")
    grip = g.get("grip_fingers")
    best = g["opened"]
    saved = d.qpos.copy()
    for frac in (RELEASE_OPEN_FRAC, 0.7, 0.55):
        op = H.approach_open_targets(rig, g["closed"], frac, grip, g.get("grip_posed", False))
        if table < 0:
            best = op
            break
        rig.set_fingers(d, op); mujoco.mj_forward(m, d)
        worst = min(H.geom_distance(m, d, gg, table, 0.05) for gg in rig.collision_gids)
        best = op
        if worst > -0.002:
            break
    d.qpos[:] = saved; mujoco.mj_forward(m, d)
    return best


def plan_lay(sp, sg, side, obj):
    """{motion: 'lay', side, object (the cable link), pose (a pinch on it: rope_pinch_pose, optimize_grasp_pose family
    'bar', plan_knot_moves' poses), target_pos ([x, y] or [x, y, z]: where the link's body origin lands; z defaults to where
    it lies now, i.e. on the table), via ([[x, y], ...] points the link passes over on the way, in order), lift_m (how far
    the link is lifted; default: its underside LAY_CLEAR_M above the highest strand under the path, at least
    LAY_MIN_LIFT_M), approach_m (0.06), carry_s (1.6), set_s (0.6), open_s (0.6), retreat_m (0.06), expect_order (the
    crossing order the move must leave, knot.crossing_order's code), knot (the knot the rope must be after this move)}."""
    m, d = sp.model, sp.pl.data
    if not obj:
        raise ValueError("lay: give `object` (the cable link to carry)")
    ob = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, obj)
    if ob < 0:
        raise ValueError(f"lay: no body {obj!r}")
    if sg.get("target_pos") is None:
        raise ValueError("lay: give target_pos [x, y] (where the link lands)")
    if sp.state["held"].get(side) not in (None, obj):
        raise ValueError(f"lay: the {side} hand holds {sp.state['held'][side]!r}; let go first")
    pre, bodies = cable_of(m, ob)
    rad = cable_radius(m, bodies)
    mujoco.mj_forward(m, d)
    p_link = d.xpos[ob].copy()
    tgt = np.asarray(sg["target_pos"], float)
    tgt = np.array([tgt[0], tgt[1], float(tgt[2]) if len(tgt) > 2 else float(p_link[2])])
    vias = [np.asarray(v, float)[:2] for v in (sg.get("via") or [])]
    i_ob = bodies.index(ob)
    skip = set(bodies[max(0, i_ob - 2):i_ob + 3])
    path = [p_link[:2]] + vias + [tgt[:2]]
    if sg.get("lift_m") is not None:
        z_carry = float(p_link[2]) + float(sg["lift_m"])
    else:
        top = strand_top_along(m, d, bodies, skip, path, rad)
        z_carry = max(float(p_link[2]) + LAY_MIN_LIFT_M, top + rad + LAY_CLEAR_M)
    pose = sp._pose(sg)
    g = sp._approach_pose(side, pose, obj, approach_m=float(sg.get("approach_m", 0.06)))
    sp.pl.attach(side, obj)
    sp.pl.fingers(side, g["press"])
    pw, Rw = sp.pl.wrist(side)
    sp.pl.move_wrist(side, pw + np.array([0.0, 0.0, z_carry - float(p_link[2])]), Rw)
    sp.pl.push(f"{side}: lift {obj}", float(sg.get("lift_s", 0.6)), 0.1, side, [obj], anchor=obj)
    legs = vias + [tgt[:2]]
    t_c = float(sg.get("carry_s", 1.6))
    L = [float(np.linalg.norm(b - a)) for a, b in zip([p_link[:2]] + legs[:-1], legs)]
    for i, xy in enumerate(legs):
        share = t_c * (L[i] / max(sum(L), 1e-9))
        sp.pl.carry_to(side, np.array([xy[0], xy[1], z_carry]), t=max(0.3, share), anchor=None, label=f"{side}: carry {obj}")
    sp.pl.carry_to(side, tgt, t=float(sg.get("set_s", 0.6)), anchor=None, label=f"{side}: set down {obj}")
    # let go where it was set down: the fingers open with the wrist still, then the hand leaves along its approach
    op = _release_open(sp, side, g)
    sp.pl.fingers(side, op)
    sp.pl.detach(side)
    sp.pl.push(f"{side}: let go of {obj}", float(sg.get("open_s", 0.6)), 0.4, side, [obj], anchor=None)
    pw, Rw = sp.pl.wrist(side)
    a = P.retreat_axis(pose, Rw)
    sp.pl.move_wrist(side, pw + a * float(sg.get("retreat_m", 0.06)), Rw)
    sp.pl.push(f"{side}: retreat", float(sg.get("retreat_s", 1.0)), 0.2, side, [obj], anchor=None)
    sp.state["held"][side] = None
    (sp.state.get("in_hand") or {}).pop(side, None)
    exp = {"object": obj, "target_pos": tgt.round(5).tolist(), "target_axes": "xy", "released": True, "at_rest": True,
           "involved": [sp.model.body(b).name for b in bodies]}
    if sg.get("expect_order"):
        exp["crossings"] = {"code": str(sg["expect_order"]), "rope": pre}
    if sg.get("knot"):
        exp["knot"] = sg["knot"]
    exp["lay"] = {"carry_z": round(z_carry, 4), "lift_m": round(z_carry - float(p_link[2]), 4), "via": [v.round(4).tolist() for v in vias]}
    return exp


def plan_drag(sp, sg, side, obj):
    """{motion: 'drag', ...lay's parameters}: the link is pinched and SLID along the table (held DRAG_LIFT_M off it unless
    `lift_m` says otherwise) to target_pos and let go in place. The rope behind it slides too: a tail pulled along
    itself through a crossing keeps its over/under there (a pull-through or tuck), a strand dragged into a resting one
    rides over it. knotplan.lay predicts it with the same lift."""
    out = plan_lay(sp, dict(sg, lift_m=float(sg["lift_m"]) if sg.get("lift_m") is not None else DRAG_LIFT_M), side, obj)
    out["lay"]["kind"] = "drag"
    return out


# ----------------------------------------------------------------------------- outcome
def _relabel(L, O):
    names, out = {}, []
    for l in L:
        names.setdefault(l, chr(ord("a") + len(names)))
        out.append(names[l])
    return "".join(out), "".join(O)


def matches_order(got_letters, got_over, want_letters, want_over):
    """The executed crossing order is the planned one, or becomes it when curls the plan did not have are dropped (a
    crossing whose two visits are next to each other: a small loop of the executed rope). The plan's own crossings are
    never dropped: a planned loop ('a- a+') is not met by a rope with no crossing."""
    want = _relabel(want_letters, want_over)
    L, O = list(got_letters), list(got_over)
    while True:
        if _relabel(L, O) == want:
            return True
        i = next((i for i in range(len(L) - 1) if L[i] == L[i + 1]), None)
        if i is None or len(L) <= len(want[0]):
            return False
        del L[i:i + 2]
        del O[i:i + 2]


def parse_code(code):
    toks = [t for t in str(code).split() if t]
    return "".join(t[:-1] for t in toks), "".join(t[-1] for t in toks)


def crossing_checks(model, data, Q, t1, expect, checks, meas):
    """expect['crossings'] = {code, rope?}: the cable's crossing order at the subgoal's last frame (kinks dropped) must be
    the planned one."""
    want = expect.get("crossings")
    if not want:
        return checks
    from tool_library.verifiers.rope_knot_check import _cable_bodies, _centreline
    pre, bodies = _cable_bodies(model, want.get("rope"))
    data.qpos[:] = Q[t1]; mujoco.mj_kinematics(model, data)
    od = K.crossing_order(_centreline(model, data, bodies, pre))
    meas["crossings"] = {"code": od["code"], "expected": want.get("code", "")}
    checks["crossings"] = matches_order(od["letters"], od["over"], *parse_code(want.get("code", "")))
    if not checks["crossings"]:
        meas["hint"] = (meas.get("hint", "") + " " if meas.get("hint") else "") + (
            f"the rope's crossings after the move are '{od['code'] or 'none'}', the plan expected '{want.get('code', '') or 'none'}': "
            "re-plan the remaining moves from the executed state (plan_knot_moves with this trajectory and frame)")
    return checks
