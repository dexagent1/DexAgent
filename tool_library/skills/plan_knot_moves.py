"""Plan the over-moves that tie a knot in a cable lying on the bench."""
from __future__ import annotations

TOOL = {
    "name": "plan_knot_moves",
    "title": "Plan knot moves",
    "type": "skill",
    "category": "Trajectory & Execution",
    "summary": "Search the sequences of over-moves (pick a link, carry it just clear of the strands to a landing point, set it "
               "down) that tie a cable on the table into a knot (an overhand / trefoil by default), each move predicted "
               "follow-the-leader and checked by its crossing order, keeping only pinches and landings this bench's arms "
               "hold; returns `lay` plan entries for execute_subgoal_plan and the crossings each move must leave.",
    "description": (
        "A hand holding a rope end cannot pass it UNDER a strand on the table (a fingers-down pinch stands about 0.18 m "
        "tall, a lifted span leaves less than 0.1 m), but every crossing a knot needs can be made by laying rope OVER rope: "
        "an end that must go under a strand is tied by laying that strand over the end. The planner works on the cable's "
        "centreline at a state (a trajectory frame or the scene keyframe): the vertices of construct_rope's links, the "
        "segment lengths and, for every crossing, which strand is on top. A move picks one link, lifts it, carries it "
        "along a straight path or, for an end, through one via point (a straight rope's first loop, a tail swept round a "
        "crossing) and sets it down. Its prediction (common/knotplan.py) is follow-the-leader with friction: a segment the "
        "move stretches is pulled back to its length by moving both its ends (the least motion that keeps the rope "
        "inextensible), slack rope does not move, the rope bends round the hand with a radius, rope that moved lies on top "
        "of rope that did not, a tail slid through a crossing keeps its order there; pinning or overlapping moves are "
        "refused, and so is a move that would close a loop tighter than a thick rope lies (18 cm of rope, or 10 radii, "
        "between a crossing's two visits) or stack two crossings on one spot. Measured on an executed 40 cm carry of a "
        "1.55 m, 3.2 cm cable end: pure follow-the-leader 0.20 m mean "
        "from the executed rope, this model 28 mm with the crossing order right. The knot is read from the predicted diagram "
        "exactly as rope_knot_check reads a real rope (knot.check; knot.quick_type screens), and each move's crossing ORDER "
        "is recorded (knot.crossing_order: 'a+ b- c+ a- b+ c-' is an overhand, a nested 'a b c c b a' a fold). An overhand "
        "is searched first as its own three over-moves (knotplan.overhand_search: a link near one end laid across the "
        "other strand, the loop; a link near the other end laid over the loop, its tail on top; a later piece of the loop "
        "laid over that tail; landings are tried just past every strand, however far, and on rings around the link), "
        "then, for anything else, by beam search: "
        "states ranked by the largest set of pairwise interleaved crossings (three only when they alternate over and under, "
        "as a trefoil's do), the free tail length at the ends and the cost (moves, carry length, a via, a change of hand, a "
        "pin, distance from the human's moves when `prefer` gives them); the beam keeps at most a third per crossing order. "
        "A pick must lie flat (no crossing on it or its neighbours, no raised link two away), nearly straight and 5 cm clear "
        "of other strands (the pads straddle the link; a pinch next to a raised link ran the thumb into it), a landing "
        "`land_clear_m` (3.5 cm) clear. Reach: one pinch per side (`pinch_poses`: a pinch with its `bar` record, best one proved by grasp stress "
        "tests such as optimize_grasp_pose family 'bar'; without one it is solved here: a level thumb-index pinch, the hand "
        "leaning in from its station, carried to a clear link the arm holds it at and optimized there with lift and hold "
        "tests: measured, a level unoptimized pinch lost a 0.86 kg cable at the start of the carry, the optimized one carried "
        "it) is carried rigidly to each candidate link, either way round, leaning 20 degrees along the link if needed, and "
        "the station's arm must hold it (6-DoF IK of the flange) at the pre-grasp, the grasp, lifted, at the via, over and "
        "at the landing and after the retreat. With `allow_pins` a move may have the OTHER hand pin a link 3-12 links away "
        "with its pinch (a grip `pin` entry before it, a `release` after): the drag stops there. Every plan found is "
        "re-screened with the arm checked against the scene (armhand.screen_poses) and simulated on the cable alone "
        "(common/ropesim.py: the cable's own asset on the table in the episode's contact physics, the link carried as the "
        "executor's lay carries it; 10-11 mm mean from executed ropes on the bench); plans whose simulated cable is not the "
        "knot are dropped and the simulated crossing orders become the entries' expect_order. Of the rest, the cheapest "
        "with the widest landing margin (the share of landings 2 cm around the planned one that still give the knot in the "
        "prediction) comes first. Execute the first move (its entries share an index), then call again from the executed "
        "trajectory (closed loop): a lay whose expect_order fails left other crossings than planned; drop expect_order to "
        "accept the rope as it lies and plan on from it. `moves` skips the search and plans the moves YOU give (from the "
        "video, or the rest of an earlier plan re-planned from the executed rope): each is predicted, screened for reach "
        "and simulated the same way, and the entries say what crossing order each must leave. When no full tie is found "
        "(or the given moves stop early) the result is `partial`: the furthest sequence, with its predicted orders, to "
        "execute and plan again from."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "Bench scene MJCF the state belongs to (Stage 2 scene.xml, or an executed scene_physics.xml)."},
        {"name": "state", "type": "str | None", "description": "trajectory.npz (qpos) to read the cable from; None: the scene keyframe."},
        {"name": "frame", "type": "int", "description": "Frame of the state. Default -1."},
        {"name": "rope", "type": "str | None", "description": "Name prefix of the cable's bodies; default the only cable."},
        {"name": "goal", "type": "str", "description": "'trefoil' (an overhand, default) or 'figure-eight'."},
        {"name": "sides", "type": "list", "description": "Hands that may act. Default ['left', 'right']."},
        {"name": "pinch_poses", "type": "dict | None", "description": "{side: pose dict or JSON path}: a verified pinch on one of the cable's links with its `bar` {p0, p1, radius}."},
        {"name": "scene_recon_json", "type": "str | None", "description": "Stage 2 manifest: used to build a hand scene and solve the pinch for a side without one."},
        {"name": "max_moves", "type": "int", "description": "Longest sequence searched. Default 3."},
        {"name": "beam", "type": "int", "description": "States kept per depth. Default 8."},
        {"name": "lift_m", "type": "float", "description": "Carry height of the picked link above the table in the prediction (the executor lifts it clear of the strands under its path). Default 0.05."},
        {"name": "prefer", "type": "list | None", "description": "Per move, what the human did: {side, link (body name) or end ('first' | 'last': the B_first or B_last end), target [x, y]}; leaving it costs."},
        {"name": "bounds", "type": "list | None", "description": "[xmin, xmax, ymin, ymax] where rope may land. Default: the table top 3 cm in from its edges."},
        {"name": "time_budget_s", "type": "float | None", "description": "Stop the search after this long. Default 900."},
        {"name": "out_dir", "type": "str", "description": "Folder for the pose JSONs and plan.json."},
        {"name": "index0", "type": "int", "description": "Index of the first entry written. Default 1."},
        {"name": "simulate", "type": "bool", "description": "Check every plan found on the cable simulated alone (common/ropesim.py, needs scene_recon_json) and take its crossing orders as expect_order. Default true."},
        {"name": "verbose", "type": "bool", "description": "Print the search frontier per depth. Default false."},
        {"name": "workers", "type": "int", "description": "Processes optimize_grasp_pose may use for a pinch it solves here. Default 2."},
        {"name": "allow_pins", "type": "bool", "description": "Also search moves during which the OTHER hand pins a link with its pinch (a grip `pin` entry before the lay, a `release` after): the drag then stops at the pinned link. Default false."},
        {"name": "moves", "type": "list | None", "description": "Plan these moves instead of searching: [{link (body name) | end ('first' | 'last') | k (link index), target [x, y] (where the link's body origin lands), via [[x, y], ...], side (default: the first hand that holds it), pins [link names]}], in order."},
        {"name": "land_clear_m", "type": "float", "description": "How far a landed link must lie from every other strand (the fingers open beside it). Default 0.035."},
    ],
    "outputs": [
        {"name": "result", "type": "dict", "description": "ok; partial (entries that do not finish the knot: the furthest sequence found, or the given moves up to the first one refused); entries (the best plan's `lay` entries, ready to paste into execute_subgoal_plan's subgoals); plans [{entries, predicted [{move, code, type}], cost, margin, simulated}]; start {code, type}; templates {side: pose json}; reachable_picks {side: n}; stats; reason when no full plan was found; plan_json."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy", "sympy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "rope", "cable", "knot", "planning", "deformable", "bimanual"],
    "used_in": ["Knot the Rope"],
    "status": "tested",
    "entrypoint": "plan_knot_moves",
}

import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from tool_library.common import armhand as AH
from tool_library.common import episode as E
from tool_library.common import hand as H
from tool_library.common import knot as K
from tool_library.common import knotplan as KP
from tool_library.common import plan as P
from tool_library.common import ropemoves as RM

APPROACH_M = 0.06
RETREAT_M = 0.06
# the hand may lean along the link about the line through its pads (the pinch itself unchanged): the arm reaches some
# links only leaning 20 deg one way or the other
LEANS_DEG = (0.0, 20.0, -20.0)


def _load(p):
    if isinstance(p, (str, Path)):
        d = json.loads(Path(p).read_text())
        if isinstance(d, dict) and "pose" in d and "pos" not in d:
            d = d["pose"]
        if isinstance(d, list):
            d = d[0]
        return d
    return dict(p)


def _frame_of_bar(bar, s=0.5):
    p0, p1 = np.asarray(bar["p0"], float), np.asarray(bar["p1"], float)
    x = p1 - p0
    x[2] = 0.0
    x /= max(np.linalg.norm(x), 1e-9)
    z = np.array([0.0, 0.0, 1.0])
    R = np.column_stack([x, np.cross(z, x), z])
    return p0 + float(s) * (p1 - p0), R


class PinchTemplate:
    """A verified pinch expressed in its link's frame (origin at the pinch point on the axis, x along the link, z up), so
    it can be put on any link of the same cable lying on the table, either way round."""

    def __init__(self, pose, source=None):
        self.pose = _load(pose)
        if not self.pose.get("bar"):
            raise ValueError("plan_knot_moves: a pinch template needs its `bar` record {p0, p1, radius} (optimize_grasp_pose family "
                             "'bar', rope_pinch_pose, handle_pinch_pose give it)")
        s = float((self.pose.get("straddle") or {}).get("along_segment", 0.5))
        self.s = s
        o, R = _frame_of_bar(self.pose["bar"], s)
        p, Rh = P.pose_from_dict(self.pose)
        self.o, self.R = o, R
        self.p_rel = R.T @ (p - o)
        self.R_rel = R.T @ Rh
        self.source = source or self.pose.get("source") or "pinch template"

    def at(self, a, b, z, flip=False, lean_deg=0.0):
        """The pose on the link from vertex a to b (xy) whose axis lies at height z; `flip` pinches it the other way round,
        `lean_deg` leans the hand along the link about the line through the two pads (the pads stay where they are)."""
        a3, b3 = np.r_[a, z], np.r_[b, z]
        if flip:
            a3, b3 = b3, a3
        o, R = _frame_of_bar({"p0": a3, "p1": b3}, self.s)
        if lean_deg:
            R = Rot.from_rotvec(np.radians(float(lean_deg)) * R[:, 1]).as_matrix() @ R
        Rw = R @ self.R_rel
        pw = o + R @ self.p_rel
        T = R @ self.R.T                      # template frame -> this link's frame (a turn about z, and the lean)
        q = dict(self.pose)
        q["pos"] = pw.tolist()
        q["euler"] = H.mat_to_euler(Rw).tolist()
        sh = lambda v: (T @ (np.asarray(v, float) - self.o) + o).tolist()
        for key in ("targets", "points"):
            if isinstance(self.pose.get(key), dict):
                q[key] = {f: sh(v) for f, v in self.pose[key].items()}
        if isinstance(self.pose.get("normals"), dict):
            q["normals"] = {f: (T @ np.asarray(v, float)).tolist() for f, v in self.pose["normals"].items()}
        for key in ("approach", "pinch_dir"):
            if self.pose.get(key) is not None:
                q[key] = (T @ np.asarray(self.pose[key], float)).tolist()
        q["bar"] = {"p0": a3.tolist(), "p1": b3.tolist(), "radius": self.pose["bar"].get("radius")}
        for key in ("optimize", "stress_tests", "anchor", "reanchored"):
            q.pop(key, None)
        return q


class Reach:
    """The station's arm holding a pose, cached by (side, cell): 6-DoF IK of the flange (armhand.ArmSide), warm-started,
    multistart only when the warm start fails."""

    def __init__(self, model, q_base, sides):
        self.model = model
        self.q = np.asarray(q_base, float).copy()
        self.parts = E.robot_parts(model)
        self.S = {s: AH.ArmSide(model, s, parts=self.parts) for s in sides if s in self.parts}
        self.rigs = P.scene_rigs(model)
        self.prev = {s: S.home.copy() for s, S in self.S.items()}
        self.cache = {}
        self.calls = 0

    def ok(self, side, pos, euler, n_seeds=6):
        key = (side, tuple(np.round(np.asarray(pos) / 0.008).astype(int)), tuple(np.round(np.asarray(euler) / np.radians(6)).astype(int)))
        if key in self.cache:
            return self.cache[key]
        S = self.S[side]
        d = S.data
        d.qpos[:] = self.q
        self.rigs[side].teleport(d, np.asarray(pos, float), np.asarray(euler, float))
        qq = d.qpos.copy()
        pf, Rf = S.flange_target(qq)
        self.calls += 1
        # a screen, not the executor's solve: the warm start, home and the library seeds nearest in pose space (the
        # executor's coupling solves the chosen plan again with its full multistart)
        good = False
        for seed in [self.prev[side], S.home] + S.nearest_seeds(pf, Rf, k=int(n_seeds)):
            ep, er, qa = S._ik(qq, pf, Rf, np.asarray(seed, float), S.cfg["ik_iters"])
            if S.ok(ep, er):
                good = True
                self.prev[side] = qa
                break
        self.cache[key] = good
        return good

    def pose_ok(self, side, pose, offsets=((0.0, 0.0, 0.0),), n_seeds=6):
        p, R = P.pose_from_dict(pose)
        e = pose["euler"]
        return all(self.ok(side, p + np.asarray(o, float), e, n_seeds=n_seeds) for o in offsets)


def _table_bounds(model, data, margin=0.03):
    g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "g_table")
    if g < 0 or int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_BOX):
        return None
    c, h = data.geom_xpos[g], model.geom_size[g]
    return (float(c[0] - h[0] + margin), float(c[0] + h[0] - margin), float(c[1] - h[1] + margin), float(c[1] + h[1] - margin))


def _solve_template(scene_recon_json, side, rp, names, model, bodies, cand, base, state, frame, out_dir, bench_scene, reach, workers=2):
    """A pinch for one side, proved by the grasp stress tests. A level thumb-index pinch (handle_pinch_pose, the hand leaning
    in from its station's side) is solved on one clear link, carried to the clear links nearest the station's reach and
    turned either way round until the arm holds it there (the reach screen), then optimized on that link
    (optimize_grasp_pose family 'bar': the arm screened, lift and hold tested). When nothing passes, the start itself
    (no stress test: its lays are proved only by their outcomes). Measured on the bench: a level handle_pinch_pose pinch
    lifted a 3.2 cm, 0.86 kg cable end 3 cm and lost it at the start of the carry; the optimized pinch of the same cable
    carried its end 40 cm."""
    from tool_library.skills.hand_scene_from_recon import hand_scene_from_recon
    from tool_library.skills.handle_pinch_pose import handle_pinch_pose
    man = json.loads(Path(scene_recon_json).read_text())
    target = next((o["id"] for o in man.get("objects", []) if names[0] in (o.get("bodies") or []) or o.get("root_body") == names[0]), None)
    if target is None:
        target = next((o["id"] for o in man.get("objects", []) if o.get("skill") in ("construct_rope", "construct_cuttable_rope")), None)

    def geom_of(k):
        return model.geom(next(i for i in range(model.ngeom) if int(model.geom_bodyid[i]) == bodies[k]
                               and int(model.geom_type[i]) == int(mujoco.mjtGeom.mjGEOM_CAPSULE))).name

    order = sorted(cand, key=lambda k: abs(float(np.linalg.norm(rp.centre(k) - base)) - 0.42))
    k0 = order[0]
    hs0 = Path(out_dir) / f"hs_{side}_start.json"
    hand_scene_from_recon(scene_recon_json, side, target, grasp_body=names[k0], state=state, frame=frame, out_json=str(hs0))
    # the hand comes in leaning from its station's side (as a person reaches): measured on the bench, a level thumb-index
    # pinch solved that way was in the right arm's reach at link headings 60-240 deg, one solved from straight above
    # only at 120-240 (with the pinch either way round, every heading against two thirds of them)
    t = np.asarray(base - rp.centre(k0), float)
    t = t / max(np.linalg.norm(t), 1e-9)
    a = np.r_[0.6 * t, 1.0]
    start = handle_pinch_pose(str(hs0), bar=geom_of(k0), s=0.5, fingers=["thumb", "index"], tilt_deg=0.0,
                              approach_dirs=[(a / np.linalg.norm(a)).tolist()])
    start["source"] = "handle_pinch_pose (thumb-index, pads level across the link, leaning from the station) in plan_knot_moves"
    T0 = PinchTemplate(start)
    z = float(start["bar"]["p0"][2])
    pick = None
    for k in order[:12]:
        for flip in (False, True):
            for lean in LEANS_DEG:
                q = T0.at(rp.xy[k], rp.xy[k + 1], z, flip=flip, lean_deg=lean)
                if reach.pose_ok(side, q, [(0, 0, 0), (0.0, 0.0, 0.05)]):
                    pick = (k, q)
                    break
            if pick:
                break
        if pick:
            break
    k, st = pick if pick else (k0, start)
    hs_path = Path(out_dir) / f"hs_{side}_template.json"
    hand_scene_from_recon(scene_recon_json, side, target, grasp_body=names[k], state=state, frame=frame, out_json=str(hs_path))
    pose = None
    try:
        from tool_library.skills.optimize_grasp_pose import optimize_grasp_pose
        st_file = Path(out_dir) / f"pinch_start_{side}.json"
        st_file.write_text(json.dumps(st, default=float))
        r = optimize_grasp_pose(str(hs_path), family="bar", starts=str(st_file), tests=["grasp_lift_test", "grasp_hold_test"],
                                lift_heights_m=(0.04, 0.07), bench_scene=str(bench_scene), arm_waypoints=[[0.0, 0.0, 0.05]], state=state,
                                bar=geom_of(k), workers=workers, out_json=str(Path(out_dir) / f"pinch_opt_{side}.json"))
        if r.get("ok"):
            pose = dict(r["pose"])
    except Exception as e:                       # the start pose is used; its lays are proved by their outcomes
        st["optimize_error"] = f"{type(e).__name__}: {e}"[:200]
    pose = pose or dict(st, source=start["source"])
    pose.setdefault("bar", st.get("bar"))
    pose["hand_scene"] = str(hs_path)
    pose["template_link"] = names[k]
    return pose


def _rope_sim(scene, scene_recon_json, model, names):
    """The cable alone (common/ropesim.py), from the Stage 2 manifest's asset; None when the manifest does not name it."""
    if not scene_recon_json:
        return None
    from tool_library.common.ropesim import RopeSim
    man_p = Path(scene_recon_json)
    man = json.loads(man_p.read_text())
    for o in man.get("objects", []):
        if names[0] in (o.get("bodies") or []) or o.get("root_body") == names[0]:
            ax = o.get("asset_xml")
            if not ax:
                return None
            ap = Path(ax) if Path(ax).is_absolute() else (man_p.parent / ax)
            am = mujoco.MjModel.from_xml_path(str(ap))
            first = next(am.body(b).name for b in range(am.nbody) if (am.body(b).name or "").endswith("B_first"))
            prefix = names[0][:-len(first)] if names[0].endswith(first) else ""
            return RopeSim(scene, str(ap), prefix)
    return None


def _simulate(sim, q0, moves, radius, z_rest, goal):
    """The plan's moves on the simulated cable, in order: each link lifted clear of the strands under its path as the
    executor's lay lifts it. Returns {ok, codes (crossing order after each move), type, why}."""
    sim.set_from_scene_qpos(q0)
    codes = []
    for i, (k, path, pins) in enumerate(moves):
        m, d = sim.model, sim.data
        b = int(sim.bodies[k])
        p = d.xpos[b].copy()
        skip = set(sim.bodies[max(0, k - 2):k + 3])
        top = RM.strand_top_along(m, d, list(sim.bodies), skip, [p[:2]] + [np.asarray(v)[:2] for v in path], radius)
        z_carry = max(float(p[2]) + RM.LAY_MIN_LIFT_M, top + radius + RM.LAY_CLEAR_M)
        P_ = sim.lay(k, [np.asarray(v)[:2] for v in path], z_carry - float(p[2]), z_land=float(p[2]), pins=pins)
        if P_ is None:
            return {"ok": False, "codes": codes, "why": f"the simulation diverged on move {i + 1}"}
        codes.append(K.crossing_order(P_)["code"])
    t = K.quick_type(P_)
    if t == goal:
        v = K.check(P_)
        t = v["type"] if v["agree"] else "ambiguous"
    out = {"ok": t == goal, "codes": codes, "type": t}
    if t != goal:
        out["why"] = f"the simulated cable ends '{t}' (crossings {codes[-1] or 'none'}), not a {goal}"
    return out


def reach_callbacks(reach, templates, z_rest, radius, lift_m):
    """The search's reach screen for one bench state: pickable(side, rope, k) -> [pick] (the side's pinch carried onto
    link k, either way round, leaning as needed, held by the arm at the pre-grasp, the grasp and lifted) and
    landable(side, rope, k, pick, path) -> bool (the arm holds it over every via, at the landing and after the
    retreat). Also link_z(rope, k) (a link lying on another strand sits a diameter higher) and the pick cache."""

    def link_z(rp, k):
        for c in rp.crossings():
            if k in (c["i"], c["j"]) and c["top"] == k:
                return z_rest + 2.0 * radius
        return z_rest

    def pick_poses(side, rp, k):
        out_ = []
        z = link_z(rp, k)
        for flip in (False, True):
            for lean in LEANS_DEG:
                pose = templates[side].at(rp.xy[k], rp.xy[k + 1], z, flip=flip, lean_deg=lean)
                ap = np.asarray(pose.get("approach", [0, 0, 1]), float)
                ap /= max(np.linalg.norm(ap), 1e-9)
                if reach.pose_ok(side, pose, [(0, 0, 0), tuple(ap * APPROACH_M), (0.0, 0.0, lift_m)]):
                    out_.append({"flip": flip, "lean_deg": lean, "pose": pose, "z": z})
                    break
        return out_

    def landable(side, rp, k, pick, path):
        pose = pick["pose"]
        c = rp.centre(k)
        ap = np.asarray(pose.get("approach", [0, 0, 1]), float)
        ap /= max(np.linalg.norm(ap), 1e-9)
        dz_land = z_rest - pick["z"]
        offs = [np.r_[np.asarray(v, float) - c, 0.0] + [0.0, 0.0, lift_m] for v in path]
        last = np.r_[np.asarray(path[-1], float) - c, dz_land]
        # the landing first (it decides most), warm-started from the pick's arm configuration with two library seeds
        return reach.pose_ok(side, pose, [last] + offs + [last + ap * RETREAT_M], n_seeds=2)

    pick_cache = {}

    def pickable(side, rp, k):
        key = (side, k, tuple(np.round(rp.xy[k] / 0.005).astype(int)), tuple(np.round(rp.xy[k + 1] / 0.005).astype(int)))
        if key not in pick_cache:
            pick_cache[key] = pick_poses(side, rp, k)
        return pick_cache[key]

    return pickable, landable, link_z, pick_cache


def _given_moves(moves, rp0, names, sides, pickable, landable, lift_m):
    """The caller's own move sequence as one plan: each {link | k | end, target [x, y] (where the link's body origin lands),
    via, side, pins}, predicted in order (knotplan.lay) and screened for reach; the first move the predictor refuses or
    no hand reaches stops it. Returns (plan or None, reason)."""
    rp, out = rp0, []

    def prefix(reason):
        return ({"moves": out, "rope": rp, "order": rp.order(), "cost": 0.0, "margin": 0.0} if out else None), reason
    for i, mv in enumerate(moves):
        if mv.get("link") is not None:
            if mv["link"] not in names:
                return prefix(f"move {i + 1}: {mv['link']!r} is not a link of the cable")
            k = names.index(mv["link"])
        elif mv.get("end") in ("first", "last"):
            k = 0 if mv["end"] == "first" else len(names) - 2
        else:
            k = int(mv["k"])
        k = min(max(k, 0), rp.n - 2)
        off = rp.centre(k) - rp.xy[k]                      # the target names the link's origin; the predictor carries its centre
        path = [np.asarray(v, float)[:2] + off for v in (mv.get("via") or [])] + [np.asarray(mv["target"], float)[:2] + off]
        pins = [names.index(p_) if isinstance(p_, str) else int(p_) for p_ in (mv.get("pins") or [])]
        nr, info = KP.lay(rp, k, path, lift_m, pins=pins)
        if nr is None:
            return prefix(f"move {i + 1} ({names[k]} to {list(np.round(mv['target'], 3))}): the prediction refuses it: {info}")
        pick, why = None, "no hand holds the pinch on the link"
        for side in ([mv["side"]] if mv.get("side") else sides):
            for pk in pickable(side, rp, k):
                if landable(side, rp, k, pk, path):
                    pick = (side, pk)
                    break
                why = f"the {side} arm holds the pinch but not the carry or the landing"
            if pick:
                break
        if pick is None:
            return prefix(f"move {i + 1} ({names[k]} to {list(np.round(mv['target'], 3))}): {why}")
        out.append({"side": pick[0], "k": int(k), "path": path, "pins": pins, "pick": pick[1], "order": nr.order(), "path_m": info["path_m"]})
        rp = nr
    return {"moves": out, "rope": rp, "order": rp.order(), "cost": 0.0, "margin": 0.0}, None


def plan_knot_moves(scene, out_dir, state=None, frame=-1, rope=None, goal="trefoil", sides=("left", "right"), pinch_poses=None,
                    scene_recon_json=None, max_moves=3, beam=8, lift_m=0.05, prefer=None, bounds=None, time_budget_s=900.0,
                    index0=1, max_results=4, simulate=True, verbose=False, workers=2, allow_pins=False, moves=None,
                    land_clear_m=0.035):
    from tool_library.verifiers.rope_knot_check import _cable_bodies, _centreline
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    goal = KP.GOALS.get(goal, goal)
    if goal not in ("trefoil", "figure-eight"):
        raise ValueError(f"plan_knot_moves: goal must be 'trefoil' or 'figure-eight', got {goal!r}")
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    if state is not None:
        # by joint name: an executed trajectory.npz (rigid robot, scene_physics.xml) is read onto the bench scene
        q0 = E.state_qpos(model, state, int(frame))
    else:
        if model.nkey:
            mujoco.mj_resetDataKeyframe(model, data, 0)
        q0 = data.qpos.copy()
    data.qpos[:] = q0
    mujoco.mj_forward(model, data)
    pre, bodies = _cable_bodies(model, rope)
    names = [model.body(b).name for b in bodies]
    Pc = _centreline(model, data, bodies, pre)
    radius = RM.cable_radius(model, bodies)
    rp0 = KP.Rope.from_points(Pc, radius, z0=0.0)
    start = {"code": rp0.order()["code"], "type": rp0.knot_type()}
    if start["type"] == goal:
        return {"ok": True, "entries": [], "plans": [], "start": start, "reason": f"the rope is already a {goal}"}
    bounds = tuple(bounds) if bounds is not None else _table_bounds(model, data)
    sides = [s for s in sides if s in P.scene_rigs(model)]
    parts = E.robot_parts(model)
    reach = Reach(model, q0, sides)

    # ---- the pinch per side: given, or solved on a link near the station
    templates, tpl_files = {}, {}
    for s_ in sides:
        src = (pinch_poses or {}).get(s_)
        if src is None:
            if not scene_recon_json:
                raise ValueError(f"plan_knot_moves: no pinch pose for the {s_} hand: give pinch_poses[{s_!r}] or scene_recon_json")
            base = data.xpos[int(model.jnt_bodyid[parts[s_]["arm_joints"][0]])][:2]
            cand = [k for k in range(len(bodies) - 1) if KP.pickable_geometry(rp0, k)]
            if not cand:
                raise ValueError("plan_knot_moves: no link lies flat, straight and clear of the others to solve a pinch on")
            src = _solve_template(scene_recon_json, s_, rp0, names, model, bodies, cand, base, state, frame, out, scene, reach, workers=workers)
        templates[s_] = PinchTemplate(src)
        tpl_files[s_] = str(out / f"pinch_template_{s_}.json")
        Path(tpl_files[s_]).write_text(json.dumps(templates[s_].pose, default=float))

    z_rest = float(np.median(Pc[:, 2]))
    pickable, landable, link_z, pick_cache = reach_callbacks(reach, templates, z_rest, radius, lift_m)

    pref = None
    if prefer:
        pref = []
        for p_ in prefer:
            e = dict(p_ or {})
            if e.get("link") in names:
                e["k"] = names.index(e["link"])
            elif e.get("end") in ("first", "last"):            # the end the human moved (Stage 1's object part)
                e["k"] = 0 if e["end"] == "first" else len(names) - 2
            pref.append(e)
    other = {"left": "right", "right": "left"}

    def pin_options(side, rp, k):
        """Links the OTHER hand can hold with its pinch (a grip pin) while this hand lays link k: 3 to 12 links away, flat,
        straight and clear, nearest a crossing first (a pin keeps the drag from pulling the crossings already made)."""
        o = other[side]
        if o not in templates:
            return []
        cr = rp.crossings()
        out_ = []
        for p_ in range(rp.n - 1):
            if not 3 <= abs(p_ - k) <= 12 or not KP.pickable_geometry(rp, p_):
                continue
            if pickable(o, rp, p_):
                d_ = min([abs(p_ - c["i"]) for c in cr] + [abs(p_ - c["j"]) for c in cr] + [99])
                out_.append((d_, p_))
        return [p_ for _, p_ in sorted(out_)[:3]]

    partial = False
    if moves:
        plan, why = _given_moves(moves, rp0, names, sides, pickable, landable, lift_m)
        plans, stats = ([plan] if plan else []), {"given_moves": len(moves)}
        if why:
            stats["moves_refused"] = why
        partial = bool(plan) and (bool(why) or plan["rope"].knot_type() != goal)   # a prefix keeps its crossing orders
    else:
        plans, stats = [], {}
        if goal == "trefoil" and max_moves >= 3 and not prefer:
            # the overhand's own three over-moves first (loop, tail over it, the loop over the tail), searched directly
            plans, stats = KP.overhand_search(rp0, pickable, landable, lift_m=lift_m, bounds=bounds, sides=sides, land_clear_m=land_clear_m,
                                              max_results=max_results, time_budget_s=0.5 * float(time_budget_s), verbose=verbose)
            stats = {"overhand_search": stats}
        if not plans:
            plans, st2 = KP.search(rp0, pickable, landable, goal=goal, max_moves=max_moves, beam=beam, lift_m=lift_m, bounds=bounds,
                                   prefer=pref, max_results=max_results, time_budget_s=time_budget_s, sides=sides, verbose=verbose,
                                   land_clear_m=land_clear_m, pin_options=pin_options if allow_pins and len(sides) == 2 else None)
            stats.update(st2)
        bp = stats.pop("best_partial", None)
        if not plans and bp is not None and bp["progress"] > KP.progress(rp0.order())[0]:
            # nothing within max_moves ties it: hand back the sequence that got furthest (execute it, then plan again)
            plans, partial = [{"moves": bp["moves"], "rope": bp["rope"], "order": bp["order"], "cost": 0.0, "margin": 0.0}], True
            stats["best_partial"] = {"code": bp["order"]["code"], "progress": bp["progress"]}
    stats["ik_calls"] = reach.calls
    reach_summary = {s_: sum(1 for (ss, *_), v in pick_cache.items() if ss == s_ and v) for s_ in sides}
    res = {"ok": False, "start": start, "templates": tpl_files, "stats": stats, "radius_m": radius, "goal": goal,
           "reachable_picks": reach_summary, "plans": []}

    # ---- the plans as entries; each re-screened with the arm checked against the scene and simulated (the cable alone in
    # the episode's contact physics, its link carried as the executor's lay carries it)
    probe = AH.ArmProbe(str(scene))
    sim = _rope_sim(scene, scene_recon_json, model, names) if simulate else None
    for pi, plan in enumerate(plans):
        entries, predicted, ok_all, notes = [], [], True, []
        sim_moves = []
        rp = rp0
        for mi, mv in enumerate(plan["moves"]):
            k, side = mv["k"], mv["side"]
            pose = dict(mv["pick"]["pose"])
            c = rp.centre(k)
            land_c = np.asarray(mv["path"][-1], float)
            tgt_v = land_c - (c - rp.xy[k])
            heading = rp.heading(k)
            if mi == 0:
                b = bodies[k]
                q = Rot.from_matrix(data.xmat[b].reshape(3, 3)).as_quat()
                anchor = {"body": names[k], "pos": data.xpos[b].round(6).tolist(), "quat_wxyz": [float(q[3]), float(q[0]), float(q[1]), float(q[2])]}
            else:
                q = Rot.from_euler("z", heading).as_quat()
                anchor = {"body": names[k], "pos": [float(rp.xy[k][0]), float(rp.xy[k][1]), float(mv["pick"]["z"])],
                          "quat_wxyz": [float(q[3]), float(q[0]), float(q[1]), float(q[2])], "predicted": True}
            pose["anchor"] = anchor
            pose["source"] = f"plan_knot_moves ({templates[side].source}, carried to {names[k]})"
            pose["template"] = tpl_files[side]
            ap = np.asarray(pose.get("approach", [0, 0, 1]), float)
            ap /= max(np.linalg.norm(ap), 1e-9)
            wps = [list(np.r_[np.asarray(v, float) - c, lift_m]) for v in mv["path"]]
            wps.append(list(np.r_[land_c - c, z_rest - mv["pick"]["z"]]))
            scr = AH.screen_poses(model, side, [pose], q0, probe=probe, approach_m=APPROACH_M, waypoints=wps)[0]
            if not scr["ok"]:
                # the pinch the reach screen took first runs the arm into something: the other ways round and leans the
                # arm also holds (the arm can fold onto its own hand for one heading and not for the other)
                for flip in (False, True):
                    for lean in LEANS_DEG:
                        if (flip, lean) == (mv["pick"].get("flip"), mv["pick"].get("lean_deg")):
                            continue
                        alt = templates[side].at(rp.xy[k], rp.xy[k + 1], mv["pick"]["z"], flip=flip, lean_deg=lean)
                        if not landable(side, rp, k, {"pose": alt, "z": mv["pick"]["z"]}, mv["path"]):
                            continue
                        alt.update(anchor=anchor, source=pose["source"], template=pose["template"])
                        s2 = AH.screen_poses(model, side, [alt], q0, probe=probe, approach_m=APPROACH_M, waypoints=wps)[0]
                        if s2["ok"]:
                            pose, scr = alt, dict(s2, note=f"turned (flip {flip}, lean {lean} deg): {scr['reason']}")
                            break
                    if scr["ok"]:
                        break
            if not scr["ok"] and scr.get("arm_hits") and all(any(str(g).startswith(f"robot_{other[side]}_") for g in h.get("geoms", []))
                                                              for h in scr["arm_hits"]):
                # only the other station is in the way: the executor parks an idle arm out of the acting one's path, so
                # the move is screened for reach alone (the executor checks the arm against everything as it plans)
                scr2 = AH.screen_poses(model, side, [pose], q0, probe=None, approach_m=APPROACH_M, waypoints=wps)[0]
                if scr2["ok"]:
                    scr = dict(scr2, note=f"the {other[side]} arm stands in the way at home ({scr['reason']}): it must be parked")
            if not scr["ok"]:
                ok_all = False
                notes.append(f"move {mi + 1}: {scr['reason']}")
            pfile = out / f"plan{pi + 1}_move{mi + 1}_{side}_{names[k]}.json"
            pfile.write_text(json.dumps(pose, default=float))
            rp_next, _ = KP.lay(rp, k, mv["path"], lift_m, pins=mv.get("pins") or ())
            od = rp_next.order()
            last = mi == len(plan["moves"]) - 1
            for p_ in mv.get("pins") or []:
                o = other[side]
                pp = None
                for flip in (False, True):
                    for lean in LEANS_DEG:
                        q_ = templates[o].at(rp.xy[p_], rp.xy[p_ + 1], link_z(rp, p_), flip=flip, lean_deg=lean)
                        if reach.pose_ok(o, q_, [(0, 0, 0), (0.0, 0.0, 0.05)]):
                            pp = q_
                            break
                    if pp is not None:
                        break
                pp = dict(pp or templates[o].at(rp.xy[p_], rp.xy[p_ + 1], link_z(rp, p_)))
                qz = Rot.from_euler("z", rp.heading(p_)).as_quat()
                pp["anchor"] = {"body": names[p_], "pos": [float(rp.xy[p_][0]), float(rp.xy[p_][1]), float(link_z(rp, p_))],
                                "quat_wxyz": [float(qz[3]), float(qz[0]), float(qz[1]), float(qz[2])]}
                pp["source"] = f"plan_knot_moves ({templates[o].source}, carried to {names[p_]})"
                ppf = out / f"plan{pi + 1}_move{mi + 1}_{o}_pin_{names[p_]}.json"
                ppf.write_text(json.dumps(pp, default=float))
                entries.append({"index": int(index0) + mi, "motion": "pin", "side": o, "object": names[p_], "pose": str(ppf), "mode": "grip",
                                "reanchor": "yaw"})
            e = {"index": int(index0) + mi, "motion": "lay", "side": side, "object": names[k], "pose": str(pfile),
                 "target_pos": [round(float(tgt_v[0]), 4), round(float(tgt_v[1]), 4)], "reanchor": "yaw",
                 "expect_order": od["code"]}
            if len(mv["path"]) > 1:
                e["via"] = [[round(float(v[0]), 4), round(float(v[1]), 4)] for v in mv["path"][:-1]]
            if last and not partial:
                e["knot"] = goal
            entries.append(e)
            for p_ in mv.get("pins") or []:
                entries.append({"index": int(index0) + mi, "motion": "release", "side": other[side], "object": names[p_]})
            predicted.append({"move": mi + 1, "code": od["code"], "type": rp_next.knot_type(), "arm_screen": scr})
            sim_moves.append((k, [np.asarray(v, float) - (c - rp.xy[k]) for v in mv["path"]], list(mv.get("pins") or [])))
            rp = rp_next
        rec = {"entries": entries, "predicted": predicted, "cost": plan["cost"], "margin": plan["margin"], "arm_ok": ok_all, "notes": notes}
        if sim is not None:
            sres = _simulate(sim, q0, sim_moves, radius, z_rest, goal)
            rec["simulated"] = sres
            if sres["ok"]:
                # the order each move must leave is the simulated one (centimetres nearer the episode than the prediction)
                for e, code in zip([e_ for e_ in entries if e_["motion"] == "lay"], sres["codes"]):
                    e["expect_order"] = code
            else:
                notes.append(f"simulated: {sres['why']}")
        res["plans"].append(rec)
    if partial:
        for rec in res["plans"]:                          # a prefix is judged by its crossing orders, not by the knot
            sres = rec.get("simulated")
            if sres and not sres["ok"] and sres.get("codes") and len(sres["codes"]) == len(rec["predicted"]):
                sres["ok"] = True
                for e, code in zip([e_ for e_ in rec["entries"] if e_["motion"] == "lay"], sres["codes"]):
                    e["expect_order"] = code
    simok = lambda p_: (p_.get("simulated") or {"ok": True})["ok"]
    res["plans"].sort(key=lambda p_: (not (p_["arm_ok"] and simok(p_)), not p_["arm_ok"], p_["cost"] - 1.5 * p_["margin"]))
    good = [p_ for p_ in res["plans"] if p_["arm_ok"] and simok(p_)]
    if good and partial:
        res.update({"ok": False, "partial": True, "entries": good[0]["entries"], "predicted": good[0]["predicted"],
                    "reason": (f"a prefix, not a {goal}: " + str(stats.get("moves_refused") or f"the moves end '{good[0]['predicted'][-1]['code']}'")
                               + "; execute it and plan again from the executed trajectory")})
    elif good:
        res.update({"ok": True, "entries": good[0]["entries"], "predicted": good[0]["predicted"]})
    else:
        tails = KP.tail_room(rp0)
        why = (f"no sequence of up to {max_moves} over-moves the stations hold ties a {goal} from this state (start '{start['code'] or 'no crossing'}'; "
               f"links each hand can pick here: {reach_summary}; free rope at the ends {tails:.2f} m)")
        if stats.get("moves_refused"):
            why = f"the given moves: {stats['moves_refused']}"
        if res["plans"]:
            why += "; the plans found fail the arm screen or the simulation: " + "; ".join(n for p_ in res["plans"] for n in p_["notes"])[:600]
        res["reason"] = why
        res["entries"] = []
    pj = out / "plan.json"
    pj.write_text(json.dumps({k_: v for k_, v in res.items()}, indent=1, default=float))
    res["plan_json"] = str(pj)
    return res


if __name__ == "__main__":
    import sys
    print(json.dumps(plan_knot_moves(sys.argv[1], sys.argv[2], state=sys.argv[3] if len(sys.argv) > 3 else None), indent=1, default=float)[:4000])
