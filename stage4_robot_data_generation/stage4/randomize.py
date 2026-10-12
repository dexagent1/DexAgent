"""Step 1: randomize the initial state and adapt the trajectory to it.

The verified trajectory proved a set of hand-object RELATIVE motions: the
approach, the grasp, the lift. Those stay valid when the object starts
somewhere else, as long as the hand follows the object. So each frame of an
end effector is anchored to something:

    anchor = an object   the hand keeps its recorded pose relative to where
                         that object started (approach, grasp, lift);
    anchor = "world"     the hand keeps its recorded world pose (home, and any
                         motion toward a goal that did not move);
    blend A -> B         the hand migrates smoothly from one to the other
                         (reaching from home, carrying to the goal).

Moving the object moves every frame anchored to it by the same rigid
transform. The arm joints are then re-solved by inverse kinematics, a carried
object rides its hand, and the result must pass checks.py. A layout that
fails is discarded and another one is drawn.

Goals can be relative to other objects. An object that is put down follows,
from its release on, the body it was put on, in or right beside (`rests_on` /
`inside` / `next_to`, or by default whatever its carrier was anchored to when it
let go), so a brick
stacked on a brick that was itself moved lands on that brick. "Where an
object is" is therefore a function of time: its start displacement until it
is put somewhere, then the displacement of what it rests on. Hands anchored
to an object use that resting displacement; the relations are resolved in
dependency order (a tower from the bottom up) and a cycle is refused.

A deformable object simulated in Isaac Sim (a cord of the task's deformables.json, see deformables.py) is drawn like
an object, by its name, and turns about the middle of its links; hands anchored to it move with it. It has no body in
the rigid runtime scene: the physics replay runs it in Isaac Sim after the robot's MuJoCo replay.
"""
from __future__ import annotations

import copy
import fnmatch
import json
import shutil
import time

import mujoco
import numpy as np

from . import checks as CK
from . import kinematics as K

IDENT = (np.eye(3), np.zeros(3))
MOVE_TOL = 0.002           # m: an object that moved less than this is still at its start


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5


def apply(D, p, R):
    """Rigid world transform D = (R, t) applied to a pose."""
    return D[0] @ p + D[1], D[0] @ R


def is_identity(D):
    return (np.allclose(D[0], np.eye(3), atol=1e-12)
            and np.allclose(D[1], 0.0, atol=1e-12))


class Episode:
    """The scene, the verified trajectory, and everything derived from them."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = K.load_model(cfg.path("scene"))
        z = np.load(cfg.path("trajectory"), allow_pickle=False)
        self.Q = np.asarray(z["qpos"], float)
        self.T = len(self.Q)
        assert self.Q.shape[1] == self.model.nq, (
            f"trajectory has {self.Q.shape[1]} qpos columns, the scene has {self.model.nq}")
        self.fps = float(z["fps"]) if "fps" in z.files else 30.0
        self.labels = ([str(x) for x in z["labels"]] if "labels" in z.files
                       else [""] * self.T)
        m = self.model

        # end effectors, and the arms bolted to them by an equality constraint
        self.sides = {}
        for side, body in cfg["robot"]["end_effectors"].items():
            bid = K.body_id(m, body)
            ee = K.Chain(m, bid)
            follower = None
            for e in range(m.neq):
                kind = int(m.eq_type[e])
                if not m.eq_active0[e] or kind not in (K.WELD, K.CONNECT):
                    continue
                b1, b2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
                if bid not in (b1, b2):
                    continue
                tip = b1 if b2 == bid else b2
                ch = K.Chain(m, tip)
                if set(ch.joints) & set(ee.joints):
                    continue
                d = m.eq_data[e]
                if kind == K.CONNECT:            # anchor in body1, then in body2
                    pos_only, point = True, (d[0:3] if tip == b1 else d[3:6])
                else:                            # weld: anchor in body2, then body2 in body1
                    pos_only = float(d[10]) == 0.0
                    point = d[0:3] if tip == b2 else d[3:6] + K.quat2mat(d[6:10]) @ d[0:3]
                # a position-only joint leaves the arm's orientation free
                follower = {"chain": ch, "point": np.array(point, float), "pos_only": pos_only}
            self.sides[side] = {"ee": ee, "follower": follower}

        # deformable objects simulated in Isaac Sim (deformables.json): not bodies of the rigid runtime scene, moved by
        # their own draw like any object (a planar rigid transform of the cord's initial link poses)
        from . import deformables as DF
        self.deformables = DF.load(cfg)
        cords = self.deformables["cords"] if self.deformables else {}

        # objects that can be re-placed: a free joint, or a static body on the world, or a cord
        self.objects = {}
        for name in cfg["randomize"]["objects"]:
            if name in cords:
                self.objects[name] = {"bid": None, "qadr": None, "cord": True}
                continue
            bid = K.body_id(m, name)
            free = [j for j in range(m.body_jntadr[bid], m.body_jntadr[bid] + m.body_jntnum[bid])
                    if int(m.jnt_type[j]) == K.FREE]
            if free:
                self.objects[name] = {"bid": bid, "qadr": int(m.jnt_qposadr[free[0]])}
            else:
                assert m.body_parentid[bid] == 0 and m.body_jntnum[bid] == 0, (
                    f"'{name}' must have a free joint or be a fixed body on the world")
                self.objects[name] = {"bid": bid, "qadr": None}

        # reference forward kinematics of the verified trajectory
        data = mujoco.MjData(m)
        track = {}
        for side, s in self.sides.items():
            track[("ee", side)] = s["ee"].bid
            if s["follower"]:
                track[("tip", side)] = s["follower"]["chain"].bid
        for name, o in self.objects.items():
            if not o.get("cord"):
                track[("obj", name)] = o["bid"]
        self.ref = {k: (np.zeros((self.T, 3)), np.zeros((self.T, 3, 3))) for k in track}
        for t in range(self.T):
            data.qpos[:] = self.Q[t]
            mujoco.mj_kinematics(m, data)
            for k, bid in track.items():
                p, R = K.body_pose(data, bid)
                if k[0] == "tip":                # track the attachment point itself
                    p = p + R @ self.sides[k[1]]["follower"]["point"]
                self.ref[k][0][t], self.ref[k][1][t] = p, R

        self.schedule = {side: self._schedule(cfg["segments"].get(side, []))
                         for side in self.sides}
        for side, sched in self.schedule.items():
            self._check_schedule(side, sched)
        self.warnings = []
        data0 = mujoco.MjData(m)
        data0.qpos[:] = self.Q[0]
        mujoco.mj_kinematics(m, data0)
        for name, o in self.objects.items():
            spec = cfg["randomize"]["objects"][name]
            if o.get("cord"):
                # a cord turns about the middle of its links; it is carried by no hand in MuJoCo (Isaac holds it)
                if spec.get("carrier") or any(spec.get(k) for k in ("rests_on", "inside", "next_to", "attached_to")):
                    raise ValueError(f"randomize.objects['{name}']: a cord simulated in Isaac Sim takes only dx, dy, dyaw_deg")
                o["t_move"] = DF.first_moved_frame(self.deformables, name, MOVE_TOL)
                o["deformable"] = True
                o["center"] = DF.cord_center(cords[name])
                self._rides(name, o, spec)
                continue
            p = self.ref[("obj", name)][0]
            # first frame the object has clearly left its start pose (a resting
            # object jitters by a fraction of a millimetre in a physics trajectory)
            moved = np.flatnonzero(np.linalg.norm(p - p[0], axis=1) > MOVE_TOL)
            o["t_move"] = int(moved[0]) if len(moved) else None
            # a deformable (a cable: a free root with a chain of joints below it) turns about the middle of its
            # bodies, not about its first segment, so a yaw does not swing its far end across the table
            o["deformable"] = bool(spec.get("deformable"))
            sub = sorted(CK.subtree(m, o["bid"]))
            o["center"] = (data0.xpos[sub].mean(axis=0) if o["deformable"] and len(sub) > 1
                           else data0.xpos[o["bid"]].copy())
            self._rides(name, o, spec)
        self._relations()
        # the collision copy of the scene: every visible geom collides
        self.cmodel, n_on = CK.collision_model(cfg.path("scene"), m, cfg["render"]["visual_groups"])
        self.n_collision_on = n_on
        self.limbs = CK.same_limb(self.cmodel, self.sides)
        self._baseline = None

    # -- anchors -----------------------------------------------------------
    def _schedule(self, segments):
        """Per frame: (anchor a, anchor b, weight of b)."""
        a, b, w = ["world"] * self.T, ["world"] * self.T, np.zeros(self.T)
        for seg in segments:
            if "frames" in seg:
                lo, hi = seg["frames"]
                idx = list(range(max(0, lo), min(self.T, hi + 1)))
            else:
                pats = seg["phases"]
                idx = [t for t, lab in enumerate(self.labels)
                       if any(fnmatch.fnmatchcase(lab, p) for p in pats)]
                assert idx, f"segment {pats} matches no frame of the trajectory"
            names = [seg["anchor"]] * 2 if "anchor" in seg else list(seg["blend"])
            for n in names:
                assert n == "world" or n in self.objects, (
                    f"segment anchor '{n}' is not 'world' or a randomized object")
            runs, start = [], 0
            for i in range(1, len(idx) + 1):
                if i == len(idx) or idx[i] != idx[i - 1] + 1:
                    runs.append(idx[start:i])
                    start = i
            for run in runs:
                for k, t in enumerate(run):
                    a[t], b[t] = names
                    w[t] = 0.0 if "anchor" in seg else float(
                        smoothstep(k / max(1, len(run) - 1)))
        return a, b, w

    def _check_schedule(self, side, sched):
        """Anchors may only change through a blend, or the hand would jump."""
        a, b, w = sched
        for t in range(1, self.T):
            if (a[t], b[t]) == (a[t - 1], b[t - 1]):
                continue                         # same anchor, or inside one blend
            end = b[t - 1] if w[t - 1] >= 0.999 else (a[t - 1] if w[t - 1] <= 0.001 else None)
            start = a[t] if w[t] <= 0.001 else None
            if end is not None and start is not None and end == start:
                continue                         # continuous hand-over
            raise ValueError(
                f"segments for '{side}' jump at frame {t} ('{self.labels[t - 1]}' -> "
                f"'{self.labels[t]}'): from {a[t - 1]}->{b[t - 1]} (w={w[t - 1]:.2f}) "
                f"to {a[t]}->{b[t]} (w={w[t]:.2f}); add a blend between the two anchors")

    def _frame_of(self, rel):
        """A release given as a phase label (wildcards allowed) or a frame index -> frame index (None stays None)."""
        if isinstance(rel, str):
            hits = [t for t, lab in enumerate(self.labels) if fnmatch.fnmatchcase(lab, rel)]
            assert hits, f"release phase '{rel}' matches no frame"
            return hits[0]
        return None if rel is None else int(rel)

    def _fully_on(self, side, name, t):
        a, b, w = self.schedule[side]
        return (a[t] == name and w[t] <= 0.001) or (b[t] == name and w[t] >= 0.999)

    def _rides(self, name, o, spec):
        """The object's carries, in order: o["rides"] = [{side, t_ride, t_release, rest_anchor}].

        `carrier` names the side that carries it (or a list of sides, one per carry: a handover, or a part put
        down by one hand and picked up again by the other), `release` the frame or phase where each carrier lets
        go (a list with one entry per carrier; only the last carry may have none). An object rides carrier i
        from the first frame that hand is fully anchored to it (for the first carry: after the object first
        moved, so a nudge during the reach moves with the object's own start, not with the half-way hand),
        until its release; in between it rests where it was put. The legacy fields (carrier, t_release,
        rest_anchor, t_ride) describe the first carrier and the final release."""
        carriers = spec.get("carrier")
        carriers = [] if carriers is None else ([carriers] if isinstance(carriers, str) else list(carriers))
        rels = spec.get("release")
        rels = list(rels) if isinstance(rels, (list, tuple)) else [rels]
        if rels != [None] and len(rels) > max(1, len(carriers)):
            raise ValueError(f"randomize.objects['{name}']: {len(rels)} releases for {len(carriers)} carriers")
        rels = rels + [None] * (len(carriers) - len(rels))
        for c in carriers:                       # only a declared carrier moves an object
            assert o["qadr"] is not None and c in self.sides, (
                f"'{name}': carrier must be a side with an end effector, and the object free")
        if not carriers and rels[0] is not None:
            raise AssertionError(f"'{name}' has a release but no carrier")
        frames = [self._frame_of(r) for r in rels]
        if any(f is None for f in frames[:-1]):
            raise ValueError(f"randomize.objects['{name}']: every carrier but the last needs a release")
        if any(b <= a for a, b in zip(frames, frames[1:]) if a is not None and b is not None):
            raise ValueError(f"randomize.objects['{name}']: the releases of its carriers are not in time order")
        rides, prev = [], None
        for i, side in enumerate(carriers):
            rel = frames[i]
            a, b, w = self.schedule[side]
            if i == 0:
                t_ride = o["t_move"]
                if o["t_move"] is not None:
                    held = [t for t in range(o["t_move"], self.T if rel is None else rel) if self._fully_on(side, name, t)]
                    if held:
                        t_ride = held[0]
            else:
                held = [t for t in range(prev, self.T if rel is None else rel) if self._fully_on(side, name, t)]
                t_ride = held[0] if held else None
                if t_ride is None:
                    self.warnings.append(f"'{name}': the {side} hand is never anchored to it between its "
                                         f"previous release and its own; that carry is ignored")
            rides.append({"side": side, "t_ride": t_ride, "t_release": rel,
                          "rest_anchor": None if rel is None else (b[rel] if w[rel] >= 0.5 else a[rel])})
            prev = rel
        o["rides"] = rides
        o["carriers"] = carriers
        o["carrier"] = carriers[0] if carriers else None
        o["t_ride"] = rides[0]["t_ride"] if rides else o["t_move"]
        last = next((r for r in reversed(rides) if r["t_release"] is not None), None)
        o["t_release"] = last["t_release"] if last else None
        o["rest_anchor"] = last["rest_anchor"] if last else None

    def _relations(self):
        """Where every released object rests: the body named by `rests_on` /
        `inside` after its final release, else the anchor of its carrier at each
        release frame. Sets o["rest"] (the final one), o["rest_steps"] [(frame,
        body)] and the dependency order self.order (bodies first)."""
        specs = self.cfg["randomize"]["objects"]
        for name, o in self.objects.items():
            spec = specs[name]
            keys = [k for k in ("rests_on", "inside", "next_to") if spec.get(k)]
            if len(keys) > 1:
                raise ValueError(f"randomize.objects['{name}']: give one of rests_on, inside, next_to")
            o["relation"], o["rest"] = None, o["rest_anchor"]
            o["frozen_from"] = None
            carried = o["carrier"] is not None and o["t_move"] is not None
            o["rest_steps"] = [(r["t_release"], r["rest_anchor"]) for r in o.get("rides", [])
                               if r["t_release"] is not None and r["t_ride"] is not None] if carried else []
            # an object rests somewhere new only once it was really carried and let go
            o["rests_from"] = o["rest_steps"][0][0] if o["rest_steps"] else None
            if not keys:
                continue
            kind, body = keys[0], str(spec[keys[0]])
            if body == name:
                raise ValueError(f"randomize.objects['{name}']: {kind} names the object itself")
            if body != "world" and mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body) < 0:
                raise ValueError(f"randomize.objects['{name}']: {kind} '{body}' is not a body of the scene")
            if o["t_release"] is None:
                raise ValueError(
                    f"randomize.objects['{name}']: {kind} needs a carrier and a release (the frame from which "
                    f"the object rests there); for an object that starts on another one, use attached_to")
            # a part of a randomized object (a drawer of a cabinet) rests with that object
            b, rest = (mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body) if body != "world" else 0), body
            while b > 0 and self.model.body(b).name not in self.objects:
                b = int(self.model.body_parentid[b])
            if b > 0:
                rest = self.model.body(b).name
            o["relation"], o["rest"] = kind, rest
            if o["rest_steps"]:                  # the final release puts it there
                o["rest_steps"][-1] = (o["rest_steps"][-1][0], rest)
                # beside a part, it keeps the place that part had when it was put down: the neighbour being carried
                # away later does not drag it along (a part put on or in another one does move with it)
                o["frozen_from"] = o["rest_steps"][-1][0] if kind == "next_to" else None
            if o["t_move"] is None:
                self.warnings.append(f"'{name}' never leaves its start pose in the trajectory, so its {kind} "
                                     f"'{body}' has no effect")
            elif o["rest_anchor"] != rest:
                lab = self.labels[o["t_release"]]
                self.warnings.append(
                    f"'{name}' {kind.replace('_', ' ')} '{body}' from '{lab}', but the {o['carrier']} hand is "
                    f"anchored to '{o['rest_anchor']}' there: it puts the object down relative to "
                    f"'{o['rest_anchor']}', so a layout that moves '{body}' differently makes the object jump "
                    f"at the release; anchor the carry and the set-down to '{body}'")
        # dependency order: an object after the body it rests on (towers bottom up)
        order, state = [], {}

        def visit(n, path):
            if state.get(n) == 2:
                return
            if state.get(n) == 1:
                cyc = path[path.index(n):] + [n]
                raise ValueError("randomize.objects: the resting relations form a cycle "
                                 + " -> ".join(f"'{x}'" for x in cyc)
                                 + "; an object cannot rest on something that rests on it")
            state[n] = 1
            o = self.objects[n]
            deps = []
            for t_r, r in o["rest_steps"]:
                if t_r == o.get("frozen_from") and r in self.objects:
                    # beside a part: only where that part is at this release; a part still at its start then is no
                    # dependency (its later carries do not matter), so placing A next to B and later B on A is no cycle
                    rf = self.objects[r]["rests_from"]
                    if rf is None or rf > t_r:
                        continue
                deps.append(r)
            if not o["rest_steps"]:
                deps.append(o["rest"])
            for r in dict.fromkeys(deps):
                if r is not None and r != n and r in self.objects:
                    visit(r, path + [n])
            state[n] = 2
            order.append(n)

        for n in self.objects:
            visit(n, [])
        self.order = order

    def baseline(self):
        if self._baseline is None:
            self._baseline = CK.penetration_profile(self.cmodel, self.Q, limbs=self.limbs)
        return self._baseline

    # -- layout ------------------------------------------------------------
    def sample_layout(self, rng):
        """One draw: an (dx, dy, dyaw) per object and a start offset per arm joint. An object attached_to another
        gets its own draw only when it has ranges of its own (an offset relative to its support, see
        _displacements); otherwise it moves with its support exactly."""
        objs = {}
        for name, spec in self.cfg["randomize"]["objects"].items():
            if spec.get("attached_to") and not any(spec.get(k) for k in ("dx", "dy", "dyaw_deg")):
                objs[name] = [0.0, 0.0, 0.0]
                continue
            dx = rng.uniform(*spec.get("dx", [0.0, 0.0]))
            dy = rng.uniform(*spec.get("dy", [0.0, 0.0]))
            dyaw = rng.uniform(*spec.get("dyaw_deg", [0.0, 0.0]))
            objs[name] = [float(dx), float(dy), float(dyaw)]
        r = float(self.cfg["robot"]["init_jitter_rad"])
        robot = {}
        for side, s in self.sides.items():
            n = len((s["follower"]["chain"] if s["follower"] else s["ee"]).qadr)
            robot[side] = [float(v) for v in rng.uniform(-r, r, n)]
        return {"objects": objs, "robot": robot}

    def _displacements(self, layout):
        """{object: (R, t)} where every object starts. Objects attached_to another are resolved bottom-up (a cap on
        a bottle on a tray follows the tray): such an object starts wherever its support starts, moved by its own
        draw (when it has ranges) about its own centre in the support's frame; a cycle is refused."""
        D = {"world": IDENT}
        specs = self.cfg["randomize"]["objects"]

        def own(name):
            dx, dy, dyaw = layout["objects"].get(name, [0.0, 0.0, 0.0])
            c = self.objects[name]["center"] if name in self.objects else self.ref[("obj", name)][0][0]
            dR = K.rot_z(np.radians(dyaw))
            return dR, c + np.array([dx, dy, 0.0]) - dR @ c

        state = {}

        def resolve(name, path):
            if name in D:
                return D[name]
            if state.get(name) == 1:
                raise ValueError("randomize.objects: attached_to forms a cycle " + " -> ".join(path + [name]))
            state[name] = 1
            sup = specs[name].get("attached_to")
            if not sup:
                D[name] = own(name)
            else:
                assert sup in specs, f"'{name}' is attached_to '{sup}', which is not a randomized object"
                Ds, Dl = resolve(sup, path + [name]), own(name)     # its own offset, then where its support went
                D[name] = (Ds[0] @ Dl[0], Ds[0] @ Dl[1] + Ds[1])
            state[name] = 2
            return D[name]

        for name in specs:
            resolve(name, [])
        return D

    def plan(self, layout):
        """Every displacement of one layout, per frame, as rigid world transforms.

        D      {object: (R, t)}      where each object starts
        rest   {object: (R[T], t[T])} where each object rests: its start until
                                      its release, then what it was put on or in
        side   {side: [(R, t)] * T}  how each hand's recorded pose moves
        obj    {object: (R[T], t[T])} how each object's recorded pose moves:
                                      resting, riding its carrier, resting again
        Hands anchored to an object follow where it rests (a hand carrying it
        keeps the place it was picked from, so a lift stays rigid with the
        object); a hand anchored to an object that the OTHER hand is carrying at
        that frame (a handover, a tip dipped into a tube the other hand holds up)
        follows that object where it is. Relations resolve in self.order, so a
        stack follows its base."""
        D = self._displacements(layout)
        T = self.T

        def const(d):
            return np.repeat(d[0][None], T, axis=0), np.repeat(d[1][None], T, axis=0)

        ident = const(IDENT)

        def follow(name, target, table, t_r=None):
            """What an object follows once it was put down on `target` (at frame t_r)."""
            if target == name:
                return const(D[name])            # put back where it started
            frozen = t_r is not None and t_r == self.objects[name].get("frozen_from")
            if target in table:
                if frozen:
                    return const((table[target][0][t_r], table[target][1][t_r]))     # next_to: where it was then
                return table[target]
            if frozen and target in D:
                return const(D[target])          # next_to a part still at its start
            return ident                         # the world, or a body that never moves

        rest = {}
        for name in self.order:
            o = self.objects[name]
            Rr, tr = const(D[name])
            for t_r, target in o["rest_steps"]:
                fr, ft = follow(name, target, rest, t_r)
                Rr[t_r:], tr[t_r:] = fr[t_r:], ft[t_r:]
            rest[name] = (Rr, tr)

        # which side carries each object at each frame (None: it rests)
        riding = {}
        for name, o in self.objects.items():
            if o["carrier"] is None or o["t_move"] is None:
                continue
            who = [None] * T
            for r in o["rides"]:
                if r["t_ride"] is None:
                    continue
                for t in range(r["t_ride"], T if r["t_release"] is None else max(r["t_ride"], r["t_release"])):
                    who[t] = r["side"]
            riding[name] = who

        def side_pass(sides, held_now=None, prev=None):
            out = dict(prev or {})
            for side in sides:
                a, b, w = self.schedule[side]
                P, R = self.ref[("ee", side)]

                def anchor(n, t):
                    if n == "world":
                        return IDENT
                    if held_now is not None and riding.get(n, [None] * T)[t] not in (None, side):
                        return held_now[n][0][t], held_now[n][1][t]
                    return rest[n][0][t], rest[n][1][t]

                Dn = []
                for t in range(T):
                    pa, Ra = apply(anchor(a[t], t), P[t], R[t])
                    if a[t] == b[t] or w[t] == 0.0:
                        p2, R2 = pa, Ra
                    else:
                        pb, Rb = apply(anchor(b[t], t), P[t], R[t])
                        p2, R2 = (1 - w[t]) * pa + w[t] * pb, K.slerp(Ra, Rb, w[t])
                    dR = R2 @ R[t].T
                    Dn.append((dR, p2 - dR @ P[t]))
                out[side] = Dn
            return out

        def objects_pass(side_D):
            obj = {}
            for name in self.order:
                o = self.objects[name]
                Ro, to = const(D[name])
                if name in riding:
                    rides = [r for r in o["rides"] if r["t_ride"] is not None]
                    rel = dict(o["rest_steps"])
                    for i, r in enumerate(rides):
                        t_end = T if r["t_release"] is None else max(r["t_ride"], r["t_release"])
                        for t in range(r["t_ride"], t_end):
                            Ro[t], to[t] = side_D[r["side"]][t]
                        if r["t_release"] is not None:  # rests where it was put until the next carry
                            t_next = rides[i + 1]["t_ride"] if i + 1 < len(rides) else T
                            fr, ft = follow(name, rel.get(r["t_release"], r["rest_anchor"]), obj, r["t_release"])
                            Ro[r["t_release"]:t_next], to[r["t_release"]:t_next] = (fr[r["t_release"]:t_next],
                                                                                    ft[r["t_release"]:t_next])
                obj[name] = (Ro, to)
            return obj

        side_D = side_pass(list(self.sides))
        obj = objects_pass(side_D)
        # a hand anchored to an object another hand carries: a second pass with the carried object where it is
        cross = sorted({side for side in self.sides for t in range(T)
                        for n in {self.schedule[side][0][t], self.schedule[side][1][t]}
                        if n in riding and riding[n][t] not in (None, side)})
        if cross:
            side_D = side_pass(cross, held_now=obj, prev=side_D)
            obj = objects_pass(side_D)
        return {"D": D, "rest": rest, "side": side_D, "obj": obj}

    # -- the adaptation ----------------------------------------------------
    def layout_models(self, layout, D):
        """(model, collision model, overrides) of a layout: static objects move in the model itself (and in its
        collision copy); free objects move through their joints and leave the model as it is."""
        m = self.model
        model, cmodel = m, self.cmodel
        overrides = {}
        for name, o in self.objects.items():
            if o.get("cord"):
                continue                             # a cord is moved in Isaac Sim, not in the MuJoCo model
            if o["qadr"] is None and not np.allclose(layout["objects"].get(name, [0.0, 0.0, 0.0]), 0.0):
                if model is m:
                    model, cmodel = copy.deepcopy(m), copy.deepcopy(self.cmodel)
                p, R = apply(D[name], m.body_pos[o["bid"]].copy(), K.quat2mat(m.body_quat[o["bid"]]))
                for mm in (model, cmodel):
                    mm.body_pos[o["bid"]], mm.body_quat[o["bid"]] = p, K.mat2quat(R)
                overrides[name] = {"pos": p.tolist(), "quat_wxyz": K.mat2quat(R).tolist()}
        return model, cmodel, overrides

    def _solver(self, data, Q, rng):
        """One frame's IK solve on `data`, retried from Q's previous frame and then from a few perturbed postures
        when the first start gets caught on a joint stop. The first start that converges wins, so the result stays
        as close to the recorded posture as the target allows."""
        tol_p, tol_r = self.cfg["checks"]["ik_pos_tol_m"], np.radians(self.cfg["checks"]["ik_rot_tol_deg"])

        def solve(ch, t, p, R=None, point=None, rest=None):
            start = data.qpos[ch.qadr].copy()
            best = (ch.solve(data, p, R, point=point, rest=rest), data.qpos[ch.qadr].copy())
            seeds = ([Q[t - 1, ch.qadr]] if t > 0 else []) + [
                start + rng.normal(0.0, 0.25, len(start)) for _ in range(10)]
            for s in seeds:
                if best[0][0] <= tol_p and best[0][1] <= tol_r:
                    break
                data.qpos[ch.qadr] = s                  # retries: no pull toward the
                err = ch.solve(data, p, R, point=point)  # posture that got stuck
                if err[0] + 0.1 * err[1] < best[0][0] + 0.1 * best[0][1]:
                    best = (err, data.qpos[ch.qadr].copy())
            data.qpos[ch.qadr] = best[1]
            return best[0]
        return solve

    def chains(self):
        out = {}
        for side, s in self.sides.items():
            out[f"{side}_ee"] = s["ee"].qadr
            if s["follower"]:
                out[f"{side}_arm"] = s["follower"]["chain"].qadr
        return out

    def free_qadr(self):
        return {n: o["qadr"] for n, o in self.objects.items() if o["qadr"] is not None}

    # -- the adaptation ----------------------------------------------------
    def adapt(self, layout, pl=None):
        """Returns (qpos trajectory, body overrides of static objects, report)."""
        cfg = self.cfg
        pl = pl or self.plan(layout)
        D = pl["D"]
        Q = self.Q.copy()
        report = {"ik": {}}
        model, cmodel, overrides = self.layout_models(layout, D)
        data = mujoco.MjData(model)

        # per side: the world displacement of every frame
        side_D = pl["side"]

        # objects: at the new start pose until they first move, then riding
        # the hand that carries them, then resting where they were put
        for name, o in self.objects.items():
            if o["qadr"] is None:
                continue
            P, R = self.ref[("obj", name)]
            adr = o["qadr"]
            Ro, to = pl["obj"][name]
            for t in range(self.T):
                d = (Ro[t], to[t])
                if is_identity(d):
                    continue                         # untouched frame: keep it exactly
                p2, R2 = apply(d, P[t], R[t])
                q2 = K.mat2quat(R2)
                if np.dot(q2, self.Q[t, adr + 3:adr + 7]) < 0:
                    q2 = -q2                         # stay on the recorded hemisphere
                Q[t, adr:adr + 3] = p2
                Q[t, adr + 3:adr + 7] = q2

        # robot: inverse kinematics on every displaced frame, then the start
        # offset of the arm joints, fading out before the first contact phase
        for side, s in self.sides.items():
            Dn = side_D[side]
            ee, fol = s["ee"], s["follower"]
            arm = fol["chain"] if fol else None
            driver = arm or ee
            jit = np.asarray(layout["robot"].get(side, np.zeros(len(driver.qadr))), float)
            fade = self._jitter_fade(side) if np.any(jit) else np.zeros(self.T)
            worst = {"pos_m": 0.0, "rot_deg": 0.0, "frame": 0}
            moved = 0

            def note(ep, er, t):
                if ep > worst["pos_m"] or np.degrees(er) > worst["rot_deg"]:
                    worst.update(pos_m=max(ep, worst["pos_m"]), frame=t,
                                 rot_deg=max(float(np.degrees(er)), worst["rot_deg"]))

            def seed(ch, t):
                """Where a frame's solve starts. A non-redundant chain carries the
                previous frame's correction forward so it stays on its branch; a
                redundant arm starts from the recorded posture every frame, so its
                solution depends on the frame alone and cannot drift."""
                redundant = ch is arm and fol["pos_only"]
                if t > 0 and not redundant:
                    data.qpos[ch.qadr] = self.Q[t, ch.qadr] + (
                        Q[t - 1, ch.qadr] - self.Q[t - 1, ch.qadr]
                        - (jit * fade[t - 1] if ch is driver else 0.0))

            solve = self._solver(data, Q, np.random.default_rng(0))

            for t in range(self.T):
                if is_identity(Dn[t]) and fade[t] == 0.0:
                    continue                         # untouched frame: keep it exactly
                moved += 1
                data.qpos[:] = Q[t]
                if not is_identity(Dn[t]):
                    seed(ee, t)
                    note(*solve(ee, t, *apply(Dn[t], *[x[t] for x in self.ref[("ee", side)]])), t)
                    Q[t, ee.qadr] = data.qpos[ee.qadr]
                    if arm:
                        seed(arm, t)
                        p, R = apply(Dn[t], *[x[t] for x in self.ref[("tip", side)]])
                        note(*solve(arm, t, p, None if fol["pos_only"] else R,
                                    point=fol["point"], rest=self.Q[t, arm.qadr]), t)
                        Q[t, arm.qadr] = data.qpos[arm.qadr]
                if fade[t] > 0.0:
                    q = Q[t, driver.qadr] + jit * fade[t]
                    q = np.where(driver.limited, np.clip(q, driver.lo, driver.hi), q)
                    if arm:                          # the hand rides the offset arm tip
                        mujoco.mj_kinematics(model, data)
                        p_tip, R_tip = K.body_pose(data, arm.bid)
                        p_tip = p_tip + R_tip @ fol["point"]
                        p_ee, R_ee = K.body_pose(data, ee.bid)
                        data.qpos[arm.qadr] = q
                        mujoco.mj_kinematics(model, data)
                        p2, R2 = K.body_pose(data, arm.bid)
                        p2 = p2 + R2 @ fol["point"]
                        dR = np.eye(3) if fol["pos_only"] else R2 @ R_tip.T
                        note(*solve(ee, t, p2 + dR @ (p_ee - p_tip), dR @ R_ee), t)
                        Q[t, ee.qadr] = data.qpos[ee.qadr]
                    Q[t, driver.qadr] = q
            worst["frames_adapted"] = moved
            report["ik"][side] = {k: (round(v, 6) if isinstance(v, float) else v)
                                  for k, v in worst.items()}

        ck = cfg["checks"]
        report["ik"]["ok"] = all(
            r["pos_m"] <= ck["ik_pos_tol_m"] and r["rot_deg"] <= ck["ik_rot_tol_deg"]
            for r in report["ik"].values() if isinstance(r, dict))
        report["joint_speed"] = CK.check_joint_speed(
            Q, self.Q, self.chains(), self.fps, ck["joint_speed_scale"], ck["joint_speed_floor"])
        report["object_steps"] = CK.check_object_steps(Q, self.Q, self.free_qadr(), ck["object_step_tol_m"])
        report["penetration"] = CK.check_penetration(
            cmodel, Q, self.baseline(), ck["penetration_tol_m"], limbs=self.limbs)
        report["ok"] = all(report[k]["ok"] for k in
                           ("ik", "joint_speed", "object_steps", "penetration"))
        report["failed"] = [k for k in ("ik", "joint_speed", "object_steps", "penetration")
                            if not report[k]["ok"]]
        return Q, overrides, report

    # -- reach-aware pre-screen ----------------------------------------------
    def prescreen(self, layout, pl=None, per_run=3, tol_scale=1.5):
        """The reach of a layout's anchored keyframes, before the full adaptation: on every run of frames in which
        a hand is fully anchored to a moved object (an approach, a grasp, a set-down), its first, middle and last
        frame are solved alone (the hand, then its arm, from the recorded posture). A layout whose keyframes miss
        by more than tol_scale times the reach tolerance cannot pass the reach check and is dropped before any
        other work. Returns {ok, frames_checked, <side>: {pos_m, rot_deg, frame}}."""
        pl = pl or self.plan(layout)
        ck = self.cfg["checks"]
        tol_p, tol_r = tol_scale * ck["ik_pos_tol_m"], tol_scale * ck["ik_rot_tol_deg"]
        data = mujoco.MjData(self.model)
        out, n = {"ok": True}, 0
        for side, s in self.sides.items():
            Dn = pl["side"][side]
            a, b, w = self.schedule[side]
            on = [t for t in range(self.T) if not is_identity(Dn[t])
                  and ((a[t] != "world" and w[t] <= 0.001) or (b[t] != "world" and w[t] >= 0.999))]
            frames, run = [], []
            for t in on + [None]:
                if run and (t is None or t != run[-1] + 1):
                    pick = np.linspace(0, len(run) - 1, min(per_run, len(run))).round().astype(int)
                    frames += [run[i] for i in sorted(set(pick))]
                    run = []
                if t is not None:
                    run.append(t)
            ee, fol = s["ee"], s["follower"]
            arm = fol["chain"] if fol else None
            solve = self._solver(data, self.Q, np.random.default_rng(0))
            worst = {"pos_m": 0.0, "rot_deg": 0.0, "frame": None}
            for t in frames:
                data.qpos[:] = self.Q[t]
                errs = [solve(ee, t, *apply(Dn[t], *[x[t] for x in self.ref[("ee", side)]]))]
                if arm:
                    p, R = apply(Dn[t], *[x[t] for x in self.ref[("tip", side)]])
                    errs.append(solve(arm, t, p, None if fol["pos_only"] else R, point=fol["point"],
                                      rest=self.Q[t, arm.qadr]))
                for ep, er in errs:
                    if ep > worst["pos_m"] or np.degrees(er) > worst["rot_deg"]:
                        worst.update(pos_m=round(max(ep, worst["pos_m"]), 6), frame=int(t),
                                     rot_deg=round(max(float(np.degrees(er)), worst["rot_deg"]), 4))
            n += len(frames)
            out[side] = worst
            out["ok"] &= worst["pos_m"] <= tol_p and worst["rot_deg"] <= tol_r
        out["frames_checked"] = n
        return out

    # -- retiming and checks of a trajectory of any length --------------------
    def retime(self, Q, speed_report):
        """Q slowed down where a robot joint moves faster than the joint-speed check allows (its limit per chain:
        joint_speed_scale x the original episode's fastest motion + joint_speed_floor). Returns (Q', src) with
        src the source frame (float) of every frame of Q'."""
        from . import retime as RT
        cols, vmax = [], []
        for k, adr in self.chains().items():
            v = speed_report.get(k)
            if isinstance(v, dict):
                cols += [int(a) for a in adr]
                vmax += [float(v["limit"])] * len(adr)
        # a joint that jumps half a radian in one frame changed IK branch: that is a failure to report, not a motion to
        # slow down for; and no layout is slowed to more than three times the original's length
        src = RT.retime(Q[:, cols], self.fps, vmax=np.asarray(vmax), ignore_step=0.5,
                        max_frames=3 * len(Q))["src"] if cols else np.arange(len(Q), dtype=float)
        return RT.resample(Q, src, RT.quaternion_slices(self.model)), src

    def check_trajectory(self, Q, cmodel, reach=None):
        """Stage 4's checks on a trajectory that is not a frame-by-frame warp of the original (a re-executed or a
        retimed episode, any length): joint speed against the original's fastest motions, object continuity
        against the original's largest step, no penetration deeper than the original's, and the reach of the arms
        (`reach`: the arm-hand weld residual report of a re-executed episode)."""
        ck = self.cfg["checks"]
        rep = {"joint_speed": CK.check_joint_speed(Q, self.Q, self.chains(), self.fps, ck["joint_speed_scale"],
                                                   ck["joint_speed_floor"]),
               "object_steps": CK.check_object_steps_max(Q, self.Q, self.free_qadr(), ck["object_step_tol_m"]),
               "penetration": CK.check_penetration(cmodel, Q, self.baseline(), ck["penetration_tol_m"], limbs=self.limbs),
               "ik": reach if reach is not None else {"ok": True, "note": "not measured"}}
        keys = ("ik", "joint_speed", "object_steps", "penetration")
        rep["ok"] = all(rep[k]["ok"] for k in keys)
        rep["failed"] = [k for k in keys if not rep[k]["ok"]]
        return rep

    def adapt_reference(self, Qref, Q):
        """The servo commands of the verified episode moved to an adapted trajectory Q.

        Qref is what the hands were commanded to do (fingers closed past the
        object's surface, so they press); self.Q is what they did. Wherever Q
        moved a hand, its command keeps the same offset from the executed hand
        in the hand's own frame: T_cmd' = T_exec' (T_exec^-1 T_cmd), solved for
        the arm joints of the rigid robot. Fingers keep their recorded commands;
        the arm motor controller follows the adapted arm commands in physics."""
        m = self.model
        data = mujoco.MjData(m)
        out = np.array(Qref, float, copy=True)
        worst = 0.0
        for side, s in self.sides.items():
            ee = s["ee"]
            changed = np.flatnonzero(np.any(Q[:, ee.qadr] != self.Q[:, ee.qadr], axis=1))
            for t in changed:
                poses = []
                for q in (self.Q[t], Qref[t], Q[t]):
                    data.qpos[:] = q
                    mujoco.mj_kinematics(m, data)
                    poses.append(K.body_pose(data, ee.bid))
                (pe, Re), (pc, Rc), (pn, Rn) = poses
                p_t, R_t = pn + Rn @ (Re.T @ (pc - pe)), Rn @ (Re.T @ Rc)
                data.qpos[:] = Qref[t]
                data.qpos[ee.qadr] = Q[t, ee.qadr] + (Qref[t, ee.qadr] - self.Q[t, ee.qadr])
                err = ee.solve(data, p_t, R_t)
                worst = max(worst, err[0])
                out[t, ee.qadr] = data.qpos[ee.qadr]
        return out, worst

    def _jitter_fade(self, side):
        """1 while the hand waits at its start pose, fading to 0 as it sets off."""
        P = self.ref[("ee", side)][0]
        moved = np.flatnonzero(np.linalg.norm(P - P[0], axis=1) > 1e-3)
        if not len(moved):
            return np.ones(self.T)
        t0 = int(moved[0])
        n = int(self.cfg["robot"]["init_blend_frames"])
        a, b, w = self.schedule[side]
        for t in range(t0, self.T):                  # finish before the first contact phase
            if (a[t] != "world" and w[t] == 0.0) or (b[t] != "world" and w[t] == 1.0):
                n = min(n, max(1, t - t0))
                break
        fade = np.ones(self.T)
        fade[t0:] = 1.0 - smoothstep((np.arange(self.T - t0)) / n)
        return fade

    def joint_names(self):
        from tool_library.common.episode import joint_names
        return joint_names(self.model)


def save_sample(ep, d, Q, layout, overrides, report, seed, replay=None, labels=None, execution=None, deformables=None):
    """Publish the checked physics states. The warped trajectory is only a proposal. A task with deformables also
    publishes the cords' Isaac episode and its mirror (deformables/), which the export renders."""
    import hashlib
    from pathlib import Path
    if not report.get("ok") or not (report.get("physics") or {}).get("ok"):
        raise ValueError("cannot publish an episode without passing kinematic and physics checks")
    if getattr(ep, "deformables", None) is not None and not (deformables and deformables.get("ok") and deformables.get("scene_mirror")):
        raise ValueError("cannot publish an episode with deformables without their passing Isaac Sim episode")
    if replay is None:
        raise ValueError("cannot publish a kinematic proposal: the actual physics replay is missing")
    if not execution or str(np.asarray(execution.get("robot_schema", "")).item()) != "dexagent.rigid_robot.v1":
        raise ValueError("cannot publish without the rigid runtime control and state record")
    replay = np.asarray(replay, float)
    labels = ep.labels if labels is None else labels
    if replay.shape != np.asarray(Q).shape or len(labels) != len(replay) or not np.isfinite(replay).all():
        raise ValueError("physics replay shape, labels or finite-state check failed")
    if not np.array_equal(np.asarray(execution.get("qpos")), replay):
        raise ValueError("recorded runtime states differ from the verified replay")
    for key, width in (("qvel", ep.model.nv), ("ctrl", ep.model.nu), ("reference", ep.model.nq),
                       ("eq_active", ep.model.neq)):
        array = np.asarray(execution.get(key), float)
        if array.shape != (len(replay), width) or not np.isfinite(array).all():
            raise ValueError(f"missing or invalid runtime {key}")
    required = {"initial_qpos", "initial_qvel", "initial_state", "final_state", "state_spec", "time_s",
                "substep_qpos", "substep_qvel", "substep_ctrl", "substep_time_s", "scene_sha256", "actuator_names"}
    if ep.model.neq:
        required |= {"eq_names", "eq_active_start", "substep_eq_active", "substep_eq_active_before", "substep_eq_data"}
    missing = sorted(required - set(execution))
    if missing:
        raise ValueError("incomplete runtime recording: " + ", ".join(missing))
    from .physics import library
    library()                                     # puts the DexAgent tool library on the path
    from tool_library.common.stage4 import copy_runtime_scene
    scene = Path(str(execution.get("_scene_path", "")))
    if not scene.is_file() or hashlib.sha256(scene.read_bytes()).hexdigest() != str(np.asarray(execution["scene_sha256"]).item()):
        raise ValueError("actual runtime scene is missing or changed after replay")
    copy_runtime_scene(scene, d / "scene_physics.xml")
    if execution.get("_scene_recon_path"):
        recon = Path(execution["_scene_recon_path"])
        if hashlib.sha256(recon.read_bytes()).hexdigest() != str(np.asarray(execution["scene_recon_sha256"]).item()):
            raise ValueError("trusted mechanism evidence changed after replay")
        shutil.copy2(recon, d / "mechanism_scene_recon.json")
    arrays = dict({k: v for k, v in execution.items() if not k.startswith("_")}, qpos=replay, fps=ep.fps, labels=np.array(labels),
                  joint_names=np.array(ep.joint_names()))
    np.savez_compressed(d / "trajectory.npz", **arrays)
    np.savez_compressed(d / "physics_replay.npz", **arrays)
    digest = hashlib.sha256((d / "trajectory.npz").read_bytes()).hexdigest()
    overrides_digest = hashlib.sha256(json.dumps(overrides, sort_keys=True, default=float).encode()).hexdigest()
    extra = {}
    if deformables is not None:
        from . import deformables as DF
        extra["deformables"] = DF.publish(deformables, d)
    (d / "sample.json").write_text(json.dumps(
        {"seed": seed, "layout": layout, "body_overrides": overrides, "checks": report, **extra,
         "trajectory_source": "physics_replay", "trajectory_sha256": digest,
         "body_overrides_sha256": overrides_digest,
         "source_scene_sha256": str(np.asarray(execution.get("source_scene_sha256", execution["scene_sha256"])).item()),
         "runtime_scene": "scene_physics.xml", "runtime_scene_sha256": str(np.asarray(execution["scene_sha256"]).item()),
         "scene_recon_sha256": str(np.asarray(execution.get("scene_recon_sha256", "")).item())},
        indent=1, default=float))


def physics_summary(rep):
    """The part of a physics replay report worth keeping per sample (the full check table is long)."""
    keep = ("ok", "failed", "not_judged", "unstable", "stopped_at_frame", "frames_simulated", "end_error_m",
            "max_hand_pen_mm", "max_object_pen_mm", "seconds", "reference_ik_m", "physical_audits")
    out = {k: rep[k] for k in keep if k in rep}
    out["n_checks"] = len(rep.get("checks", {}))
    out["subgoals"] = [r for r in rep.get("subgoals", []) if not r.get("ok")] or "all ok"
    return out


# ----------------------------------------------------------------------------- drawing layouts
LAYOUT_FAILURES = ("ik", "reach", "joint_speed", "object_steps", "penetration", "physics")
# rejections a re-execution of the Stage 3 plan on the drawn layout can fix: it re-plans every motion from the
# objects' new poses (reach, speed, collisions, jumps) and closes the loop on the contacts (a grip or a press that
# the open-loop replay of the warped commands lost)
REEXECUTE_FIXES = ("ik", "reach", "joint_speed", "object_steps", "penetration", "physics")


class Sampler:
    """Draws layouts inside the configured ranges, steering away from where earlier draws failed.

    The first draw of every attempt is exactly the uniform draw of that attempt's seed. Once draws have failed on
    a layout-dependent check (reach, speed, penetration, continuity, physics), each attempt draws `candidates`
    layouts from its seed and picks one with probability proportional to its distance weight from the earlier
    failures: every failure multiplies a candidate's weight by 1 - strength * exp(-d^2 / 2 bandwidth^2), with d the
    distance in range-normalized coordinates over the objects the failure was blamed on (the object the failing
    hand was anchored to, the bodies of a colliding pair, the objects named by a failed physics check). The support
    never changes, only the odds inside it; the seed, the candidate picked and its weight are recorded per draw."""

    def __init__(self, ep, cfg):
        sc = cfg.get("sampling") or {}
        self.ep = ep
        self.adaptive = bool(sc.get("adaptive", True))
        self.k = max(1, int(sc.get("candidates", 6)))
        self.width = float(sc.get("bandwidth", 0.5))
        self.strength = float(sc.get("strength", 0.85))
        self.fails = []                          # (normalized layout {object: [n...]}, blamed objects)
        self.ranges = {}
        for name, spec in cfg["randomize"]["objects"].items():
            dims = []
            for i, k in enumerate(("dx", "dy", "dyaw_deg")):
                lo, hi = spec.get(k, [0.0, 0.0])
                if hi > lo:
                    dims.append((i, 0.5 * (lo + hi), 0.5 * (hi - lo)))
            if dims and not (spec.get("attached_to") and not any(spec.get(k) for k in ("dx", "dy", "dyaw_deg"))):
                self.ranges[name] = dims

    def normalized(self, layout):
        return {n: np.array([(layout["objects"][n][i] - mid) / half for i, mid, half in dims])
                for n, dims in self.ranges.items() if n in layout["objects"]}

    def weight(self, layout):
        x = self.normalized(layout)
        w = 1.0
        for xf, blamed in self.fails:
            names = [n for n in (blamed or xf) if n in x and n in xf]
            if not names:
                continue
            d2 = sum(float(np.sum((x[n] - xf[n]) ** 2)) for n in names)
            w *= 1.0 - self.strength * np.exp(-d2 / (2.0 * self.width ** 2))
        return w

    def draw(self, rng):
        first = self.ep.sample_layout(rng)
        if not (self.adaptive and self.fails and self.k > 1):
            return first, {"candidates": 1, "picked": 0, "failures_seen": len(self.fails)}
        cands = [first] + [self.ep.sample_layout(rng) for _ in range(self.k - 1)]
        w = np.array([self.weight(c) for c in cands])
        i = int(rng.choice(len(cands), p=w / w.sum())) if w.sum() > 1e-12 else 0
        return cands[i], {"candidates": len(cands), "picked": i, "weight": round(float(w[i] / max(w.max(), 1e-12)), 3),
                          "failures_seen": len(self.fails)}

    def record(self, layout, failed, blamed):
        if self.adaptive and any(f in LAYOUT_FAILURES for f in failed):
            self.fails.append((self.normalized(layout), sorted(blamed or [])))


class Randomizer:
    """One draw of the randomize step, the same for Stage 4 and for the gates that try its draws:

      1. a layout from the (adaptive) sampler;
      2. the reach of its anchored keyframes (pre-screen): a layout the arms cannot reach goes no further;
      3. the warp of the verified trajectory (Episode.adapt) and the kinematic checks; a layout rejected only on
         joint speed is retimed (slowed down where it is too fast) and checked again;
      4. the physics replay of the warped commands (checks.physics_replay);
      5. when the warp is rejected for a reason re-execution fixes and Stage 3 handed off its executable plan
         (reexecute.plan): the Stage 3 executor runs the plan again on the drawn layout, closed-loop in contact
         physics, and the episode it produces must pass the executor's own outcome checks, the same Stage 4 checks
         and the physics replay.
    """

    def __init__(self, ep, cfg, log=print, physics_replay=None, reexecute=None, workdir=None):
        self.ep, self.cfg, self.log = ep, cfg, log
        self.sampler = Sampler(ep, cfg)
        sc = cfg.get("sampling") or {}
        self.prescreen_on = bool(sc.get("prescreen", True))
        self.prescreen_tol = float(sc.get("prescreen_tol_scale", 1.5))
        self.retime_on = bool((cfg.get("retime") or {}).get("enabled", True))
        pr = cfg["checks"].get("physics_replay") if physics_replay is None else physics_replay
        self.phys = None
        if pr:
            from . import physics
            self.phys = physics.Replay(ep, cfg, log)
            baseline = self.phys.original()
            if not baseline["ok"]:
                raise ValueError("original Stage 3 episode fails the physics replay: " + "; ".join(baseline["failed"][:8]))
        rx = cfg.get("reexecute") or {}
        self.reexec, self.reexec_note = None, None
        want = rx.get("enabled", True) if reexecute is None else reexecute
        if want and getattr(ep, "deformables", None) is not None:
            # the executor re-plans rigid objects only: a cord's episode is the warped one, replayed in Isaac Sim
            self.reexec_note = "re-execution is not available for a task with deformables simulated in Isaac Sim"
        elif want:
            from . import reexecute as RX
            self.reexec, self.reexec_note = RX.make(ep, cfg, log, physics=self.phys, workdir=workdir)
        elif rx.get("plan"):
            self.reexec_note = "re-execution switched off"
        self.max_reexec = rx.get("max_draws")
        self.n_reexec = 0

    # -- blame: which objects a rejection points at ----------------------------
    def _object_of_body(self, body_id):
        m = self.ep.model
        b = int(body_id)
        while b > 0 and m.body(b).name not in self.ep.objects:
            b = int(m.body_parentid[b])
        if b <= 0:
            return None
        name = m.body(b).name
        specs = self.cfg["randomize"]["objects"]
        seen = set()
        while specs.get(name, {}).get("attached_to") and name not in seen and not any(
                specs[name].get(k) for k in ("dx", "dy", "dyaw_deg")):
            seen.add(name)
            name = specs[name]["attached_to"]     # an attached part is moved by its support's draw
        return name

    def _objects_named(self, text):
        return {n for n in self.ep.objects if n in str(text)}

    def _drivers(self, name, t=None, seen=None):
        """The drawn objects whose displacement moves `name` (at frame t when given): an object put down on another
        moves with it from its release on, an attached part with its support."""
        seen = set() if seen is None else seen
        if name in seen or name not in self.ep.objects:
            return set()
        seen.add(name)
        o, out = self.ep.objects[name], {name}
        steps = [(tr, tgt) for tr, tgt in o.get("rest_steps", []) if t is None or tr <= t]
        if t is not None and steps:
            steps = steps[-1:]                  # what it rests on at that frame
        for _tr, tgt in steps:
            if tgt in self.ep.objects and tgt != name:
                out |= self._drivers(tgt, t, seen)
        return out

    def blame(self, failed, rep, prep=None):
        ep, out = self.ep, set()
        for key in ("ik", "prescreen"):
            r = rep.get(key) or {}
            for side, v in r.items():
                if isinstance(v, dict) and v.get("frame") is not None and side in ep.schedule and (
                        "ik" in failed or "reach" in failed):
                    a, b, w = ep.schedule[side]
                    t = int(v["frame"])
                    for n in (a[t], b[t]):
                        out |= self._drivers(n, t)
        if "penetration" in failed:
            pair = (rep.get("penetration") or {}).get("pair") or []
            for lab in pair:
                bid = mujoco.mj_name2id(ep.model, mujoco.mjtObj.mjOBJ_BODY, str(lab).split("/")[0])
                n = self._object_of_body(bid) if bid >= 0 else None
                if n:
                    out.add(n)
        if "joint_speed" in failed:
            for k, v in (rep.get("joint_speed") or {}).items():
                if isinstance(v, dict) and v["max"] > v["limit"]:
                    side = k.rsplit("_", 1)[0]
                    if side in ep.schedule:
                        a, b, _ = ep.schedule[side]
                        out |= {n for n in set(a) | set(b) if n in ep.objects}
        if "object_steps" in failed:
            tol = self.cfg["checks"]["object_step_tol_m"]
            out |= {n for n, v in (rep.get("object_steps") or {}).items() if n in ep.objects and v > tol}
        if "physics" in failed and prep is not None:
            for k in prep.get("failed", []):
                out |= self._objects_named(k)
            for r in prep.get("subgoals", []):
                if not r.get("ok"):
                    out |= self._objects_named(r.get("subgoal", "")) | self._objects_named(r.get("measured", ""))
        out = {d for n in out for d in self._drivers(n)} if not ({"ik", "reach"} & set(failed)) else out
        return {(n if ep.objects[n].get("cord") else self._object_of_body(ep.objects[n]["bid"])) or n for n in out}

    # -- one draw ---------------------------------------------------------------
    def warp(self, layout):
        """The warp fast path: (result dict, plan of the layout)."""
        ep = self.ep
        pl = ep.plan(layout)
        res = {"method": "warp", "labels": ep.labels, "replay": None, "timing": None}
        t0 = time.time()
        if self.prescreen_on:
            pre = ep.prescreen(layout, pl, tol_scale=self.prescreen_tol)
            res["prescreen"] = pre
            if not pre["ok"]:
                res.update(ok=False, failed=["reach"], report={"ok": False, "failed": ["reach"], "prescreen": pre},
                           Q=None, overrides={}, kinematic_seconds=round(time.time() - t0, 2))
                return res, pl
        Q, ov, rep = ep.adapt(layout, pl)
        if rep["failed"] == ["joint_speed"] and self.retime_on:
            # too fast, nothing else wrong: slow the fast frames down instead of rejecting the layout
            Qt, src = ep.retime(Q, rep["joint_speed"])
            _model, cmodel, _ov = ep.layout_models(layout, pl["D"])
            speed = CK.check_joint_speed(Qt, ep.Q, ep.chains(), ep.fps, self.cfg["checks"]["joint_speed_scale"],
                                         self.cfg["checks"]["joint_speed_floor"])
            pen = CK.check_penetration(cmodel, Qt, ep.baseline(), self.cfg["checks"]["penetration_tol_m"], limbs=ep.limbs)
            rep["retime"] = {"frames_before": len(Q), "frames_after": len(Qt),
                             "added_s": round((len(Qt) - len(Q)) / ep.fps, 2), "warped_speed": rep["joint_speed"]}
            if speed["ok"] and pen["ok"]:
                from . import retime as RT
                rep["joint_speed"], rep["penetration"] = speed, pen
                # object steps only shrink when a path is resampled more finely: the warp's check still holds
                rep["ok"], rep["failed"] = True, []
                res.update(method="warp+retime", labels=RT.resample_labels(ep.labels, src), timing={"src": src, "warp": Q})
                Q = Qt
            else:
                rep["retime"]["still_failing"] = [k for k, v in (("joint_speed", speed), ("penetration", pen)) if not v["ok"]]
        res.update(ok=rep["ok"], failed=list(rep["failed"]), report=rep, Q=Q, overrides=ov,
                   kinematic_seconds=round(time.time() - t0, 2))
        if rep["ok"] and self.phys is not None:
            t1 = time.time()
            prep, replay = self.phys.check(layout, Q, ov, timing=res["timing"])
            res["physics_seconds"] = round(time.time() - t1, 1)
            res["physics"] = prep
            rep["physics"] = physics_summary(prep)
            res["replay"] = replay
            res["execution"] = self.phys.last_execution
            res["deformables"] = self.phys.last_deformables
            if prep["ok"]:
                # Compliance can move the robot and objects away from the proposal. Check the states that will
                # actually be exported, including every visible collision shape, before accepting them.
                _model, cmodel, _ov = ep.layout_models(layout, pl["D"])
                executed = ep.check_trajectory(replay, cmodel)
                rep["executed_trajectory"] = executed
                if not executed["ok"]:
                    prep["ok"] = False
                    prep["failed"].extend(f"executed trajectory: {k}" for k in executed["failed"])
                    rep["physics"] = physics_summary(prep)
            if not prep["ok"]:
                res["ok"] = False
                res["failed"].append("physics")
                rep["ok"] = False
                rep["failed"].append("physics")
        return res, pl

    def attempt(self, attempt):
        """Draw and judge one layout. Returns the result dict: ok, failed, layout, Q, labels, overrides, report,
        replay, method, row (the line randomize.json keeps)."""
        t0 = time.time()
        rng = np.random.default_rng([int(self.cfg["seed"]), int(attempt)])
        layout, sinfo = self.sampler.draw(rng)
        res, pl = self.warp(layout)
        rep = res["report"]
        row = {"attempt": attempt, "ok": res["ok"], "failed": list(res["failed"]), "layout": layout["objects"],
               "method": res["method"], "kinematic_seconds": res.get("kinematic_seconds"), "sampling": sinfo}
        if res.get("prescreen") is not None:
            row["prescreen"] = {k: v for k, v in res["prescreen"].items() if k != "ok"}
        if rep.get("retime"):
            row["retime"] = {k: v for k, v in rep["retime"].items() if k != "warped_speed"}
        if res.get("physics_seconds") is not None:
            row["physics_seconds"] = res["physics_seconds"]
        prep = res.get("physics")
        if prep is not None and not prep["ok"]:
            row["physics_failed"] = prep["failed"][:12]
            row["physics"] = {k: prep.get(k) for k in ("unstable", "stopped_at_frame", "end_error_m")}
            bad = [r for r in prep.get("subgoals", []) if not r.get("ok")]
            if bad:
                row["physics"]["first_failed_subgoal"] = bad[0]
        for k in ("object_steps", "penetration", "ik"):    # why a kinematic check failed, in one line
            if k in row["failed"]:
                row.setdefault("why", {})[k] = {kk: v for kk, v in rep[k].items() if kk != "ok"}
        blamed = self.blame(res["failed"], {**rep, "prescreen": res.get("prescreen")}, prep)
        if not res["ok"] and blamed:
            row["blamed"] = sorted(blamed)
        # the physics replay of a contact-rich episode can turn on the arms' start offset alone (it changes every
        # contact that follows by a fraction of a millimetre): a layout rejected only there is warped again without it,
        # the objects where they were drawn, before anything costlier is tried
        jit = layout.get("robot") or {}
        if (not res["ok"] and res["failed"] == ["physics"] and self.phys is not None
                and any(np.any(np.asarray(v, float) != 0.0) for v in jit.values())):
            still = {"objects": layout["objects"], "robot": {s: [0.0] * len(v) for s, v in jit.items()}}
            res2, pl2 = self.warp(still)
            row["start_offset_dropped"] = {"ok": res2["ok"], "failed": list(res2["failed"]), "method": res2["method"]}
            if res2.get("physics_seconds") is not None:
                row["physics_seconds"] = round(float(row.get("physics_seconds") or 0.0) + res2["physics_seconds"], 1)
            if res2["ok"]:
                res, pl, layout = dict(res2, method=f"{res2['method']} (no start offset)"), pl2, still
                row.update(ok=True, failed=[], method=res["method"])
        # the fallback: re-execute the Stage 3 plan on this layout
        if (not res["ok"] and self.reexec is not None and set(res["failed"]) & set(REEXECUTE_FIXES)
                and (self.max_reexec is None or self.n_reexec < int(self.max_reexec))):
            self.n_reexec += 1
            rx = self.reexec.run(layout, pl, f"draw_{attempt:04d}")
            row["warp_failed"] = list(res["failed"])
            row["reexecute"] = rx["row"]
            if rx["ok"]:
                res = dict(rx, method="reexecuted", warp=res)
                row.update(ok=True, failed=[], method="reexecuted")
            else:
                row["failed"] = list(res["failed"]) + ["reexecute"]
                res = dict(res, ok=False, failed=row["failed"])
        self.sampler.record(layout, [] if res["ok"] else row.get("warp_failed", row["failed"]), blamed)
        row["seconds"] = round(time.time() - t0, 1)
        res["row"], res["layout"] = row, layout
        return res


def run(cfg, log=print):
    """Draw layouts until num_samples of them pass every check."""
    if not cfg["checks"].get("physics_replay"):
        raise ValueError("Stage 4 output requires physics replay; use Randomizer(physics_replay=False) for diagnostics only")
    ep = Episode(cfg)
    out = cfg.out / "samples"
    shutil.rmtree(out, ignore_errors=True)       # never mix samples of two runs
    # the render bundles describe the previous samples: the export step must run again before render/composite
    # (it reuses rendered frames whose cache key still matches and removes folders of samples that are gone)
    for f in (cfg.out / "render").glob("*/*/scene.json"):
        f.unlink()
    n_want, rows, n_ok = int(cfg["num_samples"]), [], 0
    log(f"[randomize] {ep.T} frames @ {ep.fps:g} fps; end effectors "
        f"{ {s: (('arm follows, position only' if v['follower']['pos_only'] else 'arm follows') if v['follower'] else 'direct') for s, v in ep.sides.items()} }; "
        f"objects { {n: (o['carriers'] if len(o['carriers']) > 1 else o['carrier']) for n, o in ep.objects.items()} }; "
        f"penetration checked on every visible geom ({ep.n_collision_on} visual-only geoms "
        f"made collidable)")
    rel = {n: f"{o['relation'] or 'rests with'} {o['rest']}" for n, o in ep.objects.items() if o["rests_from"] is not None}
    if rel:
        log(f"[randomize] after their release: {rel}")
    for w in ep.warnings:
        log(f"[randomize] warning: {w}")
    rz = Randomizer(ep, cfg, log, workdir=cfg.out / "reexecute")
    phys = rz.phys
    log("[randomize] re-execution of the Stage 3 plan: "
        + ("on" if rz.reexec is not None else f"off ({rz.reexec_note or 'no plan handed off: reexecute.plan'})"))
    if cfg["include_original"]:
        zero = {"objects": {n: [0.0, 0.0, 0.0] for n in ep.objects},
                "robot": {}}
        Q, ov, rep = ep.adapt(zero)
        assert np.array_equal(Q, ep.Q), "a zero layout must reproduce the trajectory exactly"
        if phys is not None:
            rep["physics"] = physics_summary(phys.baseline)
        if not rep["ok"]:
            raise ValueError("original Stage 3 episode fails Stage 4 kinematic checks: " + ", ".join(rep["failed"]))
        _model, cmodel, _ov = ep.layout_models(zero, ep.plan(zero)["D"])
        executed = ep.check_trajectory(phys.baseline_qpos, cmodel)
        if not executed["ok"]:
            raise ValueError("original physics states fail Stage 4 checks: " + ", ".join(executed["failed"]))
        rep["executed_trajectory"] = executed
        save_sample(ep, out / "original", Q, zero, ov, rep, None, replay=phys.baseline_qpos,
                    execution=phys.baseline_execution, deformables=phys.baseline_deformables)
    t_kin = t_phys = t_rx = 0.0
    methods = {}
    for attempt in range(int(cfg["max_attempts"])):
        if n_ok >= n_want:
            break
        res = rz.attempt(attempt)
        row = res["row"]
        t_kin += float(row.get("kinematic_seconds") or 0.0)
        t_phys += float(row.get("physics_seconds") or 0.0)
        t_rx += float((row.get("reexecute") or {}).get("seconds") or 0.0)
        rep = res["report"]
        if res["ok"]:
            row["sample"] = f"{n_ok:04d}"
            rep = dict(rep, method=res["method"])
            if res["method"] == "reexecuted":
                rep["warp_rejected_for"] = row.get("warp_failed")
            save_sample(ep, out / row["sample"], res["Q"], res["layout"], res["overrides"], rep, attempt,
                        res.get("replay"), labels=res.get("labels"), execution=res.get("execution"),
                        deformables=res.get("deformables"))
            n_ok += 1
            methods[res["method"]] = methods.get(res["method"], 0) + 1
        rows.append(row)
        log(f"[randomize] attempt {attempt}: "
            + (f"OK -> {row['sample']} ({res['method']})" if res["ok"] else f"rejected ({', '.join(row['failed'])})")
            + (f" {'warp ' if res['ok'] else ''}physics: {row['physics_failed'][:3]}" if row.get("physics_failed") else "")
            + (f" re-execution: {row['reexecute'].get('why')}" if row.get("reexecute") and not res["ok"] else "")
            + f"  {json.dumps(res['layout']['objects'])}  {row['seconds']}s")
    if phys is not None:
        phys.close()                                  # the Isaac work folders: their files were copied into the samples
    summary = {"accepted": n_ok, "attempts": len(rows), "wanted": n_want,
               "physics_replay": phys is not None, "reexecution": rz.reexec is not None, "methods": methods,
               "seconds": {"kinematic": round(t_kin, 1), "physics": round(t_phys, 1), "reexecute": round(t_rx, 1),
                           "physics_original": phys.baseline["seconds"] if phys is not None else 0.0},
               "sampling": {k: v for k, v in (cfg.get("sampling") or {}).items()},
               "relations": rel, "warnings": ep.warnings, "rows": rows}
    if phys is not None:
        summary["physics_original"] = physics_summary(phys.baseline)
        summary["mujoco_version"] = phys.mujoco_version
    (cfg.out / "randomize.json").write_text(json.dumps(summary, indent=1, default=float))
    log(f"[randomize] {n_ok}/{len(rows)} layouts accepted -> {out}")
    assert n_ok >= n_want, (
        f"only {n_ok} of {n_want} samples passed in {len(rows)} attempts; "
        f"narrow randomize ranges or raise max_attempts")
    return summary
