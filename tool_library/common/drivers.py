"""Force-driven drivers that put a mechanism on trial.

One driver per mechanism. They share one rule: NOTHING IS EVER POSED. A driver
only adds to `xfrc_applied`, a force or a torque on a body, and lets the solver
decide what happens together with gravity, contact, joint limits and the
mechanism's own constraints. A position servo on a cap's yaw joint would spin
the cap open just as prettily on an asset whose "thread" was a free slide; a
twisting torque on the cap body opens the bottle only if the coupling really
converts rotation into lift. Every driver measures what its forces achieved
and `verdict()` turns the measurement into a sentence.
"""
from __future__ import annotations

import numpy as np

from .mjcf import ForceGrip, apply_wrench


def _own_penetration_mm(model, data, body_ids) -> float:
    """Deepest interpenetration among contacts THIS driver's bodies are in.

    Scanning every contact in the scene instead is how a driver ends up
    reporting its neighbour's problem: the carton in the stage demo failed its
    own check on 9.5 mm of penetration that turned out to be the bottle's
    loose cap sinking into the table, two objects away.
    """
    worst = 0.0
    for c in range(data.ncon):
        b1 = model.geom_bodyid[data.contact.geom1[c]]
        b2 = model.geom_bodyid[data.contact.geom2[c]]
        if b1 in body_ids or b2 in body_ids:
            worst = max(worst, -1000.0 * float(data.contact.dist[c]))
    return worst


def _n(prefix: str, name: str) -> str:
    return f"{prefix}{name}"


def smoothstep(t, t0, t1):
    if t <= t0:
        return 0.0
    if t >= t1:
        return 1.0
    s = (t - t0) / (t1 - t0)
    return 0.5 - 0.5 * np.cos(np.pi * s)


# ==========================================================================
class BottleDriver:
    """Twist the cap with a TORQUE; tug it with a FORCE. Never touch a joint.

    Sequence: settle, push straight down (nothing), twist it closed, pull
    straight up (nothing), twist it open, thread runs out, lift it away, drop.
    """

    PULL_N = 20.0          # a hard straight tug on a 3.5 g cap
    TWIST_NM = 0.100       # must beat the thread's own friction (0.02 N*m);
                           # spin rate = (twist - friction) / yaw_damping,
                           # so 0.10 N*m turns it at ~1.6 turns/s

    def __init__(self, model, data, built: dict, prefix: str = "",
                 t0: float = 0.0):
        self.m, self.d, self.b, self.p = model, data, built, prefix
        self.t0 = float(t0)
        self.cap = model.body(_n(prefix, built["bodies"]["cap"])).id
        self.bottle = model.body(_n(prefix, built["bodies"]["bottle"])).id
        self.yaw_a = model.joint(_n(prefix, built["joints"]["yaw"])).qposadr[0]
        self.lift_a = model.joint(_n(prefix, built["joints"]["lift"])).qposadr[0]
        self.eq = model.equality(_n(prefix, built["equalities"]["engaged"])).id
        self.k = model.equality(_n(prefix, built["equalities"]["thread"])).data[1]
        self.range = float(built["total_lift_m"])
        self.grip = ForceGrip(model, self.cap, max_force=self.PULL_N * 1.2)
        # A SECOND, GENTLE grip for carrying the cap once it is free. The
        # push/pull grip is deliberately brutal (20 N against a constraint);
        # reusing it on a 3.5 g body with nothing holding it back is 6800
        # m/s^2 and the integrator dies on the spot.
        self.carry_grip = ForceGrip(model, self.cap, kp=1.5, kd=0.12,
                                    max_force=0.12)
        self.released = False
        # WHERE THE CAP STARTS decides which half of the sequence is
        # meaningful, and it differs by construction: a freshly compiled asset
        # starts SEATED (qpos0), while the standalone demo seats it at the top
        # of the thread first. Measuring "travel without turning" against the
        # wrong end of the thread reports the full 25 mm as cheating when the
        # cap has not moved at all — which is exactly what it did.
        self.lift0 = float(data.qpos[self.lift_a])
        self.open_only = self.lift0 < 0.5 * self.range
        self._phase_ref = self.lift0
        self._phase = "start"
        self.meas = {"push_N": self.PULL_N, "twist_Nm": self.TWIST_NM,
                     "travel_without_turning_mm": 0.0,
                     "thread_travel_mm": round(self.range * 1000, 2),
                     "turns_applied": 0.0, "coupling_error_mm": 0.0}
        self.T = (dict(settle=1.0, push=1.0, close=1.0, pull=2.8, open=8.0,
                       carry=8.8, drop=12.0) if self.open_only else
                  dict(settle=1.0, push=2.4, close=6.4, pull=7.8, open=12.0,
                       carry=12.8, drop=15.5))

    @property
    def axis(self):
        return self.d.xmat[self.bottle].reshape(3, 3)[:, 2]

    def lift(self):
        return float(self.d.qpos[self.lift_a])

    def turns(self):
        return float(self.d.qpos[self.yaw_a]) / (2 * np.pi)

    def step(self, t) -> str:
        t = t - self.t0
        if t < 0:
            return ""
        if not self.released:
            self.meas["coupling_error_mm"] = max(
                self.meas["coupling_error_mm"],
                1000 * abs(self.lift() - self.k * self.d.qpos[self.yaw_a]))
        self.meas["turns_applied"] = max(self.meas["turns_applied"],
                                         abs(self.turns()))
        T, ax = self.T, self.axis
        if t < T["settle"]:
            return "cap screwed on"
        if t < T["push"]:
            # THE BASELINE IS WHERE THIS PHASE STARTED, not where the thread
            # ends. Measuring against the end of the travel reported a
            # half-closed cap as 12.8 mm of cheating when it had not moved at
            # all during the push.
            if self._phase != "push":
                self._phase, self._phase_ref = "push", self.lift()
            self.grip.pull(self.d, -ax, self.PULL_N)
            self.meas["travel_without_turning_mm"] = max(
                self.meas["travel_without_turning_mm"],
                1000 * abs(self.lift() - self._phase_ref))
            return (f"PUSH {self.PULL_N:.0f} N straight down, no twist -> "
                    f"lift {self.lift()*1000:5.1f} mm")
        if t < T["close"]:
            self.grip.twist(self.d, ax, -self.TWIST_NM)
            return (f"TWIST {self.TWIST_NM*1000:.1f} mN*m closing -> "
                    f"{self.turns():4.2f} turns, lift {self.lift()*1000:5.1f} mm")
        if t < T["pull"]:
            if self._phase != "pull":
                self._phase, self._phase_ref = "pull", self.lift()
            self.grip.pull(self.d, ax, self.PULL_N)
            self.meas["travel_without_turning_mm"] = max(
                self.meas["travel_without_turning_mm"],
                1000 * abs(self.lift() - self._phase_ref))
            return (f"PULL {self.PULL_N:.0f} N straight up, no twist -> "
                    f"lift {self.lift()*1000:5.1f} mm")
        if t < T["open"] and self.lift() < 0.96 * self.range:
            self.grip.twist(self.d, ax, +self.TWIST_NM)
            return (f"TWIST {self.TWIST_NM*1000:.1f} mN*m opening -> "
                    f"{self.turns():4.2f} turns, lift {self.lift()*1000:5.1f} mm")
        if not self.released:
            self.d.eq_active[self.eq] = 0
            self.released = True
            self.t_release = t
            # what the twist ACTUALLY achieved by the moment of release —
            # reaching here by the phase TIMEOUT with the cap still far down
            # its thread is a stall, and the verdict must say so instead of
            # crediting the full travel
            self.meas["lift_at_release_mm"] = round(self.lift() * 1000, 2)
            self.meas["turns_at_release"] = round(abs(self.turns()), 2)
            self._carry_to = (self.d.xpos[self.cap] + ax * 0.055
                              + np.array([0.06, 0.0, 0.0]))
            return "THREAD RAN OUT — the cap is a free rigid body now"
        if t < getattr(self, "t_release", T["open"]) + 1.2:
            # a hand lifting it clear: still just a force on a free body
            self.carry_grip.hold(self.d, self.lift_target(ax))
            return "lifted away by hand (force only)"
        return "let go — gravity, contact, roll"

    def lift_target(self, ax):
        return getattr(self, "_carry_to", self.d.xpos[self.cap])

    def verdict(self) -> str:
        m = self.meas
        rel = m.get("lift_at_release_mm")
        stalled = rel is not None and rel < 0.9 * m["thread_travel_mm"]
        ok = (m["travel_without_turning_mm"] < 2.0
              and m["coupling_error_mm"] < 2.0 and not stalled)
        if stalled:
            return (f"TWIST STALLED — {m['twist_Nm']*1000:.1f} mN*m only "
                    f"unscrewed {rel:.1f} of {m['thread_travel_mm']:.0f} mm "
                    f"({m.get('turns_at_release', 0):.2f} turns) before the "
                    f"phase timed out and force-released the thread. The cap "
                    f"is jammed (contact interference or friction), not "
                    f"unscrewed — the drop that follows is NOT evidence.")
        return (f"THREAD OK — {m['push_N']:.0f} N of straight push/pull moved "
                f"the cap {m['travel_without_turning_mm']:.2f} mm of its "
                f"{m['thread_travel_mm']:.0f} mm travel; "
                f"{m['twist_Nm']*1000:.1f} mN*m of twist moved it all of it "
                f"in {m['turns_applied']:.1f} turns"
                if ok else
                f"NOT A THREAD — straight force alone moved the cap "
                f"{m['travel_without_turning_mm']:.1f} mm (coupling error "
                f"{m['coupling_error_mm']:.1f} mm)")


# ==========================================================================
class FoldBoxDriver:
    """Fold the carton by PUSHING each panel's far edge, then let go.

    Unfolding is not a reverse trajectory — the forces simply stop and
    gravity opens the box, which is the honest test of whether the creases
    were holding anything up.
    """

    PUSH_N = 1.2

    def __init__(self, model, data, built: dict, prefix: str = "",
                 t0: float = 0.0):
        self.m, self.d, self.b, self.p = model, data, built, prefix
        self.t0 = float(t0)
        self.walls = [_n(prefix, j) for j in built["wall_joints"]]
        self.flaps = [_n(prefix, j) for j in built["flap_joints"]]
        self.y_flaps = [j for j in self.flaps if j.endswith(("yp", "yn"))]
        self.x_flaps = [j for j in self.flaps if j.endswith(("xp", "xn"))]
        # style="lid" has ONE flap joint named <name>_lid, which matches
        # neither pair — split by suffix alone and it would be pushed by
        # nothing at all and the lid would simply never open.
        if not self.y_flaps and not self.x_flaps:
            self.y_flaps = list(self.flaps)
        self.qad = {j: model.joint(j).qposadr[0]
                    for j in self.walls + self.flaps}
        self.bid, self.local = {}, {}
        for raw, body in built["push_bodies"].items():
            j = _n(prefix, raw)
            self.bid[j] = model.body(_n(prefix, body)).id
            self.local[j] = np.asarray(built["push_local"][raw], float)
        self._own = {self.bid[j] for j in self.bid}
        self._own |= {model.body(_n(prefix, built["base_body"])).id}
        self.meas = {"push_N": self.PUSH_N, "max_wall_deg": 0.0,
                     "max_flap_deg": 0.0, "max_penetration_mm": 0.0,
                     "opened_back_deg": 0.0}
        self.T = dict(rest=1.2, walls=5.0, inner=7.4, outer=9.8, hold=12.0,
                      open=17.0)

    def _push(self, joint, gain=1.0, stop_deg=86.0):
        """Force along the panel's own normal, applied at its far edge.

        The push EASES OFF over the last 25 degrees. Pushing at full force up
        to the stop and then letting the joint limit catch the panel drives it
        10 degrees past vertical and slams the flaps together — 6.4 mm of
        panel interpenetration, measured. A hand folding card slows down too.
        """
        b = self.bid[joint]
        deg = np.degrees(self.d.qpos[self.qad[joint]])
        stop_deg = min(stop_deg, np.degrees(self.m.jnt_range[
            self.m.joint(joint).id][1]) - 2.0)
        if deg >= stop_deg:
            return
        gain = gain * min(1.0, max(0.03, (stop_deg - deg) / 35.0))
        R = self.d.xmat[b].reshape(3, 3)
        pt = self.d.xpos[b] + R @ self.local[joint]
        apply_wrench(self.d, b, force=(R @ np.array([0.0, 0.0, 1.0]))
                     * self.PUSH_N * gain, point=pt, com=self.d.xipos[b])

    def step(self, t) -> str:
        t = t - self.t0
        if t < 0:
            return ""
        wall_deg = (max(np.degrees(self.d.qpos[self.qad[j]])
                        for j in self.walls) if self.walls else 90.0)
        self.meas["max_wall_deg"] = max(self.meas["max_wall_deg"], wall_deg)
        if self.flaps:
            self.meas["max_flap_deg"] = max(
                self.meas["max_flap_deg"],
                max(np.degrees(self.d.qpos[self.qad[j]]) for j in self.flaps))
        self.meas["max_penetration_mm"] = max(
            self.meas["max_penetration_mm"],
            _own_penetration_mm(self.m, self.d, self._own))
        T = self.T
        if t < T["rest"]:
            return "flat blank resting on the table"
        if t < T["hold"]:
            g = smoothstep(t, T["rest"], T["rest"] + 0.6)
            for j in self.walls:
                self._push(j, g)
            if t > T["walls"]:
                for j in self.y_flaps:
                    self._push(j, smoothstep(t, T["walls"], T["walls"] + 0.5))
            if t > T["inner"]:
                for j in self.x_flaps:
                    self._push(j, smoothstep(t, T["inner"], T["inner"] + 0.5))
            if not self.walls:                       # style="lid"
                what = ("opening the lid" if t < T["outer"] else
                        "lid held open")
                return (f"{what}   {self.PUSH_N:.1f} N at the lid's free edge"
                        f"   lid {self.meas['max_flap_deg']:5.1f} deg")
            what = ("pushing the walls up" if t < T["walls"] else
                    "pushing the inner flaps over" if t < T["inner"] else
                    "pushing the outer flaps over" if t < T["outer"] else
                    "closed, still held")
            return (f"{what}   {self.PUSH_N:.1f} N at each panel edge   "
                    f"walls {wall_deg:5.1f} deg")
        # forces OFF: whatever happens now is gravity and the creases
        if t > T["hold"] + 1.0:
            self.meas["opened_back_deg"] = max(
                self.meas["opened_back_deg"], 90.0 - wall_deg)
        if not self.walls:
            return (f"HANDS OFF — the lid falls shut under gravity   lid "
                    f"{np.degrees(self.d.qpos[self.qad[self.flaps[0]]]):5.1f} deg")
        return f"HANDS OFF — gravity unfolds it   walls {wall_deg:5.1f} deg"

    def verdict(self) -> str:
        m = self.meas
        rigid = not self.walls                    # style="lid": nothing folds
        ok = ((m["max_flap_deg"] if rigid else m["max_wall_deg"]) > 85.0
              and m["max_penetration_mm"] < 1.5)
        if rigid:
            return (f"LID OPENS UNDER FORCE — {m['push_N']:.1f} N at the lid's "
                    f"free edge swung it to {m['max_flap_deg']:.1f} deg; worst "
                    f"interpenetration {m['max_penetration_mm']:.2f} mm"
                    if ok else
                    f"LID FAILED — {m['push_N']:.1f} N only reached "
                    f"{m['max_flap_deg']:.1f} deg")
        return (f"FOLDS UNDER FORCE — {m['push_N']:.1f} N per panel edge "
                f"raised the walls to {m['max_wall_deg']:.1f} deg and closed "
                f"the flaps to {m['max_flap_deg']:.1f} deg; worst panel "
                f"interpenetration {m['max_penetration_mm']:.2f} mm; with the "
                f"hands off it fell back open by "
                f"{m['opened_back_deg']:.1f} deg"
                if ok else
                f"FOLD FAILED — {m['push_N']:.1f} N only reached "
                f"{m['max_wall_deg']:.1f} deg (penetration "
                f"{m['max_penetration_mm']:.2f} mm)")


# ==========================================================================
class ThreadDriver:
    """Drag the thread by a capped spring-damper wrench on one end segment.

    A 15 mm segment weighs 0.6 g, so the cap on that wrench is what keeps this
    from being a teleport: the end lags its target, the rest of the cable
    follows only through its own bending and tension, and letting go really
    is letting go.
    """

    MAX_N = 0.9

    def __init__(self, model, data, built: dict, prefix: str = "",
                 t0: float = 0.0, height: float = 0.0, span: float = 0.5):
        self.m, self.d, self.b, self.p = model, data, built, prefix
        self.t0, self.z0, self.span = float(t0), float(height), float(span)
        self.first = model.body(_n(prefix, built["bodies"]["first"])).id
        self.last = model.body(_n(prefix, built["bodies"]["last"])).id
        self.ids = ([self.first]
                    + [model.body(_n(prefix, b)).id
                       for b in built["bodies"]["segments"]] + [self.last])
        self.ga = ForceGrip(model, self.first, kp=3.0, kd=0.25,
                            max_force=self.MAX_N)
        self.gb = ForceGrip(model, self.last, kp=3.0, kd=0.25,
                            max_force=self.MAX_N)
        self.L = None
        self.meas = {"max_force_N": self.MAX_N, "max_stretch_pct": 0.0,
                     "max_span_m": 0.0, "self_contacts": 0,
                     "nominal_length_m": 0.0}
        self.T = dict(settle=1.0, lift=3.0, swing=7.0, lower=9.5, coil=11.0,
                      taut=14.5, loop=18.0)

    def centreline(self):
        return self.d.xpos[self.ids].copy()

    def step(self, t) -> str:
        t = t - self.t0
        if t < 0:
            return ""
        cl = self.centreline()
        seg = float(np.linalg.norm(np.diff(cl, axis=0), axis=1).sum())
        if self.L is None:
            self.L = seg
            self.meas["nominal_length_m"] = round(seg, 4)
        self.meas["max_stretch_pct"] = max(self.meas["max_stretch_pct"],
                                           100 * (seg - self.L) / self.L)
        self.meas["max_span_m"] = max(self.meas["max_span_m"],
                                      float(np.linalg.norm(cl[-1] - cl[0])))
        sc = 0
        for c in range(self.d.ncon):
            b1 = self.m.geom_bodyid[self.d.contact.geom1[c]]
            b2 = self.m.geom_bodyid[self.d.contact.geom2[c]]
            if b1 in self.ids and b2 in self.ids and abs(b1 - b2) > 1:
                sc += 1
        self.meas["self_contacts"] = max(self.meas["self_contacts"], sc)

        T = self.T
        mid = 0.5 * (cl[0] + cl[-1])
        if t < T["settle"]:
            return "thread lying on the table"
        if t < T["lift"]:
            a = smoothstep(t, T["settle"], T["lift"])
            self.ga.hold(self.d, [cl[0][0], cl[0][1], self.z0 + 0.22 * a])
            return f"one end lifted by <= {self.MAX_N:.1f} N of pull"
        if t < T["swing"]:
            a = (t - T["lift"]) / (T["swing"] - T["lift"])
            th = 2 * np.pi * 1.5 * a
            # swing radius scales with the thread — a fixed 13 cm circle is
            # a gentle arc for a 60 cm cord and a whip that lands on whatever
            # is 13 cm away for a 26 cm one (measured: it flogged the carton
            # next to it and drove 9.5 mm into a panel).
            r = 0.35 * self.span
            self.ga.hold(self.d, [mid[0] + r * np.cos(th),
                                  mid[1] + r * np.sin(th),
                                  self.z0 + 0.22])
            return "swung in a circle — the free end trails behind"
        if t < T["lower"]:
            a = smoothstep(t, T["swing"], T["lower"])
            self.ga.hold(self.d, [mid[0], mid[1],
                                  self.z0 + 0.22 * (1 - a) + 0.01])
            return "lowered back onto the table"
        if t < T["coil"]:
            return f"let go — {self.meas['self_contacts']} self-contacts"
        if t < T["taut"]:
            a = smoothstep(t, T["coil"], T["taut"])
            d0 = cl[0] - mid
            d0 = d0 / (np.linalg.norm(d0[:2]) + 1e-9)
            for grip, sgn in ((self.ga, +1.0), (self.gb, -1.0)):
                grip.hold(self.d, [mid[0] + sgn * d0[0] * 0.46 * self.L * a,
                                   mid[1] + sgn * d0[1] * 0.46 * self.L * a,
                                   self.z0 + 0.01])
            return (f"both ends pulled apart at <= {self.MAX_N:.1f} N -> span "
                    f"{np.linalg.norm(cl[-1]-cl[0])*100:5.1f} cm of "
                    f"{self.L*100:.0f} cm")
        a = smoothstep(t, T["taut"], T["loop"])
        self.ga.hold(self.d, [mid[0] + 0.10 * a, mid[1] - 0.16 * a,
                              self.z0 + 0.01 + 0.05 * a])
        self.gb.hold(self.d, [mid[0] - 0.10 * a, mid[1] + 0.16 * a,
                              self.z0 + 0.01])
        return f"ends crossed into a loop — {self.meas['self_contacts']} self-contacts"

    def verdict(self) -> str:
        m = self.meas
        L = m["nominal_length_m"] or 1.0
        ok = (m["max_stretch_pct"] < 5.0 and m["max_span_m"] <= L * 1.02
              and m["self_contacts"] > 0)
        return (f"CABLE OK — dragged by <= {m['max_force_N']:.1f} N it "
                f"stretched {m['max_stretch_pct']:.2f}%, never spanned more "
                f"than {m['max_span_m']*100:.1f} cm of its {L*100:.0f} cm, "
                f"and reached {m['self_contacts']} self-contacts"
                if ok else
                f"CABLE SUSPECT — stretch {m['max_stretch_pct']:.2f}%, span "
                f"{m['max_span_m']*100:.1f}/{L*100:.0f} cm, self-contacts "
                f"{m['self_contacts']}")


# ==========================================================================
