"""Task and rollout gates: goal predicates measured on an executed episode.

Each gate answers one physical question about the delivered trajectory (are two bodies touching at the end of a
window, does A rest on B, is A inside B, did a joint reach its target and stay there, was the object held through the
window, did it stay where it was put, ...) by forward kinematics and collision detection on the recorded qpos, with
the same execution model the Stage 3 executor used (`episode.build_execution_model`: hand hull pairs, so hand-object
contacts register). The verifiers in tool_library/verifiers/*_gate.py are thin wrappers around the functions here;
`stage1_goal_gate` maps each Stage 1 goal sentence to predicates and evaluates them with the same functions.

Every gate returns {ok, issues, measured...}; issues name the frame, the body and what to change.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from tool_library.common import episode as E

TABLE_NAMES = ("table", "floor", "ground")
BOX, SPHERE, CAPSULE, CYLINDER, ELLIPSOID, MESH, PLANE = (int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_SPHERE),
                                                         int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                                                         int(mujoco.mjtGeom.mjGEOM_ELLIPSOID), int(mujoco.mjtGeom.mjGEOM_MESH),
                                                         int(mujoco.mjtGeom.mjGEOM_PLANE))
HINGE, SLIDE, FREE, BALL = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE), int(mujoco.mjtJoint.mjJNT_FREE),
                            int(mujoco.mjtJoint.mjJNT_BALL))
SIDES = ("left", "right")

# Contact distances: soft contacts rest a fraction of a millimetre inside each other and flicker by about as much
# between frames, so "touching" means a contact closer than 2 mm. Holding contacts are judged at 1 mm.
TOUCH_TOL_M = 0.002
HOLD_TOL_M = 0.001
# A goal that says two objects touch is judged at 4 mm: reconstructed parts carry millimetre errors and a latch that is
# closed on its joint can rest 3 mm off the wall it hooks under; the misses this gate exists for are centimetres short.
GOAL_TOUCH_TOL_M = 0.004


def _r(x, n=4):
    return None if x is None else round(float(x), n)


# ============================================================================================ episode
class Episode:
    """A scene (the execution model of the MJCF the trajectory was executed in) and its trajectory."""

    def __init__(self, scene, trajectory=None, qpos=None, fps=None, labels=None):
        if isinstance(scene, mujoco.MjModel):
            self.model = scene
            self.scene_path = None
        else:
            p = Path(str(scene))
            if not p.exists():
                raise ValueError(f"scene: {scene} does not exist (pass the MJCF the trajectory was executed in)")
            self.scene_path = str(p)
            self.model, _ = E.build_execution_model(str(p))
        m = self.model
        self.data = mujoco.MjData(m)
        self.extras = {}
        if trajectory is not None:
            tp = Path(str(trajectory))
            if not tp.exists():
                raise ValueError(f"trajectory: {trajectory} does not exist")
            z = np.load(str(tp), allow_pickle=True)
            key = "qpos" if "qpos" in z.files else ("Q" if "Q" in z.files else None)
            if key is None:
                raise ValueError("trajectory: the npz has no 'qpos' array")
            self.Q = np.asarray(z[key], float)
            self.fps = float(z["fps"]) if "fps" in z.files else 30.0
            self.labels = [str(x) for x in z["labels"]] if "labels" in z.files else [""] * len(self.Q)
            for k in ("subgoal", "eq_active", "eq_names", "eq_active_start"):
                if k in z.files:
                    self.extras[k] = np.asarray(z[k])
        elif qpos is not None:
            self.Q = np.atleast_2d(np.asarray(qpos, float))
            self.fps = float(fps or 30.0)
            self.labels = list(labels) if labels is not None else [""] * len(self.Q)
        else:
            raise ValueError("trajectory: give the episode npz (or qpos)")
        if self.Q.ndim != 2 or self.Q.shape[1] != m.nq:
            raise ValueError(f"scene: the trajectory has {self.Q.shape[-1]} qpos columns but the scene has nq={m.nq}; "
                             "pass the scene the trajectory was executed in")
        if len(self.Q) < 1 or (trajectory is not None and len(self.Q) < 2) or not np.isfinite(self.Q).all():
            raise ValueError("trajectory: nonempty finite states are required (at least two for a recorded episode)")
        if fps is not None:
            self.fps = float(fps)
        self.T = len(self.Q)
        if not np.isfinite(self.fps) or self.fps <= 0 or len(self.labels) != self.T:
            raise ValueError("trajectory: fps must be finite and positive and every frame must have a label")
        self._children = {}
        for b in range(1, m.nbody):
            self._children.setdefault(int(m.body_parentid[b]), []).append(b)
        self._sub = {}
        self._frame = None
        tb = set()
        for b in range(1, m.nbody):
            if m.body(b).name in TABLE_NAMES:
                tb |= self.subtree(b)
        if any(int(m.geom_bodyid[g]) == 0 and (m.geom_contype[g] or m.geom_conaffinity[g]) for g in range(m.ngeom)):
            tb.add(0)
        self.table = tb
        self.robot = {b for b in range(m.nbody) if m.body(b).name.startswith("robot_")}
        self._hands = None
        self._mesh = {}
        self._pair_geoms = None

    # ------------------------------------------------------------------ structure
    def subtree(self, bid):
        bid = int(bid)
        if bid not in self._sub:
            out, stack = set(), [bid]
            while stack:
                b = stack.pop()
                out.add(b)
                stack += self._children.get(b, [])
            self._sub[bid] = out
        return self._sub[bid]

    def body(self, name, param="body"):
        """Body id for a name: exact, its slug, or the one body ending in '_<slug>' (a scene prefix)."""
        m = self.model
        if isinstance(name, (int, np.integer)):
            return int(name)
        s = str(name)
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, s)
        if bid >= 0:
            return int(bid)
        slug = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, slug)
        if bid >= 0:
            return int(bid)
        cands = [b for b in range(m.nbody) if m.body(b).name.endswith(f"_{slug}")]
        if len(cands) == 1:
            return cands[0]
        raise ValueError(f"{param}: no body {name!r} in the scene (bodies: "
                         f"{[m.body(b).name for b in range(1, m.nbody) if not m.body(b).name.startswith('robot_')][:30]})")

    def bodies(self, name, param="body"):
        """The body set a name stands for: 'table' -> the support surface, a body -> its subtree, a list -> the union."""
        if isinstance(name, (list, tuple, set)):
            out = set()
            for n in name:
                out |= self.bodies(n, param)
            return out
        if str(name) in TABLE_NAMES:
            if not self.table:
                raise ValueError(f"{param}: the scene has no table body (a body named 'table' or a world plane)")
            return set(self.table)
        return set(self.subtree(self.body(name, param)))

    def name(self, bid):
        return self.model.body(int(bid)).name

    def hands(self):
        """{side: set of hand body ids}."""
        if self._hands is None:
            self._hands = {}
            try:
                for side, pt in E.robot_parts(self.model).items():
                    self._hands[side] = set(int(b) for b in pt["rig"].body_ids)
            except Exception:
                pass
            for side in SIDES:
                if side not in self._hands:
                    hb = {b for b in range(self.model.nbody) if self.name(b).startswith(f"robot_{side}_")
                          and "_arm_" not in self.name(b) and "_base" not in self.name(b)}
                    if hb:
                        self._hands[side] = hb
        return self._hands

    def hand_set(self, side=None, hand_bodies=None, param="side"):
        if hand_bodies:
            out = set()
            for n in hand_bodies:
                nm = str(n)
                hits = {b for b in range(self.model.nbody) if self.name(b) == nm or self.name(b).startswith(nm)}
                if not hits:
                    raise ValueError(f"hand_bodies: no body named or prefixed {nm!r}")
                out |= hits
            return out
        hs = self.hands()
        if side in (None, "any", "both"):
            out = set()
            for v in hs.values():
                out |= v
            if not out:
                raise ValueError(f"{param}: the scene has no robot hand; pass hand_bodies")
            return out
        if side not in hs:
            raise ValueError(f"{param}: no {side!r} hand in the scene (hands: {sorted(hs)})")
        return set(hs[side])

    def window(self, frames=None, subgoal=None):
        if frames is not None:
            if len(frames) != 2:
                raise ValueError("frames: give [first, last]")
            t0, t1 = int(frames[0]), int(frames[1])
            if t0 < 0:
                t0 += self.T
            if t1 < 0:
                t1 += self.T
            if t0 < 0 or t1 >= self.T:
                raise ValueError(f"frames: {list(frames)} is outside the trajectory of {self.T} frames")
            if t1 < t0:
                raise ValueError(f"frames: {list(frames)} is empty for a trajectory of {self.T} frames")
            return t0, t1
        if subgoal is not None:
            sg = self.extras.get("subgoal")
            if sg is None:
                raise ValueError("subgoal: the trajectory has no per-frame 'subgoal' array; pass frames")
            idx = np.nonzero(np.asarray(sg).astype(int) == int(subgoal))[0]
            if not len(idx):
                raise ValueError(f"subgoal: no frames with subgoal index {subgoal}")
            return int(idx[0]), int(idx[-1])
        return 0, self.T - 1

    # ------------------------------------------------------------------ state
    def at(self, t, collide=False):
        """Set the state to frame t (kinematics; collision detection when collide)."""
        t = int(t)
        key = (t, bool(collide))
        if self._frame != key:
            d = self.data
            d.qpos[:] = self.Q[t]
            mujoco.mj_kinematics(self.model, d)
            if collide:
                mujoco.mj_collision(self.model, d)
            self._frame = key
        return self.data

    def invalidate(self):
        self._frame = None

    def pose(self, t, bid):
        d = self.at(t)
        return d.xpos[bid].copy(), d.xmat[bid].reshape(3, 3).copy()

    def com(self, t, bodies):
        """Mass-weighted centre of a body set at frame t."""
        d = self.at(t)
        m = self.model
        ms = np.array([float(m.body_mass[b]) for b in bodies])
        P = np.array([d.xipos[b] for b in bodies])
        if ms.sum() <= 0:
            return P.mean(0)
        return (ms[:, None] * P).sum(0) / ms.sum()

    # ------------------------------------------------------------------ geometry
    def pair_geoms(self):
        if self._pair_geoms is None:
            m = self.model
            self._pair_geoms = set(int(g) for g in m.pair_geom1) | set(int(g) for g in m.pair_geom2)
        return self._pair_geoms

    def geoms(self, bodies, collision=True):
        """Geoms of a body set: collision geoms (contype/conaffinity or in an explicit pair), else every geom."""
        m = self.model
        bodies = set(bodies)
        allg = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in bodies and int(m.geom_type[g]) != PLANE]
        if not collision:
            return allg
        pg = self.pair_geoms()
        col = [g for g in allg if m.geom_contype[g] or m.geom_conaffinity[g] or g in pg]
        return col or allg

    def _verts(self, g):
        if g not in self._mesh:
            m = self.model
            mid = int(m.geom_dataid[g])
            a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
            self._mesh[g] = np.asarray(m.mesh_vert[a:a + n], float)
        return self._mesh[g]

    def support(self, t, gids, d):
        """max over the geoms of max_{x in geom} x . d (d a world direction): the extent of a body set along d."""
        dat = self.at(t)
        m = self.model
        d = np.asarray(d, float)
        d = d / max(np.linalg.norm(d), 1e-12)
        best = -np.inf
        for g in gids:
            ty = int(m.geom_type[g])
            p = dat.geom_xpos[g]
            R = dat.geom_xmat[g].reshape(3, 3)
            s = m.geom_size[g]
            dl = R.T @ d
            c = float(p @ d)
            if ty == BOX:
                v = c + float(np.abs(dl) @ s[:3])
            elif ty == SPHERE:
                v = c + float(s[0])
            elif ty == CAPSULE:
                v = c + float(s[0]) + float(s[1]) * abs(float(dl[2]))
            elif ty == CYLINDER:
                v = c + float(s[1]) * abs(float(dl[2])) + float(s[0]) * float(np.sqrt(max(0.0, 1.0 - dl[2] ** 2)))
            elif ty == ELLIPSOID:
                v = c + float(np.linalg.norm(s[:3] * dl))
            elif ty == MESH:
                v = c + float(np.max(self._verts(g) @ dl))
            elif ty == PLANE:
                continue
            else:
                v = c + float(m.geom_rbound[g])
            best = max(best, v)
        return best

    def extent(self, t, gids, R=None):
        """(lo, hi) of a body set along the columns of R (world axes by default)."""
        R = np.eye(3) if R is None else np.asarray(R, float)
        lo = np.array([-self.support(t, gids, -R[:, k]) for k in range(3)])
        hi = np.array([self.support(t, gids, R[:, k]) for k in range(3)])
        return lo, hi

    def lowest(self, t, bodies):
        return -self.support(t, self.geoms(bodies), [0, 0, -1.0])

    # ------------------------------------------------------------------ contacts
    def contacts(self, t, A, B, tol=TOUCH_TOL_M):
        """[(dist, pos, normal from A into B)] between body sets A and B at frame t; geoms of A and B are given a
        detection margin of tol (so contacts up to tol apart register and report their true gap)."""
        A, B = set(A), set(B)
        with Inflate(self, A | B, tol):
            d = self.at(t, collide=True)
            m = self.model
            out = []
            for i in range(d.ncon):
                c = d.contact[i]
                b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
                if b1 in A and b2 in B:
                    n = np.asarray(c.frame[:3], float).copy()
                elif b2 in A and b1 in B:
                    n = -np.asarray(c.frame[:3], float)
                else:
                    continue
                if float(c.dist) <= tol:
                    out.append((float(c.dist), np.asarray(c.pos, float).copy(), n))
        return out

    def gap(self, t, A, B, probe=0.05):
        """Smallest A-B distance at frame t up to `probe` (None when farther)."""
        cs = self.contacts(t, A, B, tol=probe)
        return min(c[0] for c in cs) if cs else None

    def excluded(self, A, B):
        """True when the scene filters every A-B body pair out of collision (an explicit <exclude> or a parent-child pair):
        two parts of one articulated object (a lid hinged on its box) never produce contacts."""
        m = self.model
        sig = {int(x) for x in m.exclude_signature} if m.nexclude else set()
        filt_parent = not (int(m.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT))
        ba = {int(m.geom_bodyid[g]) for g in self.geoms(A)}
        bb = {int(m.geom_bodyid[g]) for g in self.geoms(B)}
        if not ba or not bb:
            return False
        for a in ba:
            for b in bb:
                if ((a << 16) + b) in sig or ((b << 16) + a) in sig:
                    continue
                wa, wb = int(m.body_weldid[a]), int(m.body_weldid[b])
                if wa == wb:
                    continue
                if filt_parent and (int(m.body_weldid[int(m.body_parentid[wa])]) == wb and wb != 0 or
                                    int(m.body_weldid[int(m.body_parentid[wb])]) == wa and wa != 0):
                    continue
                return False
        return True

    def geom_gap(self, t, A, B, probe=0.05):
        """Smallest distance between the primitive collision geoms of A and B at frame t, ignoring collision filters
        (None when farther than probe; 'mesh' when either side has only meshes, whose convex-hull distance is unreliable)."""
        m = self.model
        prim = lambda gs: [g for g in gs if int(m.geom_type[g]) in (BOX, SPHERE, CAPSULE, CYLINDER, ELLIPSOID)]
        ga, gb = prim(self.geoms(A)), prim(self.geoms(B))
        if not ga or not gb:
            return "mesh"
        d = self.at(t)
        fr = np.zeros(6)
        best = None
        for a in ga:
            for b in gb:
                v = float(mujoco.mj_geomDistance(m, d, a, b, float(probe), fr))
                if v < probe and (best is None or v < best):
                    best = v
        return best

    def can_collide(self, A, B):
        """Can any geom of A ever collide with any geom of B (contype/conaffinity or an explicit pair)?"""
        m = self.model
        ga, gb = self.geoms(A), self.geoms(B)
        sa, sb = set(ga), set(gb)
        for i in range(m.npair):
            g1, g2 = int(m.pair_geom1[i]), int(m.pair_geom2[i])
            if (g1 in sa and g2 in sb) or (g1 in sb and g2 in sa):
                return True
        for g1 in ga:
            for g2 in gb:
                if (m.geom_contype[g1] & m.geom_conaffinity[g2]) or (m.geom_contype[g2] & m.geom_conaffinity[g1]):
                    return True
        return False


class Inflate:
    """Temporarily give the geoms of a body set (and their bodies and explicit pairs) a collision margin."""

    def __init__(self, ep, bodies, tol):
        self.ep, self.bodies, self.tol = ep, set(bodies), float(tol)

    def __enter__(self):
        m = self.ep.model
        self.g = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in self.bodies]
        gs = set(self.g)
        self.pairs = [i for i in range(m.npair) if int(m.pair_geom1[i]) in gs or int(m.pair_geom2[i]) in gs]
        self.b = sorted(self.bodies)
        self.old = (m.geom_margin[self.g].copy(), m.body_margin[self.b].copy() if hasattr(m, "body_margin") else None,
                    m.pair_margin[self.pairs].copy(), m.geom_gap[self.g].copy(), m.pair_gap[self.pairs].copy())
        if self.g:
            m.geom_margin[self.g] = np.maximum(m.geom_margin[self.g], self.tol)
            m.geom_gap[self.g] = 0.0
        if hasattr(m, "body_margin") and self.b:
            m.body_margin[self.b] = np.maximum(m.body_margin[self.b], self.tol)
        if self.pairs:
            m.pair_margin[self.pairs] = np.maximum(m.pair_margin[self.pairs], self.tol)
            m.pair_gap[self.pairs] = 0.0
        self.ep.invalidate()
        return self

    def __exit__(self, *exc):
        m = self.ep.model
        gm, bm, pm, gg, pg = self.old
        if self.g:
            m.geom_margin[self.g] = gm
            m.geom_gap[self.g] = gg
        if bm is not None and self.b:
            m.body_margin[self.b] = bm
        if self.pairs:
            m.pair_margin[self.pairs] = pm
            m.pair_gap[self.pairs] = pg
        self.ep.invalidate()
        return False


def frames_between(t0, t1, stride=1):
    st = max(1, int(stride))
    out = list(range(int(t0), int(t1) + 1, st))
    if out[-1] != int(t1):
        out.append(int(t1))
    return out


# ============================================================================================ orientation
def up_axis(R):
    """The body-frame unit axis (+-x, +-y, +-z) that points most nearly up in world orientation R."""
    k = int(np.argmax(np.abs(R[2, :])))
    u = np.zeros(3)
    u[k] = np.sign(R[2, k]) or 1.0
    return u


def tilt_deg(R, u_local):
    return float(np.degrees(np.arccos(np.clip((R @ u_local)[2], -1.0, 1.0))))


def level_deg(R):
    """Angle between the vertical and the body axis nearest to it (0 = some face is level)."""
    return float(np.degrees(np.arccos(np.clip(np.max(np.abs(R[2, :])), -1.0, 1.0))))


def yaw_deg(R):
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


def wrap180(a):
    return (float(a) + 180.0) % 360.0 - 180.0


def sym_err(a, sym_deg):
    """|a| folded by a rotational symmetry of sym_deg (90 for a square brick)."""
    if not sym_deg:
        return abs(wrap180(a))
    s = float(sym_deg)
    r = float(a) % s
    return min(r, s - r)


def _report(ok, issues, **meas):
    out = {"ok": bool(ok), "issues": list(issues)}
    out.update(meas)
    return out


# ============================================================================================ gates
def gate_contact(ep, A, B, t0, t1, tol=GOAL_TOUCH_TOL_M, throughout=False, min_frac=1.0, apart=False, apart_m=None,
                 stride=1, names=None):
    """A and B in contact at t1 (and on min_frac of [t0, t1] when throughout); apart: no contact closer than apart_m at t1."""
    na, nb = names or ("A", "B")
    issues = []
    if ep.excluded(A, B):
        # parts of one articulated object (a lid and its box) are filtered out of collision: measure the geometric gap
        g = ep.geom_gap(t1, A, B, probe=max(0.05, float(apart_m or 0) + 0.01))
        if g == "mesh":
            return _report(False, [f"{na} and {nb} are excluded from collision in this scene and are meshes, so their contact "
                                   "cannot be measured: state the relation as a joint target in goal_predicates"], measured_by="none")
        if apart:
            lim = float(apart_m if apart_m is not None else tol)
            ok = g is None or g > lim
            if not ok:
                issues.append(f"{na} and {nb} are {1000 * g:.1f} mm apart at frame {t1} (must be more than {1000 * lim:.1f} mm)")
            return _report(ok, issues, end_frame=int(t1), gap_m=_r(g), mode="apart", measured_by="geom distance (excluded pair)")
        ok = g is not None and g <= tol
        if not ok:
            issues.append(f"{na} and {nb} do not touch at frame {t1} (gap {'> 50' if g is None else f'{1000 * g:.1f}'} mm, "
                          f"tolerance {1000 * tol:.1f} mm): move {na} until it touches {nb}")
        return _report(ok, issues, end_frame=int(t1), touching_at_end=bool(ok), end_gap_m=_r(g), measured_by="geom distance (excluded pair)")
    if not ep.can_collide(A, B):
        return _report(False, [f"{na} and {nb} can never collide in this scene (contype/conaffinity, no explicit pair): "
                               "contact between them cannot be measured; check the body names"], can_collide=False)
    if apart:
        lim = float(apart_m if apart_m is not None else tol)
        g = ep.gap(t1, A, B, probe=max(lim, 0.05))
        ok = g is None or g > lim
        if not ok:
            issues.append(f"{na} and {nb} are {1000 * g:.1f} mm apart at frame {t1} (must be more than {1000 * lim:.1f} mm): "
                          f"move {na} away from {nb}")
        return _report(ok, issues, end_frame=int(t1), gap_m=_r(g), required_gap_m=lim, mode="apart")
    cs = ep.contacts(t1, A, B, tol)
    end_touch = bool(cs)
    meas = {"end_frame": int(t1), "touching_at_end": end_touch,
            "end_min_dist_m": _r(min(c[0] for c in cs)) if cs else None}
    if not end_touch:
        g = ep.gap(t1, A, B)
        meas["end_gap_m"] = _r(g)
        issues.append(f"{na} and {nb} do not touch at frame {t1} (gap {'> 50' if g is None else f'{1000 * g:.1f}'} mm, "
                      f"tolerance {1000 * tol:.1f} mm): move {na} until it touches {nb}")
    ok = end_touch
    if throughout:
        fs = frames_between(t0, t1, stride)
        hit = [bool(ep.contacts(t, A, B, tol)) for t in fs]
        frac = float(np.mean(hit))
        first_lost = next((t for t, h in zip(fs, hit) if not h), None)
        meas.update(touch_fraction=_r(frac, 3), first_lost_frame=first_lost, frames_evaluated=len(fs))
        if frac < float(min_frac) - 1e-9:
            ok = False
            issues.append(f"{na} and {nb} touch on {100 * frac:.0f}% of frames {t0}-{t1} (needed {100 * float(min_frac):.0f}%), "
                          f"first lost at frame {first_lost}: keep them in contact through the motion")
    return _report(ok, issues, **meas)


def gate_stacked_on(ep, A, Bs, t, tol=TOUCH_TOL_M, level_tol_deg=10.0, normal_min=0.7, allow_table=False,
                    yaw_tol_deg=None, yaw_sym_deg=90.0, names=None):
    """A rests on every body set in Bs at frame t: an A-B contact whose normal points up out of B (A sits on B's top),
    no A-table contact, A level within level_tol_deg, optionally yaw-aligned with the first B."""
    na = (names or {}).get("A", "A")
    nbs = (names or {}).get("B", [f"B{i}" for i in range(len(Bs))])
    issues, per = [], {}
    if all(ep.excluded(A, B) for B in Bs):
        # a hinged lid on its own box: collision is filtered, so 'on' is judged as touching (geometric gap) and level
        for B, nb in zip(Bs, nbs):
            g = ep.geom_gap(t, A, B)
            per[nb] = {"gap_m": _r(g) if g != "mesh" else None, "measured_by": "geom distance (excluded pair)"}
            if g == "mesh":
                issues.append(f"{na} and {nb} are excluded from collision and are meshes: state the relation as a joint target in goal_predicates")
            elif g is None or g > tol:
                issues.append(f"{na} is not resting on {nb} at frame {t} (gap {'> 50' if g is None else f'{1000 * g:.1f}'} mm)")
        # a hinged part always touches its base at the hinge: it rests on it only when it lies flat, i.e. its thinnest
        # axis (a lid's normal) is vertical
        _, RA = ep.pose(t, min(A))
        lo, hi = ep.extent(t, ep.geoms(A), RA)
        ext = hi - lo
        thin = [k for k in range(3) if ext[k] <= 1.2 * float(np.min(ext))]
        lv = min(float(np.degrees(np.arccos(np.clip(abs(RA[2, k]), -1.0, 1.0)))) for k in thin)
        if lv > float(level_tol_deg):
            issues.append(f"{na} is tilted {lv:.1f} deg from lying flat at frame {t} (tolerance {level_tol_deg} deg): it is not closed down on {nbs[0]}")
        return _report(not issues, issues, frame=int(t), supports=per, level_err_deg=_r(lv, 2))
    for B, nb in zip(Bs, nbs):
        cs = ep.contacts(t, A, B, tol)
        top = [c for c in cs if c[2][2] <= -float(normal_min)]       # normal from A into B points down: A is above
        per[nb] = {"contacts": len(cs), "top_contacts": len(top)}
        if not top:
            g = ep.gap(t, A, B) if not cs else 0.0
            per[nb]["gap_m"] = _r(g)
            why = (f"touches it only from the side ({len(cs)} contacts, none with an upward normal)" if cs else
                   f"does not touch it (gap {'> 50' if g is None else f'{1000 * g:.1f}'} mm)")
            issues.append(f"{na} is not resting on {nb} at frame {t}: it {why}; lower {na} onto the top of {nb}")
    tab = ep.contacts(t, A, ep.table, tol) if (ep.table and not allow_table) else []
    if tab:
        issues.append(f"{na} touches the table at frame {t}: it is not stacked (it rests on the table, or tips onto it)")
    _, RA = ep.pose(t, min(A))
    lv = level_deg(RA)
    if lv > float(level_tol_deg):
        issues.append(f"{na} is tilted {lv:.1f} deg from level at frame {t} (tolerance {level_tol_deg} deg): it is not seated flat")
    meas = {"frame": int(t), "supports": per, "touches_table": bool(tab), "level_err_deg": _r(lv, 2)}
    if yaw_tol_deg is not None and Bs:
        _, RB = ep.pose(t, min(Bs[0]))
        dy = yaw_deg(RB.T @ RA)
        e = sym_err(dy, yaw_sym_deg)
        meas["yaw_err_deg"] = _r(e, 2)
        if e > float(yaw_tol_deg):
            issues.append(f"{na} is turned {e:.1f} deg from {nbs[0]} (modulo {yaw_sym_deg} deg; tolerance {yaw_tol_deg}): align it")
    return _report(not issues, issues, **meas)


def _container_frame(ep, t, B):
    """B's root pose with its local axes re-ordered so that the third points up."""
    p, R = ep.pose(t, min(B))
    u = up_axis(R)
    k = int(np.argmax(np.abs(u)))
    perm = [i for i in range(3) if i != k] + [k]
    Rb = R[:, perm].copy()
    Rb[:, 2] *= u[k]
    if np.linalg.det(Rb) < 0:
        Rb[:, 0] *= -1
    return p, Rb


def gate_inside_container(ep, A, B, t, tol=TOUCH_TOL_M, rim_band_frac=0.15, rim_band_min_m=0.004, xy_margin_m=0.0,
                          com_above_rim_m=0.0, entirely=False, names=None):
    """A's centre of mass inside B's interior volume (B's collision extent in B's frame) and A resting on B below the
    rim band (not on the rim), not on the table. entirely: A's whole footprint inside B's."""
    na, nb = names or ("A", "B")
    issues = []
    p, Rb = _container_frame(ep, t, B)
    gB = ep.geoms(B)
    lo, hi = ep.extent(t, gB, Rb)
    c = Rb.T @ (ep.com(t, A) - p)
    # extents are support values along the frame axes (world dot products): shift them to the frame's origin
    off = np.array([float(p @ Rb[:, k]) for k in range(3)])
    lo, hi = lo - off, hi - off
    depth = float(hi[2] - lo[2])
    band = max(float(rim_band_min_m), float(rim_band_frac) * depth)
    m = float(xy_margin_m)
    in_xy = bool(np.all(c[:2] >= lo[:2] + m) and np.all(c[:2] <= hi[:2] - m))
    in_z = bool(lo[2] < c[2] <= hi[2] + float(com_above_rim_m))
    cs = ep.contacts(t, A, B, tol)
    zs = [float(Rb[:, 2] @ (pos - p)) for _, pos, _ in cs]
    below = [z for z in zs if z < hi[2] - band]
    tab = ep.contacts(t, A, ep.table, tol) if ep.table else []
    meas = {"frame": int(t), "com_in_B": [_r(v) for v in c], "B_interior_lo": [_r(v) for v in lo], "B_interior_hi": [_r(v) for v in hi],
            "com_inside_xy": in_xy, "com_inside_height": in_z, "contacts_with_B": len(cs),
            "contacts_below_rim": len(below), "rim_band_m": _r(band), "touches_table": bool(tab),
            "lowest_contact_height_m": _r(min(zs)) if zs else None}
    if not in_xy:
        issues.append(f"{na}'s centre is outside {nb}'s footprint at frame {t} (centre {np.round(c[:2], 3).tolist()} in {nb}'s frame, "
                      f"interior x {lo[0]:.3f}..{hi[0]:.3f}, y {lo[1]:.3f}..{hi[1]:.3f}): place it over the middle of {nb}")
    if not in_z:
        issues.append(f"{na}'s centre is {c[2] - hi[2]:+.3f} m from {nb}'s rim height at frame {t}: it is not down inside {nb}")
    if not cs:
        issues.append(f"{na} does not touch {nb} at frame {t}: it is not resting in it")
    elif not below:
        issues.append(f"{na} touches {nb} only in the top {1000 * band:.0f} mm (the rim) at frame {t}: it rests on the rim, "
                      f"not on {nb}'s inner floor; lower it into the middle")
    if tab:
        issues.append(f"{na} touches the table at frame {t}: it is outside {nb}")
    if entirely:
        gA = ep.geoms(A)
        loA, hiA = ep.extent(t, gA, Rb)
        loA, hiA = loA - off, hiA - off
        inside = bool(np.all(loA[:2] >= lo[:2] - 1e-4) and np.all(hiA[:2] <= hi[:2] + 1e-4))
        meas["entirely_inside_footprint"] = inside
        if not inside:
            issues.append(f"{na} sticks out over {nb}'s edge at frame {t} (A spans x {loA[0]:.3f}..{hiA[0]:.3f}, y {loA[1]:.3f}..{hiA[1]:.3f})")
    return _report(not issues, issues, **meas)


def point_in_polygon(pt, poly):
    x, y = float(pt[0]), float(pt[1])
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xc = x1 + (y - y1) * (x2 - x1) / ((y2 - y1) or 1e-12)
            if x < xc:
                inside = not inside
    return inside


def region_polygon(region):
    if not isinstance(region, dict):
        raise ValueError("region: give {'box': [[xmin, ymin], [xmax, ymax]]}, {'polygon': [[x, y], ...]} or "
                         "{'center': [x, y], 'radius': r}")
    if "box" in region:
        (x0, y0), (x1, y1) = region["box"]
        return [(min(x0, x1), min(y0, y1)), (max(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1)), (min(x0, x1), max(y0, y1))]
    if "polygon" in region:
        poly = [tuple(map(float, q[:2])) for q in region["polygon"]]
        if len(poly) < 3:
            raise ValueError("region: a polygon needs at least 3 points")
        return poly
    if "center" in region and "radius" in region:
        cx, cy = map(float, region["center"][:2])
        r = float(region["radius"])
        return [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 48, endpoint=False)]
    raise ValueError("region: needs 'box', 'polygon' or 'center'+'radius'")


def table_top(ep):
    if not ep.table:
        return 0.0
    gs = ep.geoms(ep.table - {0})
    if not gs:
        return 0.0
    return float(ep.support(0, gs, [0, 0, 1.0]))


def gate_target_region(ep, A, t0, t1, region, z_range=None, point="origin", yaw_target_deg=None, yaw_tol_deg=None,
                       yaw_sym_deg=None, upright_tol_deg=None, hold_s=0.0, names=None):
    """A ends inside a table-frame region (x, y as in the scene's world, z measured from the tabletop), optionally
    within a height band and an orientation tolerance, and stays there for the last hold_s of the window."""
    na = (names or ("A",))[0]
    poly = region_polygon(region)
    zt = table_top(ep)
    bid = min(A)
    _, R0 = ep.pose(0, bid)
    u0 = up_axis(R0)
    hold_n = int(round(float(hold_s) * ep.fps))
    fs = frames_between(max(t0, t1 - hold_n), t1)
    rows, issues = [], []
    for t in fs:
        pt = ep.com(t, A) if point == "com" else ep.pose(t, bid)[0]
        _, R = ep.pose(t, bid)
        ok_xy = point_in_polygon(pt[:2], poly)
        z = float(pt[2] - zt)
        ok_z = True if z_range is None else (float(z_range[0]) <= z <= float(z_range[1]))
        row = {"frame": int(t), "xy": [_r(pt[0]), _r(pt[1])], "z_above_table": _r(z), "in_region": ok_xy, "in_height": ok_z}
        ok_o = True
        if yaw_target_deg is not None:
            e = sym_err(yaw_deg(R) - float(yaw_target_deg), yaw_sym_deg) if yaw_sym_deg else abs(wrap180(yaw_deg(R) - float(yaw_target_deg)))
            row["yaw_err_deg"] = _r(e, 2)
            ok_o &= e <= float(yaw_tol_deg if yaw_tol_deg is not None else 15.0)
        if upright_tol_deg is not None:
            tl = tilt_deg(R, u0)
            row["tilt_deg"] = _r(tl, 2)
            ok_o &= tl <= float(upright_tol_deg)
        row["orientation_ok"] = bool(ok_o)
        row["ok"] = bool(ok_xy and ok_z and ok_o)
        rows.append(row)
    end = rows[-1]
    if not end["in_region"]:
        cx, cy = np.mean([q[0] for q in poly]), np.mean([q[1] for q in poly])
        issues.append(f"{na} ends at ({end['xy'][0]:.3f}, {end['xy'][1]:.3f}) at frame {t1}, outside the target region "
                      f"(centre ({cx:.3f}, {cy:.3f})): move it {np.hypot(cx - end['xy'][0], cy - end['xy'][1]) * 100:.1f} cm toward the region")
    if not end["in_height"]:
        issues.append(f"{na} ends {end['z_above_table']:.3f} m above the table at frame {t1}, outside the height band {list(z_range)}")
    if not end["orientation_ok"]:
        issues.append(f"{na}'s orientation is off at frame {t1} ({ {k: end[k] for k in ('yaw_err_deg', 'tilt_deg') if k in end} })")
    bad_hold = [r["frame"] for r in rows[:-1] if not r["ok"]]
    if hold_n and bad_hold:
        issues.append(f"{na} is outside the region during the final {hold_s} s (frames {bad_hold[:5]}): it must stay there")
    return _report(not issues and end["ok"], issues, end=end, frames_evaluated=len(rows))


def _quat_to_R(q):
    w, x, y, z = q
    return Rot.from_quat([x, y, z, w]).as_matrix()


def gate_relative_pose(ep, A, B, t, rel_pos=None, rel_quat=None, rel_euler_deg=None, reference_frame=None, pos_tol_m=0.01,
                       rot_tol_deg=10.0, pos_axes="xyz", sym_axis=None, sym_deg=None, names=None):
    """Pose of A's root in B's root frame at frame t against a target (rel_pos [m], rel_quat wxyz or rel_euler_deg xyz,
    or the relative pose at reference_frame)."""
    na, nb = names or ("A", "B")
    a, b = min(A), min(B)
    pA, RA = ep.pose(t, a)
    pB, RB = ep.pose(t, b)
    rp = RB.T @ (pA - pB)
    rR = RB.T @ RA
    if reference_frame is not None:
        qa, Qa = ep.pose(int(reference_frame), a)
        qb, Qb = ep.pose(int(reference_frame), b)
        rel_pos = (Qb.T @ (qa - qb)).tolist()
        tR = Qb.T @ Qa
    else:
        tR = None
        if rel_quat is not None:
            tR = _quat_to_R(rel_quat)
        elif rel_euler_deg is not None:
            tR = Rot.from_euler("xyz", rel_euler_deg, degrees=True).as_matrix()
    if rel_pos is None and tR is None:
        raise ValueError("rel_pos / rel_quat / rel_euler_deg / reference_frame: give a target relative pose")
    issues, meas = [], {"frame": int(t), "rel_pos": [_r(v) for v in rp], "rel_euler_deg": [_r(v, 2) for v in Rot.from_matrix(rR).as_euler("xyz", degrees=True)]}
    ok = True
    if rel_pos is not None:
        ax = ["xyz".index(c) for c in pos_axes]
        err = float(np.linalg.norm((rp - np.asarray(rel_pos, float))[ax]))
        meas.update(target_rel_pos=[_r(v) for v in rel_pos], pos_err_m=_r(err))
        if err > float(pos_tol_m):
            ok = False
            d = (np.asarray(rel_pos, float) - rp)
            issues.append(f"{na} is {1000 * err:.1f} mm from its target pose relative to {nb} at frame {t} (tolerance "
                          f"{1000 * float(pos_tol_m):.1f} mm); move it by {np.round(RB @ d, 4).tolist()} m (world)")
    if tR is not None:
        cands = [tR]
        if sym_axis is not None and sym_deg:
            n = int(round(360.0 / float(sym_deg)))
            ax_ = np.asarray(sym_axis, float) / np.linalg.norm(sym_axis)
            cands = [tR @ Rot.from_rotvec(ax_ * np.radians(k * float(sym_deg))).as_matrix() for k in range(n)]
        ang = min(float(np.degrees(np.linalg.norm(Rot.from_matrix(c.T @ rR).as_rotvec()))) for c in cands)
        meas["rot_err_deg"] = _r(ang, 2)
        if ang > float(rot_tol_deg):
            ok = False
            issues.append(f"{na} is rotated {ang:.1f} deg from its target orientation relative to {nb} at frame {t} "
                          f"(tolerance {rot_tol_deg} deg)")
    return _report(ok, issues, **meas)


def joint_info(ep, joint, param="joint"):
    m = ep.model
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, str(joint))
    if j < 0:
        raise ValueError(f"{param}: no joint {joint!r} in the scene")
    ty = int(m.jnt_type[j])
    if ty not in (HINGE, SLIDE):
        raise ValueError(f"{param}: {joint!r} is not a hinge or slide joint")
    lim = bool(m.jnt_limited[j])
    rng = (float(m.jnt_range[j][0]), float(m.jnt_range[j][1])) if lim else None
    return {"id": int(j), "adr": int(m.jnt_qposadr[j]), "type": "hinge" if ty == HINGE else "slide", "range": rng,
            "body": int(m.jnt_bodyid[j])}


def closed_value(rng):
    """The closed (assembled) value of an articulation: 0 when inside the range (the library's convention), else the limit nearest 0."""
    lo, hi = rng
    if lo - 1e-9 <= 0.0 <= hi + 1e-9:
        return 0.0
    return lo if abs(lo) < abs(hi) else hi


def gate_joint_target(ep, joint, t0, t1, target=None, fraction=None, state=None, tol=None, open_fraction=0.5,
                      hold=True, names=None):
    """The joint reaches its target inside [t0, t1] and (hold) stays there to t1. target: a value; fraction: of the
    range from the lower limit; state 'closed' (the closed value +- tol) or 'open' (at least open_fraction of the way
    from closed to the far limit)."""
    ji = joint_info(ep, joint)
    q = np.asarray(ep.Q[t0:t1 + 1, ji["adr"]], float)
    hinge = ji["type"] == "hinge"
    tol = float(tol) if tol is not None else (np.radians(10.0) if hinge else 0.01)
    unit = (lambda v: round(float(np.degrees(v)), 2)) if hinge else (lambda v: round(float(v), 4))
    uname = "deg" if hinge else "m"
    mode, thr = "at", None
    if state is not None:
        if ji["range"] is None:
            raise ValueError(f"state: joint {joint!r} has no range, so 'open'/'closed' cannot be resolved; pass target")
        c = closed_value(ji["range"])
        if state == "closed":
            target = c
        elif state == "open":
            far = ji["range"][1] if abs(ji["range"][1] - c) >= abs(ji["range"][0] - c) else ji["range"][0]
            thr = c + float(open_fraction) * (far - c)
            mode = "beyond"
            sgn = 1.0 if far >= c else -1.0
        else:
            raise ValueError("state: 'open' or 'closed'")
    elif fraction is not None:
        if ji["range"] is None:
            raise ValueError(f"fraction: joint {joint!r} has no range; pass target")
        target = ji["range"][0] + float(fraction) * (ji["range"][1] - ji["range"][0])
    if mode == "at" and target is None:
        raise ValueError("target / fraction / state: give one")
    good = (sgn * (q - thr) >= 0) if mode == "beyond" else (np.abs(q - float(target)) <= tol)
    hit = np.nonzero(good)[0]
    meas = {"joint": str(joint), "unit": uname, "start": unit(q[0]), "end": unit(q[-1]),
            "target": unit(target) if target is not None else None, "open_threshold": unit(thr) if thr is not None else None,
            "tol": unit(tol), "range": [unit(v) for v in ji["range"]] if ji["range"] else None}
    issues = []
    if not len(hit):
        best = float(q[np.argmin(np.abs(q - float(target)))]) if mode == "at" else float(q[np.argmax(sgn * q)])
        meas["closest"] = unit(best)
        want = f"{unit(target)} +- {unit(tol)}" if mode == "at" else f"beyond {unit(thr)}"
        issues.append(f"joint {joint} never reaches {want} {uname} in frames {t0}-{t1} (closest {unit(best)}, end {unit(q[-1])}): "
                      f"drive the part {'to its target' if mode == 'at' else 'farther open'}")
        return _report(False, issues, **meas)
    first = int(hit[0])
    meas["reached_frame"] = int(t0 + first)
    if hold:
        after = good[first:]
        if not np.all(after):
            lost = int(t0 + first + int(np.argmin(after)))
            meas["lost_frame"] = lost
            issues.append(f"joint {joint} reaches its target at frame {t0 + first} but leaves it at frame {lost} (end {unit(q[-1])} {uname}): "
                          "it must stay there to the end of the window (a latch that springs back, a lid that reopens)")
    return _report(not issues, issues, **meas)


def gate_held(ep, A, t0, t1, H, mode="in_hand", tol=HOLD_TOL_M, min_frac=0.9, max_gap_s=0.25, opposing_deg=100.0,
              min_links=2, allow_table=False, stay_tol_m=0.01, stay_tol_deg=8.0, stride=1, names=None):
    """mode 'in_hand': on every frame the hand touches A with opposing contacts on at least min_links links and A has
    no table support; 'fixed': the hand touches A and A does not move (a steadying hold)."""
    na = (names or ("A",))[0]
    fs = frames_between(t0, t1, stride)
    cos_opp = float(np.cos(np.radians(opposing_deg)))
    rows = []
    m = ep.model
    with Inflate(ep, set(A) | set(H), tol):
        for t in fs:
            d = ep.at(t, collide=True)
            links, normals, tab = set(), [], False
            for i in range(d.ncon):
                c = d.contact[i]
                if float(c.dist) > tol:
                    continue
                b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
                if b1 in H and b2 in A:
                    links.add(b1); normals.append(np.asarray(c.frame[:3], float))
                elif b2 in H and b1 in A:
                    links.add(b2); normals.append(-np.asarray(c.frame[:3], float))
                elif (b1 in A and b2 in ep.table) or (b2 in A and b1 in ep.table):
                    tab = True
            opp = False
            if len(normals) >= 2:
                N = np.array(normals)
                opp = bool(np.min(N @ N.T) <= cos_opp)
            if mode == "fixed":
                okf = len(links) >= 1
            else:
                okf = len(links) >= int(min_links) and opp and (allow_table or not tab)
            rows.append((t, okf, len(links), opp, tab))
    good = np.array([r[1] for r in rows])
    frac = float(good.mean())
    run, longest, start_bad, worst_start = 0, 0, None, None
    for (t, okf, *_rest) in rows:
        if not okf:
            if run == 0:
                start_bad = t
            run += 1
            if run > longest:
                longest, worst_start = run, start_bad
        else:
            run = 0
    step = (fs[1] - fs[0]) if len(fs) > 1 else 1
    longest_s = longest * step / ep.fps
    issues = []
    meas = {"mode": mode, "frames": [int(t0), int(t1)], "held_fraction": _r(frac, 3), "longest_unheld_s": _r(longest_s, 3),
            "longest_unheld_from": worst_start, "end_links": rows[-1][2], "end_opposing": rows[-1][3], "end_on_table": rows[-1][4]}
    if mode == "fixed":
        p0, R0 = ep.pose(t0, min(A))
        mv, rot = 0.0, 0.0
        for t in fs:
            p, R = ep.pose(t, min(A))
            mv = max(mv, float(np.linalg.norm(p - p0)))
            rot = max(rot, float(np.degrees(np.linalg.norm(Rot.from_matrix(R0.T @ R).as_rotvec()))))
        meas.update(moved_m=_r(mv), rotated_deg=_r(rot, 2))
        if mv > float(stay_tol_m) or rot > float(stay_tol_deg):
            issues.append(f"{na} moved {1000 * mv:.0f} mm / {rot:.0f} deg while held fixed in frames {t0}-{t1} "
                          f"(tolerance {1000 * float(stay_tol_m):.0f} mm / {stay_tol_deg} deg): hold it firmer or press it onto the table")
    if frac < float(min_frac) - 1e-9 or longest_s > float(max_gap_s) or not rows[-1][1]:
        bad = next((r for r in rows if not r[1]), rows[-1])
        why = ("on the table" if bad[4] else ("no opposing contacts" if bad[2] >= 1 and not bad[3] else f"{bad[2]} hand links touching"))
        what = "touched by the hand" if mode == "fixed" else "held"
        issues.append(f"{na} is {what} on only {100 * frac:.0f}% of frames {t0}-{t1} (longest gap {longest_s:.2f} s from frame "
                      f"{worst_start}; at frame {bad[0]}: {why}{'' if rows[-1][1] else '; not held at the last frame'}): "
                      "grip it with opposing fingers before the carry and keep the grip closed")
    return _report(not issues, issues, **meas)


def last_robot_contact(ep, A, t0, t1, tol=HOLD_TOL_M):
    """Last frame in [t0, t1] where any robot body touches A (None when none does)."""
    R = ep.robot
    if not R:
        return None
    with Inflate(ep, set(A) | R, tol):
        for t in range(int(t1), int(t0) - 1, -1):
            d = ep.at(t, collide=True)
            m = ep.model
            for i in range(d.ncon):
                c = d.contact[i]
                if float(c.dist) > tol:
                    continue
                b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
                if (b1 in A and b2 in R) or (b2 in A and b1 in R):
                    return t
    return None


def robot_touches(ep, t, A, tol=HOLD_TOL_M):
    return bool(ep.robot) and bool(ep.contacts(t, A, ep.robot, tol))


def gate_release_settle(ep, A, t0, t1, settle_s=0.3, pos_tol_m=0.004, rot_tol_deg=10.0, min_settle_s=0.15,
                        seat_s=0.15, seat_tol_m=0.015, seat_tol_deg=35.0,
                        orientation=None, orient_tol_deg=15.0, on=None, release_frame=None, names=None):
    """After the last robot contact (the release), A stays where it was put: within seat_s it may seat itself (a brick
    released at a slight tilt drops onto its studs) by at most seat_tol_m / seat_tol_deg, and over the following
    settle_s its centre moves less than pos_tol_m and its up axis tips less than rot_tol_deg (no fall, roll, slide or
    tip-over; spinning or rocking in place is allowed). It is supported at
    the end (and by `on` when given) and is not touched again. orientation: 'upright' / 'upside_down' (its frame-0 up
    axis within orient_tol_deg of up / down) / 'horizontal'."""
    na = (names or {}).get("A", "A") if isinstance(names, dict) else (names or ("A",))[0]
    issues = []
    r = last_robot_contact(ep, A, t0, t1) if release_frame is None else int(release_frame)
    start = (r + 1) if r is not None else t0
    seat_n = int(round(float(seat_s) * ep.fps)) if r is not None else 0
    n = seat_n + int(round(float(settle_s) * ep.fps))
    end = start + n
    stop = None
    if end > ep.T - 1:
        end = ep.T - 1
    if end > t1:
        # the window ends before the settle time has passed: keep watching the following frames until something touches A
        for t in range(t1 + 1, end + 1):
            if robot_touches(ep, t, A):
                stop = t
                break
        if stop is not None:
            end = stop - 1
    mid = min(start + seat_n, end)
    avail = (end - mid) / ep.fps
    bid = min(A)
    meas = {"release_frame": r, "settle_frames": [int(mid), int(end)], "settle_s": _r(avail, 3)}
    if start > ep.T - 1 or avail < float(min_settle_s) - 1e-9:
        issues.append(f"{na} is released at frame {r} with only {max(0.0, (end - start) / ep.fps):.2f} s of trajectory before it is touched "
                      f"again or the episode ends: let it rest at least {seat_s + min_settle_s:.2f} s after the release to show it stays")
        return _report(False, issues, **meas)

    def motion(a, b):
        # translation of the centre and tipping of the up axis: an object that spins or rocks in place (a soft toy on its
        # rounded belly) stays where it was put; one that falls, slides, rolls away or tips over does not
        pa, Ra = ep.pose(a, bid)
        ua = up_axis(Ra)
        ca = ep.com(a, A)
        mv, rot, worst = 0.0, 0.0, a
        for t in range(a, b + 1):
            p, R = ep.pose(t, bid)
            dm = float(np.linalg.norm(ep.com(t, A) - ca))
            if dm > mv:
                mv, worst = dm, t
            rot = max(rot, float(np.degrees(np.arccos(np.clip((R @ ua) @ (Ra @ ua), -1.0, 1.0)))))
        return mv, rot, worst
    def drift(a, b):
        # still after seating: the mean pose over the first third of the window against the last third, so an object
        # rocking in place (an undamped soft toy) averages out while one sliding, rolling or tipping does not
        n3 = max(1, (b - a + 1) // 3)
        _, Ra = ep.pose(a, bid)
        ua = up_axis(Ra)
        first = list(range(a, a + n3))
        last = list(range(b - n3 + 1, b + 1))
        cf = np.mean([ep.com(t, A) for t in first], axis=0)
        cl = np.mean([ep.com(t, A) for t in last], axis=0)
        uf = np.mean([ep.pose(t, bid)[1] @ ua for t in first], axis=0)
        ul = np.mean([ep.pose(t, bid)[1] @ ua for t in last], axis=0)
        ang = float(np.degrees(np.arccos(np.clip(uf @ ul / max(np.linalg.norm(uf) * np.linalg.norm(ul), 1e-12), -1.0, 1.0))))
        worst_t = max(range(a, b + 1), key=lambda t: float(np.linalg.norm(ep.com(t, A) - cf)))
        return float(np.linalg.norm(cl - cf)), ang, worst_t
    seat_mv, seat_rot, _ = motion(start, end)               # everything after the release, the seating included
    mv, rot, worst = drift(mid, end)                         # after the seating time: must be still
    p0, _ = ep.pose(start, bid)
    pe, Re = ep.pose(end, bid)
    dz = float(pe[2] - p0[2])
    meas.update(moved_m=_r(mv), rotated_deg=_r(rot, 2), worst_frame=int(worst), moved_since_release_m=_r(seat_mv),
                rotated_since_release_deg=_r(seat_rot, 2), dz_since_release_m=_r(dz))
    if seat_mv > float(seat_tol_m) or seat_rot > float(seat_tol_deg):
        kind = "falls" if dz < -0.5 * seat_mv else ("tips over" if seat_rot > float(seat_tol_deg) else "slides or rolls")
        issues.append(f"{na} {kind} after the release at frame {r}: it moves {1000 * seat_mv:.1f} mm / {seat_rot:.1f} deg before frame {end} "
                      f"(at most {1000 * float(seat_tol_m):.0f} mm / {seat_tol_deg} deg of seating is allowed); set it down on a stable spot, "
                      "lower, before letting go")
    elif mv > float(pos_tol_m) or rot > float(rot_tol_deg):
        issues.append(f"{na} is still moving {seat_s:.2f} s after the release at frame {r}: {1000 * mv:.1f} mm / {rot:.1f} deg in frames "
                      f"{mid}-{end} (tolerance {1000 * float(pos_tol_m):.0f} mm / {rot_tol_deg} deg): it slides or rolls; set it down on a stable spot")
    others = set(range(ep.model.nbody)) - set(A) - ep.robot
    sup = ep.contacts(end, A, others, TOUCH_TOL_M)
    meas["supported"] = bool(sup)
    if not sup:
        issues.append(f"{na} touches nothing at frame {end} after the release: it is falling")
    if on is not None:
        S = on if isinstance(on, set) else ep.bodies(on, "on")
        hit = ep.contacts(end, A, S, TOUCH_TOL_M)
        meas["on_support"] = bool(hit)
        if not hit:
            issues.append(f"{na} does not rest on {(names or {}).get('on', 'its support') if isinstance(names, dict) else 'its support'} "
                          f"at frame {end} after the release")
    if orientation:
        _, Rs = ep.pose(0, bid)
        u0 = up_axis(Rs)
        tl = tilt_deg(Re, u0)
        meas["tilt_from_start_up_deg"] = _r(tl, 2)
        want = {"upright": (0.0, float(orient_tol_deg)), "upside_down": (180.0 - float(orient_tol_deg), 180.0),
                "horizontal": (90.0 - float(orient_tol_deg) - 15.0, 90.0 + float(orient_tol_deg) + 15.0)}.get(orientation)
        if want is None:
            raise ValueError("orientation: 'upright', 'upside_down' or 'horizontal'")
        if not (want[0] - 1e-6 <= tl <= want[1] + 1e-6):
            issues.append(f"{na} ends {tl:.0f} deg from its starting up axis at frame {end}: it is not {orientation.replace('_', ' ')}")
    return _report(not issues, issues, **meas)


def gate_lift_clearance(ep, A, t0, t1, min_height_m=0.01, support=None, at="end", tol=HOLD_TOL_M, names=None):
    """A's lowest point rises at least min_height_m above where it rested at t0 and A no longer touches its support
    (what it touched at t0, or `support`), at t1 ('end') or on some frame ('any')."""
    na = (names or ("A",))[0]
    m = ep.model
    others = set(range(m.nbody)) - set(A) - ep.robot

    def touched(t):
        S_ = set()
        d = ep.at(t, collide=True)
        for i in range(d.ncon):
            c = d.contact[i]
            if float(c.dist) > TOUCH_TOL_M:
                continue
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if b1 in A and b2 in others:
                S_.add(b2)
            elif b2 in A and b1 in others:
                S_.add(b1)
        return S_
    ref = t0
    with Inflate(ep, set(A), TOUCH_TOL_M):
        S0 = touched(t0)
        if support is None and not S0:
            # already off its support when the window starts (a cap freed by the last turn): measure from the last frame it
            # rested on something
            for t in range(t0 - 1, max(-1, t0 - int(20 * ep.fps)), -3):
                S0 = touched(t)
                if S0:
                    ref = t
                    break
    if support is None:
        S = S0 or set(ep.table)
    else:
        S = set(support)
    z0 = ep.lowest(ref, A)
    fs = [t1] if at == "end" else frames_between(t0, t1, 3)
    if fs[-1] != t1:
        fs.append(t1)
    best = None
    for t in fs:
        rise = ep.lowest(t, A) - z0
        touch = bool(ep.contacts(t, A, S, tol))
        # clearance: how far A is from its support -- straight up from a table, or along the axis of a held bottle it was
        # unscrewed from; a support farther than the probe counts as clear
        g = None if touch else ep.gap(t, A, S, probe=max(0.05, 2 * float(min_height_m)))
        clear = 0.0 if touch else (np.inf if g is None else g)
        okf = (rise >= float(min_height_m) or clear >= float(min_height_m)) and not touch
        if best is None or (okf and not best[2]) or (okf == best[2] and rise > best[1]):
            best = (t, rise, okf, touch, clear)
    t, rise, okf, touch, clear = best
    meas = {"frame": int(t), "rise_m": _r(rise), "clearance_m": None if not np.isfinite(clear) else _r(clear), "touches_support": touch,
            "support": sorted(ep.name(b) for b in S)[:6], "start_lowest_z": _r(z0), "reference_frame": int(ref)}
    issues = []
    if not okf:
        why = []
        if rise < float(min_height_m) and clear < float(min_height_m):
            why.append(f"its bottom rose {1000 * rise:.1f} mm and it is {1000 * clear:.1f} mm from its support (needed {1000 * float(min_height_m):.0f} mm)")
        if touch:
            why.append("it still touches its support")
        issues.append(f"{na} is not lifted clear at frame {t}: {' and '.join(why)}; lift it higher before the next phase")
    return _report(okf, issues, **meas)


def gate_insertion_depth(ep, A, B, t, axis=None, entry=None, min_depth_m=None, min_fraction=0.5, point="leading",
                         tol=TOUCH_TOL_M, names=None):
    """Depth of A along an axis into B. axis: the insertion direction in B's frame; entry: a point of the entry plane in
    B's frame (default: B's outer face on the side A enters from). Passes when the chosen point of A (leading / com /
    trailing) is at least min_depth_m past the entry plane, or, without min_depth_m, when min_fraction of A's length
    along the axis is past it. Without an axis: A is seated in B's envelope (its centre inside B's collision extent
    in B's frame), touches B and not the table."""
    na, nb = names or ("A", "B")
    pB, RB = ep.pose(t, min(B))
    gA, gB = ep.geoms(A), ep.geoms(B)
    cs = ep.contacts(t, A, B, tol)
    tab = bool(ep.contacts(t, A, ep.table, tol)) if ep.table else False
    issues = []
    if axis is None:
        off = np.array([float(pB @ RB[:, k]) for k in range(3)])
        lo, hi = ep.extent(t, gB, RB)
        lo, hi = lo - off, hi - off
        c = RB.T @ (ep.com(t, A) - pB)
        inside = bool(np.all(c >= lo) and np.all(c <= hi))
        margin = float(np.min(np.concatenate([c - lo, hi - c])))
        meas = {"frame": int(t), "mode": "envelope", "com_in_B": [_r(v) for v in c], "B_lo": [_r(v) for v in lo], "B_hi": [_r(v) for v in hi],
                "depth_inside_m": _r(margin), "touches_B": bool(cs), "touches_table": tab}
        if not inside:
            k = int(np.argmin(np.concatenate([c - lo, hi - c])))
            issues.append(f"{na}'s centre is {1000 * -margin:.1f} mm outside {nb}'s envelope at frame {t} (along {nb}'s "
                          f"{'xyz'[k % 3]} axis): push it farther into {nb}")
        if not cs:
            issues.append(f"{na} does not touch {nb} at frame {t}: it is not seated in it")
        # a slot with an open floor (a battery holder) lets the seated part rest on the table: table contact is reported only
        return _report(not issues, issues, **meas)
    ax = np.asarray(axis, float)
    ax = ax / max(np.linalg.norm(ax), 1e-12)
    d = RB @ ax
    if entry is not None:
        s_entry = float((pB + RB @ np.asarray(entry, float)) @ d)
    else:
        s_entry = -ep.support(t, gB, -d)
    lead = ep.support(t, gA, d)
    trail = -ep.support(t, gA, -d)
    comd = float(ep.com(t, A) @ d)
    val = {"leading": lead, "com": comd, "trailing": trail}.get(point)
    if val is None:
        raise ValueError("point: 'leading', 'com' or 'trailing'")
    depth = val - s_entry
    L = max(lead - trail, 1e-9)
    frac = float(np.clip((lead - s_entry) / L, 0.0, 1.0))
    meas = {"frame": int(t), "mode": "axis", "axis_world": [_r(v) for v in d], "depth_m": _r(depth), "point": point,
            "inserted_fraction": _r(frac, 3), "length_along_axis_m": _r(L), "touches_B": bool(cs), "touches_table": tab}
    if min_depth_m is not None:
        ok = depth >= float(min_depth_m)
        if not ok:
            issues.append(f"{na}'s {point} point is {1000 * depth:.1f} mm past {nb}'s entry plane at frame {t} (needed "
                          f"{1000 * float(min_depth_m):.1f} mm): push it {1000 * (float(min_depth_m) - depth):.1f} mm farther along {np.round(d, 2).tolist()}")
    else:
        ok = frac >= float(min_fraction)
        if not ok:
            issues.append(f"only {100 * frac:.0f}% of {na}'s length is inside {nb} at frame {t} (needed {100 * float(min_fraction):.0f}%): "
                          f"push it farther along {np.round(d, 2).tolist()}")
    return _report(ok, issues, **meas)


def gate_rotation_turns(ep, t0, t1, min_deg, joint=None, body=None, axis=None, relative_to=None, direction=None,
                        release_joint=None, release_travel_m=None, names=None):
    """A part turned at least min_deg about its axis in [t0, t1], wraps counted: a hinge joint's travel, or a body's
    rotation about an axis (body frame) integrated frame by frame (optionally relative to another body).
    release_joint + release_travel_m: the turn must also run the thread out (the lift joint reaches the travel)."""
    if joint is None and body is None:
        raise ValueError("joint / body: give the turning joint or the turning body")
    issues, meas = [], {}
    if joint is not None:
        ji = joint_info(ep, joint)
        if ji["type"] != "hinge":
            raise ValueError(f"joint: {joint!r} is a slide; rotation needs a hinge")
        q = np.asarray(ep.Q[t0:t1 + 1, ji["adr"]], float)
        total = float(np.degrees(q[-1] - q[0]))
        peak = float(np.degrees(np.max(np.abs(q - q[0]))))
        meas.update(source=f"joint {joint}", total_deg=_r(total, 1), peak_deg=_r(peak, 1))
    else:
        a = int(body)
        ref = int(relative_to) if relative_to is not None else None
        ax = np.asarray(axis if axis is not None else [0, 0, 1.0], float)
        ax = ax / max(np.linalg.norm(ax), 1e-12)
        prev = None
        total, cum = 0.0, [0.0]
        for t in range(t0, t1 + 1):
            _, R = ep.pose(t, a)
            if ref is not None:
                _, Rr = ep.pose(t, ref)
                R = Rr.T @ R
            if prev is not None:
                rv = Rot.from_matrix(prev.T @ R).as_rotvec()     # increment in the body frame
                total += float(np.degrees(rv @ ax))
                cum.append(total)
            prev = R
        peak = float(np.max(np.abs(cum)))
        meas.update(source=f"body {ep.name(a)} about {ax.round(3).tolist()}", total_deg=_r(total, 1), peak_deg=_r(peak, 1))
    turned = abs(total) if direction is None else float(np.sign(direction)) * total
    ok = turned >= float(min_deg)
    meas["required_deg"] = _r(min_deg, 1)
    meas["turns"] = _r(turned / 360.0, 3)
    if release_joint is not None and release_travel_m is not None:
        rj = joint_info(ep, release_joint, "release_joint")
        lmax = float(np.max(np.abs(ep.Q[t0:t1 + 1, rj["adr"]])))
        rel = lmax >= float(release_travel_m) - 1e-5
        meas.update(release_travel_m=_r(lmax), release_needed_m=_r(release_travel_m), released=rel)
        if not rel:
            issues.append(f"the thread is not run out in frames {t0}-{t1}: {release_joint} travelled {1000 * lmax:.1f} mm of "
                          f"{1000 * float(release_travel_m):.1f} mm (turned {abs(total):.0f} deg): keep turning until the part comes free")
        ok = rel                                # a cap that came free has turned as far as it can
    elif not ok:
        issues.append(f"the part turned {turned:.0f} deg in frames {t0}-{t1} (needed {float(min_deg):.0f} deg"
                      f"{'' if direction is None else ' in the requested direction'}): add turning bites")
    return _report(ok, issues, **meas)


def gate_tilt_pour(ep, A, t0, t1, min_tilt_deg=60.0, target=None, region=None, mouth_local=None, up_local=None,
                   min_time_s=0.5, target_margin_m=0.01, at_end=False, names=None):
    """A tilted past min_tilt_deg (its frame-0 up axis from vertical) with its mouth over a target body's footprint or
    a table-frame region, for a contiguous min_time_s inside [t0, t1]. No target and no region: tilt alone."""
    na = (names or ("A", "target"))[0]
    nt = (names or ("A", "target"))[1] if names and len(names) > 1 else "the target"
    bid = min(A)
    _, R0 = ep.pose(0, bid)
    u0 = np.asarray(up_local, float) if up_local is not None else up_axis(R0)
    gA = ep.geoms(A)
    if mouth_local is None:
        p0, _ = ep.pose(0, bid)
        h = ep.support(0, gA, R0 @ u0) - float(p0 @ (R0 @ u0))
        mouth_local = u0 * h
    mouth_local = np.asarray(mouth_local, float)
    poly = region_polygon(region) if region is not None else None
    gT = ep.geoms(target) if target is not None else None
    run, best, best_start, start = 0, 0, None, None
    peak = 0.0
    rows = 0
    over_any = False
    for t in range(t0, t1 + 1):
        p, R = ep.pose(t, bid)
        tl = tilt_deg(R, u0)
        peak = max(peak, tl)
        mouth = p + R @ mouth_local
        over = True
        if poly is not None:
            over = point_in_polygon(mouth[:2], poly)
        elif gT is not None:
            lo, hi = ep.extent(t, gT)
            mg = float(target_margin_m)
            over = bool(lo[0] - mg <= mouth[0] <= hi[0] + mg and lo[1] - mg <= mouth[1] <= hi[1] + mg)
        over_any |= over and tl >= float(min_tilt_deg)
        good = tl >= float(min_tilt_deg) and over
        rows += 1
        if good:
            if run == 0:
                start = t
            run += 1
            if run > best:
                best, best_start = run, start
        else:
            run = 0
    if at_end:                                   # a kept tilt: the qualifying run must reach the window's last frame
        best, best_start = run, (start if run else None)
    dur = best / ep.fps
    _, Re = ep.pose(t1, bid)
    meas = {"peak_tilt_deg": _r(peak, 1), "end_tilt_deg": _r(tilt_deg(Re, u0), 1), "required_tilt_deg": _r(min_tilt_deg, 1),
            "longest_pour_s": _r(dur, 3), "pour_from_frame": best_start, "required_s": _r(min_time_s, 3), "at_end": bool(at_end)}
    issues = []
    ok = dur >= float(min_time_s) - 1e-9
    if not ok:
        if at_end and peak >= float(min_tilt_deg) and meas["end_tilt_deg"] < float(min_tilt_deg):
            issues.append(f"{na} ends the window tilted {meas['end_tilt_deg']:.0f} deg (needed {float(min_tilt_deg):.0f} deg kept to frame {t1}): "
                          "hold the tilt to the end of the subgoal")
        elif peak < float(min_tilt_deg):
            issues.append(f"{na} tilts at most {peak:.0f} deg in frames {t0}-{t1} (needed {float(min_tilt_deg):.0f} deg): tilt it further")
        elif (poly is not None or gT is not None) and not over_any:
            issues.append(f"{na} tilts to {peak:.0f} deg but its mouth is never over {nt} while tilted: bring the mouth over {nt} before tilting")
        else:
            issues.append(f"{na} stays tilted past {float(min_tilt_deg):.0f} deg{' over ' + nt if (poly is not None or gT is not None) else ''} "
                          f"for only {dur:.2f} s (needed {float(min_time_s):.2f} s): hold the tilt longer")
    return _report(ok, issues, **meas)


def count_strokes(x, d, release_frac=0.3):
    """Excursions of x (displacement, >= 0) that reach d and come back below release_frac * d."""
    n, armed = 0, True
    for v in x:
        if armed and v >= d:
            n += 1
            armed = False
        elif not armed and v <= release_frac * d:
            armed = True
    return n


def gate_press_displacement(ep, t0, t1, min_m, joint=None, body=None, direction=None, relative_to=None, min_strokes=1,
                            peak=True, names=None):
    """A pressed part moved at least min_m: a slide joint's travel from its value at t0 (|dq|), or a body's displacement
    along direction (world; default down, optionally relative to another body). peak: the largest excursion counts
    (a button that springs back); else the value at t1 (a part that stays seated). min_strokes: separate presses."""
    if joint is None and body is None:
        raise ValueError("joint / body: give the pressed joint or body")
    na = (names or ("the part",))[0]
    if joint is not None:
        ji = joint_info(ep, joint)
        q = np.asarray(ep.Q[t0:t1 + 1, ji["adr"]], float)
        x = np.abs(q - q[0])
        src = f"joint {joint}"
    else:
        dvec = np.asarray(direction if direction is not None else [0, 0, -1.0], float)
        dvec = dvec / max(np.linalg.norm(dvec), 1e-12)
        a = int(body)
        ref = int(relative_to) if relative_to is not None else None
        xs = []
        p0r = None
        for t in range(t0, t1 + 1):
            p, _ = ep.pose(t, a)
            if ref is not None:
                pr, Rr = ep.pose(t, ref)
                rel = Rr.T @ (p - pr)
                if p0r is None:
                    p0r, R0r = rel, Rr
                xs.append(float((R0r @ (rel - p0r)) @ dvec))
            else:
                if p0r is None:
                    p0r = p
                xs.append(float((p - p0r) @ dvec))
        x = np.maximum(np.asarray(xs), 0.0)
        src = f"body {ep.name(a)} along {np.round(dvec, 3).tolist()}"
    disp = float(np.max(x)) if peak else float(x[-1])
    strokes = count_strokes(x, float(min_m))
    meas = {"source": src, "displacement_m": _r(disp, 5), "end_displacement_m": _r(float(x[-1]), 5), "required_m": _r(min_m, 5),
            "strokes": int(strokes), "required_strokes": int(min_strokes)}
    issues = []
    ok = disp >= float(min_m) - 1e-9 and strokes >= int(min_strokes)
    if disp < float(min_m) - 1e-9:
        issues.append(f"{na} moved {1000 * disp:.2f} mm in frames {t0}-{t1} (needed {1000 * float(min_m):.2f} mm{'' if peak else ' at the end'}): "
                      "press farther (more force, or press on the part itself)")
    elif strokes < int(min_strokes):
        issues.append(f"{na} was pressed through {strokes} stroke(s) in frames {t0}-{t1} (needed {int(min_strokes)}): release and press again")
    return _report(ok, issues, **meas)


# ------------------------------------------------------------------------------------------ deformables
_CABLE_RE = re.compile(r"^(?P<pre>.*)B_(?P<k>first|last|\d+)$")


def cable_bodies(ep, rope=None):
    """Segment body ids of a construct_rope cable in order (first, 1..n, last), by name prefix."""
    m = ep.model
    groups = {}
    for b in range(m.nbody):
        mt = _CABLE_RE.match(m.body(b).name)
        if mt:
            k = mt.group("k")
            order = -1 if k == "first" else (10 ** 9 if k == "last" else int(k))
            groups.setdefault(mt.group("pre"), []).append((order, b))
    if rope is not None:
        keys = [k for k in groups if k == rope or k.startswith(str(rope)) or str(rope).startswith(k)]
    else:
        keys = list(groups)
    if not keys:
        raise ValueError(f"rope: no cable (bodies '<prefix>B_first..B_last') {'matching ' + repr(rope) if rope else ''} in the scene")
    if len(keys) > 1 and rope is None:
        raise ValueError(f"rope: several cables in the scene ({keys}); name one")
    key = sorted(keys, key=len)[-1] if rope is not None else keys[0]
    return key, [b for _, b in sorted(groups[key])]


def resample_polyline(P, n):
    P = np.asarray(P, float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return np.repeat(P[:1], n, axis=0)
    u = np.linspace(0, s[-1], n)
    return np.stack([np.interp(u, s, P[:, k]) for k in range(P.shape[1])], axis=1)


def frechet(P, Q):
    """Discrete Frechet distance between two polylines."""
    P, Q = np.asarray(P, float), np.asarray(Q, float)
    n, m = len(P), len(Q)
    D = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=2)
    C = np.full((n, m), np.inf)
    C[0, 0] = D[0, 0]
    for i in range(1, n):
        C[i, 0] = max(C[i - 1, 0], D[i, 0])
    for j in range(1, m):
        C[0, j] = max(C[0, j - 1], D[0, j])
    for i in range(1, n):
        for j in range(1, m):
            C[i, j] = max(min(C[i - 1, j], C[i - 1, j - 1], C[i, j - 1]), D[i, j])
    return float(C[-1, -1])


def gate_deformable_shape(ep, bodies, t, target_points=None, target_ends=None, tol_m=0.03, ends_tol_m=0.03, ordered=False,
                          planar=False, n=40, names=None):
    """A rope's centreline at frame t against a target centreline (discrete Frechet distance) and/or target end positions."""
    if target_points is None and target_ends is None:
        raise ValueError("target_points / target_ends: give a target centreline or end positions")
    na = (names or ("the rope",))[0]
    d = ep.at(t)
    P = np.array([d.xpos[b] for b in bodies])
    issues, meas = [], {"frame": int(t), "segments": len(bodies)}
    ok = True
    if target_points is not None:
        Tg = np.asarray(target_points, float)
        if Tg.ndim != 2 or Tg.shape[0] < 2 or Tg.shape[1] not in (2, 3):
            raise ValueError("target_points: a [k, 3] (or [k, 2] for planar) polyline with k >= 2")
        planar = planar or Tg.shape[1] == 2
        PP = P[:, :2] if planar else P
        TT = Tg[:, :2] if planar else Tg
        A_ = resample_polyline(PP, n)
        B_ = resample_polyline(TT, n)
        fd = frechet(A_, B_)
        if not ordered:
            fd = min(fd, frechet(A_[::-1], B_))
        meas["frechet_m"] = _r(fd)
        if fd > float(tol_m):
            ok = False
            dists = np.linalg.norm(A_[:, None] - B_[None], axis=2).min(1)
            k = int(np.argmax(dists))
            issues.append(f"{na}'s centreline is {100 * fd:.1f} cm (Frechet) from the target at frame {t} (tolerance {100 * float(tol_m):.1f} cm); "
                          f"the largest deviation is near {np.round(A_[k], 3).tolist()}")
    if target_ends is not None:
        e = target_ends
        first, last = np.asarray(e.get("first"), float) if e.get("first") is not None else None, np.asarray(e.get("last"), float) if e.get("last") is not None else None
        ends = {"first": P[0], "last": P[-1]}

        def err(pf, pl):
            out = {}
            if first is not None:
                out["first"] = float(np.linalg.norm((pf - first)[:len(first)]))
            if last is not None:
                out["last"] = float(np.linalg.norm((pl - last)[:len(last)]))
            return out
        e1 = err(ends["first"], ends["last"])
        e2 = err(ends["last"], ends["first"])
        use = e1 if (ordered or max(e1.values()) <= max(e2.values())) else e2
        meas["end_errors_m"] = {k: _r(v) for k, v in use.items()}
        meas["ends_swapped"] = use is e2
        for k, v in use.items():
            if v > float(ends_tol_m):
                ok = False
                issues.append(f"{na}'s {k} end is {100 * v:.1f} cm from its target at frame {t} (tolerance {100 * float(ends_tol_m):.1f} cm)")
    return _report(ok, issues, **meas)


def rope_joins(ep):
    """Names of CONNECT/WELD equalities between two non-robot bodies (rope joins, cut joins)."""
    m = ep.model
    out = []
    for e in range(m.neq):
        if int(m.eq_type[e]) not in (int(mujoco.mjtEq.mjEQ_CONNECT), int(mujoco.mjtEq.mjEQ_WELD)):
            continue
        b1, b2 = ep.name(int(m.eq_obj1id[e])), ep.name(int(m.eq_obj2id[e]))
        if not (b1.startswith("robot_") or b2.startswith("robot_")) and m.eq(e).name:
            out.append(m.eq(e).name)
    return out


def gate_rope_severed(ep, t0, t1, joins=None, min_gap_m=0.003):
    """Each join equality was whole when the window started and inactive at its end (the recorded eq_active), and the
    two sides of the join are at least min_gap_m apart at the end."""
    from tool_library.common import outcome as O
    ea, names = ep.extras.get("eq_active"), ep.extras.get("eq_names")
    issues = []
    if ea is None or names is None:
        return _report(False, ["the trajectory carries no eq_active / eq_names (the per-frame equality state): run the cut through "
                               "execute_subgoal_plan with the scene_recon_json that lists the rope's cut_joins so the cut is recorded"],
                       recorded=False)
    ea = np.asarray(ea).astype(bool)
    names = [str(x) for x in names]
    if t0 > 0:
        start = ea[t0 - 1]
    elif ep.extras.get("eq_active_start") is not None:
        start = np.asarray(ep.extras["eq_active_start"]).astype(bool)
    else:
        # the scene's initial state, matched by name (the execution model drops the arm welds, so indices differ)
        start = ea[0].copy()
        for i, nm in enumerate(names):
            e = mujoco.mj_name2id(ep.model, mujoco.mjtObj.mjOBJ_EQUALITY, nm)
            if e >= 0:
                start[i] = bool(ep.model.eq_active0[e])
    if joins is None:
        cand = set(rope_joins(ep))
        want = [nm for i, nm in enumerate(names) if nm in cand and start[i] and not ea[t1, i]]
        if not want:
            issues.append(f"no rope join went from whole to cut in frames {t0}-{t1} (joins recorded: "
                          f"{[n for n in names if n in cand][:8]}): the blades must close on the join")
            return _report(False, issues, joins={}, recorded=True)
    else:
        want = [joins] if isinstance(joins, str) else [str(j) for j in joins]
    per = {}
    d = ep.at(t1)
    for nm in want:
        if nm not in names:
            per[nm] = {"recorded": False}
            issues.append(f"join {nm!r} is not a column of the trajectory's eq_names: it was not recorded")
            continue
        i = names.index(nm)
        col = ea[t0:t1 + 1, i]
        off = np.nonzero(~col)[0]
        gap = O.join_end_gap(ep.model, d, nm)
        row = {"whole_at_start": bool(start[i]), "whole_at_end": bool(ea[t1, i]),
               "cut_frame": int(t0 + off[0]) if len(off) and start[i] else None, "end_gap_m": _r(gap)}
        per[nm] = row
        if not row["whole_at_start"]:
            issues.append(f"join {nm} was already cut before frame {t0}: the cut does not happen in this window")
        elif row["whole_at_end"]:
            issues.append(f"join {nm} is still whole at frame {t1}: the rope was not cut (close the blades on the join)")
        elif gap is not None and gap < float(min_gap_m):
            issues.append(f"join {nm} is switched off but its two sides are only {1000 * gap:.1f} mm apart at frame {t1} "
                          f"(needed {1000 * float(min_gap_m):.1f} mm): the halves did not separate")
    return _report(not issues, issues, joins=per, recorded=True)


def gate_knot(ep, bodies, t, expect="trefoil", pre=None):
    """The cable's centreline at frame t is knotted (common.knot: Alexander polynomial of the closed-off curve)."""
    from tool_library.common import knot as K
    d = ep.at(t)
    P = [d.xpos[b].copy() for b in bodies]
    if pre:
        sid = mujoco.mj_name2id(ep.model, mujoco.mjtObj.mjOBJ_SITE, pre + "S_last")
        if sid >= 0:
            P.append(d.site_xpos[sid].copy())
    v = K.check(np.asarray(P, float))
    hit = bool(v["knot"]) if expect == "knot" else bool(v["agree"] and v["type"] == expect)
    issues = [] if hit else [f"the rope is '{v['type']}' at frame {t} (expected {expect}; closures agree {v['agree']}, "
                             f"{v['n_self_crossings']} self-crossings): tie it through the loop"]
    return _report(hit, issues, frame=int(t), type=v["type"], agree=bool(v["agree"]), self_crossings=int(v["n_self_crossings"]))


def strand_paths(model, bodies):
    """The strands of a cable, rope or ribbon given its bodies: link paths from where it starts to each free end. A plain
    chain is one strand; at a junction (a body with several children: the knot of a braid's strands) each child starts
    a strand of its own, the junction itself belongs to none."""
    S = set(int(b) for b in bodies)
    kids = {b: sorted(c for c in S if int(model.body_parentid[c]) == b) for b in S}
    roots = sorted(b for b in S if int(model.body_parentid[b]) not in S)
    paths = []

    def walk(b, path):
        ks = kids[b]
        if not ks:
            paths.append(path + [b]); return
        if len(ks) == 1:
            walk(ks[0], path + [b]); return
        if path:
            paths.append(path + [b])
        for c in ks:
            walk(c, [])

    for r in roots:
        walk(r, [])
    return [p_ for p_ in paths if len(p_) >= 3]


def gate_strand_crossings(ep, bodies, t, max_crossings=None, min_crossings=None, junction_m=0.005):
    """Plan-view crossings of a cable's strands with each other at frame t (strand_crossing_check's count); a single strand
    is judged by its self-crossings. An unbraided ribbon has none; a braid has several."""
    from tool_library.verifiers import strand_crossing_check as SC
    d = ep.at(t)
    paths = strand_paths(ep.model, bodies)
    if not paths:
        raise _Unresolved("no strand of three or more links among the object's bodies")
    if len(paths) == 1:
        from tool_library.common import knot as K
        n = int(K.check(np.asarray([d.xpos[b].copy() for b in paths[0]], float))["n_self_crossings"])
    else:
        smap = {f"strand{i}": [ep.model.body(b).name for b in p_] for i, p_ in enumerate(paths)}
        n = len(SC._crossings(SC._positions(ep.model, d, smap), float(junction_m)))
    issues = []
    if max_crossings is not None and n > int(max_crossings):
        issues.append(f"{n} strand crossing(s) at frame {t}, at most {int(max_crossings)} required: the strands still lie across "
                      f"each other")
    if min_crossings is not None and n < int(min_crossings):
        issues.append(f"{n} strand crossing(s) at frame {t}, at least {int(min_crossings)} required")
    return _report(not issues, issues, frame=int(t), n_crossings=n, n_strands=len(paths))


# ============================================================================================ video comparison
def load_camera(camera, video_frame=None):
    """('name', camera name) for a camera in the scene, or ('dict', camera dict) for a pinhole camera given as a dict or
    a JSON file (width, height, fx/fy or fovy_deg, pos, quat_wxyz or R_cv). A moving camera is a per-frame camera: a
    list of camera dicts indexed by video frame, a dict {'frames': [{'frame': i, ...camera...}, ...]} or such a JSON;
    the camera of `video_frame` (default the last) is used."""
    cam = camera
    if isinstance(cam, (str, Path)) and str(cam).lower().endswith(".json"):
        p = Path(str(cam))
        if not p.exists():
            raise ValueError(f"camera: {cam} does not exist")
        cam = json.loads(p.read_text())
    if isinstance(cam, str):
        return "name", cam
    frames = None
    if isinstance(cam, list):
        frames = [dict(c, frame=c.get("frame", i)) for i, c in enumerate(cam)]
    elif isinstance(cam, dict) and isinstance(cam.get("frames"), list):
        base = {k: v for k, v in cam.items() if k != "frames"}
        frames = [dict(base, **c) for c in cam["frames"]]
    if frames is not None:
        if not frames:
            raise ValueError("camera: the per-frame camera list is empty")
        if video_frame is None:
            return "dict", max(frames, key=lambda c: int(c.get("frame", 0)))
        best = min(frames, key=lambda c: abs(int(c.get("frame", 0)) - int(video_frame)))
        return "dict", best
    if isinstance(cam, dict):
        if "pos" not in cam:
            raise ValueError("camera: a camera dict needs pos and quat_wxyz (or R_cv), and fovy_deg (or fy and height)")
        return "dict", cam
    raise ValueError("camera: a camera name, a camera dict / JSON, or a per-frame camera list")


def _cam_quat_fovy(cam):
    if "quat_wxyz" in cam:
        q = [float(v) for v in cam["quat_wxyz"]]
    elif "R_cv" in cam:
        R = np.asarray(cam["R_cv"], float).reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
        x, y, z, w = Rot.from_matrix(R).as_quat()
        q = [w, x, y, z]
    else:
        raise ValueError("camera: needs quat_wxyz or R_cv")
    if "fovy_deg" in cam:
        fovy = float(cam["fovy_deg"])
    elif "fy" in cam and "height" in cam:
        fovy = float(np.degrees(2 * np.arctan(0.5 * float(cam["height"]) / float(cam["fy"]))))
    else:
        raise ValueError("camera: needs fovy_deg or fy and height")
    return q, fovy


def render_silhouettes(scene_path, qpos, camera, width, height):
    """{body name: bool HxW} segmentation of the scene at qpos through a named camera or a camera dict."""
    kind, cam = camera
    spec = mujoco.MjSpec.from_file(str(scene_path))
    cname = cam if kind == "name" else "__gate_camera"
    if kind == "dict":
        q, fovy = _cam_quat_fovy(cam)
        c = spec.worldbody.add_camera()
        c.name = cname
        c.pos = [float(v) for v in cam["pos"]]
        c.quat = q
        c.fovy = fovy
    spec.visual.global_.offwidth = max(int(spec.visual.global_.offwidth), int(width))
    spec.visual.global_.offheight = max(int(spec.visual.global_.offheight), int(height))
    model = spec.compile()
    if model.nq != len(qpos):
        raise ValueError(f"scene: nq {model.nq} does not match the trajectory ({len(qpos)})")
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cname) < 0:
        raise ValueError(f"camera: no camera {cname!r} in the scene")
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    from tool_library.common.render import segmentation_renderer
    with segmentation_renderer(model, int(height), int(width)) as r:
        r.update_scene(data, camera=cname)
        seg = r.render().copy()
    gid = seg[..., 0].astype(int)
    typ = seg[..., 1].astype(int)
    out = {}
    for g in np.unique(gid):
        if g < 0 or g >= model.ngeom:
            continue
        sel = (gid == g) & (typ == int(mujoco.mjtObj.mjOBJ_GEOM))
        if not sel.any():
            continue
        nm = model.body(int(model.geom_bodyid[g])).name
        out[nm] = out.get(nm, np.zeros_like(sel)) | sel
    return out, model


def load_mask(mask, size=None):
    from tool_library.common import scene as S
    return S.load_mask(mask, size)


# ============================================================================================ manifest helpers
def no_robot_action_allowed(subgoal):
    """Only the independent Stage 1 input can declare an observation-only step.

    Waiting for an object to settle or holding it is an outcome to verify, not an
    observation that a trajectory generator may silently discard.
    """
    motion = str(subgoal.get("motion", "")).strip().lower()
    passive = {"no_robot_action", "observe", "look", "wait", "idle"}
    goal = str(subgoal.get("goal", "")).lower()
    manipulation = r"\b(grasp\w*|hold\w*|lift\w*|mov\w*|plac\w*|push\w*|pull\w*|releas\w*|sett\w*|insert\w*|open\w*|clos\w*|touch\w*|turn\w*|rotat\w*)\b"
    return (motion in passive and not subgoal.get("object_refs")
            and not subgoal.get("goal_predicates") and not re.search(manipulation, goal))


def stage1_indices(sg):
    idx = sg.get("stage1_indices")
    if idx:
        return [int(x) for x in idx]
    return [int(sg["index"])] if "index" in sg else []


def expand_entries(subs):
    """Manifest entries with simultaneous two-hand entries {motion: 'both', hands: {left: {...}, right: {...}}, sync} split
    into one entry per hand: each keeps its own object, motion and expect, takes the parent's frames, index, stage1_indices
    and skip state, and is marked as part of a synchronous pair. The parent's goal_predicates / deviation travel with the
    first hand's entry."""
    out = []
    for sg in subs or []:
        if not isinstance(sg, dict):
            continue
        hands = sg.get("hands")
        if str(sg.get("motion", "")).lower() == "both" and isinstance(hands, dict) and hands:
            first = True
            for side in ("left", "right"):
                sub = hands.get(side)
                if not isinstance(sub, dict):
                    continue
                e = dict(sub)
                e["side"] = side
                for k in ("index", "frames", "stage1_indices", "skipped", "skip_kind", "reason"):
                    if k in sg and (k not in e or e.get(k) is None):
                        e[k] = sg[k]
                if first:
                    for k in ("goal_predicates", "deviation"):
                        if k in sg and k not in e:
                            e[k] = sg[k]
                e["sync"] = sg.get("sync", "start")
                e["both_entry"] = True
                out.append(e)
                first = False
        else:
            out.append(sg)
    return out


def side_phase(label, side=None):
    """The phase text of a frame label for a side: 'left: hold box | right: grip lid' -> 'hold box' for 'left'."""
    for part in str(label).split(" | "):
        sd, _, ph = part.partition(": ")
        if side is None or sd.strip() == side:
            return ph if ph else part
    return ""


def entry_frames(ep, sg):
    fr = sg.get("frames")
    if isinstance(fr, (list, tuple)) and len(fr) == 2 and fr[0] is not None:
        return max(0, int(fr[0])), min(ep.T - 1, int(fr[1]))
    sub = ep.extras.get("subgoal")
    if sub is not None and "index" in sg:
        idx = np.nonzero(np.asarray(sub).astype(int) == int(sg["index"]))[0]
        if len(idx):
            return int(idx[0]), int(idx[-1])
    return None


def load_json(path, param):
    p = Path(str(path))
    if not p.exists():
        raise ValueError(f"{param}: {path} does not exist")
    return json.loads(p.read_text())


def recon_objects(recon):
    """{stage1 object id: {root, bodies, parts, joints, equalities, params}} from a Stage 2 scene_recon.json. `parts` are
    the separate pieces of a group object (piece_groups; [] for a single-body object), and are among its bodies."""
    from tool_library.common.piece_groups import recon_parts
    out = {}
    for o in (recon or {}).get("objects", []) or []:
        if not isinstance(o, dict) or "id" not in o:
            continue
        bodies = [str(b) for b in (o.get("bodies") if isinstance(o.get("bodies"), list) else [])]
        parts = recon_parts(o)
        out[str(o["id"])] = {"root": o.get("root_body"), "bodies": bodies + [b for b in parts if b not in bodies], "parts": parts,
                             "joints": o.get("joints"), "equalities": o.get("equalities"),
                             "params": o.get("params") if isinstance(o.get("params"), dict) else {}, "type": o.get("type")}
    return out


def gate_bimanual_sync(ep, sem, man, recon=None, min_overlap_s=0.2, stride=2, tol=HOLD_TOL_M):
    """For every Stage 1 subgoal with acting_hand 'both': each hand is active (a manifest entry of that side that is not
    a 'home' move, or touching one of the subgoal's objects) and the two activity periods overlap for at least
    min_overlap_s, unless an entry of that subgoal declares a sequential deviation {reason, sequential: true}."""
    objs = recon_objects(recon)
    rows, issues = [], []
    hands = ep.hands()
    for s1 in sem.get("subgoals", []):
        if str(s1.get("acting_hand", "")).lower() != "both":
            continue
        k = int(s1["index"])
        ents = [sg for sg in expand_entries(man.get("subgoals", [])) if k in stage1_indices(sg)]
        ex = [sg for sg in ents if not sg.get("skipped")]
        row = {"index": k, "goal": s1.get("goal"), "entries": len(ex)}
        if not ex:
            kinds = {sg.get("skip_kind") for sg in ents}
            row.update(ok=bool(ents) and kinds <= {"no_robot_action"} and no_robot_action_allowed(s1), note="no executed entry")
            if not row["ok"]:
                issues.append(f"Stage 1 subgoal {k} (both hands) has no executed manifest entry")
            rows.append(row)
            continue
        dev = next((sg.get("deviation") for sg in ex if isinstance(sg.get("deviation"), dict) and
                    (sg["deviation"].get("sequential") or str(sg["deviation"].get("kind", "")).lower() == "sequential")), None)
        wins = [w for w in (entry_frames(ep, sg) for sg in ex) if w]
        if not wins:
            row.update(ok=False, note="no frames")
            issues.append(f"Stage 1 subgoal {k}: its entries have no frames in the trajectory")
            rows.append(row)
            continue
        t0, t1 = min(w[0] for w in wins), max(w[1] for w in wins)
        B = set()
        for ref in s1.get("object_refs", []) or []:
            oid = str(ref).split(":")[0]
            o = objs.get(oid)
            for nm in ([o["root"]] if o and o.get("root") else []) + (o["bodies"] if o else []):
                try:
                    B |= ep.subtree(ep.body(nm))
                except ValueError:
                    pass
            if not o:
                try:
                    B |= ep.subtree(ep.body(oid))
                except ValueError:
                    pass
        B -= ep.robot
        act = {s: np.zeros(t1 - t0 + 1, bool) for s in SIDES}
        touched = {s: [] for s in SIDES}
        if B:
            for s in SIDES:
                if s not in hands:
                    continue
                with Inflate(ep, B | hands[s], tol):
                    for t in frames_between(t0, t1, stride):
                        if ep.contacts(t, hands[s], B, tol):
                            lo, hi = t - t0, min(t1 - t0, t - t0 + stride - 1)
                            act[s][lo:hi + 1] = True
                            touched[s].append(t)
        # Labels and a motion='both' declaration are proposals, not evidence of
        # participation. Each hand must actually contact a Stage 1 task object.
        both = act["left"] & act["right"]
        ov = float(both.sum()) / ep.fps
        spans = {}
        for s in SIDES:
            nz = np.nonzero(act[s])[0]
            spans[s] = [int(t0 + nz[0]), int(t0 + nz[-1])] if len(nz) else None
        row.update(window=[t0, t1], active=spans, overlap_s=_r(ov, 3), sequential_deviation=dev)
        ok = ov >= float(min_overlap_s)
        if not ok:
            idle = [s for s in SIDES if spans[s] is None]
            if idle:
                issues.append(f"Stage 1 subgoal {k} ('{s1.get('goal', '')[:80]}') is a two-hand step but only the "
                              f"{[s for s in SIDES if s not in idle][0] if len(idle) == 1 else 'no'} hand acts in frames {t0}-{t1}: give the "
                              f"{' and '.join(idle)} hand its part; a sequential deviation cannot satisfy a simultaneous goal")
            else:
                issues.append(f"Stage 1 subgoal {k} ('{s1.get('goal', '')[:80]}') is a two-hand step but the hands act one after the other "
                              f"(left {spans['left']}, right {spans['right']}, overlap {ov:.2f} s < {min_overlap_s} s): make them act "
                              "together; a sequential deviation cannot satisfy a simultaneous goal")
        row["ok"] = bool(ok)
        rows.append(row)
    return _report(not issues, issues, subgoals=rows)


# ============================================================================================ Stage 1 goal predicates
CLASSES = ("contact", "apart", "stacked_on", "inside_container", "target_region", "relative_pose", "joint_target", "held",
           "lift_clearance", "release_settle", "insertion_depth", "rotation_turns", "tilt_pour", "press_displacement",
           "deformable_shape", "rope_severed", "knot", "strand_crossings")

_STOP = set(("the a an of its their his her it they them this that these those to and or nor with on in at by from into onto "
             "inside over under is are was were be been has have had very still just also both each all every one two three "
             "four five held hold holds holding is s other another whole far near").split())
_PRONOUN = {"it", "its", "they", "them", "their", "both", "each", "all", "every", "either", "these", "those"}
TABLE_WORDS = {"table", "tabletop", "cloth", "tablecloth", "mat", "surface", "desk", "counter", "bench", "worktop"}
HAND_WORDS = {"hand", "hands", "finger", "fingers", "fist", "fingertips", "thumb", "palm", "grip", "grasp"}
CONTAINER_NOUNS = {"bowl", "cup", "mug", "tray", "drawer", "box", "bin", "basket", "bag", "pot", "pan", "jar", "container",
                   "glass", "bucket", "sink", "dish", "interior", "pocket", "tub"}
SLOT_NOUNS = {"slot", "port", "socket", "well", "bay", "cage", "stack", "hole", "holder", "sleeve", "receptacle", "rack", "case", "jack"}
MOVING_PART_WORDS = {"tray", "drawer", "lid", "door", "flap", "latch", "cap", "button", "head", "plunger", "pump", "handle",
                     "knob", "trigger", "blade", "slider"}
_VERB = re.compile(r"\b(is|are|was|were|has|have|had|sits?|rests?|stands?|lies|lie|lay|hangs?|touch(?:es)?|stays?|remains?|"
                   r"ends?|holds?|comes?|goes|runs?|forms?|leaves|displays?|points?|becomes?|leans?|can|will|moves?|"
                   r"passe?s?|encircles?|transfers?|rotated|slid|turned)\b")
_CUT = re.compile(r",|;|\bwith\b|\bwhile\b|\bso\b|\bwithout\b|\bwhen\b|\buntil\b|\bafter\b|\bbefore\b|\bbecause\b|\bcovering\b|"
                  r"\bbridging\b|\bleaving\b|\bthat\b|\bwhich\b|\bwhere\b|\bbetween\b|\balong\b|\bacross\b|\babove\b|\bbelow\b|"
                  r"\bbehind\b|\btoward\b|\btowards\b|\bnear\b|\bnext to\b|\bto the\b|\bin front\b|\bdirectly\b|\bat\b|\bon\b|"
                  r"\binto\b|\bonto\b|\binside\b|\bfrom\b|\bby\b|\bunder\b|\bover\b|\bagainst\b|\bthrough\b|\bhooked\b|\band (?:the|a|an)\b"
                  r"|\bstays?\b|\bremains?\b|\bin\b|\bas\b|\bfor\b|\bpointing\b|\beach\b|\bfully\b|\bwhole\b")


def _oid(ref):
    """The Stage 1 object id of a ref ('obj', 'obj:part' or 'obj:part#body')."""
    return str(ref).split("#")[0].split(":")[0]


def _norm(w):
    w = w.lower()
    if len(w) > 3 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _toks(s):
    return [_norm(w) for w in re.findall(r"[a-z0-9]+", str(s).lower())]


def _content(s):
    return [w for w in _toks(s) if w not in _STOP]


def _phrase_after(text, end):
    rest = text[end:]
    m = _CUT.search(rest)
    return rest[:m.start()] if m and m.start() > 0 else (rest if not m else rest[:m.start()])


class GoalContext:
    """Stage 1 objects, Stage 2 bodies and the episode, for resolving the noun phrases of goal sentences."""

    def __init__(self, ep, sem, recon):
        self.ep = ep
        self.sem = sem
        self.objs = {str(o["id"]): o for o in sem.get("objects", []) if isinstance(o, dict) and "id" in o}
        self.recon = recon_objects(recon)

    def obj_tokens(self, oid):
        o = self.objs.get(oid, {})
        return set(_toks(oid)) | set(_toks(o.get("name", ""))) - _STOP

    def candidates(self, refs):
        out = []
        for r in refs:
            r = str(r).split("#")[0]
            oid, _, part = r.partition(":")
            out.append({"ref": r, "oid": oid, "part": part, "is_ref": True, "id_tokens": set(_toks(oid)) - _STOP,
                        "tokens": self.obj_tokens(oid) | (set(_toks(part)) - _STOP), "part_tokens": set(_toks(part)) - _STOP})
        seen = {c["oid"] for c in out}
        have = {c["ref"] for c in out}
        for oid in self.objs:
            if oid not in seen:
                out.append({"ref": oid, "oid": oid, "part": "", "is_ref": False, "tokens": self.obj_tokens(oid), "part_tokens": set(),
                            "id_tokens": set(_toks(oid)) - _STOP})
            for part in self.objs[oid].get("parts") or []:
                ref = f"{oid}:{part}"
                if ref in have or not isinstance(part, str):
                    continue
                pt = set(_toks(part)) - _STOP
                out.append({"ref": ref, "oid": oid, "part": part, "is_ref": False, "tokens": self.obj_tokens(oid) | pt,
                            "part_tokens": pt, "id_tokens": set(_toks(oid)) - _STOP})
        return out

    def match(self, phrase, refs, exclude=()):
        """The candidates a noun phrase names (several for 'X and Y'); [] when none. exclude: refs (or object ids) to skip."""
        parts = [p for p in re.split(r",|\band\b|\bor\b|\bnor\b|\bboth\b", phrase) if p.strip()]
        found = []
        ex = {str(e).split("#")[0] for e in exclude}
        cands = [c for c in self.candidates(refs) if c["ref"] not in ex and not (c["oid"] in ex and not c["part"])]
        for p in parts or [phrase]:
            ws = set(_content(p))
            if not ws:
                continue
            best, top = 0, []
            for c in cands:
                s = len(ws & c["tokens"])
                if s > best:
                    best, top = s, [c]
                elif s == best and s > 0:
                    top.append(c)
            if best == 0:
                continue
            if len(top) > 1:
                refs_top = [c for c in top if c["is_ref"]]
                top = refs_top or top
            if len(top) > 1:
                spec = [c for c in top if c["part_tokens"] & ws]
                top = spec or top
            if len({c["oid"] for c in top}) > 1:
                # 'the battery' names the battery, not the battery holder: an object whose id is wholly in the phrase,
                # then the candidate with the fewest unmatched words
                whole = [c for c in top if c["id_tokens"] and c["id_tokens"] <= ws]
                top = whole or top
                extra = {c["ref"]: (len(c["part_tokens"] - ws), len(c["tokens"] - ws - c["part_tokens"])) for c in top}
                lo_ = min(extra.values())
                top = [c for c in top if extra[c["ref"]] == lo_]
            if len({c["oid"] for c in top}) == 1:
                found.append(top[0])
        out, seen = [], set()
        for c in found:
            if c["ref"] not in seen:
                seen.add(c["ref"])
                out.append(c)
        return out

    def is_table(self, phrase):
        return bool(set(_toks(phrase)) & TABLE_WORDS)

    # ------------------------------------------------------------------ bodies
    def object_body_names(self, oid):
        r = self.recon.get(oid)
        names = []
        if r:
            names = ([r["root"]] if r.get("root") else []) + [b for b in r["bodies"] if b != r.get("root")]
        return names

    def body_of(self, ref, acted=()):
        """Resolve a source identity without letting the manifest select its part.

        Partial name overlap and the acted body are not evidence that a named
        drawer/latch is the requested one. An ambiguous part stays unresolved.
        """
        ep = self.ep
        ref, _, bound = str(ref).partition("#")
        oid, _, part = str(ref).partition(":")
        names = self.object_body_names(oid)
        ids = []
        for n in names:
            try:
                ids.append(ep.body(n))
            except ValueError:
                pass
        if not ids:
            try:
                ids = [ep.body(oid)]
            except ValueError:
                return None
        root = ids[0]
        objset = set()
        for b in ids:
            objset |= ep.subtree(b)
        pieces = self.group_pieces(oid)
        if pieces:
            # a group object (separate loose pieces): 'a block piece' is any of them, so the piece this subgoal's
            # entries actually act on stands for it; a Stage 1 part of a group ('top_course') names a region of the
            # pieces, not a body, and resolves to the group (its root here, every piece in _resolve_set)
            for a in acted or ():
                hit = next((pb for pb in pieces if int(a) in ep.subtree(pb)), None)
                if hit is not None:
                    return hit
            return root
        if not part:
            return root
        pt = set(_toks(part)) - _STOP - set(_toks(oid))
        fixed_part = (self.objs.get(oid, {}).get("articulation") or {}).get("fixed_part")
        if fixed_part == part:
            return root
        # Library constructors use a terminal B for body names (lidB), not a
        # semantic modifier. Match that exact suffix; 'lid' must not match
        # lidFlapB, and 'left drawer' must not match an unnamed drawer.
        hits = [b for b in sorted(objset) if pt and all(
            word in set(_toks(ep.name(b))) or word + "b" in set(_toks(ep.name(b))) for word in pt)]
        if not hits and len(objset) and not (self.objs.get(oid, {}).get("articulation") or self.objs.get(oid, {}).get("type") == "articulated"):
            # a named region of a rigid object (a bowl's rim, a cup's handle) is that object's own body
            return root
        return hits[0] if len(hits) == 1 else None

    def group_pieces(self, oid):
        """Body ids of a group object's separate pieces (scene_recon `parts`); [] for a single-body object."""
        out = []
        for n in (self.recon.get(oid) or {}).get("parts") or []:
            try:
                out.append(self.ep.body(n))
            except ValueError:
                pass
        return out if len(out) > 1 else []

    def bind(self, refs, acted):
        """Keep source references; generated acted-body hints cannot bind them."""
        return [str(r).split("#")[0] for r in refs]

    def object_root(self, oid):
        for n in self.object_body_names(oid):
            try:
                return self.ep.body(n)
            except ValueError:
                pass
        try:
            return self.ep.body(oid)
        except ValueError:
            return None

    def joints_of(self, bid, oid=None):
        """Hinge/slide joints on the body or its ancestors inside its object; else every hinge/slide of the object."""
        ep = self.ep
        m = ep.model
        root = self.object_root(oid) if oid else None
        chain, b = [], int(bid)
        while b > 0:
            chain.append(b)
            if root is not None and b == root:
                break
            b = int(m.body_parentid[b])
        for b in chain:
            js = [j for j in range(m.njnt) if int(m.jnt_bodyid[j]) == b and int(m.jnt_type[j]) in (HINGE, SLIDE)]
            if js:
                return [m.joint(j).name for j in js]
        objset = ep.subtree(root) if root is not None else ep.subtree(bid)
        return [m.joint(j).name for j in range(m.njnt) if int(m.jnt_bodyid[j]) in objset and int(m.jnt_type[j]) in (HINGE, SLIDE)]


def _amount(text, default=None):
    t = text.lower()
    approx = bool(re.search(r"\b(about|roughly|around|approximately|nearly|almost|some)\b", t))
    frac = 0.75 if approx else 0.9
    if "quarter turn" in t:
        return 90.0, frac
    if "half turn" in t or "half a turn" in t:
        return 180.0, frac
    if "full turn" in t or "whole turn" in t:
        return 360.0, frac
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:deg|degrees|°)", t)
    if m:
        return float(m.group(1)), frac
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:full )?turns", t)
    if m:
        return 360.0 * float(m.group(1)), frac
    return default, frac


def _strokes(text):
    t = text.lower()
    m = re.search(r"(\d+)\s+times", t)
    if m:
        return int(m.group(1))
    if re.search(r"\b(twice|several times|repeatedly|a few times|multiple times|again and again)\b", t):
        return 2
    return 1


_MOTION = {"grasp_lift": ("held_end", "lift"), "pick": ("held_end", "lift"), "regrasp": ("held_end",), "carry": ("held_through",),
           "hold": ("held_fixed",), "place": ("release",), "release": ("release",), "insert": ("insertion",),
           "open_joint": ("joint_open",), "close_joint": ("joint_closed",), "unscrew": ("rotation",), "screw": ("rotation",),
           "pour": ("pour",), "tilt": ("tilt",), "cut": ("severed",), "tie": ("knot",)}


def _split_clauses(goal):
    g = re.sub(r"^\s*intended outcome:\s*", "", str(goal), flags=re.I)
    out = []
    for sent in re.split(r"[;:]|\.\s+|\.$", g):
        sent = sent.strip()
        if not sent:
            continue
        for c in re.split(r",\s*and\s+(?=[a-z])|\s+and\s+(?=(?:the|a|an|its|all|both|neither|no|each|every)\b)", sent):
            c = c.strip(" ,")
            if c:
                pieces = re.split(r"\s+with\s+(?=(?:the|its|their|a|an|all|both|one|two|three|four)\b)", c)
                merged = [pieces[0]]
                for w in pieces[1:]:
                    if re.search(r"\b(flush|level|aligned|parallel|contact|together|filled|covered|line|even|coaxial)\s*$", merged[-1]):
                        merged[-1] = merged[-1] + " with " + w
                    else:
                        merged.append(w)
                for k, w in enumerate(merged):
                    out.append((w.strip(" ,"), k > 0))
    return out


def infer_goal_predicates(gc, s1, entries):
    """Predicates a Stage 1 subgoal's goal sentence, motion and object refs require. Returns (predicates, notes)."""
    ep = gc.ep
    refs = [str(r) for r in (s1.get("object_refs") or [])]
    goal = str(s1.get("goal", ""))
    motion = str(s1.get("motion", "")).lower()
    acted = []
    for sg in entries:
        if sg.get("object"):
            try:
                acted.append(ep.body(sg["object"]))
            except ValueError:
                pass
    acted_oids = []
    for b in acted:
        best = None
        for c in gc.candidates(refs):
            if not c["is_ref"] or b not in _objset(gc, c["oid"]):
                continue
            rb = gc.body_of(c["ref"], [b])
            root = gc.object_root(c["oid"])
            # rank: the ref resolves to the acted body by its own part name or root, then any ref of that object
            rank = 0 if (rb == b and (root == b or (c["part"] and rb != root))) else (1 if rb == b else 2)
            if best is None or rank < best[0]:
                best = (rank, c["ref"])
        if best:
            ref_ = best[1]
            root = gc.object_root(_oid(ref_))
            body = b if (":" in ref_ or root is None) else root      # 'spray_bottle' is the bottle even when the head was acted on
            if gc.group_pieces(_oid(ref_)):
                body = gc.body_of(ref_, [b]) or body                  # a group's piece is the one acted on, not its root
            tag = f"{ref_}#{ep.name(body)}"
            if tag not in acted_oids:
                acted_oids.append(tag)
    preds, notes = [], []

    def subj_from(phrase, prev):
        ws = set(_toks(phrase))
        if ws & HAND_WORDS and not gc.match(phrase, refs):
            return "HAND"
        m = gc.match(phrase, refs)
        if m:
            return [c["ref"] for c in m]
        if ws & {"two", "three", "four", "both", "all", "each", "held"} and acted_oids:
            # 'the two held bricks': the objects the manifest acted on, when the remaining words describe them
            rest = set(_content(phrase))
            known = set()
            for r_ in acted_oids:
                known |= gc.obj_tokens(_oid(r_))
            if rest <= known:
                return list(acted_oids)
        if ws & _PRONOUN or not (set(_content(phrase))):
            return prev or (acted_oids or refs[:1])
        # 'water flows into the mug', 'the tower is three bricks tall': a subject that names no object gives no predicate
        # (inheriting the previous subject would put the bottle into the mug)
        return None

    def add(cls, A, **kw):
        preds.append(dict({"class": cls, "A": A}, **kw))

    prev = None
    for clause, is_with in _split_clauses(goal):
        low = re.split(r"\bso that\b|\bso (?=the|it|they|a|an)\b|\bin order to\b|\bto allow\b|\bcan be\b", clause.lower())[0]
        mv = _VERB.search(low) or re.search(r"\b(flush|against|onto|on|into|inside|in|touching|clear of|free of|still|separated|"
                                            r"away|above|below|under|beside|pointing|facing|resting|lying|standing|hooked)\b", low)
        subj_phrase = low[:mv.start()] if mv else low
        if not subj_phrase.strip():
            subj = prev or (acted_oids or refs[:1])
        else:
            subj = subj_from(subj_phrase, prev)
            if is_with and subj is None:
                subj = prev
        pred_text = low[mv.start():] if mv else ""
        if subj is None:
            notes.append(f"clause '{clause}': the subject names no object of the scene; no predicate from it")
            continue
        hand_subj = subj == "HAND"
        subjects = [] if hand_subj else gc.bind(list(subj), acted)
        if not hand_subj and not is_with:            # 'X ..., with its flap flush ..., and stays closed': X stays the subject
            prev = subjects
        negated = lambda pos: bool(re.search(r"\b(no longer|not|never|neither|nor|without|none)\b", low[max(0, pos - 40):pos]))
        # ---- hands as the subject: 'neither hand touches X', 'the left hand is in contact with X'
        if hand_subj:
            for mt in re.finditer(r"\b(touch(?:es|ing)?|in contact with|holds?|holding)\b", low):
                tgt = gc.match(_phrase_after(low, mt.end()), refs)
                if not tgt:
                    tgt = [{"ref": r} for r in acted_oids]
                if mt.group(1).startswith("hold"):
                    for c in tgt:
                        add("held", c["ref"], mode="in_hand", source=f"text: '{mt.group(0)}'")
                elif negated(mt.start()) or re.search(r"\b(neither|no)\b", subj_phrase):
                    for c in tgt:
                        add("release_settle", c["ref"], source=f"text: '{subj_phrase.strip()} {mt.group(0)}'")
                else:
                    for c in tgt:
                        add("held", c["ref"], mode="fixed", source=f"text: hand '{mt.group(0)}'")
            if re.search(r"\bclear of\b", low):
                for r_ in acted_oids:
                    add("lift_clearance", r_, support="table", source="text: 'clear of'")
            continue
        # ---- relations to a named body
        for mt in re.finditer(r"\bon (?:the )?(?:inner|inside) (?:bottom|floor|base|surface) of\b", pred_text):
            for c in gc.match(_phrase_after(pred_text, mt.end()), refs, exclude={_oid(s) for s in subjects}):
                for s in subjects:
                    add("inside_container", s, B=c["ref"], source=f"text: '{mt.group(0)}'")
        inner_spans = [m.span() for m in re.finditer(r"\bon (?:the )?(?:inner|inside) (?:bottom|floor|base|surface) of\b", pred_text)]
        for mt in re.finditer(r"\b(touch(?:es|ing|ed)?|in contact with|against|beside|abut(?:s|ting)?)\b", pred_text):
            ph = _phrase_after(pred_text, mt.end())
            sub_oids = {_oid(s) for s in subjects}
            tgt = gc.match(ph, refs, exclude=sub_oids)
            if not tgt and re.search(r"\b(each other|one another)\b", ph) and len(subjects) >= 2:
                cls = "apart" if negated(pred_text.find(mt.group(0)) + len(low) - len(pred_text)) else "contact"
                add(cls, subjects[0], B=subjects[1], source=f"text: '{mt.group(0)} each other'")
                continue
            if re.search(r"\bboth\b", ph) and not tgt and len(refs) >= 3:
                tgt = [c for c in gc.candidates(refs) if c["is_ref"] and c["oid"] not in sub_oids]
            neg = negated(len(low) - len(pred_text) + mt.start())
            thr = bool(re.search(r"\b(stay(?:s|ing)?|keeps?|keeping|remains?|remaining) in contact\b|\bthroughout\b|\bthrough all\b|"
                                 r"\bnever (?:loses|leaves)\b", low))
            Bs = [c["ref"] for c in tgt] or (["table"] if gc.is_table(ph) else [])
            if not Bs and not set(_content(ph)):
                named = [c for c in gc.match(pred_text[:mt.start()], refs, exclude=sub_oids) if c["is_ref"]]
                if len({c["oid"] for c in named}) == 1:
                    Bs = [named[0]["ref"]]
            for b_ in Bs:
                for s in subjects:
                    add("apart" if neg else "contact", s, B=b_, throughout=thr and not neg, source=f"text: '{mt.group(0)}'")
        for mt in re.finditer(r"\b(separated from|free of|away from|apart from|detached from|off of)\b", pred_text):
            sub_oids = {_oid(s) for s in subjects}
            for c in gc.match(_phrase_after(pred_text, mt.end()), refs, exclude=sub_oids):
                for s in subjects:
                    add("apart", s, B=c["ref"], source=f"text: '{mt.group(0)}'")
        for mt in re.finditer(r"\bclear of\b", pred_text):
            ph = _phrase_after(pred_text, mt.end())
            if gc.is_table(ph):
                for s in subjects:
                    add("lift_clearance", s, support="table", source="text: 'clear of the table'")
            else:
                sub_oids = {_oid(s) for s in subjects}
                for c in gc.match(ph, refs, exclude=sub_oids):
                    for s in subjects:
                        add("lift_clearance", s, support=c["ref"], source=f"text: 'clear of {ph.strip()}'")
        on_support = None
        for mt in re.finditer(r"\b(on top of|onto|on)\b", pred_text):
            if any(a <= mt.start() < b for a, b in inner_spans):
                continue
            ph = _phrase_after(pred_text, mt.end())
            if gc.is_table(ph):
                on_support = "table"
                continue
            # 'on its left side', 'on the right': a position, not a support; 'sits loose on': a description of a loose part
            posw = {"left", "right", "side", "front", "back", "end", "edge", "near", "far", "top", "bottom", "middle", "centre", "center"}
            if set(_content(ph)) <= posw or re.search(r"\bloose(ly)?\s*$", pred_text[:mt.start()]):
                continue
            sub_oids = {_oid(s) for s in subjects}
            tgt = gc.match(ph, refs, exclude=sub_oids)
            if tgt:
                for s in subjects:
                    add("stacked_on", s, B=[c["ref"] for c in tgt], source=f"text: '{mt.group(0)} {ph.strip()}'")
        for mt in re.finditer(r"\b(into|inside|in)\b", pred_text):
            ph = _phrase_after(pred_text, mt.end())
            if re.match(r"\s*(front|order|place|reach|the air|the middle|half|two|line|turn)\b", ph):
                continue
            nouns = set(_toks(ph))
            sub_oids = {_oid(s) for s in subjects}
            if nouns & HAND_WORDS:
                continue
            tgt = gc.match(ph, refs, exclude=sub_oids)
            slot = bool(nouns & SLOT_NOUNS)
            cont = bool(nouns & CONTAINER_NOUNS) or mt.group(1) in ("inside", "into")
            if not tgt and (slot or cont):
                others = [c for c in gc.candidates(refs) if c["is_ref"] and c["oid"] not in sub_oids]
                if len({c["oid"] for c in others}) == 1:
                    tgt = others[:1]
            if not tgt or not (slot or cont):
                continue
            for c in tgt:
                for s in subjects:
                    add("insertion_depth" if slot else "inside_container", s, B=c["ref"], source=f"text: '{mt.group(1)} {ph.strip()}'")
        for mt in re.finditer(r"\bflush with\b", pred_text):
            ph = _phrase_after(pred_text, mt.end())
            sub_oids = {_oid(s) for s in subjects}
            tgt = gc.match(ph, refs, exclude=sub_oids)
            for s in subjects:
                b = gc.body_of(s, acted)
                if motion in ("close_joint", "open_joint") or (b is not None and gc.joints_of(b, _oid(s)) and
                                                              gc.objs.get(_oid(s), {}).get("type") == "articulated"):
                    add("joint_target", s, state="closed", source="text: 'flush with'")
                else:
                    for c in tgt:
                        add("insertion_depth", s, B=c["ref"], flush=True, source=f"text: 'flush with {ph.strip()}'")
        # 'seated' / 'seated flush' naming no target ('Drive 2 is seated flush'): home in the other object the subgoal refers to
        if re.search(r"\b(seated|seats|fully home)\b", pred_text) and subjects and \
                not any(q["class"] == "insertion_depth" and q.get("A") in subjects for q in preds):
            sub_oids = {_oid(s) for s in subjects}
            others = [c for c in gc.candidates(refs) if c["is_ref"] and c["oid"] not in sub_oids]
            if len({c["oid"] for c in others}) == 1:
                fl = {"flush": True} if re.search(r"\bflush\b", pred_text) else {}
                for s in subjects:
                    add("insertion_depth", s, B=others[0]["ref"], **fl, source=f"text: 'seated' in the subgoal's {others[0]['ref']}")
        # ---- holding, release, orientation
        for mt in re.finditer(r"\b(held|holds|holding|carried|carries|carrying|grasped|gripped)\b", pred_text):
            after = pred_text[mt.end():mt.end() + 40]
            if re.match(r"\s*(fixed|steady|still|down|in place|firmly)", after):
                for s in subjects:
                    add("held", s, mode="fixed", source=f"text: '{mt.group(0)}{after.split(',')[0][:12]}'")
                continue
            if re.match(r"\s*between the", after) and not (set(_toks(after)) & HAND_WORDS):
                continue
            for s in subjects:
                add("held", s, mode="in_hand", source=f"text: '{mt.group(0)}'")
        rel = re.search(r"\bwithout (?:any |a |the )?(?:hand|support|finger)s?\b|\blet go\b|\blets go\b|\breleased\b|\bno hand on\b|"
                        r"\bstays? where\b|\bremains? where\b|\b(?:when|after) (?:the |both )?hands? (?:let go|lift|release|open|leave|withdraw)|\b(?:rests?|resting)\b|"
                        r"\bneither hand (?:touches|supports)|\bno hand contact\b|"
                        r"\bstays? (?:seated|put|in place|there)\b", pred_text)
        rest_on_table = on_support == "table" and re.search(r"\b(stands?|standing|lies|lying|sits?|sitting|rests?|resting)\b", pred_text)
        orient = "upside_down" if re.search(r"\bupside[ -]down\b|\binverted\b|\bmouth[ -]down\b", pred_text) else ("upright" if re.search(r"\bupright\b|\bstands? on\b", pred_text) else None)
        if orient and subjects:
            # the gate compares with the object's orientation when the episode starts: a cup that starts upside down and
            # must end upside down keeps its starting up axis
            init = str(gc.objs.get(_oid(subjects[0]), {}).get("initial_state", "")).lower()
            started_flipped = bool(re.search(r"\bupside[ -]down\b|\binverted\b|\bmouth[ -]down\b", init))
            if orient == "upside_down":
                orient = "upright" if started_flipped else "upside_down"
            elif orient == "upright" and started_flipped:
                orient = "upside_down"
        if rel or rest_on_table:
            for s in subjects:
                add("release_settle", s, on="table" if on_support == "table" else None, orientation=orient,
                    source=f"text: '{(rel or rest_on_table).group(0)}'")
        # ---- articulation states
        if not is_with:
            jo = re.search(r"\b(?<!still )(open(?:ed)?|slid out|pulled out)\b", pred_text)
            jc = re.search(r"\b(?<!still )(closed|shut|pushed in|slid in|latched|hooked)\b", pred_text)
            for s in subjects:
                oid = _oid(s)
                if gc.objs.get(oid, {}).get("type") != "articulated" and motion not in ("open_joint", "close_joint"):
                    continue
                if jc:
                    add("joint_target", s, state="closed", source=f"text: '{jc.group(0)}'")
                elif jo:
                    add("joint_target", s, state="open", source=f"text: '{jo.group(0)}'")
        # ---- turning, tilting, pouring, pressing, cutting, knots
        mt = re.search(r"\b(unscrewed|loosened|disengaged|turned (?:about |by |roughly |approximately |around |through )?(?:a |an )?"
                       r"(?:quarter|half|full|whole|\d+))", pred_text)
        if mt:
            amt, frac = _amount(pred_text[mt.start():])
            ax = "vertical" if "vertical" in pred_text else None
            for s in subjects:
                add("rotation_turns", s, min_deg=None if amt is None else amt * frac, axis=ax,
                    release="disengaged" in mt.group(0) or "loose" in pred_text or "unscrewed" in mt.group(0),
                    source=f"text: '{mt.group(0)}'")
        mt = re.search(r"\b(horizontal|on its side|tilted|tipped)\b", pred_text)
        if mt and motion in ("tilt", "pour", "carry", "grasp_lift"):
            amt = 90.0 if mt.group(1) in ("horizontal", "on its side") else 20.0
            frac = 0.75 if re.search(r"\b(roughly|about|nearly|approximately)\b", pred_text) else 0.85
            for s in subjects:
                add("tilt_pour", s, min_tilt_deg=amt * frac if amt == 90.0 else amt, at_end=True, source=f"text: '{mt.group(0)}'")
        mt = re.search(r"\b(pressed|snapped|clicked|pushed down|depressed)\b", pred_text)
        if mt:
            full = bool(re.search(r"\b(through its stroke|full stroke|fully|all the way|bottom(?:s|ed)? out)\b", pred_text))
            for s in subjects:
                add("press_displacement", s, stroke_fraction=0.8 if full else 0.5, min_strokes=_strokes(pred_text),
                    source=f"text: '{mt.group(0)}'")
        if re.search(r"\b(cut|severed|separated into|into two pieces|in two pieces|in two)\b", pred_text):
            add("rope_severed", subjects[0] if subjects else None, source="text: 'cut / separated into pieces'")
        if re.search(r"\bknot", pred_text) and not re.search(r"\b(untie|untied)\b", pred_text):
            for s in subjects[:1]:
                add("knot", s, source="text: 'knot'")
    # ---- strands of one cable or ribbon (the whole sentence: a clause names the strands, not the object, so the subgoal's
    # deformable object stands in)
    gl = goal.lower()
    strand_subj = [r for r in refs if _oid(r) in gc.objs and gc.objs[_oid(r)].get("type") == "deformable"][:1]
    if strand_subj and not any(q["class"] == "strand_crossings" for q in preds):
        if re.search(r"\b(unbraid(?:ed|s)?|untangled|no (?:two )?strands? cross(?:es)?|no crossings?|(?:lie|lies|lying|are) (?:apart|side by side|separate(?:d|ly)?)|fann(?:ed|ing) out|without (?:any )?crossings?)\b", gl):
            add("strand_crossings", strand_subj[0], max_crossings=0, source="text: 'unbraided / no crossings'")
        elif re.search(r"\b(braided|plaited|into a braid|in a braid)\b", gl) and not re.search(r"\bunbraid", gl):
            add("strand_crossings", strand_subj[0], min_crossings=3, source="text: 'braided'")
    # ---- motion
    msub = acted_oids or refs[:1]
    for tag in _MOTION.get(motion, ()):
        src = f"motion '{motion}'"
        if tag == "held_end":
            for s in msub:
                add("held", s, mode="in_hand", window="end", source=src)
        elif tag == "held_through":
            for s in msub:
                add("held", s, mode="in_hand", window="through", source=src)
        elif tag == "held_fixed":
            for s in msub:
                add("held", s, mode="fixed", window="hold", source=src)
        elif tag == "lift":
            for s in msub:
                oid = _oid(s)
                if gc.objs.get(oid, {}).get("type") == "deformable" and len(gc.object_body_names(oid)) > 3:
                    notes.append(f"{s}: a multi-body deformable is not judged by lift clearance")
                    continue
                add("lift_clearance", s, source=src)
        elif tag == "release":
            for s in msub:
                add("release_settle", s, source=src)
        elif tag == "insertion":
            others = [r for r in refs if _oid(r) not in {_oid(x) for x in msub}]
            for s in msub:
                add("insertion_depth", s, B=others[0] if others else None, source=src)
        elif tag in ("joint_open", "joint_closed"):
            for s in msub:
                add("joint_target", s, state="open" if tag == "joint_open" else "closed", source=src)
        elif tag == "rotation":
            for s in msub:
                add("rotation_turns", s, min_deg=None, release=motion == "unscrew", source=src)
        elif tag == "pour":
            mt = re.search(r"\binto\b", goal.lower())
            tgt = gc.match(_phrase_after(goal.lower(), mt.end()), refs) if mt else []
            others = [r for r in refs if _oid(r) not in {_oid(x) for x in msub}]
            B = tgt[0]["ref"] if tgt else (others[0] if others else None)
            for s in msub:
                add("tilt_pour", s, B=B, min_tilt_deg=45.0, source=src)
        elif tag == "tilt":
            for s in msub:
                add("tilt_pour", s, min_tilt_deg=15.0, at_end=True, source=src)
        elif tag == "severed":
            add("rope_severed", None, source=src)
        elif tag == "knot":
            add("knot", msub[0] if msub else None, source=src)
    return _merge(preds), notes


def _merge(preds):
    """One predicate per (class, A, B): text details win over motion defaults."""
    plain = {(p.get("A"), p.get("B")) for p in preds if p["class"] == "insertion_depth" and not p.get("flush")}
    preds = [p for p in preds if not (p["class"] == "insertion_depth" and p.get("flush") and (p.get("A"), p.get("B")) in plain)]
    out = {}
    for p in preds:
        B = p.get("B")
        key = (p["class"], p.get("A"), tuple(B) if isinstance(B, list) else B, p.get("mode") if p["class"] == "held" else None)
        if p["class"] == "held":
            # one holding predicate per object: a motion window refines a text 'held'
            key = ("held", p.get("A"), None, None)
        if p["class"] in ("release_settle", "lift_clearance", "tilt_pour", "joint_target", "rotation_turns", "press_displacement"):
            key = (p["class"], p.get("A"), None, None)
        if key not in out:
            out[key] = dict(p)
            continue
        q = out[key]
        for k, v in p.items():
            if k == "source":
                if v not in q["source"]:
                    q["source"] = q["source"] + " + " + v
            elif v is not None and (q.get(k) is None or k in ("window",)):
                q[k] = v
            elif k == "mode" and q.get("mode") == "in_hand" and v == "fixed" and str(q.get("source", "")).startswith("motion"):
                q[k] = v
        if p["class"] == "rotation_turns" and p.get("min_deg") is not None:
            q["min_deg"] = p["min_deg"]
        if p["class"] == "tilt_pour":
            q["min_tilt_deg"] = max(float(q.get("min_tilt_deg") or 0), float(p.get("min_tilt_deg") or 0))
    return list(out.values())


# ------------------------------------------------------------------------------------------ evaluation of predicates
PARAMS = {
    "contact": ("tol", "throughout", "min_frac"),
    "apart": ("apart_m",),
    "stacked_on": ("tol", "level_tol_deg", "allow_table", "yaw_tol_deg", "yaw_sym_deg", "normal_min"),
    "inside_container": ("rim_band_frac", "rim_band_min_m", "xy_margin_m", "com_above_rim_m", "entirely"),
    "target_region": ("region", "z_range", "point", "yaw_target_deg", "yaw_tol_deg", "yaw_sym_deg", "upright_tol_deg", "hold_s"),
    "relative_pose": ("rel_pos", "rel_quat", "rel_euler_deg", "reference_frame", "pos_tol_m", "rot_tol_deg", "pos_axes",
                      "sym_axis", "sym_deg"),
    "joint_target": ("joint", "target", "fraction", "state", "tol", "open_fraction", "hold"),
    "held": ("mode", "min_frac", "max_gap_s", "opposing_deg", "min_links", "allow_table", "side", "hand_bodies"),
    "lift_clearance": ("min_height_m", "support", "at"),
    "release_settle": ("settle_s", "pos_tol_m", "rot_tol_deg", "seat_s", "seat_tol_m", "seat_tol_deg", "orientation", "orient_tol_deg", "on"),
    "insertion_depth": ("axis", "entry", "min_depth_m", "min_fraction", "point"),
    "rotation_turns": ("joint", "body", "axis", "min_deg", "direction", "release_joint", "release_travel_m", "relative_to"),
    "tilt_pour": ("min_tilt_deg", "region", "mouth_local", "up_local", "min_time_s", "target_margin_m", "at_end"),
    "press_displacement": ("joint", "body", "direction", "min_m", "min_strokes", "peak", "relative_to"),
    "deformable_shape": ("rope", "target_points", "target_ends", "tol_m", "ends_tol_m", "ordered", "planar"),
    "rope_severed": ("joins", "min_gap_m"),
    "knot": ("rope", "expect"),
    "strand_crossings": ("max_crossings", "min_crossings", "junction_m"),
}
END_STATE = {"contact", "apart", "stacked_on", "inside_container", "insertion_depth", "joint_target", "target_region", "relative_pose",
             "strand_crossings"}
# explicit predicates may make a gate stricter, never weaker than these bounds
CAPS = {("contact", "tol"): ("max", 0.006), ("held", "min_frac"): ("min", 0.8), ("held", "max_gap_s"): ("max", 0.5),
        ("release_settle", "pos_tol_m"): ("max", 0.01), ("release_settle", "rot_tol_deg"): ("max", 15.0),
        ("release_settle", "settle_s"): ("min", 0.15), ("release_settle", "seat_tol_m"): ("max", 0.03),
        ("release_settle", "seat_tol_deg"): ("max", 45.0), ("lift_clearance", "min_height_m"): ("min", 0.005),
        ("insertion_depth", "min_fraction"): ("min", 0.3), ("stacked_on", "level_tol_deg"): ("max", 20.0),
        ("stacked_on", "tol"): ("max", 0.005), ("tilt_pour", "min_tilt_deg"): ("min", 10.0)}


def _cap(cls, params):
    out, capped = dict(params), []
    for k, v in params.items():
        c = CAPS.get((cls, k))
        if c is None or v is None or isinstance(v, (list, dict, str, bool)):
            continue
        if c[0] == "max" and float(v) > c[1]:
            out[k] = c[1]; capped.append(k)
        elif c[0] == "min" and float(v) < c[1]:
            out[k] = c[1]; capped.append(k)
    return out, capped


def explicit_predicates(items, origin):
    """Manifest goal_predicates / deviation.predicates -> internal predicates. Each item:
    {class, bodies: {A, B} | [A, B], params: {...}, frames: [t0, t1] (optional), side (optional)}."""
    out, errs = [], []
    for k, it in enumerate(items or []):
        if not isinstance(it, dict) or it.get("class") not in CLASSES:
            errs.append(f"{origin}[{k}]: needs 'class' one of {list(CLASSES)}")
            continue
        b = it.get("bodies") or {}
        if isinstance(b, (list, tuple)):
            b = {"A": b[0] if len(b) > 0 else None, "B": (list(b[1:]) if len(b) > 2 else (b[1] if len(b) > 1 else None))}
        params = dict(it.get("params") or {})
        bad = sorted(set(params) - set(PARAMS[it["class"]]))
        if bad:
            errs.append(f"{origin}[{k}] ({it['class']}): unknown params {bad}; allowed {list(PARAMS[it['class']])}")
            continue
        params, capped = _cap(it["class"], params)
        out.append({"class": it["class"], "A_body": b.get("A"), "B_body": b.get("B"), "params": params,
                    "frames": it.get("frames"), "side": it.get("side"), "capped": capped, "source": origin})
    return out, errs


class _Unresolved(Exception):
    pass


def _objset(gc, oid):
    ep = gc.ep
    s = set()
    for n in gc.object_body_names(oid):
        try:
            s |= ep.subtree(ep.body(n))
        except ValueError:
            pass
    r = gc.object_root(oid)
    if r is not None:
        s |= ep.subtree(r)
    return s


def _acting_entries(gc, entries, oid):
    ep = gc.ep
    objset = _objset(gc, oid) if oid else set()
    out = []
    for sg in entries:
        if not sg.get("object") or not objset:
            continue
        try:
            b = ep.body(sg["object"])
        except ValueError:
            continue
        if b in objset:
            out.append(sg)
    return out


def _span(ep, entries):
    ws = [w for w in (entry_frames(ep, sg) for sg in entries) if w]
    return (min(w[0] for w in ws), max(w[1] for w in ws)) if ws else None


def _resolve_set(gc, ref, acted, exclude=None):
    """Body set for a Stage 1 ref or a scene body name; 'table' -> the table."""
    ep = gc.ep
    if ref is None:
        raise _Unresolved("no body named")
    if str(ref) in TABLE_NAMES:
        return set(ep.table), "table", None
    oid = _oid(ref)
    if oid in gc.objs or oid in gc.recon:
        b = gc.body_of(ref, acted)
        if b is None:
            raise _Unresolved(f"Stage 1 object {ref!r} has no body in the scene")
        S = set(ep.subtree(b))
        pieces = gc.group_pieces(oid)
        if pieces and not any(int(x) in ep.subtree(pb) for pb in pieces for x in (acted or ())):
            # a group object no entry of this subgoal acts on a piece of: the group as a whole, every piece
            for pb in pieces:
                S |= set(ep.subtree(pb))
    else:
        b = ep.body(ref, "goal_predicates body")
        S = set(ep.subtree(b))
    if exclude:
        S -= set(exclude)
    if not S:
        raise _Unresolved(f"{ref!r} has no bodies left after removing the subject's")
    return S, ep.name(b), b


def evaluate_predicate(gc, s1, entries, pred, stride=1, next_span=None):
    """Resolve one predicate's bodies, window and parameters, run its gate, return the row."""
    ep = gc.ep
    cls = pred["class"]
    explicit = "A_body" in pred
    acted = []
    for sg in entries:
        if sg.get("object"):
            try:
                acted.append(ep.body(sg["object"]))
            except ValueError:
                pass
    A_ref = pred.get("A_body") if explicit else pred.get("A")
    row = {"class": cls, "source": pred.get("source")}
    try:
        A, A_name, A_bid = (None, None, None)
        if A_ref is not None:
            A, A_name, A_bid = _resolve_set(gc, A_ref, acted)
        oid = _oid(pred.get("A")) if (not explicit and pred.get("A")) else None
        if oid is None and A_bid is not None:
            for o in list(gc.recon) + list(gc.objs):
                if A_bid in _objset(gc, o):
                    oid = o
                    break
        mine = _acting_entries(gc, entries, oid) if oid else []
        span = _span(ep, mine) or _span(ep, entries)
        if pred.get("frames"):
            span = ep.window(frames=pred["frames"])
        if span is None:
            raise _Unresolved("the subgoal's manifest entries have no frames in the trajectory")
        t0, t1 = span
        B_ref = pred.get("B_body") if explicit else pred.get("B")
        prm = dict(pred.get("params") or {})
        row.update(A=A_name, frames=[int(t0), int(t1)])
        sides = sorted({str(sg.get("side")) for sg in (mine or entries) if sg.get("side") in SIDES})
        if cls in ("contact", "apart"):
            if A is None or B_ref is None:
                raise _Unresolved("contact needs two bodies")
            B, B_name, _ = _resolve_set(gc, B_ref, (), exclude=A)
            row["B"] = B_name
            if cls == "contact":
                rep = gate_contact(ep, A, B, t0, t1, tol=prm.get("tol", GOAL_TOUCH_TOL_M),
                                   throughout=prm.get("throughout", pred.get("throughout", False)),
                                   min_frac=prm.get("min_frac", 0.9), stride=max(stride, 2), names=(A_name, B_name))
            else:
                rep = gate_contact(ep, A, B, t0, t1, apart=True, apart_m=prm.get("apart_m"), names=(A_name, B_name))
        elif cls == "stacked_on":
            Bl = B_ref if isinstance(B_ref, list) else [B_ref]
            if A is None or not Bl or Bl[0] is None:
                raise _Unresolved("stacked_on needs A and its support(s)")
            Bs, nbs = [], []
            for b in Bl:
                S, nm, _ = _resolve_set(gc, b, (), exclude=A)
                Bs.append(S); nbs.append(nm)
            row["B"] = nbs
            kw = {k: prm[k] for k in ("tol", "level_tol_deg", "allow_table", "yaw_tol_deg", "yaw_sym_deg", "normal_min") if k in prm}
            rep = gate_stacked_on(ep, A, Bs, t1, names={"A": A_name, "B": nbs}, **kw)
        elif cls == "inside_container":
            if A is None or B_ref is None:
                raise _Unresolved("inside_container needs A and the container")
            B, B_name, _ = _resolve_set(gc, B_ref, (), exclude=A)
            row["B"] = B_name
            kw = {k: prm[k] for k in PARAMS[cls] if k in prm}
            if "entirely" not in kw and re.search(r"\bentirely\b|\bwholly\b|\b(?:completely|fully) (?:inside|contained)\b", str(s1.get("goal", "")).lower()):
                kw["entirely"] = True
            rep = gate_inside_container(ep, A, B, t1, names=(A_name, B_name), **kw)
        elif cls == "target_region":
            if A is None or "region" not in prm:
                raise _Unresolved("target_region needs A and params.region (add it in goal_predicates)")
            kw = {k: prm[k] for k in PARAMS[cls] if k in prm}
            rep = gate_target_region(ep, A, t0, t1, names=(A_name,), **kw)
        elif cls == "relative_pose":
            if A is None or B_ref is None:
                raise _Unresolved("relative_pose needs A, B and a target (add it in goal_predicates)")
            B, B_name, _ = _resolve_set(gc, B_ref, (), exclude=A)
            row["B"] = B_name
            kw = {k: prm[k] for k in PARAMS[cls] if k in prm}
            rep = gate_relative_pose(ep, A, B, t1, names=(A_name, B_name), **kw)
        elif cls == "joint_target":
            jn = prm.get("joint")
            if jn is None:
                if A_bid is None:
                    raise _Unresolved("joint_target needs the moving part")
                js = gc.joints_of(A_bid, oid)
                if len(js) == 1:
                    jn = js[0]
                elif not js:
                    raise _Unresolved(f"{A_name} has no hinge or slide joint")
                else:
                    raise _Unresolved(f"{A_name} has several joints {js}: name one in goal_predicates")
            row["joint"] = jn
            kw = {k: prm[k] for k in ("target", "fraction", "tol", "open_fraction", "hold") if k in prm}
            st = prm.get("state", pred.get("state"))
            if "target" not in kw and "fraction" not in kw:
                kw["state"] = st or "closed"
            rep = gate_joint_target(ep, jn, t0, t1, **kw)
        elif cls == "held":
            if A is None:
                raise _Unresolved("held needs the object")
            mode = prm.get("mode", pred.get("mode", "in_hand"))
            side = prm.get("side", pred.get("side"))
            ah = str(s1.get("acting_hand", "")).lower()
            if side is None and ah in SIDES:
                side = ah
            elif side is None and ah == "both" and len(sides) == 1 and mine:
                # two hands, each on its own object (one brick per hand): that object's hand; two hands on one object (a lid
                # squeezed between them) or one entry for a two-hand hold: either hand's contacts count
                other = [sg for sg in entries if sg.get("side") in SIDES and sg.get("side") != sides[0] and sg.get("object")]
                if other and all(sg not in mine for sg in other):
                    side = sides[0]
            H = ep.hand_set(side if side in SIDES else None, prm.get("hand_bodies"))
            win = pred.get("window") or {"carry": "through", "hold": "hold"}.get(str(s1.get("motion")), "end")
            if pred.get("frames"):
                win = "through"
            h0, h1 = t0, t1
            if win == "end":
                k = int(min(0.5 * ep.fps, 0.3 * (t1 - t0 + 1)))
                h0 = max(t0, t1 - k)
            elif win == "hold":
                lab = [t for t in range(t0, t1 + 1) if re.match(r"\s*(hold|grip|press|push)\b", side_phase(ep.labels[t], side))]
                h0 = lab[0] if lab else t0 + (t1 - t0) // 2
            row["frames"] = [int(h0), int(h1)]
            row["hand"] = side or "any"
            deform = bool(oid and gc.objs.get(oid, {}).get("type") == "deformable" and len(gc.object_body_names(oid)) > 3)
            kw = {k: prm[k] for k in ("min_frac", "max_gap_s", "opposing_deg", "min_links", "allow_table") if k in prm}
            if deform:
                kw.setdefault("allow_table", True)
            rep = gate_held(ep, A, h0, h1, H, mode=mode, stride=stride if (h1 - h0) < 600 else max(stride, 2), names=(A_name,), **kw)
        elif cls == "lift_clearance":
            if A is None:
                raise _Unresolved("lift_clearance needs the object")
            sup = prm.get("support", pred.get("support"))
            S = None
            if sup is not None:
                S, _, _ = _resolve_set(gc, sup, (), exclude=A)
            rep = gate_lift_clearance(ep, A, t0, t1, min_height_m=prm.get("min_height_m", 0.01), support=S, at=prm.get("at", "end"),
                                      names=(A_name,))
        elif cls == "release_settle":
            if A is None:
                raise _Unresolved("release_settle needs the object")
            on = prm.get("on", pred.get("on"))
            S = None
            if on is not None:
                S, on_name, _ = _resolve_set(gc, on, (), exclude=A)
            kw = {k: prm[k] for k in ("settle_s", "pos_tol_m", "rot_tol_deg", "seat_s", "seat_tol_m", "seat_tol_deg", "orient_tol_deg") if k in prm}
            orient = prm.get("orientation", pred.get("orientation"))
            rep = gate_release_settle(ep, A, t0, t1, orientation=orient, on=S,
                                      names={"A": A_name, "on": on if on else None}, **kw)
            if not rep["ok"] and rep.get("release_frame") is not None and rep["release_frame"] >= t1 - 1 and next_span \
                    and next_span[1] > t1:
                # still held when this subgoal ends: the Stage 1 split put the letting go in the next subgoal
                rep2 = gate_release_settle(ep, A, t0, next_span[1], orientation=orient, on=S,
                                           names={"A": A_name, "on": on if on else None}, **kw)
                if rep2["ok"]:
                    rep = rep2
                    row["note"] = f"let go in the next Stage 1 subgoal (frame {rep2['release_frame']})"
                    row["frames"] = [int(t0), int(next_span[1])]
        elif cls == "insertion_depth":
            if A is None or B_ref is None:
                raise _Unresolved("insertion_depth needs A and what it goes into")
            B, B_name, B_bid = _resolve_set(gc, B_ref, (), exclude=A)
            row["B"] = B_name
            kw = {k: prm[k] for k in PARAMS[cls] if k in prm}
            if pred.get("flush") and "axis" not in kw:
                # flush: the trailing face reaches B's entry face; the insertion axis is the way A moved into B in this window
                p0, R0 = ep.pose(t0, B_bid)
                pa0, _ = ep.pose(t0, A_bid)
                p1, R1 = ep.pose(t1, B_bid)
                pa1, _ = ep.pose(t1, A_bid)
                dv = R1.T @ (pa1 - p1) - R0.T @ (pa0 - p0)
                if np.linalg.norm(dv) < 0.005:
                    raise _Unresolved(f"{A_name} hardly moves relative to {B_name} in frames {t0}-{t1}: name the insertion axis in goal_predicates")
                k = int(np.argmax(np.abs(dv)))
                ax = np.zeros(3); ax[k] = np.sign(dv[k])
                kw.update(axis=ax.tolist(), point="trailing", min_depth_m=-0.008)
            rep = gate_insertion_depth(ep, A, B, t1, names=(A_name, B_name), **kw)
        elif cls == "rotation_turns":
            kw = {k: prm[k] for k in ("direction",) if k in prm}
            jn, body, axis = prm.get("joint"), prm.get("body"), prm.get("axis")
            rj, rt = prm.get("release_joint"), prm.get("release_travel_m")
            mind = prm.get("min_deg", pred.get("min_deg"))
            if jn is None and body is None:
                if oid is None:
                    raise _Unresolved("rotation_turns needs the turning part")
                rec = gc.recon.get(oid) or {}
                # the screw-cap contract (construct_bottle): joints yaw + lift, params thread_pitch x thread_turns
                owner = None
                for o, r_ in gc.recon.items():
                    if A_bid is not None and A_bid in _objset(gc, o) and isinstance(r_.get("params"), dict) and r_["params"].get("thread_pitch"):
                        owner = o
                        break
                objs_ = _objset(gc, owner) if owner else (_objset(gc, oid) | (ep.subtree(A_bid) if A_bid is not None else set()))
                jy = [ep.model.joint(j).name for j in range(ep.model.njnt) if int(ep.model.jnt_bodyid[j]) in objs_
                      and int(ep.model.jnt_type[j]) == HINGE]
                jl = [ep.model.joint(j).name for j in range(ep.model.njnt) if int(ep.model.jnt_bodyid[j]) in objs_
                      and int(ep.model.jnt_type[j]) == SLIDE]
                rr = gc.recon.get(owner) if owner else rec
                js = rr.get("joints") if isinstance(rr.get("joints"), dict) else {}
                prm_ = rr.get("params") or {}
                if js.get("yaw"):
                    jy = [js["yaw"]]
                if js.get("lift"):
                    jl = [js["lift"]]
                if pred.get("release") and prm_.get("thread_pitch") and prm_.get("thread_turns") and jy and jl:
                    jn = jy[0]
                    rj, rt = jl[0], 0.96 * float(prm_["thread_pitch"]) * float(prm_["thread_turns"])
                    mind = mind if mind is not None else 0.96 * 360.0 * float(prm_["thread_turns"])
                elif len(jy) == 1 and (mind is not None):
                    jn = jy[0]
                elif mind is not None and A_bid is not None:
                    body = A_bid
                    _, R0 = ep.pose(0, A_bid)
                    axis = up_axis(R0).tolist()
                else:
                    raise _Unresolved("how far the part must turn is not stated and it has no thread contract: give min_deg "
                                      "(and the joint) in goal_predicates")
            elif body is not None:
                body = ep.body(body, "rotation_turns body")
            if mind is None:
                raise _Unresolved("rotation_turns needs min_deg")
            row["joint"] = jn
            rep = gate_rotation_turns(ep, t0, t1, float(mind), joint=jn, body=body, axis=axis,
                                      relative_to=ep.body(prm["relative_to"]) if prm.get("relative_to") else None,
                                      release_joint=rj, release_travel_m=rt, **kw)
        elif cls == "tilt_pour":
            if A is None:
                raise _Unresolved("tilt_pour needs the container")
            T_ = None
            if B_ref is not None:
                T_, B_name, _ = _resolve_set(gc, B_ref, (), exclude=A)
                row["B"] = B_name
            kw = {k: prm[k] for k in ("region", "mouth_local", "up_local", "target_margin_m") if k in prm}
            at_end = prm.get("at_end", pred.get("at_end", False))
            rep = gate_tilt_pour(ep, {A_bid} | A, t0, t1, min_tilt_deg=prm.get("min_tilt_deg", pred.get("min_tilt_deg", 45.0)),
                                 target=T_, min_time_s=prm.get("min_time_s", 0.2 if at_end else 0.5), at_end=at_end,
                                 names=(A_name, row.get("B", "the target")), **kw)
        elif cls == "press_displacement":
            jn, body = prm.get("joint"), prm.get("body")
            mm = prm.get("min_m")
            if jn is None and body is None:
                if A_bid is None:
                    raise _Unresolved("press_displacement needs the pressed part")
                js = [j for j in gc.joints_of(A_bid, oid) if joint_info(ep, j)["type"] == "slide"] or gc.joints_of(A_bid, oid)
                if len(js) != 1:
                    raise _Unresolved(f"the pressed part {A_name} has {len(js)} joints: name the joint and min_m in goal_predicates")
                jn = js[0]
            if mm is None:
                ji = joint_info(ep, jn) if jn else None
                if ji is None or ji["range"] is None:
                    raise _Unresolved("how far the part must move is not stated: give min_m in goal_predicates")
                mm = float(pred.get("stroke_fraction", 0.5)) * abs(ji["range"][1] - ji["range"][0])
            row["joint"] = jn
            rep = gate_press_displacement(ep, t0, t1, float(mm), joint=jn, body=ep.body(body) if body else None,
                                          direction=prm.get("direction"), min_strokes=int(prm.get("min_strokes", pred.get("min_strokes", 1))),
                                          peak=prm.get("peak", True),
                                          relative_to=ep.body(prm["relative_to"]) if prm.get("relative_to") else None,
                                          names=(A_name or jn,))
        elif cls == "rope_severed":
            sp = _span(ep, entries) if not pred.get("frames") else (t0, t1)
            rep = gate_rope_severed(ep, sp[0], sp[1], joins=prm.get("joins"), min_gap_m=prm.get("min_gap_m", 0.003))
            row["frames"] = [int(sp[0]), int(sp[1])]
        elif cls == "knot":
            pre = prm.get("rope") or (A_name.split("B_")[0] if A_name and "B_" in A_name else A_name)
            key, bl = cable_bodies(ep, pre)
            sp = _span(ep, entries) if not pred.get("frames") else (t0, t1)
            rep = gate_knot(ep, bl, sp[1], expect=prm.get("expect", "knot"), pre=key)
            row["frames"] = [int(sp[0]), int(sp[1])]
        elif cls == "strand_crossings":
            if A is None:
                raise _Unresolved("strand_crossings needs the cable or ribbon")
            bl = sorted(A) if not oid else sorted(_objset(gc, oid) or A)
            rep = gate_strand_crossings(ep, bl, t1, max_crossings=prm.get("max_crossings", pred.get("max_crossings")),
                                        min_crossings=prm.get("min_crossings", pred.get("min_crossings")),
                                        junction_m=prm.get("junction_m", 0.005))
        elif cls == "deformable_shape":
            pre = prm.get("rope") or A_name
            key, bl = cable_bodies(ep, pre)
            kw = {k: prm[k] for k in ("target_points", "target_ends", "tol_m", "ends_tol_m", "ordered", "planar") if k in prm}
            rep = gate_deformable_shape(ep, bl, t1, **kw)
        else:
            raise _Unresolved(f"unknown class {cls}")
    except _Unresolved as exc:
        row.update(ok=False, unresolved=True, issues=[f"{cls}: {exc}"])
        return row
    except ValueError as exc:
        row.update(ok=False, unresolved=True, issues=[f"{cls}: {exc}"])
        return row
    row["ok"] = bool(rep["ok"])
    row["issues"] = list(rep.get("issues", []))
    row["measured"] = {k: v for k, v in rep.items() if k not in ("ok", "issues")}
    if pred.get("capped"):
        row["capped_params"] = pred["capped"]
    return row


def _compiled_subgoal_measurement(gc, s1, entries, predicate, span, stride):
    """Measure independently compiled state atoms at this step's boundary."""
    from tool_library.common.final_goal_language import CUSTOM_CLASSES, measure_custom
    p = dict(predicate, source="stage1.goal: compositional language")
    start, end = span
    if p["class"] in CUSTOM_CLASSES:
        return measure_custom(gc, dict(p, start_frame=start), end)
    if p.get("throughout") and p["class"] == "relative_pose":
        frames = range(start, end + 1)
    elif p.get("persistent") and p["class"] in ("knot", "joint_target"):
        frames = range(max(start, end - int(np.ceil(.3 * gc.ep.fps))), end + 1)
    else:
        # Holding has an approach before the terminal grasp; the ordinary
        # end window remains appropriate, and the actual boundary is checked.
        row = evaluate_predicate(gc, s1, entries, p, stride=stride)
        if p["class"] == "held":
            endpoint = evaluate_predicate(gc, s1, entries, dict(p, frames=[end, end]), stride=1)
            row["boundary_check"] = endpoint
            if not endpoint["ok"]:
                row["ok"] = False
                row["issues"] += endpoint.get("issues", [])
        return row
    samples = [evaluate_predicate(gc, s1, entries, dict(p, frames=[t, t]), stride=1) for t in frames]
    bad = [r for r in samples if not r["ok"]]
    row = dict(bad[0] if bad else samples[-1])
    row.update(ok=not bad, frames=[frames.start, frames.stop - 1], frames_evaluated=len(samples))
    if p.get("persistent") and (frames.stop - 1 - frames.start) / gc.ep.fps < .3 - 1e-9:
        row["ok"] = False
        row["issues"] = list(row.get("issues", [])) + ["a persistent subgoal state needs at least 0.30 s of recorded state"]
    return row


def stage1_goal_report(ep, sem, man, recon=None, only=None, stride=1):
    """Evaluate independent Stage 1 goals. Manifest predicates can add checks, never replace them."""
    gc = GoalContext(ep, sem, recon)
    rows, issues = [], []
    if only is not None and set(map(int, only)) != {int(s["index"]) for s in sem.get("subgoals", [])}:
        issues.append("a selected subgoal subset is diagnostic only; full task acceptance requires every Stage 1 subgoal and task_success")
    spans = {}
    for s1_ in sem.get("subgoals", []):
        ex_ = [sg for sg in expand_entries(man.get("subgoals", [])) if int(s1_["index"]) in stage1_indices(sg) and not sg.get("skipped")]
        spans[int(s1_["index"])] = _span(ep, ex_)
    for s1 in sem.get("subgoals", []):
        k = int(s1["index"])
        if only is not None and k not in set(int(x) for x in only):
            continue
        ents = [sg for sg in expand_entries(man.get("subgoals", [])) if k in stage1_indices(sg)]
        ex = [sg for sg in ents if not sg.get("skipped")]
        row = {"index": k, "goal": s1.get("goal"), "motion": s1.get("motion"), "acting_hand": s1.get("acting_hand"),
               "entries": [sg.get("index") for sg in ex]}
        if not ex:
            kinds = {sg.get("skip_kind") for sg in ents}
            if ents and kinds <= {"no_robot_action"} and no_robot_action_allowed(s1):
                row.update(ok=True, judged=False, note="skipped: no robot action")
            else:
                row.update(ok=False, judged=False, note="not executed")
                issues.append(f"Stage 1 subgoal {k}: no executed manifest entry covers it, so its goal cannot be checked")
            rows.append(row)
            continue
        span = _span(ep, ex)
        row["frames"] = list(span) if span else None
        inferred, notes = infer_goal_predicates(gc, s1, ex)
        required = sorted({p["class"] for p in inferred})
        gp_items = [it for sg in ex for it in (sg.get("goal_predicates") or [])]
        gp, gp_err = explicit_predicates(gp_items, "goal_predicates")
        dev = next((sg.get("deviation") for sg in ex if isinstance(sg.get("deviation"), dict) and sg["deviation"].get("predicates")), None)
        dv, dv_err = explicit_predicates((dev or {}).get("predicates"), "deviation")
        trusted, trusted_err = explicit_predicates(s1.get("goal_predicates"), "stage1.goal_predicates")
        errs = gp_err + dv_err + trusted_err
        from tool_library.common.final_goal_language import compile_final
        compiled, coverage, language_errors = compile_final(gc, str(s1.get("goal") or ""))
        row["coverage"] = coverage
        errs += [e.replace("final-state atom", "subgoal atom") for e in language_errors]
        if not compiled or not coverage:
            errs.append("subgoal goal has no complete supported state requirement; express every observable clause explicitly in Stage 1 or implement its measurement")
        # A proposer must not define its own success criterion: preserve every
        # independently inferred predicate, even when an explicit one has the
        # same class. Stage 1 may supply a reviewed predicate for ambiguous prose.
        final = list(inferred) + trusted + gp
        required = sorted({p["class"] for p in inferred + trusted + compiled})
        dev_info = None
        if dev is not None:
            reason = str(dev.get("reason") or "").strip()
            dev_info = {"reason": reason, "classes": sorted({p["class"] for p in dv}), "accepted": False}
            errs.append("deviation predicates cannot replace the independent Stage 1 goal; adapt the robot motion while preserving its objects, roles and outcome")
        row.update(required_classes=required, deviation=dev_info, notes=notes)
        prow, seen = [], {}
        for p in final:
            r_ = evaluate_predicate(gc, s1, ex, p, stride=stride, next_span=spans.get(k + 1))
            # two refs of one object ('gift_box:lid', 'gift_box:hinge') can resolve to the same body and joint: judge it once
            key = (r_["class"], r_.get("A"), json.dumps(r_.get("B")), r_.get("joint"), json.dumps(r_.get("frames")),
                   json.dumps(p, sort_keys=True, default=str))
            if key in seen and not r_.get("unresolved"):
                q = seen[key]
                if r_.get("source") and r_["source"] not in str(q.get("source")):
                    q["source"] = f"{q.get('source')} + {r_['source']}"
                continue
            seen[key] = r_
            prow.append(r_)
        if span:
            prow += [_compiled_subgoal_measurement(gc, s1, ex, p, span, stride) for p in compiled]
        row["predicates"] = prow
        unresolved = not inferred and not trusted and not compiled
        if unresolved:
            errs.append(f"no independent goal predicate could be inferred from '{s1.get('goal')}' (motion {s1.get('motion')}): add "
                        f"goal_predicates for additional measurable checks in the reviewed Stage 1 input (class one of {list(CLASSES)}, bodies, params); "
                        "unsupported prose must also be made independently measurable")
        row["errors"] = errs
        row["ok"] = bool(prow) and all(r["ok"] for r in prow) and not errs
        for r in prow:
            if not r["ok"]:
                for msg in r.get("issues", [])[:2]:
                    issues.append(f"Stage 1 subgoal {k} [{r['class']}{' ' + str(r.get('A')) if r.get('A') else ''}]: {msg}")
        for e in errs:
            issues.append(f"Stage 1 subgoal {k}: {e}")
        rows.append(row)
    # Each Stage 1 step is checked at its own declared boundary. A state reached
    # later cannot retroactively turn a failed step into a successful one.
    issues = [i for i in issues if not i.startswith("Stage 1 subgoal ")]
    for r in rows:
        if r.get("predicates") is not None:
            r["ok"] = bool(r["predicates"]) and all(q["ok"] for q in r["predicates"]) and not r.get("errors")
    for r in rows:
        for q in r.get("predicates", []) or []:
            if not q["ok"]:
                for msg in q.get("issues", [])[:2]:
                    issues.append(f"Stage 1 subgoal {r['index']} [{q['class']}{' ' + str(q.get('A')) if q.get('A') else ''}]: {msg}")
        for e in r.get("errors", []) or []:
            issues.append(f"Stage 1 subgoal {r['index']}: {e}")
        if not r.get("judged", True) and not r["ok"]:
            issues.append(f"Stage 1 subgoal {r['index']}: no executed manifest entry covers it, so its goal cannot be checked")
    judged = [r for r in rows if r.get("judged", True)]
    if not judged:
        issues.append("Stage 1 supplies no independently verifiable robot task; an unjudged episode cannot be released")
    final_task = task_success_report(gc, sem, stride=stride)
    issues.extend(f"Stage 1 task_success: {message}" for message in final_task["issues"])
    return _report(not issues, issues, subgoals=rows, task_success=final_task, counts={"subgoals": len(rows), "judged": len(judged),
                                                              "passed": sum(1 for r in judged if r["ok"])})


def task_success_report(gc, sem, stride=1):
    """Measure every independently grounded final-state atom at the real end.

    Explicit Stage 1 declarations only add checks. They cannot waive unknown
    language, replace inferred requirements, or select an earlier success frame.
    The finite compositional grammar is in common.final_goal_language.
    """
    from tool_library.common.final_goal_language import compile_final, measure_custom, CUSTOM_CLASSES
    ep = gc.ep
    goal = sem.get("task_success")
    if not isinstance(goal, str) or not goal.strip():
        return _report(False, ["missing nonempty task_success final-state sentence in the independent Stage 1 input"],
                       goal=goal, predicates=[], coverage=[])
    inferred, coverage, issues = compile_final(gc, goal)
    trusted, errors = explicit_predicates(sem.get("task_success_predicates"), "stage1.task_success_predicates")
    issues.extend(errors)
    pseudo = {"goal": goal, "motion": "other", "object_refs": list(gc.objs), "acting_hand": "both"}
    final, predicates = list(inferred) + trusted, []
    allowed = END_STATE | {"held", "lift_clearance", "release_settle", "rope_severed", "knot", "deformable_shape"} | CUSTOM_CLASSES
    if not final:
        issues.append("task_success supplies no independently verifiable final-state predicate")
    for pred in final:
        p = dict(pred)
        if p["class"] not in allowed:
            issues.append(f"{p['class']} describes an event, not a final state")
            continue
        if p.get("frames") is not None:
            issues.append("task_success_predicates cannot override the final evaluation window with frames")
        tail = max(0, ep.T - 1 - int(round(.5 * ep.fps)))
        start = tail if p["class"] in ("held", "release_settle") else 0
        if p["class"] in END_STATE and not (p["class"] == "relative_pose" and p.get("throughout")):
            start = ep.T - 1
        p["frames"] = [start, ep.T - 1]
        if p["class"] == "release_settle":
            settle_s = float((p.get("params") or {}).get("settle_s", .3))
            p["frames"] = [max(0, ep.T - 1 - int(round(settle_s * ep.fps))), ep.T - 1]
        if p["class"] in CUSTOM_CLASSES:
            row = measure_custom(gc, p, ep.T - 1)
        elif p["class"] == "relative_pose" and p.get("throughout"):
            samples = [evaluate_predicate(gc, pseudo, [], {**p, "frames": [0, t]}, stride=stride) for t in range(ep.T)]
            bad = [r for r in samples if not r["ok"]]
            row = dict(bad[0] if bad else samples[-1])
            row.update(ok=not bad, frames=[0, ep.T - 1], frames_evaluated=ep.T)
        elif p["class"] in ("knot", "joint_target") and p.get("persistent"):
            first = max(0, ep.T - 1 - int(np.ceil(.3 * ep.fps)))
            # Every recorded terminal frame is judged, not just endpoints.
            samples = [evaluate_predicate(gc, pseudo, [], {**p, "frames": [t, t]}, stride=1) for t in range(first, ep.T)]
            bad = [r for r in samples if not r["ok"]]
            row = dict(bad[0] if bad else samples[-1])
            duration = (ep.T - 1 - first) / ep.fps
            row.update(ok=not bad and duration >= .3 - 1e-9, frames=[first, ep.T - 1], frames_evaluated=len(samples))
            if duration < .3 - 1e-9:
                row["issues"] = list(row.get("issues", [])) + ["a persistent final state needs at least 0.30 s of recorded state"]
        else:
            row = evaluate_predicate(gc, pseudo, [], p, stride=stride)
        if p["class"] == "held":
            endpoint = evaluate_predicate(gc, pseudo, [], {**p, "frames": [ep.T - 1, ep.T - 1]}, stride=1)
            row["final_frame_check"] = endpoint
            if not endpoint["ok"]:
                row["ok"] = False
                row["issues"] = list(row.get("issues", [])) + list(endpoint.get("issues", []))
        predicates.append(row)
        if not row["ok"]:
            issues.extend(f"[{row['class']} {row.get('A', '')}]: {message}" for message in row.get("issues", [])[:2])
    return _report(not issues and bool(predicates), issues, goal=goal, predicates=predicates, coverage=coverage,
                   scope="finite compositional physical-state grammar; unsupported clauses fail closed")


# ============================================================================================ verifier plumbing
def open_window(scene, trajectory, frames=None, subgoal=None, frame=None):
    """(Episode, t0, t1) for a verifier: frame -> that single frame (t0 = t1), frames -> [first, last], subgoal -> the
    frames whose npz `subgoal` index matches, else the whole trajectory."""
    ep = Episode(scene, trajectory)
    if frame is not None:
        t = int(frame)
        if t < 0:
            t += ep.T
        if not 0 <= t < ep.T:
            raise ValueError(f"frame: {frame} is outside the trajectory ({ep.T} frames)")
        return ep, t, t
    t0, t1 = ep.window(frames=frames, subgoal=subgoal)
    return ep, t0, t1


def finish(report, out_json=None):
    """Make a report JSON-safe and optionally write it."""
    rep = json.loads(json.dumps(report, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(rep, indent=1))
    return rep
