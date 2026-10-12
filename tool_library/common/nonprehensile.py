"""Thin and flat parts: what a hand can do with a part that lies too low to pinch or wrap where it lies, and the
non-prehensile moves that make it graspable.

A fingertip pad is a capsule (7 mm radius on the Sharpa hand). A pad that closes on a side face touches it at the
pad's own centre height, so its centre must stay a pad radius above the table and its contact a little below the
part's top: a part lower than about pad radius + 3 mm (10 mm) cannot be pinched from its sides where it lies, and a
finger fits under a part only where the gap is a fingertip thick (14 mm). Phones (9 mm), scissor straps (10 mm) and
flat strips (1-2 mm) fall below both. What makes them graspable instead:

- slide: push the part along the table with the fingers (pads pressed on its top, dragging it by friction; or a
  fingertip against a side face) until it overhangs the support's edge or rests against something;
- edge_pinch: fingers under the overhanging part, deeper than the thumb on its top (so the pinch also carries the
  part's moment about the edge), then lift;
- tilt_up: press the overhanging end down so the part pivots about the support's edge and its other end rises; the
  other hand pinches the raised end (the pressing hand holds the tilt until it is told to let go);
- hook: one or more fingers through an opening of the part (a ring, a strap, a handle), flexed against its inner
  wall, the thumb on the outer wall, then lift;
- strip_pinch: a flat strip pinched where it arches over something (the thumb on it, a fingertip under the arch);
  a strip lying flat on this table has none: pads closing along it slide over it (their friction 1.2 against its 1.0).

`hand_dims` reads the hand from its collision capsules; `lying_geometry` measures the part as it lies; `diagnose`
compares the two and names the primitive. `install` adds the Stage 3 motions to SubgoalPlanner, and `evaluate_extra`
measures their expectations (overhang, end_raised, max_table_pen_m) on the executed frames."""
from __future__ import annotations

import copy
import re
from pathlib import Path

import mujoco
import numpy as np

from tool_library.common import hand as H
from tool_library.common import plan as P
from tool_library.common import episode as E


def hand_geoms(model, bodies):
    """The part's geoms a hand can touch: collision geoms whose masks meet the hand's contact bits
    (hand.HAND_CONTACT_BITS) or that sit in an explicit pair. A geom on other bits (a magnet's sensing shell, which
    only other magnets meet) is no surface for a finger, nor does it fill an opening; a part with no such geom keeps
    all of them."""
    gs = E.collision_geoms(model, bodies)
    paired = {int(g) for g in model.pair_geom1} | {int(g) for g in model.pair_geom2}
    keep = [g for g in gs if (int(model.geom_contype[g]) | int(model.geom_conaffinity[g])) & H.HAND_CONTACT_BITS or g in paired]
    return keep or gs

MOTIONS = ("slide", "edge_pinch", "overhang_grasp", "tilt_up", "pivot", "hook", "strip_pinch")
TABLE_CLEAR_M = 0.001        # a pad posed this far above the table touches nothing
TOP_MARGIN_M = 0.002         # a side contact this far below the part's top edge does not roll over it
EDGE_CLEAR_M = 0.003         # a finger under an overhang stays this far outside the support's edge
COM_MARGIN_M = 0.012         # an overhanging part keeps its centre of mass this far inside the support's edge
THUMB_FORCE_N = 5.0          # what a thumb pad presses with at the finger servos' 0.8 N m cap (6-8 cm lever), less margin
TABLE_PEN_M = 0.002          # the deepest a hand may press into the support during these motions (soft contact)
HAND_MU = H.HAND_PAIR_FRICTION[0]
PUSH_SLOPE_FRAC = 0.6        # a push works while the contact normal's slope stays under this share of 1 / mu_support


# ----------------------------------------------------------------------------- the hand
def hand_dims(model, rig) -> dict:
    """The hand's sizes that decide what it can get under or around: pad_radius_m (the fingertip capsule),
    tip_thickness_m (the distal segment's diameter), finger_thickness_m (the thickest finger segment), span_m (thumb
    tip to index tip with the hand opened flat). Read from the collision capsules, so any hand that follows the
    library's naming gives its own numbers."""
    pad = float(H.pad_radius(model, rig, "index"))
    tips, segs = [], []
    caps = (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_SPHERE))
    tip_g = set().union(*[rig.tip_geoms(f) for f in rig.fingertip_sites if f != "thumb"]) if rig.fingertip_sites else set()
    for g in rig.collision_gids:
        nm = model.geom(g).name or ""
        if int(model.geom_type[g]) not in caps or "_thumb" in nm or "palm" in nm:
            continue
        r = float(model.geom_size[g][0])
        (tips if g in tip_g else segs).append(r)
    tip_r = min(tips) if tips else pad
    d = mujoco.MjData(model)
    rig.teleport(d, np.zeros(3), np.zeros(3))
    rig.set_fingers(d, {k: (lo if H.joint_role(k) == "flex" else (0.0 if lo <= 0.0 <= hi else lo)) for k, (lo, hi) in rig.finger_range.items()})
    mujoco.mj_kinematics(model, d)
    span = 0.0
    if "thumb" in rig.fingertip_sites and "index" in rig.fingertip_sites:
        span = float(np.linalg.norm(d.site_xpos[rig.fingertip_sites["thumb"]] - d.site_xpos[rig.fingertip_sites["index"]]))
    return {"pad_radius_m": round(pad, 4), "tip_thickness_m": round(2 * tip_r, 4),
            "finger_thickness_m": round(2 * max(segs + [tip_r]), 4), "span_m": round(span, 4)}


# ----------------------------------------------------------------------------- the support
def table_box(model, data=None):
    """The bench table as a box {center, half_extents, quat} (the body named 'table': its largest box geom), or None
    when the scene has no table box (a ground plane)."""
    data = data or mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    tb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    best = None
    for g in range(model.ngeom):
        if int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            continue
        if not (int(model.geom_bodyid[g]) == tb or (model.geom(g).name or "") == "g_table"):
            continue
        vol = float(np.prod(model.geom_size[g]))
        if best is None or vol > best[0]:
            best = (vol, g)
    if best is None:
        return None
    g = best[1]
    q = np.zeros(4); mujoco.mju_mat2Quat(q, data.geom_xmat[g])
    return {"center": [round(float(v), 6) for v in data.geom_xpos[g]], "half_extents": [round(float(v), 6) for v in model.geom_size[g]],
            "quat": [round(float(v), 8) for v in q], "geom": model.geom(g).name}


def table_floor(scene_xml):
    """table_box of a bench MJCF file, as a hand-scene `floor` (the finite table)."""
    m = mujoco.MjModel.from_xml_path(str(scene_xml))
    tb = table_box(m)
    if tb is None:
        return True
    return {k: tb[k] for k in ("center", "half_extents", "quat")}


def with_table_floor(scene, bench_scene=None):
    """A copy of a hand-scene dict whose floor is the bench table as the finite box it is (hand.build_hand_scene): a
    hand can reach under a part that overhangs its edge, and the visible-mesh floor checks count only what is over the
    table. `bench_scene` defaults to the scene's own bench_scene (hand_scene_from_recon records it)."""
    spec = copy.deepcopy(H.load_scene_spec(scene))
    if isinstance(spec.get("floor"), dict):
        return spec
    src = bench_scene or spec.get("bench_scene")
    if src and Path(str(src)).exists():
        spec["floor"] = table_floor(src)
    return spec


def _support_rect(fl):
    """(centre xy, x axis, y axis, half x, half y, top z) of a floor/table box dict."""
    from scipy.spatial.transform import Rotation as Rot
    q = fl.get("quat", (1, 0, 0, 0))
    R = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    c = np.asarray(fl["center"], float); h = np.asarray(fl["half_extents"], float)
    return c[:2], R[:2, 0] / max(np.linalg.norm(R[:2, 0]), 1e-12), R[:2, 1] / max(np.linalg.norm(R[:2, 1]), 1e-12), float(h[0]), float(h[1]), float(c[2] + h[2])


def exit_distance(rect, p, n):
    """Distance from the xy point p along the unit xy direction n to the edge of the support rectangle (inf when the
    ray never leaves it), and the edge's outward normal."""
    c, ex, ey, hx, hy, _ = rect
    p = np.asarray(p, float)[:2]; n = np.asarray(n, float)[:2]; n = n / max(np.linalg.norm(n), 1e-12)
    lp = np.array([(p - c) @ ex, (p - c) @ ey]); ln = np.array([n @ ex, n @ ey])
    best, normal = np.inf, None
    for k, (h, ax) in enumerate(((hx, ex), (hy, ey))):
        if abs(ln[k]) < 1e-9:
            continue
        s = (np.sign(ln[k]) * h - lp[k]) / ln[k]
        if 0 <= s < best:
            best, normal = float(s), np.sign(ln[k]) * ax
    return best, (None if normal is None else normal)


# ----------------------------------------------------------------------------- the part as it lies
def _geom_points(model, data, g):
    """Points on geom g's surface hull (world): box corners, capsule/cylinder end discs, sphere/ellipsoid axes, mesh
    vertices; enough to bound the geom's footprint and height exactly for boxes and meshes."""
    c = np.asarray(data.geom_xpos[g], float); R = np.asarray(data.geom_xmat[g], float).reshape(3, 3); s = model.geom_size[g]
    t = int(model.geom_type[g]); T = mujoco.mjtGeom
    if t == int(T.mjGEOM_BOX):
        L = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float) * s
    elif t in (int(T.mjGEOM_CAPSULE), int(T.mjGEOM_CYLINDER)):
        a = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        ring = np.stack([np.cos(a) * s[0], np.sin(a) * s[0], np.zeros_like(a)], 1)
        hz = s[1] + (s[0] if t == int(T.mjGEOM_CAPSULE) else 0.0)
        L = np.vstack([ring + [0, 0, s[1]], ring - [0, 0, s[1]], [[0, 0, hz], [0, 0, -hz]]])
    elif t == int(T.mjGEOM_SPHERE):
        L = np.vstack([np.eye(3), -np.eye(3)]) * s[0]
    elif t == int(T.mjGEOM_ELLIPSOID):
        L = np.vstack([np.eye(3), -np.eye(3)]) * s
    elif t == int(T.mjGEOM_MESH) and int(model.geom_dataid[g]) >= 0:
        mid = int(model.geom_dataid[g]); v0 = int(model.mesh_vertadr[mid]); n = int(model.mesh_vertnum[mid])
        L = np.asarray(model.mesh_vert[v0:v0 + n], float)
        if len(L) > 400:
            L = L[np.linspace(0, len(L) - 1, 400).astype(int)]
    else:
        return np.asarray([c], float)
    return L @ R.T + c


def _ray(model, data, g, pnt, vec):
    t = int(model.geom_type[g])
    if t == int(mujoco.mjtGeom.mjGEOM_MESH):
        return float(mujoco.mj_rayMesh(model, data, g, pnt, vec))
    if t in (int(mujoco.mjtGeom.mjGEOM_HFIELD), int(mujoco.mjtGeom.mjGEOM_PLANE), int(mujoco.mjtGeom.mjGEOM_SDF)):
        return -1.0
    return float(mujoco.mju_rayGeom(data.geom_xpos[g], data.geom_xmat[g], model.geom_size[g], pnt, vec, t))


def part_bodies(model, body):
    b = body if isinstance(body, int) else mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    if b < 0:
        raise ValueError(f"no body {body!r} in the scene")
    return int(b), E.subtree_bodies(model, b)


def footprint(model, data, bodies):
    """The part's footprint from above: centre (xy), principal axes u (long) and v, half lengths along them, and the
    part's lowest and highest z. Exact for boxes and meshes (hull points), close for round primitives."""
    gs = hand_geoms(model, bodies)
    Pts = np.vstack([_geom_points(model, data, g) for g in gs])
    xy = Pts[:, :2]
    c0 = xy.mean(0)
    C = np.cov((xy - c0).T) if len(xy) > 2 else np.eye(2)
    w, V = np.linalg.eigh(C)
    u = V[:, int(np.argmax(w))]; v = np.array([-u[1], u[0]])
    a, b = (xy - c0) @ u, (xy - c0) @ v
    c = c0 + u * 0.5 * (a.max() + a.min()) + v * 0.5 * (b.max() + b.min())
    return {"centre": c, "u": u, "v": v, "half_u": float(0.5 * (a.max() - a.min())), "half_v": float(0.5 * (b.max() - b.min())),
            "z_low": float(Pts[:, 2].min()), "z_high": float(Pts[:, 2].max()), "points": Pts}


def extent_along(fp, n):
    """How far the footprint reaches from its centre along the unit xy direction n (exact on its hull points)."""
    n = np.asarray(n, float)[:2]; n = n / max(np.linalg.norm(n), 1e-12)
    return float(((fp["points"][:, :2] - fp["centre"]) @ n).max())


def overhang_along(fp, rect, n, origin=None):
    """How far the part sticks out past the support's edge along the xy direction n (negative: how far inside it
    stays), measured from the part's centre (or `origin`) along n."""
    o = fp["centre"] if origin is None else np.asarray(origin, float)[:2]
    s, _ = exit_distance(rect, o, n)
    n = np.asarray(n, float)[:2]; n = n / max(np.linalg.norm(n), 1e-12)
    reach = float(((fp["points"][:, :2] - o) @ n).max())
    return reach - s, s


def under_gaps(model, data, bodies, support_z, fp, grid_m=0.002, max_n=60):
    """The free gap under the part's underside on a grid over its footprint: a ray up from just above the support finds
    the part's underside, a ray down from there finds whatever lies under it (the table, another object it rests on: a
    part draped over a block has no gap where the block is). Returns {max_gap_m, at (world xy of the widest gap),
    underside_z (the part's underside there), covered (fraction of the grid the part lies over)}; a part lying flat has
    max_gap ~0, an arch over a crossing its height."""
    gs = hand_geoms(model, bodies)
    robot = {b for b in range(model.nbody) if (model.body(b).name or "").startswith("robot_")}
    others = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) not in bodies and int(model.geom_bodyid[g]) not in robot
              and (model.geom_contype[g] or model.geom_conaffinity[g]) and int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_PLANE)]
    nu = int(np.clip(np.ceil(2 * fp["half_u"] / grid_m) + 1, 3, max_n)); nv = int(np.clip(np.ceil(2 * fp["half_v"] / grid_m) + 1, 3, max_n))
    up, down = np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])
    best, at, under_z, hits, tot = 0.0, None, None, 0, 0
    for a in np.linspace(-fp["half_u"], fp["half_u"], nu):
        for b in np.linspace(-fp["half_v"], fp["half_v"], nv):
            xy = fp["centre"] + fp["u"] * a + fp["v"] * b
            pnt = np.array([xy[0], xy[1], support_z - 0.002])        # from under the support's top: a ray never starts inside the part
            tot += 1
            hs = [h for h in (_ray(model, data, g, pnt, up) for g in gs) if h >= 0]
            if not hs:
                continue
            hits += 1
            zb = support_z - 0.002 + min(hs)
            q = np.array([xy[0], xy[1], zb - 1e-4])
            ds = [x for x in (_ray(model, data, g, q, down) for g in others) if x >= 0]
            floor = (zb - 1e-4 - min(ds)) if ds else support_z
            gap = zb - max(floor, support_z)
            if gap > best:
                best, at, under_z = gap, xy, zb
    return {"max_gap_m": round(float(best), 4), "at": None if at is None else [round(float(v), 4) for v in at],
            "underside_z": None if under_z is None else round(float(under_z), 5), "covered": round(hits / max(tot, 1), 3)}


def openings(model, data, bodies, fp, cell_m=0.001, max_cells=160):
    """Through-openings of the part seen from above (a ring, a strap, a handle lying flat): its top-down occupancy on a
    grid (rays down onto its own geoms), the empty cells no path from the border reaches, grouped. Returns
    [{centre (world xy), inscribed_radius_m (the largest circle that fits), area_m2, wall_top_z (the part's top around
    it)}], widest first."""
    from scipy import ndimage
    gs = hand_geoms(model, bodies)
    pad = 3 * cell_m
    hu, hv = fp["half_u"] + pad, fp["half_v"] + pad
    cell = max(float(cell_m), 2 * max(hu, hv) / float(max_cells))
    nu, nv = int(np.ceil(2 * hu / cell)) + 1, int(np.ceil(2 * hv / cell)) + 1
    occ = np.zeros((nu, nv), bool); top = np.full((nu, nv), -np.inf)
    z0 = fp["z_high"] + 0.02
    down = np.array([0.0, 0.0, -1.0])
    A = np.linspace(-hu, hu, nu); B = np.linspace(-hv, hv, nv)
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            xy = fp["centre"] + fp["u"] * a + fp["v"] * b
            pnt = np.array([xy[0], xy[1], z0])
            ds = [_ray(model, data, g, pnt, down) for g in gs]
            ds = [x for x in ds if x >= 0]
            if ds:
                occ[i, j] = True; top[i, j] = z0 - min(ds)
    free = ~occ
    lab, n = ndimage.label(free)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    dist = ndimage.distance_transform_edt(free) * cell
    out = []
    for k in range(1, n + 1):
        if k in border:
            continue
        m = lab == k
        if m.sum() < 3:
            continue
        ii, jj = np.nonzero(m)
        r = float(dist[m].max())
        kk = int(np.argmax(np.where(m, dist, -1).ravel()))
        ci, cj = np.unravel_index(kk, m.shape)
        ring = ndimage.binary_dilation(m, iterations=3) & occ
        out.append({"centre": [round(float(v), 4) for v in fp["centre"] + fp["u"] * A[ci] + fp["v"] * B[cj]],
                    "inscribed_radius_m": round(r, 4), "area_m2": round(float(m.sum()) * cell * cell, 6),
                    "wall_top_z": round(float(top[ring].max()) if ring.any() else float(fp["z_high"]), 4)})
    return sorted(out, key=lambda o: -o["inscribed_radius_m"])


def support_under(model, data, bodies, fp, exclude=()):
    """The body the part rests on and the height of its top under the part: the table, or another object whose top
    under the part's footprint is within 3 mm of the part's bottom (the highest such)."""
    rect = (fp["centre"][0] - fp["half_u"], fp["centre"][0] + fp["half_u"], fp["centre"][1] - fp["half_u"], fp["centre"][1] + fp["half_u"])
    best = (None, -np.inf)
    robot = {b for b in range(model.nbody) if (model.body(b).name or "").startswith("robot_")}
    roots = {}
    for b in range(1, model.nbody):
        if b in bodies or b in robot or b in exclude:
            continue
        r = b
        while int(model.body_parentid[r]) != 0:
            r = int(model.body_parentid[r])
        roots.setdefault(r, set()).add(b)
    for r, bs in roots.items():
        if r in bodies:
            continue
        try:
            z, info = E.support_top(model, data, bs, rect)
        except ValueError:
            continue
        if info.get("method", "").startswith("whole") and model.body(r).name != "table":
            continue
        if z <= fp["z_low"] + 0.003 and z > best[1]:
            best = (model.body(r).name, float(z))
    if best[0] is None:
        best = ("world", 0.0)
    return best


def is_flexible(model, bodies):
    """A chain of bodies joined by hinge or ball joints (a strip, a cord, a ribbon): it bends where pushed."""
    n = 0
    for b in bodies:
        for j in range(int(model.body_jntadr[b]), int(model.body_jntadr[b]) + int(model.body_jntnum[b])):
            if int(model.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_BALL)):
                n += 1
    return n >= 3


def lying_geometry(model, data, body, table=None):
    """Everything diagnose needs about a part as it lies: its footprint and height above its support, the gap under
    it, its openings, how far it overhangs (or stays inside) each edge of the table, mass and frictions."""
    mujoco.mj_forward(model, data)
    b, bodies = part_bodies(model, body)
    fp = footprint(model, data, bodies)
    sup, sz = support_under(model, data, bodies, fp)
    tb = table if table is not None else table_box(model, data)
    edges = {}
    if tb is not None:
        rect = _support_rect(tb)
        for nm, n in (("+x", (1, 0)), ("-x", (-1, 0)), ("+y", (0, 1)), ("-y", (0, -1))):
            ov, s = overhang_along(fp, rect, n)
            edges[nm] = {"overhang_m": round(ov, 4), "edge_distance_m": round(s, 4) if np.isfinite(s) else None}
    gaps = under_gaps(model, data, bodies, sz, fp)
    holes = openings(model, data, bodies, fp)
    mass = float(sum(model.body_mass[x] for x in bodies))
    gs = hand_geoms(model, bodies)
    mu_part = max(float(model.geom_friction[g][0]) for g in gs) if gs else 1.0
    tg = [g for g in range(model.ngeom) if (model.geom(g).name or "") == (tb or {}).get("geom")]
    mu_sup = max(mu_part, float(model.geom_friction[tg[0]][0]) if tg else 1.0)     # MuJoCo takes the larger of the two
    height = float(fp["z_high"] - sz)
    flexible = is_flexible(model, bodies)
    if flexible:
        # a strip draped over something stands high but its side faces are only as tall as its links are thick
        th = [2 * float(min(model.geom_size[g][:3])) for g in hand_geoms(model, bodies) if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_BOX)]
        if th:
            height = min(height, max(th))
    return {"body": model.body(b).name, "bodies": sorted(bodies), "footprint": fp, "support": sup, "support_top_z": sz,
            "height_m": height, "edges": edges, "under": gaps, "openings": holes, "mass_kg": mass,
            "mu_support": mu_sup, "mu_hand": HAND_MU, "flexible": flexible, "table": tb,
            "com": np.asarray(data.subtree_com[b], float).copy()}


# ----------------------------------------------------------------------------- the diagnosis
def side_pinch_min_height(dims):
    """Lowest part a pad pinches from its sides where it lies: the pad's centre a pad radius (plus TABLE_CLEAR_M)
    above the support touches the side face at that height, which must lie TOP_MARGIN_M below the part's top."""
    return dims["pad_radius_m"] + TABLE_CLEAR_M + TOP_MARGIN_M


def hook_thumb_wall(dims):
    """Lowest wall a thumb closes on from outside while a finger hooks it from inside: the thumb's distal cap a pad
    radius + 1 mm above the table touches the wall at that height, 1.5 mm under its top."""
    return dims["pad_radius_m"] + TABLE_CLEAR_M + 0.0015


def under_min_gap(dims):
    """Smallest gap a fingertip slides under without touching the support or lifting the part: its thickness plus a
    millimetre on each side."""
    return dims["tip_thickness_m"] + 2 * TABLE_CLEAR_M


def overhang_need(dims, mass, d_com, thumb_inset=None):
    """How far a rigid part must overhang the support's edge for an edge pinch that carries it: the grip fingers under
    it at inset f from its edge, the thumb on its top at inset t, the part's centre of mass d_com in from the edge. The
    thumb presses at most THUMB_FORCE_N, so the moment balance about the fingers, F_t (f - t) >= m g (d_com - f), puts
    the fingers at f >= (m g d_com + F_t t) / (F_t + m g); the finger then needs its own radius and EDGE_CLEAR_M past
    the support's edge. Returns (overhang_m, finger_inset_m, thumb_inset_m)."""
    t = dims["pad_radius_m"] + 0.003 if thumb_inset is None else float(thumb_inset)
    mg = 9.81 * float(mass)
    f = (mg * float(d_com) + THUMB_FORCE_N * t) / (THUMB_FORCE_N + mg)
    f = max(f, t + dims["tip_thickness_m"])          # the finger pads at least a fingertip deeper than the thumb
    return f + 0.5 * dims["tip_thickness_m"] + EDGE_CLEAR_M, f, t


def push_contact_deg(dims, height):
    """Angle above horizontal of the contact normal when an upright fingertip (its cap a pad radius + 1 mm above the
    table) pushes a part `height` tall: 0 on a face taller than the cap's centre, the slope onto its top edge on a lower
    one. The push presses the part down by tan(angle) per newton it pushes, which the support's friction multiplies:
    it moves the part only while tan(angle) < 1 / mu_support (PUSH_SLOPE_FRAC of it, for margin)."""
    r = dims["pad_radius_m"]
    drop = max(0.0, r + TABLE_CLEAR_M - float(height))
    return float(np.degrees(np.arcsin(min(1.0, drop / r))))


def drag_force_need(mass, mu_hand, mu_support):
    """Normal force fingertips pressed on a part's top need to drag it along the support by friction: mu_hand N >
    mu_support (N + m g), so N > mu_support m g / (mu_hand - mu_support); None when mu_hand <= mu_support (the pads
    slip on the part before it moves)."""
    if mu_hand <= mu_support + 1e-6:
        return None
    return float(mu_support * 9.81 * mass / (mu_hand - mu_support))


def diagnose(model, data, body, rig, wrist_force_n=H.WRIST_FORCE_N):
    """Can the part (body: an object root or a part of it) be pinched or wrapped from where it lies, and if not, which
    primitive makes it graspable? Returns {geometry, hand, verdicts {name: {ok, why, ...}}, graspable_as_lies,
    recommend [{plan, why}]} (see thin_grasp_diagnosis)."""
    dims = hand_dims(model, rig)
    g = lying_geometry(model, data, body)
    fp, h = g["footprint"], g["height_m"]
    mass = g["mass_kg"]
    v = {}
    hmin = side_pinch_min_height(dims)
    span_need = 2 * fp["half_v"] + dims["tip_thickness_m"]
    v["side_pinch"] = {"ok": bool(h >= hmin and span_need <= dims["span_m"]), "need_height_m": round(hmin, 4),
                       "why": (f"stands {1000 * h:.1f} mm above its support; side contacts need {1000 * hmin:.1f} mm (pad radius "
                               f"{1000 * dims['pad_radius_m']:.0f} mm + {1000 * TABLE_CLEAR_M:.0f} mm off the table + {1000 * TOP_MARGIN_M:.0f} mm "
                               f"under the top edge)") if h < hmin else
                              (f"its narrow side is {1000 * 2 * fp['half_v']:.0f} mm across; the thumb-index span is {1000 * dims['span_m']:.0f} mm"
                               if span_need > dims["span_m"] else "pads fit on its side faces")}
    gap = g["under"]["max_gap_m"]; gmin = under_min_gap(dims)
    v["pinch_under"] = {"ok": bool(gap >= gmin), "gap_m": gap, "need_gap_m": round(gmin, 4), "at": g["under"]["at"],
                        "why": f"the widest gap under it is {1000 * gap:.1f} mm; a fingertip needs {1000 * gmin:.1f} mm"}
    wmin = dims["finger_thickness_m"] + 0.005
    v["wrap"] = {"ok": bool(h >= wmin and 2 * fp["half_v"] <= dims["span_m"]), "need_height_m": round(wmin, 4),
                 "why": f"a wrap curls fingers {1000 * dims['finger_thickness_m']:.0f} mm thick around it: it needs {1000 * wmin:.0f} mm of height, it has {1000 * h:.1f}"}
    # the edge pinch and the slide that makes it possible: toward the nearest edge of the table
    best_edge = None
    if g["edges"]:
        best_edge = min((e for e in g["edges"].items() if e[1]["edge_distance_m"] is not None),
                        key=lambda e: e[1]["edge_distance_m"] - e[1]["overhang_m"] * 0.0, default=None)
    n_dir = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}
    if best_edge is not None:
        nm, ed = best_edge
        n = np.asarray(n_dir[nm], float)
        reach = extent_along(fp, n)                       # centre to the part's edge along n
        com_in = float(reach - (np.asarray(g["com"][:2]) - fp["centre"]) @ n)   # the part's edge to its centre of mass
        need, f_in, t_in = overhang_need(dims, mass, com_in)
        allow = com_in - COM_MARGIN_M
        cur = ed["overhang_m"]
        v["edge_pinch"] = {"ok": bool(cur >= need and need <= allow), "edge": nm, "overhang_m": round(cur, 4), "need_overhang_m": round(need, 4),
                           "max_overhang_m": round(allow, 4), "finger_inset_m": round(f_in, 4), "thumb_inset_m": round(t_in, 4),
                           "why": (f"overhangs the {nm} edge by {1000 * cur:.0f} mm; an edge pinch that carries its moment needs {1000 * need:.0f} mm"
                                   + (f", more than it can overhang and stay on the table ({1000 * allow:.0f} mm)" if need > allow else ""))}
        Nd = drag_force_need(mass, HAND_MU, g["mu_support"])
        side_push = np.tan(np.radians(push_contact_deg(dims, h))) <= PUSH_SLOPE_FRAC / g["mu_support"]
        # a drag rides on the pads' friction margin over the support's (1.2 vs 1.0 measured too thin on the bench: the
        # pads slid 6 cm over a 170 g slab pressed with 21 N); a push does not depend on it
        drag_ok = Nd is not None and Nd * 1.5 <= wrist_force_n and HAND_MU >= 1.5 * g["mu_support"]
        dist = max(0.0, need + 0.005 - cur)
        v["slide"] = {"ok": bool((drag_ok or side_push) and need <= allow), "mode": "push" if side_push else ("drag" if drag_ok else None),
                      "edge": nm, "direction": list(n_dir[nm]), "distance_m": round(dist, 4), "target_overhang_m": round(need + 0.005, 4),
                      "drag_force_n": None if Nd is None else round(Nd, 2),
                      "push_contact_deg": round(push_contact_deg(dims, h), 1),
                      "why": ("upright fingertips behind it push it (contact {:.0f} deg above horizontal, support friction {:.2f})".format(push_contact_deg(dims, h), g["mu_support"])
                              if side_push else ("fingertips pressed on its top drag it: friction hand {:.2f} vs support {:.2f} needs {:.1f} N of press (wrist cap {:.0f} N)".format(HAND_MU, g["mu_support"], Nd, wrist_force_n)
                                                 if drag_ok else "neither a push (too low: the fingertip presses it down more than it pushes) nor a drag (the pads' friction does not beat the support's by half) moves it"))}
        # tilt up about the edge: pressing the overhanging end lifts the far end by press x (far lever / near lever)
        if cur > 0.01:
            near = cur - 0.006                                   # press point a few mm in from the overhanging end
            far = 2 * fp["half_u" if abs(n @ fp["u"]) > 0.7 else "half_v"] - cur
            lift_need = gmin + 0.006
            press = lift_need * near / max(far, 1e-6)
            torque_ok = True
            v["tilt_up"] = {"ok": bool(far > 0 and press < 0.6 * near and torque_ok), "mode": "press", "edge": nm,
                            "press_depth_m": round(press, 4), "far_lift_m": round(lift_need, 4),
                            "why": f"pressing the overhanging end {1000 * press:.0f} mm down pivots it about the {nm} edge and raises the far end {1000 * lift_need:.0f} mm"}
        else:
            v["tilt_up"] = {"ok": False, "mode": "press", "why": "nothing to pivot on: it does not overhang an edge (slide it first) and lies flat on its support"}
    hole = g["openings"][0] if g["openings"] else None
    rmin = 0.5 * dims["tip_thickness_m"] + 0.0015
    if hole is not None:
        wall = hole["wall_top_z"] - g["support_top_z"]
        v["hook"] = {"ok": bool(hole["inscribed_radius_m"] >= rmin), "opening": hole, "need_radius_m": round(rmin, 4),
                     "thumb_outside": bool(wall >= hook_thumb_wall(dims) - 1e-4), "wall_height_m": round(wall, 4),
                     "why": f"an opening {1000 * 2 * hole['inscribed_radius_m']:.0f} mm across (a fingertip needs {1000 * 2 * rmin:.0f} mm); its wall stands {1000 * wall:.1f} mm"
                            + ("" if wall >= hook_thumb_wall(dims) - 1e-4 else " (too low for the thumb on its outside: a pure hook, pull and lift)")}
    else:
        v["hook"] = {"ok": False, "why": "no through-opening seen from above"}
    thin = h < hmin
    arch = gap >= gmin
    v["strip_pinch"] = {"ok": bool(thin and g["flexible"] and arch), "flexible": g["flexible"], "gap_m": gap, "at": g["under"]["at"],
                        "why": (f"a strip arching {1000 * gap:.0f} mm at {g['under']['at']}: the thumb on it, a fingertip under the arch "
                                "(no pose tool in the library holds one yet: try handle_pinch_pose or pose_from_points across the arch, stress-tested "
                                "in the hand scene, then strip_pinch)" if thin and g["flexible"] and arch else
                                (f"a strip lying flat (its widest gap {1000 * gap:.1f} mm, a fingertip needs {1000 * gmin:.0f}): no pad gets under or "
                                 "beside it, and pads closing along it on this support slide over it rather than gather it (their friction "
                                 f"{HAND_MU} against the strip's {g['mu_support']:.2f} on the table); lift it where it arches over something"
                                 if thin and g["flexible"] else ("rigid: not a strip" if thin else "tall enough to pinch directly")))}
    as_lies = v["side_pinch"]["ok"] or v["pinch_under"]["ok"] or v["wrap"]["ok"] or v.get("edge_pinch", {}).get("ok", False)
    rec = []
    if v["side_pinch"]["ok"] or v["wrap"]["ok"]:
        rec.append({"plan": ["grasp_lift"], "why": "pinch or wrap it where it lies (generate_rigid_grasps, pose_from_points)"})
    if v["pinch_under"]["ok"]:
        rec.append({"plan": ["grasp_lift"], "why": f"a fingertip fits under it at {v['pinch_under']['at']} (handle_pinch_pose / pose_from_points there)"})
    if v.get("edge_pinch", {}).get("ok"):
        rec.append({"plan": ["edge_pinch"], "why": "it already overhangs enough: overhang_pinch_pose, then edge_pinch"})
    if v["hook"]["ok"]:
        rec.append({"plan": ["hook"], "why": "hook it through its opening (hook_grasp_pose, then hook)"})
    if v["strip_pinch"]["ok"]:
        rec.append({"plan": ["strip_pinch"], "why": "pinch it where it arches: thumb on top, a fingertip under (no pose tool holds one yet: handle_pinch_pose or "
                    "pose_from_points across the arch, stress-tested, then strip_pinch)"})
    if v.get("slide", {}).get("ok") and not v.get("edge_pinch", {}).get("ok"):
        sl = v["slide"]
        rec.append({"plan": ["slide", "edge_pinch"], "why": f"slide it {1000 * sl['distance_m']:.0f} mm toward the {sl['edge']} edge ({sl['mode']}, slide_contact_pose) "
                                                           f"until it overhangs {1000 * sl['target_overhang_m']:.0f} mm, then edge_pinch (overhang_pinch_pose)"})
        rec.append({"plan": ["slide", "tilt_up", "edge_pinch (other hand)"], "why": "or press the overhanging end down (tilt_up) and pinch the raised far end with the other hand"})
    if not rec:
        rec.append({"plan": [], "why": "no primitive in the library makes it graspable: report it blocked with these measurements"})
    geo = {"body": g["body"], "height_m": round(h, 4), "support": g["support"], "support_top_z": round(g["support_top_z"], 4),
           "footprint": {"centre": [round(float(x), 4) for x in fp["centre"]], "u": [round(float(x), 4) for x in fp["u"]],
                         "half_u_m": round(fp["half_u"], 4), "half_v_m": round(fp["half_v"], 4)},
           "edges": g["edges"], "under": g["under"], "openings": g["openings"][:3], "mass_kg": round(mass, 4),
           "mu_support": round(g["mu_support"], 3), "mu_hand": HAND_MU, "flexible": g["flexible"]}
    return {"geometry": geo, "hand": dims, "verdicts": v, "graspable_as_lies": bool(as_lies), "recommend": rec}


# ----------------------------------------------------------------------------- poses
def _unit(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-12)


def thin_axis(fp):
    """The part's thickness direction (smallest spread of its hull points), pointing up."""
    Pts = fp["points"] - fp["points"].mean(0)
    w, V = np.linalg.eigh(Pts.T @ Pts)
    n = V[:, 0]
    return n if n[2] >= 0 else -n


def surface_z(model, data, gs, xy, z_from, up=False):
    """Height where a vertical ray at xy meets the geoms gs: from z_from downward (the top surface) or upward (up=True,
    the underside). None when the ray misses."""
    vec = np.array([0.0, 0.0, 1.0 if up else -1.0])
    pnt = np.array([float(xy[0]), float(xy[1]), float(z_from)])
    ds = [x for x in (_ray(model, data, g, pnt, vec) for g in gs) if x >= 0]
    if not ds:
        return None
    return float(z_from + (1.0 if up else -1.0) * min(ds))


def footprint_chord(fp, e, depth):
    """The footprint's extent across the horizontal unit direction e (as offsets along the perpendicular t = [-e_y, e_x]
    from its centre) `depth` in from its edge along e: the chord of its convex hull there (a rotated part's corner gives
    a short chord)."""
    from scipy.spatial import ConvexHull
    e = np.asarray(e, float)[:2]; t = np.array([-e[1], e[0]])
    rel = fp["points"][:, :2] - fp["centre"]
    reach = float((rel @ e).max())
    try:
        V = rel[ConvexHull(rel).vertices]
    except Exception:
        V = rel
    c0 = reach - float(depth)
    ts = []
    for i in range(len(V)):
        a_, b_ = V[i], V[(i + 1) % len(V)]
        fa, fb = a_ @ e - c0, b_ @ e - c0
        if fa == 0:
            ts.append(a_ @ t)
        if fa * fb < 0:
            q = a_ + (b_ - a_) * (fa / (fa - fb))
            ts.append(q @ t)
    return (min(ts), max(ts)) if ts else (float((rel @ t).min()), float((rel @ t).max()))


def edge_pinch_targets(model, data, bodies, edge_dir, thumb_inset, finger_inset, fingers=("index", "middle"), s=0.5,
                       finger_spread_m=0.019):
    """Contact points of an edge pinch on a flat part: the thumb on its top surface `thumb_inset` in from its edge
    along the horizontal direction edge_dir (outward), the grip fingers on its underside `finger_inset` in, spread
    along the edge, at `s` (0..1) along it. Points come from rays onto the part's own geometry, so a tilted or curved
    part gets its real surfaces. Returns (points, normals, info)."""
    fp = footprint(model, data, bodies)
    gs = hand_geoms(model, bodies)
    e = _unit([edge_dir[0], edge_dir[1], 0.0])[:2]
    t = np.array([-e[1], e[0]])
    reach = extent_along(fp, e)
    rel = fp["points"][:, :2] - fp["centre"]

    def chord(depth):
        return footprint_chord(fp, e, depth)

    lo, hi = chord(float(finger_inset))
    half = 0.5 * float(finger_spread_m) * (len(fingers) - 1)
    if hi - lo < 2 * half + 0.008:
        raise ValueError(f"edge pinch: the part is {1000 * (hi - lo):.0f} mm across {1000 * finger_inset:.0f} mm in from its edge, "
                         f"too short for {len(fingers)} fingers {1000 * finger_spread_m:.0f} mm apart")
    along = lo + half + 0.004 + (hi - lo - 2 * half - 0.008) * float(s)
    edge_xy = fp["centre"] + e * reach + t * along
    nt = thin_axis(fp)
    z_top0, z_bot0 = fp["z_high"] + 0.05, fp["z_low"] - 0.05
    pts, nrm = {}, {}
    tl, th_ = chord(float(thumb_inset))
    xy = fp["centre"] + e * (reach - float(thumb_inset)) + t * float(np.clip(along, tl + 0.006, th_ - 0.006))
    zt = surface_z(model, data, gs, xy, z_top0)
    if zt is None:
        raise ValueError(f"edge pinch: no part surface {1000 * thumb_inset:.0f} mm in from its edge for the thumb")
    pts["thumb"] = np.array([xy[0], xy[1], zt]); nrm["thumb"] = nt.copy()
    offs = (np.arange(len(fingers)) - 0.5 * (len(fingers) - 1)) * float(finger_spread_m)
    for f, o in zip(fingers, offs):
        xy = edge_xy - e * float(finger_inset) + t * o
        zb = surface_z(model, data, gs, xy, z_bot0, up=True)
        if zb is None:
            raise ValueError(f"edge pinch: no part underside {1000 * finger_inset:.0f} mm in from its edge for the {f}")
        pts[f] = np.array([xy[0], xy[1], zb]); nrm[f] = -nt
    return pts, nrm, {"edge_xy": edge_xy.tolist(), "edge_dir": e.tolist(), "along_dir": t.tolist(), "thin_axis": nt.tolist()}


def finger_support_clearance(model, data, floor_gid, pts, fingers, dims):
    """How far each grip finger's pad, at its target under the part, stays clear of the support: the larger of its
    horizontal distance outside the support's edge and its height above the support's top, less the pad radius. A
    finger under an overhang is clear by the first, under a raised (tilted) end by the second."""
    out = {}
    if floor_gid is None or floor_gid < 0:
        return {f: 1.0 for f in fingers}
    top, n_up, fp = H.floor_surface(model, data, floor_gid)
    r = dims["pad_radius_m"]
    for f in fingers:
        p = np.asarray(pts[f], float) - n_up * r                # the pad's lowest point under its contact
        h = float((p - top) @ n_up)
        if fp is None:
            out[f] = h
            continue
        c, ex, ey, hx, hy = fp
        dx, dy = abs(float((p - c) @ ex)) - hx, abs(float((p - c) @ ey)) - hy
        outside = max(dx, dy)
        out[f] = max(h, outside - r)
    return out


def edge_pinch_seeds(model, data0, rig, targets, edge_dir, palm_up=True, front=False):
    """Wrist seeds for an edge pinch: the finger axis pointing into the part (minus edge_dir, pitched +-20 deg), the
    palm toward the part's underside, rolled about the finger axis. With `front` also a hand in front of the edge with
    its fingers pointing down past it (curled in under a raised end, the palm beside it instead of under it: under a
    tilted part there is room for fingers, not for a palm)."""
    from scipy.spatial.transform import Rotation as Rot
    pk = H.grasp_pocket(model, data0, rig)
    cen = np.mean(list(targets.values()), axis=0)
    din = -_unit([edge_dir[0], edge_dir[1], 0.0])
    ax = _unit(np.cross([0.0, 0.0, 1.0], din))
    out = []
    for roll in (0, 25, -25, 50, -50, 90, -90):
        for pitch in (0, 20, -20):
            Rp = Rot.from_rotvec(np.radians(pitch) * ax).as_matrix()
            f_w = Rp @ din
            n_w = Rp @ np.array([0.0, 0.0, 1.0 if palm_up else -1.0])
            n_w = Rot.from_rotvec(np.radians(roll) * f_w).as_matrix() @ n_w
            out.append(H.pocket_pose(pk, cen, n_w, f_w))
    if front:
        for tilt in (50, 70, 90):                     # fingers from pointing in to pointing down past the edge
            f_w = Rot.from_rotvec(np.radians(tilt) * ax).as_matrix() @ din
            for roll in (0, 30, -30):
                n_w = Rot.from_rotvec(np.radians(roll) * f_w).as_matrix() @ _unit(np.cross(f_w, ax))
                out.append(H.pocket_pose(pk, cen - np.asarray([*(-din[:2]), 0.0]) * 0.0, n_w, f_w))
                out.append(H.pocket_pose(pk, cen, -n_w, f_w))
    return out


def solve_edge_pinch(model, data0, rig, ids, pts, nrm, edge_dir, iters=240, time_budget_s=None):
    """The fingertip IK for an edge pinch (edge_pinch_seeds), the idle fingers curled clear. Returns (pose dict, worst
    residual m, overlap m)."""
    so = H.pad_radius(model, rig, "index") + 0.001
    tg = {f: np.asarray(pts[f], float) + np.asarray(nrm[f], float) * so for f in pts}
    seeds = edge_pinch_seeds(model, data0, rig, tg, edge_dir)
    worst, res, pen, q, face = H.solve_fingertip_ik(model, data0, rig, ids, tg, seeds, iters=int(iters), time_budget_s=time_budget_s)
    d = mujoco.MjData(model); d.qpos[:] = q; mujoco.mj_forward(model, d)
    H.curl_free_fingers_clear(model, d, rig, ids, set(pts))
    mujoco.mj_forward(model, d)
    pose = H.pose_from_data(model, d, rig, rig.side)
    ovl = H.deepest_overlap(model, d, rig, ids, keep_tips=tuple(pts), visual_floor=True)
    pose.update({"residual_mm": {f: round(v * 1000, 2) for f, v in res.items()}, "worst_residual_mm": round(worst * 1000, 2),
                 "overlap_mm": round(ovl * 1000, 2), "targets": {f: v.tolist() for f, v in tg.items()}, "standoff_m": so})
    return pose, worst, ovl, d


# ----------------------------------------------------------------------------- Stage 3 motions
def load_pose(pose):
    from tool_library.common.subgoal import load_pose as _lp
    return _lp(pose)


def _press_caps(sp, side, dvec, force, phases):
    """The wrist force cap during a press: `force` on the wrist slide axis nearest the press direction (the others
    keep the library's cap), on the named phases only (press_seat's rule)."""
    m, d = sp.model, sp.pl.data
    rig = sp.pl.rigs[side]
    R0 = P.root_frame(m, d, rig)[1]
    loc = R0.T @ np.asarray(dvec, float)
    k = int(np.argmax(np.abs(loc)))
    caps = [float(H.WRIST_FORCE_N)] * 3
    if abs(loc[k]) >= np.cos(np.radians(25.0)):
        caps[k] = float(force)
    else:
        caps = [float(force)] * 3
    wf = sp._rollout.setdefault("wrist_force_n", {})
    wf[side] = {"cap": caps, "phases": list(phases)}


def _horizontal(v, name):
    v = np.asarray(v, float).reshape(-1)
    if v.size < 2:
        raise ValueError(f"{name} must be [dx, dy] or [dx, dy, dz]")
    h = np.array([v[0], v[1], 0.0])
    if np.linalg.norm(h) < 1e-9:
        raise ValueError(f"{name} must have a horizontal component")
    return h / np.linalg.norm(h)


def _no_hold(sp, side, motion):
    if sp.state["held"].get(side):
        raise ValueError(f"{motion}: the {side} hand still holds {sp.state['held'][side]!r}; let go first (place with release)")


def pose_mode_hint(sg):
    """The slide mode a subgoal's pose records (slide_contact_pose's slide_mode), without re-anchoring it."""
    try:
        return load_pose(sg["pose"]).get("slide_mode")
    except Exception:
        return None


def m_slide(sp, sg, side, obj):
    """Slide a part along its support with the fingers: `mode` 'drag' (default; the pose is a flat hand whose pads rest
    on the part's top, slide_contact_pose / flat_hand_pose: the wrist presses them down with `force_n` and the part goes
    with the hand by friction, which needs the hand's friction above the support's) or 'push' (a fingertip against a
    side face, push_pose_from_point: the hand pushes it). How far: `overhang_m` (along `direction` until the part sticks
    out this far past the support's edge; refused when its centre of mass would come within COM_MARGIN_M of the edge),
    `distance_m`, or `target_pos` (its root's xy); a drag's hand travels `travel_factor` (default 1.5) times as far, since
    it slips. The part is carried in the plan, so the next entries are planned
    with it where it should end; the physics decides. Expect: displaced along the direction, the overhang reached,
    upright, at rest, released, and no finger more than max_table_pen_m (2 mm) into the support."""
    m, d = sp.model, sp.pl.data
    if not obj:
        raise ValueError("slide: give `object` (the part to slide)")
    _no_hold(sp, side, "slide")
    b, bodies = part_bodies(m, obj)
    mujoco.mj_forward(m, d)
    if sg.get("direction") is None and sg.get("target_pos") is None:
        raise ValueError("slide: give `direction` ([dx, dy]: where the part goes) with overhang_m or distance_m, or target_pos")
    fp = footprint(m, d, bodies)
    tb = table_box(m, d)
    rect = _support_rect(tb) if tb is not None else None
    p_root = d.xpos[b].copy()
    if sg.get("target_pos") is not None:
        tgt = np.asarray(sg["target_pos"], float)
        dvec = _horizontal(tgt - p_root, "target_pos - the part")
        dist = float(np.linalg.norm((tgt - p_root)[:2]))
    else:
        dvec = _horizontal(sg["direction"], "direction")
        if sg.get("overhang_m") is not None:
            if rect is None:
                raise ValueError("slide: overhang_m needs the bench table (a body named 'table' with a box geom)")
            ov, _ = overhang_along(fp, rect, dvec[:2])
            dist = float(sg["overhang_m"]) - ov
        elif sg.get("distance_m") is not None:
            dist = float(sg["distance_m"])
        else:
            raise ValueError("slide: give overhang_m, distance_m or target_pos")
    if dist <= 0.002:
        raise ValueError(f"slide: nothing to do: the part is already {1000 * -dist:.0f} mm past where it was asked to go")
    if rect is not None:
        # the whole free object's centre of mass stays on the table (a strip's end may go over the edge, its knot not)
        jid = P.free_joint_of(m, obj)
        root = int(m.jnt_bodyid[jid]) if jid is not None else b
        com = d.subtree_com[root][:2]
        s_com, _ = exit_distance(rect, com, dvec[:2])
        if s_com - dist < COM_MARGIN_M:
            raise ValueError(f"slide: {1000 * dist:.0f} mm along {np.round(dvec[:2], 2).tolist()} would leave the centre of mass "
                             f"{1000 * (s_com - dist):.0f} mm from the support's edge (at least {1000 * COM_MARGIN_M:.0f} mm): it would tip off; "
                             f"slide at most {1000 * (s_com - COM_MARGIN_M):.0f} mm")
    pose = sp._pose(sg)
    mode = str(sg.get("mode") or pose.get("slide_mode") or "drag")
    if mode not in ("drag", "push"):
        raise ValueError("slide: mode must be 'drag' or 'push'")
    g = sp._grip_from_pose(side, pose)
    posed = dict(g["closed"])
    g.update({"press": posed, "opened": posed, "wrist_press": None, "own_open": True})
    sp.state["grip"][side] = g
    p, R = P.pose_from_dict(pose)
    a = P.retreat_axis(pose, R)
    approach_m = float(sg.get("approach_m", 0.06))
    mass = float(sum(m.body_mass[x] for x in bodies))
    mu_s = lying_geometry(m, d, obj)["mu_support"] if mode == "drag" else 1.0
    # the part lags a push by a few millimetres; a drag rides on the pads' friction margin over the support's and slips
    # (a strip end dragged with 74 mm of hand travel moved 49 mm): the hand goes travel_factor times as far
    factor = float(sg.get("travel_factor", 1.5 if (sg.get("mode") or pose_mode_hint(sg)) == "drag" else 1.0))
    hand_travel = dist * factor + float(sg.get("extra_m", 0.004))
    objs = [obj]
    sp.pl.fingers(side, posed); sp.pl.move_wrist(side, p + a * approach_m, R)
    sp.pl.push(f"{side}: reach for {obj}", float(sg.get("t_reach", 2.0)), 0.1, side, objs, anchor=obj)
    sp.pl.move_wrist(side, p, R); sp.pl.push(f"{side}: descend on {obj}", 0.8, 0.1, side, objs, anchor=obj)
    exp = {"object": obj}
    lab_slide = f"{side}: slide {obj}"
    if mode == "drag":
        need = drag_force_need(mass, HAND_MU, mu_s)
        if need is None:
            raise ValueError(f"slide: a drag cannot move it: the pads' friction {HAND_MU} does not exceed the support's {mu_s:.2f} "
                             "(push a side face instead: mode 'push' with a push_pose_from_point pose)")
        force = float(min(H.WRIST_FORCE_N, sg.get("force_n") or max(15.0, 2.5 * need)))
        if force < need * 1.2:
            raise ValueError(f"slide: a {force:.0f} N press drags at most {force * (HAND_MU - mu_s) / mu_s / 9.81:.2f} kg; "
                             f"this part ({mass:.3f} kg) needs {need:.1f} N (the wrist cap is {H.WRIST_FORCE_N:.0f} N)")
        nrm = pose.get("normals") or {}
        press_dir = -_unit(np.mean([np.asarray(v, float) for v in nrm.values()], axis=0)) if nrm else np.array([0.0, 0.0, -1.0])
        travel = float(sg.get("press_travel_m", force / 10000.0 + 0.004))
        sp.pl.move_wrist(side, p + press_dir * travel, R)
        sp.pl.push(f"{side}: press {obj}", 0.5, 0.2, side, objs, anchor=obj)
        try:
            sp.pl.attach(side, obj)                        # the plan carries it (the physics decides where it goes)
        except ValueError:
            pass
        start = p + press_dir * travel
        sp.pl.move_wrist(side, start + dvec * hand_travel, R)
        sp.pl.push(lab_slide, float(sg.get("duration_s", max(1.5, hand_travel / 0.04))), 0.3, side, objs, anchor=obj)
        sp.pl.detach(side)
        pw, _ = sp.pl.wrist(side)
        sp.pl.move_wrist(side, pw - press_dir * (travel + 0.01), R)
        sp.pl.push(f"{side}: lift off {obj}", 0.5, 0.1, side, objs, anchor=None)
        _press_caps(sp, side, press_dir, force, [f"{side}: press {obj}", lab_slide])
        exp["press"] = {"force_n": round(force, 1), "need_n": round(need, 2), "travel_m": round(travel, 4), "mu_support": round(mu_s, 3)}
    else:
        try:
            sp.pl.attach(side, obj)
        except ValueError:
            pass
        sp.pl.move_wrist(side, p + dvec * hand_travel, R)
        sp.pl.push(lab_slide, float(sg.get("duration_s", max(1.5, hand_travel / 0.04))), 0.3, side, objs, anchor=obj)
        sp.pl.detach(side)
        pw, _ = sp.pl.wrist(side)
        sp.pl.move_wrist(side, pw - dvec * 0.02, R); sp.pl.push(f"{side}: back off", 0.5, 0.1, side, objs, anchor=None)
    pw, _ = sp.pl.wrist(side)
    sp.pl.move_wrist(side, pw + a * approach_m, R); sp.pl.push(f"{side}: retreat", 0.8, 0.2, side, [], anchor=None)
    sp.state["held"][side] = None
    exp.update({"displacement": {"dir": dvec.round(6).tolist(), "min_m": round(dist, 4)}, "upright": bool(sg.get("upright", True)),
                "at_rest": True, "released": True, "max_table_pen_m": float(sg.get("max_table_pen_m", TABLE_PEN_M))})
    if sg.get("overhang_m") is not None:
        exp["overhang"] = {"dir": dvec[:2].round(6).tolist(), "min_m": round(float(sg["overhang_m"]) - max(0.004, 0.15 * dist), 4),
                           "com_margin_m": 0.0}
    return exp


def _current_overhang(sp, obj, edge_dir):
    m, d = sp.model, sp.pl.data
    b, bodies = part_bodies(m, obj)
    mujoco.mj_forward(m, d)
    tb = table_box(m, d)
    if tb is None:
        return None
    return overhang_along(footprint(m, d, bodies), _support_rect(tb), edge_dir)[0]


def m_edge_pinch(sp, sg, side, obj):
    """Grasp a flat part by its edge where there is room under it, and lift: the pose (overhang_pinch_pose) has the
    grip fingers under the part, deeper than the thumb on its top, and records the edge (`edge_pinch` {edge_dir,
    need_overhang_m | raised}). Before planning, the part's current overhang past the support's edge along edge_dir is
    measured: short of what the pose needs, the entry is refused with the slide that fixes it. The hand comes in from
    beyond the edge (the pose's approach waypoints: above and outside, down outside the edge, then in under the part),
    closes, and lifts `lift_m` (default 0.08). Expect: rise, in hand, no slip, and no finger more than
    max_table_pen_m into the support."""
    raw = dict(load_pose(sg["pose"]))
    pose = sp._pose(sg)
    pose = rotate_waypoints(sp.model, sp.pl.data, dict(pose, anchor=raw.get("anchor"))) if raw.get("approach_waypoints") else pose
    info = pose.get("edge_pinch") or {}
    if info.get("edge_dir") is not None and not info.get("raised"):
        cur = _current_overhang(sp, obj, info["edge_dir"])
        need = float(info.get("need_overhang_m", 0.0))
        if cur is not None and cur < need - 0.003:
            raise ValueError(f"edge_pinch: {obj} overhangs the support's edge by {1000 * cur:.0f} mm along {info['edge_dir']}; this pinch puts "
                             f"its fingers {1000 * float(info.get('finger_inset_m', 0)):.0f} mm under it and needs {1000 * need:.0f} mm: slide it first "
                             f"(motion slide, direction {info['edge_dir']}, overhang_m {need + 0.005:.3f})")
    sub = dict(sg, pose=pose, reanchor="none")             # _pose already re-anchored it
    sub.setdefault("approach_m", float(pose.get("approach_m", 0.10)))
    exp = sp.m_grasp_lift(sub, side, obj)
    exp["max_table_pen_m"] = float(sg.get("max_table_pen_m", TABLE_PEN_M))
    return exp


def m_tilt_up(sp, sg, side, obj):
    """Tilt a flat part up about a fulcrum by pressing one end: the part overhangs the support's edge (slide it there
    first) and the hand presses the overhanging end down (the pose: a fingertip or flat fingers on its top there,
    push_pose_from_point pointing down or flat_hand_pose). The part pivots about the edge and its far end rises: the
    press goes deep enough that the far end rises `lift_m` (default: a fingertip thickness + 8 mm, room for another
    hand's fingers), by the lever ratio of the press point and the far end about the edge; the wrist force on the press
    axis is capped at `force_n` (default 3x the moment the part's weight needs, within the 40 N cap). With `hold` (default
    true) the hand stays pressed at the end, so the next entry (the other hand) can pinch the raised end; a later
    `release` entry of this hand lets go. Expect: the far end raised at least 70 % of lift_m, the part still on its
    support, no finger more than max_table_pen_m into the support."""
    m, d = sp.model, sp.pl.data
    if not obj:
        raise ValueError("tilt_up: give `object`")
    _no_hold(sp, side, "tilt_up")
    b, bodies = part_bodies(m, obj)
    mujoco.mj_forward(m, d)
    tb = table_box(m, d)
    if tb is None:
        raise ValueError("tilt_up: needs the bench table (a body named 'table' with a box geom) as the fulcrum")
    rect = _support_rect(tb)
    fp = footprint(m, d, bodies)
    pose = sp._pose(sg)
    pts = pose.get("points") or ({"p": pose["point"]} if pose.get("point") is not None else None)
    press_pt = np.asarray(sg["press_point"], float) if sg.get("press_point") is not None else (
        np.mean([np.asarray(v, float) for v in pts.values()], axis=0) if pts else None)
    if press_pt is None:
        raise ValueError("tilt_up: give press_point (world xyz on the part's top) or a pose that records its contact (point / points)")
    # the fulcrum: the support edge the press point lies beyond
    if sg.get("edge_dir") is not None:
        e = _horizontal(sg["edge_dir"], "edge_dir")[:2]
    else:
        cands = []
        for n in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            s, _ = exit_distance(rect, fp["centre"], n)
            cands.append((overhang_along(fp, rect, n)[0], n))
        e = _unit([*max(cands)[1], 0.0])[:2]
    s_edge, _ = exit_distance(rect, fp["centre"], e)
    edge_c = fp["centre"] + e * s_edge                      # a point on the pivot line
    lever_press = float((press_pt[:2] - edge_c) @ e)        # > 0: beyond the edge
    lever_far = float(((fp["points"][:, :2] - edge_c) @ -e).max())
    if lever_press <= 0.003:
        raise ValueError(f"tilt_up: the press point lies {1000 * -lever_press:.0f} mm inside the support's edge: press the overhanging end "
                         f"(beyond the edge along {np.round(e, 2).tolist()}); slide the part to overhang first")
    dims = hand_dims(m, sp.pl.rigs[side])
    lift = float(sg.get("lift_m", dims["tip_thickness_m"] + 0.008))
    ang = np.arcsin(min(0.95, lift / max(lever_far, 1e-6)))
    depth = lever_press * np.sin(ang)
    mass = float(sum(m.body_mass[x] for x in bodies))
    com_in = float((edge_c - d.subtree_com[b][:2]) @ e)      # the centre of mass inside the edge (> 0)
    need = 9.81 * mass * max(com_in, 0.0) / lever_press
    force = float(min(H.WRIST_FORCE_N, sg.get("force_n") or max(4.0, 3.0 * need)))
    if force < 1.1 * need:
        raise ValueError(f"tilt_up: pressing {1000 * lever_press:.0f} mm beyond the edge needs {need:.1f} N to lift the far end, over the "
                         f"{force:.0f} N cap: press farther out")
    g = sp._grip_from_pose(side, pose)
    posed = dict(g["closed"])
    g.update({"press": posed, "opened": posed, "wrist_press": None, "own_open": True})
    sp.state["grip"][side] = g
    p, R = P.pose_from_dict(pose)
    a = P.retreat_axis(pose, R)
    approach_m = float(sg.get("approach_m", 0.06))
    objs = [obj]
    sp.pl.fingers(side, posed); sp.pl.move_wrist(side, p + a * approach_m, R)
    sp.pl.push(f"{side}: reach for {obj}", float(sg.get("t_reach", 2.0)), 0.1, side, objs, anchor=obj)
    sp.pl.move_wrist(side, p, R); sp.pl.push(f"{side}: descend on {obj}", 0.8, 0.1, side, objs, anchor=obj)
    down = np.array([0.0, 0.0, -1.0])
    travel = depth + force / 10000.0 + float(sg.get("extra_m", 0.003))
    lab = f"{side}: tilt {obj} up"
    sp.pl.move_wrist(side, p + down * travel, R)
    sp.pl.push(lab, float(sg.get("duration_s", 1.2)), float(sg.get("hold_s", 0.6)), side, objs, anchor=obj)
    _press_caps(sp, side, down, force, [lab])
    if not sg.get("hold", True):
        sp.pl.move_wrist(side, p + a * approach_m, R); sp.pl.push(f"{side}: back off", 0.8, 0.2, side, objs, anchor=None)
        sp._rollout["wrist_force_n"][side]["phases"].append(f"{side}: back off")
    else:
        # the press stays on: the executor carries this hand's commands into the next entry, and the cap with it
        sp.state.setdefault("pressing", {})[side] = {"object": obj, "force_n": force}
    sp.state["held"][side] = None
    far_pt = fp["points"][int(np.argmax((fp["points"][:, :2] - edge_c) @ -e))]
    R_b = d.xmat[b].reshape(3, 3)
    far_local = (R_b.T @ (far_pt - d.xpos[b])).round(6).tolist()
    exp = {"object": obj, "end_raised": {"point_local": far_local, "min_m": round(0.7 * lift, 4), "planned_m": round(lift, 4),
                                          "z0": round(float(far_pt[2]), 5)},
           "tilt_deg": round(float(np.degrees(ang)), 2), "fraction": 0.6, "on_support": True,
           "press": {"force_n": round(force, 2), "need_n": round(need, 2), "depth_m": round(depth, 4), "lever_press_m": round(lever_press, 4),
                     "lever_far_m": round(lever_far, 4), "edge_dir": np.round(e, 4).tolist()},
           "max_table_pen_m": float(sg.get("max_table_pen_m", TABLE_PEN_M)), "min_links": 1}
    return exp


def m_hook(sp, sg, side, obj):
    """Hook a part through an opening (a ring, a strap, a handle) and lift: the pose (hook_grasp_pose) puts one or
    more fingers down through the opening and the thumb, when the wall is tall enough, on its outside; it comes from
    straight above (its approach), the hooking fingers flex against the inner wall, the thumb closes on the outer one,
    and the hand lifts `lift_m`. A pure hook (no thumb in grip_fingers) first pulls `pull_m` along the pose's
    pull_dir while rising, so the part tips up onto the hooked fingers, then lifts. Expect: rise, in hand (one link for
    a pure hook), no slip, no finger more than max_table_pen_m into the support."""
    raw = dict(load_pose(sg["pose"]))
    pose = sp._pose(sg)
    pose = rotate_waypoints(sp.model, sp.pl.data, dict(pose, anchor=raw.get("anchor"))) if raw.get("approach_waypoints") else pose
    grip = list(pose.get("grip_fingers") or [])
    sub = dict(sg, pose=pose, reanchor="none")
    sub.setdefault("approach_m", float(pose.get("approach_m", 0.06)))
    if "thumb" not in grip and pose.get("pull_dir") is not None and sg.get("lift_path") is None:
        pd = _horizontal(pose["pull_dir"], "pull_dir")
        pull = float(sg.get("pull_m", pose.get("pull_m", 0.012)))
        lift = float(sg.get("lift_m", 0.08))
        sub["lift_path"] = [(pd * pull + np.array([0, 0, 0.004])).tolist(), (pd * pull + np.array([0, 0, 0.5 * lift])).tolist(),
                            (pd * pull + np.array([0, 0, lift])).tolist()]
    exp = sp.m_grasp_lift(sub, side, obj)
    if "thumb" not in grip:
        exp["min_links"] = 1
    exp["max_table_pen_m"] = float(sg.get("max_table_pen_m", TABLE_PEN_M))
    return exp


def m_strip_pinch(sp, sg, side, obj):
    """Lift a flat strip (a box-link ribbon, a flat cable) where it arches over something: the pose (a pinch across the arch: handle_pinch_pose,
    pose_from_points) has the thumb on the strip and the grip fingertip in the gap under the arch; the hand comes down beside the strip,
    slides the finger in under it (the pose's approach waypoints), closes and lifts `lift_m` (default 0.06). `object`
    is the link the outcome follows (the free end's link, to see the end rise). Expect: rise, in hand, no slip, no
    finger more than max_table_pen_m into the table. A strip lying flat has no such pose (thin_grasp_diagnosis says so)."""
    raw = dict(load_pose(sg["pose"]))
    pose = sp._pose(sg)
    pose = rotate_waypoints(sp.model, sp.pl.data, dict(pose, anchor=raw.get("anchor"))) if raw.get("approach_waypoints") else pose
    sub = dict(sg, pose=pose, reanchor="none")
    sub.setdefault("approach_m", float(pose.get("approach_m", 0.08)))
    sub.setdefault("lift_m", 0.06)
    exp = sp.m_grasp_lift(sub, side, obj)
    exp["max_table_pen_m"] = float(sg.get("max_table_pen_m", TABLE_PEN_M))
    return exp


_MOTION_FUNCS = {"slide": m_slide, "edge_pinch": m_edge_pinch, "overhang_grasp": m_edge_pinch, "tilt_up": m_tilt_up,
                 "pivot": m_tilt_up, "hook": m_hook, "strip_pinch": m_strip_pinch}


def install(cls):
    """Add the motions above to a planner class as m_<motion> methods; returns their names (for MOTIONS)."""
    for name, fn in _MOTION_FUNCS.items():
        if not hasattr(cls, f"m_{name}"):
            setattr(cls, f"m_{name}", (lambda f: (lambda self, sg, side, obj: f(self, sg, side, obj)))(fn))
    return tuple(_MOTION_FUNCS)


# ----------------------------------------------------------------------------- what these motions must have achieved
EXPECT_KEYS = ("overhang", "end_raised", "max_table_pen_m", "on_support")


def _support_geoms(model):
    tb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    out = [g for g in range(model.ngeom) if (tb >= 0 and int(model.geom_bodyid[g]) == tb) or
           (int(model.geom_bodyid[g]) == 0 and int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE))]
    return out


def evaluate_extra(model, Q, t0, t1, expect, rig, checks, meas):
    """The expectations of the thin-part motions, measured on the executed frames t0..t1 of Q (called by
    outcome.evaluate): overhang {dir, min_m} (the part sticks out past the table's edge along dir by at least min_m at
    the end, its centre of mass still over the table), end_raised {point_local, min_m, z0} (that point of the part,
    in its own frame, ends at least min_m above where it started), on_support (the part's centre of mass is over the
    table at the end: it did not tip off), max_table_pen_m (no collision geom of the acting hand deeper than this into
    the table on any frame)."""
    d = mujoco.MjData(model)
    obj = expect.get("object")
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj) if obj else -1
    tb = table_box(model, d)
    if bid >= 0 and (expect.get("overhang") or expect.get("on_support")):
        d.qpos[:] = Q[t1]; mujoco.mj_forward(model, d)
        bodies = E.subtree_bodies(model, bid)
        fp = footprint(model, d, bodies)
        com = np.asarray(d.subtree_com[bid], float)
        if tb is not None:
            rect = _support_rect(tb)
            c, ex, ey, hx, hy, top = rect
            inside = min(hx - abs(float((com[:2] - c) @ ex)), hy - abs(float((com[:2] - c) @ ey)))
            meas["com_inside_table_m"] = round(inside, 4)
            if expect.get("on_support"):
                checks["on_support"] = inside > 0.0 and com[2] > top - 0.02
            ov = expect.get("overhang")
            if ov:
                got, _ = overhang_along(fp, rect, ov["dir"])
                meas["overhang_m"] = round(got, 4)
                checks["overhang"] = got >= float(ov["min_m"]) and inside >= float(ov.get("com_margin_m", 0.0))
    er = expect.get("end_raised")
    if er and bid >= 0:
        d.qpos[:] = Q[t1]; mujoco.mj_kinematics(model, d)
        pt = d.xpos[bid] + d.xmat[bid].reshape(3, 3) @ np.asarray(er["point_local"], float)
        rise = float(pt[2] - float(er["z0"]))
        meas["end_raised_m"] = round(rise, 4)
        checks["end_raised"] = rise >= float(er["min_m"])
    lim = expect.get("max_table_pen_m")
    if lim is not None and rig is not None:
        sup = set(_support_geoms(model))
        hand = set(rig.collision_gids)
        worst, at = 0.0, None
        for t in range(t0, t1 + 1, 2):
            d.qpos[:] = Q[t]; mujoco.mj_forward(model, d)
            for i in range(d.ncon):
                c = d.contact[i]
                g1, g2 = int(c.geom1), int(c.geom2)
                if ((g1 in hand and g2 in sup) or (g2 in hand and g1 in sup)) and -float(c.dist) > worst:
                    worst, at = -float(c.dist), t
        meas["max_table_pen_mm"] = round(1000 * worst, 2)
        if at is not None:
            meas["max_table_pen_frame"] = int(at)
        checks["table_clear"] = worst <= float(lim)


# ----------------------------------------------------------------------------- hand scenes for these poses
def shift_object(spec, dxyz):
    """A copy of a hand-scene dict with its last object (the grasped one) moved by dxyz (its entry's pos and any free
    joint of it in joint_init): a pose solved for where a slide will leave the part."""
    spec = copy.deepcopy(spec)
    o = spec["objects"][-1]
    o["pos"] = (np.asarray(o.get("pos", (0, 0, 0)), float) + np.asarray(dxyz, float)).tolist()
    pre = f"{o['name']}_"
    for k, v in list((spec.get("joint_init") or {}).items()):
        if k.startswith(pre) and isinstance(v, (list, tuple)) and len(v) == 7:
            spec["joint_init"][k] = (np.asarray(v[:3], float) + np.asarray(dxyz, float)).tolist() + list(v[3:])
    return spec


def body_anchor(model, data, bid):
    q = np.zeros(4); mujoco.mju_mat2Quat(q, data.xmat[bid])
    return {"body": model.body(bid).name, "pos": [round(float(v), 6) for v in data.xpos[bid]], "quat_wxyz": [round(float(v), 8) for v in q]}


def rotate_waypoints(model, data, pose):
    """The pose's approach_waypoints (world offsets) turned with its part since the pose was solved (reanchor_pose turns
    pos, euler and approach but leaves waypoints as given)."""
    from scipy.spatial.transform import Rotation as Rot
    wps = pose.get("approach_waypoints")
    anc = pose.get("anchor")
    if not wps or not anc:
        return pose
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, anc["body"])
    if b < 0:
        return pose
    mujoco.mj_kinematics(model, data)
    q = anc["quat_wxyz"]
    dR = data.xmat[b].reshape(3, 3) @ Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix().T
    out = dict(pose)
    out["approach_waypoints"] = [(dR @ np.asarray(w, float)).tolist() for w in wps]
    return out


def tip_cap_offset(model, rig, finger):
    """How far the centre of the fingertip's distal cap lies beyond the fingertip site, along the finger (knuckle to
    site): an upright fingertip touches a low wall with that cap, not at its site (the Sharpa index: 6 mm; its visible
    tip reaches 12.4 mm past the site). Read from the tip link's capsule."""
    d = mujoco.MjData(model)
    rig.teleport(d, np.zeros(3), np.zeros(3))
    mujoco.mj_kinematics(model, d)
    s = rig.fingertip_sites[finger]
    sp = np.asarray(d.site_xpos[s], float)
    ax = _unit(sp - rig.knuckle_point(d, finger))
    best = 0.0
    for g in rig.tip_geoms(finger):
        if int(model.geom_type[g]) in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
            gz = np.asarray(d.geom_xmat[g], float).reshape(3, 3)[:, 2]
            hl = float(model.geom_size[g][1])
            c = np.asarray(d.geom_xpos[g], float)
            best = max(best, float((c + gz * hl - sp) @ ax), float((c - gz * hl - sp) @ ax))
    return best


class cap_sites:
    """Within the block, each named finger's fingertip site sits at the centre of its distal cap (model.site_pos moved in
    the tip link's frame, restored on exit): a fingertip touching a low wall or a strip touches with that cap, so the
    fingertip IK solves for the cap whatever the finger's tilt."""

    def __init__(self, model, rig, fingers):
        self.model, self.rig, self.fingers = model, rig, list(fingers)
        self.saved = {}

    def __enter__(self):
        m, rig = self.model, self.rig
        for f in self.fingers:
            s = rig.fingertip_sites.get(f)
            if s is None:
                continue
            b = int(m.site_bodyid[s])
            best = None
            for g in rig.tip_geoms(f):
                if int(m.geom_bodyid[g]) != b or int(m.geom_type[g]) not in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
                    continue
                q = m.geom_quat[g]; Rg = np.zeros(9); mujoco.mju_quat2Mat(Rg, q); Rg = Rg.reshape(3, 3)
                hl = float(m.geom_size[g][1])
                for sgn in (1.0, -1.0):
                    end = np.asarray(m.geom_pos[g], float) + Rg[:, 2] * hl * sgn
                    # the distal end: the one farther from the link's origin (the joint it turns about)
                    if best is None or np.linalg.norm(end) > np.linalg.norm(best):
                        best = end
            if best is not None:
                self.saved[s] = m.site_pos[s].copy()
                m.site_pos[s] = best
        return self

    def __exit__(self, *a):
        for s, v in self.saved.items():
            self.model.site_pos[s] = v
        return False


def upright_tip_targets(model, rig, pts, nrm, up=(0.0, 0.0, 1.0)):
    """Fingertip-site targets for fingertips pointing down onto low contacts: each pad's distal cap centre a pad radius
    (+1 mm) out from its contact along the normal, the site that cap offset above it (tip_cap_offset)."""
    so = H.pad_radius(model, rig, "index") + 0.001
    upv = np.asarray(up, float)
    return {f: np.asarray(pts[f], float) + np.asarray(nrm[f], float) * so + upv * tip_cap_offset(model, rig, f) for f in pts}


def press_into(rig, pose, grip, nrm):
    """{finger: how far (m) the executor's clamp moves each grip pad into its own surface (along minus its contact
    normal)}: a pad that touches with the back of its finger, or that flexion swings sideways, moves away (<= 0) and
    never squeezes."""
    c = H.executor_clamp(rig, dict(pose, grip_fingers=list(grip), normals={f: list(map(float, v)) for f, v in nrm.items()}))
    sh = H.pad_shift_world(rig, c["closed"], c["press"], pose["euler"])
    return {f: float(sh[f] @ -_unit(nrm[f])) for f in grip if f in sh and f in nrm}


_ARM_SOLVERS = {}
_ARM_SEEDS = {}


def arm_seeds(bench_scene, side, model, d0, rig, centre, max_mm=20.0, max_deg=12.0):
    """Wrist seeds the station's arm can hold: hand orientations over a grid of finger directions (pointing down to
    level, every 30 deg about the vertical) and palm rolls, each placed so the hand's grasp pocket sits at `centre`,
    solved with the executor's arm IK on the bench; the reachable ones (residual within max_mm / max_deg), best first.
    The fingertip IK seeded from them stays near orientations the arm holds: at a bench edge an arm reaches a few
    hand headings only, and an IK seeded everywhere returns poses the arm screen then drops."""
    from tool_library.skills.optimize_pose_for_reach import _reach_solver
    key = (str(bench_scene), side)
    if key not in _ARM_SOLVERS:
        bm = mujoco.MjModel.from_xml_path(str(bench_scene))
        _ARM_SOLVERS[key] = _reach_solver(bm, side)
    solve = _ARM_SOLVERS[key]
    ck = key + (tuple(np.round(np.asarray(centre, float), 2)), float(max_mm), float(max_deg))
    if ck in _ARM_SEEDS:                      # the same place again (another wall, another contact height)
        return list(_ARM_SEEDS[ck])
    pk = H.grasp_pocket(model, d0, rig)
    out = []
    from scipy.spatial.transform import Rotation as Rot
    for el in (-90.0, -65.0, -40.0, -15.0, 10.0):
        for az in (range(0, 360, 30) if el > -89 else (0,)):
            f_w = np.array([np.cos(np.radians(el)) * np.cos(np.radians(az)), np.cos(np.radians(el)) * np.sin(np.radians(az)), np.sin(np.radians(el))])
            ref = np.array([0.0, 0.0, 1.0]) if abs(f_w[2]) < 0.95 else np.array([1.0, 0.0, 0.0])
            n0 = _unit(np.cross(f_w, np.cross(ref, f_w)))
            for roll in (range(0, 360, 45) if el <= -89 else range(0, 360, 90)):
                n_w = Rot.from_rotvec(np.radians(roll) * f_w).as_matrix() @ n0
                pos, eul = H.pocket_pose(pk, centre, n_w, f_w)
                ep, er, _ = solve(pos, H.euler_to_mat(eul))
                if ep * 1000 <= max_mm and np.degrees(er) <= max_deg:
                    out.append((ep + 0.001 * np.degrees(er), (np.asarray(pos, float), np.asarray(eul, float))))
    out.sort(key=lambda t: t[0])
    _ARM_SEEDS[ck] = [sd for _, sd in out]
    return list(_ARM_SEEDS[ck])


def solve_sector_poses(spec, built, pts, nrm, site_targets, approach_dirs=((0.0, 0.0, 1.0),), sectors=4, yaw_step_deg=30,
                       pitches_deg=(-60, -30, 0, 30, 60), iters=200, max_residual_mm=3.0, max_overlap_mm=2.0, grip=None, caps=False,
                       arm=None, extra_seeds=None):
    """pose_from_points with its wrist seeds split into `sectors` sectors of yaw about the vertical, one pose per sector
    (an arm reaches some hand headings only; the executor screens the candidates). The fingertip sites go to
    `site_targets` (e.g. upright_tip_targets); `pts` / `normals` are the true contacts and stay recorded as the pose's
    points (the outcome's pad check). With `arm` = (bench scene, side) the sectors are solved first from the seeds the
    station's arm holds (arm_seeds) with their orientation kept (fix_orientation: the fingertip IK moves the wrist's
    position and the fingers only; a free orientation drifted 40-50 deg into poses the arm could not hold), then from
    the generic sweep. With `grip` (thumb first) an opposed pose is kept only when the executor's clamp
    moves no grip pad off its own surface and closes the pads by 2 mm in all (press_into; one pad may be the anvil). Returns (accepted poses ranked, every tried {sector, worst_residual_mm,
    overlap_mm, ok})."""
    from tool_library.skills.pose_from_points import pose_from_points
    model, d0, rig, ids = built
    so = H.pad_radius(model, rig, "index") + 0.001
    if caps:                                   # solve for the distal caps: their centres a pad radius off the contacts
        site_targets = {f: np.asarray(pts[f], float) + _unit(nrm[f]) * so for f in pts}
    shifted = {f: (np.asarray(site_targets[f], float) - _unit(nrm[f]) * so).tolist() for f in pts}
    cen = np.mean([np.asarray(v, float) for v in site_targets.values()], axis=0)
    seeds = H.seed_sweep(cen, [list(a) for a in approach_dirs], yaw_step_deg=yaw_step_deg, pitches_deg=tuple(pitches_deg))
    families = [(seeds, False)]
    if arm is not None and arm[0] and Path(str(arm[0])).exists():
        # first the seeds the station's arm holds (arm_seeds), their orientation kept (a generic seed that fits the hand
        # scene better would win its sector with a pose the arm then drops), then the generic sweep
        aseeds = arm_seeds(arm[0], arm[1], model, d0, rig, cen)
        if aseeds:
            families = [(aseeds, True), (seeds, False)]
    if extra_seeds:                              # the caller's own seeds (a grip whose hand orientation it knows)
        families.insert(0, (list(extra_seeds), False))
    width = 360.0 / int(sectors)
    out, tried = [], []
    for k, (fam, fix_orient) in [(k, f) for f in families for k in range(int(sectors))]:
        sub = [s for s in fam if k * width <= (np.degrees(s[1][2]) % 360.0) < (k + 1) * width]
        if not sub:
            continue
        if caps:
            with cap_sites(model, rig, list(pts)):
                pz = pose_from_points(spec, shifted, {f: list(map(float, nrm[f])) for f in pts}, standoff_m=so, seeds=sub, iters=int(iters),
                                      clear_overlap=False, fix_orientation=fix_orient, _built=built)
            pz["cap_targets"] = {f: list(map(float, v)) for f, v in site_targets.items()}
        else:
            pz = pose_from_points(spec, shifted, {f: list(map(float, nrm[f])) for f in pts}, standoff_m=so, seeds=sub, iters=int(iters),
                                  clear_overlap=False, fix_orientation=fix_orient, _built=built)
        rec = {"sector_deg": [k * width, (k + 1) * width], "arm_seeded": bool(fix_orient), "worst_residual_mm": pz["worst_residual_mm"], "overlap_mm": pz["overlap_mm"]}
        ok = pz["worst_residual_mm"] <= max_residual_mm and pz["overlap_mm"] <= max_overlap_mm
        if ok and grip:
            # an opposed grip squeezes only when the clamp drives every pad into its own surface
            into = press_into(rig, pz, grip, nrm)
            rec["press_into_mm"] = {f: round(1000 * v, 2) for f, v in into.items()}
            # every pad stays on its surface (one may be the anvil the others press the part against) and together
            # they close by at least 2 mm
            if min(into.values(), default=-1.0) < -0.0005 or sum(into.values()) < 0.002:
                ok = False
                rec["why"] = "a pad touches with the back of its finger or swings off its surface when the grip clamps"
        rec["ok"] = bool(ok)
        tried.append(rec)
        if ok:
            pz["points"] = {f: list(map(float, pts[f])) for f in pts}
            pz["sector_deg"] = rec["sector_deg"]
            out.append(pz)
    out.sort(key=lambda p: (round(p["worst_residual_mm"]), p["overlap_mm"]))
    return out, tried


def _hand_at(model, d0, rig, pose, fingers_q, off=(0.0, 0.0, 0.0)):
    dd = mujoco.MjData(model)
    dd.qpos[:] = d0.qpos
    rig.teleport(dd, np.asarray(pose["pos"], float) + np.asarray(off, float), pose["euler"]); rig.set_fingers(dd, fingers_q)
    mujoco.mj_forward(model, dd)
    return dd


OPEN_LADDER = (0.55, 0.45, 0.35, 0.25, 0.15)   # the executor's openings, widest first (SubgoalPlanner._floor_clear_open)


def executor_keeps_open(model, d0, rig, ids, pose, fingers_q, frac):
    """Whether the executor reaches the pose with this rung of its opening ladder: it keeps the widest rung whose open
    fingers are out of the table (floor_clear_at) and the last rung whatever it measures."""
    return float(frac) <= OPEN_LADDER[-1] + 1e-9 or floor_clear_at(model, d0, rig, ids, pose, fingers_q)


def floor_clear_at(model, d0, rig, ids, pose, fingers_q):
    """The executor's test of an opening (SubgoalPlanner._floor_clear_open): every collision geom of the hand at the
    pose with these fingers no more than 2 mm into the floor."""
    fl = ids.get("floor_gid", -1)
    if fl < 0:
        return True
    dd = _hand_at(model, d0, rig, pose, fingers_q)
    return min(H.geom_distance(model, dd, g, fl, 0.05) for g in rig.collision_gids) > -0.002


def part_overlap(model, dd, rig, ids):
    """Deepest overlap (m) of the hand's collision geoms, pads included, with the scene's objects (not the floor)."""
    worst = 0.0
    for g in rig.collision_gids:
        for og in ids["obj_gids"]:
            worst = max(worst, -float(mujoco.mj_geomDistance(model, dd, g, og, 0.0, None)))
    return worst


def sweep_into_part(model, d0, rig, ids, pose, closed, opened, slide_offsets=()):
    """None when the open hand sliding in (world offsets from the pose) and the fingers closing from `opened` to
    `closed` at the pose stay out of the part, no deeper than the closed pose itself grazes it + 0.5 mm; else why.
    A thumb opened 55 % over a 12 mm box curled into its end face and shoved it 14 mm before the pads met it."""
    tol = part_overlap(model, _hand_at(model, d0, rig, pose, closed), rig, ids) + 0.0005
    w = max([part_overlap(model, _hand_at(model, d0, rig, pose, opened, o), rig, ids) for o in slide_offsets] + [0.0])
    if w > tol:
        return f"the slide in sweeps the open hand {1000 * w:.1f} mm into the part"
    w = max(part_overlap(model, _hand_at(model, d0, rig, pose, {k: opened[k] + a * (closed[k] - opened[k]) for k in closed}), rig, ids)
            for a in (0.25, 0.5, 0.75))
    if w > tol:
        return f"closing from the open hand swings a finger {1000 * w:.1f} mm through the part"
    return None


def pick_open_frac(model, d0, rig, ids, pose, grip, path_offsets=()):
    """The opening a pose is reached with, as the executor opens it (give it as the pose's open_frac: the executor
    opens that far, narrower where the open fingers would be in the table at the pose): the widest rung of its ladder
    the executor keeps (executor_keeps_open) whose open hand on its way in (`path_offsets`, world offsets from the
    pose) is clear of the part and the table (visible meshes included, 1 mm) and whose close does not swing a finger
    through the part (sweep_into_part). Returns (frac or None, why)."""
    closed = {k: float(v) for k, v in pose["finger_targets"].items()}
    why = "no opening"
    for frac in OPEN_LADDER:
        op = H.approach_open_targets(rig, closed, float(frac), list(grip), False)
        if not executor_keeps_open(model, d0, rig, ids, pose, op, frac):
            why = "the open fingers are in the table at the pose"
            continue
        w = max([H.deepest_overlap(model, _hand_at(model, d0, rig, pose, op, o), rig, ids, keep_tips=(), visual_floor=True)
                 for o in path_offsets] + [0.0])
        if w > 0.001:
            why = f"the open hand on its way in is {1000 * w:.1f} mm in the part or the table"
            continue
        sw = sweep_into_part(model, d0, rig, ids, pose, closed, op)
        if sw:
            why = sw
            continue
        return float(frac), None
    return None, why


def lift_screen(spec, poses, lift_heights=(0.06, 0.10, 0.15), phases=("approach", "close", "lift", "move", "turn", "hold")):
    """Each pose's stress rollout in its hand scene, the grasp_stress_test ladder (hand.stress_rollout: the executor's
    approach and clamp, realistic contact; lifts of 6, 10 and 15 cm, 4 cm carries in five directions, 25 deg turns
    about three axes, the hold): [(ok, first_fail, achieved lift mm)] in order. The pose tools drop the candidates that
    fail, so the executor only ever falls back on grasps that pass the stress tests in the hand scene (an edge pinch
    that lifted 4 cm but let the part turn 60 deg in the hand on the bench passed a lift alone)."""
    model, d0, rig, ids = H.open_scene(spec)
    bid = H.grasp_body(model, spec, ids)
    out = []
    for p in poses:
        r = H.stress_rollout(model, d0, rig, ids, p, bid, phases=tuple(phases), lift_heights=tuple(lift_heights))
        out.append((bool(r["ok"]), r["first_fail"], r["lifts"][-1]["achieved_mm"] if r["lifts"] else None))
    return out


def arm_screen(bench_scene, side, poses, lift_m=0.08, approach_m=None):
    """Which poses the station's arm holds on the bench (armhand.screen_poses at the pre-grasp, the grasp, every
    approach waypoint and the lift, with its links checked against the scene): [(ok, residual_mm, reason)] in order.
    Poses are solved in a hand scene that knows nothing of the arm; the bench's arms reach a band in front of them, so
    the pose tools screen their candidates before the expensive stress rollouts."""
    from tool_library.common import armhand as AH
    model = mujoco.MjModel.from_xml_path(str(bench_scene))
    d = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, d, 0)
    probe = AH.ArmProbe(str(bench_scene))
    out = []
    for p in poses:
        wps = [list(map(float, w)) for w in (p.get("approach_waypoints") or [])] + [[0.0, 0.0, float(lift_m)]]
        r = AH.screen_poses(model, side, [p], d.qpos.copy(), probe=probe, approach_m=float(approach_m or p.get("approach_m", 0.08)), waypoints=wps)[0]
        out.append((bool(r["ok"]), r.get("residual_mm"), r.get("reason")))
    return out
