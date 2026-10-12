"""A CUTTABLE ROPE laid on its measured centreline.

THE POINT
---------
MuJoCo cannot break a body chain at run time, so a rope that the task cuts is
authored as TWO elastic cables laid end to end and tied by one CONNECT
equality at the cut point (scissors.attach_cuttable_thread); deactivating that
equality IS the cut, and scissors.try_sever only does it when the blades have
really closed on the join. This module builds that rope ALONG A MEASURED
CENTRELINE (measure_rope's polyline, a hand trace back-projected to the table,
any list of world points in metres):

  * the polyline is split at the cut location -- an arc-length fraction or the
    point of the rope nearest a given world point -- and each half is built
    along its own piece (cable `rest_vertices`), so both halves end exactly on
    the cut point: the rope is continuous at the join in the reference pose
    the CONNECT is baked from, and the compiled scene (qpos0 and every
    keyframe stage_on_bench writes) already has the rope where the video has
    it, with no keyframe surgery;
  * where the measured centreline crosses itself (a loop lying flat), the
    later strand is lifted over the earlier one by a smooth bump of one rope
    diameter, so the self-colliding capsules do not start interpenetrated;
  * the asset is ONE MJCF holding both cables, the join and the exclude that
    keeps the two capsules at the join from colliding with each other, so the
    bench staging route takes it exactly like construct_rope's: stage it with
    x = y = z = 0 and yaw 0 (the points are already in the rig frame).

The Stage 2 manifest fields Stage 3 needs (`cut_joins`, `halves`, the scene
names of every body) come out of `scene_fields(built, scene_name)`, already
prefixed the way stage_on_bench names an asset `scene_name`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import cable as ROPE
from .mjcf import FLOOR_XML, LIGHTS_XML, merge_params, mjcf_header, save_params

DEFAULTS = dict(ROPE.DEFAULTS)
DEFAULTS.update({
    "name": "rope",
    # capsules for the WHOLE rope; each half gets its share by length (at
    # least `min_half_segments`). A capsule much longer than four rope
    # diameters bends visibly polygonal on a tight loop.
    "segments": 48,
    "min_half_segments": 3,
    "grips": False,              # no mocap grips in a bench scene: hands grasp it
    "include_floor": False,      # the bench has the table
    # the measured centreline (world/rig metres, >= 2 points)
    "centreline": None,
    # WHERE IT IS CUT: an arc-length fraction from the first point, or the
    # point of the rope nearest this world point (wins when given)
    "cut_frac": 0.5,
    "cut_point": None,
    # every point's height, metres; None keeps the polyline's own z. A rope
    # lying on the table has its centreline one radius up.
    "z": None,
    # lift the later strand over the earlier one where the polyline crosses
    # itself (True), or leave the crossings flat (False)
    "lift_crossings": True,
    # the CONNECT that is the cut: as stiff as a cable joint, so the rope does
    # not stretch at the join before it is cut
    "join_solref": [0.004, 1.0],
})
# computed here from the centreline: a straight cable's length, start and heading mean nothing for a laid rope
for _k in ("rest_vertices", "length", "pos", "yaw_deg"):
    DEFAULTS.pop(_k, None)


def load_centreline(src) -> np.ndarray:
    """A centreline from a list of points, a dict holding one (keys points, centreline, centreline_rig,
    centreline_m, centreline_table, polyline, path) or a JSON file with either. Returns (k, 3) metres."""
    if isinstance(src, (str, Path)):
        src = json.loads(Path(src).read_text())
    if isinstance(src, dict):
        for k in ("points", "centreline", "centreline_rig", "centreline_m", "centreline_world", "polyline", "path",
                  "centreline_table"):
            if src.get(k) is not None:
                src = src[k]
                break
        else:
            raise ValueError(f"centreline: no point list in the dict (keys {sorted(src)[:12]})")
    P = np.asarray(src, float)
    if P.ndim != 2 or P.shape[1] not in (2, 3) or len(P) < 2:
        raise ValueError("centreline: need >= 2 points of [x, y] or [x, y, z] in metres")
    if P.shape[1] == 2:
        P = np.concatenate([P, np.zeros((len(P), 1))], axis=1)
    keep = [0] + [i for i in range(1, len(P)) if np.linalg.norm(P[i] - P[i - 1]) > 1e-6]
    return P[keep]


def _arclen(P):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])


def point_at(P, s):
    """The point of polyline P at arc length s."""
    S = _arclen(P)
    s = float(np.clip(s, 0.0, S[-1]))
    i = int(np.clip(np.searchsorted(S, s, side="right") - 1, 0, len(P) - 2))
    u = (s - S[i]) / max(S[i + 1] - S[i], 1e-12)
    return P[i] + u * (P[i + 1] - P[i])


def nearest_s(P, q):
    """Arc length of the point of polyline P nearest the world point q (3-D distance)."""
    S = _arclen(P)
    q = np.asarray(q, float)
    best = (np.inf, 0.0)
    for i in range(len(P) - 1):
        a, b = P[i], P[i + 1]
        d = b - a
        u = float(np.clip((q - a) @ d / max(d @ d, 1e-12), 0.0, 1.0))
        dist = float(np.linalg.norm(a + u * d - q))
        if dist < best[0]:
            best = (dist, S[i] + u * (S[i + 1] - S[i]))
    return best[1], best[0]


def resample(P, s0, s1, n):
    """n + 1 points from arc length s0 to s1 of polyline P, equally spaced in arc length."""
    return np.stack([point_at(P, s) for s in np.linspace(s0, s1, int(n) + 1)])


def _seg_cross(a, b, c, d):
    """2-D proper intersection of segments ab and cd: (t on ab, u on cd) or None."""
    r, s = b - a, d - c
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-12:
        return None
    q = c - a
    t = (q[0] * s[1] - q[1] * s[0]) / den
    u = (q[0] * r[1] - q[1] * r[0]) / den
    if 0.0 < t < 1.0 and 0.0 < u < 1.0:
        return t, u
    return None


def lift_crossings(P, radius, window_m=None):
    """Where the polyline crosses itself in plan view, lift the LATER strand over the earlier one: a smooth
    raised-cosine bump of one rope diameter (plus a millimetre) over +-window_m of arc length around the crossing.
    Returns (P lifted, [{s_under, s_over, point}])."""
    P = np.asarray(P, float).copy()
    S = _arclen(P)
    w = float(window_m if window_m is not None else max(8.0 * radius, 0.03))
    h = 2.0 * float(radius) + 0.001
    crossings = []
    n = len(P)
    for i in range(n - 1):
        for j in range(i + 2, n - 1):
            hit = _seg_cross(P[i, :2], P[i + 1, :2], P[j, :2], P[j + 1, :2])
            if hit is None:
                continue
            t, u = hit
            s_i = S[i] + t * (S[i + 1] - S[i]); s_j = S[j] + u * (S[j + 1] - S[j])
            if abs(s_j - s_i) < 2.5 * w:          # a crossing of neighbouring stretches is a kink, not a loop
                continue
            crossings.append({"s_under": round(float(s_i), 4), "s_over": round(float(s_j), 4),
                              "point": [round(float(v), 4) for v in (P[i, :2] + t * (P[i + 1, :2] - P[i, :2]))]})
    if not crossings:
        return P, []
    # LAYERED, by clearance rather than by crossing point: wherever a later stretch of the rope passes within one
    # diameter (plan view) of an earlier stretch, it is raised until the two capsules clear each other -- a crossing at
    # a shallow angle overlaps for centimetres either side of the point where the centrelines meet -- and every raise is
    # spread by a raised-cosine bump of half-width w so the rope curves smoothly over. A few passes settle a strand that
    # lies over another one which is itself lifted over a third.
    clear = h
    dxy = np.linalg.norm(P[:, None, :2] - P[None, :, :2], axis=2)
    later = (S[:, None] - S[None, :]) > 2.5 * w                       # i is the later stretch, j the earlier one
    near = later & (dxy < clear)
    if not near.any():
        return P, crossings
    dz_need = np.sqrt(np.maximum(clear ** 2 - dxy ** 2, 0.0))
    K = np.abs(S[:, None] - S[None, :]) / w
    K = np.where(K < 1.0, 0.5 * (1.0 + np.cos(np.pi * K)), 0.0)        # bump of the raise at i, seen at k
    lift = np.zeros(n)
    for _ in range(6):
        Z = P[:, 2] + lift
        req = np.where(near, Z[None, :] + dz_need - P[:, 2][:, None], -np.inf).max(axis=1)
        req = np.maximum(req, 0.0)
        new = (req[:, None] * K).max(axis=0)
        if np.all(new <= lift + 1e-6):
            break
        lift = np.maximum(lift, new)
    P[:, 2] += lift
    return P, crossings


def plan_rope(params) -> dict:
    """The geometry of the rope before any MJCF: the centreline (z set, crossings lifted, densified so the bumps
    survive resampling), the cut arc length and point, the two halves' vertices."""
    p = params
    P = load_centreline(p["centreline"])
    r = float(p["radius"])
    if p.get("z") is not None:
        P[:, 2] = float(p["z"])
    # densify to ~2 mm so the crossing bumps and the cut point are resolved before the capsule resampling
    S = _arclen(P)
    L = float(S[-1])
    if L < 0.02:
        raise ValueError(f"centreline: {L * 1000:.1f} mm long; a rope needs a centreline of at least 2 cm")
    P = resample(P, 0.0, L, max(int(np.ceil(L / 0.002)), len(P)))
    crossings = []
    if p.get("lift_crossings", True):
        P, crossings = lift_crossings(P, r)
    S = _arclen(P); L = float(S[-1])
    if p.get("cut_point") is not None:
        s_cut, off = nearest_s(P, p["cut_point"])
    else:
        s_cut, off = float(p["cut_frac"]) * L, 0.0
    N = int(p["segments"])
    seg = L / max(N, 1)
    nmin = int(p.get("min_half_segments", 3))
    if s_cut < nmin * seg * 0.999 or L - s_cut < nmin * seg * 0.999:
        raise ValueError(f"the cut at {s_cut * 1000:.0f} mm of {L * 1000:.0f} mm leaves a half shorter than "
                         f"{nmin} capsules of {seg * 1000:.1f} mm; move the cut or raise `segments`")
    na = max(nmin, int(round(s_cut / seg)))
    nb = max(nmin, int(round((L - s_cut) / seg)))
    va = resample(P, 0.0, s_cut, na)
    vb = resample(P, s_cut, L, nb)
    vb[0] = va[-1]                                  # the same point to the last bit: the CONNECT is born satisfied
    return {"P": P, "length_m": L, "s_cut": float(s_cut), "cut_point": va[-1].copy(), "cut_point_offset_m": float(off),
            "va": va, "vb": vb, "crossings": crossings}


def build(params: dict | None = None, out_dir: str | Path = ".") -> dict:
    from .scissors import attach_cuttable_thread
    import mujoco
    p = merge_params(DEFAULTS, params)
    if p["centreline"] is None:
        raise ValueError("construct_cuttable_rope needs `centreline` (the measured polyline, metres)")
    name = str(p["name"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    geo = plan_rope(p)
    spec = mujoco.MjSpec.from_string(
        mjcf_header(f"{name}_cuttable_rope", timestep=p["timestep"])
        + '  <extension><plugin plugin="mujoco.elasticity.cable"/></extension>\n'
        + f"  <worldbody>{LIGHTS_XML}{FLOOR_XML if p['include_floor'] else ''}</worldbody>\n</mujoco>\n")
    rope_p = {k: v for k, v in p.items() if k in ROPE.DEFAULTS and k not in ("centreline", "rest_vertices")}
    rope_p.update(name=name, include_floor=False, grips=bool(p["grips"]))
    # the end caps (ferrules) belong to the rope's two real ends, never to the cut
    la = p["cap_length_a"] if p["cap_length_a"] is not None else p["cap_length"]
    lb = p["cap_length_b"] if p["cap_length_b"] is not None else p["cap_length"]
    halves_p = {"a": {"cap_length_a": float(la), "cap_length_b": 0.0}, "b": {"cap_length_a": 0.0, "cap_length_b": float(lb)}}
    res = attach_cuttable_thread(spec, rope_p, out_dir / "halves", prefix="", cut_frac=geo["s_cut"] / geo["length_m"],
                                 join_name=f"{name}_cut", vertices={"a": geo["va"], "b": geo["vb"]}, half_params=halves_p)
    for e in spec.equalities:                      # the join as stiff as asked
        if e.name == f"{name}_cut":
            e.solref = [float(v) for v in p["join_solref"]]
    spec.compile()
    xml_path = out_dir / f"{name}.xml"
    xml_path.write_text(spec.to_xml())
    model = mujoco.MjModel.from_xml_path(str(xml_path))   # the written file is what a scene will attach
    pa, pb = res["prefixes"]["a"], res["prefixes"]["b"]
    ha, hb = res["halves"]["a"], res["halves"]["b"]

    def names(h, pre):
        return [f"{pre}{h['bodies']['first']}"] + [f"{pre}{b}" for b in h["bodies"]["segments"]] + [f"{pre}{h['bodies']['last']}"]

    bodies_a, bodies_b = names(ha, pa), names(hb, pb)
    for b in bodies_a + bodies_b:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) < 0:
            raise RuntimeError(f"cuttable rope: body {b!r} missing from the written asset")
    gap = _join_gap(model, res["join"])
    saved = dict(p, centreline=[[round(float(v), 6) for v in q] for q in load_centreline(p["centreline"])],
                 cut_point=None if p["cut_point"] is None else [float(v) for v in p["cut_point"]])
    save_params(out_dir / f"{name}_params.json", saved)
    built = {
        "xml": str(xml_path), "params_json": str(out_dir / f"{name}_params.json"),
        "name": name,
        "root_body": bodies_a[0],
        "bodies": bodies_a + bodies_b,
        "halves": {"a": {"prefix": pa, "root_body": bodies_a[0], "bodies": bodies_a, "first": bodies_a[0], "last": bodies_a[-1],
                         "s_range_m": [0.0, round(geo["s_cut"], 4)], "n_capsules": len(bodies_a)},
                   "b": {"prefix": pb, "root_body": bodies_b[0], "bodies": bodies_b, "first": bodies_b[0], "last": bodies_b[-1],
                         "s_range_m": [round(geo["s_cut"], 4), round(geo["length_m"], 4)], "n_capsules": len(bodies_b)}},
        "join": dict(res["join"], s_m=round(geo["s_cut"], 4), point=[round(float(v), 5) for v in geo["cut_point"]]),
        "cut": {"s_m": round(geo["s_cut"], 4), "frac": round(geo["s_cut"] / geo["length_m"], 4),
                "point": [round(float(v), 5) for v in geo["cut_point"]],
                "off_centreline_m": round(geo["cut_point_offset_m"], 4)},
        "length_m": round(geo["length_m"], 4),
        "radius_m": float(p["radius"]),
        "crossings": geo["crossings"],
        "join_gap_mm_at_build": round(1000 * gap, 3),
        "equality": res["join"]["eq"],
        "note": ("one rope in two cables built along the measured centreline; stage it with stage_on_bench at x=y=z=0, "
                 "yaw 0 (its points are rig coordinates). scene_fields(built, scene_name) gives the Stage 2 manifest "
                 "fields in the staged scene's names (cut_joins, halves, root_body, bodies). Deactivating the join "
                 "equality IS the cut; the Stage 3 executor does it through scissors.try_sever."),
    }
    return built


def _join_gap(model, join) -> float:
    """Distance between the two sides of the join at qpos0 (0 = continuous)."""
    import mujoco
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    e = model.equality(join["eq"]).id
    b1, b2 = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
    p1 = d.xpos[b1] + d.xmat[b1].reshape(3, 3) @ model.eq_data[e][0:3]
    p2 = d.xpos[b2] + d.xmat[b2].reshape(3, 3) @ model.eq_data[e][3:6]
    return float(np.linalg.norm(p1 - p2))


def scene_fields(built: dict, scene_name: str | None = None) -> dict:
    """The Stage 2 manifest fields of this rope once stage_on_bench has staged it as asset `scene_name` (its bodies,
    joints and equalities then carry the prefix '<scene_name>_'): root_body, bodies, halves, cut_joins, and the
    stage_on_bench asset entry that places it."""
    sn = str(scene_name or built["name"])
    pre = f"{sn}_"
    j = built["join"]
    halves = {t: {"prefix": pre + h["prefix"], "root_body": pre + h["root_body"], "first": pre + h["first"],
                  "last": pre + h["last"], "bodies": [pre + b for b in h["bodies"]], "s_range_m": h["s_range_m"]}
              for t, h in built["halves"].items()}
    return {
        "root_body": pre + built["root_body"],
        "bodies": [pre + b for b in built["bodies"]],
        "halves": halves,
        "cut_joins": [{"eq": pre + j["eq"], "body": pre + j["body"], "anchor_local": list(j["anchor_local"]),
                       "diameter_m": j["diameter_m"], "s_m": j["s_m"], "point": j["point"],
                       "other_body": halves["b"]["first"]}],
        "stage_on_bench": {"name": sn, "xml": built["xml"], "x": 0.0, "y": 0.0, "z": 0.0, "yaw_deg": 0.0},
    }


def preview(built: dict, out_dir: str | Path) -> list[str]:
    """One still from above with the cut point marked: the laid rope as built."""
    import mujoco
    from PIL import Image
    m = mujoco.MjModel.from_xml_path(built["xml"])
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    pts = np.array([d.xpos[m.body(b).id] for b in built["bodies"]])
    c = 0.5 * (pts.min(0) + pts.max(0))
    span = float(np.max(pts.max(0) - pts.min(0)))
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = c
    cam.distance = max(span * 1.6, 0.15)
    cam.azimuth, cam.elevation = 90.0, -89.0
    out = []
    with mujoco.Renderer(m, 480, 480) as r:
        r.update_scene(d, cam)
        img = r.render()
    path = Path(out_dir) / f"{built['name']}_preview.png"
    Image.fromarray(img).save(path)
    out.append(str(path))
    return out
