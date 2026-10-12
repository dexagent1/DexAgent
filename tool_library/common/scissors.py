"""SCISSORS — two blades on one pivot, and a cut that really separates.

THE POINT
---------
A scanned scissors is a single fused blob: it has the SHAPE of scissors and
none of the mechanism. What makes scissors scissors is one revolute joint and
two edges that shear past each other, and neither is in the pixels — a video
of scissors lying shut on a table contains no pivot axis, no opening range and
no cutting throat. So they are CONSTRUCTED, from measured outside dimensions
plus the mechanism this builder knows:

  * two halves, each a tapered blade + a shank + a finger ring, stacked in
    the blade-thickness direction and joined by ONE hinge at the pivot. Open
    the hinge and the handles spread while the blades open by the same angle,
    because that is the same joint seen from the two ends of the lever.
  * the throat — the point where the two edges cross — travels from the tip
    toward the pivot as the blades close. `throat` reads it off the compiled
    model at the current angle; it is not a constant of the asset.
  * a CUT is a topology change, and MuJoCo has none: you cannot break a body
    chain at runtime. So a cuttable object declares its severable points as
    equality constraints (`join_threads` welds two cables end to end), and
    `try_sever` deactivates one when the throat has actually closed on it —
    measured from the geometry, never scripted by time. What separates the
    rope is that the blades got there and shut, exactly as with a real cut.

The asset contains NO ACTUATOR, on purpose (the rule this whole package
follows): scissors have no motor. `ScissorsDriver` below closes them by
pushing on the two rings with capped forces, which is what a hand does.

WHAT IS AND IS NOT MEASURABLE FROM VIDEO
    measurable  : overall length, blade/handle split, ring size, colours,
                  where it lies and which way the blades point
    NOT          : the opening range, the pivot friction, the shear clearance
                  — set here, and reported by the demo so they can be fitted.
"""
from __future__ import annotations

import numpy as np
from .mjcf import ForceGrip, apply_wrench
from pathlib import Path

from .mjcf import (FLOOR_XML, LIGHTS_XML, free_camera, load, merge_params,
                     mjcf_header, rgba, rollout, save_params, save_stills,
                     vec, write_video)

# ---------------------------------------------------------------------------
# DEFAULTS — metres, kilograms, seconds. The numbers below are the scissors
# measured on the scissor-cutting clip (129 mm overall, purple grips).
# ---------------------------------------------------------------------------
DEFAULTS = {
    "name": "scissors",

    # ---- size: pivot -> tip, and pivot -> the far end of the ring ----------
    "blade_length":     0.051,
    "blade_width":      0.0100,    # at the pivot ...
    "blade_tip_width":  0.0016,    # ... tapering to this at the point
    "blade_thickness":  0.0016,    # ONE blade; the pair is twice this
    "handle_length":    0.073,     # shank + ring, along the handle axis
    "shank_length":     0.028,     # pivot -> where the ring starts
    "ring_width":       0.028,     # ring outer size across the handle axis
    "ring_offset":      0.0126,    # each ring's lateral offset from the axis
    # HOW FAR THE HANDLES ARE CRANKED ABOVE THE BLADE PLANE. Real scissors
    # are bent this way, and it is what holds the rings clear of the table:
    # built flat (0.0) the rings settle 5-10 mm up with their undersides 0.4
    # to 4.9 mm off the surface, so a 14 mm finger cannot pass through a hole
    # without passing through the tabletop -- measured, every such pose put
    # the hand 24-39 mm inside the table and lifted nothing.
    "ring_lift":        0.0,
    "ring_tube":        0.0034,    # the moulded grip's tube radius
    # FLAT MOULDED STRAPS instead of a round tube. The video's scissors have
    # wide flat plastic mouldings around the finger holes, not bent wire —
    # a round tube fat enough to grip (4-5 mm radius) reads as wire loops,
    # while an 8-10 mm wide FLAT strap of the same grabbable cross-section
    # reads as the moulded handle it is. Set to a width in metres to use
    # straps (ring_tube then gives the strap HALF-THICKNESS, vertically);
    # None keeps the round-tube look.
    "ring_strap":       None,
    "ring_scale_b":     1.15,      # the second ring is the bigger one on
                                   #   most scissors (two fingers, not one)
    "ring_segments":    14,        # capsules per ring (a real hole to poke
                                   #   a finger through, not a filled disc)
    "pivot_radius":     0.0055,

    # ---- the mechanism ----------------------------------------------------
    "open_deg":         42.0,      # blades fully open
    "close_deg":        0.0,       # blades shut (joint zero)
    # A REAL PIVOT HAS FRICTION, and it is not decoration here: with almost
    # none, blades squeezed hard enough to cut a cord slam through the last
    # few degrees once the cord parts and the solver goes non-finite a
    # second later. These are the numbers at which the pair still opens on
    # 0.7 N and stays where it is left.
    "joint_damping":    0.015,
    "joint_frictionloss": 0.003,
    # THE PIVOT NEEDS REFLECTED INERTIA. One half of these scissors has
    # 3.7e-5 kg m2 about the pivot, so a 0.4 N m moment (8 N at the ring) is
    # 1e4 rad/s2 — the joint's own limit constraint cannot hold that at a
    # 2 ms step and the solver went non-finite at t=1.44 s. 5e-5 of armature
    # doubles the effective inertia of the hinge DOF and the same episode
    # runs clean; it is the pivot's own stiction and screw mass, not a fudge.
    "joint_armature":   5e-5,
    # A SOFT STOP. Blades squeezed at 5 N and closing on nothing hit the shut
    # limit hard enough to throw the whole tool (measured: the joint reading
    # ran to 2200 deg after the solver reset). A real pivot stop is the two
    # blades meeting face to face, which is compliant, not a wall.
    "joint_solreflimit": [0.01, 1.0],

    # ---- appearance -------------------------------------------------------
    "blade_rgba":  [0.78, 0.79, 0.82, 1.0],     # stainless
    "handle_rgba": [0.42, 0.28, 0.60, 1.0],     # measured purple grip
    "pivot_rgba":  [0.55, 0.55, 0.58, 1.0],

    # ---- placement: the PIVOT's ground position, blades along +x ----------
    "pos":       [0.0, 0.0, 0.0],
    "yaw_deg":   0.0,

    # ---- physics ----------------------------------------------------------
    "mass":      0.045,            # whole tool
    "friction":  [0.9, 0.02, 0.004],
    "condim":    6,
    "solref":    [0.005, 1.0],
    "solimp":    [0.99, 0.999, 0.001],
    "fixed_base": False,
    "timestep":  0.002,
    "include_floor": True,

    # ---- cutting ----------------------------------------------------------
    # a join is severed once the blades have closed to within this gap AT the
    # join, and the join is inside the blades' own span. Both are measured.
    # A CUT NEEDS ALL OF: in the blades' span, in the WEDGE between them, in
    # their plane, and the two edges closed to within cut_gap_m of each other
    # while CLOSING and while both blades touch the thing. See try_sever.
    "cut_gap_m":     0.0035,       # edges this close = through the material
    "cut_catch_m":   0.011,        # how far off the blade line still counts
    "cut_pinch_mm":  0.3,          # how deep EACH blade must bite in
    "cut_lat_m":     0.004,        # how far off the blades' own PLANE. A cord
                                   #   resting ON the shut blades sits one
                                   #   radius off it and can only be pushed.
}

_EPS = 1e-9


def _taper(p, n=5):
    """Blade cross-section half-widths from the pivot to the tip."""
    w0, w1 = float(p["blade_width"]) / 2, float(p["blade_tip_width"]) / 2
    L = float(p["blade_length"])
    out = []
    for i in range(n):
        x0, x1 = L * i / n, L * (i + 1) / n
        hw = w0 + (w1 - w0) * ((i + 0.5) / n)
        out.append((x0, x1, hw))
    return out


def _ring_half_w(p):
    """Half the ring material's IN-PLANE width (strap or tube)."""
    strap = p.get("ring_strap")
    return (float(strap) / 2.0) if strap else float(p["ring_tube"])


def _ring_points(p, sgn: float):
    """The ring's centreline, as an ellipse in the scissors' own plane."""
    k = 1.0 if sgn > 0 else float(p.get("ring_scale_b", 1.0))
    ra = (float(p["handle_length"]) - float(p["shank_length"])) * k  # along x
    rb = float(p["ring_width"]) * (1.0 + 0.4 * (k - 1.0))
    cx = -(float(p["shank_length"]) + ra / 2.0)
    cy = sgn * float(p["ring_offset"])
    n = int(p["ring_segments"])
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    a = ra / 2.0 - _ring_half_w(p)
    b = rb / 2.0 - _ring_half_w(p)
    return np.stack([cx + a * np.cos(th), cy + b * np.sin(th)], axis=1)


def _half(p, name: str, sgn: float, z: float, tag: str, mass_frac: float
          ) -> str:
    """One half of the tool: blade (+x) + shank + ring, all at height `z`."""
    bt = float(p["blade_thickness"])
    ga = (f'friction="{vec(p["friction"])}" condim="{int(p["condim"])}" '
          f'solref="{vec(p["solref"])}" solimp="{vec(p["solimp"])}"')
    m_blade = float(p["mass"]) * mass_frac * 0.45
    m_grip = float(p["mass"]) * mass_frac * 0.55
    g = []
    seg = _taper(p)
    for i, (x0, x1, hw) in enumerate(seg):
        g.append(
            f'      <geom name="{name}_{tag}_blade{i}" type="box" '
            f'size="{(x1 - x0) / 2:.5f} {hw:.5f} {bt / 2:.5f}" '
            f'pos="{(x0 + x1) / 2:.5f} 0 {z:.5f}" '
            f'rgba="{rgba(p["blade_rgba"])}" '
            f'mass="{m_blade / len(seg):.6f}" {ga}/>')
    # the shank: pivot -> the ring, moulded grip colour like the video's
    ring = _ring_points(p, sgn)
    x_join = ring[np.argmax(ring[:, 0]), 0]
    zr = z + float(p.get("ring_lift", 0.0))      # the cranked handle plane
    shank_r = (float(p["ring_strap"]) * 0.45 if p.get("ring_strap")
               else float(p["ring_tube"]) * 0.85)
    g.append(
        f'      <geom name="{name}_{tag}_shank" type="capsule" '
        f'fromto="0 0 {z:.5f} {x_join:.5f} {sgn * float(p["ring_offset"]):.5f}'
        f' {zr:.5f}" size="{shank_r:.5f}" '
        f'rgba="{rgba(p["handle_rgba"])}" mass="{m_grip * 0.25:.6f}" {ga}/>')
    strap = p.get("ring_strap")
    for i in range(len(ring)):
        a, b = ring[i], ring[(i + 1) % len(ring)]
        if strap:
            # a flat box along the segment, lying in the handle plane
            mid = (a + b) / 2.0
            d = b - a
            seg = float(np.hypot(d[0], d[1]))
            yaw = float(np.arctan2(d[1], d[0]))
            g.append(
                f'      <geom name="{name}_{tag}_ring{i}" type="box" '
                f'size="{seg / 2 * 1.09:.5f} {float(strap) / 2:.5f} '
                f'{float(p["ring_tube"]):.5f}" '
                f'pos="{mid[0]:.5f} {mid[1]:.5f} {zr:.5f}" '
                f'euler="0 0 {np.degrees(yaw):.2f}" '
                f'rgba="{rgba(p["handle_rgba"])}" '
                f'mass="{m_grip * 0.75 / len(ring):.6f}" {ga}/>')
            continue
        g.append(
            f'      <geom name="{name}_{tag}_ring{i}" type="capsule" '
            f'fromto="{a[0]:.5f} {a[1]:.5f} {zr:.5f} '
            f'{b[0]:.5f} {b[1]:.5f} {zr:.5f}" '
            f'size="{float(p["ring_tube"]):.5f}" '
            f'rgba="{rgba(p["handle_rgba"])}" '
            f'mass="{m_grip * 0.75 / len(ring):.6f}" {ga}/>')
    L = float(p["blade_length"])
    ra = float(p["handle_length"]) - float(p["shank_length"])
    cx = -(float(p["shank_length"]) + ra / 2.0)
    g.append(f'      <site name="{name}_{tag}_tip" pos="{L:.5f} 0 {z:.5f}" '
             f'size="0.0012" rgba="1 0.2 0.2 0.4"/>')
    g.append(f'      <site name="{name}_{tag}_ring" '
             f'pos="{cx:.5f} {sgn * float(p["ring_offset"]):.5f} {zr:.5f}" '
             f'size="0.0015" rgba="0.2 0.6 1 0.4"/>')
    return "\n".join(g)


def build(params: dict | None = None, out_dir: str | Path = ".") -> dict:
    p = merge_params(DEFAULTS, params)
    name = str(p["name"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if float(p["shank_length"]) >= float(p["handle_length"]):
        raise ValueError("shank_length must be shorter than handle_length")
    bt = float(p["blade_thickness"])
    tube = float(p["ring_tube"])
    # the grips are thicker than the blades, so a real pair of scissors rests
    # on its RINGS and the blades hover: that is where local z=0 has to sit.
    lift = bt / 2.0 + tube
    zA, zB = -bt / 2.0, +bt / 2.0
    ga = (f'friction="{vec(p["friction"])}" condim="{int(p["condim"])}" '
          f'solref="{vec(p["solref"])}" solimp="{vec(p["solimp"])}"')

    body = f"""
    <body name="{name}" pos="{vec((p['pos'][0], p['pos'][1],
                                   p['pos'][2] + lift))}"
          euler="0 0 {float(p['yaw_deg'])}">
{'' if p["fixed_base"] else f'      <freejoint name="{name}_free"/>'}
      <geom name="{name}_pivot" type="cylinder"
            size="{float(p['pivot_radius']):.5f} {bt:.5f}" pos="0 0 0"
            rgba="{rgba(p['pivot_rgba'])}" mass="0.002" {ga}/>
      <site name="{name}_pivot_site" pos="0 0 0" size="0.0015"
            rgba="1 1 0 0.4"/>
{_half(p, name, +1.0, zA, "a", 0.5)}
      <body name="{name}_upper" pos="0 0 0">
        <joint name="{name}_open" type="hinge" axis="0 0 1" pos="0 0 0"
               range="{float(p['close_deg'])} {float(p['open_deg'])}"
               damping="{float(p['joint_damping'])}"
               frictionloss="{float(p['joint_frictionloss'])}"
               armature="{float(p['joint_armature'])}"
               solreflimit="{vec(p['joint_solreflimit'])}"/>
{_half(p, name, -1.0, zB, "b", 0.5)}
      </body>
    </body>"""

    xml = mjcf_header(f"{name}_scissors", timestep=p["timestep"]) + f"""
  <worldbody>{LIGHTS_XML}{FLOOR_XML if p["include_floor"] else ""}{body}
  </worldbody>
</mujoco>
"""
    xml_path = out_dir / f"{name}.xml"
    xml_path.write_text(xml)
    save_params(out_dir / f"{name}_params.json", p)
    out = _built_fields(p)
    out.update({"xml": str(xml_path), "params_json": str(out_dir / f"{name}_params.json")})
    out["cutter"] = cutter_spec(out)
    return out


def _built_fields(p: dict) -> dict:
    """The mechanism's names and measured numbers for a parameter sheet -- the
    part of build()'s return that does not depend on where the files went."""
    name = str(p["name"])
    bt = float(p["blade_thickness"])
    lift = bt / 2.0 + float(p["ring_tube"])
    return {
        "base_body": name,                  # the lower half == the root
        "bodies": {"lower": name, "upper": f"{name}_upper"},
        "joint": f"{name}_open",
        "joint_range_deg": [float(p["close_deg"]), float(p["open_deg"])],
        "sites": {"pivot": f"{name}_pivot_site",
                  "tip_a": f"{name}_a_tip", "tip_b": f"{name}_b_tip",
                  "ring_a": f"{name}_a_ring", "ring_b": f"{name}_b_ring"},
        "ring_bodies": {"a": name, "b": f"{name}_upper"},
        "blade_length_m": round(float(p["blade_length"]), 5),
        "blade_width_m": round(float(p["blade_width"]), 5),
        "blade_tip_width_m": round(float(p["blade_tip_width"]), 5),
        "handle_length_m": round(float(p["handle_length"]), 5),
        "ring_centre_local": [round(-(float(p["shank_length"])
                                      + (float(p["handle_length"])
                                         - float(p["shank_length"])) / 2), 5),
                              round(float(p["ring_offset"]), 5)],
        # the LONGER half is the tool's length: the two rings differ (one
        # takes two fingers), and reporting half A's would under-state the
        # asset against the very measurement it was built from
        "total_length_m": round(
            float(p["blade_length"]) + float(p["shank_length"])
            + (float(p["handle_length"]) - float(p["shank_length"]))
            * max(1.0, float(p.get("ring_scale_b", 1.0))), 5),
        "axis_local": [1.0, 0.0, 0.0],       # pivot -> tips
        "pivot_height_m": round(lift, 5),
        "cut_gap_m": float(p["cut_gap_m"]),
        "cut_catch_m": float(p["cut_catch_m"]),
        "cut_lat_m": float(p["cut_lat_m"]),
        "cut_pinch_mm": float(p["cut_pinch_mm"]),
        "note": ("blades along the body's own +x from the pivot at the "
                 "origin, handles along -x. ONE hinge ('joint'); qpos 0 is "
                 "shut. A cut is not a pose: register severable joins with "
                 "join_threads() and call try_sever() every step — it fires "
                 "on the MEASURED throat gap, so the rope parts only if the "
                 "blades really closed on it. `cutter` is what try_sever "
                 "needs; record it in the Stage 2 manifest entry so the "
                 "Stage 3 executor can cut."),
    }


#: the keys of a built dict that try_sever / gap_at / blade_contact read
CUTTER_KEYS = ("bodies", "joint", "joint_range_deg", "blade_length_m", "blade_width_m",
               "blade_tip_width_m", "axis_local", "pivot_height_m", "ring_centre_local",
               "cut_gap_m", "cut_catch_m", "cut_lat_m", "cut_pinch_mm")


def cutter_spec(built: dict) -> dict:
    """The JSON-ready subset of a built pair that the cut machinery needs, in
    the ASSET's own names (a staged scene adds its prefix: see
    cutter_from_recon)."""
    return {k: built[k] for k in CUTTER_KEYS if k in built}


def cutter_from_recon(entry: dict) -> dict | None:
    """The cutter of a Stage 2 manifest entry, with the scene `prefix` its
    body and joint names carry. Reads `cutter` (construct_scissors' output,
    recorded by Stage 2), else `built.cutter`, else -- for manifests written
    before the cutter was recorded -- derives it from the entry's construct_
    scissors `params` (the same formulas build() uses). The prefix is the
    entry's `cutter_prefix`, else what its `root_body` adds in front of the
    asset's lower-half body name. Returns None when the entry is not a pair
    of scissors."""
    c = entry.get("cutter") or (entry.get("built") or {}).get("cutter")
    if c is None:
        if entry.get("skill") != "construct_scissors" or not isinstance(entry.get("params"), dict):
            return None
        c = cutter_spec(_built_fields(merge_params(DEFAULTS, dict(entry["params"]))))
    c = dict(c)
    prefix = entry.get("cutter_prefix")
    if prefix is None:
        root, lower = str(entry.get("root_body") or ""), str(c["bodies"]["lower"])
        prefix = root[:-len(lower)] if root.endswith(lower) else ""
    c["prefix"] = prefix
    return c


def cord_gates(cutter: dict, diameter_m: float):
    """The cut gates scaled to a CORD (scissors_cut_test's numbers): a cord
    thicker than the blade overlap is sheared by ONE blade bearing on it while
    the other backs it, and it sits about one radius off the blade plane by
    construction. Returns (cutter with cut_lat_m, try_sever keyword args)."""
    r = 0.5 * float(diameter_m)
    return (dict(cutter, cut_lat_m=r + 0.006),
            dict(catch_m=2.5 * r + 0.004, pinch_mm=0.2, require_both=False))


def join_near_blades(model, data, cutter: dict, join: dict, prefix: str = "",
                     margin_m: float = 0.03) -> bool:
    """Cheap pre-gate: is the join within reach of the blades at all (closer
    to the pivot than a blade length plus `margin_m`)? Lets a rollout call
    try_sever only when it could fire."""
    q = join_point(model, data, join)
    piv = np.asarray(data.xpos[_bid(model, prefix, cutter["bodies"]["lower"])], float)
    return float(np.linalg.norm(q - piv)) <= float(cutter["blade_length_m"]) + float(margin_m)


# ===========================================================================
# MEASURING A PAIR OF SCISSORS OFF ONE BENCH FRAME
# ===========================================================================
#: the landmarks a pair of scissors is measured by, in image pixels. Every one
#: of them is a point a person can point at in a still, which is the whole
#: reason the measurement is landmark-based rather than mask-based: a shut
#: pair of scissors is ONE silhouette, and no segmentation of it can say where
#: the pivot is or how far the ring reaches.
LANDMARKS = ("blade_tip", "pivot", "ring_a_near", "ring_a_far",
             "ring_a_left", "ring_a_right", "ring_b_near", "ring_b_far",
             "ring_b_left", "ring_b_right")


def rebuild(params_json: str | Path, out_dir: str | Path | None = None) -> dict:
    from .mjcf import load_params
    p = load_params(params_json)
    return build(p, out_dir or Path(params_json).parent)


# ===========================================================================
# reading the mechanism off a compiled model (never assumed from the plan)
# ===========================================================================
def _bid(model, prefix: str, name: str) -> int:
    import mujoco
    for n in (f"{prefix}{name}", name):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) >= 0:
            return model.body(n).id
    raise KeyError(f"body {name!r} not in the model (prefix {prefix!r})")


def opening_deg(model, data, built: dict, prefix: str = "") -> float:
    import mujoco
    for n in (f"{prefix}{built['joint']}", built["joint"]):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) >= 0:
            return float(np.degrees(data.qpos[model.joint(n).qposadr[0]]))
    raise KeyError(f"joint {built['joint']!r} not in the model")


def blade_frames(model, data, built: dict, prefix: str = ""):
    """((p, dir) lower, (p, dir) upper) — each blade's pivot point and its
    unit direction toward its own tip, in the world."""
    out = []
    for key in ("lower", "upper"):
        b = _bid(model, prefix, built["bodies"][key])
        R = np.asarray(data.xmat[b], float).reshape(3, 3)
        out.append((np.asarray(data.xpos[b], float).copy(),
                    R @ np.asarray(built["axis_local"], float)))
    return out[0], out[1]


def gap_at(model, data, built: dict, point, prefix: str = "") -> dict:
    """Where a world point sits relative to the two blades.

    Scissors cut what is IN THE WEDGE — the V between the two edges — and
    nothing else. A point can be near a blade and still be in none of the
    places that get cut (resting ON TOP of the shut blades, behind the pivot,
    past the tips), so this reports each of those separately instead of one
    "close enough" number:

        s_m      distance from the pivot along the lower blade
        gap_m    separation of the two blade centre-lines at that s
        off_m    distance off the lower blade's line (any direction)
        lat_m    distance off the blades' own PLANE (the shear plane).
                 A cord lying on top of shut blades has lat_m ~ its own
                 radius and can never be cut, only pushed.
        in_span  0 <= s <= blade length
        in_wedge the point is angularly BETWEEN the two blades, i.e. on the
                 inner side of both edges. This is the test that says
                 "in the scissors" rather than "near the scissors".
    """
    (pa, da), (pb, db) = blade_frames(model, data, built, prefix)
    q = np.asarray(point, float)
    s = float((q - pa) @ da)
    L = float(built["blade_length_m"])
    foot = pa + s * da
    off = float(np.linalg.norm(q - foot))
    other = pb + s * db
    n = np.cross(da, db)
    nn = float(np.linalg.norm(n))
    if nn < 1e-9:                       # blades shut: the plane normal is
        R = np.asarray(data.xmat[_bid(model, prefix,               # the
                                      built["bodies"]["lower"])],  # body's
                       float).reshape(3, 3)                        # own z
        n = R @ np.array([0.0, 0.0, 1.0])
    else:
        n = n / nn
    v = q - pa
    lat = float(abs(v @ n))
    # inner side of each edge: the cross products of (blade dir x v) must
    # point opposite ways about the plane normal
    ca = float(np.cross(da, v) @ n)
    cb = float(np.cross(db, v) @ n)
    # THE GAP THAT MATTERS IS BETWEEN THE FACING EDGES, not between the two
    # centre-lines: these blades are 10 mm wide at the pivot, so their faces
    # are already ~7 mm apart when the centre-lines are 15 mm apart. Measuring
    # centre to centre says "still 16 mm open" about a pair of blades that are
    # visibly squeezing a 7 mm cord.
    w0 = float(built.get("blade_width_m", 0.010)) / 2.0
    w1 = float(built.get("blade_tip_width_m", 0.0016)) / 2.0
    hw = w0 + (w1 - w0) * min(max(s / max(L, 1e-9), 0.0), 1.0)
    cgap = float(np.linalg.norm(foot - other))
    return {"s_m": s, "gap_m": cgap, "edge_gap_m": max(cgap - 2.0 * hw, 0.0),
            "half_width_m": hw, "off_m": off, "lat_m": lat,
            "in_span": bool(0.0 <= s <= L),
            "in_wedge": bool(ca * cb <= 0.0),
            # kept for callers written against the old name
            "inside": bool(0.0 <= s <= L)}


def blade_contact(model, data, built: dict, point, prefix: str = "",
                  near_m: float = 0.012) -> dict:
    """Is something actually being PINCHED at `point` — are BOTH blades
    bearing on it, and how deep?

    Scans by CONTACT POSITION, not by body: a cord is a chain of capsules and
    the blades bear on whichever links happen to be there, so keying this on
    one nominated body (the first version did) reports "not pinched" while
    the blades are visibly squeezing the cord two links along.
    """
    q = np.asarray(point, float)
    tool = {_bid(model, prefix, built["bodies"][k]) for k in ("lower", "upper")}
    tag_of = {}
    for key, tag in (("lower", "a"), ("upper", "b")):
        tag_of[_bid(model, prefix, built["bodies"][key])] = tag
    hit = {"a": 0.0, "b": 0.0}
    for c in range(data.ncon):
        if float(np.linalg.norm(data.contact.pos[c] - q)) > near_m:
            continue
        g1, g2 = int(data.contact.geom1[c]), int(data.contact.geom2[c])
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        for gb, ob in ((g1, b2), (g2, b1)):
            bb = int(model.geom_bodyid[gb])
            if bb not in tool or ob in tool:
                continue
            if "_blade" not in (model.geom(gb).name or ""):
                continue        # a ring or a shank touching is not a cut
            t = tag_of[bb]
            hit[t] = max(hit[t], -float(data.contact.dist[c]))
    return {"a_mm": round(1000 * hit["a"], 3), "b_mm": round(1000 * hit["b"], 3),
            "both": bool(hit["a"] > 0.0 and hit["b"] > 0.0),
            "min_mm": round(1000 * min(hit["a"], hit["b"]), 3)}


def join_point(model, data, join: dict) -> np.ndarray:
    """WHERE the cut point is, right now. The join body's ORIGIN is the start
    of its capsule, one segment short of where the two cables actually meet —
    using it put the cut point 7.5 mm off the cord and outside the wedge."""
    bid = _bid(model, "", join["body"])
    a = np.asarray(join.get("anchor_local", [0.0, 0.0, 0.0]), float)
    R = np.asarray(data.xmat[bid], float).reshape(3, 3)
    return np.asarray(data.xpos[bid], float) + R @ a


def throat(model, data, built: dict, prefix: str = "") -> np.ndarray:
    """Where the two edges currently cross — the point that does the cutting.
    Closed blades cross along their whole length, so the throat is reported at
    the pivot; as they open it travels out toward the tips."""
    (pa, da), (pb, db) = blade_frames(model, data, built, prefix)
    th = np.radians(max(opening_deg(model, data, built, prefix), 0.0))
    L = float(built["blade_length_m"])
    w = float(built.get("blade_width_m", 0.009))
    s = min(L, w / max(np.tan(th), 1e-3)) if th > 1e-4 else 0.0
    return pa + s * da


# ===========================================================================
# CUTTING — a severable join between two cable assets
# ===========================================================================
def join_threads(spec, body_a: str, body_b: str, name: str = "cut_join",
                 solref=(0.004, 1.0)) -> str:
    """Weld the END of one cable to the START of another, so the pair behaves
    as ONE rope until the weld is deactivated. THIS is the cut point: MuJoCo
    cannot break a body chain, so a rope that can be cut is authored as two
    cables plus one equality, and cutting deactivates the equality.

    `spec` is an mjSpec being assembled (after both cables are attached).
    Returns the equality's name.
    """
    import mujoco
    names = {b.name for b in spec.bodies}
    for b in (body_a, body_b):
        if b not in names:
            raise KeyError(f"join_threads: no body {b!r} in the spec")
    eq = spec.add_equality()
    eq.type = mujoco.mjtEq.mjEQ_WELD
    eq.objtype = mujoco.mjtObj.mjOBJ_BODY
    eq.name = name
    eq.name1, eq.name2 = body_a, body_b
    eq.active = True
    eq.solref = list(solref)
    eq.solimp = [0.995, 0.9995, 0.0005, 0.5, 2]
    eq.data = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1.0]
    return name


def attach_cuttable_thread(spec, params: dict, out_dir, prefix: str = "rope_",
                           cut_frac: float = 0.5, frame=None,
                           join_name: str = "rope_cut", vertices=None,
                           half_params=None) -> dict:
    """Attach a rope that CAN BE CUT: two cables laid end to end at qpos0 and
    tied by one CONNECT equality at the joint.

    Why not one cable: MuJoCo cannot break a body chain at run time, so a rope
    that is one composite can never part. Why CONNECT and not WELD: a rope
    bends freely at every point including this one — a weld would put a rigid
    elbow in the middle of the cord, which is visible the moment the rope is
    picked up.

    `params` is a normal `rope.build` parameter dict (its `length` is the
    WHOLE rope). Returns the two halves' built dicts plus the equality name
    and the body whose position IS the cut point, which is exactly what
    `try_sever` wants.

    `vertices` = {"a": [[x, y, z], ...], "b": [...]} builds each half ALONG
    a measured polyline instead of straight (cable `rest_vertices`): the last
    point of "a" must be the first point of "b" -- the cut point -- so the two
    halves meet there in the reference configuration the CONNECT is baked
    from. `cut_frac` is then only recorded; the split is the polylines'.
    `half_params` = {"a": {...}, "b": {...}} overrides rope parameters per
    half (the end caps: a rope's ferrules are at its two real ends, never at
    the cut).
    """
    import mujoco
    from pathlib import Path as _P
    from . import cable as ROPE
    out_dir = _P(out_dir)
    L = float(params.get("length", ROPE.DEFAULTS["length"]))
    p0 = np.asarray(params.get("pos", ROPE.DEFAULTS["pos"]), float)
    halves = {}
    if vertices is not None:
        va, vb = (np.asarray(vertices[t], float) for t in ("a", "b"))
        if float(np.linalg.norm(va[-1] - vb[0])) > 1e-6:
            raise ValueError("attach_cuttable_thread: the last vertex of half a must be "
                             "the first vertex of half b (the cut point)")
    for tag, frac, off in (("a", cut_frac, 0.0), ("b", 1.0 - cut_frac, cut_frac)):
        q = dict(params)
        q["name"] = f"{params.get('name', 'thread')}_{tag}"
        q["include_floor"] = False
        q["grips"] = params.get("grips", True)
        q.update((half_params or {}).get(tag, {}))
        if vertices is not None:
            # built along its own measured polyline: count, length and start
            # follow from the points (cable.build, rest_vertices)
            q.pop("centreline", None)
            q["rest_vertices"] = [[float(v) for v in pt] for pt in vertices[tag]]
        else:
            q["length"] = L * frac
            q["segments"] = max(6, int(round(int(params.get(
                "segments", ROPE.DEFAULTS["segments"])) * frac)))
            # laid end to end IN THE REFERENCE CONFIGURATION: a body-anchored
            # connect bakes body2's anchor from qpos0, so if the two halves do
            # not already meet there the constraint is born with an offset.
            q["pos"] = [float(p0[0] + L * off), float(p0[1]), float(p0[2])]
        halves[tag] = ROPE.build(q, out_dir / f"{prefix}{tag}".strip("_"))
    for tag in ("a", "b"):
        child = mujoco.MjSpec.from_file(halves[tag]["xml"])
        spec.attach(child, prefix=f"{prefix}{tag}_",
                    frame=frame if frame is not None else spec.worldbody.add_frame())
    body_a = f"{prefix}a_{halves['a']['bodies']['last']}"
    body_b = f"{prefix}b_{halves['b']['bodies']['first']}"
    names = {b.name for b in spec.bodies}
    for b in (body_a, body_b):
        if b not in names:
            raise KeyError(f"attach_cuttable_thread: no body {b!r} after attach")
    seg = float(halves["a"]["segment_length_m"])
    if vertices is not None:
        # a measured polyline's capsules are not all equal: the anchor is the
        # far end of half a's LAST capsule
        va = np.asarray(vertices["a"], float)
        seg = float(np.linalg.norm(va[-1] - va[-2]))
    eq = spec.add_equality()
    eq.type = mujoco.mjtEq.mjEQ_CONNECT
    eq.objtype = mujoco.mjtObj.mjOBJ_BODY
    eq.name = join_name
    eq.name1, eq.name2 = body_a, body_b
    eq.active = True
    eq.solref = [0.004, 1.0]
    eq.solimp = [0.995, 0.9995, 0.0005, 0.5, 2]
    eq.data = [seg, 0.0, 0.0] + [0.0] * 8      # anchor at the far end of a
    # THE TWO CAPSULES AT THE JOIN MUST NOT COLLIDE WITH EACH OTHER. A chain
    # of capsules laid end to end always overlaps at every join by up to one
    # diameter — their hemispherical caps do — and inside ONE cable MuJoCo's
    # parent-child filter hides it. Across the two cables nothing does, so
    # the scene reports a 6 mm penetration that is not a modelling error but
    # the same overlap every other segment already has. (It stays excluded
    # after the cut: two severed tips can pass through each other at the very
    # end, which is invisible and costs nothing.)
    ex = spec.add_exclude()
    ex.name = f"{join_name}_selfpair"
    ex.bodyname1, ex.bodyname2 = body_a, body_b
    return {"halves": halves, "prefixes": {t: f"{prefix}{t}_" for t in halves},
            "join": {"eq": join_name, "body": body_a,
                     # the cut point is the capsule's FAR END, not the body's
                     # origin — one segment apart, which put it outside the
                     # wedge and made every gate read the wrong place
                     "anchor_local": [seg, 0.0, 0.0],
                     "diameter_m": round(2 * float(
                         params.get("radius", 0.0035)), 5)},
            "cut_point_body": body_a,
            "note": ("one rope in two cables; `join` goes straight into "
                     "scissors.try_sever. Deactivating that equality IS the "
                     "cut — nothing else changes.")}


def join_threads_eq(model, data, join: dict) -> bool:
    """Is the rope still whole?"""
    import mujoco
    eid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, join["eq"])
    return bool(eid >= 0 and data.eq_active[eid])


def try_sever(model, data, built: dict, joins, prefix: str = "",
              gap_m: float | None = None, catch_m: float | None = None,
              pinch_mm: float | None = None, require_contact: bool = True,
              require_closing: bool = True, require_both: bool = True) -> list:
    """Cut every registered join the blades have REALLY closed on.

    `joins` = [{"eq": <equality name>, "body": <the join's own body>,
    "anchor_local": [dx,0,0], "diameter_m": <what is being cut>}, ...] — the
    shape `attach_cuttable_thread` returns.

    SIX conditions, all measured, all of which a real cut also satisfies. The
    first version had only the last two, and that is exactly how a cord could
    "part" while the blades were still OPENING and merely sweeping it aside:

      1. the join is inside the blades' own SPAN (0 <= s <= blade length) —
         nothing is cut past the tips or behind the pivot;
      2. it is in the WEDGE, angularly between the two edges. A cord lying
         across SHUT blades, or on the outside of one of them, is near the
         scissors and in none of the places that get cut;
      3. it is in the blades' own PLANE (|lat| <= cut_lat_m) — same reason,
         stated as a distance;
      4. it is close to the blade line (off_m <= catch_m);
      5. the blades are CLOSING (joint velocity < 0). Opening blades push
         material out of the way; they have never cut anything;
      6. BOTH blades bear on it there, each biting at least `pinch_mm`, and
         the two edges have closed to within the MATERIAL'S OWN THICKNESS
         (`diameter_m`, or `gap_m`). Blunt box blades cannot pass through a
         7 mm cord — they stall on it — so "the edges met" is the wrong
         question; "the edges closed onto it and bit in" is the right one.

    Returns the joins cut on this call, each with the numbers it was cut on.
    """
    import mujoco
    catch_m = float(built["cut_catch_m"] if catch_m is None else catch_m)
    lat_m = float(built.get("cut_lat_m", 0.5 * catch_m))
    pinch_mm = float(built.get("cut_pinch_mm", 0.3)
                     if pinch_mm is None else pinch_mm)
    closing = True
    if require_closing:
        for n in (f"{prefix}{built['joint']}", built["joint"]):
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) >= 0:
                closing = bool(data.qvel[model.joint(n).dofadr[0]] < -1e-3)
                break
    cut = []
    for j in joins:
        eid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, j["eq"])
        if eid < 0 or not data.eq_active[eid]:
            continue
        q = join_point(model, data, j)
        g = gap_at(model, data, built, q, prefix)
        thr = float(gap_m if gap_m is not None
                    else j.get("diameter_m", built["cut_gap_m"]))
        con = (blade_contact(model, data, built, q, prefix)
               if require_contact else {"both": True, "min_mm": 9e9,
                                        "a_mm": None, "b_mm": None})
        # A FAT CORD AGAINST PLATE BLADES: "both blades biting" assumes a
        # thread thinner than the blade overlap. A 16 mm rope closed on by
        # 5 mm plates is sheared by ONE blade bearing on it while the other
        # backs it; `require_both=False` asks for one bite (the rest of the
        # gates -- span, wedge, plane, catch, closing -- still all hold).
        if not require_contact:
            bite_ok = True      # no blade reading asked for (the stub has no per-blade depths)
        elif require_both:
            bite_ok = con["both"] and con["min_mm"] >= pinch_mm
        else:
            bite_ok = max(con["a_mm"], con["b_mm"]) >= pinch_mm
        if (g["in_span"] and g["in_wedge"] and g["lat_m"] <= lat_m
                and g["off_m"] <= catch_m and g["edge_gap_m"] <= thr
                and closing and bite_ok):
            data.eq_active[eid] = 0
            cut.append({**{k: v for k, v in j.items() if k != "anchor_local"},
                        "s_m": round(g["s_m"], 4),
                        "gap_mm": round(1000 * g["gap_m"], 2),
                        "edge_gap_mm": round(1000 * g["edge_gap_m"], 2),
                        "lat_mm": round(1000 * g["lat_m"], 2),
                        "pinch_mm": [con["a_mm"], con["b_mm"]]})
    return cut


def sever_report(model, data, built: dict, joins, prefix: str = "") -> list:
    """Every gate's current value for every join — the sensor behind a cut
    that did NOT happen, which is the harder thing to explain."""
    import mujoco
    out = []
    vel = None
    for n in (f"{prefix}{built['joint']}", built["joint"]):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) >= 0:
            vel = float(data.qvel[model.joint(n).dofadr[0]])
            break
    for j in joins:
        eid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, j["eq"])
        q = join_point(model, data, j)
        g = gap_at(model, data, built, q, prefix)
        con = blade_contact(model, data, built, q, prefix)
        out.append({"eq": j["eq"],
                    "whole": bool(eid >= 0 and data.eq_active[eid]),
                    "s_mm": round(1000 * g["s_m"], 1),
                    "gap_mm": round(1000 * g["gap_m"], 1),
                    "edge_gap_mm": round(1000 * g["edge_gap_m"], 1),
                    "off_mm": round(1000 * g["off_m"], 1),
                    "lat_mm": round(1000 * g["lat_m"], 1),
                    "in_span": g["in_span"], "in_wedge": g["in_wedge"],
                    "pinch_mm": con["min_mm"], "pinched_by_both": con["both"],
                    "joint_vel": None if vel is None else round(vel, 3)})
    return out


# ===========================================================================
def preview(built: dict, out_dir: str | Path) -> list[str]:
    """Three stills: shut, half open, fully open — the mechanism's extremes."""
    import mujoco
    from .mjcf import key_states, load_params
    m = load(built["xml"])
    d = mujoco.MjData(m)
    p = load_params(built["params_json"])
    jadr = int(m.joint(built["joint"]).qposadr[0])
    hi = np.radians(float(built["joint_range_deg"][1]))

    def at(frac):
        def setter(mm, dd):
            dd.qpos[jadr] = hi * frac
        return setter

    L = built["total_length_m"]
    c = np.asarray(p["pos"], float)
    return key_states(
        m, d, [("shut", at(0.0)), ("half_open", at(0.5)), ("open", at(1.0))],
        out_dir, f"{built['base_body']}_preview",
        camera=free_camera([c[0], c[1], 0.01], L * 2.2, 90, -55),
        width=520, height=460)


def demo(built: dict, out_mp4: str | Path, fps: int = 30,
         width: int = 640, height: int = 480) -> dict:
    """Open and shut the blades with FORCES on the two rings — no actuator,
    no scripted qpos — with a cuttable cord lying in the throat. The caption
    is the measurement: how wide they opened, how hard the rings were pushed,
    and at what gap the cord actually parted."""
    import mujoco
    from .mjcf import clear_forces, load_params
    p = load_params(built["params_json"])
    m = load(built["xml"])
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    drv = ScissorsDriver(m, d, built)
    L = built["total_length_m"]
    cam = free_camera([0.5 * built["blade_length_m"], 0.0, 0.01], L * 2.0,
                      100, -35)

    def control(t, model, data):
        clear_forces(data)
        return drv.step(t)

    frames = rollout(m, d, seconds=drv.T["end"], fps=fps, control=control,
                     camera=cam, width=width, height=height)
    out = write_video(frames, out_mp4, fps=fps)
    out["stills"] = save_stills(frames, Path(out_mp4).parent,
                                Path(out_mp4).stem + "_still", n=4)
    out["measured"] = drv.meas
    out["verdict"] = drv.verdict()
    return out


# ---------------------------------------------------------------------------


# ===========================================================================
class ScissorsDriver:
    """Open and shut a pair of scissors by PUSHING ON THE RINGS.

    Two capped forces, applied at the two ring centres in the scissors' own
    plane, perpendicular to each handle — which is what a thumb and a finger
    do. Nothing writes the hinge's qpos, so a blade that jams on what it is
    cutting simply stops, and the force needed to close is a measurement
    rather than a setting.

    The cut is the same kind of evidence: `joins` (from
    `scissors.join_threads`) are severed by `scissors.try_sever`, which fires
    on the MEASURED gap between the blades at the join, so nothing parts
    unless the blades really shut on it.
    """

    # WHAT A FINGER ACTUALLY PUSHES WITH. These scissors weigh 45 g and one
    # half has 3.7e-5 kg m2 about the pivot; 8 N at the ring is 1e4 rad/s2,
    # which is not a hand opening scissors, it is an explosion (measured: the
    # solver went non-finite). 0.7 N opens them fully in 1.6 s.
    MAX_N = 0.7

    def __init__(self, model, data, built: dict, prefix: str = "",
                 t0: float = 0.0, joins=None, hold_root: bool = True):
        import sys
        self.SC = sys.modules[__name__]
        self.m, self.d, self.b, self.p = model, data, built, prefix
        self.t0 = float(t0)
        self.joins = list(joins or [])
        self.lower = model.body(_n(prefix, built["bodies"]["lower"])).id
        self.upper = model.body(_n(prefix, built["bodies"]["upper"])).id
        # the hand that HOLDS the tool, not the one that works it: gentle,
        # and gentle in ROTATION above all. A 45 g tool has ~1e-5 kg m2 of
        # inertia, so ForceGrip's stock 0.8 N m cap is 8e4 rad/s2 of angular
        # acceleration — measured, that alone blew the solver up at t=4.65 s.
        self.ga = ForceGrip(model, self.lower, kp=8.0, kd=0.6,
                            max_force=1.0, kr=0.02, kdr=0.002,
                            max_torque=0.008)
        self.gb = ForceGrip(model, self.upper, kp=8.0, kd=0.6,
                            max_force=1.0, kr=0.02, kdr=0.002,
                            max_torque=0.008)
        self.hold_root = bool(hold_root)
        self.home = np.asarray(data.xpos[self.lower], float).copy()
        self.home_q = np.asarray(data.xquat[self.lower], float).copy()
        self.ring_local = np.asarray(
            built.get("ring_centre_local", [-0.05, 0.012]), float)
        self.meas = {"max_open_deg": 0.0, "min_gap_at_join_mm": 1e9,
                     "ring_force_N": self.MAX_N, "cut": [],
                     "closed_deg": None}
        # settle -> open -> let go -> squeeze shut. A caller whose blades
        # START open sets open == settle, which skips the opening stroke:
        # opening INTO the thing you mean to cut only sweeps it aside.
        self.T = dict(settle=0.6, open=2.2, hold=2.8, close=5.0, end=6.2)
        # extra keyword arguments for try_sever (catch_m, pinch_mm,
        # require_both ...): a fat cord against plate blades is sheared by
        # one blade bearing on it while the other backs it
        self.sever_kwargs: dict = {}

    # ---- geometry -------------------------------------------------------
    def _ring_world(self, which: str):
        bid = self.lower if which == "a" else self.upper
        R = np.asarray(self.d.xmat[bid], float).reshape(3, 3)
        sgn = 1.0 if which == "a" else -1.0
        loc = np.array([self.ring_local[0], sgn * self.ring_local[1], 0.0])
        return np.asarray(self.d.xpos[bid], float) + R @ loc

    def _plane_normal(self):
        R = np.asarray(self.d.xmat[self.lower], float).reshape(3, 3)
        return R @ np.array([0.0, 0.0, 1.0])

    def _push(self, which: str, sign: float, force: float):
        """Force at the ring, perpendicular to that handle, in the plane."""
        bid = self.lower if which == "a" else self.upper
        pivot = np.asarray(self.d.xpos[self.lower], float)
        ring = self._ring_world(which)
        arm = ring - pivot
        n = self._plane_normal()
        tang = np.cross(n, arm)
        tang /= (np.linalg.norm(tang) + 1e-12)
        f = sign * force * tang
        apply_wrench(self.d, bid, force=f, point=ring,
                     com=np.asarray(self.d.xipos[bid], float))
        return f

    # ---- the episode ----------------------------------------------------
    def step(self, t) -> str:
        t = t - self.t0
        if t < 0:
            return ""
        SC = self.SC
        ang = SC.opening_deg(self.m, self.d, self.b, self.p)
        self.meas["max_open_deg"] = max(self.meas["max_open_deg"], ang)
        if self.hold_root:
            # the tool is held in place (a hand round the whole thing), so
            # what the ring forces do is OPEN it, not skate it across the table
            # IN-PLANE ONLY (axes z off): a hold that can pull up is a hold
            # that lifts a 45 g tool off the table the moment it tips.
            self.ga.hold(self.d, self.home, self.home_q, axes=(1, 1, 0))
        for j in self.joins:
            g = SC.gap_at(self.m, self.d, self.b,
                          self.d.xpos[self.m.body(j["body"]).id], self.p)
            if g["inside"] and g["off_m"] <= float(self.b["cut_catch_m"]):
                self.meas["min_gap_at_join_mm"] = min(
                    self.meas["min_gap_at_join_mm"], 1000 * g["gap_m"])
        cut = SC.try_sever(self.m, self.d, self.b, self.joins, self.p,
                           **self.sever_kwargs)
        if cut:
            self.meas["cut"] += cut
            self.meas["closed_deg"] = round(ang, 2)

        T = self.T
        if t < T["settle"]:
            return f"scissors at rest — blades {ang:4.1f} deg apart"
        if t < T["open"]:
            a = smoothstep(t, T["settle"], T["open"])
            self._push("b", +1.0, self.MAX_N * a)
            return (f"upper ring pushed at <= {self.MAX_N:.1f} N -> "
                    f"{ang:4.1f} deg open")
        if t < T["hold"]:
            return f"let go — the pivot holds at {ang:4.1f} deg"
        if t < T["close"]:
            a = smoothstep(t, T["hold"], T["close"])
            # EASE OFF ONCE IT IS CUT. A hand stops squeezing the moment the
            # cord parts; keeping full force on drives the freed blades into
            # each other at speed and the solver blows up a second later.
            self._push("b", -1.0, self.MAX_N * a
                       * (0.15 if self.meas["cut"] else 1.0))
            done = "; cord PARTED" if self.meas["cut"] else ""
            return (f"upper ring squeezed at <= {self.MAX_N:.1f} N -> "
                    f"{ang:4.1f} deg{done}")
        return (f"shut at {ang:4.1f} deg"
                + (f" — cut at {self.meas['closed_deg']} deg"
                   if self.meas["cut"] else ""))

    def verdict(self) -> str:
        m = self.meas
        hi = float(self.b["joint_range_deg"][1])
        opened = m["max_open_deg"] >= 0.6 * hi
        if not self.joins:
            return (f"SCISSORS OK — opened to {m['max_open_deg']:.1f} of "
                    f"{hi:.0f} deg on <= {m['ring_force_N']:.1f} N at the "
                    f"rings, and shut again"
                    if opened else
                    f"SCISSORS SUSPECT — only {m['max_open_deg']:.1f} of "
                    f"{hi:.0f} deg on {m['ring_force_N']:.1f} N")
        if m["cut"]:
            return (f"CUT — opened to {m['max_open_deg']:.1f} deg, then shut "
                    f"on the cord and parted it at "
                    f"{m['cut'][0]['gap_mm']:.1f} mm of blade gap, "
                    f"{1000*m['cut'][0]['s_m']:.0f} mm from the pivot")
        return (f"NO CUT — blades reached {m['max_open_deg']:.1f} deg and "
                f"closed to {m['min_gap_at_join_mm']:.1f} mm at the cord, "
                f"never under the {1000*float(self.b['cut_gap_m']):.1f} mm "
                f"the cut needs")


def smoothstep(t, t0, t1):
    if t <= t0:
        return 0.0
    if t >= t1:
        return 1.0
    s = (t - t0) / (t1 - t0)
    return 0.5 - 0.5 * np.cos(np.pi * s)


def _n(prefix, name):
    return f"{prefix}{name}"
