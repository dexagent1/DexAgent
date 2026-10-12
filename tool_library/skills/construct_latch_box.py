"""Construct latching container: A food container with a separate lid and edge latches that really clip shut: the lid seats in the rim, then straps fold over the walls and snap onto nubs.

LATCHING FOOD CONTAINER — a box, a SEPARATE lid, and four edge latches
that really clip shut.

THE POINT
---------
A Lock&Lock-style meal box is not "a box with a lid on it". Closing it is two
different mechanisms in series, and only the first one is placement:

  1. SEATING   the lid's gasket ring drops into the body's rim. Nothing holds
               it: lift the lid and it comes straight off, and the build
               MEASURES that (n_latched = 0 after seating).
  2. LATCHING  four straps hinged on the lid's four edges fold down over the
               body's walls. Each strap carries a WINDOW near its end, and
               each wall carries a NUB that must pop through that window. The
               pop is the task: it is what makes closing the box a force
               problem instead of a placement problem, and it is why the box
               then stays shut when you pick it up by the lid.

Neither part is in a scan. SAM3D fuses a transparent container into one shell
with no rim, no nub and no straps, so the mechanism is CONSTRUCTED from
measured outer dimensions — the same move construct_bottle makes for a thread.

WHY A WINDOW AND A NUB, AND NOT A HOOK AND A LEDGE
--------------------------------------------------
The obvious latch is a hook at the strap's tip that catches under a ledge on
the wall. It was built first and it FAILS, measurably: a hook sitting *below*
its ledge can leave sideways for free — nothing is in its way at its own
height — so under a sustained lift the strap creeps open a degree at a time
(measured: radial engagement fell 2.06 -> 0.66 mm in 2.4 s of hanging, on its
way to letting go). A window is different in exactly the way that matters:
the nub is captured at the strap's own height, so leaving requires bowing the
strap outward over the nub — the same act, and the same force, as closing it.

HOW THE SNAP IS MODELLED
------------------------
A real latch works because the plastic strap BENDS: the nub pushes it out,
it springs back once the nub is through the window, and it must be pushed out
again to come off. Putting that "give" in the contact solver instead would
make it neither measurable nor a property of the object, so the strap's
elasticity is an explicit DOF:

    <hinge fold>      0 deg = LATCHED, +ve = swung open
      <slide flex>    the strap's own outward springiness, 0 = relaxed

and the snap is then pure geometry against that spring: the nub stands
`snap_interference` (default 1.2 mm) proud of where the strap wants to lie,
so getting past it costs `flex_stiffness * snap_interference` newtons — a
number this build prints, that a fingertip exceeds and a resting lid never
will.

No actuators exist anywhere in the asset, so nothing in the demo can be
produced by a servo: every fold is a force applied to a body.

    python -m tool_library.skills.construct_latch_box [out_dir]      # build + preview stills
"""
from __future__ import annotations

TOOL = {'name': 'construct_latch_box',
 'title': 'Construct latching container',
 'type': 'skill',
 'category': 'Object Reconstruction',
 'summary': 'A food container with a separate lid and edge latches that really clip shut: the '
            'lid seats in the rim, then straps fold over the walls and snap onto nubs.',
 'description': 'Builds a rounded-rectangular body, a separate free lid with a gasket ring '
                'that seats in the body rim, and one hinged strap per edge whose window must '
                'pop over a nub on the wall, from outer size, corner radius, wall and lid '
                'thickness, latch count, strap and nub geometry, colours and masses. Seating '
                'and latching are two mechanisms in series and both are measured on the '
                'compiled model: lid_seating() reports whether the lid is in the rim, '
                'latch_state() how many straps are snapped. The latch_snap_test verifier '
                'presses the straps shut with forces and checks they hold.',
 'inputs': [{'name': 'out_dir',
             'type': 'path',
             'description': 'Folder that receives the MJCF, its meshes and '
                            '<name>_params.json.'},
            {'name': 'preview',
             'type': 'bool',
             'description': 'Also render preview stills of the mechanism at its extremes. '
                            'Default False.'},
            {'name': 'name', 'type': 'str', 'description': 'Name. Default mealbox'},
            {'name': 'length', 'type': 'float', 'description': 'Along +x. Default 0.2025'},
            {'name': 'width', 'type': 'float', 'description': 'Along +y. Default 0.1519'},
            {'name': 'height',
             'type': 'float',
             'description': 'Tabletop to rim. Default 0.0607'},
            {'name': 'corner_r',
             'type': 'float',
             'description': 'Plan-view corner radius. Default 0.0273'},
            {'name': 'wall',
             'type': 'float',
             'description': 'Glass wall thickness. Default 0.005'},
            {'name': 'floor_t',
             'type': 'float',
             'description': 'Glass floor thickness. Default 0.006'},
            {'name': 'rim_h',
             'type': 'float',
             'description': 'The proud band at the top of the wall. Default 0.007'},
            {'name': 'rim_out',
             'type': 'float',
             'description': 'How far that band stands out from the wall. Default 0.0015'},
            {'name': 'lid_over',
             'type': 'float',
             'description': 'Lid outer edge beyond the body wall, per side. Default 0.0015'},
            {'name': 'lid_t', 'type': 'float', 'description': 'The clear panel. Default 0.004'},
            {'name': 'gasket_w',
             'type': 'float',
             'description': 'Sealing ring cross-section, across. Default 0.0055'},
            {'name': 'gasket_t',
             'type': 'float',
             'description': '...and how far it hangs below the panel. Default 0.005'},
            {'name': 'gasket_clear',
             'type': 'float',
             'description': "Slack between the ring and the body's opening. Default 0.0025"},
            {'name': 'latch_edges',
             'type': 'list[4]',
             'description': "Latch edges. Default ['xp', 'xn', 'yp', 'yn']"},
            {'name': 'latch_frac',
             'type': 'float',
             'description': 'Strap width as a fraction of its own edge. Default 0.42'},
            {'name': 'latch_len',
             'type': 'float',
             'description': "Hinge to the strap's free end. Default 0.04"},
            {'name': 'latch_t',
             'type': 'float',
             'description': 'Strap thickness. Default 0.0025'},
            {'name': 'latch_gap',
             'type': 'float',
             'description': "Strap inner face to the body's wall face —. Default 0.0025"},
            {'name': 'bar_h',
             'type': 'float',
             'description': 'The catch bar below the window: what bears. Default 0.008'},
            {'name': 'window_h',
             'type': 'float',
             'description': "The window's own height. Default 0.009"},
            {'name': 'window_frac',
             'type': 'float',
             'description': 'Window width, as a fraction of the strap. Default 0.62'},
            {'name': 'nub_out',
             'type': 'float',
             'description': "The wall nub's protrusion — WITH latch_gap. Default 0.0037"},
            {'name': 'nub_h', 'type': 'float', 'description': 'Nub h. Default 0.005'},
            {'name': 'nub_frac',
             'type': 'float',
             'description': 'Nub width as a fraction of the strap. Default 0.5'},
            {'name': 'nub_z',
             'type': 'float | None',
             'description': 'None = derived so the catch bar just clears it'},
            {'name': 'nub_ramp',
             'type': 'float',
             'description': 'A half-height lead-in above the nub, so the. Default 0.004'},
            {'name': 'lock_gap',
             'type': 'float',
             'description': 'Slack between the catch bar and the nub when. Default 0.0015'},
            {'name': 'latch_open_deg',
             'type': 'float',
             'description': "The fold joint's open stop. Default 115.0"},
            {'name': 'latch_start_deg',
             'type': 'float | None',
             'description': 'Where the latches ARE at t=0. None = splayed'},
            {'name': 'flex_travel',
             'type': 'float',
             'description': 'How far the strap may bow outward. Default 0.006'},
            {'name': 'flex_stiffness',
             'type': 'float',
             'description': 'N/m — SETS THE SNAP FORCE (x interference). Default 6000.0'},
            {'name': 'flex_damping',
             'type': 'float',
             'description': 'Flex damping, joint damping. Default 6.0'},
            {'name': 'flex_armature',
             'type': 'float',
             'description': 'Kg: a 5 g strap on a 4 kN/m spring is a 1 ms. Default 0.01'},
            {'name': 'fold_damping',
             'type': 'float',
             'description': 'Fold damping, joint damping. Default 0.004'},
            {'name': 'fold_frictionloss',
             'type': 'float',
             'description': 'An OPEN latch must stay open on its own. Default 0.006'},
            {'name': 'fold_armature',
             'type': 'float',
             'description': 'Fold armature. Default 0.002'},
            {'name': 'glass_rgba',
             'type': 'list[4]',
             'description': 'Glass rgba, colour as rgba in 0..1. Default [0.82, 0.89, 0.87, '
                            '0.38]'},
            {'name': 'lid_rgba',
             'type': 'list[4]',
             'description': 'Lid rgba, colour as rgba in 0..1. Default [0.88, 0.92, 0.94, '
                            '0.34]'},
            {'name': 'gasket_rgba',
             'type': 'list[4]',
             'description': 'Gasket rgba, colour as rgba in 0..1. Default [0.6, 0.65, 0.7, '
                            '1.0]'},
            {'name': 'latch_rgba',
             'type': 'list[4]',
             'description': 'Latch rgba, colour as rgba in 0..1. Default [0.88, 0.92, 0.94, '
                            '0.82]'},
            {'name': 'pos',
             'type': 'list[3]',
             'description': 'Body footprint centre. Default [0.0, 0.0, 0.0]'},
            {'name': 'yaw_deg',
             'type': 'float',
             'description': 'Yaw deg, degrees. Default 0.0'},
            {'name': 'lid_layout',
             'type': 'str',
             'description': 'Beside | seated | closed | off. Default beside'},
            {'name': 'lid_offset',
             'type': 'list[2]',
             'description': "'beside': lid centre from the body's,. Default [0.24, 0.0]"},
            {'name': 'lid_yaw_deg',
             'type': 'float',
             'description': "In the body's own frame. Default 0.0"},
            {'name': 'glass_density',
             'type': 'float',
             'description': 'Glass density. Default 2500.0'},
            {'name': 'plastic_density',
             'type': 'float',
             'description': 'Plastic density. Default 950.0'},
            {'name': 'body_mass',
             'type': 'float | None',
             'description': 'None = from density; a number rescales it'},
            {'name': 'lid_mass', 'type': 'float | None', 'description': 'Lid mass, kilograms'},
            {'name': 'friction',
             'type': 'str',
             'description': 'Friction, MuJoCo friction triple. Default 0.9 0.02 0.001'},
            {'name': 'body_free',
             'type': 'bool',
             'description': 'A 1 kg dish DOES slide when you press a latch. Default True'},
            {'name': 'timestep',
             'type': 'float',
             'description': 'The flex spring needs it (see flex_armature). Default 0.001'},
            {'name': 'include_floor',
             'type': 'bool',
             'description': 'False when attaching into a world. Default True'}],
 'outputs': [{'name': 'built',
              'type': 'dict',
              'description': 'xml: the MJCF path; params_json: every parameter as built; the '
                             'names of the bodies, joints and equality constraints the '
                             'mechanism exposes, and the derived measurements (sizes, travel, '
                             'masses); stills when preview is set.'}],
 'requirements': ['mujoco>=3.2', 'numpy', 'trimesh', 'imageio', 'Pillow'],
 'simulators': ['MuJoCo'],
 'tags': ['container', 'lid', 'latch', 'snap', 'articulated', 'parametric', 'reconstruct', 'object reconstruction'],
 'used_in': ['Meal Box (extra sample)'],
 'status': 'tested',
 'entrypoint': 'construct_latch_box',
 'cli': 'python -m tool_library.skills.construct_latch_box out_dir'}


import json
import numpy as np
from pathlib import Path

from tool_library.common.mjcf import (FLOOR_XML, LIGHTS_XML, apply_wrench, clear_forces,
                     free_camera, load, merge_params, mjcf_header, rgba,
                     save_params, save_stills, vec, write_video)

# ---------------------------------------------------------------------------
# DEFAULTS — metres, kilograms, degrees. These numbers are the meal-box
# clip's own: outer box and lid from a two-view silhouette fit on
# the bench's table plane (IoU 0.93), the latch proportions from the splayed
# straps in its own frame 0.
# ---------------------------------------------------------------------------
DEFAULTS = {
    "name": "mealbox",

    # ---- BODY: the outer glass box, measured ----
    "length":      0.2025,    # along +x
    "width":       0.1519,    # along +y
    "height":      0.0607,    # tabletop to rim
    "corner_r":    0.0273,    # plan-view corner radius
    "wall":        0.0050,    # glass wall thickness
    "floor_t":     0.0060,    # glass floor thickness
    "rim_h":       0.0070,    # the proud band at the top of the wall
    "rim_out":     0.0015,    # how far that band stands out from the wall

    # ---- LID ----
    "lid_over":    0.0015,    # lid outer edge beyond the body wall, per side
    "lid_t":       0.0040,    # the clear panel
    "gasket_w":    0.0055,    # sealing ring cross-section, across
    "gasket_t":    0.0050,    # ...and how far it hangs below the panel
    "gasket_clear": 0.0025,   # slack between the ring and the body's opening

    # ---- LATCHES: strap with a window, wall with a nub ----
    "latch_edges": ["xp", "xn", "yp", "yn"],
    "latch_frac":  0.42,      # strap width as a fraction of its own edge
    "latch_len":   0.0400,    # hinge to the strap's free end
    "latch_t":     0.0025,    # strap thickness
    "latch_gap":   0.0025,    # strap inner face to the body's wall face —
                              #   must clear rim_out on the way down
    "bar_h":       0.0080,    # the catch bar below the window: what bears
    "window_h":    0.0090,    # the window's own height
    "window_frac": 0.62,      # window width, as a fraction of the strap
    "nub_out":     0.0037,    # the wall nub's protrusion — WITH latch_gap
                              #   this sets the snap: interference =
                              #   nub_out - latch_gap
    "nub_h":       0.0050,
    "nub_frac":    0.50,      # nub width as a fraction of the strap
    "nub_z":       None,      # None = derived so the catch bar just clears it
    "nub_ramp":    0.0040,    # a half-height lead-in above the nub, so the
                              #   strap meets the interference in two steps
    "lock_gap":    0.0015,    # slack between the catch bar and the nub when
                              #   latched: this IS the lid's vertical play
    "latch_open_deg": 115.0,  # the fold joint's open stop
    "latch_start_deg": None,  # where the latches ARE at t=0. None = splayed
                              #   (100 deg, the clip's own frame 0) unless the
                              #   layout is 'closed'. Baked into the latch
                              #   body's pose with the joint's `ref`, so the
                              #   model STARTS there while the joint still
                              #   reads 0 = latched — MJCF has no other way
                              #   to say "initial qpos", and a keyframe does
                              #   not survive being attached into a world.
    "flex_travel": 0.0060,    # how far the strap may bow outward
    "flex_stiffness": 6000.0, # N/m — SETS THE SNAP FORCE (x interference)
    "flex_damping": 6.0,
    "flex_armature": 0.010,   # kg: a 5 g strap on a 4 kN/m spring is a 1 ms
                              #   oscillator; armature is what makes it
                              #   integrable (foldbox's remedy for creases)
    "fold_damping": 0.004,
    "fold_frictionloss": 0.006,   # an OPEN latch must stay open on its own
    "fold_armature": 0.0020,

    # ---- appearance (transparent: alpha is the object, not a style) ----
    "glass_rgba":  [0.82, 0.89, 0.87, 0.38],
    "lid_rgba":    [0.88, 0.92, 0.94, 0.34],
    "gasket_rgba": [0.60, 0.65, 0.70, 1.0],
    "latch_rgba":  [0.88, 0.92, 0.94, 0.82],

    # ---- placement + layout ----
    "pos":        [0.0, 0.0, 0.0],   # body footprint centre
    "yaw_deg":    0.0,
    "lid_layout": "beside",          # beside | seated | closed | off
    "lid_offset": [0.24, 0.0],       # 'beside': lid centre from the body's,
    "lid_yaw_deg": 0.0,              #   in the body's own frame

    # ---- physics ----
    "glass_density": 2500.0,
    "plastic_density": 950.0,
    "body_mass":  None,       # None = from density; a number rescales it
    "lid_mass":   None,
    "friction":   "0.9 0.02 0.001",
    "body_free":  True,       # a 1 kg dish DOES slide when you press a latch
    "timestep":   0.001,      # the flex spring needs it (see flex_armature)
    "include_floor": True,    # False when attaching into a world
}

_EDGES = {                       # key -> outward normal in the body frame
    "xp": np.array([1.0, 0.0, 0.0]),
    "xn": np.array([-1.0, 0.0, 0.0]),
    "yp": np.array([0.0, 1.0, 0.0]),
    "yn": np.array([0.0, -1.0, 0.0]),
}


# ---------------------------------------------------------------------------
# geometry helpers — a rounded rectangle, as a slab and as a band
# ---------------------------------------------------------------------------
def _rr_segments(L, W, r, band_w, n_corner=4):
    """The centre line of a rounded-rectangle BAND, as (start, end) chords,
    with the four STRAIGHT runs kept exact.

    Sampling the whole outline as one polygon (the obvious version) puts the
    straight edges on chords between arc samples, which insets them by the
    arc's own sagitta — 0.6 mm here. Invisible in a render, fatal to a latch:
    the strap's clearance to the wall it must not touch is a millimetre or
    two by construction, so a 0.6 mm phantom inset is most of it. Measured:
    with the polygon version the strap was driven 6 mm into the glass.
    """
    cl = band_w / 2.0
    a, b = L / 2 - r, W / 2 - r          # corner centres
    rc = max(r - cl, 1e-4)               # centre-line corner radius
    X, Y = L / 2 - cl, W / 2 - cl
    segs = [((X, -b), (X, b)), ((-a, Y), (a, Y)),
            ((-X, b), (-X, -b)), ((a, -Y), (-a, -Y))]
    for (cx, cy, t0) in ((a, b, 0.0), (-a, b, 90.0),
                         (-a, -b, 180.0), (a, -b, 270.0)):
        ts = np.linspace(t0, t0 + 90.0, n_corner + 1)
        pts = [(cx + rc * np.cos(np.radians(t)),
                cy + rc * np.sin(np.radians(t))) for t in ts]
        segs += [(pts[i], pts[i + 1]) for i in range(n_corner)]
    return segs


def _band(name, L, W, r, band_w, z0, z1, rgba_s, gattr, n_corner=4):
    """A rounded-rectangle WALL/RING: the outer face lands on (L, W, r) and
    the band grows inward by band_w. One box per run, corners chorded — which
    is why the gasket reads as a ring and not as four bars with holes in the
    corners, and that ring is the most visible thing about this object."""
    zc, hz = (z0 + z1) / 2, (z1 - z0) / 2
    out = []
    for i, (p0, p1) in enumerate(_rr_segments(L, W, r, band_w, n_corner)):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        d = p1 - p0
        ln = float(np.linalg.norm(d))
        if ln < 1e-6:
            continue
        ang = float(np.degrees(np.arctan2(d[1], d[0])))
        c = (p0 + p1) / 2
        out.append(
            f'      <geom name="{name}_{i}" type="box" '
            f'size="{vec((ln / 2 + band_w / 4, band_w / 2, hz))}" '
            f'pos="{vec((c[0], c[1], zc))}" euler="0 0 {ang:.4f}" '
            f'rgba="{rgba_s}" {gattr}/>')
    return out


def _slab(name, L, W, r, z0, z1, rgba_s, gattr):
    """A solid rounded-rectangle slab: two boxes + four corner cylinders."""
    a, b = L / 2 - r, W / 2 - r
    zc, hz = (z0 + z1) / 2, (z1 - z0) / 2
    out = [f'      <geom name="{name}_a" type="box" '
           f'size="{vec((L / 2, b, hz))}" pos="0 0 {zc:.6f}" '
           f'rgba="{rgba_s}" {gattr}/>',
           f'      <geom name="{name}_b" type="box" '
           f'size="{vec((a, W / 2, hz))}" pos="0 0 {zc:.6f}" '
           f'rgba="{rgba_s}" {gattr}/>']
    for i, (sx, sy) in enumerate(((1, 1), (-1, 1), (-1, -1), (1, -1))):
        out.append(f'      <geom name="{name}_c{i}" type="cylinder" '
                   f'size="{r:.6f} {hz:.6f}" '
                   f'pos="{vec((sx * a, sy * b, zc))}" '
                   f'rgba="{rgba_s}" {gattr}/>')
    return out


def _edge_frame(key, L, W):
    """(outward unit normal, tangent unit, edge half-length, edge radius)."""
    n = _EDGES[key]
    if abs(n[0]) > 0:
        return n, np.array([0.0, 1.0, 0.0]), W / 2, L / 2
    return n, np.array([1.0, 0.0, 0.0]), L / 2, W / 2


# ---------------------------------------------------------------------------
def build(params: dict | None = None, out_dir: str | Path = ".") -> dict:
    p = merge_params(DEFAULTS, params)
    name = str(p["name"])
    L, W, H = float(p["length"]), float(p["width"]), float(p["height"])
    r, wall, ft = float(p["corner_r"]), float(p["wall"]), float(p["floor_t"])
    lid_t, over = float(p["lid_t"]), float(p["lid_over"])
    gw, gt, gc = (float(p["gasket_w"]), float(p["gasket_t"]),
                  float(p["gasket_clear"]))
    lg, lt, gap = (float(p["latch_len"]), float(p["latch_t"]),
                   float(p["latch_gap"]))
    bar_h, win_h = float(p["bar_h"]), float(p["window_h"])
    nub_out, nub_h, ramp = (float(p["nub_out"]), float(p["nub_h"]),
                            float(p["nub_ramp"]))
    rim_out, rim_h = float(p["rim_out"]), float(p["rim_h"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- the numbers that decide whether this thing can latch at all -----
    # A strap that fouls the rim, a nub that does not reach into the window,
    # or a window the nub cannot fit inside are all builds that compile,
    # render beautifully and can never be closed. Say so now, with numbers.
    if gap <= rim_out:
        raise ValueError(
            f"latch_gap {1000*gap:.1f} mm does not clear the rim it must "
            f"ride past ({1000*rim_out:.1f} mm) — the strap would have to "
            f"bow over the rim as well as the nub.")
    interf = nub_out - gap
    if interf <= 0:
        raise ValueError(
            f"nub_out - latch_gap = {1000*interf:.2f} mm: the nub never "
            f"reaches into the window, so the strap folds shut over nothing "
            f"and latches nothing. Raise nub_out or lower latch_gap.")
    if interf >= float(p["flex_travel"]):
        raise ValueError(
            f"snap interference {1000*interf:.2f} mm >= flex_travel "
            f"{1000*float(p['flex_travel']):.2f} mm — the strap cannot bow "
            f"far enough to get over its own nub, so the latch can never "
            f"close. Raise flex_travel.")
    if nub_h >= win_h - 0.001:
        raise ValueError(
            f"the nub ({1000*nub_h:.1f} mm) does not fit in the window "
            f"({1000*win_h:.1f} mm) with any play — raise window_h.")
    z_hinge = lid_t / 2.0                       # in the LID's own frame
    z_w1 = lg - bar_h                           # window bottom, below hinge
    z_w0 = z_w1 - win_h                         # window top, below hinge
    if z_w0 <= 0.002:
        raise ValueError(
            f"bar_h + window_h ({1000*(bar_h+win_h):.1f} mm) leaves no strap "
            f"above the window on a {1000*lg:.1f} mm latch — shorten them or "
            f"lengthen latch_len.")
    bar_top = H + z_hinge - z_w1                # body frame, latched
    nub_z = (float(p["nub_z"]) if p["nub_z"] is not None
             else bar_top + float(p["lock_gap"]))
    if nub_z <= ft or nub_z + nub_h + ramp >= H - rim_h:
        raise ValueError(
            f"the nub lands at z {1000*nub_z:.1f} mm on a wall that runs "
            f"floor {1000*ft:.1f} .. rim {1000*(H-rim_h):.1f} mm — change "
            f"latch_len (now {1000*lg:.1f} mm) so the window sits over the "
            f"middle of the wall")
    snap_N = float(p["flex_stiffness"]) * interf

    gl, ld = rgba(p["glass_rgba"]), rgba(p["lid_rgba"])
    gk, lc = rgba(p["gasket_rgba"]), rgba(p["latch_rgba"])
    gA = f'density="{p["glass_density"]}" friction="{p["friction"]}" condim="4"'
    pA = (f'density="{p["plastic_density"]}" friction="{p["friction"]}" '
          f'condim="4"')
    # The nub/window pair IS the mechanism: its contact must be STIFF, so the
    # force it takes to get past is the strap's own spring and not the
    # solver's squish. Everything else keeps MuJoCo's defaults.
    kA = (f'density="{p["plastic_density"]}" friction="0.5 0.02 0.001" '
          f'condim="3" solref="0.002 1" solimp="0.95 0.99 0.0005"')

    # ---------------- body: floor + wall ring + rim band + nubs ----------
    body = _slab(f"{name}_floor", L, W, r, 0.0, ft, gl, gA)
    body += _band(f"{name}_wall", L, W, r, wall, ft, H - rim_h, gl, gA)
    body += _band(f"{name}_rim", L + 2 * rim_out, W + 2 * rim_out,
                  r + rim_out, wall + rim_out, H - rim_h, H, gl, gA)
    for key in p["latch_edges"]:
        n, t, half, R = _edge_frame(key, L, W)
        lw = float(p["latch_frac"]) * 2 * half
        nw = float(p["nub_frac"]) * lw
        for tag, prot, z0, z1 in (("nub", nub_out, nub_z, nub_z + nub_h),
                                  ("ramp", nub_out / 2, nub_z + nub_h,
                                   nub_z + nub_h + ramp)):
            c = n * (R + prot / 2)
            size = (np.abs(n) * (prot / 2) + np.abs(t) * (nw / 2)
                    + np.array([0.0, 0.0, (z1 - z0) / 2]))
            body.append(
                f'      <geom name="{name}_{tag}_{key}" type="box" '
                f'size="{vec(size)}" pos="{vec((c[0], c[1], (z0 + z1) / 2))}" '
                f'rgba="{gl}" {kA if tag == "nub" else gA}/>')

    # ---------------- lid: panel + gasket ring + the four latches --------
    lid = _slab(f"{name}_panel", L + 2 * over, W + 2 * over, r + over,
                0.0, lid_t, ld, pA)
    lid += _band(f"{name}_gasket", L - 2 * (wall + gc), W - 2 * (wall + gc),
                 max(r - wall - gc, 1e-3), gw, -gt, 0.0, gk, pA)

    if p["latch_start_deg"] is not None:
        start = float(p["latch_start_deg"])
    elif str(p["lid_layout"]) == "closed":
        start = 0.0
    elif str(p["lid_layout"]) == "beside":
        # THE LID IS LYING ON A SURFACE, so its straps have fallen until they
        # touch it. That angle is not a style choice, it is arccos(hinge
        # height / strap length) — and it matters: a strap left standing at
        # 100 deg hovers 14 mm over the very edge a hand has to reach, and
        # every finger arrives through it.
        start = float(np.degrees(np.arccos(np.clip(
            (gt + z_hinge) / max(lg, 1e-6), -1.0, 1.0))))
        start = float(np.clip(start, 0.0, float(p["latch_open_deg"])))
    else:
        start = min(100.0, float(p["latch_open_deg"]))
    latches, folds, flexes = [], [], []
    for key in p["latch_edges"]:
        n, t, half, R = _edge_frame(key, L, W)
        lw = float(p["latch_frac"]) * 2 * half
        ww = float(p["window_frac"]) * lw
        hinge = n * (R + gap + lt / 2) + np.array([0.0, 0.0, z_hinge])
        axis = np.cross(n, np.array([0.0, 0.0, 1.0]))   # +q swings it OPEN
        # the body is DRAWN at `start`; ref says so, so the applied rotation
        # is qpos - start and qpos = 0 is still the latched pose
        ca, sa = np.cos(np.radians(start / 2)), np.sin(np.radians(start / 2))
        bq = [ca, sa * axis[0], sa * axis[1], sa * axis[2]]
        jn, fn = f"{name}_fold_{key}", f"{name}_flex_{key}"
        folds.append(jn)
        flexes.append(fn)

        def _bar(tag, z0, z1, y0, y1):
            """one box of the strap frame, in the strap's own plane
            (depth below the hinge z0..z1, tangential y0..y1)"""
            c = t * ((y0 + y1) / 2) + np.array([0.0, 0.0, -(z0 + z1) / 2])
            sz = (np.abs(n) * (lt / 2) + np.abs(t) * abs(y1 - y0) / 2
                  + np.array([0.0, 0.0, (z1 - z0) / 2]))
            return (f'          <geom name="{name}_{tag}_{key}" type="box" '
                    f'size="{vec(sz)}" pos="{vec(c)}" rgba="{lc}" '
                    f'{kA if tag == "barlo" else pA}/>')

        frame = "\n".join([
            _bar("barhi", 0.0, z_w0, -lw / 2, lw / 2),          # above window
            _bar("side0", z_w0, z_w1, ww / 2, lw / 2),          # window sides
            _bar("side1", z_w0, z_w1, -lw / 2, -ww / 2),
            _bar("barlo", z_w1, lg, -lw / 2, lw / 2),           # THE CATCH
            _bar("tab", lg, lg + 0.006, -lw / 2 * 0.7, lw / 2 * 0.7)])
        latches.append(f"""
      <body name="{name}_latch_{key}" pos="{vec(hinge)}"
            quat="{' '.join(f'{v:.8f}' for v in bq)}">
        <joint name="{jn}" type="hinge" axis="{vec(axis)}"
               range="-1 {p['latch_open_deg']}" ref="{start}"
               damping="{p['fold_damping']}"
               frictionloss="{p['fold_frictionloss']}"
               armature="{p['fold_armature']}"/>
        <!-- the hinge carries no geometry of its own (the strap below it is
             what bends), and MuJoCo will not move a massless body — so it
             gets a token inertial; the DOF's real inertia is fold_armature. -->
        <inertial pos="0 0 {-lg / 2:.6f}" mass="0.0005"
                  diaginertia="1e-7 1e-7 1e-7"/>
        <body name="{name}_strap_{key}">
          <joint name="{fn}" type="slide" axis="{vec(n)}"
                 range="0 {p['flex_travel']}"
                 stiffness="{p['flex_stiffness']}"
                 damping="{p['flex_damping']}"
                 armature="{p['flex_armature']}"
                 solreflimit="0.004 1" solimplimit="0.99 0.999 0.0002"/>
{frame}
        </body>
      </body>""")

    # ---------------- where the lid starts -------------------------------
    layout = str(p["lid_layout"])
    if layout in ("seated", "closed"):
        lid_pos, lid_yaw = np.array([0.0, 0.0, H]), 0.0
    elif layout == "beside":
        lid_pos = np.array([float(p["lid_offset"][0]),
                            float(p["lid_offset"][1]), gt])
        lid_yaw = float(p["lid_yaw_deg"])
    else:
        lid_pos, lid_yaw = np.array([0.0, 0.0, H + 0.12]), 0.0
    body_free = (f'      <freejoint name="{name}_body_free"/>\n'
                 if p["body_free"] else "")
    floor = FLOOR_XML if p["include_floor"] else ""

    def _xml(gd, pd):
        bx = [b.replace(f'density="{p["glass_density"]}"', f'density="{gd}"')
              .replace(f'density="{p["plastic_density"]}"', f'density="{pd}"')
              for b in body]
        lx = [b.replace(f'density="{p["plastic_density"]}"', f'density="{pd}"')
              for b in lid]
        la = [b.replace(f'density="{p["plastic_density"]}"', f'density="{pd}"')
              for b in latches]
        return mjcf_header(f"{name}_latchbox", timestep=p["timestep"]) + f"""
  <worldbody>{LIGHTS_XML}{floor}
    <body name="{name}_body" pos="{vec(p['pos'])}" euler="0 0 {p['yaw_deg']}">
{body_free}{chr(10).join(bx)}
    </body>
    <body name="{name}_lid" pos="{vec(lid_pos + np.asarray(p['pos'], float))}"
          euler="0 0 {float(p['yaw_deg']) + lid_yaw}">
      <freejoint name="{name}_lid_free"/>
{chr(10).join(lx)}{''.join(la)}
    </body>
  </worldbody>
</mujoco>
"""

    xml_path = out_dir / f"{name}.xml"
    xml_path.write_text(_xml(p["glass_density"], p["plastic_density"]))
    # densities double-count where geoms overlap, so MEASURE the compiled
    # mass and rescale once: a 1 kg glass dish that weighs 1.6 kg slides
    # differently under the same finger.
    masses = _measure_mass(xml_path, name)
    gd, pd = float(p["glass_density"]), float(p["plastic_density"])
    if p["body_mass"] and masses["body"] > 0:
        gd *= float(p["body_mass"]) / masses["body"]
    if p["lid_mass"] and masses["lid"] > 0:
        pd *= float(p["lid_mass"]) / masses["lid"]
    if (gd, pd) != (float(p["glass_density"]), float(p["plastic_density"])):
        xml_path.write_text(_xml(gd, pd))
        masses = _measure_mass(xml_path, name)

    save_params(out_dir / f"{name}_params.json", p)
    built = {
        "xml": str(xml_path),
        "params_json": str(out_dir / f"{name}_params.json"),
        "name": name,
        "body_body": f"{name}_body",
        "lid_body": f"{name}_lid",
        "fold_joints": folds,
        "flex_joints": flexes,
        "latch_edges": list(p["latch_edges"]),
        "latched_deg": 0.0,
        "open_deg": float(p["latch_open_deg"]),
        "start_deg": round(float(start), 2),
        "seat_height_m": round(H, 5),
        "nub_bottom_m": round(nub_z, 5),
        "nub_top_m": round(nub_z + nub_h, 5),
        "nub_outer_m": {k: round(float(_edge_frame(k, L, W)[3]) + nub_out, 5)
                        for k in p["latch_edges"]},
        "edge_radius_m": {k: round(float(_edge_frame(k, L, W)[3]), 5)
                          for k in p["latch_edges"]},
        "catch_bar_top_m": round(bar_top, 5),
        "snap_interference_mm": round(1000 * interf, 3),
        "snap_force_N": round(snap_N, 2),
        "lock_gap_mm": round(1000 * (nub_z - bar_top), 3),
        "strap_wall_clearance_mm": round(1000 * gap, 2),
        "body_mass_kg": round(masses["body"], 4),
        "lid_mass_kg": round(masses["lid"], 4),
        "inner_volume_l": round(1000 * (L - 2 * wall) * (W - 2 * wall)
                                * (H - ft), 3),
        "note": (
            f"fold joints: 0 deg = LATCHED, {p['latch_open_deg']:.0f} deg = "
            f"splayed open, and the flex slide beside each is the strap's own "
            f"springiness. The wall's nub stands {1000*interf:.2f} mm proud "
            f"of where the strap wants to lie, so popping it through the "
            f"window costs {snap_N:.1f} N against that spring — the same in "
            f"both directions, which is why a latched lid does not lift off. "
            f"A latch can only close when the lid is SEATED: the catch bar "
            f"sits {1000*(nub_z-bar_top):.1f} mm under the nub, so a lid held "
            f"even 2 mm proud cannot reach it."),
    }
    (out_dir / f"{name}_built.json").write_text(json.dumps(built, indent=1))
    return built


def _measure_mass(xml_path, name) -> dict:
    m = load(xml_path)
    out = {}
    for k, b in (("body", f"{name}_body"), ("lid", f"{name}_lid")):
        try:
            out[k] = float(m.body(b).subtreemass[0])
        except Exception:
            out[k] = 0.0
    return out


def rebuild(params_json: str | Path, out_dir: str | Path | None = None) -> dict:
    params_json = Path(params_json)
    return build(json.loads(params_json.read_text()),
                 out_dir or params_json.parent)


# ---------------------------------------------------------------------------
# state: setting a layout, and MEASURING what the latches are actually doing
# ---------------------------------------------------------------------------
def _bname(model, suffix):
    for i in range(model.nbody):
        n = model.body(i).name or ""
        if n == suffix or n.endswith("_" + suffix) or n.endswith(suffix):
            return n
    raise KeyError(suffix)


def _gname(model, suffix):
    for i in range(model.ngeom):
        n = model.geom(i).name or ""
        if n == suffix or n.endswith(suffix):
            return n
    raise KeyError(suffix)


def _joint(model, suffix):
    for i in range(model.njnt):
        n = model.joint(i).name or ""
        if n == suffix or n.endswith(suffix):
            return int(model.joint(i).qposadr[0])
    return None


def _free_adr(model, body_suffix):
    import mujoco
    try:
        b = model.body(_bname(model, body_suffix))
    except KeyError:
        return None
    for j in range(int(b.jntnum[0])):
        jid = int(b.jntadr[0]) + j
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            return int(model.jnt_qposadr[jid])
    return None


def set_layout(model, data, built: dict, layout: str = "beside",
               latch_deg: float | None = None) -> None:
    """Put the asset into a named start state, kinematically.

    'beside'  lid flat on the table beside the body, latches splayed — the
              clip's own frame 0, and the state the task starts from
    'seated'  lid on the body, latches still open — the halfway state
    'closed'  lid on the body, all four latched
    Joint and body names are prefixed when the asset is attached into a
    world, so every lookup here is by suffix.
    """
    import mujoco
    q = float(built["open_deg"] if latch_deg is None else latch_deg)
    if layout == "closed" and latch_deg is None:
        q = 0.0
    for jn in built["fold_joints"]:
        a = _joint(model, jn)
        if a is not None:
            data.qpos[a] = np.radians(q)
    for jn in built["flex_joints"]:
        a = _joint(model, jn)
        if a is not None:
            data.qpos[a] = 0.0
    if layout in ("seated", "closed"):
        lj, bj = _free_adr(model, built["lid_body"]), \
            _free_adr(model, built["body_body"])
        if lj is not None:
            if bj is not None:
                base, quat = data.qpos[bj:bj + 3].copy(), \
                    data.qpos[bj + 3:bj + 7].copy()
            else:
                b = model.body(_bname(model, built["body_body"]))
                base, quat = np.asarray(b.pos).copy(), np.asarray(b.quat).copy()
            data.qpos[lj:lj + 3] = base + np.array(
                [0.0, 0.0, built["seat_height_m"]])
            data.qpos[lj + 3:lj + 7] = quat
    mujoco.mj_forward(model, data)


def latch_state(model, data, built: dict) -> dict:
    """THE GATE — per latch, and it asks the question that can fail.

    "Is the strap folded down?" is not the question: a strap folded against a
    box whose lid is 10 mm proud looks identical and holds nothing. What is
    measured instead, in the BODY's own frame so it survives the assembly
    being carried about, is whether the wall's NUB is actually inside the
    strap's WINDOW: the catch bar under the nub, and still overlapping it
    radially. Both, because either alone passes for free.
    """
    import mujoco
    mujoco.mj_forward(model, data)
    bb = model.body(_bname(model, built["body_body"]))
    pb, Rb = data.xpos[bb.id].copy(), data.xmat[bb.id].reshape(3, 3).copy()
    out, n_locked = {}, 0
    for key in built["latch_edges"]:
        fold = _joint(model, f"{built['name']}_fold_{key}")
        flex = _joint(model, f"{built['name']}_flex_{key}")
        g = model.geom(_gname(model, f"{built['name']}_barlo_{key}"))
        gp = Rb.T @ (data.geom_xpos[g.id] - pb)          # catch bar, body frame
        gR = Rb.T @ data.geom_xmat[g.id].reshape(3, 3)
        sz = np.asarray(model.geom_size[g.id], float)
        n = _EDGES[key]
        top = float(gp[2] + float(np.abs(gR[2]) @ sz))
        inner = float(gp[:3] @ n - float(np.abs(n @ gR) @ sz))
        deg = float(np.degrees(data.qpos[fold])) if fold is not None else None
        bow = float(1000 * data.qpos[flex]) if flex is not None else None
        overlap = 1000.0 * (built["nub_outer_m"][key] - inner)
        under = bool(top <= built["nub_bottom_m"] + 0.0015)
        engaged = bool(under and overlap > 0.2 and deg is not None
                       and abs(deg) < 15.0)
        n_locked += int(engaged)
        out[key] = {"fold_deg": round(deg, 2) if deg is not None else None,
                    "strap_bow_mm": round(bow, 3) if bow is not None else None,
                    "catch_bar_top_m": round(top, 5),
                    "bar_under_nub": under,
                    "nub_window_overlap_mm": round(overlap, 2),
                    "latched": engaged}
    return {"latches": out, "n_latched": n_locked,
            "n_total": len(built["latch_edges"]),
            "all_latched": n_locked == len(built["latch_edges"])}


def set_folds(model, data, built: dict, degs) -> None:
    """Put the fold joints where you want them, per latch.

    `degs` is either one angle for all of them or {edge_key: deg}. 0 = latched,
    `open_deg` = splayed. Needed because a scene mid-task has latches in
    DIFFERENT states — the two already shut, the two still standing out —
    and a pose solved against "all four splayed" is solved against geometry
    that is not there any more.
    """
    import mujoco
    if not isinstance(degs, dict):
        degs = {k: float(degs) for k in built["latch_edges"]}
    for key, q in degs.items():
        a = _joint(model, f"{built['name']}_fold_{key}")
        if a is not None:
            data.qpos[a] = np.radians(float(q))
    mujoco.mj_forward(model, data)


def latch_hinge(built: dict, key: str, params: dict | None = None):
    """(hinge point in the LID's own frame, fold axis) for one latch.

    THE HINGE IS THE WHOLE MECHANISM'S ONE DEGREE OF FREEDOM, so a hand that
    closes a latch by rotating rigidly about this line keeps its pads on the
    strap EXACTLY, at every angle, with no sliding — which is what makes the
    closing motion a single smooth rotation instead of a sequence of
    re-solved poses that each slip somewhere else on the tab.
    """
    p = params or json.loads(Path(built["params_json"]).read_text())
    n = _EDGES[key]
    off = (float(built["edge_radius_m"][key]) + float(p["latch_gap"])
           + float(p["latch_t"]) / 2.0)
    hinge = n * off + np.array([0.0, 0.0, float(p["lid_t"]) / 2.0])
    return hinge, np.cross(n, np.array([0.0, 0.0, 1.0]))


def latch_press_targets(model, data, built: dict, key: str,
                        fingers=("index", "middle"), spread_m: float = 0.011,
                        part: str = "barlo", prefix: str = ""):
    """Where a hand should put its pads to close ONE latch, and which way.

    Read off the strap's ACTUAL geom frame at whatever angle it is folded to,
    not derived from the number sheet: the strap is on a hinge, so where its
    outer face is depends on how far it has already been folded, and that is
    the difference between pressing a tab and pressing the air beside it.

    THE STRAP'S THIN AXIS IS NOT ALWAYS THE GEOM'S LOCAL y. It is whichever
    half-size is smallest — the box is built in the edge's own (n, t) frame,
    so the thin direction is local x on the xp/xn straps and local y on
    yp/yn. Assuming y silently returns points 32 mm out in mid-air on half
    the latches.
    """
    g = model.geom(_gname(model, f"{prefix}{built['name']}_{part}_{key}"))
    gp = np.asarray(data.geom_xpos[g.id], float)
    R = np.asarray(data.geom_xmat[g.id], float).reshape(3, 3)
    half = np.asarray(model.geom_size[g.id], float)
    thin = int(np.argmin(half))
    lid = model.body(_bname(model, built["lid_body"]))
    lc = np.asarray(data.xpos[lid.id], float)
    nrm = R[:, thin]
    if float((gp - lc) @ nrm) < 0:
        nrm = -nrm
    # the wide axis of the strap face — the other one that is not "down it"
    down = -R[:, 2]                      # the strap runs along its own -z
    tang = np.cross(down, nrm)
    tang /= np.linalg.norm(tang) + 1e-12
    face = gp + nrm * float(half[thin])
    k = (len(fingers) - 1) / 2.0
    pts = {f: face + tang * ((i - k) * spread_m)
           for i, f in enumerate(fingers)}
    nms = {f: nrm.copy() for f in fingers}
    return pts, nms, {"face": face, "normal": nrm, "tangent": tang,
                      "down": down}


def lid_seating(model, data, built: dict) -> dict:
    """How the lid sits on the body: 0 mm proud = fully home on the rim."""
    import mujoco
    mujoco.mj_forward(model, data)
    bb = model.body(_bname(model, built["body_body"]))
    lb = model.body(_bname(model, built["lid_body"]))
    d = data.xmat[bb.id].reshape(3, 3).T @ (data.xpos[lb.id] - data.xpos[bb.id])
    return {"lid_over_body_mm": [round(1000 * float(v), 2) for v in d[:2]],
            "lid_proud_mm": round(1000 * float(d[2] - built["seat_height_m"]),
                                  2),
            "seated": bool(abs(d[0]) < 0.006 and abs(d[1]) < 0.006
                           and abs(d[2] - built["seat_height_m"]) < 0.004)}


# ---------------------------------------------------------------------------
# preview + the FORCE-DRIVEN demo
# ---------------------------------------------------------------------------
def _cam(params, az=126.0, el=-22.0):
    L = float(params["length"])
    return free_camera([0.10 * L, 0.0, 0.6 * float(params["height"])],
                       2.5 * L, az, el)


def preview_stills(built: dict, out_dir: str | Path) -> list[str]:
    """Three stills — lid beside, lid seated with latches open, all four
    latched — rendered from the built asset, so what is attached is what is
    shown."""
    import mujoco
    from PIL import Image
    from tool_library.common.mjcf import _label
    p = json.loads(Path(built["params_json"]).read_text())
    m, d = load(built["xml"]), None
    d = mujoco.MjData(m)
    paths = []
    with mujoco.Renderer(m, height=520, width=680) as r:
        for i, (lab, layout, q) in enumerate(
                (("lid beside, latches open", "beside", None),
                 ("lid seated, latches still open", "seated", None),
                 ("all four latched", "closed", 0.0))):
            mujoco.mj_resetData(m, d)
            set_layout(m, d, built, layout, latch_deg=q)
            r.update_scene(d, camera=_cam(p))
            f = Path(out_dir) / (f"{built['name']}_preview_{i}_"
                                 f"{lab.split(',')[0].replace(' ', '_')}.png")
            Image.fromarray(_label(r.render().copy(), lab)).save(f)
            paths.append(str(f))
    return paths


class LatchDriver:
    """Force-only closing of one latch, as a PINCH.

    The force is applied at the strap's free end along the STRAP's OWN
    outward normal — which rotates with the strap. Pushing along the box's
    fixed normal instead is the trap: a strap splayed at 115 deg is above its
    own hinge, so a horizontal inward push there OPENS it further. Measured:
    that version left all four sitting at 90 deg, having driven them open.

    The equal and opposite force goes on the body, so the pair is a finger
    and a thumb and not a rocket: closing a latch pushes nothing anywhere.
    """

    def __init__(self, model, data, built, key, force_N=10.0):
        self.m, self.built, self.key = model, built, key
        self.body = model.body(_bname(model, f"{built['name']}_latch_{key}")).id
        self.fold = _joint(model, f"{built['name']}_fold_{key}")
        self.flex = _joint(model, f"{built['name']}_flex_{key}")
        self.body_b = model.body(_bname(model, built["body_body"])).id
        self.f = float(force_N)
        self.len = float(json.loads(
            Path(built["params_json"]).read_text())["latch_len"])
        self.peak_bow = 0.0

    def push(self, data, sign=-1.0):
        """sign -1 pushes the strap CLOSED (inward), +1 prises it open."""
        R = data.xmat[self.body].reshape(3, 3)
        n = R @ _EDGES[self.key]                     # the STRAP's own normal
        tip = data.xpos[self.body] + R @ np.array([0.0, 0.0, -self.len])
        apply_wrench(data, self.body, force=sign * self.f * n,
                     point=tip, com=data.xipos[self.body])
        apply_wrench(data, self.body_b, force=-sign * self.f * n)
        self.peak_bow = max(self.peak_bow, float(data.qpos[self.flex]))


def _pd_lift(data, body_id, z_target, kp=250.0, kd=60.0, mass_hint=1.6):
    """A FORCE that lifts, not a constraint that teleports: gravity
    compensation plus a capped PD toward a height. The closest honest
    analogue of a hand picking the lid up, and it cannot drag the box along
    unless the latches really carry it."""
    z = float(data.xpos[body_id][2])
    v = float(data.cvel[body_id][5])
    cap = 2.2 * mass_hint * 9.81
    f = mass_hint * 9.81 + kp * (z_target - z) - kd * v
    apply_wrench(data, body_id, force=[0.0, 0.0, float(np.clip(f, -cap, cap))])


def demo(built: dict, out_mp4: str | Path, fps: int = 30,
         seconds: tuple = (1.4, 3.0, 2.6, 2.0, 2.2),
         close_force_N: float = 10.0) -> dict:
    """The build's own proof, in five measured phases and one video:

      seat   the lid is dropped onto the body — it seats, and NOTHING holds
             it (n_latched = 0)
      close  each latch is PINCHED shut by a force at its tip; the strap bows
             out over the nub and springs back (the snap, measured in mm)
      lift   the whole box is lifted BY THE LID: the body has to come with
             it, which is the only proof the latches bear load
      shake  the body is yanked sideways while it hangs
      open   the same pinch, reversed — the latches let go

    Everything moves by gravity and applied wrenches; the asset has no
    actuators, so none of this can be produced by a servo.
    """
    import mujoco
    from tool_library.common.mjcf import _label
    p = json.loads(Path(built["params_json"]).read_text())
    m = load(built["xml"])
    d = mujoco.MjData(m)
    set_layout(m, d, built, "seated")
    lj = _free_adr(m, built["lid_body"])
    d.qpos[lj + 2] += 0.012                  # let its own weight do the seating
    mujoco.mj_forward(m, d)
    drivers = {k: LatchDriver(m, d, built, k, force_N=close_force_N)
               for k in built["latch_edges"]}
    lid_id = m.body(_bname(m, built["lid_body"])).id
    body_id = m.body(_bname(m, built["body_body"])).id
    z_start = float(d.xpos[body_id][2])
    total_kg = built["body_mass_kg"] + built["lid_mass_kg"]
    t0, t1, t2, t3, t4 = np.cumsum(seconds)
    z_lid0 = float(d.xpos[lid_id][2])
    lift_to = z_lid0 + 0.10

    def control(t, mm, dd):
        clear_forces(dd)
        if t < t0:
            return "seat: the lid is dropped on — gravity only, nothing holds it"
        if t < t1:
            for k in built["latch_edges"]:
                drivers[k].push(dd, -1.0)
            return (f"close: a {close_force_N:.0f} N pinch at each strap tip; "
                    f"the window must pop over a nub standing "
                    f"{built['snap_interference_mm']:.1f} mm proud "
                    f"({built['snap_force_N']:.1f} N)")
        if t < t2:
            u = float(np.clip((t - t1) / max(t2 - t1, 1e-6), 0.0, 1.0))
            _pd_lift(dd, lid_id, z_lid0 + (lift_to - z_lid0)
                     * (10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5),
                     mass_hint=total_kg)
            return "lift BY THE LID: a latched box comes up with it"
        if t < t3:
            _pd_lift(dd, lid_id, lift_to, mass_hint=total_kg)
            g = 9.81 * total_kg
            apply_wrench(dd, body_id, force=[0.45 * g * np.sin(5.0 * t),
                                             0.35 * g * np.cos(4.0 * t), 0.0])
            return ("shake: +-0.45 g sideways while it hangs — the nubs "
                    "stay in the windows")
        for k in built["latch_edges"]:
            drivers[k].push(dd, +1.0)
        return "open: the same pinch, reversed — the latches let go"

    def snap():
        return {**latch_state(m, d, built), **lid_seating(m, d, built),
                "body_z_m": round(float(d.xpos[body_id][2]), 4)}

    frames, marks = [], {}
    phase_end = {"seated": t0, "closed": t1, "lifted": t2, "shaken": t3}
    dt = float(m.opt.timestep)
    spf = max(1, int(round((1.0 / fps) / dt)))
    with mujoco.Renderer(m, height=520, width=680) as r:
        for i in range(int(round(t4 * fps))):
            lab = None
            for _ in range(spf):
                lab = control(d.time, m, d) or lab
                mujoco.mj_step(m, d)
            if not np.isfinite(d.qpos).all():
                raise RuntimeError(f"latchbox demo went non-finite at "
                                   f"t={d.time:.3f}s")
            for tag, tt in phase_end.items():        # at the END of a phase
                if tag not in marks and d.time >= tt - 1.0 / fps:
                    marks[tag] = snap()
            cam = _cam(p)
            cam.lookat[:] = 0.5 * (d.xpos[body_id] + d.xpos[lid_id])
            r.update_scene(d, camera=cam)
            frames.append(_label(r.render().copy(), lab or ""))
    end = snap()
    n_edges = len(built["latch_edges"])
    rose = round(1000 * (marks.get("lifted", {}).get("body_z_m", z_start)
                         - z_start), 1)
    meas = {
        "after_seat": {k: marks.get("seated", {}).get(k)
                       for k in ("seated", "lid_proud_mm", "n_latched")},
        "after_close": {**{k: marks.get("closed", {}).get(k)
                           for k in ("n_latched", "all_latched", "seated",
                                     "lid_proud_mm")},
                        "peak_strap_bow_mm": {k: round(1000 * v.peak_bow, 2)
                                              for k, v in drivers.items()}},
        "after_lift": {"n_latched": marks.get("lifted", {}).get("n_latched"),
                       "body_rose_mm": rose,
                       "lid_proud_mm": marks.get("lifted", {}).get(
                           "lid_proud_mm")},
        "after_shake": {k: marks.get("shaken", {}).get(k)
                        for k in ("n_latched", "all_latched", "lid_proud_mm")},
        "after_prise_open": {"n_latched": end["n_latched"],
                             "fold_deg": {k: v["fold_deg"] for k, v
                                          in end["latches"].items()}},
        "snap_force_N": built["snap_force_N"],
        "close_force_N": close_force_N,
    }
    ok_close = marks.get("closed", {}).get("n_latched", 0) == n_edges
    ok_lift = (marks.get("lifted", {}).get("n_latched", 0) >= n_edges - 1
               and rose > 40)
    ok_hold = marks.get("shaken", {}).get("n_latched", 0) >= n_edges - 1
    ok_open = end["n_latched"] == 0
    fails = [t for t, ok in (("did not latch", ok_close),
                             ("the box did not come up with the lid", ok_lift),
                             ("shook loose", ok_hold),
                             ("did not release when prised", ok_open))
             if not ok]
    verdict = (f"all {n_edges} latch under a {close_force_N:.0f} N pinch "
               f"(snap {built['snap_force_N']:.1f} N), carry the "
               f"{total_kg:.2f} kg box {rose:.0f} mm off the table by the lid "
               f"alone, survive a shake, and release when prised"
               if not fails else "MECHANISM FAILED: " + ", ".join(fails))
    out = write_video(frames, out_mp4, fps=fps)
    out["stills"] = save_stills(frames, Path(out_mp4).parent,
                                f"{built['name']}_demo", n=5)
    out["measured"] = meas
    out["verdict"] = verdict
    return out


# ---------------------------------------------------------------------------


def construct_latch_box(out_dir, preview=False, **params):
    """Build the asset into out_dir from the parameters above (any key of
    DEFAULTS; unknown keys raise). Returns the build dictionary."""
    built = build(params, out_dir)
    if preview:
        built["stills"] = preview_stills(built, out_dir)
    return built


if __name__ == "__main__":
    import json as _json
    import sys as _sys
    out = _sys.argv[1] if len(_sys.argv) > 1 else "."
    print(_json.dumps(construct_latch_box(out, preview=True), indent=1, default=str))
