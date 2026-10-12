"""Construct bottle with screw cap and drinking straw: the threaded bottle of construct_bottle plus a drinking tube that hangs from the cap into the body and leaves with the cap.

BOTTLE + SCREW CAP — a parametric threaded asset that behaves like a thread.

THE POINT
---------
A cap that is merely "placed on top and pushed down" is not a bottle. What
makes opening a bottle a task is that lift and rotation are LOCKED TOGETHER:
you cannot pull the cap off without turning it, and you cannot push it on
without turning it. This builder encodes exactly that, and nothing else about
the mechanism is left to the demonstrator's luck.

HOW THE THREAD IS MODELLED
--------------------------
MuJoCo has no screw joint, and URDF cannot express one at all — which is why
this asset is MJCF, not URDF. The mechanism is three parts:

  1. a `carrier` body inside the bottle carrying TWO scalar joints —
     `<name>_yaw`   (hinge about the bottle axis)
     `<name>_lift`  (slide  along the bottle axis)
  2. an equality constraint  lift = (pitch / 2pi) * yaw  — the screw itself.
     One turn buys exactly `thread_pitch` metres of travel. Nothing else can
     produce travel: push straight down on a stationary cap and the constraint
     absorbs the force.
  3. an equality WELD from the carrier to the cap, which is a real free body.
     Deactivating that weld (`data.eq_active[thread_eq] = 0`) is the thread
     running out: the cap becomes an ordinary 6-DoF rigid body mid-simulation,
     with whatever velocity it had, and falls.

  4. DRY FRICTION on the yaw joint, because a frictionless helix is just a
     ramp: push axially and the constraint happily converts the force into
     rotation and screws itself down (measured: 20 N drove the frictionless
     version 22 mm down the thread at 27 rad/s). Real threads are
     self-locking — 3.4 deg lead angle against a ~17 deg friction angle — so
     this one is too. Measured on the final asset: 20 N of straight push or
     pull moves the cap 0.18 mm of its 25 mm travel; 100 mN*m of twist takes
     it the whole way in 5 turns.

The helical ridge on the neck is a visual mesh. It has to be: MuJoCo collides
meshes by their convex hull, so a helix cannot be a collision geom. The
geometry is what a person (or a VLM looking at the render) reads as a thread;
the constraint is what a policy has to solve.

The asset has NO ACTUATORS. Nothing in it can pose the mechanism; the only way
to open it is to apply a torque to the cap (see tool_library/common/drivers.py).

FITTING THIS TO A REAL CLIP
---------------------------
Every number is in `<name>_params.json` next to the built asset. Measure the
bottle in the video, edit those numbers, call `rebuild(json_path)`. No code
edit fits an asset to a clip.

    python -m tool_library.skills.construct_bottle_straw [out_dir]      # build + preview stills
"""
from __future__ import annotations

TOOL = {'name': 'construct_bottle_straw',
 'title': 'Construct bottle with screw cap and drinking straw',
 'type': 'skill',
 'category': 'Object Reconstruction',
 'summary': 'A threaded screw-cap bottle (sports bottle) whose cap carries a drinking tube hanging down '
            'into the body, so lifting the cap off withdraws the tube through the mouth.',
 'description': 'Fork of construct_bottle that adds a drinking straw: a thin tube of straw_length '
                'hanging from the cap ceiling down the bottle axis, part of the cap body (it moves '
                'with the cap, rotates with it while unscrewing and leaves with it). The straw is '
                'visual-only (no collider), so it never jams the thread; the build reports '
                'straw_clear_lift_m, the cap lift above its seat at which the straw end leaves the '
                'mouth, which a planner must exceed before moving the cap sideways. straw_length 0 '
                'gives the plain bottle. A straw longer than the interior raises ValueError. '
                'Builds the bottle as a lathe mesh over primitive colliders and the cap as a '
                'free body with a hollow skirt, from one dictionary of numbers: body radius '
                'and height, shoulder and neck, cap size and wall, thread pitch and turns, '
                'colours, masses, friction. The thread is modelled, because MuJoCo has no '
                'screw joint and URDF cannot express one: a carrier body inside the bottle '
                'carries a yaw hinge and a lift slide, an equality constraint locks lift = '
                'pitch/2pi x yaw, dry friction on the yaw joint makes the thread self-locking '
                '(an axial push or pull cannot back-drive it), and a weld between carrier and '
                'cap is the engagement, which disengages when the cap has travelled '
                'thread_pitch x thread_turns. A visible helical ridge is drawn on the neck. '
                'The asset has no actuators; the only way to open it is torque on the cap. '
                'Every parameter is written to <name>_params.json so a build can be repeated '
                'or refitted by editing numbers. The thread_self_locking_test verifier proves '
                'a built bottle.',
 'inputs': [{'name': 'out_dir',
             'type': 'path',
             'description': 'Folder that receives the MJCF, its meshes and '
                            '<name>_params.json.'},
            {'name': 'preview',
             'type': 'bool',
             'description': 'Also render preview stills of the mechanism at its extremes. '
                            'Default False.'},
            {'name': 'name',
             'type': 'str',
             'description': 'Asset name; prefixes every body, joint and mesh. Default bottle'},
            {'name': 'body_radius',
             'type': 'float',
             'description': 'Widest radius of the barrel, metres. Default 0.035'},
            {'name': 'body_height',
             'type': 'float',
             'description': 'Barrel height from base to shoulder, metres. Default 0.15'},
            {'name': 'shoulder_height',
             'type': 'float',
             'description': 'Taper height from shoulder to neck, metres. Default 0.045'},
            {'name': 'neck_radius',
             'type': 'float',
             'description': 'Threaded neck radius (the thread crest, what a caliper measures), '
                            'metres. Default 0.0135'},
            {'name': 'neck_height',
             'type': 'float',
             'description': 'Neck height from the taper to the mouth, metres. Default 0.022'},
            {'name': 'cap_radius',
             'type': 'float',
             'description': 'Cap outer radius, metres. Default 0.016'},
            {'name': 'cap_height',
             'type': 'float',
             'description': 'Cap height, metres. Default 0.024'},
            {'name': 'cap_wall',
             'type': 'float',
             'description': 'Cap side-wall thickness, metres. Default 0.002'},
            {'name': 'cap_top_wall',
             'type': 'float | None',
             'description': 'Cap ceiling thickness, metres; None uses cap_wall. cap_height '
                            'minus this is the engagement depth'},
            {'name': 'thread_pitch',
             'type': 'float',
             'description': 'Metres of lift per full turn of the cap. Default 0.005'},
            {'name': 'thread_turns',
             'type': 'float',
             'description': 'Turns from seated to disengaged. Default 5.0'},
            {'name': 'thread_right_handed',
             'type': 'bool',
             'description': 'True: counter-clockwise (seen from above) unscrews, like a real '
                            'bottle. Default True'},
            {'name': 'thread_ridge_radius',
             'type': 'float',
             'description': 'Visual ridge tube radius. Default 0.0011'},
            {'name': 'body_rgba',
             'type': 'list[4]',
             'description': 'Bottle colour as rgba in 0..1; alpha below 1 renders a '
                            'translucent bottle. Default [0.35, 0.62, 0.88, 0.45]'},
            {'name': 'cap_rgba',
             'type': 'list[4]',
             'description': 'Cap colour as rgba. Default [0.2, 0.45, 0.85, 1.0]'},
            {'name': 'thread_rgba',
             'type': 'list[4]',
             'description': 'Visible thread ridge colour as rgba. Default [0.3, 0.58, 0.84, '
                            '0.85]'},
            {'name': 'cap_stripe_count',
             'type': 'int',
             'description': 'Visual grip ribs on the cap so its rotation reads on camera; 0 '
                            'for none. Default 0'},
            {'name': 'cap_stripe_rgba',
             'type': 'list[4]',
             'description': 'Cap stripe rgba, colour as rgba in 0..1. Default [0.62, 0.62, '
                            '0.6, 1.0]'},
            {'name': 'pos',
             'type': 'list[3]',
             'description': 'Bottle base centre in the world, metres. Default [0.0, 0.0, 0.0]'},
            {'name': 'yaw_deg',
             'type': 'float',
             'description': 'Bottle yaw, degrees. Default 0.0'},
            {'name': 'body_mass',
             'type': 'float',
             'description': 'Bottle mass, kilograms. Default 0.3'},
            {'name': 'cap_mass',
             'type': 'float',
             'description': 'Cap mass, kilograms. Default 0.0035'},
            {'name': 'friction',
             'type': 'list[3]',
             'description': 'Contact friction triple of every bottle and cap geom. Default '
                            '[1.8, 0.02, 0.001]'},
            {'name': 'contact_solref',
             'type': 'list[2]',
             'description': 'Contact solref, MuJoCo solref. Default [0.004, 1.0]'},
            {'name': 'yaw_frictionloss',
             'type': 'float',
             'description': 'Dry friction on the thread rotation, N*m; what makes the thread '
                            'self-locking. Default 0.02'},
            {'name': 'yaw_damping',
             'type': 'float',
             'description': 'Sets the spin rate under a given twist. Default 0.008'},
            {'name': 'lift_damping',
             'type': 'float',
             'description': 'Lift damping, joint damping. Default 0.002'},
            {'name': 'cap_skirt_segments',
             'type': 'int',
             'description': 'Hollow cap skirt = N boxes in a ring. Default 12'},
            {'name': 'bottle_fixed',
             'type': 'bool',
             'description': 'True: the bottle is fixed to the world (another hand holds it); '
                            'False: it is a free body. Default True'},
            {'name': 'timestep',
             'type': 'float',
             'description': 'Timestep, seconds. Default 0.00025'},
            {'name': 'thread_solref',
             'type': 'list[2]',
             'description': 'Thread solref, MuJoCo solref. Default [0.0005, 1.0]'},
            {'name': 'thread_solimp',
             'type': 'list[3]',
             'description': 'Thread solimp, MuJoCo solimp. Default [0.99, 0.9999, 1e-05]'},
            {'name': 'carrier_mass',
             'type': 'float',
             'description': 'Apparent inertia of the thread carrier. Default 0.02'},
            {'name': 'carrier_inertia',
             'type': 'float',
             'description': 'Carrier inertia. Default 0.0001'},
            {'name': 'straw_length', 'type': 'float', 'description': 'Drinking tube length below the cap ceiling, metres; 0 = no straw. Default 0.0'},
            {'name': 'straw_radius', 'type': 'float', 'description': 'Drinking tube outer radius, metres. Default 0.0035'},
            {'name': 'straw_rgba', 'type': 'list[float]', 'description': 'Drinking tube colour (clear plastic). Default [0.85, 0.92, 0.95, 0.35]'},
            {'name': 'include_floor',
             'type': 'bool',
             'description': 'Add a floor plane to the standalone asset. Default True'}],
 'outputs': [{'name': 'built',
              'type': 'dict',
              'description': 'xml: the MJCF path; params_json: every parameter as built; the '
                             'names of the bodies, joints and equality constraints the '
                             'mechanism exposes, and the derived measurements (sizes, travel, '
                             'masses); straw: geom name, length, bottom height when seated and straw_clear_lift_m; stills when preview is set.'}],
 'requirements': ['mujoco>=3.2', 'numpy', 'trimesh', 'imageio', 'Pillow'],
 'simulators': ['MuJoCo'],
 'tags': ['bottle', 'screw cap', 'straw', 'drinking tube', 'sports bottle', 'thread', 'articulated', 'parametric', 'reconstruct', 'object reconstruction'],
 'used_in': ['Open the Bottle'],
 'status': 'tested',
 'entrypoint': 'construct_bottle_straw',
 'cli': 'python -m tool_library.skills.construct_bottle_straw out_dir'}


import numpy as np
from pathlib import Path

from tool_library.common.mjcf import (FLOOR_XML, LIGHTS_XML, free_camera, helix_tube, load,
                     merge_params, mjcf_header, revolve_profile, rgba,
                     rollout, save_params, save_stills, vec, write_mesh,
                     write_video)

# ---------------------------------------------------------------------------
# DEFAULTS — every one of these is a number you may change per clip.
# Lengths metres, mass kg, angles degrees. Colours rgba 0..1.
# ---------------------------------------------------------------------------
DEFAULTS = {
    "name": "bottle",

    # ---- size (measure these off the video) ----
    "body_radius":      0.035,   # widest radius of the bottle barrel
    "body_height":      0.150,   # barrel height (base -> shoulder)
    "shoulder_height":  0.045,   # taper height (shoulder -> neck)
    "neck_radius":      0.0135,  # threaded neck radius
    "neck_height":      0.022,   # neck height (taper top -> mouth)
    "cap_radius":       0.0160,  # cap outer radius
    "cap_height":       0.0240,  # cap height
    "cap_wall":         0.0020,  # cap SIDE wall thickness
    # ceiling thickness; None -> cap_wall. inner depth = cap_height - this,
    # and inner depth is the ENGAGEMENT: how far the neck reaches into the
    # cap, i.e. how far the cap must travel before it is free. A deep
    # decorative cap (a wide-mouth sports bottle's 40 mm skirt) engages only
    # its top ~18 mm — model that by a thick ceiling, NOT by a 40 mm inner
    # depth: with the whole skirt hollow the neck would have to fill it, the
    # shoulder cylinder ends up INSIDE the skirt wall, and the seated cap is
    # radially jammed on the shoulder (measured: 100 mN*m stalled at 0.3
    # turns on a cap whose skirt overlapped the shoulder by 1.5 mm).
    "cap_top_wall":     None,

    # ---- thread ----
    "thread_pitch":     0.005,   # metres of lift per FULL turn
    "thread_turns":     5.0,     # turns from seated to disengaged
    "thread_right_handed": True, # True: counter-clockwise (seen from above)
                                 #   unscrews, like every real bottle
    "thread_ridge_radius": 0.0011,   # visual ridge tube radius

    # ---- appearance ----
    "body_rgba":  [0.35, 0.62, 0.88, 0.45],   # translucent PET by default
    "cap_rgba":   [0.20, 0.45, 0.85, 1.0],
    "thread_rgba": [0.30, 0.58, 0.84, 0.85],
    # N visual-only stripes on the cap's side (a real cap's mould seam /
    # grip ribs). A plain cap is ROTATIONALLY SYMMETRIC: it can spin 4.8
    # turns on camera and render identically to a cap standing still, so a
    # correct unscrew is indistinguishable from a straight lift. 0 = none.
    "cap_stripe_count": 0,
    "cap_stripe_rgba": [0.62, 0.62, 0.60, 1.0],

    # ---- placement in the world ----
    "pos":      [0.0, 0.0, 0.0],   # bottle base centre
    "yaw_deg":  0.0,

    # ---- physics ----
    "body_mass":        0.300,   # a full 500 ml bottle
    "cap_mass":         0.0035,
    # RUBBER PADS ON A PLASTIC BOTTLE. 1.0 is a generic slippery default and
    # it is the reason the bottle rotates in the hand while the other hand
    # unscrews the cap: a wrap covering only ~160 deg of a smooth cylinder
    # resists twist through FRICTION alone, so at mu=1.0 the hand has to make
    # up the difference by clamping -- measured 1675-3100 N of contact load on
    # the barrel, and the bottle still twisted ~10 deg on every bite.
    # Sharpa's fingertips are rubber and the bottle is plastic; mu 1.8 is what
    # that pair actually gives, and it roughly doubles the torque the same
    # grip can hold without squeezing any harder.
    "friction":         [1.8, 0.02, 0.001],
    # A 3.5 g cap is split across 13 small collision geoms, and MuJoCo's
    # contact stiffness scales with the mass it acts on, so at the default
    # softness the loose cap SINKS 9.5 mm into the table it lands on — half
    # its own height, and visible in the render as a cap embedded in the
    # tabletop. Same fix as the thread's contact_solref.
    "contact_solref":   [0.004, 1.0],
    # THREAD FRICTION — what makes a bottle cap stay on.
    #
    # A frictionless helix is a RAMP: push axially on the cap and the
    # constraint converts that force straight into rotation, so the cap
    # obligingly spins itself down the thread. Measured on the frictionless
    # version: 20 N of straight push drove the cap 22 mm down the thread at
    # 27 rad/s, i.e. "push it on without turning" WORKED, which is the exact
    # behaviour this asset exists to forbid.
    #
    # Real threads are self-locking: lead angle atan(pitch / 2*pi*r) = 3.4 deg
    # for a 5 mm pitch on a 13.5 mm neck, against a plastic friction angle of
    # ~17 deg. So the thread is modelled with DRY FRICTION on its rotation.
    # An axial force F back-drives with torque F * pitch/2pi, so this number
    # is the axial force the cap resists: 0.02 N*m holds 25 N. A hand must
    # exceed it to turn the cap at all.
    "yaw_frictionloss": 0.020,
    "yaw_damping":      0.008,   # sets the spin rate under a given twist
    "lift_damping":     0.002,
    "cap_skirt_segments": 12,    # hollow cap skirt = N boxes in a ring
    "bottle_fixed":     True,    # True: the other hand holds the bottle
    # THREAD STIFFNESS — the three numbers that decide whether this is a
    # thread or a loose slot, and the least obvious part of the whole file.
    #
    # MuJoCo's constraint stiffness is ACCELERATION-based: what a constraint
    # can hold scales as (1/timeconst^2) x (mass of what it holds), and a
    # bottle cap is 3 g. At MuJoCo's DEFAULT softness the coupling behaves
    # like a 0.5 N/m spring — the cap sags 65 mm through its own thread under
    # nothing but gravity, i.e. the asset silently degrades into "push it
    # down and it goes on", the exact failure it exists to prevent. Measured
    # travel of the cap under a straight pull, no turning:
    #
    #     dt / timeconst        2 N       10 N      20 N
    #     2.00 / 2.0 ms      12.4 mm    29.5 mm   35.0 mm   <- unusable
    #     1.00 / 2.0 ms       3.1 mm     8.6 mm   15.5 mm
    #     0.50 / 1.0 ms       0.8 mm     2.2 mm    3.9 mm
    #     0.25 / 0.5 ms       0.2 mm     0.5 mm    1.0 mm   <- default
    #
    # timeconst must stay >= 2*timestep or the integrator goes unstable, so
    # the two numbers move together. The default costs 0.03 s of CPU per
    # simulated second for this asset — nothing. SOFTEN them on purpose if
    # you want a thread that can be stripped.
    "timestep":         0.00025,
    "thread_solref":    [0.0005, 1.0],
    "thread_solimp":    [0.99, 0.9999, 1e-5],
    "carrier_mass":     0.02,    # apparent inertia of the thread carrier
    # APPARENT ROTATIONAL INERTIA OF THE THREAD, and the number that makes it
    # self-locking in practice. The friction constraint on the yaw joint can
    # only resist in proportion to the inertia it acts on, so with the cap's
    # own 9e-7 kg*m^2 a sustained 20 N pull CREEPS the cap open one thread at
    # a time no matter how large frictionloss is. Measured creep under a
    # ramped 20 N pull, against how fast a 50 mN*m twist still turns it:
    #     9e-7  ->  2.03 mm creep   1.79 turns / 2 s
    #     1e-4  ->  0.14 mm creep   1.18 turns / 2 s   <- default
    #     1e-3  ->  0.02 mm creep   1.11 turns / 2 s
    "carrier_inertia":  1e-4,
    "straw_length":     0.0,
    "straw_radius":     0.0035,
    "straw_rgba":       [0.85, 0.92, 0.95, 0.35],
    "include_floor":    True,
}


# ---------------------------------------------------------------------------
def _derive(p: dict) -> dict:
    """Everything the mechanism needs that follows from the numbers."""
    mouth_z = p["body_height"] + p["shoulder_height"] + p["neck_height"]
    top_wall = float(p["cap_top_wall"] if p["cap_top_wall"] is not None
                     else p["cap_wall"])
    inner_depth = p["cap_height"] - top_wall
    seat_z = mouth_z - inner_depth + 0.0005          # cap seated on the mouth
    lift_range = p["thread_pitch"] * p["thread_turns"]
    yaw_range_deg = 360.0 * p["thread_turns"]
    sign = 1.0 if p["thread_right_handed"] else -1.0
    return {
        "mouth_z": mouth_z,
        "top_wall": top_wall,
        "inner_depth": inner_depth,
        "seat_z": seat_z,
        "lift_range": lift_range,
        "yaw_range_deg": yaw_range_deg,
        # lift = k * yaw   (yaw in RADIANS, the unit MuJoCo constrains in)
        "k_lift_per_rad": sign * p["thread_pitch"] / (2 * np.pi),
        "clears_neck": lift_range >= inner_depth,
    }


def build(params: dict | None = None, out_dir: str | Path = ".") -> dict:
    """Write <name>.xml + meshes + <name>_params.json. Returns the handles."""
    p = merge_params(DEFAULTS, params)
    d = _derive(p)
    name = str(p["name"])
    out_dir = Path(out_dir)
    (out_dir / "meshes").mkdir(parents=True, exist_ok=True)

    # ---------------- meshes (visual) ----------------
    R, H = p["body_radius"], p["body_height"]
    nr, sh, nh = p["neck_radius"], p["shoulder_height"], p["neck_height"]
    # `neck_radius` is the thread CREST — the widest part of the neck, which
    # is what a caliper (and a mask) measures. The smooth core the cap slides
    # over is two ridge-heights narrower, exactly as on a real bottle. Getting
    # this backwards puts the helix OUTSIDE the cap wall and the render shows
    # a thread drawn on top of the closed cap.
    ridge = min(p["thread_ridge_radius"],
                max(1e-4, (p["cap_radius"] - p["cap_wall"] - nr) * 0.9))
    core_r = nr - 2 * ridge
    profile = [(0.0, 0.0), (R * 0.92, 0.0), (R, 0.012),
               (R, H), (core_r, H + sh), (core_r, H + sh + nh),
               (0.0, d["mouth_z"])]
    write_mesh(revolve_profile(profile, sections=48),
               out_dir / "meshes" / f"{name}_body.obj")

    cr, ch, cw = p["cap_radius"], p["cap_height"], p["cap_wall"]
    tw = d["top_wall"]
    cap_profile = [(0.0, ch), (cr, ch), (cr, 0.0), (cr - cw, 0.0),
                   (cr - cw, ch - tw), (0.0, ch - tw)]
    write_mesh(revolve_profile(cap_profile, sections=40),
               out_dir / "meshes" / f"{name}_cap.obj")

    # the helix rides on the neck, spanning the travel the thread provides
    thread_h = min(d["lift_range"], nh * 0.9)
    turns_shown = max(1.0, thread_h / max(p["thread_pitch"], 1e-4))
    write_mesh(helix_tube(coil_radius=nr - ridge,
                          pitch=p["thread_pitch"], turns=turns_shown,
                          tube_radius=ridge,
                          z0=d["mouth_z"] - thread_h - 0.001,
                          right_handed=bool(p["thread_right_handed"])),
               out_dir / "meshes" / f"{name}_thread.obj")

    # ---------------- collision primitives ----------------
    # The bottle's VISUAL is the lathe mesh; its COLLISION is primitives,
    # because a bottle is not convex (the shoulder is a dent around the neck)
    # and MuJoCo would collide its convex hull — a cone that the cap could
    # never come down over.
    mid_z = H + sh * 0.5
    coll = f"""
      <geom name="{name}_c_barrel" type="cylinder" size="{R:.5f} {H/2:.5f}"
            pos="0 0 {H/2:.5f}" rgba="0 0 0 0" friction="{vec(p['friction'])}"
            solref="{vec(p['contact_solref'])}"/>
      <geom name="{name}_c_shoulder" type="cylinder"
            size="{(R+nr)/2:.5f} {sh/2:.5f}" pos="0 0 {mid_z:.5f}"
            rgba="0 0 0 0" friction="{vec(p['friction'])}"
            solref="{vec(p['contact_solref'])}"/>
      <geom name="{name}_c_neck" type="cylinder" size="{core_r:.5f} {nh/2:.5f}"
            pos="0 0 {H+sh+nh/2:.5f}" rgba="0 0 0 0"
            friction="{vec(p['friction'])}"
            solref="{vec(p['contact_solref'])}"/>"""

    # The cap must be HOLLOW or it cannot sit around the neck at all: a ring
    # of boxes is a shell, a mesh's convex hull is a plug.
    nseg = int(p["cap_skirt_segments"])
    skirt_h = ch - tw
    chord = np.pi * cr / nseg
    skirt = []
    for i in range(nseg):
        a = 2 * np.pi * i / nseg
        rr = cr - cw / 2
        skirt.append(
            f'      <geom name="{name}_cap_s{i}" type="box" '
            f'size="{cw/2:.5f} {chord:.5f} {skirt_h/2:.5f}" '
            f'pos="{rr*np.cos(a):.5f} {rr*np.sin(a):.5f} {skirt_h/2:.5f}" '
            f'euler="0 0 {np.degrees(a):.2f}" rgba="0 0 0 0" '
            f'friction="{vec(p["friction"])}" '
            f'solref="{vec(p["contact_solref"])}"/>')
    skirt.append(f'      <geom name="{name}_cap_top" type="cylinder" '
                 f'size="{cr:.5f} {tw/2:.5f}" pos="0 0 {ch-tw/2:.5f}" '
                 f'rgba="0 0 0 0" friction="{vec(p["friction"])}" '
                 f'solref="{vec(p["contact_solref"])}"/>')
    # visible stripes (visual-only) so rotation reads on camera
    for i in range(int(p["cap_stripe_count"])):
        a = 2 * np.pi * i / max(1, int(p["cap_stripe_count"]))
        rr = cr + 0.0004
        skirt.append(
            f'      <geom name="{name}_cap_stripe{i}" type="box" '
            f'size="0.0006 {min(0.0035, 0.25*chord):.5f} {0.40*ch:.5f}" '
            f'pos="{rr*np.cos(a):.5f} {rr*np.sin(a):.5f} {0.52*ch:.5f}" '
            f'euler="0 0 {np.degrees(a):.2f}" '
            f'rgba="{rgba(p["cap_stripe_rgba"])}" contype="0" '
            f'conaffinity="0" group="2"/>')

    # ---- drinking straw: visual tube hanging from the cap ceiling ----
    straw = None
    sl = float(p["straw_length"]); srad = float(p["straw_radius"])
    if sl < 0 or srad <= 0:
        raise ValueError("straw_length must be >= 0 and straw_radius > 0")
    if sl > 0:
        ceil_local = ch - tw                       # cap ceiling, cap frame
        bottom_seated = d["seat_z"] + ceil_local - sl
        if bottom_seated < 0.002:
            raise ValueError(
                f"straw_length {sl:.4f} m reaches below the bottle floor when the cap "
                f"is seated (bottom at {bottom_seated:.4f} m); shorten straw_length")
        if srad >= core_r:
            raise ValueError("straw_radius must be smaller than the neck core radius")
        skirt.append(
            f'      <geom name="{name}_straw" type="cylinder" '
            f'size="{srad:.5f} {sl/2:.5f}" pos="0 0 {ceil_local - sl/2:.5f}" '
            f'rgba="{rgba(p["straw_rgba"])}" contype="0" conaffinity="0" '
            f'group="2" mass="0"/>')
        straw = {"geom": f"{name}_straw", "length_m": sl, "radius_m": srad,
                 "bottom_z_seated_m": round(bottom_seated, 5),
                 "straw_clear_lift_m": round(d["mouth_z"] - bottom_seated, 5)}

    m_cap = p["cap_mass"]
    cap_ixx = m_cap * (cr * cr / 2 + ch * ch / 3)
    cap_izz = m_cap * cr * cr
    m_b = p["body_mass"]
    b_ixx = m_b * (R * R / 4 + H * H / 3)

    free = "" if p["bottle_fixed"] else '      <freejoint name="%s_free"/>\n' % name
    floor = FLOOR_XML if p["include_floor"] else ""
    px, py, pz = p["pos"]

    xml = mjcf_header(f"{name}_screwcap", timestep=p["timestep"],
                      meshdir="meshes") + f"""
  <asset>
    <mesh name="{name}_body_m" file="{name}_body.obj"/>
    <mesh name="{name}_cap_m"  file="{name}_cap.obj"/>
    <mesh name="{name}_thr_m"  file="{name}_thread.obj"/>
  </asset>

  <worldbody>{LIGHTS_XML}{floor}
    <body name="{name}" pos="{vec([px, py, pz])}" euler="0 0 {p['yaw_deg']}">
{free}      <inertial pos="0 0 {0.4*H:.4f}" mass="{m_b}"
                diaginertia="{b_ixx:.6f} {b_ixx:.6f} {m_b*R*R/2:.6f}"/>
      <geom name="{name}_v_body" type="mesh" mesh="{name}_body_m"
            rgba="{rgba(p['body_rgba'])}" contype="0" conaffinity="0"
            group="2"/>
      <geom name="{name}_v_thread" type="mesh" mesh="{name}_thr_m"
            rgba="{rgba(p['thread_rgba'])}" contype="0" conaffinity="0"
            group="2"/>{coll}
      <!-- carrier: the two scalar joints the thread constraint couples -->
      <body name="{name}_carrier" pos="0 0 {d['seat_z']:.5f}">
        <joint name="{name}_yaw" type="hinge" axis="0 0 1"
               range="{-d['yaw_range_deg'] - 20:.2f} {d['yaw_range_deg'] + 20:.2f}"
               damping="{p['yaw_damping']}"
               frictionloss="{p['yaw_frictionloss']}"/>
        <!-- 15% of headroom past the thread's travel. Without it the slide
             hits its limit a hair before the driver decides the thread has
             run out, the limit and the coupling fight each other, and the cap
             spins up to 4000 rad/s and takes the integrator with it. -->
        <joint name="{name}_lift" type="slide" axis="0 0 1"
               range="-0.002 {d['lift_range'] * 1.15:.5f}"
               damping="{p['lift_damping']}"/>
        <inertial pos="0 0 {ch/2:.4f}" mass="{p['carrier_mass']:.6f}"
                  diaginertia="{max(cap_ixx, 0.6*p['carrier_inertia']):.8f} {max(cap_ixx, 0.6*p['carrier_inertia']):.8f} {p['carrier_inertia']:.8f}"/>
      </body>
    </body>

    <!-- the cap is a REAL free body: once the thread lets go it just falls -->
    <body name="{name}_cap" pos="{vec([px, py, pz + d['seat_z']])}"
          euler="0 0 {p['yaw_deg']}">
      <freejoint name="{name}_cap_free"/>
      <inertial pos="0 0 {ch/2:.4f}" mass="{m_cap}"
                diaginertia="{cap_ixx:.8f} {cap_ixx:.8f} {cap_izz:.8f}"/>
      <geom name="{name}_v_cap" type="mesh" mesh="{name}_cap_m"
            rgba="{rgba(p['cap_rgba'])}" contype="0" conaffinity="0" group="2"/>
{chr(10).join(skirt)}
    </body>
  </worldbody>

  <equality>
    <!-- THE THREAD: lift = (pitch/2pi) * yaw. Turning is the only way up. -->
    <joint name="{name}_thread" joint1="{name}_lift" joint2="{name}_yaw"
           polycoef="0 {d['k_lift_per_rad']:.8f} 0 0 0"
           solref="{vec(p['thread_solref'])}" solimp="{vec(p['thread_solimp'])}"/>
    <!-- thread engagement: break it and the cap is a plain rigid body -->
    <weld name="{name}_engaged" body1="{name}_carrier" body2="{name}_cap"
          relpose="0 0 0 1 0 0 0" solref="{vec(p['thread_solref'])}"
          solimp="{vec(p['thread_solimp'])}"/>
  </equality>

  <!-- NO ACTUATORS. A bottle has no motors, and shipping a position servo
       on the cap's own yaw joint is how a "demo" stops being evidence: the
       servo holds or drives the mechanism regardless of whether the thread
       works, and it silently fights anything else that pushes (measured: a
       forgotten kp=6 servo at target 0 held every carton panel flat against
       1.2 N of push, which read as a broken fold). Motion comes from
       external wrenches only — see tool_library/common/drivers.py. -->
</mujoco>
"""
    xml_path = out_dir / f"{name}.xml"
    xml_path.write_text(xml)
    save_params(out_dir / f"{name}_params.json", p)

    return {
        "xml": str(xml_path),
        "params_json": str(out_dir / f"{name}_params.json"),
        "joints": {"yaw": f"{name}_yaw", "lift": f"{name}_lift"},
        "bodies": {"bottle": name, "cap": f"{name}_cap",
                   "carrier": f"{name}_carrier"},
        "equalities": {"thread": f"{name}_thread",
                       "engaged": f"{name}_engaged"},
        "mouth_z": round(d["mouth_z"], 5),
        "lift_per_turn_m": p["thread_pitch"],
        "turns_to_open": p["thread_turns"],
        "total_lift_m": round(d["lift_range"], 5),
        "cap_inner_depth_m": round(d["inner_depth"], 5),
        "cap_clears_neck": bool(d["clears_neck"]),
        "straw": straw,
        "note": ("open/close is ONE number: thread_pitch metres per turn. "
                 "The cap cannot translate without rotating — that is an "
                 "equality constraint, not a contact heuristic. Set "
                 "data.eq_active[<engaged>] = 0 to disengage the thread."),
    }


def rebuild(params_json: str | Path, out_dir: str | Path | None = None) -> dict:
    """Re-run the build from an edited params file — the fitting workflow."""
    from tool_library.common.mjcf import load_params
    p = load_params(params_json)
    return build(p, out_dir or Path(params_json).parent)


# ---------------------------------------------------------------------------
def seat_cap(model, data, built: dict, lift: float) -> None:
    """Put the mechanism at a given thread position and SNAP the free cap onto
    the carrier there. Anything that moves the joints must call this — the cap
    is a separate free body and does not follow qpos on its own."""
    import mujoco
    k = model.equality(built["equalities"]["thread"]).data[1]
    yaw = float(lift) / k if abs(k) > 1e-12 else 0.0
    data.qpos[model.joint(built["joints"]["yaw"]).qposadr[0]] = yaw
    data.qpos[model.joint(built["joints"]["lift"]).qposadr[0]] = float(lift)
    mujoco.mj_forward(model, data)
    cid = model.body(built["bodies"]["carrier"]).id
    adr = model.joint(f'{built["bodies"]["cap"]}_free').qposadr[0]
    data.qpos[adr:adr + 3] = data.xpos[cid]
    data.qpos[adr + 3:adr + 7] = data.xquat[cid]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)


def preview_stills(built: dict, out_dir: str | Path) -> list[str]:
    """Three stills: seated, half a thread up, fully unscrewed."""
    import mujoco
    from tool_library.common.mjcf import key_states
    m = load(built["xml"])
    dat = mujoco.MjData(m)
    lr = float(built["total_lift_m"])
    return key_states(
        m, dat,
        [("closed", lambda mm, dd: seat_cap(mm, dd, built, 0.0)),
         ("half_open", lambda mm, dd: seat_cap(mm, dd, built, lr / 2)),
         ("open", lambda mm, dd: seat_cap(mm, dd, built, lr))],
        out_dir, f"{built['bodies']['bottle']}_preview",
        camera=free_camera([0, 0, built["mouth_z"] * 0.72],
                           built["mouth_z"] * 1.9, 120, -8))


def demo(built: dict, out_mp4: str | Path, fps: int = 30,
         width: int = 640, height: int = 480) -> dict:
    """Push it, twist it closed, pull it, twist it open, watch it fall off —
    all of it from forces and torques on the CAP BODY (see tool_library/common/drivers.py). The
    asset has no actuators, so there is nothing here that could fake it."""
    import mujoco
    from tool_library.common.mjcf import clear_forces
    from tool_library.common.drivers import BottleDriver
    m = load(built["xml"])
    dat = mujoco.MjData(m)
    seat_cap(m, dat, built, float(built["total_lift_m"]))   # start at the top
    drv = BottleDriver(m, dat, built)
    cam = free_camera([0.03, 0, built["mouth_z"] * 0.72],
                      built["mouth_z"] * 2.0, 120, -8)

    def control(t, model, data):
        clear_forces(data)
        return drv.step(t)

    frames = rollout(m, dat, seconds=drv.T["drop"], fps=fps, control=control,
                     camera=cam, width=width, height=height, spin_deg_per_s=3.0)
    out = write_video(frames, out_mp4, fps=fps)
    out["stills"] = save_stills(frames, Path(out_mp4).parent,
                                f"{built['bodies']['bottle']}_still", n=5)
    out["measured"] = {k: (round(float(v), 3) if isinstance(v, float) else v)
                       for k, v in drv.meas.items()}
    out["verdict"] = drv.verdict()
    return out


# ---------------------------------------------------------------------------


def construct_bottle_straw(out_dir, preview=False, **params):
    """Build the asset into out_dir from the parameters above (any key of
    DEFAULTS; unknown keys raise). Returns the build dictionary."""
    built = build(params, out_dir)
    if preview:
        built["stills"] = preview_stills(built, out_dir)
    return built


# seat_cap() puts the mechanism at a thread position and snaps the free cap onto it.

if __name__ == "__main__":
    import json as _json
    import sys as _sys
    out = _sys.argv[1] if len(_sys.argv) > 1 else "."
    print(_json.dumps(construct_bottle_straw(out, preview=True), indent=1, default=str))
