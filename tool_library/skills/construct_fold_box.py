"""Construct folding box: A box whose creases are real hinge joints: a folding carton, an open tray, or a rigid box with a hinged lid (a gift box).

FOLDING CARTON — a box whose creases are real joints.

THE POINT
---------
A folded box is not a deformable object and it is not a rigid one. It is a
tree of rigid panels connected by hinges that all live on measurable crease
lines, which is precisely what a scanned mesh of a box cannot give you: a scan
returns one fused shell with no idea where the paper bends.

So the creases are CONSTRUCTED, in one of three joint topologies (`style`):

    carton   4 wall creases + 4 top flaps   ships flat, folds into a box
    tray     4 wall creases                 open-topped tray or punnet
    lid      1 hinge                        RIGID box with a lid: shoebox,
                                            chest, pizza box — walls do NOT fold

From five numbers — outer length, width, height, panel thickness, and how far
the top flaps reach — the carton style emits a nine-body tree:

    base
      +- wall_xp -- flap_xp        4 walls hinged on the base's four edges
      +- wall_xn -- flap_xn        4 top flaps hinged on the wall tops
      +- wall_yp -- flap_yp
      +- wall_yn -- flap_yn

Joint zero is FLAT (the carton as it ships, laid out on the table); 90 degrees
is assembled. Every fold in between is a real articulated state, so "fold the
box" is a joint trajectory a policy can be scored against, and "unfold it" is
the same trajectory backwards.

Three geometric details that are not optional, because without them the box
jitters itself apart instead of closing:
  * the x-walls are narrowed by one thickness so they tuck BETWEEN the y-walls
    instead of sharing the corner volume with them;
  * the second pair of flaps closes onto the first pair and is stopped by
    contact, not by its joint limit — which is what happens with real card;
  * the creases stop AT vertical: a few degrees past and the x-walls lean in
    over the closed inner flaps and grind 2 mm into them.

No actuators: folding in the demos is a force pushed against each panel's far
edge (common.drivers.FoldBoxDriver), and unfolding is that force stopping.

    python -m tool_library.skills.construct_fold_box [out_dir]      # build + preview stills
"""
from __future__ import annotations

TOOL = {'name': 'construct_fold_box',
 'title': 'Construct folding box',
 'type': 'skill',
 'category': 'Object Reconstruction',
 'summary': 'A box whose creases are real hinge joints: a folding carton, an open tray, or a '
            'rigid box with a hinged lid (a gift box).',
 'description': 'Builds the box as a tree of rigid panels connected by hinges on measurable '
                'crease lines, in one of three topologies chosen by style: carton (four wall '
                'creases and four top flaps, ships flat and folds into a box), tray (four wall '
                'creases) or lid (a rigid box with one hinged lid). Panel sizes, card '
                'thickness, crease stiffness and damping, colours and mass are parameters. '
                'Returns the wall and flap joint names and, per panel, the body and the local '
                'point a finger pushes to fold it. The fold_box_fold_test verifier folds a '
                'built box with forces and checks it closes and stays.',
 'inputs': [{'name': 'out_dir',
             'type': 'path',
             'description': 'Folder that receives the MJCF, its meshes and '
                            '<name>_params.json.'},
            {'name': 'preview',
             'type': 'bool',
             'description': 'Also render preview stills of the mechanism at its extremes. '
                            'Default False.'},
            {'name': 'name', 'type': 'str', 'description': 'Name. Default box'},
            {'name': 'style',
             'type': 'str',
             'description': '"carton" (4 wall creases + 4 top flaps, ships flat), "tray" (4 '
                            'wall creases) or "lid" (rigid box with one hinged lid). Default '
                            'carton'},
            {'name': 'length', 'type': 'float', 'description': 'Along +x. Default 0.24'},
            {'name': 'width', 'type': 'float', 'description': 'Along +y. Default 0.18'},
            {'name': 'height', 'type': 'float', 'description': 'Wall height. Default 0.12'},
            {'name': 'thickness',
             'type': 'float',
             'description': 'Panel thickness (corrugated ~3-4 mm). Default 0.0035'},
            {'name': 'with_top_flaps',
             'type': 'bool',
             'description': 'With top flaps. Default True'},
            {'name': 'flap_x_frac',
             'type': 'float',
             'description': 'Flap on an x-wall, as a fraction of length/2. Default 0.92'},
            {'name': 'flap_y_frac',
             'type': 'float',
             'description': 'Flap on a y-wall, as a fraction of width/2. Default 0.92'},
            {'name': 'lid_flap', 'type': 'bool', 'description': 'Lid flap. Default False'},
            {'name': 'lid_flap_h',
             'type': 'float | None',
             'description': 'None = the wall height, so it covers the face'},
            {'name': 'lid_flap_rgba',
             'type': 'float | None',
             'description': 'None = same as the lid'},
            {'name': 'lid_flap_frictionloss',
             'type': 'float | None',
             'description': 'None = enough to hold its own weight'},
            {'name': 'panel_rgba',
             'type': 'list[4]',
             'description': 'Kraft card. Default [0.76, 0.6, 0.4, 1.0]'},
            {'name': 'flap_rgba',
             'type': 'list[4]',
             'description': 'Slightly darker so folds read. Default [0.7, 0.54, 0.35, 1.0]'},
            {'name': 'pos',
             'type': 'list[3]',
             'description': "Centre of the base panel's underside. Default [0.0, 0.0, 0.0]"},
            {'name': 'yaw_deg',
             'type': 'float',
             'description': 'Yaw deg, degrees. Default 0.0'},
            {'name': 'panel_density',
             'type': 'float',
             'description': 'Kg/m3, corrugated board. Default 250.0'},
            {'name': 'hinge_damping',
             'type': 'float',
             'description': 'Creases are lossy, not springy. Default 0.02'},
            {'name': 'hinge_frictionloss',
             'type': 'float',
             'description': 'A crease HOLDS its angle when released. Default 0.004'},
            {'name': 'hinge_stiffness',
             'type': 'float',
             'description': '>0 = springy card that wants to reopen. Default 0.0'},
            {'name': 'hinge_solreflimit', 'type': 'list[float]',
             'description': 'Hinge limit solref [timeconst, dampratio]. Default [0.005, 1.0].'},
            {'name': 'hinge_armature',
             'type': 'float',
             'description': 'Hinge armature. Default 0.005'},
            {'name': 'lid_armature', 'type': 'float | None', 'description': 'Lid armature'},
            {'name': 'lid_damping',
             'type': 'float | None',
             'description': 'None = follow hinge_damping'},
            {'name': 'lid_frictionloss',
             'type': 'float | None',
             'description': 'None = follow hinge_frictionloss'},
            {'name': 'fold_range_deg',
             'type': 'list[2]',
             'description': 'Fold range deg, degrees. Default [-5.0, 90.0]'},
            {'name': 'timestep',
             'type': 'float',
             'description': 'Timestep, seconds. Default 0.002'},
            {'name': 'fixed_base',
             'type': 'bool',
             'description': 'A carton on a table does not slide about. Default True'},
            {'name': 'include_floor',
             'type': 'bool',
             'description': 'Include floor. Default True'}],
 'outputs': [{'name': 'built',
              'type': 'dict',
              'description': 'xml: the MJCF path; params_json: every parameter as built; the '
                             'names of the bodies, joints and equality constraints the '
                             'mechanism exposes, and the derived measurements (sizes, travel, '
                             'masses); stills when preview is set.'}],
 'requirements': ['mujoco>=3.2', 'numpy', 'trimesh', 'imageio', 'Pillow'],
 'simulators': ['MuJoCo'],
 'tags': ['box', 'carton', 'lid', 'hinge', 'articulated', 'parametric', 'reconstruct', 'object reconstruction'],
 'used_in': ['Giftbox Packing'],
 'status': 'tested',
 'entrypoint': 'construct_fold_box',
 'cli': 'python -m tool_library.skills.construct_fold_box out_dir'}


import numpy as np
from pathlib import Path

from tool_library.common.mjcf import (FLOOR_XML, LIGHTS_XML, free_camera, load, merge_params,
                     mjcf_header, rgba, rollout, save_params, save_stills,
                     vec, write_video)

# ---------------------------------------------------------------------------
# DEFAULTS — metres, kilograms, degrees.
# ---------------------------------------------------------------------------
DEFAULTS = {
    "name": "box",
    "style": "carton",       # carton | tray | lid  (see _STYLES)

    # ---- size (outer dimensions of the assembled carton) ----
    "length":     0.240,     # along +x
    "width":      0.180,     # along +y
    "height":     0.120,     # wall height
    "thickness":  0.0035,    # panel thickness (corrugated ~3-4 mm)

    # ---- top flaps ----
    "with_top_flaps": True,
    "flap_x_frac": 0.92,     # flap on an x-wall, as a fraction of length/2
    "flap_y_frac": 0.92,     # flap on a y-wall, as a fraction of width/2

    # ---- style="lid" only: the SKIRT creased onto the lid's free edge ----
    # A magnetic gift box is not a one-hinge lid. The lid carries a second
    # crease near its free edge, and the strip beyond it folds down over the
    # front wall and latches there. That is a genuinely different object to
    # manipulate: closing it is two subgoals, not one, and the second one
    # cannot be reached until the first has finished.
    "lid_flap": False,
    "lid_flap_h": None,      # None = the wall height, so it covers the face
    "lid_flap_rgba": None,   # None = same as the lid
    "lid_flap_frictionloss": None,   # None = enough to hold its own weight

    # ---- appearance ----
    "panel_rgba": [0.76, 0.60, 0.40, 1.0],   # kraft card
    "flap_rgba":  [0.70, 0.54, 0.35, 1.0],   # slightly darker so folds read

    # ---- placement ----
    "pos":     [0.0, 0.0, 0.0],   # centre of the base panel's underside
    "yaw_deg": 0.0,

    # ---- physics ----
    "panel_density":   250.0,     # kg/m3, corrugated board
    "hinge_damping":   0.02,      # creases are lossy, not springy
    "hinge_frictionloss": 0.004,  # a crease HOLDS its angle when released
    "hinge_stiffness": 0.0,       # >0 = springy card that wants to reopen
    # A crease on a 16 g flap has almost no rotational inertia, and dry
    # friction on a near-inertialess DOF is solved badly however large it is:
    # measured, a skirt carrying 7x its own gravity torque in frictionloss
    # still crept from 0 to 26 deg in 15 s, and raising the friction from
    # 0.035 to 0.20 N.m changed nothing at all. Armature is the remedy, and
    # it is also the right story for a crease, which carries the stiffness of
    # the board either side of it. Measured over 15 s, released mid-travel:
    #     armature   skirt 0 deg ->    lid 60 deg ->
    #      0.0000        26.14            -1.17     (falls shut on its own)
    #      0.0001        16.61            13.59
    #      0.0010         3.84            44.77
    #      0.0050         0.87            55.63     (holds)
    # Without this the hands get no credit for closing anything, because
    # gravity closes it for them.
    "hinge_armature": 0.005,
    # THE END OF A FOLD IS A HARD STOP: the two panels meet face to face.
    # MuJoCo's default joint-limit reference (0.02 s) is soft enough that a
    # 1.2 N push at the far edge of an 11 cm flap carries the crease 1.5 deg
    # past 90, and that tip then dips 2.8 mm INTO the flap beneath it
    # (measured on the default carton). A 5 ms reference holds the fold to
    # 0.3 deg of overshoot and the flap-on-flap penetration to 0.6 mm.
    "hinge_solreflimit": [0.005, 1.0],
    # THE LID AND THE SKIRT WANT DIFFERENT ANSWERS. The skirt has to hold the
    # angle it is left at or gravity folds the box and the hands get the
    # credit. The lid, in the demo, is meant to FALL backward on camera when
    # the clip opens, and then be dragged forward by the hands — so it wants
    # to be free enough for gravity to carry it to its own stop. One number
    # cannot be both; None = follow hinge_armature.
    "lid_armature": None,
    "lid_damping": None,          # None = follow hinge_damping
    "lid_frictionloss": None,     # None = follow hinge_frictionloss
    # The crease STOPS at vertical. Letting the panels swing a few degrees past
    # is not harmless: at 93 deg the x-walls lean 7 mm inward over the closed
    # inner flaps and grind 2 mm into them. Real walls stop each other at 90.
    "fold_range_deg":  [-5.0, 90.0],
    "timestep":        0.002,
    "fixed_base":      True,      # a carton on a table does not slide about
    "include_floor":   True,
}

# THREE BOX STYLES, three different joint topologies. Which one a clip needs
# is a question about the object, not about the numbers:
#   carton  8 revolute creases  — ships flat, folds into a box, folds back
#   tray    4 revolute creases  — an open tray/punnet, no top
#   lid     1 revolute hinge    — a rigid box with a lid: shoebox, chest,
#                                 pizza box. The walls do NOT fold.
#   flap    1 revolute crease   — the simplest fold there is: one panel
#                                   hinged to a base, folded over onto it.
#                                   The right place to start a hand demo.
_STYLES = {"carton", "tray", "lid", "flap"}

# panel key -> (hinge offset from base centre, hinge axis, panel out-direction)
_WALLS = {
    "xp": (np.array([+1, 0, 0]), (0, -1, 0)),
    "xn": (np.array([-1, 0, 0]), (0, +1, 0)),
    "yp": (np.array([0, +1, 0]), (+1, 0, 0)),
    "yn": (np.array([0, -1, 0]), (-1, 0, 0)),
}


def build(params: dict | None = None, out_dir: str | Path = ".") -> dict:
    p = merge_params(DEFAULTS, params)
    name = str(p["name"])
    L, W, H, t = (p["length"], p["width"], p["height"], p["thickness"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lo, hi = p["fold_range_deg"]
    jattr = (f'range="{lo} {hi}" damping="{p["hinge_damping"]}" '
             f'frictionloss="{p["hinge_frictionloss"]}" '
             f'armature="{p["hinge_armature"]}" '
             f'stiffness="{p["hinge_stiffness"]}" '
             f'solreflimit="{vec(p["hinge_solreflimit"])}"')
    gattr = f'density="{p["panel_density"]}" friction="0.9 0.02 0.001"'
    # x-walls are narrowed by a thickness so the four corners do not overlap
    half_y_x = W / 2 - t
    fx = p["flap_x_frac"] * L / 2
    fy = p["flap_y_frac"] * W / 2

    style = str(p["style"])
    if style not in _STYLES:
        raise ValueError(f"style must be one of {sorted(_STYLES)}, not {style!r}")
    if style == "lid":
        return _build_lid(p, out_dir, jattr, gattr)
    if style == "flap":
        return _build_flap(p, out_dir, gattr)
    if style == "tray":
        p = dict(p, with_top_flaps=False)

    bodies, joints, flap_joints = [], [], []
    push_local, push_bodies = {}, {}
    for key, (dirv, axis) in _WALLS.items():
        along_x = abs(dirv[0]) > 0
        hinge = np.array([dirv[0] * L / 2, dirv[1] * W / 2, t / 2])
        s = float(dirv[0] + dirv[1])                    # +1 or -1
        if along_x:
            g_size, g_pos = (H / 2, half_y_x, t / 2), (s * H / 2, 0, 0)
            f_size, f_pos = (fx / 2, half_y_x, t / 2), (s * fx / 2, 0, 0)
            # OUTER flaps crease one thickness further up the wall, so when
            # they close they LIE ON the inner pair instead of trying to
            # occupy the same plane. Without this the two pairs fight for the
            # same 3.5 mm and the closed box buzzes with 3 mm of panel
            # interpenetration — the fold looks right and the physics is junk.
            f_hinge = (s * (H + t), 0, 0)
        else:
            g_size, g_pos = (L / 2, H / 2, t / 2), (0, s * H / 2, 0)
            # short of the x-walls by a thickness, or the closing flap grinds
            # into the wall it is supposed to sit between
            f_size, f_pos = (L / 2 - 2 * t, fy / 2, t / 2), (0, s * fy / 2, 0)
            f_hinge = (0, s * H, 0)

        jn = f"{name}_fold_{key}"
        joints.append(jn)
        push_local[jn] = [2 * v for v in g_pos]        # the panel's far edge
        push_bodies[jn] = f"{name}_wall_{key}"
        flap = ""
        if p["with_top_flaps"]:
            fj = f"{name}_flap_{key}"
            flap_joints.append(fj)
            push_local[fj] = [2 * v for v in f_pos]
            push_bodies[fj] = f"{name}_flapb_{key}"
            flap = f"""
        <body name="{name}_flapb_{key}" pos="{vec(f_hinge)}">
          <joint name="{fj}" type="hinge" axis="{vec(axis)}" {jattr}/>
          <geom name="{name}_flap_{key}" type="box" size="{vec(f_size)}"
                pos="{vec(f_pos)}" rgba="{rgba(p['flap_rgba'])}" {gattr}/>
        </body>"""
        bodies.append(f"""
      <body name="{name}_wall_{key}" pos="{vec(hinge)}">
        <joint name="{jn}" type="hinge" axis="{vec(axis)}" {jattr}/>
        <geom name="{name}_wall_{key}" type="box" size="{vec(g_size)}"
              pos="{vec(g_pos)}" rgba="{rgba(p['panel_rgba'])}" {gattr}/>{flap}
      </body>""")

    # Sibling walls share their corner volume the instant both stand up. That
    # contact is spurious (real card just meets edge to edge) and it fights
    # the folding force, so it is excluded rather than tuned.
    # A crease is ONE piece of card, so the panels either side of it must not
    # collide. MuJoCo filters parent/child pairs automatically — but not when
    # the parent is welded to the world, which a fixed carton base is, so the
    # base and its own walls collided at exactly half a thickness (the corner
    # the crease sweeps through). Both exclusions are geometry, not tuning.
    wk = list(_WALLS)
    excl = "\n".join(
        [f'    <exclude body1="{name}" body2="{name}_wall_{a}"/>' for a in wk]
        + [f'    <exclude body1="{name}_wall_{a}" body2="{name}_wall_{b}"/>'
           for i, a in enumerate(wk) for b in wk[i + 1:]])

    free = "" if p["fixed_base"] else f'      <freejoint name="{name}_free"/>\n'
    floor = FLOOR_XML if p["include_floor"] else ""
    xml = mjcf_header(f"{name}_foldbox", timestep=p["timestep"]) + f"""
  <worldbody>{LIGHTS_XML}{floor}
    <body name="{name}" pos="{vec(p['pos'])}" euler="0 0 {p['yaw_deg']}">
{free}      <geom name="{name}_base" type="box"
            size="{vec((L/2, W/2, t/2))}" pos="0 0 {t/2:.5f}"
            rgba="{rgba(p['panel_rgba'])}" {gattr}/>{''.join(bodies)}
    </body>
  </worldbody>

  <contact>
{excl}
  </contact>

  <!-- NO ACTUATORS: creases are not motors; every fold in
       the demos comes from a force pushed against a panel edge. -->
</mujoco>
"""
    xml_path = out_dir / f"{name}.xml"
    xml_path.write_text(xml)
    save_params(out_dir / f"{name}_params.json", p)

    return {
        "xml": str(xml_path),
        "params_json": str(out_dir / f"{name}_params.json"),
        "style": style,
        "wall_joints": joints,
        "flap_joints": flap_joints,
        # WHERE A FINGER WOULD PUSH, in each panel's own frame: its far edge,
        # and the panel normal to push along. The moment arm is the whole
        # mechanism — a force at the crease folds nothing.
        "push_local": push_local,
        "push_bodies": push_bodies,
        "base_body": name,
        "flat_deg": 0.0,
        "assembled_deg": 90.0,
        "inner_volume_l": round(1000 * (L - 2*t) * (W - 2*t) * (H - t), 3),
        "note": ("joint 0 deg = flat blank, 90 deg = assembled. Fold order "
                 "that does not jam: walls, then the y-flaps, then the "
                 "x-flaps on top of them. Unfold is the same in reverse."),
    }


def _build_flap(p: dict, out_dir: Path, gattr: str) -> dict:
    """style="flap": a base panel and ONE panel creased to it.

    The whole object is two rectangles and one hinge, which is the point: it
    is the smallest thing that is genuinely a fold, so a hand demo can be
    judged on whether the crease closed rather than on whether four panels
    were sequenced correctly.
    """
    name = str(p["name"])
    L, W, t = p["length"], p["width"], p["thickness"]
    jn = f"{name}_fold"
    lo, hi = -5.0, 170.0            # folds all the way over onto the base
    xml = mjcf_header(f"{name}_flap", timestep=p["timestep"]) + f"""
  <worldbody>{LIGHTS_XML}{FLOOR_XML if p["include_floor"] else ""}
    <body name="{name}" pos="{vec(p['pos'])}" euler="0 0 {p['yaw_deg']}">
{'' if p["fixed_base"] else f'      <freejoint name="{name}_free"/>'}
      <geom name="{name}_base" type="box" size="{vec((L/2, W/2, t/2))}"
            pos="0 0 {t/2:.5f}" rgba="{rgba(p['panel_rgba'])}" {gattr}/>
      <body name="{name}_flapb" pos="{L/2:.5f} 0 {t/2:.5f}">
        <joint name="{jn}" type="hinge" axis="0 -1 0" range="{lo} {hi}"
               damping="{p['hinge_damping']}"
               frictionloss="{p['hinge_frictionloss']}"
               armature="{p['hinge_armature']}"
               stiffness="{p['hinge_stiffness']}"/>
        <geom name="{name}_flap" type="box" size="{vec((L/2, W/2, t/2))}"
              pos="{L/2:.5f} 0 0" rgba="{rgba(p['flap_rgba'])}" {gattr}/>
      </body>
    </body>
  </worldbody>

  <contact>
    <!-- one piece of card: the two panels either side of a crease never
         collide with each other, they just meet -->
    <exclude body1="{name}" body2="{name}_flapb"/>
  </contact>
</mujoco>
"""
    xml_path = out_dir / f"{name}.xml"
    xml_path.write_text(xml)
    save_params(out_dir / f"{name}_params.json", p)
    return {
        "xml": str(xml_path),
        "params_json": str(out_dir / f"{name}_params.json"),
        "style": "flap",
        "wall_joints": [], "flap_joints": [jn],
        "push_local": {jn: [L, 0.0, 0.0]},        # the panel's free edge
        "push_bodies": {jn: f"{name}_flapb"},
        "base_body": name,
        "flat_deg": 0.0, "assembled_deg": 180.0, "open_deg": hi,
        "panel_len_m": round(float(L), 4), "panel_wid_m": round(float(W), 4),
        "inner_volume_l": 0.0,
        "note": ("one crease: 0 deg = flat, 180 deg = folded over onto the "
                 "base. The free edge is at local (+length, 0, 0) of "
                 f"{name}_flapb."),
    }


def _build_lid(p: dict, out_dir: Path, jattr: str, gattr: str) -> dict:
    """style="lid": a RIGID box with one hinged lid, optionally with a SKIRT.

    The walls are geoms of the base body, not panels on creases — a shoebox
    does not fold flat, and pretending it does would give a policy four joints
    that do not exist. One revolute joint, one thing to learn.

    `lid_flap=True` adds the second one: the strip creased onto the lid's own
    free edge that comes down over the front wall, which is what a magnetic
    gift box actually has. It is a separate joint on a separate body, so
    closing the box becomes two subgoals in a fixed order — the skirt cannot
    be pressed against the wall until the lid it hangs from is down.
    """
    name = str(p["name"])
    L, W, H, t = (p["length"], p["width"], p["height"], p["thickness"])
    lo, hi = p["fold_range_deg"]
    lid_hi = max(float(hi), 100.0)
    walls = []
    for key, (dirv, _axis) in _WALLS.items():
        along_x = abs(dirv[0]) > 0
        s = float(dirv[0] + dirv[1])
        size = ((t / 2, W / 2 - t, H / 2) if along_x else (L / 2, t / 2, H / 2))
        pos = ((s * (L / 2 - t / 2), 0, t / 2 + H / 2) if along_x
               else (0, s * (W / 2 - t / 2), t / 2 + H / 2))
        walls.append(f'      <geom name="{name}_wall_{key}" type="box" '
                     f'size="{vec(size)}" pos="{vec(pos)}" '
                     f'rgba="{rgba(p["panel_rgba"])}" {gattr}/>')
    lid_j = f"{name}_lid"
    lid_arm = (p["hinge_armature"] if p.get("lid_armature") is None
               else p["lid_armature"])
    lid_damp = (p["hinge_damping"] if p.get("lid_damping") is None
                else p["lid_damping"])
    lid_fric = (p["hinge_frictionloss"] if p.get("lid_frictionloss") is None
                else p["lid_frictionloss"])
    # ---- the skirt creased onto the lid's free edge -----------------------
    # Angles are stated AGAINST THE LID: 0 = the skirt continues the lid in
    # the same plane, 90 = folded square to it. With the lid shut that is
    # straight down the front face; with the lid open flat behind the box it
    # is straight up, which is exactly how the real one sits when it is open.
    # The crease is offset by half a thickness so the folded skirt lies
    # OUTSIDE the front wall instead of trying to occupy it — the same
    # reasoning as the carton's outer flaps, and for the same 2 mm of
    # interpenetration it avoids.
    skirt_j, skirt_body, skirt_h = None, None, 0.0
    skirt = ""
    if p.get("lid_flap"):
        skirt_h = float(p["lid_flap_h"] or H)
        skirt_j = f"{name}_lidflap"
        skirt_body = f"{name}_lidflapb"
        s_rgba = p.get("lid_flap_rgba") or p["flap_rgba"]
        # a skirt that flops open under its own weight cannot BE a subgoal:
        # it has to hold whatever angle it was last left at, which is what
        # the crease in stiff card does. Gravity torque about the crease is
        # m*g*h/2; the friction has to beat it with margin.
        s_mass = L * skirt_h * t * float(p["panel_density"])
        s_fric = float(p.get("lid_flap_frictionloss")
                       or max(float(p["hinge_frictionloss"]),
                              3.0 * s_mass * 9.81 * skirt_h / 2))
        skirt = f"""
        <body name="{skirt_body}" pos="0 {-W:.5f} 0">
          <joint name="{skirt_j}" type="hinge" axis="1 0 0" range="-5 90"
                 damping="{p['hinge_damping']}" frictionloss="{s_fric:.5f}"
                 armature="{p['hinge_armature']}"
                 stiffness="{p['hinge_stiffness']}"/>
          <geom name="{name}_lidflap" type="box"
                size="{vec((L/2, skirt_h/2, t/2))}"
                pos="0 {-skirt_h/2:.5f} {t/2:.5f}"
                rgba="{rgba(s_rgba)}" {gattr}/>
        </body>"""
    xml = mjcf_header(f"{name}_lidbox", timestep=p["timestep"]) + f"""
  <worldbody>{LIGHTS_XML}{FLOOR_XML if p["include_floor"] else ""}
    <body name="{name}" pos="{vec(p['pos'])}" euler="0 0 {p['yaw_deg']}">
{'' if p["fixed_base"] else f'      <freejoint name="{name}_free"/>'}
      <geom name="{name}_base" type="box" size="{vec((L/2, W/2, t/2))}"
            pos="0 0 {t/2:.5f}" rgba="{rgba(p['panel_rgba'])}" {gattr}/>
{chr(10).join(walls)}
      <body name="{name}_lidb" pos="0 {W/2:.5f} {t/2 + H:.5f}">
        <joint name="{lid_j}" type="hinge" axis="-1 0 0"
               range="{lo} {lid_hi}" damping="{lid_damp}"
               frictionloss="{lid_fric}"
               armature="{lid_arm}"
               stiffness="{p['hinge_stiffness']}"/>
        <geom name="{name}_lid" type="box"
              size="{vec((L/2, W/2, t/2))}" pos="0 {-W/2:.5f} 0"
              rgba="{rgba(p['flap_rgba'])}" {gattr}/>{skirt}
      </body>
    </body>
  </worldbody>

  <contact>
    <!-- The lid and the box are ONE object, creased together: shut, the lid
         shares exactly half a board thickness with every wall it rests on
         (measured 2.00 mm on 4 mm board) — the same corner any crease sweeps
         through, and the same reason the carton excludes base-vs-wall. The
         joint limit at 0 deg is what stops the lid, not the walls. -->
    <exclude body1="{name}" body2="{name}_lidb"/>
{f'    <exclude body1="{name}" body2="{skirt_body}"/>' if skirt_body else ''}
  </contact>

  <!-- NO ACTUATORS: a lid has no motor either. -->
</mujoco>
"""
    xml_path = out_dir / f"{name}.xml"
    xml_path.write_text(xml)
    save_params(out_dir / f"{name}_params.json", p)
    joints = [lid_j] + ([skirt_j] if skirt_j else [])
    push_local = {lid_j: [0.0, -W, 0.0]}          # the lid's free edge
    push_bodies = {lid_j: f"{name}_lidb"}
    if skirt_j:
        push_local[skirt_j] = [0.0, -skirt_h, t / 2]   # the skirt's free edge
        push_bodies[skirt_j] = skirt_body
    return {
        "xml": str(xml_path),
        "params_json": str(out_dir / f"{name}_params.json"),
        "style": "lid",
        "wall_joints": [],                 # the walls are rigid, by design
        "flap_joints": joints,
        # NAMED, not positional. Two joints that both close the box are not
        # interchangeable and a caller that indexes [0]/[1] is one parameter
        # change away from pressing the wrong one.
        "lid_joint": lid_j, "lid_body": f"{name}_lidb",
        "lid_geom": f"{name}_lid",
        "skirt_joint": skirt_j, "skirt_body": skirt_body,
        "skirt_geom": (f"{name}_lidflap" if skirt_j else None),
        "skirt_h_m": round(float(skirt_h), 4) if skirt_j else None,
        "lid_len_m": round(float(L), 4), "lid_dep_m": round(float(W), 4),
        "thickness_m": round(float(t), 4),
        "push_local": push_local,
        "push_bodies": push_bodies,
        "base_body": name,
        "flat_deg": 0.0, "assembled_deg": 0.0, "open_deg": lid_hi,
        "inner_volume_l": round(1000 * (L - 2*t) * (W - 2*t) * (H - t), 3),
        "note": ("lid joint: 0 deg = shut, "
                 f"{lid_hi:.0f} deg = wide open. The walls are rigid."
                 + (" Skirt joint is stated against the LID: 0 = in plane "
                    "with it, 90 = folded square, which with the lid shut is "
                    "flat down the front wall." if skirt_j else "")),
    }


def rebuild(params_json: str | Path, out_dir: str | Path | None = None) -> dict:
    from tool_library.common.mjcf import load_params
    p = load_params(params_json)
    return build(p, out_dir or Path(params_json).parent)


# ---------------------------------------------------------------------------
def preview_stills(built: dict, out_dir: str | Path) -> list[str]:
    """Three stills: flat blank, walls up, closed carton."""
    import mujoco
    from tool_library.common.mjcf import key_states, load_params
    m = load(built["xml"])
    dat = mujoco.MjData(m)
    p = load_params(built["params_json"])
    span = max(p["length"], p["width"]) * 2.6

    def _set(walls_deg, flaps_deg):
        def f(mm, dd):
            for j in built["wall_joints"]:
                dd.qpos[mm.joint(j).qposadr[0]] = np.radians(walls_deg)
            for j in built["flap_joints"]:
                dd.qpos[mm.joint(j).qposadr[0]] = np.radians(flaps_deg)
        return f

    states = ([("shut", _set(0, 0)), ("half_open", _set(0, 55)),
               ("open", _set(0, float(built.get("open_deg", 100))))]
              if built.get("style") == "lid" else
              [("flat", _set(0, 0)), ("walls_up", _set(90, 0)),
               ("closed", _set(90, 90))])
    return key_states(
        m, dat, states, out_dir, f"{built['base_body']}_preview",
        camera=free_camera([0, 0, p["height"] * 0.3], span, 135, -24))


def demo(built: dict, out_mp4: str | Path, fps: int = 30,
         width: int = 640, height: int = 480) -> dict:
    """Push each panel's far edge until the carton stands and closes, then
    take the hands away and let gravity decide what falls back open."""
    import mujoco
    from tool_library.common.mjcf import clear_forces, load_params
    from tool_library.common.drivers import FoldBoxDriver
    m = load(built["xml"])
    dat = mujoco.MjData(m)
    p = load_params(built["params_json"])
    drv = FoldBoxDriver(m, dat, built)
    span = max(p["length"], p["width"]) * 3.0
    cam = free_camera([0, 0, p["height"] * 0.35], span, 135, -22)

    def control(t, model, data):
        clear_forces(data)
        return drv.step(t)

    frames = rollout(m, dat, seconds=drv.T["open"] + 1.0, fps=fps,
                     control=control, camera=cam, width=width, height=height,
                     spin_deg_per_s=4.0)
    out = write_video(frames, out_mp4, fps=fps)
    out["stills"] = save_stills(frames, Path(out_mp4).parent,
                                f"{built['base_body']}_still", n=5)
    out["measured"] = {k: (round(float(v), 3) if isinstance(v, float) else v)
                       for k, v in drv.meas.items()}
    out["verdict"] = drv.verdict()
    return out


# ---------------------------------------------------------------------------


def construct_fold_box(out_dir, preview=False, **params):
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
    print(_json.dumps(construct_fold_box(out, preview=True), indent=1, default=str))
