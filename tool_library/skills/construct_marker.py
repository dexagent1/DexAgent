"""Construct whiteboard marker: A whiteboard marker: barrel, posted cap and tapered nib, with a declared tip site that writes on a whiteboard.

HAND TOOLS THAT ACT ON A SURFACE — a whiteboard marker and an eraser.

Both are rigid bodies, so neither needs a mechanism. What they DO need, and
what a scanned mesh does not carry, is a declared WORKING POINT: the marker's
nib and the eraser's felt face are the parts that write and wipe, and every
downstream question — where the ink goes, how much board a wipe covers, which
end the hand must not cover — is asked about those and not about the shell.

    style="marker"   barrel + posted cap + tapered nib, with a `<name>_tip`
                     site at the nib's point and a `tip_local` offset in the
                     body's own frame
    style="eraser"   a block with a felt pad face, with `<name>_pad` at the
                     centre of that face and the pad's half-extents declared

Which end is which is a property of the object, not of the plan, so both are
returned by `build()` and read off the compiled model at run time
(`tip_world`, `pad_world`). A task that guesses instead — "the tip is at
+x" — breaks the moment the object is yawed or picked up.

The barrel is a CYLINDER on purpose: the fingertip grasp skills read a target
as a segment plus a radius, and a cylinder gives them that exactly, so a pen
grasp is solved against the pen's own axis rather than against a box's diagonal.

    python -m tool_library.skills.construct_marker [out_dir]      # build it + preview stills
"""
from __future__ import annotations

TOOL = {'name': 'construct_marker',
 'title': 'Construct whiteboard marker',
 'type': 'skill',
 'category': 'Object Reconstruction',
 'summary': 'A whiteboard marker: barrel, posted cap and tapered nib, with a declared tip site '
            'that writes on a whiteboard.',
 'description': 'Builds the marker as a free rigid body from its barrel radius and length, cap '
                'and nib dimensions, colours and mass, and declares its working point: a '
                '<name>_tip site at the nib with a tip_local offset in the body frame, which '
                'is where ink goes when the nib meets a writable surface and which end a hand '
                'must leave uncovered. Helpers read the tip position and axis off a compiled '
                'model.',
 'inputs': [{'name': 'out_dir',
             'type': 'path',
             'description': 'Folder that receives the MJCF, its meshes and '
                            '<name>_params.json.'},
            {'name': 'preview',
             'type': 'bool',
             'description': 'Also render preview stills of the mechanism at its extremes. '
                            'Default False.'},
            {'name': 'name', 'type': 'str', 'description': 'Name. Default marker'},
            {'name': 'length',
             'type': 'float',
             'description': 'Overall, cap to nib point. Default 0.14'},
            {'name': 'radius', 'type': 'float', 'description': 'Barrel. Default 0.0085'},
            {'name': 'cap_length',
             'type': 'float',
             'description': 'The posted cap on the back end. Default 0.038'},
            {'name': 'cap_radius',
             'type': 'float',
             'description': 'Cap radius, metres. Default 0.0098'},
            {'name': 'nib_length',
             'type': 'float',
             'description': 'The tapered black cone. Default 0.014'},
            {'name': 'nib_radius',
             'type': 'float',
             'description': 'The writing point. Default 0.0026'},
            {'name': 'collar_length',
             'type': 'float',
             'description': 'The black shoulder the nib comes out of. Default 0.008'},
            {'name': 'barrel_rgba',
             'type': 'list[4]',
             'description': 'Barrel rgba, colour as rgba in 0..1. Default [0.93, 0.93, 0.94, '
                            '1.0]'},
            {'name': 'cap_rgba',
             'type': 'list[4]',
             'description': 'Cap rgba, colour as rgba in 0..1. Default [0.07, 0.07, 0.08, '
                            '1.0]'},
            {'name': 'nib_rgba',
             'type': 'list[4]',
             'description': 'Nib rgba, colour as rgba in 0..1. Default [0.05, 0.05, 0.06, '
                            '1.0]'},
            {'name': 'label_rgba',
             'type': 'list[4]',
             'description': 'Label rgba, colour as rgba in 0..1. Default [0.1, 0.11, 0.14, '
                            '1.0]'},
            {'name': 'mass', 'type': 'float', 'description': 'Mass, kilograms. Default 0.013'},
            {'name': 'pos',
             'type': 'list[3]',
             'description': "The object's own footprint centre on the table (local z = 0 is the table surface). "
                            'Default [0.0, 0.0, 0.0]'},
            {'name': 'yaw_deg',
             'type': 'float',
             'description': 'Yaw deg, degrees. Default 0.0'},
            {'name': 'roll_deg',
             'type': 'float',
             'description': 'A pen lying on a table can be rolled. Default 0.0'},
            {'name': 'friction',
             'type': 'list[3]',
             'description': 'Friction, MuJoCo friction triple. Default [0.85, 0.02, 0.006]'},
            {'name': 'condim',
             'type': 'int',
             'description': '3 = frictionless spin AND roll; 6 = both resisted. Default 6'},
            {'name': 'nib_friction',
             'type': 'list[3]',
             'description': 'Nib friction, MuJoCo friction triple. Default [0.16, 0.004, '
                            '0.0001]'},
            {'name': 'fixed_base',
             'type': 'bool',
             'description': 'These get picked up: free by default. Default False'},
            {'name': 'timestep',
             'type': 'float',
             'description': 'Timestep, seconds. Default 0.002'},
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
 'tags': ['marker', 'pen', 'tool', 'parametric', 'reconstruct', 'object reconstruction'],
 'used_in': ['Drawing and Wiping'],
 'status': 'tested',
 'entrypoint': 'construct_marker',
 'cli': 'python -m tool_library.skills.construct_marker out_dir'}


import numpy as np
from pathlib import Path

from tool_library.common.mjcf import (FLOOR_XML, LIGHTS_XML, free_camera, load, merge_params,
                     mjcf_header, rgba, rollout, save_params, save_stills,
                     vec, write_video)

DEFAULTS = {
    "name": "marker",
    "style": "marker",            # marker | eraser

    # ---- style="marker" ----
    "length":      0.140,         # overall, cap to nib point
    "radius":      0.0085,        # barrel
    "cap_length":  0.038,         # the posted cap on the back end
    "cap_radius":  0.0098,
    "nib_length":  0.014,         # the tapered black cone
    "nib_radius":  0.0026,        # the writing point
    "collar_length": 0.008,       # the black shoulder the nib comes out of
    "barrel_rgba": [0.93, 0.93, 0.94, 1.0],
    "cap_rgba":    [0.07, 0.07, 0.08, 1.0],
    "nib_rgba":    [0.05, 0.05, 0.06, 1.0],
    "label_rgba":  [0.10, 0.11, 0.14, 1.0],
    "mass":        0.013,

    # ---- style="eraser" ----
    "block_l":     0.108,         # long axis, local x
    "block_w":     0.048,         # local y
    "block_h":     0.030,         # local z, table to top
    "pad_h":       0.007,         # the felt layer at the BOTTOM
    "block_rgba":  [0.30, 0.30, 0.32, 1.0],
    "pad_rgba":    [0.22, 0.22, 0.24, 1.0],
    "eraser_mass": 0.048,

    # ---- placement + physics (both styles) ----
    "pos":      [0.0, 0.0, 0.0],  # the object's own footprint centre, on the
                                  #   table: local z=0 IS the table surface
    "yaw_deg":  0.0,
    "roll_deg": 0.0,              # a pen lying on a table can be rolled
    # slide / spin / ROLL. The third one is not decoration on a marker: a
    # cylinder on a plane with condim 3 has no rolling resistance at all, so
    # any nudge — the millimetre it is dropped from when it is placed — sets
    # it rolling and nothing ever stops it. Measured before this existed: the
    # pen travelled 126 mm across the table during a 2.8 s opening shot in
    # which nothing touched it, and every grasp afterwards was solved against
    # a pen that had left. A real marker on a tablecloth stops.
    "friction": [0.85, 0.02, 0.006],
    "condim": 6,             # 3 = frictionless spin AND roll; 6 = both resisted
    # THE NIB GETS ITS OWN, MUCH LOWER FRICTION. The barrel needs grip — it
    # is what the fingers hold — and the writing point needs the opposite: a
    # dry-erase tip on melamine glides. At the barrel's 0.85 the nib STUCK to
    # the board and released in jumps, and the trace shows exactly that: the
    # pen on the surface for every sample (gap 0.0-0.6 mm) while its in-plane
    # error swung 1-21 mm, so what landed was a row of dots at the stick
    # points instead of a line.
    "nib_friction": [0.16, 0.004, 0.0001],
    "fixed_base": False,          # these get picked up: free by default
    "timestep": 0.002,
    "include_floor": True,
}

_STYLES = {"marker", "eraser"}


def build(params: dict | None = None, out_dir: str | Path = ".") -> dict:
    p = merge_params(DEFAULTS, params)
    style = str(p["style"])
    if style not in _STYLES:
        raise ValueError(f"style must be one of {sorted(_STYLES)}, not {style!r}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return (_build_marker if style == "marker" else _build_eraser)(p, out_dir)


def rebuild(params_json: str | Path, out_dir: str | Path | None = None) -> dict:
    import json
    p = json.loads(Path(params_json).read_text())
    return build(p, out_dir or Path(params_json).parent)


# ---------------------------------------------------------------------------
def _build_marker(p: dict, out_dir: Path) -> dict:
    name = str(p["name"])
    L, r = float(p["length"]), float(p["radius"])
    cl, cr = float(p["cap_length"]), float(p["cap_radius"])
    nl, nr = float(p["nib_length"]), float(p["nib_radius"])
    kl = float(p["collar_length"])
    if cl + nl + kl >= L:
        raise ValueError(f"cap+collar+nib ({cl+kl+nl:.3f} m) is longer than "
                         f"the marker ({L:.3f} m)")
    # LYING ON THE TABLE, ALONG LOCAL +X. The body origin is on the table
    # under the barrel's centre, so `pos` means the same thing it means for
    # every other builder here: where the object's footprint sits.
    x0, x1 = -L / 2, L / 2                    # cap end .. nib point
    axis_z = cr                               # the barrel rests on the cap's
                                              #   radius, which is the widest
    gattr = (f'friction="{" ".join(str(x) for x in p["friction"])}" '
             f'condim="{int(p["condim"])}" '
             f'solref="0.005 1" solimp="0.99 0.999 0.001"')
    xb0, xb1 = x0 + cl, x1 - nl - kl          # the white barrel
    # the mass lives on the barrel; everything else is a shell
    body = f"""
    <body name="{name}" pos="{vec((p['pos'][0], p['pos'][1],
                                   p['pos'][2] + axis_z))}"
          euler="{p['roll_deg']} 0 {p['yaw_deg']}">
{'' if p["fixed_base"] else f'      <freejoint name="{name}_free"/>'}
      <geom name="{name}_barrel" type="cylinder"
            fromto="{xb0:.5f} 0 0 {xb1:.5f} 0 0" size="{r:.5f}"
            rgba="{rgba(p['barrel_rgba'])}" mass="{float(p['mass']):.5f}"
            {gattr}/>
      <geom name="{name}_label" type="cylinder"
            fromto="{xb0 + 0.018:.5f} 0 0 {xb0 + 0.050:.5f} 0 0"
            size="{r * 1.02:.5f}" rgba="{rgba(p['label_rgba'])}"
            mass="0.00001" contype="0" conaffinity="0" group="2"/>
      <geom name="{name}_cap" type="cylinder"
            fromto="{x0:.5f} 0 0 {xb0:.5f} 0 0" size="{cr:.5f}"
            rgba="{rgba(p['cap_rgba'])}" mass="0.00001" {gattr}/>
      <geom name="{name}_collar" type="cylinder"
            fromto="{xb1:.5f} 0 0 {xb1 + kl:.5f} 0 0" size="{r * 0.72:.5f}"
            rgba="{rgba(p['nib_rgba'])}" mass="0.00001" {gattr}/>
      <geom name="{name}_nib" type="cylinder"
            fromto="{x1 - nl:.5f} 0 0 {x1:.5f} 0 0" size="{nr:.5f}"
            rgba="{rgba(p['nib_rgba'])}" mass="0.00001"
            friction="{" ".join(str(x) for x in p["nib_friction"])}"
            condim="{int(p["condim"])}"
            solref="0.005 1" solimp="0.99 0.999 0.001"/>
      <site name="{name}_tip" pos="{x1:.5f} 0 0" size="0.0015"
            rgba="1 0 0 0.35"/>
      <site name="{name}_back" pos="{x0:.5f} 0 0" size="0.0015"
            rgba="0 0 1 0.35"/>
    </body>"""
    xml = mjcf_header(f"{name}_marker", timestep=p["timestep"]) + f"""
  <worldbody>{LIGHTS_XML}{FLOOR_XML if p["include_floor"] else ""}{body}
  </worldbody>
</mujoco>
"""
    xml_path = out_dir / f"{name}.xml"
    xml_path.write_text(xml)
    save_params(out_dir / f"{name}_params.json", p)
    return {
        "xml": str(xml_path),
        "params_json": str(out_dir / f"{name}_params.json"),
        "style": "marker",
        "base_body": name,
        "barrel_geom": f"{name}_barrel",     # what a grasp is solved against
        "nib_geom": f"{name}_nib",
        "cap_geom": f"{name}_cap",
        "tip_site": f"{name}_tip",
        "back_site": f"{name}_back",
        # the nib point and the barrel axis IN THE BODY'S OWN FRAME, so a
        # task can recover them after the pen has been picked up and turned
        "tip_local": [round(x1, 5), 0.0, 0.0],
        "axis_local": [1.0, 0.0, 0.0],
        "grip_span_local": [round(xb0, 5), round(xb1, 5)],
        "length_m": round(L, 4), "radius_m": round(r, 5),
        "nib_radius_m": round(nr, 5),
        "axis_height_m": round(axis_z, 5),
        "note": ("a rigid marker lying along its own local +x with the nib "
                 "at +length/2. Grasp it on <barrel_geom> (a cylinder, so "
                 "a grasp solver reads its true axis). Ink goes "
                 "wherever <tip_site> touches a common.ink.InkSurface — "
                 "never where the plan says it should."),
    }


# ---------------------------------------------------------------------------
def _build_eraser(p: dict, out_dir: Path) -> dict:
    name = str(p["name"])
    L, W, H = (float(p["block_l"]), float(p["block_w"]), float(p["block_h"]))
    ph = float(p["pad_h"])
    if ph >= H:
        raise ValueError("pad_h must be less than block_h")
    gattr = (f'friction="{" ".join(str(x) for x in p["friction"])}" '
             f'condim="{int(p["condim"])}" '
             f'solref="0.005 1" solimp="0.99 0.999 0.001"')
    body = f"""
    <body name="{name}" pos="{vec(p['pos'])}"
          euler="0 0 {p['yaw_deg']}">
{'' if p["fixed_base"] else f'      <freejoint name="{name}_free"/>'}
      <geom name="{name}_body" type="box"
            size="{vec((L / 2, W / 2, (H - ph) / 2))}"
            pos="0 0 {ph + (H - ph) / 2:.5f}" rgba="{rgba(p['block_rgba'])}"
            mass="{float(p['eraser_mass']):.5f}" {gattr}/>
      <geom name="{name}_pad" type="box"
            size="{vec((L / 2 * 0.985, W / 2 * 0.985, ph / 2))}"
            pos="0 0 {ph / 2:.5f}" rgba="{rgba(p['pad_rgba'])}"
            mass="0.00001" {gattr}/>
      <site name="{name}_pad_site" pos="0 0 0" size="0.002"
            rgba="0 1 0 0.35"/>
    </body>"""
    xml = mjcf_header(f"{name}_eraser", timestep=p["timestep"]) + f"""
  <worldbody>{LIGHTS_XML}{FLOOR_XML if p["include_floor"] else ""}{body}
  </worldbody>
</mujoco>
"""
    xml_path = out_dir / f"{name}.xml"
    xml_path.write_text(xml)
    save_params(out_dir / f"{name}_params.json", p)
    return {
        "xml": str(xml_path),
        "params_json": str(out_dir / f"{name}_params.json"),
        "style": "eraser",
        "base_body": name,
        "block_geom": f"{name}_body",         # what a grasp is solved against
        "pad_geom": f"{name}_pad",
        "pad_site": f"{name}_pad_site",
        "pad_local": [0.0, 0.0, 0.0],         # centre of the wiping face
        "pad_half": [round(L / 2 * 0.985, 5), round(W / 2 * 0.985, 5)],
        "pad_normal_local": [0.0, 0.0, -1.0],
        "block_lwh": [round(L, 4), round(W, 4), round(H, 4)],
        "note": ("a felt block whose WIPING FACE is the underside: <pad_site> "
                 "is its centre and pad_half its extent, both in the body's "
                 "own frame. common.ink.InkSurface.wipe takes exactly those "
                 "and whitens the rectangle the pad really covers."),
    }


# ===========================================================================
def tip_world(model, data, built: dict, prefix: str = "") -> np.ndarray:
    """Where the working point IS, right now — read off the model, never
    assumed from the plan."""
    key = "tip_site" if built["style"] == "marker" else "pad_site"
    name = _site(model, prefix + built[key], built[key])
    return np.asarray(data.site_xpos[model.site(name).id], float).copy()


def pad_frame(model, data, built: dict, prefix: str = ""):
    """(centre, R) of an eraser's wiping face in the world."""
    name = _site(model, prefix + built["pad_site"], built["pad_site"])
    sid = model.site(name).id
    bid = int(model.site_bodyid[sid])
    return (np.asarray(data.site_xpos[sid], float).copy(),
            np.asarray(data.xmat[bid], float).reshape(3, 3).copy())


def axis_world(model, data, built: dict, prefix: str = "") -> np.ndarray:
    """The marker's own axis, nib-ward, in the world."""
    name = _body(model, prefix + built["base_body"], built["base_body"])
    R = np.asarray(data.xmat[model.body(name).id], float).reshape(3, 3)
    return R @ np.asarray(built["axis_local"], float)


def _site(model, *names):
    import mujoco
    for n in names:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) >= 0:
            return n
    hits = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
            for i in range(model.nsite)]
    hits = [h for h in hits if h and h.endswith(names[-1])]
    if len(hits) == 1:
        return hits[0]
    raise KeyError(f"cannot resolve site {names[-1]!r} (matches {hits})")


def _body(model, *names):
    import mujoco
    for n in names:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) >= 0:
            return n
    hits = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in range(model.nbody)]
    hits = [h for h in hits if h and h.endswith(names[-1])]
    if len(hits) == 1:
        return hits[0]
    raise KeyError(f"cannot resolve body {names[-1]!r} (matches {hits})")


# ===========================================================================
def preview_stills(built: dict, out_dir: str | Path) -> list[str]:
    import mujoco
    from PIL import Image
    model = load(built["xml"])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    out_dir = Path(out_dir)
    paths = []
    with mujoco.Renderer(model, height=420, width=560) as r:
        for tag, az, el in (("side", 90, -12), ("top", 90, -80),
                            ("end", 0, -18)):
            cam = free_camera(np.asarray(data.xpos[1], float), 0.34, az, el)
            r.update_scene(data, camera=cam)
            p = out_dir / f"{built['base_body']}_{tag}.png"
            Image.fromarray(r.render().copy()).save(p)
            paths.append(str(p))
    return paths


def demo(built: dict, out_mp4: str | Path, fps: int = 30,
         seconds: float = 3.0) -> dict:
    """Drop it on the floor and let it settle — the cheapest proof that the
    thing is a sane rigid body (right mass, right footprint, does not jitter
    or sink) before any hand goes near it."""
    import mujoco
    model = load(built["xml"])
    data = mujoco.MjData(model)
    bid = model.body(built["base_body"]).id
    adr = int(model.jnt_qposadr[model.body_jntadr[bid]])
    data.qpos[adr + 2] += 0.06
    mujoco.mj_forward(model, data)
    cam = free_camera([0, 0, 0.03], 0.42, 105, -20)
    frames = rollout(model, data, seconds, fps, camera=cam,
                     width=560, height=420)
    out = write_video(frames, out_mp4, fps=fps)
    out["stills"] = save_stills(frames, Path(out_mp4).parent,
                                Path(out_mp4).stem + "_still", n=3)
    rest = float(data.xpos[bid][2])
    out["measured"] = {"rest_height_m": round(rest, 5),
                       "mass_kg": round(float(model.body_mass[bid]), 5)}
    out["verdict"] = (f"settled at z={rest*1000:.1f} mm carrying "
                      f"{model.body_mass[bid]*1000:.1f} g")
    return out


# ---------------------------------------------------------------------------


def construct_marker(out_dir, preview=False, **params):
    """Build a marker (style is fixed to "marker"). Any other DEFAULTS key may be set."""
    built = build(dict(params, style="marker"), out_dir)
    if preview:
        built["stills"] = preview_stills(built, out_dir)
    return built


if __name__ == "__main__":
    import json as _json
    import sys as _sys
    print(_json.dumps(construct_marker(_sys.argv[1] if len(_sys.argv) > 1 else ".", preview=True), indent=1, default=str))
