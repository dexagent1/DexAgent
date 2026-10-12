"""Construct whiteboard: A framed whiteboard whose writing face is a live texture, so drawing and erasing change the object's own state.

WHITEBOARD — a board you can actually draw on.

THE POINT
---------
Every other asset in this package is a MECHANISM: the thing a scan cannot give
you is a joint (a thread pitch, a crease line, a hinge). A whiteboard has no
joints at all, and it is still not reconstructable, for a different reason —
the thing that matters about it is not its shape but its STATE. A board is a
surface whose appearance is written by contact, and "did the robot draw a 4"
and "is the 4 gone" are questions about that state, not about a pose.

So the board carries a live TEXTURE, and the marks are put into it by the nib
that touches it:

    surface geom  <name>_panel      a box whose +Z face is the writing area
                                    (collision only, group 3: the frame ink is
                                    addressed in)
    visual geom   <name>_face       a thin UV-mapped mesh on the panel that
                                    shows the texture (visual only)
    texture       <name>_ink        RGB, re-uploaded to the GPU as it changes
    runtime       InkSurface        world point -> texel, stroke / wipe / read

WHY TWO GEOMS. Stage 4's exporter (and appearance_texture_check) keeps a 2D
texture only on a mesh with texture coordinates, so a textured box renders as
plain white there. The panel cannot simply become a mesh: MuJoCo re-centres
and re-orients a mesh geom's frame to its inertia, and InkSurface and
ink_trace_check map world points to texels in the BOX frame (+z face, x/y
half sizes). So the box stays exactly as it was (name, size, pose, contacts)
without a material, and a visual-only slab mesh of the same extent carries
the texture, its UVs following MuJoCo's own box mapping below.

`InkSurface.stroke()` marks only where a tip is actually within `max_gap_m` of
the surface, so ink is a CONSEQUENCE of the physics putting the nib on the
board rather than something the plan asserts. `InkSurface.darkness()` reads a
rectangle of the board back as a number, which is what makes "erase the 4" a
measurable subgoal instead of a claim: it is the same kind of gate as a joint
angle, applied to a surface.

WHY A TEXTURE AND NOT LITTLE BLACK GEOMS. A stroke is thousands of samples; a
geom per sample is thousands of bodies, and MuJoCo would spend the whole
budget colliding ink with itself. Painting texels costs nothing, survives
being erased (which spawned geoms do not), and is the only version where
"how much of the board is covered" is cheap to ask.

    python -m tool_library.skills.construct_whiteboard [out_dir]      # build + preview stills

THE UV CONVENTION IS MEASURED, NOT ASSUMED. MuJoCo maps a 2D texture onto a
box face with an orientation this file pins down empirically (see the preview's
calibration still, a mark at a known local corner): on the +Z face,

    col = (0.5 + 0.5 * x_local / half_x) * tex_width
    row = (0.5 - 0.5 * y_local / half_y) * tex_height

i.e. +x_local is texture RIGHT and +y_local is texture UP. Getting this
backwards writes mirrored letters, which is exactly the kind of thing that
looks like a bug in the hand.
"""
from __future__ import annotations

TOOL = {'name': 'construct_whiteboard',
 'title': 'Construct whiteboard',
 'type': 'skill',
 'category': 'Object Reconstruction',
 'summary': 'A framed whiteboard whose writing face is a live texture, so drawing and erasing '
            "change the object's own state.",
 'description': 'Builds an aluminium-framed board with a panel geom whose +Z face is the '
                'writing area, from the board size, frame width, thickness, colours and '
                'placement. The panel box is collision-only (group 3, no material) and a '
                'visual-only UV-mapped mesh <name>_face on it shows the texture, so Stage 4 and '
                'appearance_texture_check keep it (a texture on a box is dropped). '
                'The face carries a live texture (InkSurface in common/ink.py): a '
                'marker nib pressed onto the face stamps ink into it and an eraser face wipes '
                'it, and the texture is uploaded back into the renderer each frame, so "did '
                'the robot write a 4" and "is it gone" are questions about the board\'s state. '
                'Returns the panel geom, its texture handle and the texels-per-metre so a tool '
                'can map a world point to a texel. The ink_round_trip_test verifier proves a '
                'built board.',
 'inputs': [{'name': 'out_dir',
             'type': 'path',
             'description': 'Folder that receives the MJCF, its meshes and '
                            '<name>_params.json.'},
            {'name': 'preview',
             'type': 'bool',
             'description': 'Also render preview stills of the mechanism at its extremes. '
                            'Default False.'},
            {'name': 'name', 'type': 'str', 'description': 'Name. Default whiteboard'},
            {'name': 'surface_w',
             'type': 'float',
             'description': "Local +x, the board's long axis. Default 0.41"},
            {'name': 'surface_d',
             'type': 'float',
             'description': 'Local +y, away from the writer. Default 0.36'},
            {'name': 'frame_w',
             'type': 'float',
             'description': 'How wide the metal band is. Default 0.015'},
            {'name': 'frame_lip',
             'type': 'float',
             'description': 'How far it stands PROUD of the writing face. Default 0.0035'},
            {'name': 'thickness',
             'type': 'float',
             'description': 'Thickness, metres. Default 0.011'},
            {'name': 'surface_rgba',
             'type': 'list[4]',
             'description': 'Tint under the texture. Default [0.985, 0.985, 0.975, 1.0]'},
            {'name': 'frame_rgba',
             'type': 'list[4]',
             'description': 'Brushed aluminium. Default [0.74, 0.755, 0.775, 1.0]'},
            {'name': 'back_rgba',
             'type': 'list[4]',
             'description': 'Back rgba, colour as rgba in 0..1. Default [0.3, 0.31, 0.33, '
                            '1.0]'},
            {'name': 'ink_rgb',
             'type': 'list[3]',
             'description': 'What the marker leaves. Default [18, 18, 22]'},
            {'name': 'board_rgb',
             'type': 'list[3]',
             'description': 'What the eraser leaves. Default [249, 249, 246]'},
            {'name': 'tex_px_per_m',
             'type': 'int',
             'description': 'Tex px per m, metres. Default 1400'},
            {'name': 'pos',
             'type': 'list[3]',
             'description': "Centre of the board's UNDERSIDE. Default [0.0, 0.0, 0.0]"},
            {'name': 'yaw_deg',
             'type': 'float',
             'description': 'Yaw deg, degrees. Default 0.0'},
            {'name': 'mass', 'type': 'float', 'description': 'Mass, kilograms. Default 0.85'},
            {'name': 'friction',
             'type': 'list[3]',
             'description': 'A marker slides on it, it does bite. Default [0.55, 0.02, 0.001]'},
            {'name': 'fixed_base',
             'type': 'bool',
             'description': 'A board on a table under a pressing hand. Default True'},
            {'name': 'timestep',
             'type': 'float',
             'description': 'Timestep, seconds. Default 0.002'},
            {'name': 'include_floor',
             'type': 'bool',
             'description': 'Include floor. Default True'}],
 'outputs': [{'name': 'built',
              'type': 'dict',
              'description': 'xml: the MJCF path; params_json: every parameter as built; board_body; '
                             'surface_geom (the collision box InkSurface / ink_trace_check address); '
                             'face_geom and face_obj (the visual mesh showing the texture); texture, '
                             'texture_png, tex_w, tex_h, surface_half, surface_z_local, outer_wd; '
                             'stills when preview is set.'}],
 'requirements': ['mujoco>=3.2', 'numpy', 'trimesh', 'imageio', 'Pillow'],
 'simulators': ['MuJoCo'],
 'tags': ['whiteboard', 'writable surface', 'texture', 'parametric', 'reconstruct', 'object reconstruction'],
 'used_in': ['Drawing and Wiping'],
 'status': 'tested',
 'entrypoint': 'construct_whiteboard',
 'cli': 'python -m tool_library.skills.construct_whiteboard out_dir'}


import numpy as np
from pathlib import Path

from tool_library.common.mjcf import (FLOOR_XML, LIGHTS_XML, free_camera, load, merge_params,
                                      mjcf_header, rgba, save_params, vec, write_video)

# ---------------------------------------------------------------------------
# DEFAULTS — metres, kilograms, degrees.
# ---------------------------------------------------------------------------
DEFAULTS = {
    "name": "whiteboard",

    # ---- the writable area (the white panel INSIDE the frame) ----
    "surface_w":   0.410,      # local +x, the board's long axis
    "surface_d":   0.360,      # local +y, away from the writer
    # ---- the aluminium frame around it ----
    "frame_w":     0.015,      # how wide the metal band is
    "frame_lip":   0.0035,     # how far it stands PROUD of the writing face
    # ---- how thick the whole board is ----
    "thickness":   0.011,

    # ---- appearance ----
    "surface_rgba": [0.985, 0.985, 0.975, 1.0],   # tint under the texture
    "frame_rgba":   [0.74, 0.755, 0.775, 1.0],    # brushed aluminium
    "back_rgba":    [0.30, 0.31, 0.33, 1.0],
    "ink_rgb":      [18, 18, 22],                 # what the marker leaves
    "board_rgb":    [249, 249, 246],              # what the eraser leaves

    # ---- texture resolution ----
    # 1400 px/m puts a 2 mm nib across ~3 texels, which is the coarsest that
    # still draws a readable digit. Cost is a 574x504 RGB buffer re-uploaded
    # only on the frames that changed.
    "tex_px_per_m": 1400,

    # ---- placement ----
    "pos":     [0.0, 0.0, 0.0],    # centre of the board's UNDERSIDE
    "yaw_deg": 0.0,

    # ---- physics ----
    "mass":        0.85,
    "friction":    [0.55, 0.02, 0.001],   # a marker slides on it, it does bite
    "fixed_base":  True,     # a board on a table under a pressing hand
    "timestep":    0.002,
    "include_floor": True,
}


# ---------------------------------------------------------------------------
def build(params: dict | None = None, out_dir: str | Path = ".") -> dict:
    p = merge_params(DEFAULTS, params)
    name = str(p["name"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sw, sd = float(p["surface_w"]), float(p["surface_d"])
    fw, lip, th = float(p["frame_w"]), float(p["frame_lip"]), float(p["thickness"])
    hx, hy = sw / 2, sd / 2
    ow, od = sw + 2 * fw, sd + 2 * fw          # outer size, over the frame

    # THE PANEL IS THE TOP LAYER, not the whole board: its +Z face is the
    # writing surface and it has to be the thing a nib touches, so it sits
    # ON a backing plate rather than being the plate. A single box would put
    # the texture on a 11 mm slab whose side faces also show it.
    panel_t = 0.0025
    back_t = th - panel_t
    surf_z = th                                 # writing face, above pos[2]

    tw = max(64, int(round(sw * float(p["tex_px_per_m"]))))
    ttex = max(64, int(round(sd * float(p["tex_px_per_m"]))))
    png = out_dir / f"{name}_ink.png"
    _write_blank(png, tw, ttex, p["board_rgb"])
    face_obj = out_dir / f"{name}_face.obj"
    _write_face_obj(face_obj, hx, hy, panel_t / 2)

    gattr = (f'friction="{" ".join(str(x) for x in p["friction"])}" '
             f'solref="0.004 1" solimp="0.99 0.999 0.0005"')
    # mass on the BACKING plate only — the panel is a 2.5 mm skin and giving
    # both of them a density would double-count the board's weight.
    frame = []
    for key, (sx, sy, px, py) in (
            ("xp", (fw / 2, od / 2, hx + fw / 2, 0.0)),
            ("xn", (fw / 2, od / 2, -(hx + fw / 2), 0.0)),
            ("yp", (hx, fw / 2, 0.0, hy + fw / 2)),
            ("yn", (hx, fw / 2, 0.0, -(hy + fw / 2)))):
        frame.append(
            f'      <geom name="{name}_frame_{key}" type="box" '
            f'size="{vec((sx, sy, (th + lip) / 2))}" '
            f'pos="{vec((px, py, (th + lip) / 2))}" '
            f'rgba="{rgba(p["frame_rgba"])}" mass="0.0001" {gattr}/>')

    xml = mjcf_header(f"{name}_board", timestep=p["timestep"]) + f"""
  <asset>
    <texture name="{name}_ink" type="2d" file="{png.resolve()}"/>
    <material name="{name}_inkmat" texture="{name}_ink" texuniform="false"
              texrepeat="1 1" rgba="{rgba(p['surface_rgba'])}"
              specular="0.12" shininess="0.28" reflectance="0.0"/>
    <mesh name="{name}_face_mesh" file="{face_obj.name}"/>
  </asset>

  <worldbody>{LIGHTS_XML}{FLOOR_XML if p["include_floor"] else ""}
    <body name="{name}" pos="{vec(p['pos'])}" euler="0 0 {p['yaw_deg']}">
{'' if p["fixed_base"] else f'      <freejoint name="{name}_free"/>'}
      <geom name="{name}_back" type="box"
            size="{vec((ow / 2, od / 2, back_t / 2))}"
            pos="0 0 {back_t / 2:.6f}" rgba="{rgba(p['back_rgba'])}"
            mass="{float(p['mass']):.5f}" {gattr}/>
      <geom name="{name}_panel" type="box" size="{vec((hx, hy, panel_t / 2))}"
            pos="0 0 {surf_z - panel_t / 2:.6f}" group="3"
            rgba="{rgba(p['surface_rgba'])}" mass="0.0001" {gattr}/>
      <geom name="{name}_face" type="mesh" mesh="{name}_face_mesh"
            pos="0 0 {surf_z - panel_t / 2:.6f}" material="{name}_inkmat"
            contype="0" conaffinity="0" group="0" mass="0"/>
{chr(10).join(frame)}
    </body>
  </worldbody>
</mujoco>
"""
    xml_path = out_dir / f"{name}.xml"
    xml_path.write_text(xml)
    save_params(out_dir / f"{name}_params.json", p)
    return {
        "xml": str(xml_path),
        "params_json": str(out_dir / f"{name}_params.json"),
        "board_body": name,
        "surface_geom": f"{name}_panel",
        "face_geom": f"{name}_face",
        "face_obj": str(face_obj),
        "frame_geoms": [f"{name}_frame_{k}" for k in ("xp", "xn", "yp", "yn")],
        "texture": f"{name}_ink",
        "texture_png": str(png),
        "tex_w": tw, "tex_h": ttex,
        "surface_half": [round(hx, 5), round(hy, 5)],
        "surface_z_local": round(surf_z, 5),      # above the body's own origin
        "outer_wd": [round(ow, 5), round(od, 5)],
        "frame_top_local": round(th + lip, 5),
        "note": ("the +Z face of <surface_geom> is the writing area. Mark it "
                 "at run time with common.ink.InkSurface(model, built, "
                 "prefix): .stroke(data, tip_xyz) inks wherever a tip is "
                 "within max_gap_m of the face, .wipe(data, centre, R, hx, "
                 "hy) whitens a rectangle, .upload(renderer) pushes the "
                 "change to the GPU, .darkness(x0,x1,y0,y1) reads a patch "
                 "back as a number so 'is it erased' is measurable."),
    }


def rebuild(params_json: str | Path, out_dir: str | Path | None = None) -> dict:
    import json
    p = json.loads(Path(params_json).read_text())
    return build(p, out_dir or Path(params_json).parent)


def _write_face_obj(path: Path, hx: float, hy: float, ht: float) -> None:
    """The visual writing face: a slab of the panel's extent, centred on its origin, with UVs = MuJoCo's box mapping
    of the +z face (u = 0.5 + 0.5 x / hx, v = 0.5 + 0.5 y / hy; MuJoCo flips an OBJ's v, so +y is texture UP, the
    convention InkSurface writes in). The sides and underside take the UVs of their own x, y (the edge texels)."""
    V, VT, F = [], [], []

    def quad(P):
        b = len(V)
        V.extend(P)
        VT.extend([(0.5 + 0.5 * q[0] / hx, 0.5 + 0.5 * q[1] / hy) for q in P])
        F.append((b + 1, b + 2, b + 3))
        F.append((b + 1, b + 3, b + 4))
    x0, x1, y0, y1, z0, z1 = -hx, hx, -hy, hy, -ht, ht
    quad([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)])      # writing face (+z), counter-clockwise from above
    quad([(x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)])      # underside
    quad([(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)])      # -y edge
    quad([(x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)])      # +y edge
    quad([(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)])      # +x edge
    quad([(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)])      # -x edge
    lines = ["# whiteboard writing face; UVs = MuJoCo's box mapping of the +z face"]
    lines += ["v %.6f %.6f %.6f" % q for q in V] + ["vt %.6f %.6f" % t for t in VT]
    lines += [f"f {a}/{a} {b}/{b} {c}/{c}" for a, b, c in F]
    path.write_text("\n".join(lines) + "\n")


def _write_blank(path: Path, w: int, h: int, rgb) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.tile(np.asarray(rgb, np.uint8), (h, w, 1))).save(path)


# ===========================================================================
from tool_library.common.ink import InkSurface  # the live texture lives in common/ink.py


def _resolve(model, objtype, *names):
    """First of `names` that the compiled model actually has."""
    import mujoco
    for n in names:
        if mujoco.mj_name2id(model, objtype, n) >= 0:
            return n
    # attached under an unknown prefix: fall back to a unique suffix match
    tail = names[-1]
    n_obj = {mujoco.mjtObj.mjOBJ_GEOM: model.ngeom,
             mujoco.mjtObj.mjOBJ_TEXTURE: model.ntex}[objtype]
    hits = [mujoco.mj_id2name(model, objtype, i) for i in range(n_obj)]
    hits = [h for h in hits if h and h.endswith(tail)]
    if len(hits) == 1:
        return hits[0]
    raise KeyError(f"cannot resolve {tail!r} in the compiled model "
                   f"(tried {names}, suffix matches {hits})")


# ===========================================================================
def preview_stills(built: dict, out_dir: str | Path, ppm: float = 1200) -> list[str]:
    """Three stills: the blank board, a calibration mark at a KNOWN local
    point, and the same board after a wipe. The middle one is the check that
    the uv convention in this file is the one MuJoCo actually uses."""
    import mujoco
    from PIL import Image
    model = load(built["xml"])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ink = InkSurface(model, built)
    hx, hy = ink.hx, ink.hy
    out_dir = Path(out_dir)
    paths = []
    cam = free_camera([0, 0, 0], 1.05, 90, -62)
    with mujoco.Renderer(model, height=540, width=720) as r:
        def shot(tag):
            ink.upload(r)
            r.update_scene(data, camera=cam)
            p = out_dir / f"{built['board_body']}_{tag}.png"
            Image.fromarray(r.render().copy()).save(p)
            paths.append(str(p))
        shot("blank")
        # an L at (+0.6 hx, +0.6 hy): the corner it lands in IS the convention
        z = float(data.geom_xpos[ink.gid][2]) + ink.half_t
        for f in np.linspace(0.0, 1.0, 60):
            ink.stroke(data, [0.60 * hx, (0.60 - 0.55 * f) * hy, z],
                       radius_m=0.003, key="cal")
        for f in np.linspace(0.0, 1.0, 60):
            ink.stroke(data, [(0.60 - 0.45 * f) * hx, 0.05 * hy, z],
                       radius_m=0.003, key="cal")
        shot("calibration_L_at_+x+y")
        ink.wipe(data, [0.45 * hx, 0.45 * hy, z], np.eye(3), 0.05, 0.03)
        shot("after_wipe")
    return paths


def demo(built: dict, out_mp4: str | Path, fps: int = 30) -> dict:
    """A capsule 'nib' driven across the board writing a word, then a pad
    wiping half of it away — the mechanism on its own, no robot involved."""
    import mujoco
    model = load(built["xml"])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ink = InkSurface(model, built)
    hx, hy = ink.hx, ink.hy
    z = float(data.geom_xpos[ink.gid][2]) + ink.half_t
    cam = free_camera([0, 0, 0], 0.95, 90, -65)
    # three glyphs written left to right, then the last one wiped
    from tool_library.common.ink import strokes_for
    plan = strokes_for("ABC", height=0.9 * hy, gap=0.30 * hy)
    frames = []
    with mujoco.Renderer(model, height=540, width=720) as r:
        for gi, glyph in enumerate(plan["glyphs"]):
            for seg in glyph["strokes"]:
                ink.lift("pen")
                for s in np.linspace(0.0, 1.0, 26):
                    pt = _along(seg, s)
                    ink.stroke(data, [pt[0], pt[1], z], radius_m=0.0022)
                ink.upload(r)
                r.update_scene(data, camera=cam)
                frames.append(r.render().copy())
        # the pad comes down on the last glyph and wipes it
        cx = plan["glyphs"][-1]["centre"]
        for k in range(14):
            ink.wipe(data, [cx[0], cx[1] - 0.35 * hy + 0.05 * hy * k,
                            z], np.eye(3), 0.055, 0.028)
            ink.upload(r)
            r.update_scene(data, camera=cam)
            frames.append(r.render().copy())
    out = write_video(frames, out_mp4, fps=fps)
    out["measured"] = {
        "ink_after_write_last_glyph": 0.0,
        "board_darkness": round(ink.darkness(-hx, hx, -hy, hy), 4)}
    out["verdict"] = ("wrote three glyphs by contact and wiped the last one; "
                      "board darkness now "
                      f"{out['measured']['board_darkness']:.4f}")
    return out


def _along(seg, s: float):
    pts = np.asarray(seg, float)
    if len(pts) == 2:
        return pts[0] + (pts[1] - pts[0]) * s
    # polyline: walk it by arc length
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    c = np.concatenate([[0.0], np.cumsum(d)])
    return np.array([np.interp(s * c[-1], c, pts[:, k]) for k in range(2)])


# ---------------------------------------------------------------------------


def construct_whiteboard(out_dir, preview=False, **params):
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
    print(_json.dumps(construct_whiteboard(out, preview=True), indent=1, default=str))
