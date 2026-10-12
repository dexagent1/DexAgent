"""Construct open PC tower case: An open tower PC case as a static asset: approximate shell and interior dressing, with a real drive-bay stack whose slots accept the built hard disks.

AN OPEN PC TOWER CASE — the static backdrop object of a drive-insertion
task, with the drive cage EMBEDDED as real, slot-by-slot geometry.

A computer case cannot be scanned into something a task can use: SAM3D fuses
the interior into one blob and the drive bays disappear. But the task never
touches the motherboard either — what it needs is (a) a shell of the right
outer size standing in the right place, (b) interior dressing that reads as
"an open PC" on camera, and (c) a REAL bay stack whose slots accept the
disks. So the shell and dressing are approximate on purpose, and the cage is
the exact `diskbay` rack, generated inside the case body by the same code
that builds it standalone (`diskbay.rack_geoms` — one source of truth).

Local frame (pos = footprint centre, local z=0 IS the support surface):
    +x  toward the case FRONT (the cage column lives at the front)
    +y  the OPEN side (side panel removed) — the robot/camera side
    -y  the intact side panel (motherboard tray against it)
     z  up

The embedded cage's slots open toward +y like every diskbay rack: a disk
enters travelling -y, exactly as in the standalone builder, so
`diskbay.slot_world` / `fits` work unchanged with prefix `<name>_cage`.

    python -m tool_library.skills.construct_pc_case [out_dir]     # build + previews
"""
from __future__ import annotations

TOOL = {'name': 'construct_pc_case',
 'title': 'Construct open PC tower case',
 'type': 'skill',
 'category': 'Object Reconstruction',
 'summary': 'An open tower PC case as a static asset: approximate shell and interior dressing, '
            'with a real drive-bay stack whose slots accept the built hard disks.',
 'description': 'Builds the case shell to its measured outer size with interior dressing that '
                'reads as an open PC, and embeds the exact drive cage of construct_disk_rack '
                'inside it, slot by slot, so the task geometry (where a disk enters and seats) '
                'is exact while the rest is approximate on purpose. Case size, side, bay count '
                'and pitch, colours and placement are parameters; flip_x mirrors the interior.',
 'inputs': [{'name': 'out_dir',
             'type': 'path',
             'description': 'Folder that receives the MJCF, its meshes and '
                            '<name>_params.json.'},
            {'name': 'preview',
             'type': 'bool',
             'description': 'Also render preview stills of the mechanism at its extremes. '
                            'Default False.'},
            {'name': 'name', 'type': 'str', 'description': 'Name. Default pccase'},
            {'name': 'depth',
             'type': 'float',
             'description': 'Local x, front to back. Default 0.5'},
            {'name': 'width',
             'type': 'float',
             'description': 'Local y, open side to closed side. Default 0.235'},
            {'name': 'height', 'type': 'float', 'description': 'Local z. Default 0.5'},
            {'name': 'panel_t', 'type': 'float', 'description': 'Panel t. Default 0.005'},
            {'name': 'shell_rgba',
             'type': 'list[4]',
             'description': 'Shell rgba, colour as rgba in 0..1. Default [0.09, 0.09, 0.1, '
                            '1.0]'},
            {'name': 'psu',
             'type': 'bool',
             'description': 'Dark PSU block, bottom rear. Default True'},
            {'name': 'psu_l', 'type': 'float', 'description': 'Psu l. Default 0.19'},
            {'name': 'psu_h', 'type': 'float', 'description': 'Psu h. Default 0.095'},
            {'name': 'psu_rgba',
             'type': 'list[4]',
             'description': 'Psu rgba, colour as rgba in 0..1. Default [0.13, 0.12, 0.1, 1.0]'},
            {'name': 'psu_accent_rgba',
             'type': 'list[4]',
             'description': 'The gold brand band. Default [0.72, 0.6, 0.22, 1.0]'},
            {'name': 'cooler',
             'type': 'bool',
             'description': 'White tower CPU cooler, mid rear-upper. Default True'},
            {'name': 'cooler_w', 'type': 'float', 'description': 'Cooler w. Default 0.13'},
            {'name': 'cooler_h', 'type': 'float', 'description': 'Cooler h. Default 0.16'},
            {'name': 'cooler_d', 'type': 'float', 'description': 'Cooler d. Default 0.1'},
            {'name': 'cooler_rgba',
             'type': 'list[4]',
             'description': 'Cooler rgba, colour as rgba in 0..1. Default [0.88, 0.88, 0.9, '
                            '1.0]'},
            {'name': 'board_rgba',
             'type': 'list[4]',
             'description': 'Motherboard tray. Default [0.16, 0.16, 0.18, 1.0]'},
            {'name': 'ram_rgba',
             'type': 'list[4]',
             'description': 'Ram rgba, colour as rgba in 0..1. Default [0.3, 0.3, 0.34, 1.0]'},
            {'name': 'cage', 'type': 'dict', 'description': 'Cage'},
            {'name': 'cage_inset_front',
             'type': 'float',
             'description': 'Cage inset front. Default 0.02'},
            {'name': 'cage_lift', 'type': 'float', 'description': 'Cage lift. Default 0.0'},
            {'name': 'flip_x', 'type': 'bool', 'description': 'Flip x. Default False'},
            {'name': 'pos',
             'type': 'list[3]',
             'description': 'Pos, metres, xyz. Default [0.0, 0.0, 0.0]'},
            {'name': 'yaw_deg',
             'type': 'float',
             'description': 'Yaw deg, degrees. Default 0.0'},
            {'name': 'friction',
             'type': 'list[3]',
             'description': 'Friction, MuJoCo friction triple. Default [0.9, 0.02, 0.002]'},
            {'name': 'condim', 'type': 'int', 'description': 'Condim. Default 3'},
            {'name': 'fixed_base',
             'type': 'bool',
             'description': 'A full tower does not slide around. Default True'},
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
 'tags': ['computer', 'case', 'receptacle', 'parametric', 'reconstruct', 'object reconstruction'],
 'used_in': ['Computer Installation'],
 'status': 'tested',
 'entrypoint': 'construct_pc_case',
 'cli': 'python -m tool_library.skills.construct_pc_case out_dir'}


from pathlib import Path

from tool_library.common.mjcf import (FLOOR_XML, LIGHTS_XML, free_camera, load, merge_params,
                     mjcf_header, rgba, save_params, vec)
from tool_library.common import diskbay

DEFAULTS = {
    "name": "pccase",

    # outer shell (metres)
    "depth":   0.50,          # local x, front to back
    "width":   0.235,         # local y, open side to closed side
    "height":  0.50,          # local z
    "panel_t": 0.005,
    "shell_rgba": [0.09, 0.09, 0.10, 1.0],

    # interior dressing (visual-only, no contacts)
    "psu": True,              # dark PSU block, bottom rear
    "psu_l": 0.19, "psu_h": 0.095,
    "psu_rgba":    [0.13, 0.12, 0.10, 1.0],
    "psu_accent_rgba": [0.72, 0.60, 0.22, 1.0],   # the gold brand band
    "cooler": True,           # white tower CPU cooler, mid rear-upper
    "cooler_w": 0.13, "cooler_h": 0.16, "cooler_d": 0.10,
    "cooler_rgba": [0.88, 0.88, 0.90, 1.0],
    "board_rgba":  [0.16, 0.16, 0.18, 1.0],       # motherboard tray
    "ram_rgba":    [0.30, 0.30, 0.34, 1.0],

    # the embedded drive cage (diskbay rack params, or None for a bare case).
    # cage_inset_*: where the cage column sits, measured from the case's
    # FRONT-inner face (+x) and the bottom-inner face.
    "cage": dict(diskbay.DEFAULTS, style="rack"),
    "cage_inset_front": 0.02,
    "cage_lift": 0.0,

    # flip_x mirrors the INTERIOR layout along x (cage column at -x, PSU at
    # +x). A yaw can face the open side anywhere, but it cannot mirror the
    # inside: when the clip shows the cage on the CAMERA'S RIGHT with the
    # PSU on the left (and the reconstruction must face the robot the same
    # way), only a mirrored interior can match — real towers come both ways.
    "flip_x": False,

    # placement + physics
    "pos": [0.0, 0.0, 0.0],
    "yaw_deg": 0.0,
    "friction": [0.9, 0.02, 0.002],
    "condim": 3,
    "fixed_base": True,       # a full tower does not slide around
    "timestep": 0.002,
    "include_floor": True,
}


def build(params: dict | None = None, out_dir: str | Path = ".") -> dict:
    p = merge_params(DEFAULTS, params)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = str(p["name"])
    D, W, H = float(p["depth"]), float(p["width"]), float(p["height"])
    t = float(p["panel_t"])
    sh = rgba(p["shell_rgba"])
    gattr = (f'friction="{" ".join(str(x) for x in p["friction"])}" '
             f'condim="{int(p["condim"])}" '
             f'solref="0.005 1" solimp="0.99 0.999 0.001"')
    vattr = 'mass="0.0001" contype="0" conaffinity="0" group="2"'
    g = []
    # ---- shell: bottom, top, closed side (-y), front (+x), rear (-x) ----
    g.append(f'<geom name="{name}_bottom" type="box" '
             f'size="{vec((D / 2, W / 2, t / 2))}" pos="0 0 {t / 2:.5f}" '
             f'rgba="{sh}" {gattr}/>')
    g.append(f'<geom name="{name}_top" type="box" '
             f'size="{vec((D / 2, W / 2, t / 2))}" '
             f'pos="0 0 {H - t / 2:.5f}" rgba="{sh}" {gattr}/>')
    g.append(f'<geom name="{name}_side" type="box" '
             f'size="{vec((D / 2, t / 2, H / 2))}" '
             f'pos="0 {-(W / 2 - t / 2):.5f} {H / 2:.5f}" '
             f'rgba="{sh}" {gattr}/>')
    g.append(f'<geom name="{name}_front" type="box" '
             f'size="{vec((t / 2, W / 2, H / 2))}" '
             f'pos="{D / 2 - t / 2:.5f} 0 {H / 2:.5f}" rgba="{sh}" {gattr}/>')
    g.append(f'<geom name="{name}_rear" type="box" '
             f'size="{vec((t / 2, W / 2, H / 2))}" '
             f'pos="{-(D / 2 - t / 2):.5f} 0 {H / 2:.5f}" rgba="{sh}" {gattr}/>')
    # ---- interior dressing (visual only) ----
    sgn = -1.0 if p.get("flip_x") else 1.0   # mirrored interior layout
    board_x = sgn * -D * 0.10                # motherboard region, mid-rear
    if p["psu"]:
        pl, ph = float(p["psu_l"]), float(p["psu_h"])
        g.append(f'<geom name="{name}_psu" type="box" '
                 f'size="{vec((pl / 2, W / 2 * 0.88, ph / 2))}" '
                 f'pos="{sgn * -(D / 2 - t - pl / 2):.5f} 0 {t + ph / 2:.5f}" '
                 f'rgba="{rgba(p["psu_rgba"])}" {vattr}/>')
        g.append(f'<geom name="{name}_psu_accent" type="box" '
                 f'size="{vec((pl / 2 * 0.7, 0.001, ph / 2 * 0.28))}" '
                 f'pos="{sgn * -(D / 2 - t - pl / 2):.5f} {W / 2 * 0.88:.5f} '
                 f'{t + ph * 0.5:.5f}" '
                 f'rgba="{rgba(p["psu_accent_rgba"])}" {vattr}/>')
    g.append(f'<geom name="{name}_board" type="box" '
             f'size="{vec((D * 0.26, 0.002, H * 0.30))}" '
             f'pos="{board_x:.5f} {-(W / 2 - t - 0.004):.5f} {H * 0.55:.5f}" '
             f'rgba="{rgba(p["board_rgba"])}" {vattr}/>')
    if p["cooler"]:
        cw, ch, cd = (float(p["cooler_w"]), float(p["cooler_h"]),
                      float(p["cooler_d"]))
        g.append(f'<geom name="{name}_cooler" type="box" '
                 f'size="{vec((cw / 2, cd / 2, ch / 2))}" '
                 f'pos="{board_x:.5f} {-(W / 2 - t - 0.006 - cd / 2):.5f} '
                 f'{H * 0.62:.5f}" '
                 f'rgba="{rgba(p["cooler_rgba"])}" {vattr}/>')
        for k, dx in enumerate((-cw * 0.28, 0.0, cw * 0.28)):
            g.append(f'<geom name="{name}_pipe{k}" type="cylinder" '
                     f'size="0.004 {cd / 2 * 0.8:.5f}" '
                     f'pos="{board_x + dx:.5f} '
                     f'{-(W / 2 - t - 0.006 - cd / 2):.5f} '
                     f'{H * 0.62 + ch / 2 + 0.008:.5f}" '
                     f'euler="90 0 0" rgba="0.75 0.74 0.72 1" {vattr}/>')
        for k in range(3):              # RAM sticks beside the cooler
            g.append(f'<geom name="{name}_ram{k}" type="box" '
                     f'size="0.002 0.004 0.055" '
                     f'pos="{board_x + sgn * (cw / 2 + 0.015 + k * 0.009):.5f} '
                     f'{-(W / 2 - t - 0.012):.5f} {H * 0.62:.5f}" '
                     f'rgba="{rgba(p["ram_rgba"])}" {vattr}/>')
    # ---- the cage: REAL slot geometry, shared with the standalone rack ----
    cage_built = None
    if p["cage"]:
        cp = merge_params(diskbay.DEFAULTS, dict(p["cage"], style="rack"))
        cage_name = f"{name}_cage"
        _, probe = diskbay.rack_geoms(cp, cage_name)
        # the cage's slots open toward +y (the open panel), so its bay DEPTH
        # (outer_d, a drive length) runs along the case's y and its WIDTH
        # (outer_w, a drive width) along the case's x, in the front column
        cage_x = sgn * (D / 2 - t - float(p["cage_inset_front"])
                        - probe["outer_w_m"] / 2)
        cage_y = W / 2 - probe["outer_d_m"] / 2 - 0.004
        cage_z = t + float(p["cage_lift"])
        frag, cage_built = diskbay.rack_geoms(cp, cage_name,
                                              origin=(cage_x, cage_y, cage_z))
        g.append(frag)
        cage_built = {**cage_built, "params": cp, "base_body": name,
                      "name": cage_name,
                      "origin_local": [round(cage_x, 5), round(cage_y, 5),
                                       round(cage_z, 5)],
                      "insert_axis_local": [0.0, -1.0, 0.0]}
    fixed = bool(p["fixed_base"])
    body = f"""
    <body name="{name}" pos="{vec(p['pos'])}" euler="0 0 {p['yaw_deg']}">
{'' if fixed else f'      <freejoint name="{name}_free"/>'}
      {chr(10).join('      ' + x for x in g).lstrip()}
    </body>"""
    xml = mjcf_header(f"{name}_case", timestep=p["timestep"]) + f"""
  <worldbody>{LIGHTS_XML}{FLOOR_XML if p["include_floor"] else ""}{body}
  </worldbody>
</mujoco>
"""
    xml_path = out_dir / f"{name}.xml"
    xml_path.write_text(xml)
    save_params(out_dir / f"{name}_params.json", p)
    out = {
        "xml": str(xml_path),
        "params_json": str(out_dir / f"{name}_params.json"),
        "base_body": name, "params": p,
        "outer_dwh": [round(D, 4), round(W, 4), round(H, 4)],
        "open_side_local": [0.0, 1.0, 0.0],
        "cage": cage_built,
        "note": ("an open tower case; the removed-panel side is local +y — "
                 "face it toward the robot. The embedded cage's bays open "
                 "the same way: disks enter along local -y. Slot/mouth "
                 "sites are named <name>_cage_slot<i>/_mouth<i> (i=0 the "
                 "BOTTOM bay); use diskbay.slot_world with the cage dict "
                 "returned here. Shell+dressing are APPROXIMATE by design; "
                 "the cage is exact."),
    }
    return out


def rebuild(params_json: str | Path, out_dir: str | Path | None = None) -> dict:
    import json
    p = json.loads(Path(params_json).read_text())
    return build(p, out_dir or Path(params_json).parent)


def preview_stills(built: dict, out_dir: str | Path) -> list[str]:
    import mujoco
    from PIL import Image
    model = load(built["xml"])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    out_dir = Path(out_dir)
    D, W, H = built["outer_dwh"]
    paths = []
    with mujoco.Renderer(model, height=480, width=640) as r:
        for tag, az, el, dist in (("open_side", -90, -8, 1.9 * max(D, H)),
                                  ("three_quarter", -135, -22, 2.1 * max(D, H)),
                                  ("cage_side", -60, -10, 1.5 * max(D, H))):
            cam = free_camera([0, 0, H * 0.5], dist, az, el)
            r.update_scene(data, camera=cam)
            p = out_dir / f"{built['base_body']}_{tag}.png"
            Image.fromarray(r.render().copy()).save(p)
            paths.append(str(p))
    return paths


# ---------------------------------------------------------------------------


def construct_pc_case(out_dir, preview=False, **params):
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
    print(_json.dumps(construct_pc_case(out, preview=True), indent=1, default=str))
