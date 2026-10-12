"""A DRIVE-BAY STACK AND THE DISKS THAT GO INTO IT.

Two styles, built to fit each other, because the task IS the fit: "put the
hard disks into the stack" only exists in sim if every slot's inner opening
really is a few millimetres wider than the disk that must enter it. A SAM3D
scan of a drive cage is a fused blob — no slot in it can accept anything —
so the cage is CONSTRUCTED from measured numbers instead:

    style="rack"   an open skeletal column of N identical horizontal bays
                   (a PC hard-disk cage, a dish rack, a tray tower): four
                   corner posts, per-level shelf rails the disk slides on,
                   thin side/back walls, top+bottom plates. STATIC by
                   default. Every slot declares itself: `<name>_slot<i>`
                   (site ON the slot floor at bay centre, i=0 is the BOTTOM
                   bay) and `<name>_mouth<i>` (site in the middle of that
                   slot's opening). Insertion is along local -y: the mouth
                   faces +y.

    style="disk"   a 3.5-inch hard-disk drive in a black tray sled: silver
                   top plate, dark body, tray side rails and a front handle
    style="mount"  the SAME sled with no drive in it — an empty black
                   caddy: floor plate, side walls with screw bosses, an
                   open top framed by two rails, and the front handle.
                   Its collision geometry is identical to style="disk" of
                   the same dimensions, so a trajectory recorded against
                   one replays against the other frame for frame.
                   bar. FREE body. The handle end is local +y — the end
                   that stays out of the cage; the drive enters back-first
                   (local -y first).

The rack does not guess the disk: `slot_for(disk_params, ...)` computes the
slot opening FROM the disk's own outer dims plus explicit clearances, so
rack and disk stay consistent by construction. Use `fits(rack, disk)` to
verify the pair numerically before compiling a scene around it.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

from .mjcf import (FLOOR_XML, LIGHTS_XML, free_camera, load, merge_params,
                     mjcf_header, rgba, rollout, save_params, save_stills,
                     vec, write_video)

DEFAULTS = {
    "name": "diskbay",
    "style": "rack",              # rack | disk

    # ---- style="rack" ----
    "n_slots":     8,
    "slot_w":      0.118,         # inner opening, across the disk width (x)
    "slot_h":      0.040,         # inner opening height (z), floor to ceiling
    "bay_depth":   0.155,         # how deep a disk can go (y)
    "rail_reach":  0.014,         # how far each shelf rail sticks into the
                                  #   slot — the ledge the disk rests on
    "rail_t":      0.004,         # shelf rail thickness (z)
    "wall_t":      0.004,         # side/back wall thickness
    "post_t":      0.010,         # corner post square section
    "plate_t":     0.006,         # top + bottom plates
    "frame_rgba":  [0.10, 0.10, 0.11, 1.0],   # matte near-black steel
    "rail_rgba":   [0.16, 0.16, 0.17, 1.0],
    "open_back":   False,         # True drops the back wall (pure ledge rack)

    # ---- style="disk" ----
    "disk_l":      0.147,         # drive length (y, insertion direction)
    "disk_w":      0.102,         # drive width (x)
    "disk_h":      0.026,         # drive height (z)
    "tray_t":      0.006,         # tray side-rail thickness (adds to width)
    "tray_h_frac": 0.85,          # tray rail height as a fraction of disk_h
    "handle_l":    0.012,         # handle bar sticking out past the +y end
    "top_rgba":    [0.71, 0.71, 0.73, 1.0],   # bare-drive silver top
    "body_rgba":   [0.22, 0.22, 0.23, 1.0],
    "tray_rgba":   [0.07, 0.07, 0.08, 1.0],
    "label_rgba":  None,          # e.g. [0.92,0.92,0.90,1] paints a white
                                  #   label patch on the top face
    "mass":        0.45,

    # ---- placement + physics (both styles) ----
    "pos":      [0.0, 0.0, 0.0],  # footprint centre on the support:
                                  #   local z=0 IS the support surface
    "yaw_deg":  0.0,
    "friction": [0.75, 0.02, 0.002],
    "condim":   6,                # a slid tray must not glide forever
    "fixed_base": None,           # None -> rack fixed, disk free
    "timestep": 0.002,
    "include_floor": True,
}

_STYLES = {"rack", "disk", "mount"}


def build(params: dict | None = None, out_dir: str | Path = ".") -> dict:
    p = merge_params(DEFAULTS, params)
    style = str(p["style"])
    if style not in _STYLES:
        raise ValueError(f"style must be one of {sorted(_STYLES)}, not {style!r}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if style == "rack":
        return _build_rack(p, out_dir)
    return _build_disk(p, out_dir, mount=(style == "mount"))


def rebuild(params_json: str | Path, out_dir: str | Path | None = None) -> dict:
    import json
    p = json.loads(Path(params_json).read_text())
    return build(p, out_dir or Path(params_json).parent)


def slot_for(disk: dict, side_clear: float = 0.004,
             top_clear: float = 0.012) -> dict:
    """The slot opening that FITS a given disk build's params: outer tray
    width + side clearance each side, disk height + top clearance, and a bay
    deep enough to swallow the drive body. Feed the result straight into the
    rack params."""
    w = float(disk["disk_w"]) + 2 * float(disk.get("tray_t", 0.006))
    return {"slot_w": round(w + 2 * side_clear, 4),
            "slot_h": round(float(disk["disk_h"]) + top_clear, 4),
            "bay_depth": round(float(disk["disk_l"]) + 0.008, 4)}


def fits(rack: dict, disk: dict) -> dict:
    """Numeric fit check between a built rack and a built disk (their return
    dicts or their params): per-axis clearance in metres, and a verdict. A
    negative number names the axis that blocks insertion."""
    rp, dp = rack.get("params", rack), disk.get("params", disk)
    tray_w = float(dp["disk_w"]) + 2 * float(dp.get("tray_t", 0.006))
    dx = float(rp["slot_w"]) - tray_w
    dz = float(rp["slot_h"]) - float(dp["disk_h"])
    dy = float(rp["bay_depth"]) - float(dp["disk_l"])
    ok = dx > 0 and dz > 0 and dy > -0.02
    return {"clear_x_m": round(dx, 4), "clear_z_m": round(dz, 4),
            "depth_margin_m": round(dy, 4), "ok": bool(ok),
            "verdict": ("disk fits every slot with "
                        f"{dx*1000:.0f} mm lateral / {dz*1000:.0f} mm vertical "
                        f"clearance" if ok else
                        "disk does NOT fit — enlarge the slot or shrink the disk")}


# ---------------------------------------------------------------------------
def rack_geoms(p: dict, name: str, origin=(0.0, 0.0, 0.0)) -> tuple[str, dict]:
    """The rack's geoms + sites as an XML fragment in an ENCLOSING body's
    frame, with the rack's own footprint centre at `origin` (origin[2] = the
    surface its bottom plate sits on). Used both by _build_rack (origin at
    the body origin) and by pccase, which embeds the cage inside the case
    body — one source of truth for the geometry either way.

    Returns (xml, info) where info carries slot_floor_z / slot centres /
    mouths in the ENCLOSING body's frame."""
    ox, oy, oz = (float(v) for v in origin)
    n = int(p["n_slots"])
    sw, sh = float(p["slot_w"]), float(p["slot_h"])
    bd = float(p["bay_depth"])
    rr, rt = float(p["rail_reach"]), float(p["rail_t"])
    wt, pt = float(p["wall_t"]), float(p["post_t"])
    plt = float(p["plate_t"])
    fr, rl = rgba(p["frame_rgba"]), rgba(p["rail_rgba"])
    gattr = (f'friction="{" ".join(str(x) for x in p["friction"])}" '
             f'condim="{int(p["condim"])}" '
             f'solref="0.005 1" solimp="0.99 0.999 0.001"')
    pitch = sh + rt
    inner_h = n * pitch - rt          # bottom plate top .. top plate underside
    total_h = inner_h + 2 * plt
    half_x_out = sw / 2 + wt          # outside face of the rail lips
    # the lip must out-reach the headroom, or a lifted disk could ride over
    # it and leave sideways: lip_h > slot_h - disk_h for every sane disk
    lip_h = 0.65 * sh
    xml, sites = [], []
    info = {"n_slots": n, "pitch_m": round(pitch, 5),
            "total_h_m": round(total_h, 5),
            "outer_w_m": round(2 * half_x_out, 5),
            "outer_d_m": round(bd + (0.0 if p["open_back"] else wt), 5),
            "slot_floor_z": [], "slot_centre_local": [], "mouth_local": []}
    # plates
    xml.append(f'<geom name="{name}_plate_bot" type="box" '
               f'size="{vec((half_x_out, bd / 2, plt / 2))}" '
               f'pos="{vec((ox, oy, oz + plt / 2))}" rgba="{fr}" {gattr}/>')
    xml.append(f'<geom name="{name}_plate_top" type="box" '
               f'size="{vec((half_x_out, bd / 2, plt / 2))}" '
               f'pos="{vec((ox, oy, oz + total_h - plt / 2))}" '
               f'rgba="{fr}" {gattr}/>')
    # back wall only — the sides stay OPENWORK (per-level rail lips + posts),
    # which is both the look of a real cage and all the physics needs
    if not p["open_back"]:
        xml.append(f'<geom name="{name}_back" type="box" '
                   f'size="{vec((half_x_out, wt / 2, inner_h / 2))}" '
                   f'pos="{vec((ox, oy - bd / 2 - wt / 2, oz + plt + inner_h / 2))}" '
                   f'rgba="{fr}" {gattr}/>')
    # corner posts, proud of the lips — the skeletal look of a real cage
    for sx in (-1, 1):
        for sy in (-1, 1):
            xml.append(f'<geom name="{name}_post{"LR"[sx > 0]}{"BF"[sy > 0]}" '
                       f'type="box" size="{vec((pt / 2, pt / 2, total_h / 2))}" '
                       f'pos="{vec((ox + sx * (half_x_out + pt / 2), oy + sy * (bd / 2 - pt / 2), oz + total_h / 2))}" '
                       f'rgba="{fr}" {gattr}/>')
    # per-level L-rails: a ledge the disk RESTS on + a lip that pens it in.
    # Level i's floor is the TOP of its pair of ledges; the lips read as the
    # horizontal bands of the cage's open lattice from the side.
    for i in range(n):
        z_floor = oz + plt + i * pitch
        for sx in (-1, 1):
            if i:                      # level 0 rests on the bottom plate
                cx = ox + sx * (sw / 2 - rr / 2 + wt / 2)
                xml.append(f'<geom name="{name}_rail{i}{"LR"[sx > 0]}" '
                           f'type="box" '
                           f'size="{vec(((rr + wt) / 2, bd / 2, rt / 2))}" '
                           f'pos="{vec((cx, oy, z_floor - rt / 2))}" '
                           f'rgba="{rl}" {gattr}/>')
            xml.append(f'<geom name="{name}_lip{i}{"LR"[sx > 0]}" '
                       f'type="box" '
                       f'size="{vec((wt / 2, bd / 2, lip_h / 2))}" '
                       f'pos="{vec((ox + sx * (sw / 2 + wt / 2), oy, z_floor + lip_h / 2))}" '
                       f'rgba="{fr}" {gattr}/>')
        sites.append(f'<site name="{name}_slot{i}" '
                     f'pos="{vec((ox, oy, z_floor))}" size="0.002" group="4" '
                     f'rgba="0 1 0 0.3"/>')
        sites.append(f'<site name="{name}_mouth{i}" '
                     f'pos="{vec((ox, oy + bd / 2 + 0.02, z_floor + sh / 2))}" '
                     f'size="0.002" group="4" rgba="1 0 0 0.3"/>')
        info["slot_floor_z"].append(round(z_floor - oz, 5))
        info["slot_centre_local"].append([round(ox, 5), round(oy, 5),
                                          round(z_floor - oz + sh / 2, 5)])
        info["mouth_local"].append([round(ox, 5),
                                    round(oy + bd / 2 + 0.02, 5),
                                    round(z_floor - oz + sh / 2, 5)])
    return "\n      ".join(xml + sites), info


def _build_rack(p: dict, out_dir: Path) -> dict:
    name = str(p["name"])
    fixed = True if p["fixed_base"] is None else bool(p["fixed_base"])
    frag, info = rack_geoms(p, name)
    body = f"""
    <body name="{name}" pos="{vec(p['pos'])}" euler="0 0 {p['yaw_deg']}">
{'' if fixed else f'      <freejoint name="{name}_free"/>'}
      {frag}
    </body>"""
    xml = mjcf_header(f"{name}_rack", timestep=p["timestep"]) + f"""
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
        "style": "rack", "base_body": name, "params": p,
        **info,
        "insert_axis_local": [0.0, -1.0, 0.0],
        "slot_inner": [round(float(p["slot_w"]), 4),
                       round(float(p["bay_depth"]), 4),
                       round(float(p["slot_h"]), 4)],
        "note": (f"an open {info['n_slots']}-bay stack; slot i's floor is at "
                 f"local z=slot_floor_z[i] (i=0 the BOTTOM bay), its opening "
                 f"faces local +y at site <{name}_mouth<i>>. To insert: "
                 f"approach the mouth, then translate along local -y until "
                 f"the disk's centre reaches <{name}_slot<i>> + disk_h/2. "
                 f"Verify the pair with diskbay.fits(rack, disk) before "
                 f"building a scene on it."),
    }


# ---------------------------------------------------------------------------
def _build_disk(p: dict, out_dir: Path, mount: bool = False) -> dict:
    """style="disk", or with mount=True the empty caddy of the same size.

    The COLLISION set is byte-identical between the two — same boxes, same
    positions, same mass and friction — because the point of the mount is
    to be the same object wearing a different face: a trajectory recorded
    against the drive replays against the mount without re-solving
    anything. Only what you SEE differs: the drive's body and top plate go
    invisible and a tray's own detail (floor, walls, screw bosses, top
    rails) is drawn in visual-only geoms.
    """
    name = str(p["name"])
    L, W, H = float(p["disk_l"]), float(p["disk_w"]), float(p["disk_h"])
    tt = float(p["tray_t"])
    th = H * float(p["tray_h_frac"])
    hl = float(p["handle_l"])
    fixed = False if p["fixed_base"] is None else bool(p["fixed_base"])
    gattr = (f'friction="{" ".join(str(x) for x in p["friction"])}" '
             f'condim="{int(p["condim"])}" '
             f'solref="0.005 1" solimp="0.99 0.999 0.001"')
    top_t = 0.0015
    vis = 'contype="0" conaffinity="0" group="2" mass="0.00001"'
    tray_c = rgba(p["tray_rgba"])
    # a mount is black, but a black-on-black tray renders as one flat slab:
    # step the floor, walls and top rails apart so the sled reads as an
    # empty tray at bench-camera distance
    _b = p["tray_rgba"]
    floor_c = rgba([min(1.0, c + 0.035) for c in _b[:3]] + [_b[3]])
    wall_c = rgba([min(1.0, c + 0.105) for c in _b[:3]] + [_b[3]])
    rail_c = rgba([min(1.0, c + 0.065) for c in _b[:3]] + [_b[3]])
    boss_c = rgba([0.42, 0.42, 0.44, 1.0])
    caddy = ""
    if mount:
        # an EMPTY sled: floor, two inner walls with screw bosses, the open
        # top framed by a rail each side and a cross bar at the nose end
        fl_t, rail_w, rl_t = 0.0012, 0.008, 0.0012
        wh = H * 0.42
        caddy = f"""
      <geom name="{name}_floor" type="box"
            size="{vec((W / 2, L / 2, fl_t))}" pos="0 0 {fl_t:.5f}"
            rgba="{floor_c}" {vis}/>
      <geom name="{name}_wallL" type="box"
            size="{vec((0.0015, L / 2, wh / 2))}"
            pos="{-(W / 2 - 0.0015):.5f} 0 {wh / 2:.5f}"
            rgba="{wall_c}" {vis}/>
      <geom name="{name}_wallR" type="box"
            size="{vec((0.0015, L / 2, wh / 2))}"
            pos="{W / 2 - 0.0015:.5f} 0 {wh / 2:.5f}"
            rgba="{wall_c}" {vis}/>
      <geom name="{name}_railL" type="box"
            size="{vec((rail_w / 2, L / 2, rl_t))}"
            pos="{-(W / 2 - rail_w / 2):.5f} 0 {H - rl_t:.5f}"
            rgba="{rail_c}" {vis}/>
      <geom name="{name}_railR" type="box"
            size="{vec((rail_w / 2, L / 2, rl_t))}"
            pos="{W / 2 - rail_w / 2:.5f} 0 {H - rl_t:.5f}"
            rgba="{rail_c}" {vis}/>
      <geom name="{name}_nosebar" type="box"
            size="{vec((W / 2, 0.006, rl_t))}"
            pos="0 {-(L / 2 - 0.006):.5f} {H - rl_t:.5f}"
            rgba="{rail_c}" {vis}/>""" + "".join(
            f"""
      <geom name="{name}_boss{i}" type="cylinder"
            size="0.0035 0.0006"
            pos="{sx * W * 0.33:.5f} {sy * L * 0.33:.5f} {2 * fl_t:.5f}"
            rgba="{boss_c}" {vis}/>"""
            for i, (sx, sy) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1))))
    label = ""
    if p["label_rgba"]:
        label = (f'<geom name="{name}_label" type="box" '
                 f'size="{vec((W * 0.30, L * 0.26, 0.0008))}" '
                 f'pos="0 {-L * 0.08:.5f} {H:.5f}" '
                 f'rgba="{rgba(p["label_rgba"])}" mass="0.00001" '
                 f'contype="0" conaffinity="0" group="2"/>')
    body = f"""
    <body name="{name}" pos="{vec((p['pos'][0], p['pos'][1], p['pos'][2]))}"
          euler="0 0 {p['yaw_deg']}">
{'' if fixed else f'      <freejoint name="{name}_free"/>'}
      <geom name="{name}_body" type="box"
            size="{vec((W / 2, L / 2, (H - top_t) / 2))}"
            pos="0 0 {(H - top_t) / 2:.5f}"
            rgba="{'0 0 0 0' if mount else rgba(p['body_rgba'])}"
            mass="{float(p['mass']):.5f}" {gattr}/>
      <geom name="{name}_top" type="box"
            size="{vec((W / 2 * 0.995, L / 2 * 0.995, top_t / 2))}"
            pos="0 0 {H - top_t / 2:.5f}"
            rgba="{'0 0 0 0' if mount else rgba(p['top_rgba'])}"
            mass="0.00001" {gattr}/>
      {label}{caddy}
      <geom name="{name}_trayL" type="box"
            size="{vec((tt / 2, L / 2, th / 2))}"
            pos="{-(W / 2 + tt / 2):.5f} 0 {th / 2:.5f}"
            rgba="{rgba(p['tray_rgba'])}" mass="0.00001" {gattr}/>
      <geom name="{name}_trayR" type="box"
            size="{vec((tt / 2, L / 2, th / 2))}"
            pos="{W / 2 + tt / 2:.5f} 0 {th / 2:.5f}"
            rgba="{rgba(p['tray_rgba'])}" mass="0.00001" {gattr}/>
      <geom name="{name}_handle" type="box"
            size="{vec((W / 2 + tt, hl / 2, H * 0.28))}"
            pos="0 {L / 2 + hl / 2:.5f} {H * 0.28:.5f}"
            rgba="{rgba(p['tray_rgba'])}" mass="0.00001" {gattr}/>
      <site name="{name}_grip" pos="0 {L * 0.30:.5f} {H:.5f}" size="0.002"
            group="4" rgba="0 0 1 0.35"/>
      <site name="{name}_nose" pos="0 {-L / 2:.5f} {H / 2:.5f}" size="0.002"
            group="4" rgba="1 0 0 0.35"/>
    </body>"""
    xml = mjcf_header(f"{name}_{'mount' if mount else 'disk'}",
                      timestep=p["timestep"]) + f"""
  <worldbody>{LIGHTS_XML}{FLOOR_XML if p["include_floor"] else ""}{body}
  </worldbody>
</mujoco>
"""
    xml_path = out_dir / f"{name}.xml"
    xml_path.write_text(xml)
    save_params(out_dir / f"{name}_params.json", p)
    tray_w = W + 2 * tt
    return {
        "xml": str(xml_path),
        "params_json": str(out_dir / f"{name}_params.json"),
        "style": "mount" if mount else "disk",
        "base_body": name, "params": p,
        "body_geom": f"{name}_body",          # what a grasp is solved against
        "grip_site": f"{name}_grip",
        "nose_site": f"{name}_nose",          # enters the cage first
        "outer_lwh": [round(L + hl, 4), round(tray_w, 4), round(H, 4)],
        "drive_lwh": [round(L, 4), round(W, 4), round(H, 4)],
        "insert_dir_local": [0.0, -1.0, 0.0],
        "note": (f"a {W*1000:.0f}x{L*1000:.0f}x{H*1000:.0f} mm drive in a "
                 f"tray {tray_w*1000:.0f} mm wide over the rails. It enters "
                 f"a rack nose-first (local -y); the handle end (+y, "
                 f"<{name}_handle>) stays at the mouth. Lying flat its top "
                 f"is the silver <{name}_top>."),
    }


# ===========================================================================
def slot_world(model, data, built: dict, i: int, prefix: str = ""):
    """(slot_centre, mouth, insert_axis) of bay i in the WORLD, read off the
    compiled model — never assumed from the plan."""
    name = built["base_body"]
    sid = model.site(f"{prefix}{name}_slot{i}").id
    mid = model.site(f"{prefix}{name}_mouth{i}").id
    bid = int(model.site_bodyid[sid])
    R = np.asarray(data.xmat[bid], float).reshape(3, 3)
    return (np.asarray(data.site_xpos[sid], float).copy(),
            np.asarray(data.site_xpos[mid], float).copy(),
            R @ np.asarray(built.get("insert_axis_local", [0, -1, 0]), float))


# ===========================================================================
def preview(built: dict, out_dir: str | Path) -> list[str]:
    import mujoco
    from PIL import Image
    model = load(built["xml"])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    out_dir = Path(out_dir)
    paths = []
    h = built.get("total_h_m", 0.05)
    dist = max(0.45, 2.6 * h)
    with mujoco.Renderer(model, height=420, width=560) as r:
        for tag, az, el in (("front", 90, -10), ("three_quarter", 135, -25),
                            ("top", 90, -75)):
            cam = free_camera([0, 0, h / 2], dist, az, el)
            r.update_scene(data, camera=cam)
            p = out_dir / f"{built['base_body']}_{tag}.png"
            Image.fromarray(r.render().copy()).save(p)
            paths.append(str(p))
    return paths


def demo(built: dict, out_mp4: str | Path, disk_built: dict | None = None,
         fps: int = 30, seconds: float = 4.0) -> dict:
    """rack + a real disk: spawn the disk at the TOP slot's mouth, push it
    in with a plain force (no actuator, no scripted qpos), let go, and
    MEASURE where it ends up — the cheapest end-to-end proof that a disk
    really enters and really rests. For style='disk' alone: the settle drop."""
    import mujoco
    # the attachable asset is built floorless (scene.add_mjcf contract);
    # a drop/insert demo needs ground under it, so re-build a floored twin
    if not built["params"].get("include_floor", True):
        built = build(dict(built["params"], include_floor=True,
                           name=built["params"]["name"] + "_df"),
                      Path(out_mp4).parent)
        if disk_built is not None and \
                not disk_built["params"].get("include_floor", True):
            disk_built = build(dict(disk_built["params"], include_floor=True,
                                    name=disk_built["params"]["name"] + "_df"),
                               Path(out_mp4).parent)
    if built["style"] == "disk" or disk_built is None:
        model = load(built["xml"])
        data = mujoco.MjData(model)
        bid = model.body(built["base_body"]).id
        adr = int(model.jnt_qposadr[model.body_jntadr[bid]])
        data.qpos[adr + 2] += 0.05
        mujoco.mj_forward(model, data)
        frames = rollout(model, data, seconds, fps,
                         camera=free_camera([0, 0, 0.03], 0.5, 105, -18),
                         width=560, height=420)
        out = write_video(frames, out_mp4, fps=fps)
        out["stills"] = save_stills(frames, Path(out_mp4).parent,
                                    Path(out_mp4).stem + "_still", n=3)
        rest = float(data.xpos[bid][2])
        out["measured"] = {"rest_height_m": round(rest, 5)}
        out["verdict"] = f"settled flat at z={rest*1000:.1f} mm"
        return out

    # composite: rack + disk in one throwaway model
    import xml.etree.ElementTree as ET
    rx = ET.parse(built["xml"]).getroot()
    dx = ET.parse(disk_built["xml"]).getroot()
    dbody = next(b for b in dx.find("worldbody").iter("body"))
    rx.find("worldbody").append(dbody)
    tmp = Path(out_mp4).parent / "_rack_demo.xml"
    tmp.write_text(ET.tostring(rx, encoding="unicode"))
    model = mujoco.MjModel.from_xml_string(tmp.read_text())
    data = mujoco.MjData(model)
    i = built["n_slots"] - 1
    mujoco.mj_forward(model, data)
    slot, mouth, axis = slot_world(model, data, built, i)
    dl = float(disk_built["drive_lwh"][0])
    bid = model.body(disk_built["base_body"]).id
    adr = int(model.jnt_qposadr[model.body_jntadr[bid]])
    # start the disk nose 40 mm into the bay, resting near the slot floor —
    # the hand has carried it to the mouth; the demo proves the SLIDE + SEAT
    bd = float(built["params"]["bay_depth"])
    z_floor = slot[2]                     # slot site sits ON the slot floor
    data.qpos[adr:adr + 3] = [slot[0],
                              slot[1] + bd / 2 - 0.04 + dl / 2,
                              z_floor + 0.002]
    mujoco.mj_forward(model, data)
    mg = float(model.body_mass[bid]) * 9.81
    frames = []
    cam = free_camera(slot + np.array([0, 0.05, 0.0]), 0.75, 115, -12)
    steps = int(seconds / model.opt.timestep)
    render_every = max(1, int(1 / (fps * model.opt.timestep)))
    with mujoco.Renderer(model, height=420, width=560) as r:
        for k in range(steps):
            if k < steps * 0.6:           # pushed level, as a hand would
                data.xfrc_applied[bid, :3] = 6.0 * axis + [0, 0, mg * 0.9]
            else:                          # then LET GO
                data.xfrc_applied[bid, :3] = 0.0
            mujoco.mj_step(model, data)
            if k % render_every == 0:
                r.update_scene(data, camera=cam)
                frames.append(r.render().copy())
    data.xfrc_applied[bid] = 0
    end = np.asarray(data.xpos[bid], float)
    depth_in = float(np.dot(end - mouth, axis))
    seated = abs(end[2] - (slot[2] - float(built["params"]["slot_h"]) / 2
                           + float(disk_built["drive_lwh"][2]) / 2)) < 0.01 \
        and depth_in > dl * 0.5
    out = write_video(frames, out_mp4, fps=fps)
    out["stills"] = save_stills(frames, Path(out_mp4).parent,
                                Path(out_mp4).stem + "_still", n=4)
    out["measured"] = {"push_N": 6.0, "depth_in_m": round(depth_in, 4),
                       "end_z_m": round(float(end[2]), 4),
                       "slot_floor_z_m": round(float(slot[2])
                                               - float(built["params"]["slot_h"]) / 2, 4)}
    out["verdict"] = (f"a 6 N push slid the disk {depth_in*1000:.0f} mm into "
                      f"bay {i} and it stayed seated after release"
                      if seated else
                      f"disk did NOT seat (went {depth_in*1000:.0f} mm in, "
                      f"ended at z={end[2]*1000:.1f} mm) — check fits()")
    out["seated"] = bool(seated)
    return out


# ---------------------------------------------------------------------------
