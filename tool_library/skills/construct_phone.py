"""Construct a smartphone: a thin rounded slab with a real charging-port cavity and a camera bump."""
from __future__ import annotations

TOOL = {'name': 'construct_phone',
 'title': 'Construct a smartphone with a charging port',
 'type': 'skill',
 'category': 'Object Reconstruction',
 'summary': 'A smartphone as a free rigid slab with a real charging-port cavity on one short edge, '
            'a camera bump on the back, and declared port frame sites.',
 'description': 'Builds a phone-shaped slab (rounded-rectangle visual mesh) whose collision model is a '
                'set of boxes that leave an open receptacle cavity centred on the bottom short edge, '
                'so a plug of the port size can slide in along the long axis and is stopped by the '
                'cavity floor, while an offset plug is blocked by the edge face. Two rounded pads set '
                'into the upper and lower cavity walls protrude port_grip_m into the opening and squeeze a '
                'seated plug like the latch springs of a real receptacle, so the plug is held by friction; '
                'their rounded front ends are the lead-in. The body origin is the centre '
                'of the face resting on the table; +y points from the port edge to the camera-bump '
                'edge; +z points out of the back (the back faces up when the phone lies back-up). '
                'Sites: <name>_port_mouth and <name>_port_floor (z axis = insertion direction, x across the port\'s thin '
                'side, y across its width: the frame construct_plug_cable\'s <name>_plug_tip has when the plug is '
                'seated; port_floor is the site_b of the connector latch, add_connector_latch), <name>_bump, '
                '<name>_center. Sizes default to a 6.1-inch phone; set length, width and '
                'thickness from the measured footprint or the known model.',
 'inputs': [
  {'name': 'out_dir', 'type': 'path', 'description': 'Folder that receives <name>.xml, the visual mesh and <name>_params.json.'},
  {'name': 'name', 'type': 'str', 'description': 'Asset and body name prefix. Default phone'},
  {'name': 'length', 'type': 'float', 'description': 'Long-edge length in metres. Default 0.1476'},
  {'name': 'width', 'type': 'float', 'description': 'Short-edge width in metres. Default 0.0716'},
  {'name': 'thickness', 'type': 'float', 'description': 'Slab thickness in metres. Default 0.0078'},
  {'name': 'corner_r', 'type': 'float', 'description': 'Plan-view corner radius of the visual slab in metres. Default 0.010'},
  {'name': 'port_w', 'type': 'float', 'description': 'Port opening width (across the edge) in metres. Default 0.0084 (USB-C)'},
  {'name': 'port_h', 'type': 'float', 'description': 'Port opening height (through the thickness) in metres. Default 0.0026 (USB-C)'},
  {'name': 'port_depth', 'type': 'float', 'description': 'Cavity depth along the long axis in metres. Default 0.0068'},
  {'name': 'port_offset_x', 'type': 'float', 'description': 'Lateral offset of the port from the edge centre in metres. Default 0.0'},
  {'name': 'port_grip_m', 'type': 'float', 'description': 'Protrusion of the two retention pads into the opening (retention by friction); 0 = loose fit; the squeeze on a plug is port_grip_m minus the plug clearance. Default 0.00013'},
  {'name': 'bump', 'type': 'bool', 'description': 'Add the camera bump on the back. Default True'},
  {'name': 'bump_side', 'type': 'str', 'description': "'left' (-x) or 'right' (+x) corner of the far (+y) end, seen from the back. Default left"},
  {'name': 'bump_size', 'type': 'float', 'description': 'Camera bump square side in metres. Default 0.032'},
  {'name': 'bump_h', 'type': 'float', 'description': 'Camera bump height above the back in metres. Default 0.0015'},
  {'name': 'bump_inset', 'type': 'float', 'description': 'Bump inset from the two edges in metres. Default 0.006'},
  {'name': 'n_lenses', 'type': 'int', 'description': 'Lenses on the bump (1-3, diagonal). Default 2'},
  {'name': 'screen', 'type': 'bool', 'description': 'Draw the screen glass on the front face (visual only). Default True'},
  {'name': 'screen_rgba', 'type': 'list[4]', 'description': 'Screen colour (a dark, switched-off screen by default). Default [0.04, 0.04, 0.05, 1]'},
  {'name': 'lens_r', 'type': 'float', 'description': 'Lens ring radius in metres. Default 0.0058'},
  {'name': 'body_rgba', 'type': 'list[4]', 'description': 'Back/frame colour. Default [0.85, 0.6, 0.52, 1]'},
  {'name': 'lens_rgba', 'type': 'list[4]', 'description': 'Lens colour. Default [0.08, 0.08, 0.1, 1]'},
  {'name': 'port_rgba', 'type': 'list[4]', 'description': 'Port opening decal colour. Default [0.05, 0.05, 0.05, 1]'},
  {'name': 'mass', 'type': 'float', 'description': 'Total mass in kg. Default 0.171'},
  {'name': 'friction', 'type': 'list[3]', 'description': 'MuJoCo friction triple. Default [0.9, 0.01, 0.001]'},
  {'name': 'port_friction', 'type': 'list[3]', 'description': 'Friction triple of the cavity walls. Default [1.0, 0.01, 0.001]'},
  {'name': 'free', 'type': 'bool', 'description': 'Free body (True) or fixed to its parent. Default True'},
  {'name': 'pos', 'type': 'list[3]', 'description': 'Body position in the asset frame. Default [0, 0, 0]'},
  {'name': 'yaw_deg', 'type': 'float', 'description': 'Yaw about z in degrees. Default 0'},
  {'name': 'include_floor', 'type': 'bool', 'description': 'Add a floor plane (for standalone tests). Default False'},
  {'name': 'timestep', 'type': 'float', 'description': 'Simulation timestep written into the asset. Default 0.001'}],
 'outputs': [{'name': 'built', 'type': 'dict',
              'description': 'xml: the MJCF path; params_json; body, joint, geoms; sites (port_mouth, port_floor: the latch\'s port site, bump, center); '
                             'port: {width, height, depth, mouth_local, axis_local, grip_m, cavity_geoms}; dims and mass_kg.'}],
 'requirements': ['mujoco>=3.2', 'numpy'],
 'simulators': ['MuJoCo'],
 'tags': ['phone', 'smartphone', 'charging port', 'usb-c', 'receptacle', 'slab', 'rigid', 'parametric', 'object reconstruction'],
 'status': 'tested',
 'entrypoint': 'construct_phone',
 'cli': 'python -m tool_library.skills.construct_phone out_dir'}

import json
import math
from pathlib import Path

DEFAULTS = {
    "name": "phone", "length": 0.1476, "width": 0.0716, "thickness": 0.0078, "corner_r": 0.010,
    "port_w": 0.0084, "port_h": 0.0026, "port_depth": 0.0068, "port_offset_x": 0.0, "port_grip_m": 0.00013,
    "bump": True, "bump_side": "left", "bump_size": 0.032, "bump_h": 0.0015, "bump_inset": 0.006,
    "n_lenses": 2, "lens_r": 0.0058, "screen": True, "screen_rgba": [0.04, 0.04, 0.05, 1.0],
    "body_rgba": [0.85, 0.6, 0.52, 1.0], "lens_rgba": [0.08, 0.08, 0.1, 1.0], "port_rgba": [0.05, 0.05, 0.05, 1.0],
    "mass": 0.171, "friction": [0.9, 0.01, 0.001], "port_friction": [1.0, 0.01, 0.001],
    "free": True, "pos": [0.0, 0.0, 0.0], "yaw_deg": 0.0, "include_floor": False, "timestep": 0.001,
}


def _rounded_slab_obj(path, L, W, T, r, n_arc=8):
    r = max(1e-4, min(r, 0.5 * W - 1e-4, 0.5 * L - 1e-4))
    pts = []
    for cx, cy, a0 in ((W / 2 - r, L / 2 - r, 0), (-W / 2 + r, L / 2 - r, 90),
                       (-W / 2 + r, -L / 2 + r, 180), (W / 2 - r, -L / 2 + r, 270)):
        for k in range(n_arc + 1):
            a = math.radians(a0 + 90.0 * k / n_arc)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    n = len(pts)
    v = [(x, y, 0.0) for x, y in pts] + [(x, y, T) for x, y in pts] + [(0, 0, 0.0), (0, 0, T)]
    cb, ct = 2 * n + 1, 2 * n + 2
    f = []
    for i in range(n):
        j = (i + 1) % n
        f.append((cb, j + 1, i + 1))
        f.append((ct, n + i + 1, n + j + 1))
        f.append((i + 1, j + 1, n + j + 1))
        f.append((i + 1, n + j + 1, n + i + 1))
    with open(path, "w") as fh:
        for p in v:
            fh.write("v %.6f %.6f %.6f\n" % p)
        for t in f:
            fh.write("f %d %d %d\n" % t)


def _s(v):
    return " ".join("%.6g" % float(x) for x in v)


def build(params=None, out_dir="."):
    params = dict(params or {})
    unknown = sorted(set(params) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"construct_phone: unknown parameter(s) {unknown}; allowed: {sorted(DEFAULTS)}")
    p = {**DEFAULTS, **params}
    L, W, T = float(p["length"]), float(p["width"]), float(p["thickness"])
    pw, ph, pd = float(p["port_w"]), float(p["port_h"]), float(p["port_depth"])
    ox = float(p["port_offset_x"])
    for k in ("length", "width", "thickness", "port_w", "port_h", "port_depth", "mass"):
        if float(p[k]) <= 0:
            raise ValueError(f"construct_phone: {k} must be > 0")
    if W > L:
        raise ValueError("construct_phone: width must not exceed length (the port is on a short edge)")
    if ph >= T - 0.001:
        raise ValueError("construct_phone: port_h must leave at least 0.5 mm of wall above and below (reduce port_h or raise thickness)")
    if abs(ox) + pw / 2 >= W / 2 - 0.002:
        raise ValueError("construct_phone: port_w/port_offset_x put the port outside the edge")
    if pd >= 0.5 * L:
        raise ValueError("construct_phone: port_depth must be under half the length")
    if p["bump_side"] not in ("left", "right"):
        raise ValueError("construct_phone: bump_side must be 'left' or 'right'")
    if not 0 <= int(p["n_lenses"]) <= 3:
        raise ValueError("construct_phone: n_lenses must be 0..3")
    name = str(p["name"])
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    mesh = out / f"{name}_slab.obj"
    _rounded_slab_obj(mesh, L, W, T, float(p["corner_r"]))

    fr, pfr = _s(p["friction"]), _s(p["port_friction"])
    grip = float(p["port_grip_m"])
    y0 = -L / 2                      # port edge
    zc = T / 2
    # collision boxes: (name, centre, half sizes, is_cavity_wall)
    boxes = []
    boxes.append(("slab", (0, y0 + pd + (L - pd) / 2, zc), (W / 2, (L - pd) / 2, T / 2), False))
    xl0, xl1 = -W / 2, ox - pw / 2
    xr0, xr1 = ox + pw / 2, W / 2
    boxes.append(("port_left", ((xl0 + xl1) / 2, y0 + pd / 2, zc), ((xl1 - xl0) / 2, pd / 2, T / 2), True))
    boxes.append(("port_right", ((xr0 + xr1) / 2, y0 + pd / 2, zc), ((xr1 - xr0) / 2, pd / 2, T / 2), True))
    lay = (T - ph) / 2
    boxes.append(("port_below", (ox, y0 + pd / 2, lay / 2), (pw / 2, pd / 2, lay / 2), True))
    boxes.append(("port_above", (ox, y0 + pd / 2, T - lay / 2), (pw / 2, pd / 2, lay / 2), True))

    vol = sum(8 * h[0] * h[1] * h[2] for _, _, h, _ in boxes)
    if p["bump"]:
        vol += float(p["bump_size"]) ** 2 * float(p["bump_h"])
    dens = float(p["mass"]) / vol
    geoms, cav = [], []
    for gname, c, h, wall in boxes:
        g = f"{name}_{gname}"
        extra = f' friction="{pfr}"' if wall else f' friction="{fr}"'
        geoms.append(f'      <geom name="{g}" type="box" pos="{_s(c)}" size="{_s(h)}" density="{dens:.6g}" '
                     f'group="3" rgba="0.5 0.5 0.5 0.3" condim="4"{extra}/>')
        if wall:
            cav.append(g)
    if grip > 0:
        # retention: two rounded pads (capsules along the insertion axis) set into the upper and
        # lower cavity walls, protruding port_grip_m into the opening; their rounded front ends are
        # the lead-in, so a plug rides up onto them instead of stopping at a face.
        rp = min(0.0008, 0.45 * lay)
        ya, yb = y0 + rp + 0.0008, y0 + pd - rp
        if yb <= ya:
            raise ValueError("construct_phone: port_depth too short for the retention pads; raise port_depth or set port_grip_m=0")
        for sgn, tag in ((1, "grip_above"), (-1, "grip_below")):
            zz = zc + sgn * (ph / 2 + rp - grip)
            g = f"{name}_{tag}"
            geoms.append(f'      <geom name="{g}" type="capsule" fromto="{_s((ox, ya, zz, ox, yb, zz))}" size="{rp:.6g}" '
                         f'density="0" group="3" rgba="0.2 0.2 0.2 0.5" condim="4" friction="{pfr}"/>')
            cav.append(g)
    body_rgba = _s(p["body_rgba"])
    # visual: rounded slab, port decal, camera bump and lenses
    geoms.append(f'      <geom name="{name}_visual" type="mesh" mesh="{name}_slab" contype="0" conaffinity="0" '
                 f'density="0" group="1" rgba="{body_rgba}"/>')
    geoms.append(f'      <geom name="{name}_port_decal" type="box" pos="{_s((ox, y0 - 0.0001, zc))}" '
                 f'size="{_s((pw / 2, 0.00015, ph / 2))}" contype="0" conaffinity="0" density="0" group="1" '
                 f'rgba="{_s(p["port_rgba"])}"/>')
    if p["screen"]:
        # the glass on the front face (z = 0, the face the camera bump is not on), inset so its corners stay inside the
        # rounded outline; visual only
        r_c = float(p["corner_r"]); bez = max(0.0015, r_c * (1 - 1 / math.sqrt(2)) + 0.0005)
        geoms.append(f'      <geom name="{name}_screen" type="box" pos="{_s((0.0, 0.0, -0.00005))}" '
                     f'size="{_s((W / 2 - bez, L / 2 - bez, 0.00005))}" contype="0" conaffinity="0" density="0" group="1" '
                     f'rgba="{_s(p["screen_rgba"])}"/>')
    sites = {}
    bump_c = None
    if p["bump"]:
        bs, bh, bi = float(p["bump_size"]), float(p["bump_h"]), float(p["bump_inset"])
        sx = -1.0 if p["bump_side"] == "left" else 1.0
        bx = sx * (W / 2 - bi - bs / 2)
        by = L / 2 - bi - bs / 2
        bump_c = (bx, by, T + bh / 2)
        geoms.append(f'      <geom name="{name}_bump" type="box" pos="{_s(bump_c)}" size="{_s((bs / 2, bs / 2, bh / 2))}" '
                     f'friction="{fr}" condim="4" density="{dens:.6g}" rgba="{body_rgba}"/>')
        nl, lr = int(p["n_lenses"]), float(p["lens_r"])
        offs = {1: [(0, 0)], 2: [(-0.22, 0.22), (0.22, -0.22)], 3: [(-0.22, 0.22), (0.22, -0.22), (0.22, 0.22)]}.get(nl, [])
        for k, (u, v) in enumerate(offs):
            geoms.append(f'      <geom name="{name}_lens{k}" type="cylinder" pos="{_s((bx + sx * u * bs, by + v * bs, T + bh + 0.0004))}" '
                         f'size="{lr:.6g} 0.0004" contype="0" conaffinity="0" density="0" group="1" rgba="{_s(p["lens_rgba"])}"/>')
        sites["bump"] = f"{name}_bump_site"
    mouth = (ox, y0, zc)
    floor = (ox, y0 + pd, zc)
    # port frames: z along the insertion direction (+y), x across the thin side (+z), y across the width (+x): the frame of
    # construct_plug_cable's plug_tip site when the plug is seated, so a latch's seated relpose is the identity between them
    site_xml = [f'      <site name="{name}_port_mouth" pos="{_s(mouth)}" xyaxes="0 0 1 1 0 0" size="0.0015" rgba="0 1 0 0.6" group="4"/>',
                f'      <site name="{name}_port_floor" pos="{_s(floor)}" xyaxes="0 0 1 1 0 0" size="0.0015" rgba="1 0 0 0.6" group="4"/>',
                f'      <site name="{name}_center" pos="0 0 {zc:.6g}" size="0.002" group="4"/>']
    if bump_c is not None:
        site_xml.append(f'      <site name="{name}_bump_site" pos="{_s(bump_c)}" size="0.002" group="4"/>')
    sites.update({"port_mouth": f"{name}_port_mouth", "port_floor": f"{name}_port_floor", "center": f"{name}_center"})

    yaw = float(p["yaw_deg"])
    joint = f"{name}_free"
    jxml = f'      <freejoint name="{joint}"/>\n' if p["free"] else ""
    floor_xml = ('    <geom name="floor" type="plane" size="1 1 0.05" rgba="0.8 0.8 0.8 1"/>\n'
                 '    <light pos="0 0 1" dir="0 0 -1"/>\n') if p["include_floor"] else ""
    xml = (f'<mujoco model="{name}">\n'
           f'  <compiler angle="degree"/>\n'
           f'  <option timestep="{float(p["timestep"]):.6g}"/>\n'
           f'  <asset>\n    <mesh name="{name}_slab" file="{mesh.name}"/>\n  </asset>\n'
           f'  <worldbody>\n{floor_xml}'
           f'    <body name="{name}" pos="{_s(p["pos"])}" euler="0 0 {yaw:.6g}">\n{jxml}'
           + "\n".join(geoms) + "\n" + "\n".join(site_xml) + "\n"
           '    </body>\n  </worldbody>\n</mujoco>\n')
    xml_path = out / f"{name}.xml"
    xml_path.write_text(xml)
    pj = out / f"{name}_params.json"
    pj.write_text(json.dumps(p, indent=1))
    return {"xml": str(xml_path), "params_json": str(pj), "body": name, "joint": joint if p["free"] else None,
            "geoms": [f"{name}_{b[0]}" for b in boxes], "sites": sites,
            "port": {"width": pw, "height": ph, "depth": pd, "mouth_local": list(mouth), "axis_local": [0.0, 1.0, 0.0],
                     "grip_m": grip, "cavity_geoms": cav},
            "dims": {"length": L, "width": W, "thickness": T}, "mass_kg": float(p["mass"])}


def construct_phone(out_dir, **params):
    """Build the phone into out_dir. Returns the build dictionary."""
    return build(params, out_dir)


if __name__ == "__main__":
    import sys
    print(json.dumps(construct_phone(sys.argv[1] if len(sys.argv) > 1 else "."), indent=1))
