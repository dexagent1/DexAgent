"""Construct an interlocking stud brick (a LEGO / DUPLO / MEGA BLOKS style toy brick).

One free rigid body: a top plate, four skirt walls that leave the underside hollow,
underside tubes (or posts) between the stud rows, and a grid of studs on top. The
underside is sized from the studs it must receive, so a brick lowered onto another
brick of the same pitch seats over its studs and is held laterally by them (walls
outside, tubes inside), which is what makes a stack stand and interlock.

Studs carry a flat-topped cylinder for looks and, for contact, a straight-sided
cylinder (the lateral interlock) under a rounded ellipsoid cap: the cap is the lead-in
that slides a slightly misaligned brick onto the studs instead of leaving it perched.
"""
from __future__ import annotations

TOOL = {
    "name": "construct_stud_brick",
    "title": "Construct an interlocking stud brick",
    "type": "skill",
    "category": "Object Reconstruction",
    "summary": "A toy building brick (LEGO, DUPLO, MEGA BLOKS style) as one free rigid body with an N x M stud grid on top and a hollow underside sized to seat over the studs of a brick below, so bricks stack and interlock.",
    "description": (
        "Builds an MJCF with one free body: a top plate, four skirt walls from the table up to the plate, "
        "underside tubes at the centres between every 2x2 group of studs (posts between the studs of a "
        "one-stud-wide brick), and studs_x x studs_y studs at the given pitch. The outer size is "
        "studs * pitch - edge_gap. The skirt wall thickness and tube radius are DERIVED from the stud radius "
        "and the requested clearance, so the studs of a brick below fit inside the skirt with `clearance` "
        "metres of play at the walls and the tubes. Each stud is a visual cylinder (no contact) plus a "
        "collision cylinder with straight sides (the lateral interlock) topped by an ellipsoid cap covering "
        "lead_in_frac of the stud height: the round cap is a lead-in that centres a brick released up to about "
        "clearance + stud_radius off. The body frame is at the centre of the bottom face (z = 0 on the table), x along "
        "studs_x. Mass comes from `mass` when given (density is rescaled), otherwise from `density` on the "
        "geom volumes. The stud caps take contact priority with a low cap_friction so the lead-in works; "
        "clutch_margin gives the straight stud sides a contact margin larger than the clearance, so a seated "
        "brick is held by a soft press-fit preload (friction clutch) instead of geometry alone; 0 turns it off. "
        "Assumes a rigid plastic brick."
    ),
    "inputs": [
        {"name": "out_dir", "type": "path", "description": "Folder that receives <name>.xml and <name>_params.json."},
        {"name": "name", "type": "str", "description": "Asset and body name prefix. Default brick"},
        {"name": "studs_x", "type": "int", "description": "Studs along the body x axis. Default 2"},
        {"name": "studs_y", "type": "int", "description": "Studs along the body y axis. Default 2"},
        {"name": "pitch", "type": "float", "description": "Stud spacing in metres (LEGO 0.008, DUPLO 0.016, large first-builder bricks about 0.03). Default 0.016"},
        {"name": "body_height", "type": "float", "description": "Height of the brick body without studs, metres. Default 0.0192"},
        {"name": "stud_radius", "type": "float", "description": "Stud radius at the plate, metres. Default 0.0047"},
        {"name": "stud_height", "type": "float", "description": "Stud height above the top face, metres. Default 0.0045"},
        {"name": "top_thickness", "type": "float", "description": "Top plate thickness, metres. Default 0.0018"},
        {"name": "clearance", "type": "float", "description": "Play between a received stud and the skirt wall or tube, metres per side. Default 0.0004"},
        {"name": "lead_in_frac", "type": "float", "description": "Fraction of the stud height that is a rounded (ellipsoid) cap: 0 flat-topped, 1 a dome; the straight lower part holds the brick above laterally, the cap centres it. Default 0.5"},
        {"name": "cap_friction", "type": "float", "description": "Sliding friction of the stud caps (they take priority in their contacts), low so a misaligned brick slides down the lead-in instead of sticking on it. Default 0.15"},
        {"name": "clutch_margin", "type": "float", "description": "Contact margin of the straight stud sides, metres: when larger than clearance, a seated brick's walls and tubes are pressed softly by the studs (a press-fit clutch made of soft contact) so an overhanging brick stays put. 0 = no clutch. Default 0.0005 (0.1 mm beyond the default clearance)"},
        {"name": "edge_gap", "type": "float", "description": "Outer size is studs * pitch - edge_gap, metres. Default 0.0002"},
        {"name": "rgba", "type": "list[4]", "description": "Colour. Default [0.85, 0.2, 0.15, 1]"},
        {"name": "density", "type": "float", "description": "Plastic density kg/m3 (ABS 1050). Default 1050"},
        {"name": "mass", "type": "float|None", "description": "Total mass in kg; overrides density. Default None"},
        {"name": "friction", "type": "list[3]", "description": "MuJoCo friction triple. Default [0.9, 0.01, 0.001]"},
        {"name": "condim", "type": "int", "description": "Contact dimensionality. Default 4"},
        {"name": "solref", "type": "list[2]", "description": "Contact solref. Default [0.01, 1.0]"},
        {"name": "solimp", "type": "list[3]", "description": "Contact solimp. Default [0.95, 0.99, 0.001]"},
        {"name": "pos", "type": "list[3]", "description": "Initial position of the body frame (bottom centre). Default [0, 0, 0]"},
        {"name": "yaw_deg", "type": "float", "description": "Initial yaw about z, degrees. Default 0"},
        {"name": "free", "type": "bool", "description": "Free joint (True) or fixed to the world. Default True"},
        {"name": "include_floor", "type": "bool", "description": "Add a floor plane (for standalone tests; off for staging). Default False"},
    ],
    "outputs": [
        {"name": "built", "type": "dict", "description": (
            "xml, params_json, body (root body name), joint (free joint name or None), geoms, "
            "size_m [x, y, z without studs], total_height_m, pitch, studs [x, y], stud_sites (top stud centres in the body frame), "
            "wall_thickness_m, tube_radius_m, clearance, mass_kg, top_z_m (body_height), "
            "seat_offset_m (z of a brick seated on this one, in this brick's frame)")},
    ],
    "requirements": ["mujoco>=3.1", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["brick", "lego", "duplo", "mega bloks", "stud", "stack", "interlock", "toy", "rigid", "parametric", "object reconstruction"],
    "status": "tested",
    "entrypoint": "construct_stud_brick",
    "cli": "python -m tool_library.skills.construct_stud_brick out_dir",
}

import json
import math
from pathlib import Path

DEFAULTS = {
    "name": "brick",
    "studs_x": 2,
    "studs_y": 2,
    "pitch": 0.016,
    "body_height": 0.0192,
    "stud_radius": 0.0047,
    "stud_height": 0.0045,
    "top_thickness": 0.0018,
    "clearance": 0.0004,
    "lead_in_frac": 0.5,
    "cap_friction": 0.15,
    "clutch_margin": 0.0005,
    "edge_gap": 0.0002,
    "rgba": [0.85, 0.2, 0.15, 1.0],
    "density": 1050.0,
    "mass": None,
    "friction": [0.9, 0.01, 0.001],
    "condim": 4,
    "solref": [0.01, 1.0],
    "solimp": [0.95, 0.99, 0.001],
    "pos": [0.0, 0.0, 0.0],
    "yaw_deg": 0.0,
    "free": True,
    "include_floor": False,
}


def _f(x):
    return " ".join(f"{float(v):.6g}" for v in (x if isinstance(x, (list, tuple)) else [x]))


def _check(p):
    unknown = set(p) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"construct_stud_brick: unknown parameter(s) {sorted(unknown)}; allowed: {sorted(DEFAULTS)}")
    for k in ("studs_x", "studs_y"):
        if int(p[k]) < 1 or int(p[k]) != p[k]:
            raise ValueError(f"{k} must be a positive integer, got {p[k]!r}")
    for k in ("pitch", "body_height", "stud_radius", "stud_height", "top_thickness", "density"):
        if not float(p[k]) > 0:
            raise ValueError(f"{k} must be > 0, got {p[k]!r}")
    if not 0.0 <= float(p["lead_in_frac"]) <= 1.0:
        raise ValueError(f"lead_in_frac must be in [0, 1], got {p['lead_in_frac']!r}")
    if float(p["cap_friction"]) < 0 or float(p["clutch_margin"]) < 0:
        raise ValueError("cap_friction and clutch_margin must be >= 0")
    if float(p["clutch_margin"]) > 0 and float(p["clutch_margin"]) > float(p["clearance"]) + 0.002:
        raise ValueError("clutch_margin more than 2 mm beyond clearance would push bricks apart visibly; lower clutch_margin")
    if float(p["clearance"]) < 0 or float(p["edge_gap"]) < 0:
        raise ValueError("clearance and edge_gap must be >= 0")
    if p["mass"] is not None and not float(p["mass"]) > 0:
        raise ValueError(f"mass must be > 0 or None, got {p['mass']!r}")
    if float(p["stud_radius"]) >= float(p["pitch"]) / 2:
        raise ValueError("stud_radius must be smaller than pitch / 2 (neighbouring studs would overlap); lower stud_radius or raise pitch")
    if float(p["top_thickness"]) >= float(p["body_height"]) * 0.8:
        raise ValueError("top_thickness must be well below body_height; the underside must stay hollow")
    if float(p["stud_height"]) >= float(p["body_height"]) - float(p["top_thickness"]):
        raise ValueError("stud_height must be smaller than the underside cavity height (body_height - top_thickness), "
                         "or a seated brick would rest on its studs")
    if len(p["rgba"]) != 4:
        raise ValueError("rgba needs four values")


def build(params: dict | None = None, out_dir: str | Path = ".") -> dict:
    p = dict(DEFAULTS)
    p.update(params or {})
    _check(p)
    nx, ny = int(p["studs_x"]), int(p["studs_y"])
    P, H = float(p["pitch"]), float(p["body_height"])
    rs, hs, tt = float(p["stud_radius"]), float(p["stud_height"]), float(p["top_thickness"])
    c, gap = float(p["clearance"]), float(p["edge_gap"])
    Lx, Ly = nx * P - gap, ny * P - gap
    # skirt wall: inner face sits clearance outside the outermost stud of a brick below
    wall = Lx / 2 - ((nx - 1) * P / 2 + rs) - c
    wall_y = Ly / 2 - ((ny - 1) * P / 2 + rs) - c
    wall = min(wall, wall_y)
    if wall < 0.0005:
        raise ValueError(f"derived skirt wall is {wall * 1000:.2f} mm thick: lower stud_radius or clearance, or raise pitch")
    cav_h = H - tt
    name = str(p["name"])
    fr, sr, si = _f(p["friction"]), _f(p["solref"]), _f(p["solimp"])
    rgba = _f(p["rgba"])
    common = f'friction="{fr}" condim="{int(p["condim"])}" solref="{sr}" solimp="{si}" rgba="{rgba}"'

    geoms, gnames = [], []

    def add(gname, xml):
        geoms.append(xml)
        gnames.append(gname)

    # top plate
    add(f"{name}_top", f'<geom name="{name}_top" type="box" size="{_f([Lx / 2, Ly / 2, tt / 2])}" pos="0 0 {H - tt / 2:.6g}" {common}/>')
    # skirt walls (x walls span full length, y walls fit between them)
    add(f"{name}_wall_px", f'<geom name="{name}_wall_px" type="box" size="{_f([wall / 2, Ly / 2, cav_h / 2])}" pos="{Lx / 2 - wall / 2:.6g} 0 {cav_h / 2:.6g}" {common}/>')
    add(f"{name}_wall_nx", f'<geom name="{name}_wall_nx" type="box" size="{_f([wall / 2, Ly / 2, cav_h / 2])}" pos="{-(Lx / 2 - wall / 2):.6g} 0 {cav_h / 2:.6g}" {common}/>')
    add(f"{name}_wall_py", f'<geom name="{name}_wall_py" type="box" size="{_f([Lx / 2 - wall, wall / 2, cav_h / 2])}" pos="0 {Ly / 2 - wall / 2:.6g} {cav_h / 2:.6g}" {common}/>')
    add(f"{name}_wall_ny", f'<geom name="{name}_wall_ny" type="box" size="{_f([Lx / 2 - wall, wall / 2, cav_h / 2])}" pos="0 {-(Ly / 2 - wall / 2):.6g} {cav_h / 2:.6g}" {common}/>')

    # underside tubes / posts
    sx = [(i - (nx - 1) / 2) * P for i in range(nx)]
    sy = [(j - (ny - 1) / 2) * P for j in range(ny)]
    tubes, tube_r = [], 0.0
    if nx >= 2 and ny >= 2:
        tube_r = P / math.sqrt(2) - rs - c
        tubes = [((sx[i] + sx[i + 1]) / 2, (sy[j] + sy[j + 1]) / 2) for i in range(nx - 1) for j in range(ny - 1)]
    elif nx * ny > 1:
        tube_r = P / 2 - rs - c
        if nx >= 2:
            tubes = [((sx[i] + sx[i + 1]) / 2, 0.0) for i in range(nx - 1)]
        else:
            tubes = [(0.0, (sy[j] + sy[j + 1]) / 2) for j in range(ny - 1)]
    if tube_r < 0.0005:
        tubes, tube_r = [], 0.0
    for k, (x, y) in enumerate(tubes):
        add(f"{name}_tube{k}", f'<geom name="{name}_tube{k}" type="cylinder" size="{_f([tube_r, cav_h / 2])}" pos="{x:.6g} {y:.6g} {cav_h / 2:.6g}" {common}/>')

    # studs: visual cylinder (no contact) + collision cylinder with straight sides for the lower part and an
    # ellipsoid cap (radius rs at its equator, rising lead_in_frac * hs) as the lead-in on top
    f = float(p["lead_in_frac"])
    side_h = (1.0 - f) * hs
    cm = float(p["clutch_margin"])
    cap_common = (f'friction="{_f([p["cap_friction"], 0.005, 0.0001])}" condim="{int(p["condim"])}" '
                  f'solref="{sr}" solimp="{si}" rgba="{rgba}"')
    stud_sites = []
    k = 0
    for x in sx:
        for y in sy:
            add(f"{name}_stud{k}_vis", f'<geom name="{name}_stud{k}_vis" type="cylinder" size="{_f([rs, hs / 2])}" pos="{x:.6g} {y:.6g} {H + hs / 2:.6g}" contype="0" conaffinity="0" group="1" rgba="{rgba}" mass="0"/>')
            if side_h > 1e-5:
                add(f"{name}_stud{k}", f'<geom name="{name}_stud{k}" type="cylinder" size="{_f([rs, side_h / 2])}" pos="{x:.6g} {y:.6g} {H + side_h / 2:.6g}" group="3" margin="{cm:.6g}" {common}/>')
            if f > 0:
                add(f"{name}_stud{k}_cap", f'<geom name="{name}_stud{k}_cap" type="ellipsoid" size="{_f([rs, rs, f * hs])}" pos="{x:.6g} {y:.6g} {H + side_h:.6g}" group="3" priority="1" {cap_common}/>')
            stud_sites.append([round(x, 6), round(y, 6), round(H + hs, 6)])
            k += 1
    cap_type = "flat" if f == 0 else ("dome" if side_h <= 1e-5 else "rounded")

    # volume -> density so that mass matches when requested
    vol = Lx * Ly * tt + 2 * wall * Ly * cav_h + 2 * (Lx - 2 * wall) * wall * cav_h
    vol += len(tubes) * math.pi * tube_r ** 2 * cav_h
    vol += nx * ny * math.pi * rs * rs * (side_h + 2.0 / 3.0 * f * hs)
    dens = float(p["density"]) if p["mass"] is None else float(p["mass"]) / vol
    mass = dens * vol
    geoms = [g.replace("/>", f' density="{dens:.6g}"/>') if 'mass="0"' not in g else g for g in geoms]

    yaw = math.radians(float(p["yaw_deg"]))
    quat = _f([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    joint = f"{name}_free" if p["free"] else None
    jxml = f'      <freejoint name="{joint}"/>\n' if joint else ""
    body = (f'    <body name="{name}" pos="{_f(p["pos"])}" quat="{quat}">\n' + jxml +
            f'      <site name="{name}_top_site" pos="0 0 {H:.6g}" size="0.002" rgba="0 0 0 0"/>\n' +
            "".join(f"      {g}\n" for g in geoms) + "    </body>\n")
    floor = ('    <geom name="floor" type="plane" size="1 1 0.01" rgba="0.3 0.3 0.35 1" friction="0.8 0.01 0.001"/>\n'
             if p["include_floor"] else "")
    xml = (f'<mujoco model="{name}">\n  <compiler angle="radian"/>\n  <option timestep="0.002"/>\n'
           f"  <worldbody>\n{floor}{body}  </worldbody>\n</mujoco>\n")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    xml_path = out_dir / f"{name}.xml"
    xml_path.write_text(xml)
    built = {
        "xml": str(xml_path), "body": name, "joint": joint, "geoms": gnames,
        "size_m": [round(Lx, 6), round(Ly, 6), round(H, 6)], "total_height_m": round(H + hs, 6),
        "pitch": P, "studs": [nx, ny], "stud_sites": stud_sites, "stud_radius_m": rs, "stud_height_m": hs,
        "wall_thickness_m": round(wall, 6), "tube_radius_m": round(tube_r, 6), "n_tubes": len(tubes),
        "clearance": c, "clutch_margin": cm, "cap_friction": float(p["cap_friction"]), "mass_kg": round(mass, 6), "top_z_m": round(H, 6), "seat_offset_m": round(H, 6),
        "cap": cap_type,
    }
    pj = out_dir / f"{name}_params.json"
    pj.write_text(json.dumps({"params": p, "built": built}, indent=1))
    built["params_json"] = str(pj)
    return built


def construct_stud_brick(out_dir, **params):
    """Build one stud brick into out_dir. Returns the build dictionary."""
    return build(params, out_dir)


if __name__ == "__main__":
    import sys
    print(json.dumps(construct_stud_brick(sys.argv[1] if len(sys.argv) > 1 else "."), indent=1))
