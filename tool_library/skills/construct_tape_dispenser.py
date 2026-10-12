"""Construct a desk tape dispenser: a weighted base, a tape roll between two cheeks and a sloped cutter arm that
overhangs the front of the base, leaving a clamp gap under its tip where a cord or strip can be held down.
"""
from __future__ import annotations

TOOL = {
    "name": "construct_tape_dispenser",
    "title": "Construct a desk tape dispenser",
    "type": "skill",
    "category": "Object Reconstruction",
    "summary": "A desk tape dispenser as a static (or free) rigid body: weighted base, tape roll between two cheeks and a sloped cutter arm overhanging the front with a clamp gap under its tip, plus a declared clamp site where a cord end held under the front can be anchored.",
    "description": (
        "Builds one MJCF body whose origin is the bottom centre of the footprint, +x pointing to the cutter (front), z "
        "up. The base is a box over the rear (1 - nose_frac) of the length and base_height high. Two cheek plates rise "
        "from the base at the rear to the full height and hold a tape roll (a cylinder along y, radius roll_radius, "
        "width roll_width; its cardboard core is a visual-only disk). A sloped cutter arm runs from the cheeks' front "
        "down to the front end of the footprint, where its underside stays clamp_gap above the table: the space "
        "under the overhang is free of geometry, so a cord, ribbon or tape end can lie there. The site "
        "'<name>_clamp' marks that spot on the table under the arm (the point a cord end held under the front is "
        "anchored to). Every touchable part (base, cheeks, roll, arm) is a colliding box or cylinder whose surface "
        "is the visual one; the mass is spread over them by volume. fixed=True builds a jointless body (scenery: the "
        "dispenser stays where it is staged); fixed=False adds a free joint. Unknown parameters and a roll that does "
        "not fit between the base and the top (or a nose so long that the cheeks overhang the base) are refused with "
        "a message naming the parameter."
    ),
    "inputs": [
        {"name": "out_dir", "type": "path", "description": "Folder that receives <name>.xml and <name>_params.json."},
        {"name": "name", "type": "str", "description": "Asset and body name. Default tape_dispenser"},
        {"name": "length", "type": "float", "description": "Footprint length along the cutter direction (body x), metres. Default 0.15"},
        {"name": "width", "type": "float", "description": "Footprint width (body y), metres. Default 0.06"},
        {"name": "height", "type": "float", "description": "Overall height (top of the cheeks), metres. Default 0.075"},
        {"name": "base_height", "type": "float", "description": "Height of the weighted base slab, metres. Default 0.02"},
        {"name": "nose_frac", "type": "float", "description": "Front fraction of the length with no base under the cutter arm (the overhang), 0.05-0.6. Default 0.3"},
        {"name": "clamp_gap", "type": "float", "description": "Clearance between the table and the underside of the arm's tip, metres. Default 0.012"},
        {"name": "arm_thickness", "type": "float", "description": "Thickness of the cutter arm, metres. Default 0.006"},
        {"name": "roll_radius", "type": "float | None", "description": "Outer radius of the tape roll, metres; None fits the largest roll between the base and the top. Default None"},
        {"name": "roll_width", "type": "float | None", "description": "Width of the tape roll, metres; None takes 0.6 of the width. Default None"},
        {"name": "core_frac", "type": "float", "description": "Core radius as a fraction of roll_radius (visual). Default 0.6"},
        {"name": "body_rgba", "type": "list[4]", "description": "Base, cheeks and arm colour. Default [0.06, 0.06, 0.06, 1]"},
        {"name": "roll_rgba", "type": "list[4]", "description": "Tape roll colour. Default [0.9, 0.88, 0.82, 1]"},
        {"name": "core_rgba", "type": "list[4]", "description": "Cardboard core colour. Default [0.72, 0.58, 0.4, 1]"},
        {"name": "cutter_rgba", "type": "list[4]", "description": "Cutter blade strip colour (visual). Default [0.55, 0.55, 0.55, 1]"},
        {"name": "mass", "type": "float", "description": "Total mass, kg (weighted dispensers are heavy). Default 0.6"},
        {"name": "friction", "type": "list[3]", "description": "Friction triple of every colliding part. Default [1.0, 0.02, 0.001]"},
        {"name": "solref", "type": "list[2]", "description": "Contact solref. Default [0.004, 1.0]"},
        {"name": "fixed", "type": "bool", "description": "True: no joint (scenery fixed where staged); False: a free joint. Default True"},
        {"name": "pos", "type": "list[3]", "description": "Body position in the asset frame. Default [0, 0, 0]"},
        {"name": "yaw_deg", "type": "float", "description": "Body yaw about z, degrees. Default 0"},
        {"name": "include_floor", "type": "bool", "description": "Add a floor plane (stand-alone tests). Default False"},
        {"name": "timestep", "type": "float | None", "description": "Simulation timestep written into the asset, or None. Default None"},
    ],
    "outputs": [
        {"name": "built", "type": "dict", "description": "xml, params_json, body, joint (None when fixed), collision_geoms, visual_geoms, sites {clamp}, clamp_local [x, y, z] (body frame), length_m, width_m, height_m, roll_radius_m, nose {length_m, gap_m, arm_tip_local}, mass_kg, fixed."},
    ],
    "requirements": ["numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["tape dispenser", "dispenser", "anchor", "clamp", "static", "desk", "block", "rigid", "parametric", "object reconstruction"],
    "used_in": ["EgoDex thread_unthread_bead_necklace"],
    "status": "tested",
    "entrypoint": "construct_tape_dispenser",
    "cli": "python -m tool_library.skills.construct_tape_dispenser out_dir",
}

import json
import math
from pathlib import Path

DEFAULTS = {
    "name": "tape_dispenser",
    "length": 0.15,
    "width": 0.06,
    "height": 0.075,
    "base_height": 0.02,
    "nose_frac": 0.3,
    "clamp_gap": 0.012,
    "arm_thickness": 0.006,
    "roll_radius": None,
    "roll_width": None,
    "core_frac": 0.6,
    "body_rgba": [0.06, 0.06, 0.06, 1.0],
    "roll_rgba": [0.9, 0.88, 0.82, 1.0],
    "core_rgba": [0.72, 0.58, 0.4, 1.0],
    "cutter_rgba": [0.55, 0.55, 0.55, 1.0],
    "mass": 0.6,
    "friction": [1.0, 0.02, 0.001],
    "solref": [0.004, 1.0],
    "fixed": True,
    "pos": [0.0, 0.0, 0.0],
    "yaw_deg": 0.0,
    "include_floor": False,
    "timestep": None,
}

_WALL = 0.004      # clearance between the roll and the rear end / the top of the cheeks


def _fmt(v):
    return " ".join(f"{float(x):.6g}" for x in v)


def _resolve(p):
    unknown = sorted(set(p) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"construct_tape_dispenser: unknown parameter(s) {unknown}; allowed: {sorted(DEFAULTS)}")
    L, W, H, hb = (float(p[k]) for k in ("length", "width", "height", "base_height"))
    if min(L, W, H, hb) <= 0:
        raise ValueError("construct_tape_dispenser: length, width, height and base_height must be > 0")
    if hb >= H - 0.01:
        raise ValueError("construct_tape_dispenser: base_height must be at least 1 cm below height")
    nf = float(p["nose_frac"])
    if not 0.05 <= nf <= 0.6:
        raise ValueError("construct_tape_dispenser: nose_frac must lie in [0.05, 0.6]")
    gap, at = float(p["clamp_gap"]), float(p["arm_thickness"])
    if gap < 0 or at <= 0 or gap + at >= H - hb:
        raise ValueError("construct_tape_dispenser: need clamp_gap >= 0, arm_thickness > 0 and clamp_gap + arm_thickness "
                         "below height - base_height")
    r_max = (H - hb - _WALL) / 2.0
    r = r_max if p["roll_radius"] is None else float(p["roll_radius"])
    if not 0 < r <= r_max + 1e-9:
        raise ValueError(f"construct_tape_dispenser: roll_radius must be in (0, {r_max:.4f}] for this height and "
                         f"base_height (the roll sits on the base under the top)")
    rw = 0.6 * W if p["roll_width"] is None else float(p["roll_width"])
    if not 0 < rw < W - 0.002:
        raise ValueError("construct_tape_dispenser: roll_width must be > 0 and leave at least 1 mm per cheek inside width")
    tower = 2 * r + 2 * _WALL
    base_len = (1.0 - nf) * L
    if tower > base_len + 1e-9:
        raise ValueError(f"construct_tape_dispenser: the roll housing ({tower:.4f} m) is longer than the base "
                         f"({base_len:.4f} m): lower nose_frac or roll_radius, or raise length")
    cf = float(p["core_frac"])
    if not 0 < cf < 1:
        raise ValueError("construct_tape_dispenser: core_frac must lie in (0, 1)")
    if float(p["mass"]) <= 0:
        raise ValueError("construct_tape_dispenser: mass must be > 0")
    return L, W, H, hb, nf, gap, at, r, rw, tower, base_len, cf


def build(params: dict | None = None, out_dir: str | Path = ".") -> dict:
    p = dict(DEFAULTS)
    p.update(params or {})
    L, W, H, hb, nf, gap, at, r, rw, tower, base_len, cf = _resolve(p)
    name = str(p["name"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x0 = -L / 2.0
    cheek_t = (W - rw) / 2.0 - 0.0005
    parts = []   # (geom name, type, size, pos, quat, rgba, volume)
    # base slab
    parts.append((f"{name}_base", "box", [base_len / 2, W / 2, hb / 2], [x0 + base_len / 2, 0, hb / 2], [1, 0, 0, 0],
                  p["body_rgba"], base_len * W * hb))
    # cheeks
    ch_h = H - hb
    for s, tag in ((1, "left"), (-1, "right")):
        parts.append((f"{name}_cheek_{tag}", "box", [tower / 2, cheek_t / 2, ch_h / 2],
                      [x0 + tower / 2, s * (W / 2 - cheek_t / 2), hb + ch_h / 2], [1, 0, 0, 0], p["body_rgba"],
                      tower * cheek_t * ch_h))
    # tape roll (cylinder along y)
    xr, zr = x0 + _WALL + r, H - _WALL / 2 - r
    qy = [math.sqrt(0.5), math.sqrt(0.5), 0, 0]          # cylinder z -> y
    parts.append((f"{name}_roll", "cylinder", [r, rw / 2], [xr, 0, zr], qy, p["roll_rgba"], math.pi * r * r * rw))
    # cutter arm: from the cheeks' front at 60 % of the cheek height down to the front end, underside at clamp_gap
    xs, zs = x0 + tower, hb + 0.6 * ch_h
    xe, ze = L / 2.0, gap + at / 2.0
    dx, dz = xe - xs, ze - zs
    arm_len = math.hypot(dx, dz)
    pitch = math.atan2(-dz, dx)                           # rotation about y that tilts +x down toward the front
    qa = [math.cos(pitch / 2), 0, math.sin(pitch / 2), 0]
    parts.append((f"{name}_arm", "box", [arm_len / 2, W / 2, at / 2], [(xs + xe) / 2, 0, (zs + ze) / 2], qa,
                  p["body_rgba"], arm_len * W * at))
    vol = sum(v for *_, v in parts)
    rho = float(p["mass"]) / vol
    fr, sr = _fmt(p["friction"]), _fmt(p["solref"])
    lines, coll = [], []
    for gname, gtype, size, pos, quat, rgba, v in parts:
        coll.append(gname)
        lines.append(f'      <geom name="{gname}" type="{gtype}" size="{_fmt(size)}" pos="{_fmt(pos)}" quat="{_fmt(quat)}" '
                     f'rgba="{_fmt(rgba)}" mass="{rho * v:.6g}" friction="{fr}" solref="{sr}" condim="4" '
                     f'contype="1" conaffinity="1"/>')
    vis = []
    core = f"{name}_core_vis"
    vis.append(core)
    lines.append(f'      <geom name="{core}" type="cylinder" size="{cf * r:.6g} {rw / 2 + 0.0004:.6g}" pos="{_fmt([xr, 0, zr])}" '
                 f'quat="{_fmt(qy)}" rgba="{_fmt(p["core_rgba"])}" contype="0" conaffinity="0" mass="0" group="1"/>')
    cut = f"{name}_cutter_vis"
    vis.append(cut)
    lines.append(f'      <geom name="{cut}" type="box" size="{0.003:.6g} {W / 2 + 0.0005:.6g} {at / 2 + 0.0005:.6g}" '
                 f'pos="{_fmt([xe - 0.003 * math.cos(pitch), 0, ze + 0.003 * math.sin(pitch)])}" quat="{_fmt(qa)}" '
                 f'rgba="{_fmt(p["cutter_rgba"])}" contype="0" conaffinity="0" mass="0" group="1"/>')
    clamp = [L / 2.0 - 0.25 * nf * L, 0.0, 0.001]
    site = f"{name}_clamp"
    lines.append(f'      <site name="{site}" pos="{_fmt(clamp)}" size="0.003" rgba="0.9 0.2 0.2 0.5" group="3"/>')
    yaw = math.radians(float(p["yaw_deg"]))
    bq = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    joint = None if p["fixed"] else f"{name}_free"
    jline = "" if p["fixed"] else f'      <freejoint name="{joint}"/>\n'
    floor = ('    <geom name="floor" type="plane" size="0.5 0.5 0.01" rgba="0.8 0.8 0.8 1" contype="1" '
             'conaffinity="1"/>\n') if p["include_floor"] else ""
    opt = f'  <option timestep="{float(p["timestep"]):.6g}"/>\n' if p["timestep"] else ""
    xml = (f'<mujoco model="{name}">\n  <compiler angle="radian"/>\n{opt}  <worldbody>\n{floor}'
           f'    <body name="{name}" pos="{_fmt(p["pos"])}" quat="{_fmt(bq)}">\n{jline}'
           + "\n".join(lines) + "\n    </body>\n  </worldbody>\n</mujoco>\n")
    xml_path = out_dir / f"{name}.xml"
    xml_path.write_text(xml)
    pj = out_dir / f"{name}_params.json"
    pj.write_text(json.dumps(p, indent=1))
    return {"xml": str(xml_path), "params_json": str(pj), "body": name, "joint": joint, "collision_geoms": coll,
            "visual_geoms": vis, "sites": {"clamp": site}, "clamp_local": clamp, "length_m": L, "width_m": W,
            "height_m": H, "roll_radius_m": r, "nose": {"length_m": nf * L, "gap_m": gap, "arm_tip_local": [xe, 0.0, gap]},
            "mass_kg": float(p["mass"]), "fixed": bool(p["fixed"])}


def construct_tape_dispenser(out_dir, **params):
    """Build one tape dispenser into out_dir. Returns the build dictionary."""
    return build(params, out_dir)


if __name__ == "__main__":
    import sys as _sys
    print(json.dumps(construct_tape_dispenser(_sys.argv[1] if len(_sys.argv) > 1 else "."), indent=1, default=str))
