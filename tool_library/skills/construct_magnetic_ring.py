"""Construct a magnetic ring: a hollow rigid ring (short tube) that clings rim to rim to other magnetic rings.

The ring wall is a circle of box segments (hollow bore, flat faces for a pinch) on a free body. The magnet is a
jointless child body holding one sphere (or cylinder) of the ring's outer radius that collides only with other magnet
shells (its own contype/conaffinity bit), plus a body-transmission actuator with zero gain and a constant bias:
MuJoCo's adhesion mechanism with a fixed force, so the attraction needs no control signal and acts only on
magnet-to-magnet contacts (table, fingers and other objects see the ring wall, never the magnet).
"""
from __future__ import annotations

TOOL = {
    "name": "construct_magnetic_ring",
    "title": "Construct a magnetic ring",
    "type": "skill",
    "category": "Object Reconstruction",
    "summary": "A hollow rigid ring (short tube) as a free body whose outer surface magnetically clings rim to rim to other rings built by this skill, so rings can hang from, roll around and be pulled off one another.",
    "description": (
        "Builds one MJCF holding a single ring: n_segments box geoms arranged around the axis between inner_radius "
        "and outer_radius over the ring's height (hollow bore, the mass split evenly over the wall), on a free body "
        "whose origin is the point that rests on the table (bottom centre when the axis is vertical, the lowest rim "
        "point when the ring stands on its rim with the axis horizontal). Every accent_every-th segment takes "
        "accent_rgba so the pattern reads in renders. The magnet is a jointless child body '<name>_magnet' carrying "
        "a sphere (default; or a cylinder over the wall height) of radius outer_radius + magnet_pad_m in its own collision bit (magnet_bit): magnet shells collide "
        "only with each other, with margin = gap = magnet_range_m so a contact appears just before touching without "
        "pushing. A <general> actuator with body transmission on that child, gainprm 0 and biasprm [magnet_force_N] "
        "applies MuJoCo adhesion with a constant force: the ring is pulled toward every magnet shell it touches, the "
        "force split over its magnet contacts. Two touching rings therefore attract with about twice magnet_force_N, "
        "a ring hangs from another, rolls around its rim, and separates under a larger pull. A sphere magnet keeps the centres at a fixed distance whatever the tilt, so a ring wedged between two others cannot be levered out of plane by edge contacts (a cylinder magnet can). Assumes all rings of a "
        "scene are built by this skill with the same magnet_bit; the actuator needs no control (ctrl is ignored). "
        "magnet_force_N 0 builds the ring without any magnet parts (no child body, shell or actuator)."
    ),
    "inputs": [
        {"name": "out_dir", "type": "path", "description": "Folder that receives <name>.xml and <name>_params.json."},
        {"name": "name", "type": "str", "description": "Asset and body name. Default magnetic_ring"},
        {"name": "outer_radius", "type": "float", "description": "Outer radius of the ring wall in metres. Default 0.0155"},
        {"name": "inner_radius", "type": "float", "description": "Bore radius in metres (< outer_radius). Default 0.0105"},
        {"name": "height", "type": "float", "description": "Ring height along its axis in metres. Default 0.012"},
        {"name": "n_segments", "type": "int", "description": "Box segments around the wall (>= 8). Default 24"},
        {"name": "body_rgba", "type": "list[4]", "description": "Main wall colour. Default [0.16, 0.13, 0.11, 1]"},
        {"name": "accent_rgba", "type": "list[4]", "description": "Accent (pattern) colour. Default [0.78, 0.66, 0.24, 1]"},
        {"name": "accent_every", "type": "int", "description": "Every k-th segment in the accent colour, 0 for none. Default 3"},
        {"name": "mass", "type": "float", "description": "Ring mass in kg. Default 0.012"},
        {"name": "friction", "type": "list[3]", "description": "Wall friction triple. Default [1.0, 0.01, 0.001]"},
        {"name": "solref", "type": "list[2]", "description": "Wall contact solref. Default [0.005, 1.0]"},
        {"name": "solimp", "type": "list[3]", "description": "Wall contact solimp. Default [0.95, 0.99, 0.0005]"},
        {"name": "magnet_force_N", "type": "float", "description": "Constant adhesion force of this ring's magnet in newtons (>= 0). 0 builds a plain ring (a bead, a washer): no magnet body, shell or actuator, and magnet_body / magnet_geom / magnet_actuator come back None. Default 2.0"},
        {"name": "magnet_range_m", "type": "float", "description": "Distance below which magnet shells see each other (margin = gap; MuJoCo keeps touching shells within about twice this). Default 0.0003"},
        {"name": "magnet_pad_m", "type": "float", "description": "Magnet shell radius beyond the outer radius, so shells touch before walls. Default 0.0003"},
        {"name": "magnet_friction", "type": "list[3]", "description": "Friction triple of magnet-to-magnet contact (rolling vs sliding). Default [0.8, 0.01, 0.001]"},
        {"name": "magnet_bit", "type": "int", "description": "Collision bit (power of two, not 1) shared by magnet shells only. Default 4"},
        {"name": "magnet_shape", "type": "str", "description": "'sphere' (radius outer_radius + magnet_pad_m: rings attract centre to centre and stay coaxially aligned, no edge contacts) or 'cylinder' (over the wall height). Default sphere"},
        {"name": "magnet_condim", "type": "int", "description": "Contact dimensionality of magnet-to-magnet contact: 3, 4 or 6 (6 adds rolling resistance). Default 4"},
        {"name": "axis", "type": "str", "description": "'vertical' (lying flat on a face) or 'horizontal' (standing on its rim, axis along body y). Default vertical"},
        {"name": "pos", "type": "list[3]", "description": "Body position in the asset frame. Default [0, 0, 0]"},
        {"name": "yaw_deg", "type": "float", "description": "Body yaw about z in degrees. Default 0"},
        {"name": "include_floor", "type": "bool", "description": "Add a floor plane (for stand-alone tests). Default False"},
        {"name": "timestep", "type": "float", "description": "Simulation timestep written into the asset, or None. Default None"},
    ],
    "outputs": [
        {"name": "built", "type": "dict", "description": "xml, params_json, body, joint, magnet_body, magnet_geom, magnet_actuator (None without a magnet), wall_geoms, outer_radius_m, inner_radius_m, height_m, mass_kg, axis, centre_local (ring centre in the body frame)."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["ring", "magnet", "magnetic", "adhesion", "tube", "fidget", "rigid", "parametric", "object reconstruction"],
    "status": "tested",
    "entrypoint": "construct_magnetic_ring",
    "cli": "python -m tool_library.skills.construct_magnetic_ring out_dir",
}

import json
import math
from pathlib import Path

DEFAULTS = {
    "name": "magnetic_ring",
    "outer_radius": 0.0155,
    "inner_radius": 0.0105,
    "height": 0.012,
    "n_segments": 24,
    "body_rgba": [0.16, 0.13, 0.11, 1.0],
    "accent_rgba": [0.78, 0.66, 0.24, 1.0],
    "accent_every": 3,
    "mass": 0.012,
    "friction": [1.0, 0.01, 0.001],
    "solref": [0.005, 1.0],
    "solimp": [0.95, 0.99, 0.0005],
    "magnet_force_N": 2.0,
    "magnet_range_m": 0.0003,
    "magnet_pad_m": 0.0003,
    "magnet_friction": [0.8, 0.01, 0.001],
    "magnet_bit": 4,
    "magnet_shape": "sphere",
    "magnet_condim": 4,
    "axis": "vertical",
    "pos": [0.0, 0.0, 0.0],
    "yaw_deg": 0.0,
    "include_floor": False,
    "timestep": None,
}


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def _qrot(q, v):
    w, x, y, z = q
    vx, vy, vz = v
    # v' = q v q*
    t = [2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)]
    return [vx + w * t[0] + (y * t[2] - z * t[1]), vy + w * t[1] + (z * t[0] - x * t[2]),
            vz + w * t[2] + (x * t[1] - y * t[0])]


def _fmt(v):
    return " ".join(f"{float(x):.6g}" for x in v)


def _validate(p):
    unknown = sorted(set(p) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"construct_magnetic_ring: unknown parameter(s) {unknown}; allowed: {sorted(DEFAULTS)}")
    ro, ri, h = float(p["outer_radius"]), float(p["inner_radius"]), float(p["height"])
    if not (0 < ri < ro):
        raise ValueError("construct_magnetic_ring: need 0 < inner_radius < outer_radius")
    if h <= 0:
        raise ValueError("construct_magnetic_ring: height must be > 0")
    if int(p["n_segments"]) < 8:
        raise ValueError("construct_magnetic_ring: n_segments must be >= 8")
    if float(p["mass"]) <= 0:
        raise ValueError("construct_magnetic_ring: mass must be > 0")
    if float(p["magnet_force_N"]) < 0:
        raise ValueError("construct_magnetic_ring: magnet_force_N must be >= 0")
    if float(p["magnet_range_m"]) <= 0 or float(p["magnet_pad_m"]) < 0:
        raise ValueError("construct_magnetic_ring: magnet_range_m must be > 0 and magnet_pad_m >= 0")
    b = int(p["magnet_bit"])
    if b <= 1 or (b & (b - 1)) != 0:
        raise ValueError("construct_magnetic_ring: magnet_bit must be a power of two greater than 1")
    if p["magnet_shape"] not in ("sphere", "cylinder"):
        raise ValueError("construct_magnetic_ring: magnet_shape must be 'sphere' or 'cylinder'")
    if int(p["magnet_condim"]) not in (3, 4, 6):
        raise ValueError("construct_magnetic_ring: magnet_condim must be 3, 4 or 6")
    if p["axis"] not in ("vertical", "horizontal"):
        raise ValueError("construct_magnetic_ring: axis must be 'vertical' or 'horizontal'")


def build(params: dict | None = None, out_dir: str | Path = ".") -> dict:
    p = dict(DEFAULTS)
    p.update(params or {})
    _validate(p)
    name = str(p["name"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ro, ri, h = float(p["outer_radius"]), float(p["inner_radius"]), float(p["height"])
    n = int(p["n_segments"])
    wall = ro - ri
    rmid = 0.5 * (ro + ri)
    half_chord = rmid * math.tan(math.pi / n) * 1.04
    if p["axis"] == "vertical":
        q_axis = [1.0, 0.0, 0.0, 0.0]
        centre = [0.0, 0.0, h / 2.0]
    else:  # ring axis along body y, standing on its rim
        s = math.sqrt(0.5)
        q_axis = [s, -s, 0.0, 0.0]   # local z -> body y
        centre = [0.0, 0.0, ro]
    fr = _fmt(p["friction"])
    sr = _fmt(p["solref"])
    si = _fmt(p["solimp"])
    m_seg = float(p["mass"]) / n
    geoms, wall_names = [], []
    for k in range(n):
        th = 2 * math.pi * k / n
        qz = [math.cos(th / 2), 0.0, 0.0, math.sin(th / 2)]
        q = _qmul(q_axis, qz)
        local = [rmid * math.cos(th), rmid * math.sin(th), 0.0]
        rp = _qrot(q_axis, local)
        gp = [centre[i] + rp[i] for i in range(3)]
        accent = int(p["accent_every"]) > 0 and k % int(p["accent_every"]) == 0
        rgba = p["accent_rgba"] if accent else p["body_rgba"]
        gname = f"{name}_wall{k:02d}"
        wall_names.append(gname)
        geoms.append(
            f'      <geom name="{gname}" type="box" size="{wall / 2:.6g} {half_chord:.6g} {h / 2:.6g}" '
            f'pos="{_fmt(gp)}" quat="{_fmt(q)}" rgba="{_fmt(rgba)}" mass="{m_seg:.6g}" friction="{fr}" '
            f'solref="{sr}" solimp="{si}" condim="4" contype="1" conaffinity="1"/>')
    bit = int(p["magnet_bit"])
    rng = float(p["magnet_range_m"])
    mr = ro + float(p["magnet_pad_m"])
    mfr = _fmt(p["magnet_friction"])
    mshape = p["magnet_shape"]
    msize = f"{mr:.6g}" if mshape == "sphere" else f"{mr:.6g} {0.45 * h:.6g}"
    mcd = int(p["magnet_condim"])
    yaw = math.radians(float(p["yaw_deg"]))
    bq = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    # A ring with no magnet (magnet_force_N 0, a plain bead or washer) gets no magnet parts at all: a shell and a
    # zero-force actuator would be a collision-only body and an actuator that does nothing, which the scene gates flag
    # (hand_contact_pair_mask_check, actuator_mapping_check) and which no consumer needs.
    magnetic = float(p["magnet_force_N"]) > 0.0
    magnet = (
        f'      <body name="{name}_magnet" pos="{_fmt(centre)}" quat="{_fmt(q_axis)}">\n'
        f'        <geom name="{name}_magnet_shell" type="{mshape}" size="{msize}" mass="1e-05" '
        f'contype="{bit}" conaffinity="{bit}" margin="{rng:.6g}" gap="{rng:.6g}" condim="{mcd}" friction="{mfr}" '
        f'solref="{sr}" solimp="{si}" group="3" rgba="0.6 0.1 0.1 0.25"/>\n'
        f'      </body>\n') if magnetic else ""
    actuator = (
        f'  <actuator>\n'
        f'    <general name="{name}_magnet_pull" body="{name}_magnet" gainprm="0" biastype="affine" '
        f'biasprm="{float(p["magnet_force_N"]):.6g} 0 0" ctrllimited="true" ctrlrange="0 1"/>\n'
        f'  </actuator>\n') if magnetic else ""
    floor = ('    <geom name="floor" type="plane" size="0.5 0.5 0.01" rgba="0.8 0.8 0.8 1" contype="1" '
             'conaffinity="1"/>\n') if p["include_floor"] else ""
    opt = f'  <option timestep="{float(p["timestep"]):.6g}"/>\n' if p["timestep"] else ""
    xml = (
        f'<mujoco model="{name}">\n'
        f'  <compiler angle="radian"/>\n{opt}'
        f'  <worldbody>\n{floor}'
        f'    <body name="{name}" pos="{_fmt(p["pos"])}" quat="{_fmt(bq)}">\n'
        f'      <freejoint name="{name}_free"/>\n'
        + "\n".join(geoms) + "\n" + magnet +
        '    </body>\n'
        '  </worldbody>\n'
        + actuator +
        '</mujoco>\n')
    xml_path = out_dir / f"{name}.xml"
    xml_path.write_text(xml)
    pj = out_dir / f"{name}_params.json"
    pj.write_text(json.dumps(p, indent=1))
    return {"xml": str(xml_path), "params_json": str(pj), "body": name, "joint": f"{name}_free",
            "magnet_body": f"{name}_magnet" if magnetic else None,
            "magnet_geom": f"{name}_magnet_shell" if magnetic else None,
            "magnet_actuator": f"{name}_magnet_pull" if magnetic else None, "wall_geoms": wall_names,
            "outer_radius_m": ro, "inner_radius_m": ri, "height_m": h, "mass_kg": float(p["mass"]),
            "magnet_force_N": float(p["magnet_force_N"]), "axis": p["axis"], "centre_local": centre}


def construct_magnetic_ring(out_dir, **params):
    """Build one magnetic ring into out_dir. Returns the build dictionary."""
    return build(params, out_dir)


if __name__ == "__main__":
    import sys as _sys
    print(json.dumps(construct_magnetic_ring(_sys.argv[1] if len(_sys.argv) > 1 else "."), indent=1, default=str))
