"""Construct a small plush toy (stuffed animal) as a light, compliant free body.

    from tool_library.skills.construct_plush_toy import construct_plush_toy
    out = construct_plush_toy(out_dir, length=0.075, width=0.06, height=0.03,
                              body_rgba=[0.84, 0.65, 0.33, 1])
    out["xml"], out["body"], out["mass_kg"], out["size_m"]

A plush toy is a soft, stuffed fabric object whose exact shape rarely matters to a
manipulation task: what matters is its overall size (so it fits where the video
shows it, e.g. inside a drawer), its low mass, its fabric friction, and that it
gives under a pinch. This builds it from overlapping ellipsoids (a torso, a head
and optional ears or limbs scaled to the measured length x width x height) on one
free joint, and makes it compliant through its contact parameters: a long solref
time constant and a wide solimp band let fingers sink several millimetres into the
surface under a grasp force instead of meeting a rigid shell. The softness is
checked by a squeeze test (soft_squeeze_test). Optional flattened rest shape:
`height` may be smaller than the free-standing height when the toy is squashed
into a confined space at the start of the clip.
"""
from __future__ import annotations

TOOL = {
    "name": "construct_plush_toy",
    "title": "Construct plush toy",
    "type": "skill",
    "category": "Object Reconstruction",
    "summary": "A small stuffed toy as a light free body of overlapping ellipsoids (torso, head, ears) sized to a measured length x width x height, made compliant through soft contact parameters so fingers sink into it under a grasp.",
    "description": (
        "Builds an MJCF asset with one free body: a torso ellipsoid, a head ellipsoid placed "
        "toward +x, and n_ears small ellipsoids on top of the head, all scaled so the union "
        "spans the given length (x), width (y) and height (z) with its base at z=0. Every "
        "geom collides with solref=(soft_timeconst, soft_dampratio) and solimp=(soft_dmin, "
        "soft_dmax, soft_width): a long time constant and a wide impedance band let a pinch "
        "push the contact several millimetres in, which is how a rigid-body simulator shows "
        "a plush squashing in the hand. Mass is the measured or estimated mass (default 30 g), "
        "friction is fabric-like (default 1.0, condim 4 for torsional grip). The accent "
        "colour (eyes / nose) is a pair of tiny visual-only spheres on the head. Sites: "
        "'<name>_grasp' at the torso centre and '<name>_top' on the top surface. Assumes "
        "x is the long axis of the toy; set yaw_deg to orient it. Unknown parameters raise "
        "ValueError."
    ),
    "inputs": [
        {"name": "out_dir", "type": "str", "description": "Directory receiving <name>.xml and <name>_params.json."},
        {"name": "name", "type": "str", "description": "Body/geom prefix. Default 'plush'."},
        {"name": "length", "type": "float, metres", "description": "Overall length along x (measured). Default 0.08."},
        {"name": "width", "type": "float, metres", "description": "Overall width along y (measured). Default 0.06."},
        {"name": "height", "type": "float, metres", "description": "Overall height along z, at rest (may be the squashed height). Default 0.05."},
        {"name": "head_frac", "type": "float", "description": "Head diameter as a fraction of length. Default 0.45 (0 = no head)."},
        {"name": "n_ears", "type": "int", "description": "Small ellipsoids on top of the head: 0, 1 or 2. Default 2."},
        {"name": "body_rgba", "type": "list[float]", "description": "Fabric colour (measured). Default pale yellow."},
        {"name": "accent_rgba", "type": "list[float] | None", "description": "Eye / nose colour; None for no accents. Default dark brown."},
        {"name": "mass", "type": "float, kg", "description": "Total mass. Default 0.03."},
        {"name": "friction", "type": "float", "description": "Sliding friction. Default 1.0."},
        {"name": "soft_timeconst", "type": "float, s", "description": "Contact solref time constant; larger = softer. Default 0.03."},
        {"name": "soft_dampratio", "type": "float", "description": "Contact solref damping ratio. Default 1.0."},
        {"name": "soft_dmin", "type": "float", "description": "solimp dmin. Default 0.5."},
        {"name": "soft_dmax", "type": "float", "description": "solimp dmax. Default 0.95."},
        {"name": "soft_width", "type": "float, m", "description": "solimp width: the depth over which impedance ramps. Default 0.012."},
        {"name": "pos", "type": "list[float]", "description": "Initial base position [x, y, z]. Default [0, 0, 0]."},
        {"name": "yaw_deg", "type": "float", "description": "Initial yaw about z. Default 0."},
        {"name": "free", "type": "bool", "description": "Free joint (True) or fixed to the world. Default True."},
        {"name": "include_floor", "type": "bool", "description": "Add a floor plane for stand-alone testing. Default False."},
        {"name": "timestep", "type": "float, s", "description": "Option timestep. Default 0.002."},
    ],
    "outputs": [
        {"name": "xml", "type": "str", "description": "The MJCF asset path."},
        {"name": "params_json", "type": "str", "description": "Every parameter as built."},
        {"name": "body", "type": "str", "description": "Root body name."},
        {"name": "bodies", "type": "list[str]", "description": "All body names."},
        {"name": "joint", "type": "str | None", "description": "Free joint name."},
        {"name": "sites", "type": "list[str]", "description": "Grasp and top sites."},
        {"name": "mass_kg", "type": "float", "description": "Total mass as compiled."},
        {"name": "size_m", "type": "list[float]", "description": "Compiled bounding box [dx, dy, dz] of the collision geoms."},
    ],
    "status": "tested",
    "entrypoint": "construct_plush_toy",
    "requirements": ["numpy", "mujoco"],
    "tags": ["plush", "plushie", "stuffed animal", "soft", "toy", "compliant", "deformable"],
    "used_in": ["Place Toy in Drawer"],
    "cli": "python -m tool_library.skills.construct_plush_toy out_dir",
}

import json
import math
import os
import sys
from pathlib import Path

DEFAULTS = dict(
    name="plush", length=0.08, width=0.06, height=0.05, head_frac=0.45, n_ears=2,
    body_rgba=[0.88, 0.78, 0.45, 1.0], accent_rgba=[0.25, 0.15, 0.08, 1.0], mass=0.03,
    friction=1.0, soft_timeconst=0.03, soft_dampratio=1.0, soft_dmin=0.5, soft_dmax=0.95,
    soft_width=0.012, pos=[0.0, 0.0, 0.0], yaw_deg=0.0, free=True, include_floor=False,
    timestep=0.002,
)


def _f(v):
    return " ".join(f"{float(x):.6g}" for x in v)


def construct_plush_toy(out_dir, **kw):
    unknown = sorted(set(kw) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"unknown parameter(s) {unknown}; allowed: {sorted(DEFAULTS)}")
    p = dict(DEFAULTS); p.update(kw)
    L, Wd, H = float(p["length"]), float(p["width"]), float(p["height"])
    for k, v in (("length", L), ("width", Wd), ("height", H)):
        if not (0.005 <= v <= 0.6):
            raise ValueError(f"{k}={v} m is outside 0.005..0.6 m: give metres")
    if Wd > L:
        raise ValueError("width must not exceed length: x is the long axis (swap them and set yaw_deg)")
    hf = float(p["head_frac"])
    if not (0.0 <= hf <= 0.7):
        raise ValueError("head_frac must be within 0..0.7")
    ne = int(p["n_ears"])
    if ne not in (0, 1, 2):
        raise ValueError("n_ears must be 0, 1 or 2")
    if float(p["mass"]) <= 0:
        raise ValueError("mass must be positive")
    if not (0 < p["soft_dmin"] <= p["soft_dmax"] < 1):
        raise ValueError("need 0 < soft_dmin <= soft_dmax < 1")
    if p["soft_timeconst"] < 2 * p["timestep"]:
        raise ValueError("soft_timeconst must be at least 2 x timestep")
    n = p["name"]
    # geometry: head radius rh (sphere-ish), torso semi-axes; ears add height on the head
    ear_h = 0.18 * H if ne else 0.0
    if hf > 0:
        rh = min(hf * L / 2, Wd / 2, (H - ear_h) / 2)
        tx = (L - 2 * rh * 0.6) / 2  # torso semi-axis x (head overlaps the torso by 40 %)
        tz = (H - ear_h) / 2
        tcx = -L / 2 + tx
        hcx = L / 2 - rh
    else:
        rh, tx, tz, tcx, hcx = 0.0, L / 2, H / 2, 0.0, 0.0
    ty = Wd / 2
    soft = f'solref="{p["soft_timeconst"]:.6g} {p["soft_dampratio"]:.6g}" solimp="{p["soft_dmin"]:.6g} {p["soft_dmax"]:.6g} {p["soft_width"]:.6g}"'
    col = f'friction="{p["friction"]:.6g} 0.01 0.001" condim="4" {soft} rgba="{_f(p["body_rgba"])}"'
    geoms = [f'<geom name="{n}_torso" type="ellipsoid" size="{tx:.6g} {ty:.6g} {tz:.6g}" pos="{tcx:.6g} 0 {tz:.6g}" {col}/>']
    if rh > 0:
        geoms.append(f'<geom name="{n}_head" type="ellipsoid" size="{rh:.6g} {min(rh, ty):.6g} {rh:.6g}" pos="{hcx:.6g} 0 {rh:.6g}" {col}/>')
        if ne:
            er = (0.35 * rh, 0.25 * rh, ear_h / 2 + 0.25 * rh)
            ez = 2 * rh + ear_h / 2 - 0.25 * rh
            offs = [0.0] if ne == 1 else [-0.5 * rh, 0.5 * rh]
            for i, oy in enumerate(offs):
                geoms.append(f'<geom name="{n}_ear{i}" type="ellipsoid" size="{er[0]:.6g} {er[1]:.6g} {er[2]:.6g}" pos="{hcx:.6g} {oy:.6g} {ez:.6g}" {col}/>')
        if p["accent_rgba"] is not None:
            ar = max(0.06 * rh, 0.001)
            for i, oy in enumerate((-0.35 * rh, 0.35 * rh)):
                geoms.append(f'<geom name="{n}_eye{i}" type="sphere" size="{ar:.6g}" pos="{hcx + 0.85 * rh:.6g} {oy:.6g} {1.25 * rh:.6g}" contype="0" conaffinity="0" mass="0" rgba="{_f(p["accent_rgba"])}"/>')
    # distribute mass by volume over collision geoms
    vols = []
    for g in geoms:
        if 'contype="0"' in g:
            vols.append(0.0); continue
        s = [float(x) for x in g.split('size="')[1].split('"')[0].split()]
        vols.append(4 / 3 * math.pi * s[0] * s[1] * s[2])
    tot = sum(vols)
    geoms = [g if v == 0 else g.replace("/>", f' mass="{p["mass"] * v / tot:.6g}"/>') for g, v in zip(geoms, vols)]
    yaw = math.radians(float(p["yaw_deg"]))
    quat = f"{math.cos(yaw / 2):.6g} 0 0 {math.sin(yaw / 2):.6g}"
    joint = f'<freejoint name="{n}_free"/>' if p["free"] else ""
    floor = '<geom name="floor" type="plane" size="1 1 0.01" rgba="0.8 0.8 0.8 1"/>' if p["include_floor"] else ""
    xml = f"""<mujoco model="{n}">
  <option timestep="{p['timestep']:.6g}"/>
  <worldbody>
    {floor}
    <body name="{n}" pos="{_f(p['pos'])}" quat="{quat}">
      {joint}
      {chr(10).join('      ' + g for g in geoms).strip()}
      <site name="{n}_grasp" pos="{tcx:.6g} 0 {tz:.6g}" size="0.004" rgba="1 0 0 0"/>
      <site name="{n}_top" pos="{tcx:.6g} 0 {2 * tz:.6g}" size="0.004" rgba="1 0 0 0"/>
    </body>
  </worldbody>
</mujoco>
"""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    xp = out / f"{n}.xml"; xp.write_text(xml)
    pp = out / f"{n}_params.json"; pp.write_text(json.dumps(p, indent=1))
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    import numpy as np
    m = mujoco.MjModel.from_xml_path(str(xp)); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    bid = m.body(n).id
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != bid or m.geom_contype[g] == 0:
            continue
        R = d.geom_xmat[g].reshape(3, 3); s = m.geom_size[g]
        ext = np.abs(R) @ s
        lo = np.minimum(lo, d.geom_xpos[g] - ext); hi = np.maximum(hi, d.geom_xpos[g] + ext)
    return {"xml": str(xp), "params_json": str(pp), "body": n, "bodies": [n],
            "joint": f"{n}_free" if p["free"] else None, "sites": [f"{n}_grasp", f"{n}_top"],
            "mass_kg": float(m.body_subtreemass[bid]), "size_m": [round(float(x), 5) for x in (hi - lo)]}


if __name__ == "__main__":
    print(json.dumps(construct_plush_toy(sys.argv[1]), indent=1))
