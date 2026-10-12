"""A finger-pump mist sprayer: a bottle whose pump head rides a sprung, preloaded prismatic joint along the bottle axis."""
from __future__ import annotations

TOOL = {
    "name": "construct_pump_sprayer_down_axis",
    "title": "Construct pump sprayer (stroke as positive joint travel)",
    "type": "skill",
    "category": "Object Reconstruction",
    "summary": "A finger-pump spray bottle as a free rigid body whose pump head is a separate body on a preloaded spring slide joint along the bottle axis: pressed down through a short stroke by a finger, it springs back.",
    "description": (
        "Fork of construct_pump_sprayer with the library's joint convention: the slide axis points DOWN the bottle (0 0 -1) "
        "and the range is [0, stroke], so the rest state is the lower limit and pressing increases the joint value, as "
        "articulation_check and the travel tests expect. Builds one MJCF: a cylindrical bottle body (with a visual colour band), a shoulder, neck and collar, and a pump "
        "head (stem, actuator cylinder and a side nozzle) as a child body on a slide joint along the bottle's z axis. The "
        "joint range is [0, stroke]; its spring has stiffness `spring_stiffness` and a springref placed beyond the rest stop "
        "so the head rests at 0 with `preload_N` pressing it against the stop: a finger must exceed the preload to start "
        "the stroke and needs preload + stiffness * stroke to bottom out; released, it returns. Geom densities are set so "
        "the body and head weigh `mass` and `head_mass`; the body-head contact pair is excluded explicitly (the parent filter does not apply once the bottle is welded to the world). The bottle origin is the centre of its base, z up; the nozzle "
        "points along +x rotated by `nozzle_azimuth_deg`. Sites: press (top of the head), nozzle (tip), grasp (mid body). "
        "Assumes a round bottle; sizes in metres. Unknown parameters and inconsistent sizes raise ValueError."
    ),
    "inputs": [
        {"name": "out_dir", "type": "path", "description": "Receives <name>.xml and <name>_params.json."},
        {"name": "name", "type": "str", "description": "Asset name and prefix. Default 'sprayer'."},
        {"name": "body_radius", "type": "float", "description": "Bottle radius. Default 0.020."},
        {"name": "body_height", "type": "float", "description": "Cylindrical body height. Default 0.085."},
        {"name": "shoulder_height", "type": "float", "description": "Shoulder height between body and neck. Default 0.010."},
        {"name": "neck_radius", "type": "float", "description": "Neck radius. Default 0.010."},
        {"name": "neck_height", "type": "float", "description": "Neck height. Default 0.008."},
        {"name": "collar_radius", "type": "float", "description": "Pump collar radius. Default 0.012."},
        {"name": "collar_height", "type": "float", "description": "Pump collar height (sits on top of the neck). Default 0.008."},
        {"name": "stem_radius", "type": "float", "description": "Pump stem radius. Default 0.004."},
        {"name": "stem_height", "type": "float", "description": "Exposed stem between collar and head at rest; must exceed the stroke. Default 0.009."},
        {"name": "head_radius", "type": "float", "description": "Actuator head radius. Default 0.010."},
        {"name": "head_height", "type": "float", "description": "Actuator head height. Default 0.020."},
        {"name": "nozzle_length", "type": "float", "description": "Side nozzle length. Default 0.005."},
        {"name": "nozzle_radius", "type": "float", "description": "Side nozzle radius. Default 0.0025."},
        {"name": "nozzle_azimuth_deg", "type": "float", "description": "Nozzle direction about z from +x. Default 0."},
        {"name": "stroke", "type": "float", "description": "Pump stroke in metres. Default 0.006."},
        {"name": "spring_stiffness", "type": "float", "description": "Return spring N/m. Default 500."},
        {"name": "preload_N", "type": "float", "description": "Spring force holding the head on its top stop. Default 2.0."},
        {"name": "head_damping", "type": "float", "description": "Slide damping N s/m. Default 1.0."},
        {"name": "head_armature", "type": "float", "description": "Slide armature kg (numerical stability). Default 0.002."},
        {"name": "head_mass", "type": "float", "description": "Pump head mass kg. Default 0.005."},
        {"name": "band_z", "type": "float", "description": "Bottom of the colour band above the base. Default 0.0."},
        {"name": "band_height", "type": "float", "description": "Colour band height; 0 for none. Default 0.025."},
        {"name": "body_rgba", "type": "list[4]", "description": "Bottle colour."},
        {"name": "band_rgba", "type": "list[4]", "description": "Band colour."},
        {"name": "head_rgba", "type": "list[4]", "description": "Pump head and collar colour."},
        {"name": "mass", "type": "float", "description": "Bottle (without head) mass kg, liquid included. Default 0.08."},
        {"name": "friction", "type": "list[3]", "description": "MuJoCo friction triple. Default [1.0, 0.01, 0.001]."},
        {"name": "pos", "type": "list[3]", "description": "Base centre position. Default [0, 0, 0]."},
        {"name": "yaw_deg", "type": "float", "description": "Yaw about z. Default 0."},
        {"name": "free", "type": "bool", "description": "Free body (True) or fixed to its parent. Default True."},
        {"name": "include_floor", "type": "bool", "description": "Add a floor plane (stand-alone previews only). Default False."},
        {"name": "timestep", "type": "float", "description": "Option timestep. Default 0.002."},
    ],
    "outputs": [
        {"name": "built", "type": "dict", "description": "xml, params_json, body, head_body, joint, sites {press, nozzle, grasp}, stroke_m, spring_stiffness, preload_N, full_stroke_force_N, height_m, radius_m, mass_kg."},
    ],
    "status": "tested",
    "entrypoint": "construct_pump_sprayer_down_axis",
    "requirements": ["mujoco>=3.1", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["spray bottle", "pump", "sprayer", "prismatic", "spring", "articulated", "object reconstruction"],
    "used_in": [],
    "cli": "python -m tool_library.skills.construct_pump_sprayer_down_axis out_dir",
}

import json
import math
import sys
from pathlib import Path

DEFAULTS = {
    "name": "sprayer",
    "body_radius": 0.020, "body_height": 0.085, "shoulder_height": 0.010,
    "neck_radius": 0.010, "neck_height": 0.008,
    "collar_radius": 0.012, "collar_height": 0.008,
    "stem_radius": 0.004, "stem_height": 0.009,
    "head_radius": 0.010, "head_height": 0.020,
    "nozzle_length": 0.005, "nozzle_radius": 0.0025, "nozzle_azimuth_deg": 0.0,
    "stroke": 0.006, "spring_stiffness": 500.0, "preload_N": 2.0,
    "head_damping": 1.0, "head_armature": 0.002, "head_mass": 0.005,
    "band_z": 0.0, "band_height": 0.025,
    "body_rgba": [0.20, 0.25, 0.60, 1.0], "band_rgba": [0.55, 0.75, 0.15, 1.0], "head_rgba": [0.95, 0.95, 0.95, 1.0],
    "mass": 0.08, "friction": [1.0, 0.01, 0.001],
    "pos": [0.0, 0.0, 0.0], "yaw_deg": 0.0, "free": True,
    "include_floor": False, "timestep": 0.002,
}

POSITIVE = ["body_radius", "body_height", "neck_radius", "collar_radius", "collar_height", "stem_radius", "stem_height",
            "head_radius", "head_height", "nozzle_length", "nozzle_radius", "stroke", "spring_stiffness", "head_mass", "mass",
            "head_armature", "timestep"]


def _f(v):
    return " ".join(f"{float(x):.6g}" for x in v)


def construct_pump_sprayer_down_axis(out_dir, **params):
    unknown = sorted(set(params) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"unknown parameter(s) {unknown}; valid: {sorted(DEFAULTS)}")
    p = {**DEFAULTS, **params}
    for k in POSITIVE:
        if float(p[k]) <= 0:
            raise ValueError(f"{k} must be > 0 (got {p[k]})")
    for k in ("shoulder_height", "neck_height", "preload_N", "head_damping", "band_height", "band_z"):
        if float(p[k]) < 0:
            raise ValueError(f"{k} must be >= 0 (got {p[k]})")
    if p["stroke"] >= p["stem_height"]:
        raise ValueError(f"stroke {p['stroke']} must be smaller than stem_height {p['stem_height']} (the head would sink into the collar)")
    if p["neck_radius"] > p["body_radius"] or p["collar_radius"] > p["body_radius"]:
        raise ValueError("neck_radius and collar_radius must not exceed body_radius")
    if p["stem_radius"] >= p["head_radius"]:
        raise ValueError("stem_radius must be smaller than head_radius")
    if p["band_z"] + p["band_height"] > p["body_height"]:
        raise ValueError("band_z + band_height must lie within body_height")
    n = str(p["name"])
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rb, hb, sh = float(p["body_radius"]), float(p["body_height"]), float(p["shoulder_height"])
    rn, hn = float(p["neck_radius"]), float(p["neck_height"])
    rc, hc = float(p["collar_radius"]), float(p["collar_height"])
    rs, hs = float(p["stem_radius"]), float(p["stem_height"])
    rh, hh = float(p["head_radius"]), float(p["head_height"])
    rsh = 0.5 * (rb + rn)
    z_neck = hb + sh
    z_top = z_neck + hn + hc                      # collar top = head body origin
    v_body = math.pi * (rb * rb * hb + rsh * rsh * sh + rn * rn * hn + rc * rc * hc)
    dens_body = float(p["mass"]) / v_body
    v_head = math.pi * (rs * rs * hs + rh * rh * hh) + math.pi * p["nozzle_radius"] ** 2 * p["nozzle_length"]
    dens_head = float(p["head_mass"]) / v_head
    k = float(p["spring_stiffness"]); stroke = float(p["stroke"])
    springref = -float(p["preload_N"]) / k
    fr = _f(p["friction"])
    az = math.radians(float(p["nozzle_azimuth_deg"]))
    zn = hs + 0.65 * hh
    nx0, ny0 = (rh - 0.001) * math.cos(az), (rh - 0.001) * math.sin(az)
    nx1, ny1 = (rh + p["nozzle_length"]) * math.cos(az), (rh + p["nozzle_length"]) * math.sin(az)
    body_rgba, band_rgba, head_rgba = _f(p["body_rgba"]), _f(p["band_rgba"]), _f(p["head_rgba"])
    geoms = [
        f'<geom name="{n}_body_geom" type="cylinder" size="{rb:.6g} {hb / 2:.6g}" pos="0 0 {hb / 2:.6g}" rgba="{body_rgba}" density="{dens_body:.6g}" friction="{fr}" condim="4"/>',
    ]
    if sh > 0:
        geoms.append(f'<geom name="{n}_shoulder_geom" type="cylinder" size="{rsh:.6g} {sh / 2:.6g}" pos="0 0 {hb + sh / 2:.6g}" rgba="{body_rgba}" density="{dens_body:.6g}" friction="{fr}" condim="4"/>')
    if hn > 0:
        geoms.append(f'<geom name="{n}_neck_geom" type="cylinder" size="{rn:.6g} {hn / 2:.6g}" pos="0 0 {z_neck + hn / 2:.6g}" rgba="{body_rgba}" density="{dens_body:.6g}" friction="{fr}" condim="4"/>')
    geoms.append(f'<geom name="{n}_collar_geom" type="cylinder" size="{rc:.6g} {hc / 2:.6g}" pos="0 0 {z_neck + hn + hc / 2:.6g}" rgba="{head_rgba}" density="{dens_body:.6g}" friction="{fr}" condim="4"/>')
    if p["band_height"] > 0:
        bz = float(p["band_z"]) + float(p["band_height"]) / 2
        geoms.append(f'<geom name="{n}_band_visual" type="cylinder" size="{rb * 1.012:.6g} {p["band_height"] / 2:.6g}" pos="0 0 {bz:.6g}" rgba="{band_rgba}" contype="0" conaffinity="0" group="1" density="0"/>')
    geoms.append(f'<site name="{n}_grasp_site" pos="0 0 {hb / 2:.6g}" size="0.003"/>')
    head = [
        f'<joint name="{n}_pump_slide" type="slide" axis="0 0 -1" limited="true" range="0 {stroke:.6g}" stiffness="{k:.6g}" springref="{springref:.6g}" damping="{float(p["head_damping"]):.6g}" armature="{float(p["head_armature"]):.6g}" solreflimit="0.004 1"/>',
        f'<geom name="{n}_stem_geom" type="cylinder" size="{rs:.6g} {hs / 2:.6g}" pos="0 0 {hs / 2:.6g}" rgba="{head_rgba}" density="{dens_head:.6g}" friction="{fr}"/>',
        f'<geom name="{n}_head_geom" type="cylinder" size="{rh:.6g} {hh / 2:.6g}" pos="0 0 {hs + hh / 2:.6g}" rgba="{head_rgba}" density="{dens_head:.6g}" friction="{fr}" condim="4"/>',
        f'<geom name="{n}_nozzle_geom" type="cylinder" size="{float(p["nozzle_radius"]):.6g}" fromto="{nx0:.6g} {ny0:.6g} {zn:.6g} {nx1:.6g} {ny1:.6g} {zn:.6g}" rgba="{head_rgba}" density="{dens_head:.6g}" friction="{fr}"/>',
        f'<site name="{n}_press_site" pos="0 0 {hs + hh:.6g}" size="0.003"/>',
        f'<site name="{n}_nozzle_site" pos="{nx1:.6g} {ny1:.6g} {zn:.6g}" size="0.002"/>',
    ]
    pos = _f(p["pos"])
    fj = f'<freejoint name="{n}_free"/>' if p["free"] else ""
    floor = '<geom name="floor" type="plane" size="1 1 0.01" rgba="0.8 0.8 0.8 1"/>' if p["include_floor"] else ""
    xml = (f'<mujoco model="{n}">\n  <compiler angle="degree"/>\n  <option timestep="{float(p["timestep"]):.6g}"/>\n  <worldbody>\n    {floor}\n'
           f'    <body name="{n}" pos="{pos}" euler="0 0 {float(p["yaw_deg"]):.6g}">\n      {fj}\n      '
           + "\n      ".join(geoms) +
           f'\n      <body name="{n}_head" pos="0 0 {z_top:.6g}">\n        ' + "\n        ".join(head) +
           '\n      </body>\n    </body>\n  </worldbody>\n'
           f'  <contact>\n    <exclude body1="{n}" body2="{n}_head"/>\n  </contact>\n</mujoco>\n')
    xp = out / f"{n}.xml"; xp.write_text(xml)
    pj = out / f"{n}_params.json"; pj.write_text(json.dumps(p, indent=1))
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(xp))
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    total = float(m.body_subtreemass[bid])
    return {"xml": str(xp), "params_json": str(pj), "body": n, "head_body": f"{n}_head", "joint": f"{n}_pump_slide",
            "free_joint": f"{n}_free" if p["free"] else None,
            "sites": {"press": f"{n}_press_site", "nozzle": f"{n}_nozzle_site", "grasp": f"{n}_grasp_site"},
            "stroke_m": stroke, "spring_stiffness": k, "preload_N": float(p["preload_N"]),
            "full_stroke_force_N": float(p["preload_N"]) + k * stroke, "rest_stop": "lower", "press_direction": "+joint",
            "height_m": z_top + hs + hh, "radius_m": rb, "mass_kg": total}


if __name__ == "__main__":
    print(json.dumps(construct_pump_sprayer_down_axis(sys.argv[1] if len(sys.argv) > 1 else "sprayer_out"), indent=1))
