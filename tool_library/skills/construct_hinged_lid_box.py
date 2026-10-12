TOOL = {
    "name": "construct_hinged_lid_box",
    "title": "Hinged-lid box with a set initial lid state",
    "type": "skill",
    "category": "Object Reconstruction",
    "summary": "A rigid box (base tray with four walls) whose lid is hinged along the top of the back wall, optionally with a creased skirt on the lid's free edge, built so that the asset's initial state (qpos0) already shows the lid at a measured opening angle.",
    "description": (
        "A gift box, shoebox or chest: the base and its four walls are one rigid body, the lid is a child body "
        "on ONE revolute joint along the top outer edge of the back (+y) wall, and an optional skirt is a second "
        "revolute crease on the lid's free edge that folds down over the outside of the front wall. Joint angles: "
        "lid 0 = shut on the walls, 90 = vertical, 180 = folded flat behind the box at hinge height; skirt 0 = in "
        "plane with the lid, 90 = square to it (down the front face when shut, straight up when the lid is folded "
        "back). Unlike a box built shut, the lid and skirt bodies are written AT their initial angles and each joint's "
        "`ref` equals that angle, so a scene that starts from qpos0 (or a keyframe padded with qpos0) shows the "
        "open lid of the first video frame while joint value 0 still means shut. lid_init_deg='rest' computes the "
        "angle past 180 at which the lid's free edge rests on the plane under the base (a lid folded back onto the "
        "table). Shut is a hard joint stop (stiff solreflimit); lid-vs-base and skirt-vs-base contacts are excluded "
        "because hinge and crease are one piece of board. Assumes a rectangular box, uniform board thickness."
    ),
    "inputs": [
        {"name": "out_dir", "type": "path", "description": "Folder for <name>.xml and <name>_params.json."},
        {"name": "name", "type": "str", "description": "Body name prefix. Default 'box'."},
        {"name": "length", "type": "float", "description": "Outer size along x (left-right, along the hinge), m. Default 0.24."},
        {"name": "width", "type": "float", "description": "Outer size along y (front-back), m. Default 0.18."},
        {"name": "height", "type": "float", "description": "Outer wall height of the base including its floor, m. Default 0.06."},
        {"name": "thickness", "type": "float", "description": "Board thickness, m. Default 0.003."},
        {"name": "lid_flap", "type": "bool", "description": "Add the creased skirt on the lid's free edge. Default True."},
        {"name": "lid_flap_h", "type": "float | None", "description": "Skirt height, m; None = the wall height. Default None."},
        {"name": "lid_init_deg", "type": "float | 'rest'", "description": "Initial lid angle (0 shut, 180 flat at hinge height); 'rest' = folded back onto the support plane. Default 0.0."},
        {"name": "skirt_init_deg", "type": "float", "description": "Initial skirt angle against the lid. Default 90.0."},
        {"name": "lid_range_deg", "type": "list[2]", "description": "Lid joint range, deg; the upper end must admit a lid folded back onto the table. Default [0.0, 240.0]."},
        {"name": "skirt_range_deg", "type": "list[2]", "description": "Skirt joint range, deg; the upper stop is the skirt flush on the front wall. Default [-5.0, 90.0]."},
        {"name": "body_rgba", "type": "list[4]", "description": "Base and walls colour. Default [0.72, 0.19, 0.17, 1]."},
        {"name": "lid_rgba", "type": "list[4] | None", "description": "Lid colour; None = body_rgba. Default None."},
        {"name": "density", "type": "float", "description": "Board density, kg/m3. Default 600."},
        {"name": "friction", "type": "list[3]", "description": "Geom friction. Default [0.9, 0.02, 0.001]."},
        {"name": "hinge_damping", "type": "float", "description": "Lid and skirt joint damping. Default 0.02."},
        {"name": "hinge_frictionloss", "type": "float", "description": "Lid joint dry friction, N m. Default 0.004."},
        {"name": "skirt_frictionloss", "type": "float | None", "description": "Skirt dry friction; None = 3x its gravity torque so it holds its angle. Default None."},
        {"name": "hinge_armature", "type": "float", "description": "Joint armature. Default 0.005."},
        {"name": "hinge_solreflimit", "type": "list[2]", "description": "Joint-limit solref. Default [0.005, 1.0]."},
        {"name": "pos", "type": "list[3]", "description": "Centre of the base underside. Default [0, 0, 0]."},
        {"name": "yaw_deg", "type": "float", "description": "Yaw about z, deg. Default 0."},
        {"name": "fixed_base", "type": "bool", "description": "No free joint on the base. Default False."},
        {"name": "timestep", "type": "float", "description": "Seconds. Default 0.002."},
        {"name": "include_floor", "type": "bool", "description": "Floor plane for standalone previews; off for the bench. Default True."},
    ],
    "outputs": [
        {"name": "built", "type": "dict", "description": "xml, params_json, base_body, lid_body, lid_joint, lid_geom, skirt_body, skirt_joint, skirt_geom, lid_init_deg, skirt_init_deg, closed_deg (0), skirt_closed_deg (90), push_local/push_bodies (free edges), inner_volume_l, mass_kg."},
    ],
    "status": "tested",
    "entrypoint": "construct_hinged_lid_box",
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["box", "gift box", "lid", "hinge", "skirt", "articulated", "initial state", "object reconstruction"],
    "used_in": ["Giftbox Packing"],
    "cli": "python -m tool_library.skills.construct_hinged_lid_box out_dir",
}

import math
from pathlib import Path

import numpy as np

from tool_library.common.mjcf import (FLOOR_XML, LIGHTS_XML, merge_params, mjcf_header,
                                      rgba, save_params, vec)

DEFAULTS = {
    "name": "box", "length": 0.24, "width": 0.18, "height": 0.06, "thickness": 0.003,
    "lid_flap": True, "lid_flap_h": None, "lid_init_deg": 0.0, "skirt_init_deg": 90.0,
    "lid_range_deg": [0.0, 240.0], "skirt_range_deg": [-5.0, 90.0],
    "body_rgba": [0.72, 0.19, 0.17, 1.0], "lid_rgba": None,
    "density": 600.0, "friction": [0.9, 0.02, 0.001],
    "hinge_damping": 0.02, "hinge_frictionloss": 0.004, "skirt_frictionloss": None,
    "hinge_armature": 0.005, "hinge_solreflimit": [0.005, 1.0],
    "pos": [0.0, 0.0, 0.0], "yaw_deg": 0.0, "fixed_base": False,
    "timestep": 0.002, "include_floor": True,
}


def _rest_angle(W, H, t, clearance=0.001):
    """Lid angle past 180 at which the lid's lower free corner sits `clearance` above the base underside plane."""
    hz = H + t / 2.0                      # hinge (lid mid-plane) height above the underside
    for a in np.linspace(0.0, 89.0, 89001):
        r = math.radians(a)
        low = W * math.sin(r) + (t / 2.0) * math.cos(r)
        if low >= hz - clearance:
            return 180.0 + float(a)
    raise ValueError("lid_init_deg='rest' impossible: the lid is shorter than the wall height; give a number")


def construct_hinged_lid_box(out_dir, **params):
    try:
        p = merge_params(DEFAULTS, params)
    except KeyError as e:
        raise ValueError(str(e))
    name = str(p["name"])
    L, W, H, t = (float(p[k]) for k in ("length", "width", "height", "thickness"))
    if min(L, W, H, t) <= 0 or t * 4 >= min(L, W, H):
        raise ValueError("length, width, height must be positive and well above 4x thickness")
    lo, hi = (float(v) for v in p["lid_range_deg"])
    slo, shi = (float(v) for v in p["skirt_range_deg"])
    lid0 = _rest_angle(W, H, t) if p["lid_init_deg"] == "rest" else float(p["lid_init_deg"])
    sk0 = float(p["skirt_init_deg"])
    if not (lo - 1e-6 <= lid0 <= hi + 1e-6):
        raise ValueError(f"lid_init_deg {lid0:.1f} outside lid_range_deg {p['lid_range_deg']}")
    if p["lid_flap"] and not (slo - 1e-6 <= sk0 <= shi + 1e-6):
        raise ValueError(f"skirt_init_deg {sk0:.1f} outside skirt_range_deg {p['skirt_range_deg']}")
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fr = vec(p["friction"])
    g = f'density="{p["density"]}" friction="{fr}"'
    body_c = rgba(p["body_rgba"]); lid_c = rgba(p["lid_rgba"] or p["body_rgba"])
    wall_h = H - t                                   # walls stand on the floor panel
    walls = [
        (f"{name}_wall_front", (L / 2, t / 2, wall_h / 2), (0, -(W / 2 - t / 2), t + wall_h / 2)),
        (f"{name}_wall_back", (L / 2, t / 2, wall_h / 2), (0, +(W / 2 - t / 2), t + wall_h / 2)),
        (f"{name}_wall_left", (t / 2, W / 2 - t, wall_h / 2), (-(L / 2 - t / 2), 0, t + wall_h / 2)),
        (f"{name}_wall_right", (t / 2, W / 2 - t, wall_h / 2), (+(L / 2 - t / 2), 0, t + wall_h / 2)),
    ]
    wall_xml = "\n".join(f'      <geom name="{n}" type="box" size="{vec(s)}" pos="{vec(q)}" rgba="{body_c}" {g}/>'
                         for n, s, q in walls)
    jcommon = (f'damping="{p["hinge_damping"]}" armature="{p["hinge_armature"]}" '
               f'solreflimit="{vec(p["hinge_solreflimit"])}"')
    skirt_xml, skirt_h, s_fric = "", 0.0, 0.0
    if p["lid_flap"]:
        skirt_h = float(p["lid_flap_h"] or H)
        s_mass = L * skirt_h * t * float(p["density"])
        s_fric = float(p["skirt_frictionloss"] if p["skirt_frictionloss"] is not None
                       else max(float(p["hinge_frictionloss"]), 3.0 * s_mass * 9.81 * skirt_h / 2))
        # crease on the lid's free edge, half a board below the lid mid-plane so the folded skirt lies OUTSIDE the front wall
        skirt_xml = f"""
        <body name="{name}_lidflapb" pos="0 {-W:.5f} 0" euler="{sk0:.4f} 0 0">
          <joint name="{name}_lidflap" type="hinge" axis="1 0 0" range="{slo} {shi}" ref="{sk0:.4f}"
                 frictionloss="{s_fric:.5f}" {jcommon}/>
          <geom name="{name}_lidflap" type="box" size="{vec((L / 2, skirt_h / 2, t / 2))}"
                pos="0 {-skirt_h / 2:.5f} {t / 2:.5f}" rgba="{lid_c}" {g}/>
        </body>"""
    free = "" if p["fixed_base"] else f'      <freejoint name="{name}_free"/>\n'
    floor = FLOOR_XML if p["include_floor"] else ""
    xml = mjcf_header(f"{name}_hingedlidbox", timestep=p["timestep"]) + f"""
  <worldbody>{LIGHTS_XML}{floor}
    <body name="{name}" pos="{vec(p['pos'])}" euler="0 0 {float(p['yaw_deg'])}">
{free}      <geom name="{name}_base" type="box" size="{vec((L / 2, W / 2, t / 2))}" pos="0 0 {t / 2:.5f}" rgba="{body_c}" {g}/>
{wall_xml}
      <body name="{name}_lidb" pos="0 {W / 2:.5f} {H + t / 2:.5f}" euler="{-lid0:.4f} 0 0">
        <joint name="{name}_lid" type="hinge" axis="-1 0 0" range="{lo} {hi}" ref="{lid0:.4f}"
               frictionloss="{p['hinge_frictionloss']}" {jcommon}/>
        <geom name="{name}_lid" type="box" size="{vec((L / 2, W / 2, t / 2))}" pos="0 {-W / 2:.5f} 0" rgba="{lid_c}" {g}/>{skirt_xml}
      </body>
    </body>
  </worldbody>
  <contact>
    <exclude body1="{name}" body2="{name}_lidb"/>
{f'    <exclude body1="{name}" body2="{name}_lidflapb"/>' if p["lid_flap"] else ''}
  </contact>
</mujoco>
"""
    xml_path = out_dir / f"{name}.xml"
    xml_path.write_text(xml)
    save_params(out_dir / f"{name}_params.json", dict(p, lid_init_deg=lid0))
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    root = m.body(name).id
    mass = float(m.body_subtreemass[root])
    return {
        "xml": str(xml_path), "params_json": str(out_dir / f"{name}_params.json"),
        "base_body": name, "lid_body": f"{name}_lidb", "lid_joint": f"{name}_lid", "lid_geom": f"{name}_lid",
        "skirt_body": f"{name}_lidflapb" if p["lid_flap"] else None,
        "skirt_joint": f"{name}_lidflap" if p["lid_flap"] else None,
        "skirt_geom": f"{name}_lidflap" if p["lid_flap"] else None,
        "skirt_h_m": round(skirt_h, 4), "skirt_frictionloss": round(s_fric, 5),
        "lid_init_deg": round(lid0, 3), "skirt_init_deg": sk0 if p["lid_flap"] else None,
        "closed_deg": 0.0, "skirt_closed_deg": 90.0,
        "push_local": {f"{name}_lid": [0.0, -W, 0.0], **({f"{name}_lidflap": [0.0, -skirt_h, t / 2]} if p["lid_flap"] else {})},
        "push_bodies": {f"{name}_lid": f"{name}_lidb", **({f"{name}_lidflap": f"{name}_lidflapb"} if p["lid_flap"] else {})},
        "inner_volume_l": round(1000 * (L - 2 * t) * (W - 2 * t) * (H - t), 3),
        "mass_kg": round(mass, 4),
        "note": "lid 0 deg = shut, 180 = folded flat behind at hinge height; joint refs equal the initial angles so qpos0 shows the open lid.",
    }


if __name__ == "__main__":
    import json, sys
    print(json.dumps(construct_hinged_lid_box(sys.argv[1] if len(sys.argv) > 1 else "."), indent=1))
