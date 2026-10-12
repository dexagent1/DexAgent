"""Construct capped marker: a whiteboard marker whose cap sits ON the nib and
slides off along the barrel axis.

construct_marker builds the pen in its writing state: the cap posted on the
back end, the nib bare. A pen that lies on the table capped, is uncapped in
the hand and is recapped afterwards needs the cap as a MOVING PART: here it
is a child body on a slide joint along the pen axis (local +x, nib-ward).
q = 0 is seated (the cap hides the nib, nothing can write); q >= the engaged
depth (collar + nib + clearance) is off (the nib is exposed and the cap is
free of it). A frictionloss on the slide is the press-fit: the cap does not
fall off under gravity or a shake, but a fingertip pull of a few newtons
draws it. The joint range goes a margin past the engaged depth so a plan can
park the cap clear of the nib.
"""
from __future__ import annotations

TOOL = {
    "name": "construct_marker_capped",
    "title": "Construct capped marker",
    "type": "skill",
    "category": "Object Reconstruction",
    "summary": "A whiteboard marker whose cap covers the nib on a press-fit slide joint along the "
               "barrel axis, so uncapping and recapping are real motions of the asset.",
    "description": (
        "Builds a free rigid marker lying along its local +x (nib at +length/2, back plug at "
        "-length/2) from barrel, cap, collar and nib dimensions, and puts the cap on a child "
        "body with a slide joint <name>_cap_slide along +x. q = 0 seats the cap over the collar "
        "and nib; q >= engaged_depth (collar + nib + cap_clearance) uncovers the nib; the range "
        "runs to engaged_depth + off_margin. The press fit is a joint frictionloss equal to "
        "hold_force_N, so the cap stays seated under gravity and comes off under a larger pull. "
        "Declares a <name>_tip site at the nib point, a <name>_cap_grip site on the cap and the "
        "body-frame tip and axis so writing tools can read the working point. The body origin "
        "is on the pen axis, a cap radius above `pos` on the table: the pen rests on its widest part (the cap)."),
    "inputs": [
        {"name": "out_dir", "type": "path", "description": "Folder for the MJCF and <name>_params.json."},
        {"name": "name", "type": "str", "description": "Name. Default marker"},
        {"name": "length", "type": "float", "description": "Overall capped length, back plug to cap end, metres. Default 0.138"},
        {"name": "radius", "type": "float", "description": "Barrel radius, metres. Default 0.0075"},
        {"name": "cap_length", "type": "float", "description": "Cap length, metres. Default 0.036"},
        {"name": "cap_radius", "type": "float", "description": "Cap outer radius, metres. Default 0.0085"},
        {"name": "nib_length", "type": "float", "description": "Nib length, metres. Default 0.010"},
        {"name": "nib_radius", "type": "float", "description": "Nib radius, metres. Default 0.0026"},
        {"name": "collar_length", "type": "float", "description": "Collar the nib comes out of, metres. Default 0.010"},
        {"name": "back_length", "type": "float", "description": "Dark back plug length, metres. Default 0.012"},
        {"name": "cap_clearance", "type": "float", "description": "Cap end beyond the nib point, metres. Default 0.003"},
        {"name": "off_margin", "type": "float", "description": "Extra slide travel past the engaged depth, metres. Default 0.03"},
        {"name": "hold_force_N", "type": "float", "description": "Press-fit retention (slide frictionloss), newtons. Default 1.5"},
        {"name": "cap_damping", "type": "float", "description": "Slide damping, N s/m. Default 0.5"},
        {"name": "barrel_rgba", "type": "list[4]", "description": "Barrel colour. Default [0.93, 0.93, 0.94, 1]"},
        {"name": "cap_rgba", "type": "list[4]", "description": "Cap colour. Default [0.07, 0.07, 0.08, 1]"},
        {"name": "nib_rgba", "type": "list[4]", "description": "Nib, collar and back plug colour. Default [0.05, 0.05, 0.06, 1]"},
        {"name": "mass", "type": "float", "description": "Pen mass without cap, kg. Default 0.012"},
        {"name": "cap_mass", "type": "float", "description": "Cap mass, kg. Default 0.003"},
        {"name": "friction", "type": "list[3]", "description": "Barrel/cap friction. Default [0.85, 0.02, 0.006]"},
        {"name": "nib_friction", "type": "list[3]", "description": "Nib friction (glides on a board). Default [0.16, 0.004, 0.0001]"},
        {"name": "condim", "type": "int", "description": "Contact dimension; 6 resists rolling. Default 6"},
        {"name": "pos", "type": "list[3]", "description": "Footprint centre on the table. Default [0, 0, 0]"},
        {"name": "yaw_deg", "type": "float", "description": "Yaw of the nib direction from +x, degrees. Default 0"},
        {"name": "fixed_base", "type": "bool", "description": "No free joint. Default False"},
        {"name": "detachable_cap", "type": "bool", "description": "The cap comes OFF: a free body welded (<name>_cap_engaged) to an invisible carrier on the press-fit slide; once the slide has travelled the engaged depth the executor's release rule switches the weld off and the cap is free in the hand that pulled it (the scene manifest lists equalities.engaged, joints.lift and params.release_travel_m for the rule). Default False (the cap stays on its slide, parked in front of the nib, and the pen cannot write). Measured: gravity and brief bumps leave the cap seated, a sustained pull at 40% of hold_force_N creeps it ~3 mm/s (MuJoCo's soft dry friction), a pull above it releases the cap in a few hundredths of a second."},
        {"name": "timestep", "type": "float", "description": "Timestep, seconds. Default 0.002"},
        {"name": "include_floor", "type": "bool", "description": "Include a floor. Default True"},
    ],
    "outputs": [
        {"name": "built", "type": "dict",
         "description": "xml, params_json, base_body, cap_body, cap_joint, cap_range, engaged_depth_m "
                        "(q at which the nib is exposed), off_q (cap parked clear), tip_site, tip_local, axis_local, barrel_geom, "
                        "nib_geom, cap_geom, grip_span_local, length_m, axis_height_m."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["marker", "pen", "cap", "uncap", "slide joint", "press fit", "whiteboard", "object reconstruction"],
    "used_in": ["Drawing and Wiping"],
    "status": "tested",
    "entrypoint": "construct_marker_capped",
    "cli": "python -m tool_library.skills.construct_marker_capped out_dir",
}

import json
from pathlib import Path

from tool_library.common.mjcf import FLOOR_XML, LIGHTS_XML, mjcf_header, rgba, vec

DEFAULTS = {
    "name": "marker",
    "length": 0.138, "radius": 0.0075,
    "cap_length": 0.036, "cap_radius": 0.0085,
    "nib_length": 0.010, "nib_radius": 0.0026,
    "collar_length": 0.010, "back_length": 0.012,
    "cap_clearance": 0.003, "off_margin": 0.03,
    "hold_force_N": 1.5, "cap_damping": 0.5,
    "barrel_rgba": [0.93, 0.93, 0.94, 1.0],
    "cap_rgba": [0.07, 0.07, 0.08, 1.0],
    "nib_rgba": [0.05, 0.05, 0.06, 1.0],
    "mass": 0.012, "cap_mass": 0.003,
    "friction": [0.85, 0.02, 0.006],
    "nib_friction": [0.16, 0.004, 0.0001],
    "condim": 6,
    "pos": [0.0, 0.0, 0.0], "yaw_deg": 0.0,
    "fixed_base": False, "detachable_cap": False,
    "timestep": 0.002, "include_floor": True,
}


def _merge(params):
    params = dict(params or {})
    unknown = sorted(set(params) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"unknown parameter(s) {unknown}; allowed: {sorted(DEFAULTS)}")
    p = dict(DEFAULTS)
    p.update(params)
    return p


def _cap_shell_obj(path, x0, x1, r, n=32):
    """The cap's visible surface: the side wall from x0 to x1 and the closed end at x1, open at x0 (the barrel end)."""
    import math
    v = [(x, r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n)) for x in (x0, x1) for k in range(n)]
    v.append((x1, 0.0, 0.0))
    f = []
    for k in range(n):
        a0, a1, b0, b1 = k, (k + 1) % n, n + k, n + (k + 1) % n
        f += [(a0, b0, b1), (a0, b1, a1), (b0, 2 * n, b1)]
    Path(path).write_text("\n".join(["# marker cap: side wall and closed end, open at the barrel end"]
                                     + ["v %.6f %.6f %.6f" % q for q in v] + ["f %d %d %d" % (a + 1, b + 1, c + 1) for a, b, c in f]) + "\n")


def build(params=None, out_dir="."):
    p = _merge(params)
    name = str(p["name"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    L, r = float(p["length"]), float(p["radius"])
    cl, cr = float(p["cap_length"]), float(p["cap_radius"])
    nl, nr = float(p["nib_length"]), float(p["nib_radius"])
    kl, bl = float(p["collar_length"]), float(p["back_length"])
    clr, margin = float(p["cap_clearance"]), float(p["off_margin"])
    for k in ("length", "radius", "cap_length", "cap_radius", "nib_length", "nib_radius",
              "collar_length", "back_length", "mass", "cap_mass"):
        if float(p[k]) <= 0:
            raise ValueError(f"{k} must be positive, got {p[k]}")
    if cr <= r:
        raise ValueError(f"cap_radius ({cr}) must exceed radius ({r}): the cap slides over the barrel")
    if nr >= r:
        raise ValueError(f"nib_radius ({nr}) must be smaller than radius ({r})")
    engaged = kl + nl + clr
    if cl <= engaged:
        raise ValueError(f"cap_length ({cl}) must exceed collar+nib+clearance ({engaged:.4f}) "
                         "or the cap cannot cover the nib")
    if bl + (cl - engaged) + 0.02 >= L:
        raise ValueError(f"length ({L}) too short for back plug + cap overlap + a 2 cm barrel")
    if margin < 0 or float(p["hold_force_N"]) < 0:
        raise ValueError("off_margin and hold_force_N must be non-negative")

    x0 = -L / 2                     # back plug end
    xcap1 = L / 2                   # cap end (when seated)
    xtip = xcap1 - clr              # nib point
    xcol = xtip - nl                # collar / nib junction
    xbar1 = xcol - kl               # barrel end / collar start
    xb0 = x0 + bl                   # barrel start
    xcap0 = xcap1 - cl              # cap open end, overlapping the barrel
    axis_z = cr
    fr = " ".join(str(x) for x in p["friction"])
    gattr = f'friction="{fr}" condim="{int(p["condim"])}" solref="0.005 1" solimp="0.99 0.999 0.001"'
    pos = p["pos"]
    cap_mid = (xcap0 + xcap1) / 2
    slide = (f'<joint name="{name}_cap_slide" type="slide" axis="1 0 0" range="0 {engaged + margin:.5f}" limited="true" '
             f'solreflimit="0.004 1" solimplimit="0.95 0.99 0.001" frictionloss="{float(p["hold_force_N"]):.4f}" '
             f'damping="{float(p["cap_damping"]):.4f}" armature="0.001" ref="0"'
             + ' solreffriction="0.004 1" solimpfriction="0.999 0.9999 0.001"/>')
    if p["detachable_cap"]:
        # a 3 g cap on a massless carrier: MuJoCo's soft dry-friction constraint scales with the dof's inertia, so the
        # press-fit let a 0.6 N pull slide it off; armature gives the slide the inertia the friction needs
        slide = slide.replace('armature="0.001"', 'armature="0.05"')
    # the cap collides as a solid cylinder (the fingers grip its outside, the press-fit rule acts on it) but is drawn as
    # what it is, a hollow shell open toward the barrel: a solid visual would sit over the collar and the nib
    _cap_shell_obj(out_dir / f"{name}_cap_shell.obj", xcap0, xcap1, cr)
    shell = (f'<geom name="{name}_cap_shell" type="cylinder" fromto="{xcap0:.5f} 0 0 {xcap1:.5f} 0 0" size="{cr:.5f}" '
             f'rgba="{rgba(p["cap_rgba"])}" mass="{float(p["cap_mass"]):.5f}" group="3" {gattr}/>'
             f'<geom name="{name}_cap_visual" type="mesh" mesh="{name}_cap_shell" contype="0" conaffinity="0" density="0" '
             f'group="1" rgba="{rgba(p["cap_rgba"])}"/>'
             f'<site name="{name}_cap_grip" pos="{cap_mid:.5f} 0 0" size="0.0015" rgba="0 1 0 0.35"/>')
    if not p["detachable_cap"]:
        cap_inner = f'<body name="{name}_cap" pos="0 0 0">{slide}{shell}</body>'
        cap_outer = ""
    else:
        # the cap is its own free body; the press-fit lives on a massless carrier the cap is welded to
        cap_inner = (f'<body name="{name}_cap_carrier" pos="0 0 0">{slide}'
                     f'<inertial pos="0 0 0" mass="0.0005" diaginertia="1e-8 1e-8 1e-8"/></body>')
        cap_outer = (f'\n    <body name="{name}_cap" pos="{vec((pos[0], pos[1], pos[2] + axis_z))}" euler="0 0 {p["yaw_deg"]}">'
                     f'<freejoint name="{name}_cap_free"/>{shell}</body>')
    eq_xml = (f'\n  <equality><weld name="{name}_cap_engaged" body1="{name}_cap_carrier" body2="{name}_cap" solref="0.004 1"/></equality>'
              f'\n  <contact><exclude body1="{name}" body2="{name}_cap"/></contact>') if p["detachable_cap"] else ""
    body = f"""
    <body name="{name}" pos="{vec((pos[0], pos[1], pos[2] + axis_z))}" euler="0 0 {p['yaw_deg']}">
{'' if p["fixed_base"] else f'      <freejoint name="{name}_free"/>'}
      <geom name="{name}_barrel" type="cylinder" fromto="{xb0:.5f} 0 0 {xbar1:.5f} 0 0" size="{r:.5f}"
            rgba="{rgba(p['barrel_rgba'])}" mass="{float(p['mass']):.5f}" {gattr}/>
      <geom name="{name}_back" type="cylinder" fromto="{x0:.5f} 0 0 {xb0:.5f} 0 0" size="{0.85 * r:.5f}"
            rgba="{rgba(p['nib_rgba'])}" mass="0.00001" {gattr}/>
      <geom name="{name}_collar" type="cylinder" fromto="{xbar1:.5f} 0 0 {xcol:.5f} 0 0" size="{0.72 * r:.5f}"
            rgba="{rgba(p['nib_rgba'])}" mass="0.00001" {gattr}/>
      <geom name="{name}_nib" type="cylinder" fromto="{xcol:.5f} 0 0 {xtip:.5f} 0 0" size="{nr:.5f}"
            rgba="{rgba(p['nib_rgba'])}" mass="0.00001" condim="{int(p['condim'])}"
            friction="{' '.join(str(x) for x in p['nib_friction'])}" solref="0.005 1" solimp="0.99 0.999 0.001"/>
      <site name="{name}_tip" pos="{xtip:.5f} 0 0" size="0.0015" rgba="1 0 0 0.35"/>
      <site name="{name}_back_end" pos="{x0:.5f} 0 0" size="0.0015" rgba="0 0 1 0.35"/>
      {cap_inner}
    </body>{cap_outer}"""
    xml = mjcf_header(f"{name}_capped_marker", timestep=p["timestep"]) + f"""
  <asset><mesh name="{name}_cap_shell" file="{name}_cap_shell.obj"/></asset>
  <worldbody>{LIGHTS_XML}{FLOOR_XML if p["include_floor"] else ""}{body}
  </worldbody>{eq_xml}
</mujoco>
"""
    xml_path = out_dir / f"{name}.xml"
    xml_path.write_text(xml)
    pj = out_dir / f"{name}_params.json"
    pj.write_text(json.dumps(p, indent=1))
    return {
        "xml": str(xml_path), "params_json": str(pj),
        "style": "marker", "base_body": name,
        "cap_body": f"{name}_cap", "cap_joint": f"{name}_cap_slide",
        "cap_range": [0.0, round(engaged + margin, 5)],
        "engaged_depth_m": round(engaged, 5),
        "off_q": round(engaged + margin, 5),
        "barrel_geom": f"{name}_barrel", "nib_geom": f"{name}_nib", "cap_geom": f"{name}_cap_shell",
        "tip_site": f"{name}_tip", "cap_grip_site": f"{name}_cap_grip",
        "tip_local": [round(xtip, 5), 0.0, 0.0], "axis_local": [1.0, 0.0, 0.0],
        "grip_span_local": [round(xb0, 5), round(xbar1, 5)],
        "length_m": round(L, 4), "radius_m": round(r, 5), "axis_height_m": round(axis_z, 5),
        "hold_force_N": float(p["hold_force_N"]),
        **({"detachable_cap": True, "cap_free_body": f"{name}_cap", "cap_carrier": f"{name}_cap_carrier",
            "equalities": {"engaged": f"{name}_cap_engaged"}, "joints": {"lift": f"{name}_cap_slide"},
            "params": {"release_travel_m": round(engaged, 5)}} if p["detachable_cap"] else {}),
    }


def construct_marker_capped(out_dir, **params):
    """Build a capped marker; any DEFAULTS key may be set, unknown keys raise ValueError."""
    return build(params, out_dir)


if __name__ == "__main__":
    import sys
    print(json.dumps(construct_marker_capped(sys.argv[1] if len(sys.argv) > 1 else "."), indent=1))
