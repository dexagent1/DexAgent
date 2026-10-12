"""CYLINDRICAL CELL (AA battery) — a free rigid body with real terminals.

Trivial geometry, but the two ENDS are the whole task: the flat negative
face is what a holder's spring pushes on, the positive BUTTON is what must
end up against the contact bump, and telling them apart in a render is how
you tell a correctly-inserted battery from a backwards one. So the builder's
job is (a) honest terminal geometry — the button is a real colliding
cylinder, proud of the top face by its real ~1 mm — and (b) the two-tone
jacket (black sleeve + copper crown + copper base ring) that makes polarity
readable at a glance, matching the photographed cell.

Local frame: axis +z, NEGATIVE face at z=0, POSITIVE button tip at
z=`length`. Free body (own freejoint) — stand it, lie it, drop it.
"""
from __future__ import annotations

from pathlib import Path

from .mjcf import (FLOOR_XML, LIGHTS_XML, free_camera, load, merge_params,
                     mjcf_header, rgba, rollout, save_params, save_stills,
                     vec, write_video)

# ---------------------------------------------------------------------------
DEFAULTS = {
    "name": "battery",

    # ---- size: a standard AA (IEC LR6) ----
    "diameter":      0.0145,
    "length":        0.0505,   # INCLUDING the positive button
    "button_radius": 0.0027,
    "button_height": 0.001,

    # ---- the photographed jacket: black sleeve, copper crown ----
    "crown_frac":    0.30,     # top fraction of the sleeve that is copper
    "base_ring_h":   0.0012,   # thin copper ring at the negative rim
    "body_rgba":     [0.055, 0.055, 0.06, 1.0],
    "crown_rgba":    [0.72, 0.45, 0.20, 1.0],
    "terminal_rgba": [0.75, 0.76, 0.78, 1.0],
    "body_specular": 0.30, "body_shininess": 0.55,   # glossy shrink-wrap
    "metal_specular": 0.75, "metal_shininess": 0.9,

    # ---- dynamics ----
    "mass":     0.023,         # an alkaline AA
    "friction": [0.85, 0.008, 0.0002],   # >= 0.85 with the holder's own
                               #   contact friction: the tilted press on the
                               #   spring plate must sit INSIDE the friction
                               #   cone at the ~28 deg entry tilt the slot
                               #   geometry dictates

    # ---- placement (standalone build only) ----
    "pos":       [0.0, 0.0, 0.0],   # of the NEGATIVE face centre
    "upright":   True,              # False -> built lying along +x
    "include_floor": True,
    "timestep":  0.002,
}


def _neg_ring(name: str, p: dict, r_sph: float = 0.0012,
              ring_r: float = 0.003, proud: float = 0.00015) -> str:
    """Three collision spheres on the negative face (see the comment in the
    XML above). ring_r stays INSIDE a holder spring's tip plate so a flat
    press lands on spheres, never on the disc rim."""
    import numpy as np
    out = ""
    for k in range(3):
        a = 2 * np.pi * k / 3
        out += f"""
      <geom type="sphere" size="{r_sph:.6f}"
            pos="{ring_r*np.cos(a):.6f} {ring_r*np.sin(a):.6f} {r_sph - proud:.6f}"
            mass="0" rgba="{rgba(p['terminal_rgba'])}" group="3"
            friction="{vec(p['friction'])}"/>"""
    return out


def build(params: dict | None = None, out_dir: str | Path = ".") -> dict:
    p = merge_params(DEFAULTS, params)
    name = str(p["name"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    r = float(p["diameter"]) / 2.0
    Lb = float(p["length"]) - float(p["button_height"])   # the can itself
    crown = min(max(float(p["crown_frac"]), 0.0), 0.95) * Lb
    ring = float(p["base_ring_h"])
    quat = "1 0 0 0" if p["upright"] else "0.7071068 0 0.7071068 0"
    floor = FLOOR_XML if p["include_floor"] else ""
    # one colliding can + one colliding button; the jacket colours are
    # visual-only shells a hair fatter so they never z-fight the can
    rv = r + 5e-5
    xml = mjcf_header(f"{name}_cell", timestep=p["timestep"]) + f"""
  <asset>
    <material name="{name}_black" rgba="{rgba(p['body_rgba'])}"
              specular="{p['body_specular']}" shininess="{p['body_shininess']}"/>
    <material name="{name}_copper" rgba="{rgba(p['crown_rgba'])}"
              specular="{p['metal_specular']}" shininess="{p['metal_shininess']}"
              reflectance="0.25"/>
    <material name="{name}_steel" rgba="{rgba(p['terminal_rgba'])}"
              specular="{p['metal_specular']}" shininess="{p['metal_shininess']}"
              reflectance="0.3"/>
  </asset>
  <worldbody>{LIGHTS_XML}{floor}
    <body name="{name}" pos="{vec(p['pos'])}" quat="{quat}">
      <freejoint name="{name}_free"/>
      <geom name="{name}_can" type="cylinder"
            size="{r:.6f} {Lb/2:.6f}" pos="0 0 {Lb/2:.6f}"
            mass="{float(p['mass']):.4f}" material="{name}_black"
            rgba="{rgba(p['body_rgba'])}" friction="{vec(p['friction'])}"/>
      <!-- TERMINALS COLLIDE AS SPHERES, faces stay visual. A flat cylinder
           end pressed on a flat plate is a degenerate libccd pair — the
           single contact point wanders across the patch every step and the
           torque noise rattles the cell against any holder (measured: the
           press bounced at ~3 Hz however hard it was damped). sphere-box
           is an analytic pair. The + button is one sphere protruding its
           real height; the - face is a tripod ring poking 0.15 mm proud,
           which also keeps an upright cell a stable three-point stand. -->
      <geom name="{name}_button" type="sphere"
            size="{float(p['button_radius']):.6f}"
            pos="0 0 {Lb + float(p['button_height']) - float(p['button_radius']):.6f}"
            mass="0" rgba="{rgba(p['terminal_rgba'])}" group="3"
            friction="{vec(p['friction'])}"/>
      <geom type="cylinder"
            size="{float(p['button_radius']):.6f} {float(p['button_height'])/2:.6f}"
            pos="0 0 {Lb + float(p['button_height'])/2:.6f}" mass="0"
            material="{name}_steel" rgba="{rgba(p['terminal_rgba'])}"
            contype="0" conaffinity="0"/>{_neg_ring(name, p)}
      <geom type="cylinder" size="{rv:.6f} {crown/2:.6f}"
            pos="0 0 {Lb - crown/2:.6f}" material="{name}_copper"
            rgba="{rgba(p['crown_rgba'])}" contype="0" conaffinity="0"
            mass="0"/>
      <geom type="cylinder" size="{rv:.6f} {ring/2:.6f}"
            pos="0 0 {ring/2:.6f}" material="{name}_copper"
            rgba="{rgba(p['crown_rgba'])}" contype="0" conaffinity="0"
            mass="0"/>
      <geom type="cylinder" size="{r*0.55:.6f} 0.0001"
            pos="0 0 0.0001" material="{name}_steel"
            rgba="{rgba(p['terminal_rgba'])}" contype="0" conaffinity="0"
            mass="0"/>
    </body>
  </worldbody>
</mujoco>
"""
    xml_path = out_dir / f"{name}.xml"
    xml_path.write_text(xml)
    save_params(out_dir / f"{name}_params.json", p)
    return {"xml": str(xml_path),
            "params_json": str(out_dir / f"{name}_params.json"),
            "body": name, "free_joint": f"{name}_free",
            "can_geom": f"{name}_can", "button_geom": f"{name}_button",
            "radius_m": r, "length_m": float(p["length"]),
            # where a tilted press actually touches: the negative-face
            # contact ring (see _neg_ring), NOT the face rim
            "neg_contact_ring_r_m": 0.003,
            "note": "axis +z: NEGATIVE flat face at z=0, POSITIVE button "
                    "tip at z=length"}


def rebuild(params_json: str | Path, out_dir: str | Path | None = None):
    from .mjcf import load_params
    return build(load_params(params_json), out_dir or Path(params_json).parent)


def demo(built: dict, out_mp4: str | Path, fps: int = 30) -> dict:
    """Tip it over and let it roll out — mass, friction and both terminals
    on camera."""
    import mujoco
    m = load(built["xml"])
    d = mujoco.MjData(m)
    bid = m.body(built["body"]).id

    def control(t, model, data):
        data.xfrc_applied[:] = 0.0
        if 0.8 < t < 1.0:
            data.xfrc_applied[bid, 0] = 0.35     # a finger-flick at the top
            data.xfrc_applied[bid, 4] = 0.004
        return "flick -> topple -> roll" if t > 0.8 else "standing"

    cam = free_camera([0.03, 0, 0.02], 0.28, 110, -18)
    frames = rollout(m, d, seconds=4.0, fps=fps, control=control, camera=cam,
                     width=640, height=480)
    out = write_video(frames, out_mp4, fps=fps)
    out["stills"] = save_stills(frames, Path(out_mp4).parent,
                                Path(out_mp4).stem, n=4)
    return out


# ---------------------------------------------------------------------------
