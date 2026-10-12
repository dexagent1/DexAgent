"""BATTERY CASE — an open cell holder whose slots contain REAL springs.

THE POINT
---------
The one thing that makes "insert the battery" a manipulation task instead of
a pick-and-place is the spring: you must press the negative end into it,
shorten it past the contact bump, lay the cell down and let the stored
energy seat it — and if you let go early the spring throws the battery back
out. So the springs here are not decoration: each slot embeds a
`tool_library.common.spring` chain (slide-joint coils, measured stiffness, coil-bound
hard stop), and the INTERIOR LENGTH of the case is DERIVED from the battery
plus the spring at its chosen seated preload. A case built for a different
cell or a different spring is different numbers, same code.

Geometry (asset local frame): slots run along +x, z=0 is the footprint on
the table. Each slot names which end its spring lives at ("-x"/"+x" —
diagonal in a real 2-cell holder, because the cells sit antiparallel). The
opposite end wall carries the positive contact BUMP, proud enough that a
battery cannot reach it without compressing the spring first: seating is
gated by the mechanism, not by scripting.

`probe()` is the build's physics receipt, run headless in seconds:
  seated   a battery placed at the computed seated pose, spring at its
           seated squeeze, must STAY (drift < 1 mm over 1.5 s) with the
           spring still pushing — the holder actually holds.
  eject    the same battery pressed 2 mm deeper and released with its
           positive end lifted clear of the bump must be THROWN out along
           +axis — the spring actually springs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import spring as SPR
from .mjcf import (FLOOR_XML, LIGHTS_XML, free_camera, load, merge_params,
                     mjcf_header, rgba, save_params, vec)

# ---------------------------------------------------------------------------
DEFAULTS = {
    "name": "battery_case",

    # ---- the cell this case is FOR (drives every interior number) ----
    "battery_diameter": 0.0145,
    "battery_length":   0.0505,

    # ---- slots ----
    "n_slots":        2,
    "springs_at":     ["-x", "+x"],   # per slot, slot 0 = smallest y.
                                      #   Diagonal = antiparallel cells,
                                      #   like the photographed holder
    "slot_clearance": 0.0016,         # slot width - battery diameter
    "seated_preload": 0.60,           # fraction of spring travel used up
                                      #   when seated: the push that HOLDS
                                      #   the cell (leave headroom to get
                                      #   past the bump: <= ~0.75)

    # ---- the spring in every slot (tool_library.common.spring keys) ----
    "spring_stiffness":   250.0,
    "spring_free_length": 0.0115,
    "spring_coil_radius": 0.0042,
    "spring_wire_radius": 0.00045,
    "spring_n_coils":     5,

    # ---- positive contact ----
    # The COLLIDING contact is the flat PLATE (large face, unambiguous
    # normal). The domed bump on it is VISUAL ONLY: two small coaxial
    # cylinder faces (button vs bump) are a degenerate libccd pair whose
    # contact normal wanders, and a seated cell crept over a colliding bump
    # at ~1 mm/s until the spring shot it out (measured). Axial retention
    # comes from plate + spring, exactly like the real holder — removal is
    # levering the cell UP, which nothing prevents.
    "bump_radius": 0.0035,
    "bump_proud":  0.0012,            # visual dome height above the wall
    "plate_t":     0.0005,            # the colliding contact face

    # ---- shell ----
    "wall":        0.0018,
    "end_wall":    0.0020,
    "divider":     0.0020,
    "floor_t":     0.0016,
    "height":      0.0145,
    "body_rgba":   [0.06, 0.06, 0.065, 1.0],
    "body_specular": 0.35, "body_shininess": 0.65,   # glossy ABS
    "metal_rgba":  [0.75, 0.76, 0.78, 1.0],
    "density":     1250.0,            # ABS -> the shell's mass
    "friction":    [0.7, 0.008, 0.0002],
    "fixed_base":  False,             # True bolts it to the world

    # ---- the photographed leads (visual only, rigid with the shell) ----
    "with_wires":  True,
    "wire_radius": 0.0008,
    "wire_length": 0.075,             # how far the loop wanders on the table

    "include_floor": True,
    "timestep": 0.002,
}


# ---------------------------------------------------------------------------
def _slot_layout(p: dict, spr: dict) -> dict:
    """Every derived interior number, one place."""
    L_bat = float(p["battery_length"])
    travel = spr["travel_m"]
    c_seat = float(p["seated_preload"]) * travel
    if travel - c_seat < 1e-3:
        raise ValueError(
            f"seated_preload {p['seated_preload']} leaves only "
            f"{(travel-c_seat)*1e3:.2f} mm of spare squeeze over the seated "
            f"state — no room to manoeuvre the cell in. Lower the preload "
            f"or raise spring travel.")
    L_int = L_bat + float(p["plate_t"]) + spr["tip_free_z"] - c_seat
    w_slot = float(p["battery_diameter"]) + float(p["slot_clearance"])
    n = int(p["n_slots"])
    W_int = n * w_slot + (n - 1) * float(p["divider"])
    return {
        "L_int": L_int, "w_slot": w_slot, "W_int": W_int,
        "L_out": L_int + 2 * float(p["end_wall"]),
        "W_out": W_int + 2 * float(p["wall"]),
        "c_seat": c_seat,
        "z_axis": float(p["floor_t"]) + float(p["battery_diameter"]) / 2,
        "slot_y": [(-W_int / 2 + w_slot / 2 + i * (w_slot + float(p["divider"])))
                   for i in range(n)],
    }


def build(params: dict | None = None, out_dir: str | Path = ".") -> dict:
    p = merge_params(DEFAULTS, params)
    name = str(p["name"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spring_p = {
        "stiffness": float(p["spring_stiffness"]),
        "free_length": float(p["spring_free_length"]),
        "coil_radius": float(p["spring_coil_radius"]),
        "wire_radius": float(p["spring_wire_radius"]),
        "n_coils": int(p["spring_n_coils"]),
        "timestep": float(p["timestep"]),
    }
    lay = _slot_layout(p, SPR.derived(SPR.merge_params(SPR.DEFAULTS, spring_p)))
    h, ft = float(p["height"]), float(p["floor_t"])
    zc = lay["z_axis"]

    assets, bodies, slots = "", "", []
    ends = list(p["springs_at"])
    # a slot without an entry alternates with its neighbour, like a real holder
    while len(ends) < len(lay["slot_y"]):
        ends.append("+x" if ends and ends[-1] == "-x" else "-x")
    for i, y in enumerate(lay["slot_y"]):
        end = str(ends[i])
        if end not in ("-x", "+x"):
            raise ValueError(f"springs_at[{i}] must be '-x' or '+x'")
        sgn = -1.0 if end == "-x" else 1.0          # wall the spring sits on
        frag = SPR.emit(spring_p, out_dir, f"{name}_s{i}spring")
        assets += frag["assets_xml"]
        # +z of the spring frame must point INTO the slot
        quat = ("0.7071068 0 0.7071068 0" if sgn < 0
                else "0.7071068 0 -0.7071068 0")
        x_mount = sgn * lay["L_int"] / 2
        bodies += f"""
    <body name="{name}_s{i}spring_mount" pos="{x_mount:.6f} {y:.6f} {zc:.6f}"
          quat="{quat}">{frag['chain_xml']}
    </body>
    <geom type="box" size="0.0004 {lay['w_slot']*0.36:.6f} {lay['w_slot']*0.36:.6f}"
          pos="{x_mount - sgn*0.0004:.6f} {y:.6f} {zc:.6f}"
          material="{name}_steel" rgba="{rgba(p['metal_rgba'])}"
          contype="0" conaffinity="0" mass="0"/>
    <geom name="{name}_s{i}plate" type="box"
          size="{float(p['plate_t'])/2:.6f} {lay['w_slot']*0.40:.6f} {(h-ft)*0.38:.6f}"
          pos="{-x_mount + sgn*float(p['plate_t'])/2:.6f} {y:.6f} {zc:.6f}"
          material="{name}_steel" rgba="{rgba(p['metal_rgba'])}" mass="0"
          friction="0.3 0.005 0.0001"/>
    <!-- 0.3, not the ABS shell's 0.7: nickel button on nickel plate is
         slick, and at 0.7 the seating button HANGS on the plate face
         instead of sliding down it (the cell parks cocked ~5.5 deg) -->
    <geom name="{name}_s{i}bump" type="cylinder"
          fromto="{-x_mount + sgn*float(p['plate_t']):.6f} {y:.6f} {zc:.6f} {-x_mount + sgn*float(p['bump_proud']):.6f} {y:.6f} {zc:.6f}"
          size="{float(p['bump_radius']):.6f}" material="{name}_steel"
          rgba="{rgba(p['metal_rgba'])}" mass="0"
          contype="0" conaffinity="0"/>"""
        # seated pose: negative face on the seated plate, button on the bump.
        # The spring grows INTO the slot, i.e. along -sgn.
        x_plate_seated = x_mount - sgn * (frag["tip_free_z"] - lay["c_seat"])
        x_center = x_plate_seated - sgn * float(p["battery_length"]) / 2
        slots.append({
            "index": i, "spring_at": end, "y": round(y, 6),
            "axis_into_spring": [sgn, 0.0, 0.0],
            "z_axis": round(zc, 6),
            "spring_joints": frag["joints"],
            "spring_tip_body": frag["tip_body"],
            "spring_travel_m": frag["travel_m"],
            "seated_compression_m": round(lay["c_seat"], 6),
            "seated_center": [round(x_center, 6), round(y, 6), round(zc, 6)],
            # the battery BODY origin is its negative-face centre (see
            # tool_library.common.battery_cell): this is what a free joint is set to
            "seated_freejoint_pos": [round(x_plate_seated, 6), round(y, 6),
                                     round(zc, 6)],
            "seated_battery_quat_wxyz": ([0.7071068, 0.0, 0.7071068, 0.0]
                                         if sgn < 0 else
                                         [0.7071068, 0.0, -0.7071068, 0.0]),
            "plate_free_x": round(x_mount - sgn * frag["tip_free_z"], 6),
            "contact_plate_geom": f"{name}_s{i}plate",
            "contact_face_x": round(-x_mount + sgn * float(p["plate_t"]), 6),
            "bump_tip_x": round(-x_mount + sgn * float(p["bump_proud"]), 6),
            "mouth_center": [round(x_center, 6), round(y, 6),
                             round(h + 0.5 * float(p["battery_diameter"]), 6)],
        })

    # ---- shell boxes (all colliding, density -> mass) ----
    Li, Wo, Wi = lay["L_int"], lay["W_out"], lay["W_int"]
    ew, wl, dv = float(p["end_wall"]), float(p["wall"]), float(p["divider"])
    shell = f"""
    <geom name="{name}_floor" type="box"
          size="{lay['L_out']/2:.6f} {Wo/2:.6f} {ft/2:.6f}"
          pos="0 0 {ft/2:.6f}" material="{name}_abs"
          rgba="{rgba(p['body_rgba'])}" density="{p['density']}"
          friction="{vec(p['friction'])}"/>"""
    # every shell geom is NAMED: hand assets ship contype 0, so hand-case
    # contact exists only through explicit <pair> entries, and a pair needs
    # both geom names — an unnamed wall is a wall a finger passes through
    for sx, tag in ((-1, "a"), (1, "b")):
        shell += f"""
    <geom name="{name}_endwall_{tag}" type="box"
          size="{ew/2:.6f} {Wo/2:.6f} {(h-ft)/2:.6f}"
          pos="{sx*(Li/2+ew/2):.6f} 0 {(h+ft)/2:.6f}" material="{name}_abs"
          rgba="{rgba(p['body_rgba'])}" density="{p['density']}"
          friction="{vec(p['friction'])}"/>
    <geom name="{name}_sidewall_{tag}" type="box"
          size="{Li/2:.6f} {wl/2:.6f} {(h-ft)/2:.6f}"
          pos="0 {sx*(Wi/2+wl/2):.6f} {(h+ft)/2:.6f}" material="{name}_abs"
          rgba="{rgba(p['body_rgba'])}" density="{p['density']}"
          friction="{vec(p['friction'])}"/>"""
    for i in range(int(p["n_slots"]) - 1):
        yd = lay["slot_y"][i] + lay["w_slot"] / 2 + dv / 2
        shell += f"""
    <geom name="{name}_divider_{i}" type="box"
          size="{Li/2:.6f} {dv/2:.6f} {(h-ft)/2:.6f}"
          pos="0 {yd:.6f} {(h+ft)/2:.6f}" material="{name}_abs"
          rgba="{rgba(p['body_rgba'])}" density="{p['density']}"
          friction="{vec(p['friction'])}"/>"""

    # ---- the leads (visual only): red+black pair arcing out to a plug ----
    wires = ""
    if p["with_wires"]:
        r_w = float(p["wire_radius"])
        exit_y = lay["slot_y"][0]
        t = np.linspace(0, 1, 12)
        # a lazy S on the tabletop, leaving the spring end of slot 0
        bez = lambda a, b, c, dpt: ((1-t)**3)[:, None]*a + \
            (3*(1-t)**2*t)[:, None]*b + (3*(1-t)*t**2)[:, None]*c + \
            (t**3)[:, None]*dpt
        x0 = -lay["L_out"] / 2
        Lw = float(p["wire_length"])
        for k, (col, dy) in enumerate((([0.85, 0.08, 0.08, 1], 0.004),
                                       ([0.05, 0.05, 0.05, 1], -0.004))):
            a = np.array([x0, exit_y + dy, 0.006])
            b = np.array([x0 - 0.45*Lw, exit_y + dy - 0.01, r_w])
            c = np.array([x0 - 0.25*Lw, exit_y - 0.055, r_w])
            dpt = np.array([x0 - 0.55*Lw, exit_y - 0.065 + 0.01*k, r_w])
            pts = bez(a, b, c, dpt)
            for j in range(len(pts) - 1):
                wires += f"""
    <geom type="capsule" fromto="{vec(pts[j])} {vec(pts[j+1])}"
          size="{r_w:.6f}" rgba="{rgba(col)}" contype="0" conaffinity="0"
          mass="0"/>"""
        px, py = x0 - 0.55*Lw, exit_y - 0.06 + 0.005
        wires += f"""
    <geom type="box" size="0.0045 0.003 0.0028" pos="{px:.6f} {py:.6f} 0.0028"
          quat="0.9239 0 0 0.3827" rgba="0.92 0.92 0.9 1"
          contype="0" conaffinity="0" mass="0"/>"""

    root_joint = "" if p["fixed_base"] else f'<freejoint name="{name}_free"/>'
    floor = FLOOR_XML if p["include_floor"] else ""
    xml = mjcf_header(f"{name}_asset", timestep=p["timestep"]) + f"""
  <asset>{assets}
    <material name="{name}_abs" rgba="{rgba(p['body_rgba'])}"
              specular="{p['body_specular']}" shininess="{p['body_shininess']}"/>
    <material name="{name}_steel" rgba="{rgba(p['metal_rgba'])}"
              specular="0.75" shininess="0.9" reflectance="0.3"/>
  </asset>
  <worldbody>{LIGHTS_XML}{floor}
    <body name="{name}" pos="0 0 0">
      {root_joint}{shell}{wires}{bodies}
    </body>
  </worldbody>
</mujoco>
"""
    xml_path = out_dir / f"{name}.xml"
    xml_path.write_text(xml)
    save_params(out_dir / f"{name}_params.json", p)
    return {"xml": str(xml_path),
            "params_json": str(out_dir / f"{name}_params.json"),
            "body": name,
            "outer_lwh_m": [round(lay["L_out"], 5), round(lay["W_out"], 5),
                            round(h, 5)],
            "slots": slots,
            "note": ("slot axes run along local x, z=0 on the table. "
                     "seated_center/quat are where a correctly inserted "
                     "battery ENDS UP (negative face on the spring plate at "
                     "seated squeeze, button on the bump). Read a spring "
                     "live with tool_library.common.spring.compression(model, data, "
                     "slot['spring_joints']).")}


def rebuild(params_json: str | Path, out_dir: str | Path | None = None):
    from .mjcf import load_params
    return build(load_params(params_json), out_dir or Path(params_json).parent)


# ---------------------------------------------------------------------------
def _attach_battery(case_xml: str, battery_built: dict, prefix: str = "bat_"):
    """case + one free battery in one compiled model (MjSpec attach, the
    same mechanism stage_on_bench places assets with)."""
    import mujoco
    spec = mujoco.MjSpec.from_file(str(case_xml))
    # bench-ready assets carry no floor of their own — the probe world needs
    # one or everything simply falls for the whole test
    if not any(g.type == mujoco.mjtGeom.mjGEOM_PLANE for g in spec.geoms):
        g = spec.worldbody.add_geom()
        g.type = mujoco.mjtGeom.mjGEOM_PLANE
        g.size = [2.0, 2.0, 0.05]
    # the battery must be built include_floor=False (its lights are fine —
    # the prefix keeps their names distinct)
    child = mujoco.MjSpec.from_file(str(battery_built["xml"]))
    f = spec.worldbody.add_frame()
    spec.attach(child, prefix=prefix, frame=f)
    m = spec.compile()
    d = mujoco.MjData(m)
    return m, d


def probe(built: dict, battery_built: dict, slot: int = 0,
          out_json: str | Path | None = None) -> dict:
    """The physics receipt: seated battery STAYS, released battery EJECTS."""
    import mujoco
    from scipy.spatial.transform import Rotation as Rot
    s = built["slots"][slot]
    m, d = _attach_battery(built["xml"], battery_built)
    jid = m.joint(f"bat_{battery_built['free_joint']}").id
    qadr = m.jnt_qposadr[jid]
    bid = m.body(f"bat_{battery_built['body']}").id
    sgn = s["axis_into_spring"][0]

    def place(depth_extra=0.0, tilt_deg=0.0):
        """Negative face on the spring plate (squeezed seated+extra); a
        positive tilt raises the button end, as a mid-insertion hand does."""
        mujoco.mj_resetData(m, d)
        pos = np.array(s["seated_freejoint_pos"], float)
        pos[0] += sgn * depth_extra          # deeper INTO the spring
        d.qpos[qadr:qadr+3] = pos
        phi = np.radians(90.0 if sgn < 0 else -90.0) \
            + np.radians(-tilt_deg if sgn < 0 else tilt_deg)
        q = Rot.from_euler("y", phi).as_quat()      # xyzw
        d.qpos[qadr+3:qadr+7] = [q[3], q[0], q[1], q[2]]
        per = (s["seated_compression_m"] + depth_extra) / len(s["spring_joints"])
        for j in s["spring_joints"]:
            d.qpos[m.joint(j).qposadr[0]] = -per
        mujoco.mj_forward(m, d)

    # --- seated: must stay ---
    place()
    p0 = d.xpos[bid].copy()
    for _ in range(int(1.5 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    drift = float(np.linalg.norm(d.xpos[bid] - p0))
    c_end = SPR.compression(m, d, s["spring_joints"])
    seated_ok = drift < 1e-3 and c_end > 0.5 * s["seated_compression_m"]

    # --- eject: pressed 2 mm deeper, button end tilted up clear of the
    # bump (the mid-insertion pose), then released: the spring must throw
    # the battery back out along the slot ---
    place(depth_extra=0.002, tilt_deg=25.0)
    x0 = float(d.xpos[bid][0])
    for _ in range(int(0.8 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    moved = float((d.xpos[bid][0] - x0) * -sgn)
    c_after = SPR.compression(m, d, s["spring_joints"])
    # ejected = the spring re-extended fully (it is no longer holding
    # anything) AND the battery was thrown down the slot. The absolute
    # distance is capped by the slot itself (~8 mm of free slide before the
    # far wall), so the spring letting go is the primary signal.
    eject_ok = moved > 0.006 and c_after < 1e-3

    out = {"slot": slot,
           "seated": {"drift_mm": round(drift * 1e3, 2),
                      "spring_holds_mm": round(c_end * 1e3, 2),
                      "ok": bool(seated_ok)},
           "eject": {"thrown_mm": round(moved * 1e3, 1),
                     "spring_after_mm": round(c_after * 1e3, 2),
                     "ok": bool(eject_ok)},
           "ok": bool(seated_ok and eject_ok)}
    if out_json:
        Path(out_json).write_text(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------------------
