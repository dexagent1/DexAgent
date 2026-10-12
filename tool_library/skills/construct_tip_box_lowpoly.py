"""Construct pipette tip box (low-poly collision): the hinged-lid tip box of construct_tip_box with a collision model a
hand scene can afford and a physics that matches what is drawn.

A fork of construct_tip_box (tool_library/common/pipettekit.py) that also names the free tips' bodies (`tip_names`), so a
scene whose task lists the tip as an object of its own can give it a body the library's name resolver finds
('<prefix>_tipbox_tip' ends in '_tip'). What changes and why:
  * the tray of tip collars collides as a few merged slabs (the solid regions between the free wells, rising to just
    under the collars' tops) instead of one box per cell, so the moulded collars that are only drawn lie inside physics;
  * a free well is a square hole through those slabs that cages the real tip's collar, plus the platform column its
    foot stands on (no separate shaft cage);
  * the lid's mass and inertia come from its own plates (no hand-written <inertial>);
  * the free tips carry a lighter snap ring (lip_segments tangential capsules, default 12) and a merged fine shaft.
"""
from __future__ import annotations

TOOL = {
    "name": "construct_tip_box_lowpoly",
    "title": "Construct pipette tip box (low-poly collision, named tips)",
    "type": "skill",
    "category": "Object Reconstruction",
    "summary": "A hinged-lid pipette tip box (free tips named by the caller) whose tray of tips collides as a few merged slabs with caged sockets for "
               "the real (free) tips, a lid with inertia from its own plates, and free tips with a lighter snap ring: "
               "the construct_tip_box mechanism at a contact budget a hand scene can afford.",
    "description": (
        "A fork of construct_tip_box, with caller-named free tips (tip_names), for scenes a robot hand works in. The original builds one collision box per tray "
        "cell (96+), draws the standing tips as visual-only stubs that stick 15 mm out of the physics, gives the lid a "
        "hand-written 1e-7 kg m^2 inertia and puts 37 collision geoms on every free tip, so a hand near the box is "
        "paired with 150 geoms and could sink into the drawn tips without contact. Here: the base is four walls and "
        "a floor (a hollow box, as box_enclosure_check expects); the tray is the region of the grid minus the free "
        "wells, cut into the fewest axis-aligned slabs (bands of rows split around each free well's square hole) that "
        "rise from the tray's underside to `slab_rise` above the tray top (default: 4 mm under the tips' collar tops, so "
        "the drawn collars stick out less than a 5 mm visual tolerance); the tray margins are part of those slabs; "
        "every free well keeps the platform column its tip's foot stands on and the slab hole (the collar's polygon "
        "plus `hole_clear`) cages the collar; the drawn collars of the other wells are visual cylinders on the slab "
        "top. The lid is the same friction hinge (with a stiff friction impedance, so it holds its own weight) and the same plates; MuJoCo infers its mass and inertia from them at "
        "`lid_density` (lid_end_walls_collide False makes the two short skirt walls visual only, inside the hull of the "
        "rest). Free tips are the micropipette family's tip (same collar, rim, lip radius and snap depth) with "
        "`lip_segments` lip capsules (>= 12 keeps the worst-yaw interference with the nose jaws positive; 8 does not) "
        "and the two fine shaft sections merged into one cylinder. Returns the same build dictionary as "
        "construct_tip_box plus the collision geom counts."
    ),
    "inputs": [
        {"name": "out_dir", "type": "path", "description": "Folder that receives the MJCF, its meshes and <name>_params.json."},
        {"name": "name", "type": "str", "description": "Name. Default tipbox."},
        {"name": "length / width / base_h / wall_t / tray_t", "type": "float, metres",
         "description": "Box outer length (x) and width (y), base height to the tray top, wall and tray thickness. Defaults 0.127, 0.096, 0.072, 0.003, 0.004."},
        {"name": "rows / cols / pitch / hole_r", "type": "int / float", "description": "Tip grid and pitch; hole_r kept for the original's parameter set. Defaults 8, 12, 0.009, 0.00375."},
        {"name": "lid_h / lid_t / lid_deg / lid_open_max / lid_friction / lid_damping", "type": "float",
         "description": "Lid depth and plate thickness (m), initial and maximum opening (deg, 0 = closed), hinge friction (N m) and damping. Defaults 0.059, 0.0022, 100, 118, 0.055, 0.02."},
        {"name": "free_wells", "type": "list[[row, col]]", "description": "Wells holding REAL (free) tips. Default []."},
        {"name": "tip_names", "type": "list[str] | None", "description": "Body name suffix of each free tip in free_wells order: the tip body is '<name>_<suffix>'. Default None: 'tip0', 'tip1', ..."},
        {"name": "tip_params", "type": "dict", "description": "Overrides of the tip's parameters (length, collar_or, collar_ir, collar_len, lip_in, lip_depth, tip_rgba, density, friction)."},
        {"name": "lip_segments", "type": "int", "description": "Tangential capsules of each free tip's snap lip (>= 12). Default 12."},
        {"name": "slab_rise", "type": "float | None", "description": "Height of the tray slabs above the tray top, metres, between 4.5 mm and 1 mm under the collar tops. Default None: 4 mm under them."},
        {"name": "hole_clear", "type": "float", "description": "Clearance between a free tip's collar polygon and its slab hole, metres. Default 0.0004."},
        {"name": "lid_end_walls_collide", "type": "bool", "description": "The lid's two short skirt walls collide (else visual only). Default True."},
        {"name": "base_rgba / tray_rgba / lid_rgba / stub_rgba / label_rgba", "type": "list[4]", "description": "Colours. Defaults as construct_tip_box."},
        {"name": "density / slab_density / lid_density", "type": "float, kg/m3", "description": "Base plastic density (default 480), the slabs' density (default None: density x tray_t / (tray_t + slab_rise), the original tray's mass) and the lid plates' density (default 280: a ~25 g lid)."},
        {"name": "lid_solreffriction", "type": "list[2]", "description": "solreffriction of the lid hinge's friction loss: stiff, so the friction holds the lid's own weight instead of letting it creep (default [0.002, 1]: 0.03 deg/s against 2.5 deg/s at MuJoCo's default)."},
        {"name": "friction", "type": "str", "description": "MuJoCo friction triple. Default '0.85 0.015 0.0004'."},
        {"name": "pos / yaw_deg", "type": "list[3] / float", "description": "Placement of the box. Defaults [0, 0, 0], 0."},
        {"name": "timestep", "type": "float", "description": "Timestep written into the asset, seconds. Default 0.001."},
        {"name": "include_floor", "type": "bool", "description": "Floor and lights in the asset. Default True."},
    ],
    "outputs": [
        {"name": "built", "type": "dict",
         "description": "xml, params_json, name, kind 'tipbox', box_body, lid_body, lid_joint, tray_top_m, tip_seat_z_m, "
                        "tip_collar_top_m, slab_top_m, free_tips [{body, row, col, xy_local}], rows, cols, pitch_m, "
                        "lid_start_deg, lid_open_max_deg, mass_kg, lid_mass_kg, collision_geoms {box, lid, per_tip}."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "trimesh"],
    "simulators": ["MuJoCo"],
    "tags": ["tip box", "lid", "hinge", "lab", "receptacle", "parametric", "reconstruct", "object reconstruction",
             "low poly", "collision budget"],
    "used_in": ["Biology Experiment"],
    "status": "tested",
    "entrypoint": "construct_tip_box_lowpoly",
    "cli": "python -m tool_library.skills.construct_tip_box_lowpoly out_dir",
}

import json
from pathlib import Path

import numpy as np

from tool_library.common.mjcf import (FLOOR_XML, LIGHTS_XML, load, merge_params, mjcf_header, rgba, save_params,
                                      shell_ring, vec)
from tool_library.common.pipettekit import TIP_DEFAULTS, TIPBOX_DEFAULTS, _tip_mesh, tipbox_well_xy

DEFAULTS = dict(TIPBOX_DEFAULTS)
DEFAULTS.update({"lip_segments": 12, "slab_rise": None, "hole_clear": 0.0004, "lid_end_walls_collide": True,
                 "slab_density": None, "lid_density": 280.0, "lid_solreffriction": [0.002, 1.0], "tip_names": None})


def _tip_xml(tp, prefix, pos, mesh_name, n_lip):
    """The micropipette family's free tip with an n-capsule snap lip and a merged fine shaft (local z = 0 at the fine end)."""
    L = float(tp["length"])
    cor, cir, cl = float(tp["collar_or"]), float(tp["collar_ir"]), float(tp["collar_len"])
    lip, lipd = float(tp["lip_in"]), float(tp["lip_depth"])
    if cir >= cor:
        raise ValueError("collar_ir must be smaller than collar_or (the collar needs a wall)")
    if lip >= cir:
        raise ValueError("lip_in closes the bore: shrink lip_in")
    if cl >= L:
        raise ValueError("collar_len must be shorter than the tip length")
    gA = f'density="{tp["density"]}" friction="{tp["friction"]}" condim="3" solref="0.004 1"'
    inv = 'rgba="0 0 0 0"'
    g = [f'      <geom name="{prefix}_vis" type="mesh" mesh="{mesh_name}" rgba="{rgba(tp["tip_rgba"])}" '
         f'contype="0" conaffinity="0" group="2" density="1"/>']
    g += shell_ring(f"{prefix}_collar", cir, cor - cir, L - cl, L, "0 0 0 0", gA, n=8)
    for i in range(8):                       # rounded entrance rim: the nose jaws cam inward on it
        th = np.radians(22.5 + 45.0 * i)
        g.append(f'      <geom name="{prefix}_rim{i}" type="capsule" size="0.0005 {0.42 * cir:.6f}" '
                 f'pos="{vec(((cir + 0.0002) * np.cos(th), (cir + 0.0002) * np.sin(th), L - 0.0003))}" '
                 f'zaxis="{vec((-np.sin(th), np.cos(th), 0.0))}" rgba="0 0 0 0" friction="0.3 0.008 0.0002" '
                 f'condim="3" solref="0.006 1" density="{tp["density"]}"/>')
    half = cir * np.tan(np.pi / n_lip) * 1.05          # chord half-length: neighbouring capsules overlap
    for i in range(n_lip):                   # the snap lip: stiff, absolute contact (negative solref)
        th = np.radians(180.0 / n_lip + 360.0 * i / n_lip)
        g.append(f'      <geom name="{prefix}_lip{i}" type="capsule" size="{lip:.6f} {half:.6f}" '
                 f'pos="{vec((cir * np.cos(th), cir * np.sin(th), L - lipd))}" '
                 f'zaxis="{vec((-np.sin(th), np.cos(th), 0.0))}" rgba="0 0 0 0" friction="0.3 0.008 0.0002" '
                 f'condim="3" solref="-20000 -40" solimp="0.95 0.99 0.0005" density="{tp["density"]}"/>')
    for i, (r, z0, z1) in enumerate(((0.0036, L - cl - 0.014, L - cl + 0.001),
                                     (0.0028, L - cl - 0.030, L - cl - 0.013),
                                     (0.0018, 0.0015, L - cl - 0.029))):
        g.append(f'      <geom name="{prefix}_s{i}" type="cylinder" size="{r:.6f} {(z1 - z0) / 2:.6f}" '
                 f'pos="0 0 {(z0 + z1) / 2:.6f}" {inv} {gA}/>')
    g.append(f'      <geom name="{prefix}_foot" type="sphere" size="0.0011" pos="0 0 0.0011" {inv} {gA}/>')
    return (f'    <body name="{prefix}" pos="{vec(pos)}">\n      <freejoint name="{prefix}_free"/>\n'
            + "\n".join(g) + "\n    </body>")


def _bands(lo, hi, holes):
    """Split [lo, hi] into the intervals outside the (merged) hole intervals and the hole intervals themselves."""
    out, x = [], lo
    for a, b in sorted(holes):
        if a > x + 1e-6:
            out.append((x, a, False))
        out.append((max(a, x), b, True))
        x = max(x, b)
    if hi > x + 1e-6:
        out.append((x, hi, False))
    return out


def _merge(iv):
    iv = sorted(iv)
    out = []
    for a, b in iv:
        if out and a <= out[-1][1] + 1e-9:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def construct_tip_box_lowpoly(out_dir, **params):
    unknown = sorted(set(params) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"unknown parameters {unknown}; known: {sorted(DEFAULTS)}")
    p = merge_params(DEFAULTS, params)
    name = str(p["name"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    L, W, H = float(p["length"]), float(p["width"]), float(p["base_h"])
    wt, tt = float(p["wall_t"]), float(p["tray_t"])
    rows, cols, pitch = int(p["rows"]), int(p["cols"]), float(p["pitch"])
    n_lip = int(p["lip_segments"])
    if n_lip < 12:
        raise ValueError("lip_segments < 12 leaves junctions with no interference against the nose jaws (an 8-chord "
                         "lip lets a mounted tip slide off): use 12 or more")
    if cols * pitch > L - 2 * wt or rows * pitch > W - 2 * wt:
        raise ValueError("the tip grid does not fit inside the walls: reduce rows/cols/pitch or enlarge length/width")
    bc, yc, lc, sc = rgba(p["base_rgba"]), rgba(p["tray_rgba"]), rgba(p["lid_rgba"]), rgba(p["stub_rgba"])
    gA = f'density="{p["density"]}" friction="{p["friction"]}" condim="3" solref="0.004 1"'
    tp = merge_params(TIP_DEFAULTS, dict(p["tip_params"], include_floor=False))
    _tip_mesh(tp, out_dir, f"{name}_tiplathe")
    tlen, cor, cir, cl = float(tp["length"]), float(tp["collar_or"]), float(tp["collar_ir"]), float(tp["collar_len"])
    if tlen - cl > H - 0.004:
        raise ValueError(f"a {1000 * tlen:.0f} mm tip hangs deeper than the {1000 * H:.0f} mm base: raise base_h")
    tray_top = H
    plat_top = tray_top - (tlen - cl) + 0.005          # the collar rides ~5 mm proud of the tray
    tip_z = plat_top + 0.0002
    collar_top = tip_z + tlen
    rise = float(p["slab_rise"]) if p["slab_rise"] is not None else (collar_top - tray_top) - 0.004
    if not collar_top - tray_top - 0.0045 <= rise <= collar_top - tray_top - 0.001:
        raise ValueError(f"slab_rise {rise:.4f} m must lie between 4.5 mm and 1 mm under the collar tops "
                         f"({1000 * (collar_top - tray_top):.1f} mm above the tray): lower, the drawn collars stand out "
                         f"of the physics; higher, the slabs bury the free tips' collars")
    slab_top = tray_top + rise
    slab_density = float(p["slab_density"]) if p["slab_density"] is not None else float(p["density"]) * tt / (tt + rise)
    gS = f'density="{slab_density:.3f}" friction="{p["friction"]}" condim="3" solref="0.004 1"'
    # the collar polygon (8 boxes of shell_ring) reaches its corners at this radius; the hole clears it
    rc = cir + (cor - cir) / 2
    r_corner = float(np.hypot(cor, rc * np.tan(np.pi / 8) * 1.06))
    hh = max(pitch / 2, r_corner + float(p["hole_clear"]))
    free = list(dict.fromkeys((int(r), int(c)) for r, c in p["free_wells"]))     # the given order: tip_names follow it
    for r, c in free:
        if not (0 <= r < rows and 0 <= c < cols):
            raise ValueError(f"free well {(r, c)} is outside the {rows}x{cols} grid")

    g = []
    for i, (sx, sy, px, py) in enumerate(((L / 2, wt / 2, 0.0, W / 2 - wt / 2), (L / 2, wt / 2, 0.0, -(W / 2 - wt / 2)),
                                          (wt / 2, W / 2 - wt, L / 2 - wt / 2, 0.0),
                                          (wt / 2, W / 2 - wt, -(L / 2 - wt / 2), 0.0))):
        g.append(f'      <geom name="{name}_wall{i}" type="box" size="{vec((sx, sy, H / 2))}" '
                 f'pos="{vec((px, py, H / 2))}" rgba="{bc}" {gA}/>')
    g.append(f'      <geom name="{name}_floor" type="box" size="{vec((L / 2 - wt, W / 2 - wt, 0.0015))}" '
             f'pos="0 0 0.0015" rgba="{bc}" {gA}/>')
    g.append(f'      <geom name="{name}_label" type="box" size="0.020 0.0006 0.006" '
             f'pos="0 {-W / 2 - 0.0004:.6f} {0.30 * H:.6f}" rgba="{rgba(p["label_rgba"])}" contype="0" '
             f'conaffinity="0" density="1"/>')
    # tray slabs: the inside of the walls minus the free wells' holes, as bands of y split in x around the holes
    x0, x1, y0, y1 = -(L / 2 - wt), L / 2 - wt, -(W / 2 - wt), W / 2 - wt
    holes = [(tipbox_well_xy(p, r, c), (r, c)) for r, c in free]
    ybands = _bands(y0, y1, _merge([(xy[1] - hh, xy[1] + hh) for xy, _ in holes]))
    zc, hz = (tray_top - tt + slab_top) / 2, (slab_top - (tray_top - tt)) / 2
    k = 0
    for ya, yb, has in ybands:
        xs = [(x0, x1, False)] if not has else _bands(
            x0, x1, _merge([(xy[0] - hh, xy[0] + hh) for xy, _ in holes if xy[1] - hh < (ya + yb) / 2 < xy[1] + hh]))
        for xa, xb, hole in xs:
            if hole or xb - xa < 1e-4 or yb - ya < 1e-4:
                continue
            g.append(f'      <geom name="{name}_slab{k}" type="box" size="{vec(((xb - xa) / 2, (yb - ya) / 2, hz))}" '
                     f'pos="{vec(((xa + xb) / 2, (ya + yb) / 2, zc))}" rgba="{yc}" {gS}/>')
            k += 1
    for (x, y), (r, c) in holes:              # the foot's platform column
        g.append(f'      <geom name="{name}_p{r}_{c}" type="box" size="{vec((0.0044, 0.0044, (plat_top - 0.003) / 2))}" '
                 f'pos="{vec((x, y, (plat_top + 0.003) / 2))}" rgba="{bc}" {gA}/>')
    stub_lo = slab_top - 0.001
    for r in range(rows):                     # the drawn collars of the moulded (non-free) tips
        for c in range(cols):
            if (r, c) in free:
                continue
            x, y = tipbox_well_xy(p, r, c)
            g.append(f'      <geom name="{name}_stub{r}_{c}" type="cylinder" size="{cor - 0.0001:.5f} {(collar_top - stub_lo) / 2:.5f}" '
                     f'pos="{vec((x, y, (stub_lo + collar_top) / 2))}" rgba="{sc}" contype="0" conaffinity="0" density="1"/>')

    lh, lt = float(p["lid_h"]), float(p["lid_t"])
    start = float(p["lid_deg"])
    ca, sa = np.cos(np.radians(-start / 2)), np.sin(np.radians(-start / 2))
    lg = 0.0020
    gL = f'density="{p["lid_density"]}" friction="{p["friction"]}" condim="3" solref="0.004 1"'
    lidg = [f'        <geom name="{name}_lidtop" type="box" size="{vec((L / 2 + lt + lg, W / 2 + lt + lg, lt / 2))}" '
            f'pos="{vec((0.0, -W / 2 - lg, lh - lt / 2))}" rgba="{lc}" {gL}/>']
    end_attr = gL if p["lid_end_walls_collide"] else f'contype="0" conaffinity="0" density="{p["lid_density"]}"'
    for i, (sx, sy, px, py) in enumerate(((L / 2 + lt + lg, lt / 2, 0.0, -W - 2 * lg - lt / 2),
                                          (L / 2 + lt + lg, lt / 2, 0.0, -lg + lt / 2),
                                          (lt / 2, W / 2 + lg, L / 2 + lg + lt / 2, -W / 2 - lg),
                                          (lt / 2, W / 2 + lg, -(L / 2 + lg + lt / 2), -W / 2 - lg))):
        lidg.append(f'        <geom name="{name}_lidw{i}" type="box" size="{vec((sx, sy, lh / 2 - 0.001))}" '
                    f'pos="{vec((px, py, lh / 2))}" rgba="{lc}" {gL if i < 2 else end_attr}/>')
    hinge_z = H - 0.032
    if hinge_z + lh - lt < collar_top + 0.003:
        raise ValueError(f"lid_h {1000 * lh:.0f} mm: the closed lid's ceiling would sit on the standing tips; raise lid_h")
    lid = f"""
    <body name="{name}_lid" pos="{vec((0.0, W / 2 + lg, hinge_z))}" quat="{ca:.8f} {sa:.8f} 0 0">
      <joint name="{name}_hinge" type="hinge" axis="-1 0 0" range="-2 {p['lid_open_max']}" ref="{start:.3f}"
             damping="{p['lid_damping']}" frictionloss="{p['lid_friction']}" armature="0.0004"
             solreffriction="{vec(p['lid_solreffriction'])}" solimpfriction="0.99 0.999 0.001"/>
{chr(10).join(lidg)}
    </body>"""
    tips_xml, tips = [], []
    names = list(p["tip_names"]) if p["tip_names"] is not None else [f"tip{i}" for i in range(len(holes))]
    if len(names) != len(holes) or len(set(names)) != len(names) or any(not str(n).replace("_", "").isalnum() for n in names):
        raise ValueError(f"tip_names {p['tip_names']} must give one distinct alphanumeric suffix per free well ({len(holes)})")
    for i, ((x, y), (r, c)) in enumerate(holes):
        pref = f"{name}_{names[i]}"
        tips_xml.append(_tip_xml(tp, pref, [x + float(p["pos"][0]), y + float(p["pos"][1]), tip_z + float(p["pos"][2])],
                                 f"{name}_tiplathe_m", n_lip))
        tips.append({"body": pref, "row": r, "col": c, "xy_local": [round(x, 6), round(y, 6)]})
    if tips and abs(float(p["yaw_deg"])) > 1e-9:
        raise ValueError("free tips are written in the asset frame: build with yaw_deg 0 and turn the box when staging")
    floor = FLOOR_XML if p["include_floor"] else ""
    xml = (mjcf_header(f"{name}_tipbox", timestep=p["timestep"], meshdir="meshes")
           + f"""  <asset>
    <mesh name="{name}_tiplathe_m" file="{name}_tiplathe.obj"/>
  </asset>
  <worldbody>{LIGHTS_XML}{floor}
    <body name="{name}" pos="{vec(p['pos'])}" euler="0 0 {p['yaw_deg']}">
      <freejoint name="{name}_free"/>
{chr(10).join(g)}{lid}
    </body>
{chr(10).join(tips_xml)}
  </worldbody>
</mujoco>
""")
    xml_path = out_dir / f"{name}.xml"
    xml_path.write_text(xml)
    m = load(xml_path)
    save_params(out_dir / f"{name}_params.json", p)

    def ncol(body):
        b = m.body(body).id
        return int(sum(1 for gi in range(m.ngeom) if int(m.geom_bodyid[gi]) == b and (m.geom_contype[gi] or m.geom_conaffinity[gi])))
    built = {
        "xml": str(xml_path), "params_json": str(out_dir / f"{name}_params.json"), "name": name, "kind": "tipbox",
        "box_body": name, "lid_body": f"{name}_lid", "lid_joint": f"{name}_hinge",
        "tray_top_m": tray_top, "tip_seat_z_m": round(tip_z, 5), "tip_collar_top_m": round(collar_top, 5),
        "slab_top_m": round(slab_top, 5), "hole_half_width_m": round(hh, 5),
        "free_tips": tips, "rows": rows, "cols": cols, "pitch_m": pitch,
        "lid_start_deg": start, "lid_open_max_deg": float(p["lid_open_max"]),
        "mass_kg": round(float(m.body(name).subtreemass[0]), 4), "lid_mass_kg": round(float(m.body(f"{name}_lid").mass[0]), 4),
        "collision_geoms": {"box": ncol(name), "lid": ncol(f"{name}_lid"), "per_tip": ncol(tips[0]["body"]) if tips else 0},
        "note": (f"lid hinge: 0 = closed, +ve = open, friction {p['lid_friction']} N m holds it where a push leaves it; "
                 f"free tips stand in caged sockets (collar in a {2000 * hh:.1f} mm square slab hole, foot on a "
                 f"platform) and snap onto a micropipette nose; the other wells are drawn collars on the tray slabs."),
    }
    (out_dir / f"{name}_built.json").write_text(json.dumps(built, indent=1))
    return built


if __name__ == "__main__":
    import sys as _sys
    print(json.dumps(construct_tip_box_lowpoly(_sys.argv[1] if len(_sys.argv) > 1 else "."), indent=1, default=str))
