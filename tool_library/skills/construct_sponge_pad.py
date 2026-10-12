"""A compliant foam pad (sponge, scrubber, eraser pad): a thin backing slab with a grid of sprung foam cells that squash under a palm."""
from __future__ import annotations

TOOL = {
    "name": "construct_sponge_pad",
    "title": "Construct sponge pad",
    "type": "skill",
    "category": "Object Reconstruction",
    "summary": "A soft rectangular foam pad as a free body: a backing slab plus an nx x ny grid of foam cells, each on a sprung slide joint normal to the pad, so a pressing palm squashes it locally with a measured overall stiffness while it slides on the table with friction.",
    "description": (
        "Compliance is modelled as a bed of springs: the pad's bottom `base_frac` of the thickness is a rigid slab (the "
        "free root body, which touches the table) and the rest is split into nx x ny box cells, each a child body on a "
        "slide joint along the pad normal with range [-max_compression_frac x cell height, 0], stiffness "
        "stiffness_N_per_m / (nx ny) (cells act in parallel, so a uniform press of F newtons squashes the pad by "
        "F / stiffness_N_per_m), critical damping and a small armature for numerical stability. An uneven press (fingers "
        "on one end) squashes only the cells under it, which is how the pad 'bends' under the fingers; the slab itself "
        "stays flat. The mass is density x volume (or `mass`), shared by volume. Origin at the centre of the bottom face, "
        "length along x, width along y. Sites: top (centre of the top face) and grip (centre of the slab). Unknown "
        "parameters raise ValueError."
    ),
    "inputs": [
        {"name": "out_dir", "type": "path", "description": "Receives <name>.xml and <name>_params.json."},
        {"name": "name", "type": "str", "description": "Asset name and prefix. Default 'sponge'."},
        {"name": "length", "type": "float", "description": "Pad length (x), m. Default 0.12."},
        {"name": "width", "type": "float", "description": "Pad width (y), m. Default 0.08."},
        {"name": "thickness", "type": "float", "description": "Uncompressed thickness, m. Default 0.02."},
        {"name": "base_frac", "type": "float", "description": "Fraction of the thickness that is the rigid backing slab. Default 0.35."},
        {"name": "nx", "type": "int", "description": "Cells along the length. Default 4."},
        {"name": "ny", "type": "int", "description": "Cells along the width. Default 3."},
        {"name": "stiffness_N_per_m", "type": "float", "description": "Whole-pad compression stiffness under a uniform press. Default 2000."},
        {"name": "max_compression_frac", "type": "float", "description": "Largest squash as a fraction of the cell height. Default 0.6."},
        {"name": "cell_gap", "type": "float", "description": "Gap between neighbouring cells, m. Default 0.001."},
        {"name": "cell_armature", "type": "float", "description": "Armature of each cell joint, kg. Default 0.004."},
        {"name": "damping_ratio", "type": "float", "description": "Cell damping as a fraction of critical. Default 1.0."},
        {"name": "density", "type": "float", "description": "Foam density kg/m3 (ignored when mass is given). Default 40."},
        {"name": "mass", "type": "float | None", "description": "Total mass kg. Default None (from density)."},
        {"name": "rgba", "type": "list[4]", "description": "Foam colour."},
        {"name": "base_rgba", "type": "list[4] | None", "description": "Slab colour; default the foam colour."},
        {"name": "friction", "type": "list[3]", "description": "MuJoCo friction triple. Default [1.0, 0.01, 0.001]."},
        {"name": "contact_solref", "type": "list[2]", "description": "Contact solref of the pad geoms, applied with priority 1 so it wins over the table's: a gram-scale pad under a newton-scale press otherwise sinks millimetres into a soft contact and creeps under any drag. Default [0.004, 1.0]."},
        {"name": "contact_solimp", "type": "list[3]", "description": "Contact solimp of the pad geoms. Default [0.99, 0.999, 0.001]."},
        {"name": "pos", "type": "list[3]", "description": "Bottom-centre position. Default [0, 0, 0]."},
        {"name": "yaw_deg", "type": "float", "description": "Yaw about z. Default 0."},
        {"name": "include_floor", "type": "bool", "description": "Add a floor plane (stand-alone previews only). Default False."},
        {"name": "timestep", "type": "float", "description": "Option timestep. Default 0.002."},
    ],
    "outputs": [
        {"name": "built", "type": "dict", "description": "xml, params_json, body, base_geom, cells, cell_joints, cell_stiffness, stiffness_N_per_m, max_compression_m, size_m [l, w, t], mass_kg, sites."},
    ],
    "status": "tested",
    "entrypoint": "construct_sponge_pad",
    "requirements": ["mujoco>=3.1", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["sponge", "foam", "pad", "soft", "compliant", "deformable", "scrub", "wipe", "object reconstruction"],
    "used_in": [],
    "cli": "python -m tool_library.skills.construct_sponge_pad out_dir",
}

import json
import math
import sys
from pathlib import Path

DEFAULTS = {
    "name": "sponge", "length": 0.12, "width": 0.08, "thickness": 0.02, "base_frac": 0.35, "nx": 4, "ny": 3,
    "stiffness_N_per_m": 2000.0, "max_compression_frac": 0.6, "cell_gap": 0.001, "cell_armature": 0.004,
    "damping_ratio": 1.0, "density": 40.0, "mass": None, "rgba": [0.55, 0.70, 0.95, 1.0], "base_rgba": None,
    "friction": [1.0, 0.01, 0.001], "contact_solref": [0.004, 1.0], "contact_solimp": [0.99, 0.999, 0.001],
    "pos": [0.0, 0.0, 0.0], "yaw_deg": 0.0, "include_floor": False, "timestep": 0.002,
}


def _f(v):
    return " ".join(f"{float(x):.6g}" for x in v)


def construct_sponge_pad(out_dir, **params):
    unknown = sorted(set(params) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"unknown parameter(s) {unknown}; valid: {sorted(DEFAULTS)}")
    p = {**DEFAULTS, **params}
    for k in ("length", "width", "thickness", "stiffness_N_per_m", "cell_armature", "timestep", "density"):
        if float(p[k]) <= 0:
            raise ValueError(f"{k} must be > 0 (got {p[k]})")
    if not 0.05 <= float(p["base_frac"]) <= 0.9:
        raise ValueError("base_frac must lie in [0.05, 0.9]")
    if not 0 < float(p["max_compression_frac"]) <= 0.95:
        raise ValueError("max_compression_frac must lie in (0, 0.95]")
    nx, ny = int(p["nx"]), int(p["ny"])
    if nx < 1 or ny < 1 or nx * ny > 400:
        raise ValueError("nx and ny must be >= 1 with nx*ny <= 400")
    L, W, T = float(p["length"]), float(p["width"]), float(p["thickness"])
    gap = float(p["cell_gap"])
    if gap < 0 or gap >= min(L / nx, W / ny) / 2:
        raise ValueError("cell_gap must be >= 0 and smaller than half a cell")
    if p["mass"] is not None and float(p["mass"]) <= 0:
        raise ValueError("mass must be > 0 or None")
    n = str(p["name"])
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    tb = T * float(p["base_frac"]); hc = T - tb
    total = float(p["mass"]) if p["mass"] is not None else float(p["density"]) * L * W * T
    cx, cy = L / nx, W / ny
    hx, hy = cx / 2 - gap / 2, cy / 2 - gap / 2
    dens = total / (L * W * tb + nx * ny * (2 * hx) * (2 * hy) * hc)     # the geoms' real volume, gaps excluded
    cell_mass = dens * (2 * hx) * (2 * hy) * hc
    kc = float(p["stiffness_N_per_m"]) / (nx * ny)
    arm = float(p["cell_armature"])
    damp = float(p["damping_ratio"]) * 2.0 * math.sqrt(kc * (cell_mass + arm))
    springref = max(2.0 * cell_mass * 9.81 / kc, 1e-4)      # holds every cell on its top stop against its own weight
    travel = hc * float(p["max_compression_frac"])
    fr = _f(p["friction"])
    if float(p["contact_solref"][0]) < 2 * float(p["timestep"]):
        raise ValueError("contact_solref[0] must be at least twice the timestep")
    con = f'priority="1" solref="{_f(p["contact_solref"])}" solimp="{_f(p["contact_solimp"])}"'
    rgba = _f(p["rgba"]); brgba = _f(p["base_rgba"] or p["rgba"])
    cells, joints, lines = [], [], []
    for i in range(nx):
        for j in range(ny):
            x = -L / 2 + cx * (i + 0.5); y = -W / 2 + cy * (j + 0.5)
            b = f"{n}_cell_{i}_{j}"; jn = f"{b}_z"
            cells.append(b); joints.append(jn)
            lines.append(
                f'      <body name="{b}" pos="{x:.6g} {y:.6g} {tb:.6g}">\n'
                f'        <joint name="{jn}" type="slide" axis="0 0 1" limited="true" range="{-travel:.6g} 0" stiffness="{kc:.6g}" springref="{springref:.6g}" damping="{damp:.6g}" armature="{arm:.6g}" solreflimit="0.004 1"/>\n'
                f'        <geom name="{b}_geom" type="box" size="{hx:.6g} {hy:.6g} {hc / 2:.6g}" pos="0 0 {hc / 2:.6g}" rgba="{rgba}" density="{dens:.6g}" friction="{fr}" condim="4" {con}/>\n'
                f'      </body>')
    floor = '<geom name="floor" type="plane" size="1 1 0.01" rgba="0.8 0.8 0.8 1"/>' if p["include_floor"] else ""
    xml = (f'<mujoco model="{n}">\n  <compiler angle="degree"/>\n  <option timestep="{float(p["timestep"]):.6g}"/>\n  <worldbody>\n    {floor}\n'
           f'    <body name="{n}" pos="{_f(p["pos"])}" euler="0 0 {float(p["yaw_deg"]):.6g}">\n'
           f'      <freejoint name="{n}_free"/>\n'
           f'      <geom name="{n}_base_geom" type="box" size="{L / 2:.6g} {W / 2:.6g} {tb / 2:.6g}" pos="0 0 {tb / 2:.6g}" rgba="{brgba}" density="{dens:.6g}" friction="{fr}" condim="4" {con}/>\n'
           f'      <site name="{n}_top_site" pos="0 0 {T:.6g}" size="0.003"/>\n'
           f'      <site name="{n}_grip_site" pos="0 0 {tb / 2:.6g}" size="0.003"/>\n'
           + "\n".join(lines) + '\n    </body>\n  </worldbody>\n</mujoco>\n')
    xp = out / f"{n}.xml"; xp.write_text(xml)
    pj = out / f"{n}_params.json"; pj.write_text(json.dumps(p, indent=1))
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(xp))
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    return {"xml": str(xp), "params_json": str(pj), "body": n, "free_joint": f"{n}_free", "base_geom": f"{n}_base_geom",
            "cells": cells, "cell_joints": joints, "cell_stiffness": kc, "stiffness_N_per_m": float(p["stiffness_N_per_m"]),
            "max_compression_m": travel, "size_m": [L, W, T], "base_thickness_m": tb,
            "mass_kg": float(m.body_subtreemass[bid]), "sites": {"top": f"{n}_top_site", "grip": f"{n}_grip_site"}}


if __name__ == "__main__":
    print(json.dumps(construct_sponge_pad(sys.argv[1] if len(sys.argv) > 1 else "sponge_out"), indent=1))
