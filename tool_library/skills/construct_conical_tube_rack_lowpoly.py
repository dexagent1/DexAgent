"""Construct rack for 15 ml conical tubes, low-poly: the open-frame rack of construct_conical_tube_rack with its top plate
as the fewest boxes (two long bars and one cross bar between neighbouring wells) and square catch holes in its
bottom plate under chosen wells, so what stands in such a well (a pipette's nose, a conical tube's
tip) sits in a socket and leans on its edges instead of on friction alone.

A fork of construct_conical_tube_rack. The original top plate is
four boxes per well (16 for four wells): neighbouring bars are merged here (2 + cols + 1 boxes, the same openings), so a
hand reaching a tube in the rack is paired with fewer geoms (a rack, two tubes and a pipette came to 72 geoms within
6 cm of each other, over the 48 a hand scene should pay for). The catch holes, unchanged:
a pipette standing leaning in a well of the original rack rests its nose on the flat bottom plate and the barrel on the
hole edge; the nose is held sideways by friction only, MuJoCo's soft friction creeps under that sustained sideways load
and the pipette slid and fell within 3 s (9 s with the executor's no-slip iterations), measured. With the nose in a
catch hole the edges hold it by normal force and it stands (0.1 mm in 12 s).

    python -m tool_library.skills.construct_conical_tube_rack_lowpoly [out_dir]
"""
from __future__ import annotations

TOOL = {'name': 'construct_conical_tube_rack_lowpoly',
 'title': 'Construct rack for 15 ml conical tubes (low-poly, catch holes)',
 'type': 'skill',
 'category': 'Object Reconstruction',
 'summary': 'An open-frame rack for 15 ml conical tubes with a merged low-poly top plate (2 + cols + 1 boxes) and square '
            'catch holes in the bottom plate under chosen wells, so a pipette nose or a cone tip stands in a socket.',
 'description': 'A fork of construct_conical_tube_rack with the top plate merged into two long bars and one cross '
                'bar between neighbouring wells (same openings, 2 + cols + 1 boxes instead of 4 per well; one row of wells, as '
                'the original top plate). The same moulded open frame (top plate with real hole openings, '
                'corner posts, side walls, visual perforations and hole rims) from its footprint, height, hole grid and '
                'colours, with the bottom plate cut into boxes around square openings (`catch_holes`: [[row, col, '
                'half_width_m], ...]) under the named wells. The table below a catch hole is the socket floor; the '
                'hole edges hold whatever stands in it sideways by normal force (a leaning micropipette whose nose '
                'stood on the flat plate slid out and fell within seconds under MuJoCo soft friction). Holes are '
                'indexed by row and column; hole_xy() and hole_top_z() return where a tube stands. A catch hole must '
                'fit inside its well pitch and leave plate around it.',
 'inputs': [{'name': 'out_dir',
             'type': 'path',
             'description': 'Folder that receives the MJCF and <name>_params.json.'},
            {'name': 'catch_holes', 'type': 'list[[row, col, half_width_m]]',
             'description': 'Square openings in the bottom plate under these wells (half width in metres). Default [] (the plain rack).'},

            {'name': 'name', 'type': 'str', 'description': 'Name. Default tuberack15'},
            {'name': 'length',
             'type': 'float',
             'description': 'Outer envelope, m. Default 0.18'},
            {'name': 'width', 'type': 'float', 'description': 'Width, metres. Default 0.062'},
            {'name': 'height', 'type': 'float', 'description': 'Height, metres. Default 0.085'},
            {'name': 'plate_t',
             'type': 'float',
             'description': 'Top / bottom plate thickness. Default 0.004'},
            {'name': 'post',
             'type': 'float',
             'description': 'Corner post square section. Default 0.006'},
            {'name': 'wall_t',
             'type': 'float',
             'description': 'Side walls (long faces). Default 0.003'},
            {'name': 'rows', 'type': 'int', 'description': 'Rows. Default 1'},
            {'name': 'cols', 'type': 'int', 'description': 'Cols. Default 4'},
            {'name': 'hole_r', 'type': 'float', 'description': 'Hole r. Default 0.0105'},
            {'name': 'pitch_x',
             'type': 'float',
             'description': 'Pitch x, metres per turn. Default 0.04'},
            {'name': 'pitch_y',
             'type': 'float',
             'description': 'Pitch y, metres per turn. Default 0.03'},
            {'name': 'rim_r',
             'type': 'float',
             'description': 'The moulded ring around each top hole (visual). Default 0.0125'},
            {'name': 'rim_h', 'type': 'float', 'description': 'Rim h. Default 0.0015'},
            {'name': 'perf_rows',
             'type': 'int',
             'description': 'Side-wall drainage holes (visual). Default 4'},
            {'name': 'perf_cols', 'type': 'int', 'description': 'Perf cols. Default 9'},
            {'name': 'perf_r', 'type': 'float', 'description': 'Perf r. Default 0.0042'},
            {'name': 'rack_rgba',
             'type': 'list[4]',
             'description': 'Orange. Default [0.96, 0.42, 0.06, 1.0]'},
            {'name': 'perf_rgba',
             'type': 'list[4]',
             'description': 'The holes read darker. Default [0.55, 0.2, 0.03, 1.0]'},
            {'name': 'pos',
             'type': 'list[3]',
             'description': 'Pos, metres, xyz. Default [0.0, 0.0, 0.0]'},
            {'name': 'yaw_deg',
             'type': 'float',
             'description': 'Yaw deg, degrees. Default 0.0'},
            {'name': 'density', 'type': 'float', 'description': 'Density. Default 900.0'},
            {'name': 'friction',
             'type': 'list[3]',
             'description': 'Friction, MuJoCo friction triple. Default [0.8, 0.01, 0.0002]'},
            {'name': 'fixed_base', 'type': 'bool', 'description': 'Fixed base. Default False'},
            {'name': 'timestep',
             'type': 'float',
             'description': 'Timestep, seconds. Default 0.002'},
            {'name': 'include_floor',
             'type': 'bool',
             'description': 'Include floor. Default True'}],
 'outputs': [{'name': 'built',
              'type': 'dict',
              'description': 'xml: the MJCF path; params_json: every parameter as built; the '
                             'names of the bodies, joints and equality constraints the '
                             'mechanism exposes, and the derived measurements (sizes, travel, '
                             'masses); stills when preview is set.'}],
 'requirements': ['mujoco>=3.2', 'numpy', 'trimesh', 'imageio', 'Pillow'],
 'simulators': ['MuJoCo'],
  'tags': ['rack', 'lab', 'parametric', 'reconstruct', 'object reconstruction', 'catch hole', 'socket', 'pipette stand', 'low poly', 'collision budget'],
 'used_in': ['Biology Experiment'],
 'status': 'tested',
 'entrypoint': 'construct_conical_tube_rack_lowpoly',
 'cli': 'python -m tool_library.skills.construct_conical_tube_rack_lowpoly out_dir'}

import json
from pathlib import Path
import numpy as np
from tool_library.common.mjcf import FLOOR_XML, LIGHTS_XML, load, merge_params, mjcf_header, rgba, save_params, vec

DEFAULTS = {
    "name": "tuberack15",
    "length": 0.180, "width": 0.062, "height": 0.085,      # outer envelope, m
    "plate_t": 0.004,                                      # top / bottom plate thickness
    "post": 0.006,                                         # corner post square section
    "wall_t": 0.003,                                       # side walls (long faces)
    "rows": 1, "cols": 4, "hole_r": 0.0105, "pitch_x": 0.040, "pitch_y": 0.030,
    "rim_r": 0.0125, "rim_h": 0.0015,                      # the moulded ring around each top hole (visual)
    "perf_rows": 4, "perf_cols": 9, "perf_r": 0.0042,      # side-wall drainage holes (visual)
    "rack_rgba": [0.96, 0.42, 0.06, 1.0],                  # orange
    "perf_rgba": [0.55, 0.20, 0.03, 1.0],                  # the holes read darker
    "pos": [0.0, 0.0, 0.0], "yaw_deg": 0.0,
    "density": 900.0, "friction": [0.8, 0.01, 0.0002],
    "fixed_base": False, "timestep": 0.002, "include_floor": True,
    "catch_holes": [],                                     # [[row, col, half_width_m]] square openings in the bottom plate
}


def hole_xy(p: dict, row: int, col: int):
    x = (col - (int(p["cols"]) - 1) / 2) * float(p["pitch_x"])
    y = (row - (int(p["rows"]) - 1) / 2) * float(p["pitch_y"])
    return float(x), float(y)


def hole_top_z(p: dict) -> float:
    return float(p["height"])


def _merge(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1] + 1e-9:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _bands(lo, hi, holes):
    out, x = [], lo
    for a, b in holes:
        if a > x + 1e-6:
            out.append((x, a, False))
        out.append((max(a, x), b, True))
        x = max(x, b)
    if hi > x + 1e-6:
        out.append((x, hi, False))
    return out


def _bottom_plate(p, name, L, W, pt, oc, ga):
    """The bottom plate as the fewest boxes around the square catch holes (bands of y split in x around the holes)."""
    holes = []
    for item in p["catch_holes"]:
        if len(item) != 3:
            raise ValueError(f"catch_holes entries are [row, col, half_width_m], got {item!r}")
        r, c, hw = int(item[0]), int(item[1]), float(item[2])
        if not (0 <= r < int(p["rows"]) and 0 <= c < int(p["cols"])):
            raise ValueError(f"catch hole {(r, c)} is outside the {p['rows']}x{p['cols']} hole grid")
        lim = min(float(p["pitch_x"]) if int(p["cols"]) > 1 else L, float(p["pitch_y"]) if int(p["rows"]) > 1 else W) / 2 - 0.002
        if not 0.001 <= hw <= lim:
            raise ValueError(f"catch hole half width {hw * 1e3:.1f} mm must lie between 1 mm and {lim * 1e3:.1f} mm (half the well pitch "
                             f"minus 2 mm of plate)")
        x, y = hole_xy(p, r, c)
        if abs(y) + hw > W / 2 - 0.002 or abs(x) + hw > L / 2 - 0.002:
            raise ValueError(f"catch hole {(r, c)} reaches the plate edge: shrink its half width")
        holes.append((x, y, hw))
    if not holes:
        return [f'      <geom name="{name}_bottom" type="box" size="{vec((L/2, W/2, pt/2))}" pos="{vec((0, 0, pt/2))}" rgba="{oc}" {ga}/>']
    out, k = [], 0
    for ya, yb, has in _bands(-W / 2, W / 2, _merge([(y - hw, y + hw) for _, y, hw in holes])):
        xs = [(-L / 2, L / 2, False)] if not has else _bands(-L / 2, L / 2, _merge([(x - hw, x + hw) for x, y, hw in holes
                                                                                    if y - hw < (ya + yb) / 2 < y + hw]))
        for xa, xb, hole in xs:
            if hole or xb - xa < 1e-4 or yb - ya < 1e-4:
                continue
            out.append(f'      <geom name="{name}_bottom{k}" type="box" size="{vec(((xb - xa) / 2, (yb - ya) / 2, pt / 2))}" '
                       f'pos="{vec(((xa + xb) / 2, (ya + yb) / 2, pt / 2))}" rgba="{oc}" {ga}/>')
            k += 1
    return out


def build(params: dict | None = None, out_dir: str | Path = ".") -> dict:
    p = merge_params(DEFAULTS, params)
    name = str(p["name"]); out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    L, W, H, pt, ps, wt = (float(p[k]) for k in ("length", "width", "height", "plate_t", "post", "wall_t"))
    oc, hc = rgba(p["rack_rgba"]), rgba(p["perf_rgba"])
    ga = f'density="{p["density"]}" friction="{vec(p["friction"])}" condim="3"'
    g = []
    # bottom plate and the top plate as strips BETWEEN the holes (so the holes are real openings)
    g.extend(_bottom_plate(p, name, L, W, pt, oc, ga))
    xs = sorted(set(hole_xy(p, r, c)[0] for r in range(int(p["rows"])) for c in range(int(p["cols"]))))
    edges = [-L / 2] + [0.5 * (a + b) for a, b in zip(xs[:-1], xs[1:])] + [L / 2]
    hr = float(p["hole_r"]); zt = H - pt / 2
    if 2 * hr >= W - 2 * wt:
        raise ValueError(f"hole_r {hr*1e3:.1f} mm does not fit between the side walls of a "
                         f"{W*1e3:.0f} mm wide rack with {wt*1e3:.1f} mm walls — shrink hole_r or widen the rack")
    if int(p["rows"]) != 1:
        raise ValueError("the top plate is built for one row of wells (rows = 1), as in construct_conical_tube_rack")
    for i, xc in enumerate(xs):
        x0, x1 = edges[i], edges[i + 1]
        if xc - hr - x0 <= 1e-4 or x1 - xc - hr <= 1e-4:
            raise ValueError(f"hole column {i} (hole_r {hr*1e3:.1f} mm) leaves no top plate between holes or at the "
                             f"rack end: pitch_x {float(p['pitch_x'])*1e3:.1f} mm, length {L*1e3:.0f} mm — "
                             f"shrink hole_r, raise pitch_x or lengthen the rack")
        # moulded ring round the hole (visual)
        g.append(f'      <geom name="{name}_rim{i}" type="cylinder" size="{vec((float(p["rim_r"]), float(p["rim_h"])/2))}" pos="{vec((xc, 0, H + float(p["rim_h"])/2))}" rgba="{oc}" contype="0" conaffinity="0" mass="0"/>')
    # two long bars in front of and behind the row of holes, and one cross bar between neighbouring holes (and at the ends)
    g.append(f'      <geom name="{name}_topA" type="box" size="{vec((L/2, (W/2 - hr)/2, pt/2))}" pos="{vec((0, -(W/2 + hr)/2, zt))}" rgba="{oc}" {ga}/>')
    g.append(f'      <geom name="{name}_topB" type="box" size="{vec((L/2, (W/2 - hr)/2, pt/2))}" pos="{vec((0, (W/2 + hr)/2, zt))}" rgba="{oc}" {ga}/>')
    cuts = [-L / 2] + [v for xc in xs for v in (xc - hr, xc + hr)] + [L / 2]
    for i in range(0, len(cuts), 2):
        a, b = cuts[i], cuts[i + 1]
        g.append(f'      <geom name="{name}_topX{i // 2}" type="box" size="{vec(((b - a)/2, hr, pt/2))}" pos="{vec(((a + b)/2, 0, zt))}" rgba="{oc}" {ga}/>')
    for k, (sx, sy) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1))):
        g.append(f'      <geom name="{name}_post{k}" type="box" size="{vec((ps/2, ps/2, H/2))}" pos="{vec((sx*(L/2 - ps/2), sy*(W/2 - ps/2), H/2))}" rgba="{oc}" {ga}/>')
    for k, sy in enumerate((-1, 1)):                  # long side walls
        g.append(f'      <geom name="{name}_wall{k}" type="box" size="{vec((L/2, wt/2, H/2))}" pos="{vec((0, sy*(W/2 - wt/2), H/2))}" rgba="{oc}" {ga}/>')
        # perforations: dark discs on the wall face (visual)
        pr, pc, rr = int(p["perf_rows"]), int(p["perf_cols"]), float(p["perf_r"])
        for r in range(pr):
            z = pt + (r + 0.5) * (H - 2 * pt) / pr
            for c in range(pc):
                x = (c - (pc - 1) / 2) * (L - 2 * ps) / pc
                g.append(f'      <geom name="{name}_perf{k}_{r}_{c}" type="cylinder" size="{vec((rr, 0.0006))}" pos="{vec((x, sy*(W/2 + 0.0003), z))}" quat="0.70710678 0.70710678 0 0" rgba="{hc}" contype="0" conaffinity="0" mass="0"/>')
    yaw = np.radians(float(p["yaw_deg"])); q = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    floor = FLOOR_XML if p["include_floor"] else ""
    joint = "" if p["fixed_base"] else f'      <freejoint name="{name}_free"/>\n'
    xml = (mjcf_header(f"{name}_rack", timestep=p["timestep"]) + f"""
  <worldbody>{LIGHTS_XML}{floor}
    <body name="{name}" pos="{vec(p['pos'])}" quat="{vec(q)}">
{joint}{chr(10).join(g)}
    </body>
  </worldbody>
</mujoco>
""")
    xml_path = out_dir / f"{name}.xml"; xml_path.write_text(xml)
    m = load(xml_path); save_params(out_dir / f"{name}_params.json", p)
    built = {"xml": str(xml_path), "params_json": str(out_dir / f"{name}_params.json"), "name": name, "kind": "tuberack15", "catch_holes": [list(h) for h in p["catch_holes"]],
             "holes": [{"row": r, "col": c, "xy": hole_xy(p, r, c)} for r in range(int(p["rows"])) for c in range(int(p["cols"]))],
             "hole_top_z_m": hole_top_z(p), "mass_kg": round(float(m.body(name).subtreemass[0]), 4)}
    (out_dir / f"{name}_built.json").write_text(json.dumps(built, indent=1))
    return built


def construct_conical_tube_rack_lowpoly(out_dir, **params):
    """Build the rack (with its catch holes) into out_dir. Returns the build dictionary."""
    return build(params, out_dir)


if __name__ == "__main__":
    import json as _json
    import sys as _sys
    print(_json.dumps(construct_conical_tube_rack_lowpoly(_sys.argv[1] if len(_sys.argv) > 1 else "."), indent=1, default=str))
