"""Construct rack for 15 ml conical tubes: An open-frame rack for 15 ml conical tubes: top plate with round holes, catch plate, posts and perforated walls, every touchable part a box collider.

OPEN-FRAME RACK FOR 15 mL CONICAL TUBES — the orange plastic rack of the biology bench.

Not the 80-well microtube block (`labtube.build_tube_rack`): this is a moulded open frame — a TOP plate with one or two
rows of large round holes (a 15 mL tube's 17 mm body drops through and hangs by its cone/flange), a BOTTOM plate that
catches the cones, four corner posts, and side walls perforated by a grid of small round drainage holes. Everything a
robot can touch is a box collider (plates, posts, walls); the perforations and the hole rims are visual only.

Footprint at z = 0 (the bottom plate's underside), length along local +x, `yaw_deg` CCW about z. Holes are indexed
(row, col); `hole_xy(p, row, col)` gives a hole centre in the rack frame, `hole_top_z(p)` the top plate's upper face.

    python -m tool_library.skills.construct_conical_tube_rack [out_dir]     # build the default
"""
from __future__ import annotations

TOOL = {'name': 'construct_conical_tube_rack',
 'title': 'Construct rack for 15 ml conical tubes',
 'type': 'skill',
 'category': 'Object Reconstruction',
 'summary': 'An open-frame rack for 15 ml conical tubes: top plate with round holes, catch '
            'plate, posts and perforated walls, every touchable part a box collider.',
 'description': 'Builds the moulded open frame as box colliders (plates, posts, walls) with '
                'visual-only perforations and hole rims, from its footprint, height, hole grid '
                'and colours. Holes are indexed by row and column; hole_xy() and hole_top_z() '
                'return where a tube stands so a trajectory can address a well by index. The '
                'bottom plate is flat: something standing leaning in a well (a pipette, a cone '
                'tip) is held sideways by friction only and creeps out within seconds under '
                "MuJoCo's soft friction; when an object must stand in a well like that, or a robot "
                'hand works near the rack (its top plate is four boxes per well), use '
                'construct_conical_tube_rack_lowpoly (merged top plate, catch holes in the bottom '
                'plate under chosen wells) instead.',
 'inputs': [{'name': 'out_dir',
             'type': 'path',
             'description': 'Folder that receives the MJCF, its meshes and '
                            '<name>_params.json.'},
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
 'tags': ['rack', 'lab', 'parametric', 'reconstruct', 'object reconstruction'],
 'used_in': ['Biology Experiment'],
 'status': 'tested',
 'entrypoint': 'construct_conical_tube_rack',
 'cli': 'python -m tool_library.skills.construct_conical_tube_rack out_dir'}

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
}


def hole_xy(p: dict, row: int, col: int):
    x = (col - (int(p["cols"]) - 1) / 2) * float(p["pitch_x"])
    y = (row - (int(p["rows"]) - 1) / 2) * float(p["pitch_y"])
    return float(x), float(y)


def hole_top_z(p: dict) -> float:
    return float(p["height"])


def build(params: dict | None = None, out_dir: str | Path = ".") -> dict:
    p = merge_params(DEFAULTS, params)
    name = str(p["name"]); out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    L, W, H, pt, ps, wt = (float(p[k]) for k in ("length", "width", "height", "plate_t", "post", "wall_t"))
    oc, hc = rgba(p["rack_rgba"]), rgba(p["perf_rgba"])
    ga = f'density="{p["density"]}" friction="{vec(p["friction"])}" condim="3"'
    g = []
    # bottom plate and the top plate as strips BETWEEN the holes (so the holes are real openings)
    g.append(f'      <geom name="{name}_bottom" type="box" size="{vec((L/2, W/2, pt/2))}" pos="{vec((0, 0, pt/2))}" rgba="{oc}" {ga}/>')
    rows, cols = int(p["rows"]), int(p["cols"])
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be >= 1")
    xs = sorted(set(hole_xy(p, 0, c)[0] for c in range(cols)))
    ys = sorted(set(hole_xy(p, r, 0)[1] for r in range(rows)))
    edges = [-L / 2] + [0.5 * (a + b) for a, b in zip(xs[:-1], xs[1:])] + [L / 2]
    yedges = [-W / 2] + [0.5 * (a + b) for a, b in zip(ys[:-1], ys[1:])] + [W / 2]
    hr = float(p["hole_r"]); zt = H - pt / 2
    if ys[0] - hr <= -W / 2 + wt or ys[-1] + hr >= W / 2 - wt:
        raise ValueError(f"{rows} row(s) of {hr*1e3:.1f} mm holes at pitch_y {float(p['pitch_y'])*1e3:.1f} mm do not fit between "
                         f"the side walls of a {W*1e3:.0f} mm wide rack with {wt*1e3:.1f} mm walls: shrink hole_r or pitch_y, "
                         f"or widen the rack")
    # the top plate as strips around every hole of the grid (so each hole is a real opening): per cell two bars in
    # front of and behind the hole over the cell's width, and two between the hole and the cell's side edges
    for i, xc in enumerate(xs):
        x0, x1 = edges[i], edges[i + 1]
        if xc - hr - x0 <= 1e-4 or x1 - xc - hr <= 1e-4:
            raise ValueError(f"hole column {i} (hole_r {hr*1e3:.1f} mm) leaves no top plate between holes or at the "
                             f"rack end: pitch_x {float(p['pitch_x'])*1e3:.1f} mm, length {L*1e3:.0f} mm — "
                             f"shrink hole_r, raise pitch_x or lengthen the rack")
        for j, yc in enumerate(ys):
            y0, y1 = yedges[j], yedges[j + 1]
            if yc - hr - y0 <= 1e-4 or y1 - yc - hr <= 1e-4:
                raise ValueError(f"hole row {j} (hole_r {hr*1e3:.1f} mm) leaves no top plate between rows: raise pitch_y "
                                 f"or shrink hole_r")
            sfx = f"{i}" if rows == 1 else f"{j}_{i}"
            g.append(f'      <geom name="{name}_topA{sfx}" type="box" size="{vec(((x1-x0)/2, (yc - hr - y0)/2, pt/2))}" pos="{vec(((x0+x1)/2, (y0 + yc - hr)/2, zt))}" rgba="{oc}" {ga}/>')
            g.append(f'      <geom name="{name}_topB{sfx}" type="box" size="{vec(((x1-x0)/2, (y1 - yc - hr)/2, pt/2))}" pos="{vec(((x0+x1)/2, (yc + hr + y1)/2, zt))}" rgba="{oc}" {ga}/>')
            g.append(f'      <geom name="{name}_topL{sfx}" type="box" size="{vec(((xc - hr - x0)/2, hr, pt/2))}" pos="{vec(((x0 + xc - hr)/2, yc, zt))}" rgba="{oc}" {ga}/>')
            g.append(f'      <geom name="{name}_topR{sfx}" type="box" size="{vec(((x1 - xc - hr)/2, hr, pt/2))}" pos="{vec(((xc + hr + x1)/2, yc, zt))}" rgba="{oc}" {ga}/>')
            # moulded ring round the hole (visual)
            g.append(f'      <geom name="{name}_rim{sfx}" type="cylinder" size="{vec((float(p["rim_r"]), float(p["rim_h"])/2))}" pos="{vec((xc, yc, H + float(p["rim_h"])/2))}" rgba="{oc}" contype="0" conaffinity="0" mass="0"/>')
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
    built = {"xml": str(xml_path), "params_json": str(out_dir / f"{name}_params.json"), "name": name, "kind": "tuberack15",
             "holes": [{"row": r, "col": c, "xy": hole_xy(p, r, c)} for r in range(int(p["rows"])) for c in range(int(p["cols"]))],
             "hole_top_z_m": hole_top_z(p), "mass_kg": round(float(m.body(name).subtreemass[0]), 4)}
    (out_dir / f"{name}_built.json").write_text(json.dumps(built, indent=1))
    return built


def construct_conical_tube_rack(out_dir, **params):
    """Build the rack into out_dir. Returns the build dictionary."""
    return build(params, out_dir)


if __name__ == "__main__":
    import json as _json
    import sys as _sys
    print(_json.dumps(construct_conical_tube_rack(_sys.argv[1] if len(_sys.argv) > 1 else "."), indent=1, default=str))
