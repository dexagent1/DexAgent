"""Construct a charging cable: an elastic MuJoCo cable laid along a centreline or as a flat coil, with a connector plug carried by the cord's end link itself (no extra body), sized to mate with a receptacle."""
from __future__ import annotations

TOOL = {
    "name": "construct_plug_cable",
    "title": "Construct a cable with a connector plug (USB-C, Lightning, micro-USB, barrel)",
    "type": "skill",
    "category": "Object Reconstruction",
    "summary": "A charging or data cable as an elastic MuJoCo cable laid along a measured centreline or a flat "
               "non-crossing coil, with a rigid connector plug (tip plus housing) on the end link itself (the rope's link "
               "graph stays one jointed chain) whose tip is sized from the mating port minus a clearance, so the plug "
               "can be inserted into a built receptacle.",
    "description": "Builds the cord with tool_library.common.cable (the construct_rope builder: inextensible capsule "
                   "chain, bending and twisting stiffness, self-collision, a mocap grip per end) laid at build time "
                   "along rest vertices, then adds the plug's geoms and sites to the chosen end segment's own body "
                   "(not a jointless child body, which deformable_topology_check rejects as a rigid part fused to the "
                   "rope): a box tip of "
                   "(port_w - 2 clearance) x (port_h - 2 clearance) x tip_len in front of a box housing that is "
                   "larger than the port so it stops at the receptacle face. The plug frame has +x along the "
                   "insertion direction (outward from the cord end), +y across the tip's wide side and +z across "
                   "its thin side; with plug_flat the plug is rolled about the cord axis so its wide side is "
                   "horizontal (lying flat on the table, as a plug at rest does) and the last vertices are raised "
                   "so the housing clears the table. Layout: either `centreline` (a polyline in the asset frame, "
                   "resampled to `segments` evenly spaced vertices) or `coil` (a flat spiral of `turns` turns "
                   "shrinking inward from an outer ellipse by `pitch_m` per turn, so the loops never cross, plus "
                   "an optional smooth tail from the outer end to `tail_end`, where the plug end lies). A coiled "
                   "cord whose loops cross in the video cannot be ordered from one mask; the non-crossing spiral "
                   "of the same outer extent and loop count is its physically equivalent resting state. Sites: "
                   "<name>_plug_tip (z axis = insertion direction, x across the tip's thin side, y across its wide side, at "
                   "the tip's front face: the site_a of the connector latch, add_connector_latch, which holds the plug "
                   "once it bottoms out in a construct_phone port) and <name>_plug_grip (housing centre). The returned dict keeps construct_rope's keys (xml, params_json, bodies, "
                   "grips, welds) so cable verifiers and stage_on_bench treat it like a rope; params_json holds "
                   "the cord-only parameters (rest vertices included) and plug_params_json every parameter.",
    "inputs": [
        {"name": "out_dir", "type": "path", "description": "Folder that receives <name>.xml, <name>_params.json and <name>_plug_cable_params.json."},
        {"name": "name", "type": "str", "description": "Asset and body prefix. Default cable"},
        {"name": "centreline", "type": "list[[x,y,z]] | None", "description": "Polyline (metres, asset frame) the cord is laid along; mutually exclusive with coil."},
        {"name": "coil", "type": "dict | None", "description": "Flat coil layout: center [x, y] and rx, ry (outer semi-axes, m) required; yaw_deg (0), turns (2.0), pitch_m (3 x radius), winding ('ccw' | 'cw'), end_angle_deg (where the spiral meets the outer ellipse, ellipse frame; default: toward tail_end), tail_end [x, y] (optional plug-end position), tail_dir [dx, dy] (direction the plug points at the tail end; default from the outer end to tail_end), z (default: radius)."},
        {"name": "segments", "type": "int", "description": "Vertex count of the laid cord (capsules = segments - 1). Default 60"},
        {"name": "radius", "type": "float", "description": "Cord radius, m. Default 0.0016 (a 3.2 mm USB cable)"},
        {"name": "rgba", "type": "list[4]", "description": "Cord colour. Default [0.92, 0.92, 0.9, 1]"},
        {"name": "density", "type": "float", "description": "Cord density, kg/m3 (copper wires in a jacket: about 2700). Default 2700"},
        {"name": "bend_stiffness", "type": "float", "description": "Young's modulus of the cable plugin, Pa. Default 5e6"},
        {"name": "twist_stiffness", "type": "float", "description": "Shear modulus of the cable plugin, Pa. Default 1e6"},
        {"name": "plug_end", "type": "str", "description": "'b' (last vertex) or 'a' (first vertex). Default b"},
        {"name": "port_w", "type": "float", "description": "Width of the receptacle opening the plug mates with, m. Default 0.0084 (USB-C)"},
        {"name": "port_h", "type": "float", "description": "Height of the receptacle opening, m. Default 0.0026 (USB-C)"},
        {"name": "tip_clearance_m", "type": "float", "description": "Gap between tip and opening on each side, m. Default 0.0001"},
        {"name": "tip_len", "type": "float", "description": "Tip length, m. Default 0.0066"},
        {"name": "housing_w", "type": "float", "description": "Housing width, m (must exceed port_w). Default 0.0122"},
        {"name": "housing_h", "type": "float", "description": "Housing height, m (must exceed port_h). Default 0.0062"},
        {"name": "housing_len", "type": "float", "description": "Housing length, m. Default 0.018"},
        {"name": "plug_mass", "type": "float", "description": "Tip plus housing mass, kg. Default 0.006"},
        {"name": "plug_rgba", "type": "list[4]", "description": "Housing colour. Default the cord colour"},
        {"name": "tip_rgba", "type": "list[4]", "description": "Tip colour. Default [0.72, 0.72, 0.74, 1]"},
        {"name": "plug_friction", "type": "list[3]", "description": "Friction of the plug geoms. Default [1.0, 0.01, 0.001]"},
        {"name": "plug_solref", "type": "list[2] | None", "description": "Contact solref of the plug geoms (a gram-scale plug in a tight port needs a short time constant, at least 2 timesteps). Default None (MuJoCo default)"},
        {"name": "plug_flat", "type": "bool", "description": "Roll the plug so its wide side is horizontal and raise the plug end so the housing clears z = 0. Default True"},
        {"name": "include_floor", "type": "bool", "description": "Floor plane for standalone tests (False for a scene asset). Default False"},
        {"name": "...", "type": "any", "description": "Every other construct_rope parameter (joint_damping, friction, condim, grips, timestep, cap_*, ...) is passed to the cord builder."},
    ],
    "outputs": [
        {"name": "built", "type": "dict", "description": "construct_rope's dict (xml, params_json, bodies {first, last, segments}, grips, welds, segment_length_m) plus plug {body (= the end link), end, tip_geom, housing_geom, tip_site, grip_site, tip_w, tip_h, tip_len, housing, clearance_m, axis_local, roll_deg, tip_pos, insert_dir}, plug_params_json, length_m, radius_m, centreline (laid vertices), min_self_gap_m and layout."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["cable", "usb-c", "plug", "connector", "charging", "coil", "deformable", "insertion", "object reconstruction"],
    "used_in": ["EgoDex charge_uncharge_device"],
    "status": "tested",
    "entrypoint": "construct_plug_cable",
    "cli": "python -m tool_library.skills.construct_plug_cable out_dir",
}

import json
import math
from pathlib import Path

import numpy as np

from tool_library.common import cable as _cable

CORD_OVERRIDES = {
    "name": "cable", "segments": 60, "radius": 0.0016, "rgba": [0.92, 0.92, 0.9, 1.0], "density": 2700.0,
    "bend_stiffness": 5.0e6, "twist_stiffness": 1.0e6, "pos": [0.0, 0.0, 0.0], "include_floor": False,
}
PLUG_DEFAULTS = {
    "centreline": None, "coil": None,
    "plug_end": "b", "port_w": 0.0084, "port_h": 0.0026, "tip_clearance_m": 0.0001, "tip_len": 0.0066,
    "housing_w": 0.0122, "housing_h": 0.0062, "housing_len": 0.018, "plug_mass": 0.006,
    "plug_rgba": None, "tip_rgba": [0.72, 0.72, 0.74, 1.0], "plug_friction": [1.0, 0.01, 0.001], "plug_flat": True,
    "plug_solref": None,
}
COIL_KEYS = {"center", "rx", "ry", "yaw_deg", "turns", "pitch_m", "winding", "end_angle_deg", "tail_end", "tail_dir", "z"}
DEFAULTS = {**_cable.DEFAULTS, **CORD_OVERRIDES, **PLUG_DEFAULTS}
DEFAULTS.pop("rest_vertices", None)


def _resample(P, n):
    P = np.asarray(P, float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-9])
    P = P[keep]
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    t = np.linspace(0.0, s[-1], int(n))
    return np.stack([np.interp(t, s, P[:, k]) for k in range(3)], axis=1)


def _coil_polyline(c, radius):
    unknown = sorted(set(c) - COIL_KEYS)
    if unknown:
        raise ValueError(f"construct_plug_cable: unknown coil key(s) {unknown}; allowed: {sorted(COIL_KEYS)}")
    for k in ("center", "rx", "ry"):
        if c.get(k) is None:
            raise ValueError(f"construct_plug_cable: coil needs '{k}'")
    cx, cy = (float(v) for v in c["center"])
    rx, ry = float(c["rx"]), float(c["ry"])
    turns = float(c.get("turns", 2.0))
    pitch = float(c.get("pitch_m") if c.get("pitch_m") is not None else 3.0 * radius)
    if turns <= 0:
        raise ValueError("construct_plug_cable: coil turns must be > 0")
    if pitch < 2.5 * radius:
        raise ValueError(f"construct_plug_cable: coil pitch_m {pitch:.4f} is under 2.5 x radius; neighbouring loops would touch (raise pitch_m)")
    if min(rx, ry) - pitch * turns < 2.0 * pitch:
        raise ValueError("construct_plug_cable: the coil's inner loop collapses (min(rx, ry) - pitch_m x turns < 2 pitch_m); lower turns or pitch_m")
    winding = str(c.get("winding", "ccw"))
    if winding not in ("ccw", "cw"):
        raise ValueError("construct_plug_cable: coil winding must be 'ccw' or 'cw'")
    sg = 1.0 if winding == "ccw" else -1.0
    yaw = math.radians(float(c.get("yaw_deg", 0.0)))
    Ry = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    z = float(c.get("z") if c.get("z") is not None else radius)
    tail_end = c.get("tail_end")
    if c.get("end_angle_deg") is not None:
        th_end = math.radians(float(c["end_angle_deg"]))
    elif tail_end is not None:
        q = Ry.T @ (np.asarray(tail_end, float) - [cx, cy])
        th_end = math.atan2(q[1] / ry, q[0] / rx) - sg * math.radians(30.0)
    else:
        th_end = 0.0
    n = max(200, int(400 * turns))
    t = np.linspace(0.0, turns, n)
    th = th_end - sg * 2 * math.pi * (turns - t)
    a, b = rx - pitch * (turns - t), ry - pitch * (turns - t)
    E = np.stack([a * np.cos(th), b * np.sin(th)], axis=1) @ Ry.T + [cx, cy]
    spiral = np.column_stack([E, np.full(len(E), z)])
    tail = np.zeros((0, 3))
    if tail_end is not None:
        P0 = spiral[-1, :2]
        T = spiral[-1, :2] - spiral[-2, :2]; T /= np.linalg.norm(T)
        P3 = np.asarray(tail_end, float)
        dist = float(np.linalg.norm(P3 - P0))
        if dist < 4 * radius:
            raise ValueError("construct_plug_cable: tail_end is on the coil's outer end; move tail_end or end_angle_deg")
        D = np.asarray(c["tail_dir"], float) if c.get("tail_dir") is not None else (P3 - P0)
        D = D / np.linalg.norm(D)
        P1, P2 = P0 + T * 0.4 * dist, P3 - D * 0.4 * dist
        u = np.linspace(0.0, 1.0, 120)[1:, None]
        B = ((1 - u) ** 3) * P0 + 3 * ((1 - u) ** 2) * u * P1 + 3 * (1 - u) * u ** 2 * P2 + u ** 3 * P3
        tail = np.column_stack([B, np.full(len(B), z)])
    return np.vstack([spiral, tail]), {"spiral_points": len(spiral), "tail_points": len(tail), "pitch_m": pitch,
                                       "turns": turns, "winding": winding, "end_angle_deg": math.degrees(th_end)}


def _min_self_gap(V, skip):
    V = np.asarray(V, float)
    best = np.inf
    for i in range(len(V)):
        j0 = i + skip
        if j0 >= len(V):
            break
        d = np.linalg.norm(V[j0:] - V[i], axis=1)
        best = min(best, float(d.min()))
    return best


def _quat_axis(axis, ang):
    s = math.sin(ang / 2)
    return [math.cos(ang / 2), axis[0] * s, axis[1] * s, axis[2] * s]


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def build(params=None, out_dir="."):
    import mujoco
    params = dict(params or {})
    unknown = sorted(set(params) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"construct_plug_cable: unknown parameter(s) {unknown}; allowed: {sorted(DEFAULTS)}")
    p = {**DEFAULTS, **params}
    name, r = str(p["name"]), float(p["radius"])
    if p["centreline"] is not None and p["coil"] is not None:
        raise ValueError("construct_plug_cable: give either centreline or coil, not both")
    if p["plug_end"] not in ("a", "b"):
        raise ValueError("construct_plug_cable: plug_end must be 'a' or 'b'")
    c2 = float(p["tip_clearance_m"])
    tip_w, tip_h = float(p["port_w"]) - 2 * c2, float(p["port_h"]) - 2 * c2
    hw, hh, hl, tl = float(p["housing_w"]), float(p["housing_h"]), float(p["housing_len"]), float(p["tip_len"])
    if min(tip_w, tip_h) <= 0:
        raise ValueError("construct_plug_cable: tip_clearance_m leaves no tip (port_w/port_h - 2 clearance <= 0)")
    if hw <= float(p["port_w"]) or hh <= float(p["port_h"]):
        raise ValueError("construct_plug_cable: housing_w/housing_h must exceed port_w/port_h so the housing stops at the receptacle face")
    if min(hl, tl, float(p["plug_mass"])) <= 0:
        raise ValueError("construct_plug_cable: housing_len, tip_len and plug_mass must be > 0")
    n = int(p["segments"])
    if n < 4:
        raise ValueError("construct_plug_cable: segments must be >= 4")

    layout = {"mode": "straight"}
    if p["coil"] is not None:
        poly, info = _coil_polyline(dict(p["coil"]), r)
        layout = {"mode": "coil", **info}
    elif p["centreline"] is not None:
        poly = np.asarray(p["centreline"], float)
        if poly.ndim != 2 or poly.shape[1] != 3 or len(poly) < 2:
            raise ValueError("construct_plug_cable: centreline must be a list of >= 2 [x, y, z] points")
        layout = {"mode": "centreline"}
    else:
        L = float(p["length"])
        poly = np.array([[0.0, 0.0, r], [L, 0.0, r]]) + np.asarray(p["pos"], float) * [1, 1, 0]
    V = _resample(poly, n)
    if p["plug_flat"]:
        zh = 0.5 * hh + 0.0003
        k = (-1, -2) if p["plug_end"] == "b" else (0, 1)
        k3 = -3 if p["plug_end"] == "b" else 2
        V[list(k), 2] = np.maximum(V[list(k), 2], zh)
        V[k3, 2] = max(V[k3, 2], 0.5 * (V[k3, 2] + zh))
    seg = np.linalg.norm(np.diff(V, axis=0), axis=1)
    if seg.min() < 2.2 * r:
        raise ValueError("construct_plug_cable: segments too many for this length (segment shorter than 2.2 x radius); lower segments")
    gap = _min_self_gap(V, skip=max(3, int(math.ceil(4 * r / seg.mean())) + 2))
    if gap < 2 * r + 0.0005:
        raise ValueError(f"construct_plug_cable: the laid cord comes within {gap * 1000:.1f} mm of itself (crossing or touching); "
                         "change the coil pitch_m / tail_end or the centreline")

    cord = {k: v for k, v in p.items() if k in _cable.DEFAULTS and k not in ("centreline",)}
    cord["rest_vertices"] = V.round(6).tolist()
    cord["centreline"] = V.round(6).tolist()
    cord["grip_rgba"] = p["grip_rgba"]
    built = _cable.build(cord, out_dir)
    xml_path = Path(built["xml"])

    # attach the plug to the end segment
    spec = mujoco.MjSpec.from_file(str(xml_path))
    end_body = built["bodies"]["last"] if p["plug_end"] == "b" else built["bodies"]["first"]
    m0 = spec.compile(); d0 = mujoco.MjData(m0); mujoco.mj_forward(m0, d0)
    R = d0.xmat[m0.body(end_body).id].reshape(3, 3)
    if p["plug_end"] == "b":
        L_last = float(np.linalg.norm(V[-1] - V[-2]))
        base_pos, flip = [L_last, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
        out_dir_local = np.array([1.0, 0.0, 0.0])
    else:
        base_pos, flip = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]       # 180 deg about z: plug +x = segment -x
        out_dir_local = np.array([-1.0, 0.0, 0.0])
    roll = 0.0
    if p["plug_flat"]:
        d_w = R @ out_dir_local
        w = np.cross([0.0, 0.0, 1.0], d_w)
        if np.linalg.norm(w) > 1e-6:
            w /= np.linalg.norm(w)
            Rf = np.eye(3) if p["plug_end"] == "b" else np.diag([-1.0, -1.0, 1.0])
            w_loc = (R @ Rf).T @ w                      # in the plug frame before the roll
            roll = math.atan2(w_loc[2], w_loc[1])
    quat = _qmul(flip, _quat_axis([1.0, 0.0, 0.0], roll))
    Rq = np.zeros(9); mujoco.mju_quat2Mat(Rq, np.asarray(quat, float)); Rq = Rq.reshape(3, 3)
    base = np.asarray(base_pos, float)

    def at(local):                                   # plug-frame point -> end-link frame
        return (base + Rq @ np.asarray(local, float)).tolist()

    eb = spec.body(end_body)
    vol_h, vol_t = hw * hh * hl, tip_w * tip_h * tl
    dens = float(p["plug_mass"]) / (vol_h + vol_t)
    prgba = p["plug_rgba"] if p["plug_rgba"] is not None else p["rgba"]
    gh = eb.add_geom(); gh.name = f"{name}_plug_housing"; gh.type = mujoco.mjtGeom.mjGEOM_BOX
    gh.size = [hl / 2, hw / 2, hh / 2]; gh.pos = at([hl / 2, 0, 0]); gh.quat = list(quat); gh.density = dens
    gh.rgba = [float(v) for v in prgba]; gh.friction = [float(v) for v in p["plug_friction"]]; gh.condim = 4
    gt = eb.add_geom(); gt.name = f"{name}_plug_tip"; gt.type = mujoco.mjtGeom.mjGEOM_BOX
    gt.size = [tl / 2, tip_w / 2, tip_h / 2]; gt.pos = at([hl + tl / 2, 0, 0]); gt.quat = list(quat); gt.density = dens
    gt.rgba = [float(v) for v in p["tip_rgba"]]; gt.friction = [float(v) for v in p["plug_friction"]]; gt.condim = 4
    if p["plug_solref"] is not None:
        if len(p["plug_solref"]) != 2:
            raise ValueError("construct_plug_cable: plug_solref must be [timeconst, dampratio]")
        gh.solref = [float(v) for v in p["plug_solref"]]; gt.solref = [float(v) for v in p["plug_solref"]]
    st = eb.add_site(); st.name = f"{name}_plug_tip"; st.pos = at([hl + tl, 0, 0])
    st.quat = _qmul(list(quat), [math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]); st.size = [0.0015, 0, 0]; st.group = 4
    sf = eb.add_site(); sf.name = f"{name}_plug_frame"; sf.pos = at([0, 0, 0]); sf.quat = list(quat); sf.size = [0.001, 0, 0]; sf.group = 4
    sg = eb.add_site(); sg.name = f"{name}_plug_grip"; sg.pos = at([hl / 2, 0, 0]); sg.quat = list(quat); sg.size = [0.002, 0, 0]; sg.group = 4
    xml_path.write_text(spec.to_xml())

    m = mujoco.MjModel.from_xml_path(str(xml_path)); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    tip_pos = d.site_xpos[m.site(f"{name}_plug_tip").id].copy()
    Rp = d.site_xmat[m.site(f"{name}_plug_frame").id].reshape(3, 3)
    zmin = np.inf
    for g in (f"{name}_plug_housing", f"{name}_plug_tip"):
        gid = m.geom(g).id
        Rg = d.geom_xmat[gid].reshape(3, 3); hs = m.geom_size[gid]
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * hs
        zmin = min(zmin, float((corners @ Rg.T + d.geom_xpos[gid])[:, 2].min()))
    full = {k: v for k, v in p.items()}
    full["centreline_laid"] = V.round(6).tolist()
    pj = Path(out_dir) / f"{name}_plug_cable_params.json"
    pj.write_text(json.dumps(full, indent=1, default=float))
    built.update({
        "plug_params_json": str(pj), "length_m": round(float(seg.sum()), 4), "radius_m": r,
        "centreline": V.round(5).tolist(), "min_self_gap_m": round(gap, 4), "layout": layout,
        "plug": {"body": end_body, "end": p["plug_end"], "end_body": end_body, "frame_site": f"{name}_plug_frame",
                 "tip_geom": f"{name}_plug_tip", "housing_geom": f"{name}_plug_housing",
                 "tip_site": f"{name}_plug_tip", "grip_site": f"{name}_plug_grip",
                 "tip_w": round(tip_w, 6), "tip_h": round(tip_h, 6), "tip_len": tl,
                 "housing": [hl, hw, hh], "clearance_m": c2, "mates_port": [float(p["port_w"]), float(p["port_h"])],
                 "axis_local": [1.0, 0.0, 0.0], "roll_deg": round(math.degrees(roll), 2),
                 "tip_pos": tip_pos.round(5).tolist(), "insert_dir": Rp[:, 0].round(4).tolist(),
                 "wide_axis": Rp[:, 1].round(4).tolist(), "min_z_m": round(zmin, 5), "mass_kg": float(p["plug_mass"])},
    })
    return built


def construct_plug_cable(out_dir, **params):
    """Build the cable with its plug into out_dir. Returns the build dictionary."""
    return build(params, out_dir)


if __name__ == "__main__":
    import sys
    print(json.dumps(construct_plug_cable(sys.argv[1] if len(sys.argv) > 1 else "."), indent=1, default=str))
