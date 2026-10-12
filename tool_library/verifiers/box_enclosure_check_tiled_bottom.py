"""Is this container a box: a bottom (possibly tiled from several slabs), four walls and a hollow inside?"""
from __future__ import annotations

TOOL = {
    "name": "box_enclosure_check_tiled_bottom",
    "title": "Box enclosure check (tiled bottom)",
    "type": "verifier",
    "category": "Scene Compile Checks",
    "summary": "A box-like object must enclose a cavity: rays from the centre of its interior must hit its own bottom and all four side walls, the interior must be hollow, and the top is reported open or closed; the bottom may be tiled from several thin slabs (a rounded-rectangle dish), whose union sets the interior extent.",
    "description": (
        "Fork of box_enclosure_check for containers whose floor is not one box geom but a tiling of thin "
        "horizontal slabs (a rounded-rectangle glass dish built from a centre slab plus side strips, a tray "
        "with a segmented base). Collects the object's bodies (the named body with its subtree, or every body "
        "with the name as prefix) and their visible or collidable geoms, expresses every box geom in the root "
        "body frame, finds the largest thin horizontal box near the lowest point, then takes as the bottom the "
        "UNION of every thin horizontal box whose underside lies within `tile_tol` of that slab's underside and "
        "whose top is within `tile_tol` of its top. The interior extent is the union's footprint and the wall "
        "height is the tallest geom rising from inside it. Hollowness: a 7x7x5 grid of interior points (extent "
        "shrunk by `wall_margin`) is tested for containment in the object's primitive geoms; the cavity fraction "
        "must exceed `min_cavity`. Faces: rays from the interior centre along -z, +x, -x, +y, -y (object frame) "
        "must hit one of the object's geoms before leaving the extent (+2 cm); +z tells whether the top is "
        "closed, checked against `expect_top`. Reports face hits with distances, cavity fraction, interior "
        "size, the bottom tiles used, and one issue per missing face."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF file, XML string, or compiled model (an asset or the whole scene)."},
        {"name": "object", "type": "str", "description": "Root body name or name prefix of the box."},
        {"name": "min_walls", "type": "int", "description": "Side walls required (of 4). Default 4."},
        {"name": "require_bottom", "type": "bool", "description": "Default True."},
        {"name": "expect_top", "type": "str | None", "description": "'open', 'closed' or None (not checked)."},
        {"name": "min_cavity", "type": "float", "description": "Fraction of interior sample points that must be empty. Default 0.6."},
        {"name": "wall_margin", "type": "float, metres", "description": "Shrink of the extent for the interior samples. Default 0.012."},
        {"name": "tile_tol", "type": "float, metres", "description": "How close in height another slab must be to join the bottom tiling. Default 0.003."},
        {"name": "qpos", "type": "list | None", "description": "State to evaluate; default keyframe 0 or qpos0."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; faces: {bottom, x_pos, x_neg, y_pos, y_neg, top: {hit, geom, distance_m}}; cavity_fraction; interior_size_m; bottom_tiles; n_geoms; top_state; walls_found."},
    ],
    "status": "tested",
    "entrypoint": "box_enclosure_check_tiled_bottom",
    "requirements": ["mujoco>=3.1", "numpy"],
    "tags": ["box", "container", "enclosure", "walls", "cavity", "rounded rectangle", "dish", "stage 2"],
    "used_in": [],
    "cli": "python -m tool_library.verifiers.box_enclosure_check_tiled_bottom scene.xml box_body",
}

import json
import sys

import mujoco
import numpy as np

_KNOWN_TOP = (None, "open", "closed")


def _load(scene):
    if isinstance(scene, mujoco.MjModel):
        return scene
    s = str(scene)
    if s.lstrip().startswith("<"):
        return mujoco.MjModel.from_xml_string(s)
    return mujoco.MjModel.from_xml_path(s)


def _forward(model, qpos):
    data = mujoco.MjData(model)
    if qpos is not None:
        q = np.asarray(qpos, float).ravel()
        if q.size != model.nq:
            raise ValueError(f"qpos has {q.size} values, the model has nq={model.nq}; pass a full state")
        data.qpos[:] = q
    elif model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return data


def _subtree(model, bid):
    out = {bid}
    changed = True
    while changed:
        changed = False
        for b in range(1, model.nbody):
            if b not in out and int(model.body_parentid[b]) in out:
                out.add(b); changed = True
    return out


def _object_bodies(model, name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    out = set()
    if bid >= 0:
        out |= _subtree(model, bid)
    for b in range(1, model.nbody):
        n = model.body(b).name
        if n == name or n.startswith(name + "_"):
            out.add(b)
    return out, bid


def _inside(model, data, g, P):
    t = int(model.geom_type[g]); s = model.geom_size[g]
    R = data.geom_xmat[g].reshape(3, 3); c = data.geom_xpos[g]
    L = (P - c) @ R
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        return (np.abs(L) <= s[:3] + 1e-6).all(axis=1)
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return np.linalg.norm(L, axis=1) <= s[0]
    if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return (np.hypot(L[:, 0], L[:, 1]) <= s[0]) & (np.abs(L[:, 2]) <= s[1])
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        z = np.clip(L[:, 2], -s[1], s[1])
        return np.linalg.norm(L - np.c_[np.zeros(len(L)), np.zeros(len(L)), z], axis=1) <= s[0]
    if t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        return ((L / s[:3]) ** 2).sum(axis=1) <= 1.0
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        return np.linalg.norm(L, axis=1) <= 0.5 * float(model.geom_rbound[g])
    return np.zeros(len(P), bool)


def box_enclosure_check_tiled_bottom(scene, object, min_walls=4, require_bottom=True, expect_top=None,
                                     min_cavity=0.6, wall_margin=0.012, tile_tol=0.003, qpos=None):
    if expect_top not in _KNOWN_TOP:
        raise ValueError(f"expect_top must be one of {_KNOWN_TOP}, got {expect_top!r}")
    if not (0 <= int(min_walls) <= 4):
        raise ValueError("min_walls must be between 0 and 4")
    if float(tile_tol) < 0 or float(wall_margin) < 0:
        raise ValueError("tile_tol and wall_margin must be non-negative")
    model = _load(scene)
    data = _forward(model, qpos)
    bodies, rb = _object_bodies(model, object)
    if not bodies:
        return {"ok": False, "issues": [f"no body named or prefixed {object!r}; pass the box's root body name"]}
    geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies and model.geom_rgba[g][3] > 0.05
             and (model.geom_contype[g] or model.geom_conaffinity[g] or model.geom_group[g] <= 2)]
    if not geoms:
        return {"ok": False, "issues": [f"{object!r} has no visible or collidable geoms"]}
    if rb < 0:
        rb = min(bodies)
    R = data.xmat[rb].reshape(3, 3); c0 = data.xpos[rb].copy()
    boxes = []
    for g in geoms:
        if int(model.geom_type[g]) != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        Rg = R.T @ data.geom_xmat[g].reshape(3, 3); ext = np.abs(Rg) @ model.geom_size[g][:3]
        p = (data.geom_xpos[g] - c0) @ R
        boxes.append((g, p, ext))
    if not boxes:
        return {"ok": False, "issues": [f"{object!r} has no box geoms; this check reads boxes built from panels"], "n_geoms": len(geoms)}
    zmin = min(p[2] - e[2] for _, p, e in boxes); zmax = max(p[2] + e[2] for _, p, e in boxes)
    thin = [(g, p, e) for g, p, e in boxes if e[2] < 0.35 * e[0] and e[2] < 0.35 * e[1]]
    cands = [(g, p, e) for g, p, e in thin if p[2] - e[2] < zmin + 0.25 * (zmax - zmin)]
    if not cands:
        return {"ok": False, "issues": [f"{object!r}: no thin horizontal panel near the lowest point that could be the bottom"], "n_geoms": len(geoms)}
    gb, pb, eb = max(cands, key=lambda t: t[2][0] * t[2][1])
    lo_b, hi_b = pb[2] - eb[2], pb[2] + eb[2]
    # the bottom tiling: thin slabs at the same height band as the main slab and touching its footprint chain
    tiles = [(gb, pb, eb)]
    pool = [(g, p, e) for g, p, e in boxes if g != gb and abs((p[2] - e[2]) - lo_b) <= tile_tol
            and abs((p[2] + e[2]) - hi_b) <= tile_tol and e[2] < 0.5 * max(e[0], e[1])]
    grown = True
    while grown:
        grown = False
        for t in list(pool):
            g, p, e = t
            if any(abs(p[0] - q[0]) <= e[0] + f[0] + 1e-3 and abs(p[1] - q[1]) <= e[1] + f[1] + 1e-3 for _, q, f in tiles):
                tiles.append(t); pool.remove(t); grown = True
    xlo = min(p[0] - e[0] for _, p, e in tiles); xhi = max(p[0] + e[0] for _, p, e in tiles)
    ylo = min(p[1] - e[1] for _, p, e in tiles); yhi = max(p[1] + e[1] for _, p, e in tiles)
    tile_ids = {g for g, _, _ in tiles}
    bottom_top = hi_b
    tops = [p[2] + e[2] for g, p, e in boxes if g not in tile_ids and xlo - 0.005 <= p[0] <= xhi + 0.005
            and ylo - 0.005 <= p[1] <= yhi + 0.005 and p[2] + e[2] > bottom_top + 0.005]
    wall_h = (max(tops) - bottom_top) if tops else 0.02
    lo = np.array([xlo, ylo, bottom_top]); hi = np.array([xhi, yhi, bottom_top + wall_h])
    size = hi - lo
    centre_local = np.array([(xlo + xhi) / 2, (ylo + yhi) / 2, bottom_top + 0.5 * wall_h])
    lo[2] = lo_b
    m = min(float(wall_margin), 0.3 * float(min(size[0], size[1])), 0.3 * wall_h)
    xs = np.linspace(lo[0] + m, hi[0] - m, 7); ys = np.linspace(lo[1] + m, hi[1] - m, 7)
    zs = np.linspace(bottom_top + m, hi[2] - m, 5)
    G = np.array(np.meshgrid(xs, ys, zs, indexing="ij")).reshape(3, -1).T
    Wp = c0 + G @ R.T
    occupied = np.zeros(len(Wp), bool)
    for g in geoms:
        occupied |= _inside(model, data, g, Wp)
    cavity = float(1.0 - occupied.mean())
    cw = c0 + R @ centre_local
    geom_set = set(geoms)
    dirs = {"bottom": -R[:, 2], "x_pos": R[:, 0], "x_neg": -R[:, 0], "y_pos": R[:, 1], "y_neg": -R[:, 1], "top": R[:, 2]}
    reach = {"bottom": centre_local[2] - lo[2], "x_pos": hi[0] - centre_local[0], "x_neg": centre_local[0] - lo[0],
             "y_pos": hi[1] - centre_local[1], "y_neg": centre_local[1] - lo[1], "top": max(hi[2] - centre_local[2], 0.05)}
    gid = np.zeros(1, np.int32)

    def cast(origin, v, budget):
        o = np.array(origin, np.float64); travelled = 0.0; last = None
        for _ in range(8):
            dist = mujoco.mj_ray(model, data, o, np.asarray(v, np.float64), None, 1, -1, gid)
            if dist < 0:
                return None, last
            travelled += float(dist); last = model.geom(int(gid[0])).name
            if int(gid[0]) in geom_set:
                return travelled, last
            if travelled > budget + 0.02:
                return None, last
            o = o + np.asarray(v) * (dist + 1e-4)
        return None, last

    faces = {}
    for k, v in dirs.items():
        dist, gname = cast(cw, v, reach[k])
        hit = dist is not None and dist <= reach[k] + 0.02
        faces[k] = {"hit": bool(hit), "geom": gname, "distance_m": round(float(dist), 4) if dist is not None else None}
    issues = []
    if any(_inside(model, data, g, cw[None])[0] for g in geoms):
        issues.append(f"{object!r}: the interior centre lies inside a geom; this is a solid block, not a box with a cavity")
    if cavity < float(min_cavity):
        issues.append(f"{object!r}: only {cavity:.0%} of the interior is empty (need {float(min_cavity):.0%}); a box encloses a hollow space")
    if require_bottom and not faces["bottom"]["hit"]:
        issues.append(f"{object!r}: no bottom face under the interior")
    walls = [k for k in ("x_pos", "x_neg", "y_pos", "y_neg") if faces[k]["hit"]]
    if len(walls) < int(min_walls):
        missing = [k for k in ("x_pos", "x_neg", "y_pos", "y_neg") if k not in walls]
        issues.append(f"{object!r}: {len(walls)} of 4 side walls found; missing {missing}. Add the missing walls to the asset")
    top_state = "closed" if faces["top"]["hit"] else "open"
    if expect_top in ("open", "closed") and top_state != expect_top:
        issues.append(f"{object!r}: top is {top_state}, expected {expect_top}")
    return {"ok": not issues, "issues": issues, "faces": faces, "cavity_fraction": round(cavity, 3), "top_state": top_state,
            "interior_size_m": [round(float(v), 4) for v in size],
            "bottom_tiles": [model.geom(g).name for g in sorted(tile_ids)],
            "n_geoms": len(geoms), "walls_found": walls}


def main(argv=None):
    a = argv if argv is not None else sys.argv[1:]
    if len(a) < 2:
        print(__doc__); return 2
    r = box_enclosure_check_tiled_bottom(a[0], a[1])
    print(json.dumps(r, indent=1)); return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
