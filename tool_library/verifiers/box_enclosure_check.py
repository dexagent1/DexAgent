"""Is this container a box: a bottom, four walls and a hollow inside, not a slab with a lid?"""
from __future__ import annotations

TOOL = {
    "name": "box_enclosure_check",
    "title": "Box enclosure check",
    "type": "verifier",
    "category": "Scene Compile Checks",
    "summary": "A box-like object must enclose a cavity: from the centre of its interior, rays must hit its own bottom and all four side walls within its extent, the interior must be hollow (sample points not inside any of its geoms), and the top is reported as open or closed (a lid); a slab with a hinged panel, or a solid block, fails with the missing faces named.",
    "description": (
        "Collects the object's bodies (the named body with its subtree, or every body with the name as prefix) "
        "and their collidable or visible geoms, builds the object's axis-aligned extent in its root body frame, "
        "and takes the interior centre at mid height. Hollowness: a grid of interior sample points (the extent "
        "shrunk by `wall_margin`) is tested for containment in the object's box, cylinder, sphere and capsule "
        "geoms (mesh geoms by their bounding sphere); the cavity fraction must exceed `min_cavity`. Faces: rays "
        "cast from the centre along -z, +x, -x, +y, -y in the object frame must hit one of the object's geoms "
        "before leaving the extent; +z tells whether the top is closed. With `expect_top` ('open' | 'closed' | "
        "None) the top state is also checked. Reports the face hits with distances, the cavity fraction, the "
        "interior size, and one issue per missing face. Meant for cardboard boxes, trays, drawers, bins and gift "
        "boxes; a cup is a vessel (vessel_containment_test), not a box."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF file, XML string, or compiled model (an asset or the whole scene)."},
        {"name": "object", "type": "str", "description": "Root body name or name prefix of the box."},
        {"name": "min_walls", "type": "int", "description": "Side walls required (of 4). Default 4."},
        {"name": "require_bottom", "type": "bool", "description": "Default True."},
        {"name": "expect_top", "type": "str | None", "description": "'open', 'closed' or None (not checked)."},
        {"name": "min_cavity", "type": "float", "description": "Fraction of interior sample points that must be empty. Default 0.6."},
        {"name": "wall_margin", "type": "float, metres", "description": "Shrink of the extent for the interior samples. Default 0.012."},
        {"name": "qpos", "type": "list | None", "description": "State to evaluate; default keyframe 0 or qpos0."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; faces: {bottom, x_pos, x_neg, y_pos, y_neg, top: {hit, geom, distance_m}}; cavity_fraction; interior_size_m; n_geoms; top_state."},
    ],
    "status": "tested",
    "entrypoint": "box_enclosure_check",
    "requirements": ["mujoco>=3.1", "numpy"],
    "tags": ["box", "container", "enclosure", "walls", "cavity", "stage 2"],
    "used_in": ["Place Toy in Drawer", "Giftbox Packing", "Computer Installation"],
    "cli": "python -m tool_library.verifiers.box_enclosure_check scene.xml box_box",
}

import json
import sys

import mujoco
import numpy as np

from tool_library.common import scene as S


def _object_bodies(model, name):
    bid = S.resolve_body(model, name)
    out = set()
    if bid is not None and bid >= 0:
        out |= S.subtree(model, bid)
    for b in range(1, model.nbody):
        n = model.body(b).name
        if n == name or n.startswith(name + "_"):
            out.add(b)
    return out


def _inside(model, data, g, P):
    """Boolean mask: which world points P lie inside geom g."""
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


def box_enclosure_check(scene, object, min_walls=4, require_bottom=True, expect_top=None, min_cavity=0.6, wall_margin=0.012, qpos=None):
    model = S.load_model(scene)
    data = S.forward(model, qpos)
    bodies = _object_bodies(model, object)
    if not bodies:
        return {"ok": False, "issues": [f"no body named or prefixed {object!r}"]}
    geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies and model.geom_rgba[g][3] > 0.05
             and (model.geom_contype[g] or model.geom_conaffinity[g] or model.geom_group[g] <= 2)]
    if not geoms:
        return {"ok": False, "issues": [f"{object!r} has no visible or collidable geoms"]}
    rb = S.resolve_body(model, object)
    if rb is None or rb < 0:
        rb = min(bodies)
    R = data.xmat[rb].reshape(3, 3); c0 = data.xpos[rb]
    # the bottom: the largest thin horizontal box geom; the interior sits on it, inside the walls that rise from it
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
    cands = [(g, p, e) for g, p, e in boxes if e[2] < 0.35 * e[0] and e[2] < 0.35 * e[1] and p[2] - e[2] < zmin + 0.25 * (zmax - zmin)]
    if not cands:
        return {"ok": False, "issues": [f"{object!r}: no thin horizontal panel near the lowest point that could be the bottom"], "n_geoms": len(geoms)}
    gb, pb, eb = max(cands, key=lambda t: t[2][0] * t[2][1])
    bottom_top = pb[2] + eb[2]
    # walls: geoms whose footprint centre lies inside the bottom's footprint and that rise above it
    tops = [p[2] + e[2] for g, p, e in boxes if g != gb and abs(p[0] - pb[0]) <= eb[0] + 0.005 and abs(p[1] - pb[1]) <= eb[1] + 0.005 and p[2] + e[2] > bottom_top + 0.005]
    wall_h = (max(tops) - bottom_top) if tops else 0.02
    lo = np.array([pb[0] - eb[0], pb[1] - eb[1], bottom_top]); hi = np.array([pb[0] + eb[0], pb[1] + eb[1], bottom_top + wall_h])
    size = hi - lo
    centre_local = np.array([pb[0], pb[1], bottom_top + 0.5 * wall_h])
    lo[2] = pb[2] - eb[2]                                     # the extent used for the bottom ray includes the plate
    # hollowness: interior samples
    m = min(float(wall_margin), 0.3 * float(min(size[0], size[1])), 0.3 * wall_h)
    xs = np.linspace(lo[0] + m, hi[0] - m, 7); ys = np.linspace(lo[1] + m, hi[1] - m, 7); zs = np.linspace(bottom_top + m, hi[2] - m, 5)
    G = np.array(np.meshgrid(xs, ys, zs, indexing="ij")).reshape(3, -1).T
    W = c0 + G @ R.T
    occupied = np.zeros(len(W), bool)
    for g in geoms:
        occupied |= _inside(model, data, g, W)
    cavity = float(1.0 - occupied.mean())
    # faces by ray casting from the interior centre
    cw = c0 + R @ centre_local
    geom_set = set(geoms)
    dirs = {"bottom": -R[:, 2], "x_pos": R[:, 0], "x_neg": -R[:, 0], "y_pos": R[:, 1], "y_neg": -R[:, 1], "top": R[:, 2]}
    reach = {"bottom": centre_local[2] - lo[2], "x_pos": hi[0] - centre_local[0], "x_neg": centre_local[0] - lo[0],
             "y_pos": hi[1] - centre_local[1], "y_neg": centre_local[1] - lo[1], "top": max(hi[2] - centre_local[2], 0.05)}
    faces = {}
    gid = np.zeros(1, np.int32)
    def cast(origin, v, budget):
        """First hit on one of the object's own geoms along v, looking through contents and other bodies."""
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
            o = o + np.asarray(v) * (dist + 1e-4)           # step past the foreign geom (a watch in the box, a hand)
        return None, last
    for k, v in dirs.items():
        dist, gname = cast(cw, v, reach[k])
        hit = dist is not None and dist <= reach[k] + 0.02
        faces[k] = {"hit": bool(hit), "geom": gname, "distance_m": round(float(dist), 4) if dist is not None else None}
    issues = []
    centre_inside = bool(any(_inside(model, data, g, cw[None])[0] for g in geoms))
    if centre_inside:
        issues.append(f"{object!r}: the interior centre lies inside a geom; this is a solid block, not a box with a cavity")
    if cavity < float(min_cavity):
        issues.append(f"{object!r}: only {cavity:.0%} of the interior is empty (need {float(min_cavity):.0%}); a box encloses a hollow space")
    if require_bottom and not faces["bottom"]["hit"]:
        issues.append(f"{object!r}: no bottom face under the interior")
    walls = [k for k in ("x_pos", "x_neg", "y_pos", "y_neg") if faces[k]["hit"]]
    if len(walls) < int(min_walls):
        missing = [k for k in ("x_pos", "x_neg", "y_pos", "y_neg") if k not in walls]
        issues.append(f"{object!r}: {len(walls)} of 4 side walls found; missing {missing}. A box has a bottom and four walls, not a panel on a hinge")
    top_state = "closed" if faces["top"]["hit"] else "open"
    if expect_top in ("open", "closed") and top_state != expect_top:
        issues.append(f"{object!r}: top is {top_state}, expected {expect_top}")
    return {"ok": not issues, "issues": issues, "faces": faces, "cavity_fraction": round(cavity, 3), "top_state": top_state,
            "interior_size_m": [round(float(v), 4) for v in size], "n_geoms": len(geoms), "walls_found": walls}


def main(argv=None):
    a = argv if argv is not None else sys.argv[1:]
    if len(a) < 2:
        print(__doc__); return 2
    r = box_enclosure_check(a[0], a[1])
    print(json.dumps(r, indent=1)); return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
