"""Mesh validity check: are the scene's meshes non-degenerate, at a plausible scale, and closed where they define mass?"""
from __future__ import annotations

TOOL = {
    "name": "mesh_validity_check",
    "title": "Mesh validity check",
    "type": "verifier",
    "category": "Scene Compile Checks",
    "summary": "Every mesh geom is non-degenerate (enough vertices and thickness to collide), at a plausible metric scale "
               "(no millimetre-as-metre slips, no part a hundred times smaller than its object), and closed wherever MuJoCo "
               "computes mass and inertia from its volume.",
    "description": (
        "Meshes are where most reconstruction slips hide, and MuJoCo compiles many of them without complaint. The check "
        "compiles the scene through MjSpec (so each geom's mass, density, inertia mode and its body's inertial setting are "
        "known) and inspects every mesh geom on the compiled vertices (scale already applied): (1) a collidable mesh needs "
        "at least 4 vertices and 4 faces and a thickness above `min_thickness_m` in every axis, or MuJoCo collides a "
        "flat hull; (2) a mesh of a scene object larger than `max_extent_m` or smaller than `min_extent_m`, or more than "
        "`ratio_limit` times smaller than the largest geom of its own object, is reported as a unit slip (a cup authored in "
        "millimetres becomes 80 m; a part scaled twice becomes 0.08 mm); (3) a mesh that contributes mass from its volume "
        "(no explicit inertial on its body, mass not set to 0, volumetric inertia with MuJoCo's 'legacy' or 'exact' mode) "
        "must be closed: an open surface makes the legacy algorithm over-count volume, so the body's mass and inertia are an "
        "artifact. An open mass-carrying mesh fails only when its estimated mass is more than `open_mass_share` of its "
        "object's mass; smaller ones are listed as notes. Robot meshes are checked for degeneracy only (the setup ships "
        "them with explicit inertials). A scene that does not compile fails with MuJoCo's message."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF file, XML string, or compiled model (a model skips the mass attribution)."},
        {"name": "min_extent_m", "type": "float", "description": "Smallest plausible size of an object mesh, metres. Default 0.001."},
        {"name": "max_extent_m", "type": "float", "description": "Largest plausible size of an object mesh, metres. Default 1.5."},
        {"name": "min_thickness_m", "type": "float", "description": "Thinnest axis a collidable mesh may have, metres. Default 1e-5."},
        {"name": "ratio_limit", "type": "float", "description": "How many times smaller than its object's largest geom a mesh may be. Default 50."},
        {"name": "open_mass_share", "type": "float", "description": "Share of its object's mass an open, volume-massed mesh may carry. Default 0.1."},
        {"name": "robot_prefix", "type": "str", "description": "Name prefix of robot bodies. Default 'robot_'."},
    ],
    "outputs": [
        {"name": "report", "type": "dict",
         "description": "ok; issues; notes; n_mesh_geoms; meshes: [{geom, mesh, body, object, robot, collidable, nvert, nface, extent_m, "
                        "thickness_m, open_edges, mass_from_volume, est_mass_kg, object_mass_kg, problem}] (problems first); worst."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["compile", "mesh", "scale", "watertight", "inertia"],
    "status": "tested",
    "entrypoint": "mesh_validity_check",
    "cli": "python -m tool_library.verifiers.mesh_validity_check scene.xml",
}

import json
import math
import sys

import mujoco
import numpy as np

from tool_library.common import compile_checks as C

_VOLUMETRIC = {"legacy", "exact"}


def _compile_with_spec(scene):
    if isinstance(scene, mujoco.MjModel):
        return scene, None, None
    p = C.scene_path(scene)
    try:
        with C.capture_mj_warnings():
            spec = mujoco.MjSpec.from_file(str(p)) if p is not None else mujoco.MjSpec.from_string(str(scene))
            model = spec.compile()
        return model, spec, None
    except Exception as exc:
        m, err = C.compile_scene(scene)
        return m, None, err or f"{type(exc).__name__}: {exc}"


def mesh_validity_check(scene, min_extent_m=0.001, max_extent_m=1.5, min_thickness_m=1e-5, ratio_limit=50.0,
                        open_mass_share=0.1, robot_prefix=C.ROBOT_PREFIX):
    C.require_scene(scene)
    model, spec, err = _compile_with_spec(scene)
    if model is None:
        hint = ("a mesh MuJoCo rejects (too few vertices, zero volume, inverted or open under inertia='exact') or a missing "
                "mesh file: fix the mesh named in the message")
        return C.compile_failure(hint, err, meshes=[])
    attr = {}
    if spec is not None:
        comp = spec.compiler
        ifg = int(comp.inertiafromgeom)          # 0 false, 1 true, 2 auto
        lo, hi = [int(x) for x in comp.inertiagrouprange]
        mesh_mode = {me.name: str(me.inertia).split("_")[-1].lower() for me in spec.meshes}
        for g in spec.geoms:
            body = g.parent
            explicit = bool(getattr(body, "explicitinertial", False))
            uses_geom = ifg == 1 or (ifg == 2 and not explicit)
            mass_attr = float(g.mass)
            in_group = lo <= int(g.group) <= hi
            massless = (not math.isnan(mass_attr) and mass_attr == 0.0) or (math.isnan(mass_attr) and float(g.density) <= 0)
            vol = str(g.typeinertia).endswith("VOLUME")
            attr[int(g.id)] = {"contributes": bool(uses_geom and in_group and not massless), "volumetric": vol,
                               "mass_attr": None if math.isnan(mass_attr) else mass_attr, "density": float(g.density),
                               "mode": mesh_mode.get(g.meshname, "legacy")}
    paired = set(int(x) for x in model.pair_geom1) | set(int(x) for x in model.pair_geom2)
    # object of each body, its total mass and its largest geom
    obj_of = {b: C.top_body(model, b) for b in range(model.nbody)}
    obj_mass = {}
    obj_size = {}
    for b in range(1, model.nbody):
        if int(model.body_parentid[b]) == 0:
            obj_mass[b] = float(model.body_subtreemass[b])
    for g in range(model.ngeom):
        if int(model.geom_type[g]) == C.PLANE:
            continue
        o = obj_of[int(model.geom_bodyid[g])]
        ext = 2.0 * float(np.max(model.geom_aabb[g][3:]))
        obj_size[o] = max(obj_size.get(o, 0.0), ext)
    rows = []
    for g in range(model.ngeom):
        if int(model.geom_type[g]) != C.MESH:
            continue
        k = int(model.geom_dataid[g])
        V, F = C.mesh_arrays(model, k)
        b = int(model.geom_bodyid[g])
        o = obj_of[b]
        robot = C.is_robot(C.body_name(model, b), robot_prefix)
        col = C.collidable(model, g) or g in paired
        ext = (V.max(axis=0) - V.min(axis=0)) if len(V) else np.zeros(3)
        nopen = C.open_edges(F)
        sv, uv = C.mesh_volumes(V, F)
        a = attr.get(g, {"contributes": True, "volumetric": True, "mass_attr": None, "density": 1000.0, "mode": "legacy"})
        from_volume = a["contributes"] and a["volumetric"] and a["mass_attr"] is None and a["mode"] in _VOLUMETRIC
        est = 0.0 if not a["contributes"] else (a["mass_attr"] if a["mass_attr"] is not None else a["density"] * uv)
        row = {"geom": C.geom_name(model, g), "mesh": C.name_of(model, mujoco.mjtObj.mjOBJ_MESH, k, "mesh"),
               "body": C.body_name(model, b), "object": C.body_name(model, o) if o else "world", "robot": robot,
               "collidable": bool(col), "nvert": int(len(V)), "nface": int(len(F)), "extent_m": float(ext.max()),
               "thickness_m": float(ext.min()), "open_edges": nopen, "mass_from_volume": bool(from_volume),
               "est_mass_kg": float(est), "object_mass_kg": float(obj_mass.get(o, 0.0)), "problem": None, "severity": None}
        probs, sev = [], None
        if col and (len(V) < 4 or len(F) < 4 or float(ext.min()) < min_thickness_m):
            probs.append(f"degenerate collision mesh ({len(V)} vertices, {len(F)} faces, thinnest axis {1000 * float(ext.min()):.4f} mm): "
                         "MuJoCo collides a flat hull; give it thickness or replace it by a primitive")
            sev = "issue"
        if not robot and o:
            if float(ext.max()) > max_extent_m:
                probs.append(f"{float(ext.max()):.2f} m across, larger than any tabletop object ({max_extent_m} m): authored in millimetres "
                             "or centimetres and read as metres? set the mesh scale (e.g. scale=\"0.001 0.001 0.001\")")
                sev = "issue"
            elif float(ext.max()) < min_extent_m:
                probs.append(f"{1000 * float(ext.max()):.4f} mm across, smaller than any real part ({1000 * min_extent_m} mm): scaled twice? "
                             "check the mesh scale")
                sev = "issue"
            big = obj_size.get(o, 0.0)
            if float(ext.max()) > 0 and big / float(ext.max()) > ratio_limit and not probs:
                probs.append(f"{big / float(ext.max()):.0f} times smaller than the largest geom of its object ({big:.3f} m): a unit slip "
                             "between this mesh and the rest of the object")
                sev = "issue"
        if from_volume and nopen > 0 and not robot:
            share = float(est) / max(float(obj_mass.get(o, 0.0)), 1e-12)
            msg = (f"open surface ({nopen} boundary/non-manifold edges) carries mass from its volume ({a['mode']} inertia): "
                   f"about {1000 * float(est):.2f} g of {1000 * float(obj_mass.get(o, 0.0)):.2f} g of its object ({100 * share:.0f}%)")
            if share > open_mass_share:
                probs.append(msg + ": the mass and inertia are an artifact of the open mesh; close it, set the body's <inertial> or "
                                   "the geom's mass, or use inertia=\"convex\"/\"shell\"")
                sev = "issue"
            else:
                probs.append(msg + " (below the share that matters)")
                sev = sev or "note"
        if probs:
            row["problem"] = "; ".join(probs)
            row["severity"] = sev
        rows.append(row)
    rows.sort(key=lambda r: (r["severity"] != "issue", r["severity"] != "note"))
    issues = [f"mesh geom '{r['geom']}' (mesh '{r['mesh']}', body '{r['body']}'): {r['problem']}" for r in rows if r["severity"] == "issue"]
    notes = [f"mesh geom '{r['geom']}' (body '{r['body']}'): {r['problem']}" for r in rows if r["severity"] == "note"]
    return C.jsonable({"ok": not issues, "compiled": True, "issues": issues, "notes": notes[:20], "n_mesh_geoms": len(rows),
                       "meshes": rows[:60], "worst": rows[0] if rows and rows[0]["severity"] else None,
                       "mass_attribution": spec is not None})


def main(argv=None):
    rep = mesh_validity_check((sys.argv[1:] if argv is None else argv)[0])
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
