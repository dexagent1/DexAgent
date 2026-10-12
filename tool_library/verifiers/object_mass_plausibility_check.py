"""Object mass plausibility check: every object weighs what something of its size and material can weigh."""
from __future__ import annotations

TOOL = {
    "name": "object_mass_plausibility_check",
    "title": "Object mass plausibility check",
    "type": "verifier",
    "category": "Scene Assertions",
    "summary": "Divides each object's total mass by the volume of its collision geometry's convex hull and flags densities outside the band of its material (from a keyword such as plastic, glass, metal, foam, plush, cardboard, given or read from the Stage 1 appearance) or, without one, outside a broad band: weightless objects that a breath would move and lead-heavy ones no hand could lift.",
    "description": (
        "A rigid object's mass decides whether a grasp can lift it and whether a push moves it; a constructor called "
        "with mass=1e-5 or with a density of 50000 builds an object that makes Stage 3 trivially succeed or fail. "
        "For each object (the Stage 2 manifest's objects, else every top-level non-robot body with its subtree) the "
        "mass of all its bodies is summed and divided by the volume of the convex hull of its collision geoms (all "
        "visible geoms when it has none): the bulk density, which is below the material density for hollow things "
        "(a box, a cup) and equal to it for solid ones. The bulk density must lie inside the band of the object's "
        "material: `materials` maps an object id to a keyword, otherwise the Stage 1 appearance text (scene_semantic) "
        "is searched for material words, and several materials widen the band to their union. Without a material the "
        "broad band [5, 9000] kg/m^3 applies (air is 1.2, steel 7850, lead 11340). Objects that a hand manipulates "
        "(Stage 1 role manipulated or tool) must also weigh at most max_manipulated_kg. Massless objects fail."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "Scene MJCF."},
        {"name": "recon", "type": "str | dict | None", "description": "scene_recon.json (Stage 2 manifest) naming the objects and their bodies. Default: top-level bodies."},
        {"name": "semantic", "type": "str | dict | None", "description": "scene_semantics.json (Stage 1) for materials (appearance) and roles."},
        {"name": "materials", "type": "dict | None", "description": "{object id: material keyword} overriding the Stage 1 text."},
        {"name": "objects", "type": "list[str] | None", "description": "Only these objects (ids or body names)."},
        {"name": "bands", "type": "dict | None", "description": "{keyword: [lo, hi] kg/m^3} added to / replacing the built-in material bands."},
        {"name": "max_manipulated_kg", "type": "float", "description": "Heaviest object a hand may manipulate. Default 3.0."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; objects [{id, mass_kg, hull_volume_m3, bulk_density, material, band, role, verdict}]; worst (the density furthest outside its band, as a ratio)."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["assertion", "mass", "density", "physics", "material", "stage2"],
    "status": "tested",
    "entrypoint": "object_mass_plausibility_check",
}

import re

import mujoco
import numpy as np

from tool_library.common import scene_assert as SA

# bulk-density bands (mass / convex-hull volume), kg/m^3; low ends allow hollow things (an empty carton is ~10)
BANDS = {
    "foam": (5.0, 300.0), "sponge": (5.0, 300.0), "styrofoam": (5.0, 100.0),
    "plush": (5.0, 800.0), "fabric": (5.0, 800.0), "cloth": (5.0, 1500.0), "textile": (5.0, 1500.0), "felt": (5.0, 800.0),
    "stuffed": (5.0, 800.0), "paper": (5.0, 1200.0), "cardboard": (5.0, 1200.0), "carton": (5.0, 1200.0),
    "wood": (10.0, 1300.0), "wooden": (10.0, 1300.0),
    "plastic": (5.0, 2500.0), "acrylic": (5.0, 2500.0), "rubber": (5.0, 2500.0), "silicone": (5.0, 2500.0), "nylon": (5.0, 2500.0),
    "glass": (20.0, 4000.0), "ceramic": (20.0, 4000.0), "porcelain": (20.0, 4000.0), "stone": (20.0, 4000.0),
    "metal": (20.0, 9000.0), "steel": (20.0, 9000.0), "aluminium": (20.0, 3500.0), "aluminum": (20.0, 3500.0),
    "iron": (20.0, 9000.0), "brass": (20.0, 9000.0), "copper": (20.0, 9000.0),
}
BROAD = (5.0, 9000.0)


def _materials_in(text, bands):
    words = set(re.findall(r"[a-z]+", str(text).lower()))
    return sorted(k for k in bands if k in words)


def _object_points(model, data, bodies):
    gs = SA.body_geoms(model, bodies)
    col = [g for g in gs if SA.geom_collidable(model, g)]
    use = col or [g for g in gs if SA.geom_visible(model, g)]
    pts = [SA.geom_world_points(model, data, g) for g in use]
    pts = [p for p in pts if len(p)]
    return (np.vstack(pts) if pts else np.zeros((0, 3))), use


def object_mass_plausibility_check(scene, recon=None, semantic=None, materials=None, objects=None, bands=None, max_manipulated_kg=3.0):
    model, err = SA.try_load_model(scene)
    if model is None:
        return SA.report([f"the scene does not compile: {err}"], compiled=False)
    st = SA.load_setup(None)
    B = dict(BANDS)
    if bands:
        for k, v in bands.items():
            if len(v) != 2 or float(v[0]) >= float(v[1]):
                raise ValueError(f"bands['{k}'] must be [lo, hi] with lo < hi, got {v}")
            B[str(k).lower()] = (float(v[0]), float(v[1]))
    sem = {str(o.get("id")): o for o in SA.semantic_objects(semantic)} if semantic is not None else {}
    groups = SA.object_groups(model, recon, objects, ref=st["model"], cfg=st["cfg"])
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    from scipy.spatial import ConvexHull
    rows, issues = [], []
    worst = 1.0
    for g in groups:
        if g["root"] < 0:
            issues.append(f"object {g['id']}: body {g.get('missing')} is not in the scene")
            continue
        mass = float(sum(model.body_mass[b] for b in g["bodies"]))
        P, used = _object_points(model, data, g["bodies"])
        try:
            vol = float(ConvexHull(P).volume) if len(P) >= 4 else 0.0
        except Exception:                                    # noqa: BLE001 - flat geometry
            vol = 0.0
        if vol <= 0.0:
            vol = float(sum(SA.geom_volume(model, x) for x in used))
        so = sem.get(str(g["id"]), {})
        if materials and str(g["id"]) in materials:
            mats = _materials_in(materials[str(g["id"])], B) or [str(materials[str(g["id"])]).lower()]
            mats = [m for m in mats if m in B]
            if not mats:
                raise ValueError(f"materials['{g['id']}']: unknown material '{materials[str(g['id'])]}'; known: {sorted(B)}")
        else:
            text = " ".join([str(so.get("name", "")), str(so.get("appearance", ""))])
            mats = _materials_in(text, B)
        band = (min(B[m][0] for m in mats), max(B[m][1] for m in mats)) if mats else BROAD
        role = so.get("role")
        dens = mass / vol if vol > 0 else float("inf")
        verdict = "ok"
        if mass <= 1e-9:
            verdict = "massless"
            issues.append(f"object {g['id']} has no mass: give its constructor a real mass (or density)")
        elif vol <= 0:
            verdict = "no geometry"
            issues.append(f"object {g['id']} has no geometry to judge its mass against")
        elif dens < band[0]:
            verdict = "too light"
            worst = max(worst, band[0] / dens)
            issues.append(f"object {g['id']} weighs {mass * 1000:.4g} g in a {vol * 1e6:.4g} cm^3 hull: {dens:.3g} kg/m^3 is below the "
                          f"{'/'.join(mats) or 'plausible'} band {band[0]:g}-{band[1]:g}; raise its mass to at least {band[0] * vol * 1000:.3g} g")
        elif dens > band[1]:
            verdict = "too heavy"
            worst = max(worst, dens / band[1])
            issues.append(f"object {g['id']} weighs {mass:.4g} kg in a {vol * 1e6:.4g} cm^3 hull: {dens:.4g} kg/m^3 is above the "
                          f"{'/'.join(mats) or 'plausible'} band {band[0]:g}-{band[1]:g}; lower its mass to at most {band[1] * vol:.3g} kg")
        if role in ("manipulated", "tool") and mass > max_manipulated_kg:
            verdict = "too heavy to manipulate"
            issues.append(f"object {g['id']} is manipulated by a hand in the video but weighs {mass:.3g} kg (> {max_manipulated_kg} kg); check its mass")
        rows.append({"id": g["id"], "mass_kg": round(mass, 6), "hull_volume_m3": float(f"{vol:.6g}"), "bulk_density": round(dens, 2) if np.isfinite(dens) else None,
                     "material": mats, "band": list(band), "role": role, "verdict": verdict})
    return SA.report(issues, objects=rows, worst=round(worst, 3))
