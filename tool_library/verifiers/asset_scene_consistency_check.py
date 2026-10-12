"""Asset / scene consistency check: every object's recorded asset is the object the scene holds."""
from __future__ import annotations

TOOL = {
    "name": "asset_scene_consistency_check",
    "title": "Asset / scene consistency check",
    "type": "verifier",
    "category": "Scene Assertions",
    "summary": "For every object of scene_recon.json, the bodies of its recorded asset_xml are the bodies the scene has "
               "under that object's root: Stage 3 builds its hand scenes from the asset file, so an asset that lacks a part the "
               "scene has (a cable's plug) or has parts the scene lacks hides them from every grasp tool.",
    "description": (
        "Compiles each object's `asset_xml`, takes its body names, and finds the object's bodies in the scene: the subtree of "
        "its `root_body` (plus the bodies the entry lists). The scene names an asset's bodies with a prefix "
        "(<prefix>_<asset body>, the prefix stage_on_bench gave it), found from the root body. Issues: an asset body the scene "
        "does not have, a scene body of the object the asset does not have (both with names), an asset that does not compile, "
        "a root body missing from the scene. Deformables simulated only in Isaac (no MuJoCo body) and objects without an "
        "asset are skipped and reported."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "The Stage 2 scene.xml."},
        {"name": "recon", "type": "str | dict", "description": "scene_recon.json (objects with id, root_body, asset_xml, bodies)."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok, issues, objects {id: {prefix, asset_bodies, scene_bodies, missing_in_scene, missing_in_asset}}, skipped {id: reason}."},
    ],
    "requirements": ["mujoco>=3.2"],
    "simulators": ["MuJoCo"],
    "tags": ["stage2", "assertion", "assets", "hand scene", "consistency"],
    "status": "tested",
    "entrypoint": "asset_scene_consistency_check",
}

import json
from pathlib import Path

import mujoco


def _subtree(model, b):
    out, stack = set(), [b]
    while stack:
        x = stack.pop(); out.add(x)
        stack += [c for c in range(model.nbody) if int(model.body_parentid[c]) == x and c != x]
    return out


def _geoms_of(model, b):
    return sum(1 for g in range(model.ngeom) if int(model.geom_bodyid[g]) == b)


def asset_scene_consistency_check(scene, recon):
    scene = Path(scene)
    rec = json.loads(Path(recon).read_text()) if isinstance(recon, (str, Path)) else dict(recon)
    base = Path(recon).parent if isinstance(recon, (str, Path)) else scene.parent
    try:
        model = mujoco.MjModel.from_xml_path(str(scene))
    except Exception as e:                                   # a scene that does not compile is a verdict
        return {"ok": False, "issues": [f"the scene does not compile: {e}"], "objects": {}, "skipped": {}}
    issues, objects, skipped = [], {}, {}
    # objects sharing one asset (a box and its free lid, a bottle and its cap) are compared together
    groups = {}
    for o in rec.get("objects", []):
        oid, root, ax = o.get("id"), o.get("root_body"), o.get("asset_xml")
        if not ax:
            skipped[oid] = "no asset_xml"; continue
        rb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root) if root else -1
        if rb < 0:
            if o.get("type") == "deformable":
                skipped[oid] = "deformable without a MuJoCo body (Isaac only)"
            else:
                issues.append(f"{oid}: root body {root!r} is not in the scene")
            continue
        path = ((base / ax) if not Path(ax).is_absolute() else Path(ax)).resolve()
        groups.setdefault(str(path), []).append((o, rb))
    for path, members in groups.items():
        ids = [o["id"] for o, _ in members]
        label = "/".join(ids)
        try:
            am = mujoco.MjModel.from_xml_path(path)
        except Exception as e:
            issues.append(f"{label}: its asset {Path(path).name} does not compile: {str(e)[:160]}"); continue
        asset_bodies = {am.body(b).name: b for b in range(1, am.nbody) if am.body(b).name}
        prefix = None
        for o, rb in members:
            root = o["root_body"]
            for bn in sorted(asset_bodies, key=len):
                if root.endswith("_" + bn) or root == bn:
                    cand = root[: -len(bn) - 1] if root != bn else ""
                    prefix = cand if prefix is None or len(cand) < len(prefix) else prefix
        if prefix is None:
            issues.append(f"{label}: no body of its asset {Path(path).name} names the scene's root body: the asset is not this object"); continue
        name = (lambda n: f"{prefix}_{n}" if prefix else n)
        scene_bodies = set()
        for o, rb in members:
            top = rb
            while int(model.body_parentid[top]) > 0 and (model.body(int(model.body_parentid[top])).name or "").startswith(prefix + "_"):
                top = int(model.body_parentid[top])
            scene_bodies |= {model.body(b).name for b in _subtree(model, top) if model.body(b).name}
        # free bodies of the same asset sit at the scene's top level under the same prefix
        scene_bodies |= {model.body(b).name for b in range(1, model.nbody)
                         if (model.body(b).name or "") in {name(n) for n in asset_bodies}}
        want = {name(n) for n in asset_bodies}
        missing_in_scene = sorted(want - scene_bodies)
        missing_in_asset = sorted(n for n in scene_bodies - want if (not prefix or n.startswith(prefix + "_")))
        # parts carried as geoms of a body (a cable's plug on its last link): the same body must carry as many geoms
        fewer = []
        for n, ab in asset_bodies.items():
            sb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name(n))
            if sb >= 0 and _geoms_of(model, sb) > _geoms_of(am, ab):
                fewer.append(f"{name(n)} ({_geoms_of(am, ab)} geoms in the asset, {_geoms_of(model, sb)} in the scene)")
        objects[label] = {"prefix": prefix, "asset_bodies": len(asset_bodies), "scene_bodies": len(scene_bodies),
                          "missing_in_scene": missing_in_scene[:10], "missing_in_asset": missing_in_asset[:10], "fewer_geoms": fewer[:10]}
        if missing_in_asset or fewer:
            what = ", ".join((missing_in_asset[:3] + fewer[:3])[:4])
            issues.append(f"{label}: the scene has parts its asset {Path(path).name} lacks ({what}): Stage 3's hand scenes are built "
                          f"from the asset and will not see them; record the asset file the scene was staged from")
        if missing_in_scene:
            issues.append(f"{label}: its asset {Path(path).name} has {len(missing_in_scene)} body(ies) the scene lacks "
                          f"({', '.join(missing_in_scene[:4])}): the recorded asset is not the one staged")
    return {"ok": not issues, "issues": issues, "objects": objects, "skipped": skipped}
