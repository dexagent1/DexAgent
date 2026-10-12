"""Appearance texture check: every object has a colour or texture Stage 4 will actually render."""
from __future__ import annotations

TOOL = {
    "name": "appearance_texture_check",
    "title": "Appearance texture check",
    "type": "verifier",
    "category": "Scene Assertions",
    "summary": "Every object has at least one visible geom whose colour is not MuJoCo's default grey or that carries a texture Stage 4 can render; every texture and mesh file the MJCF names exists and loads; a texture on a geom that cannot carry it (a primitive, a mesh without UV coordinates) is reported because Stage 4's Cycles export silently drops it.",
    "description": (
        "Stage 4 renders the scene photo-realistically from the compiled model: a geom's colour is its rgba, or its "
        "material's rgba when the geom keeps the default (0.5, 0.5, 0.5, 1); a 2D texture is exported only for mesh "
        "geoms with texture coordinates, and Cycles drops a missing image without an error. So an object built "
        "without colours renders as default grey, and a textured material put on a box or on a mesh without UVs "
        "renders as its plain (often white) rgba. The check (1) parses the MJCF for every <texture file=...> and "
        "<mesh file=...>, resolving them as MuJoCo does (compiler assetdir/texturedir/meshdir relative to the file) "
        "and opening each texture image; (2) for each object (Stage 2 manifest objects, else top-level non-robot "
        "bodies) lists its visible geoms (groups 0-2, alpha > 0) with their effective colour and whether a texture "
        "survives; an object fails when none of its visible geoms has a non-default colour or a surviving texture, "
        "or when it has no visible geom at all; a dropped texture is an issue naming the geom. Compiling is attempted "
        "after the file checks, so a missing file is reported rather than raised."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "Scene MJCF path (the files are resolved relative to it)."},
        {"name": "recon", "type": "str | dict | None", "description": "scene_recon.json naming the objects. Default: top-level non-robot bodies."},
        {"name": "objects", "type": "list[str] | None", "description": "Only these objects (ids or body names)."},
        {"name": "visual_groups", "type": "list[int]", "description": "Geom groups Stage 4 renders. Default [0, 1, 2]."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; files {textures, meshes, missing, unreadable}; objects [{id, n_visible, colours, textured, verdict}]; dropped_textures [{geom, why}]."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "pillow"],
    "simulators": ["MuJoCo"],
    "tags": ["assertion", "appearance", "texture", "colour", "stage4", "rendering", "stage2"],
    "status": "tested",
    "entrypoint": "appearance_texture_check",
}

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from tool_library.common import scene_assert as SA


def _files(xml_path):
    """[(kind, attribute value, resolved path)] for every texture/mesh/hfield/skin file the MJCF (and its includes) name."""
    out = []
    seen = set()

    def walk(path, assetdir=None, texturedir=None, meshdir=None):
        path = Path(path).resolve()
        if path in seen or not path.exists():
            return
        seen.add(path)
        root = ET.parse(str(path)).getroot()
        for c in root.iter("compiler"):
            assetdir = c.get("assetdir", assetdir)
            texturedir = c.get("texturedir", texturedir)
            meshdir = c.get("meshdir", meshdir)
        base = path.parent
        for el in root.iter():
            f = el.get("file")
            if not f:
                continue
            if el.tag == "include":
                walk(base / f, assetdir, texturedir, meshdir)
                continue
            if el.tag not in ("texture", "mesh", "hfield", "skin"):
                continue
            d = (texturedir if el.tag == "texture" else meshdir if el.tag in ("mesh", "skin") else None) or assetdir
            p = Path(f)
            if not p.is_absolute():
                p = base / d / f if d else base / f
            out.append((el.tag, f, p))

    walk(xml_path)
    return out


def appearance_texture_check(scene, recon=None, objects=None, visual_groups=(0, 1, 2)):
    if not isinstance(scene, (str, Path)) or str(scene).lstrip().startswith("<"):
        raise ValueError("scene: give the MJCF file path (texture files are resolved relative to it)")
    issues = []
    files = {"textures": 0, "meshes": 0, "missing": [], "unreadable": []}
    for kind, raw, p in _files(scene):
        files["textures" if kind == "texture" else "meshes"] += 1
        if not p.exists():
            files["missing"].append(f"{kind} {raw}")
            continue
        if kind == "texture":
            try:
                from PIL import Image
                with Image.open(p) as im:
                    im.load()
                    if im.size[0] == 0 or im.size[1] == 0:
                        raise ValueError("empty image")
            except Exception as e:                           # noqa: BLE001 - an unreadable texture is a finding
                files["unreadable"].append(f"{raw} ({type(e).__name__})")
    for m in files["missing"]:
        issues.append(f"{m} does not exist (resolved relative to {Path(scene).parent}); Stage 4 would render the object without it: fix the path")
    for u in files["unreadable"]:
        issues.append(f"texture {u} cannot be read as an image; re-export it as PNG")
    model, err = SA.try_load_model(scene)
    if model is None:
        issues.append(f"the scene does not compile: {err}")
        return SA.report(issues, files=files, objects=[], dropped_textures=[])
    st = SA.load_setup(None)
    groups = SA.object_groups(model, recon, objects, ref=st["model"], cfg=st["cfg"])
    rows, dropped = [], []
    rgb_role = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)
    for grp in groups:
        if grp["root"] < 0:
            issues.append(f"object {grp['id']}: body {grp.get('missing')} is not in the scene")
            continue
        vis = [g for g in SA.body_geoms(model, grp["bodies"]) if SA.geom_visible(model, g, tuple(visual_groups))]
        colours, textured = set(), False
        for g in vis:
            rgba = SA.effective_rgba(model, g)
            mid = int(model.geom_matid[g])
            tex = -1
            if mid >= 0:
                tid = model.mat_texid[mid]
                tex = int(np.atleast_1d(tid)[rgb_role]) if np.ndim(tid) else int(tid)
            if tex >= 0:
                is_mesh = int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH)
                has_uv = is_mesh and int(model.mesh_texcoordadr[int(model.geom_dataid[g])]) >= 0
                is_2d = int(model.tex_type[tex]) == int(mujoco.mjtTexture.mjTEXTURE_2D)
                if is_mesh and has_uv and is_2d:
                    textured = True
                else:
                    why = ("not a 2D texture" if not is_2d else "the geom is a primitive" if not is_mesh else "the mesh has no texture coordinates")
                    dropped.append({"geom": SA.geom_label(model, g), "object": grp["id"], "why": why, "renders_as": SA.r6(rgba, 3)})
            if not np.allclose(rgba, SA.DEFAULT_RGBA, atol=1e-6):
                colours.add(tuple(round(float(x), 3) for x in rgba))
        verdict = "ok"
        if not vis:
            verdict = "invisible"
            issues.append(f"object {grp['id']} has no visible geom (groups {list(visual_groups)}, alpha > 0): Stage 4 renders nothing; give it a visual")
        elif not colours and not textured:
            verdict = "default grey"
            issues.append(f"object {grp['id']} renders in MuJoCo's default grey on every visible geom: give it the colour measured from the "
                          f"video (measure_object_color) via rgba or a material")
        rows.append({"id": grp["id"], "n_visible": len(vis), "colours": [list(c) for c in sorted(colours)][:6], "textured": textured, "verdict": verdict})
    for d in dropped[:8]:
        issues.append(f"geom {d['geom']} of {d['object']} has a textured material but {d['why']}: Stage 4 drops the texture and renders "
                      f"rgba {d['renders_as']}; use a mesh with UVs or a plain colour")
    if len(dropped) > 8:
        issues.append(f"... and {len(dropped) - 8} more dropped textures")
    return SA.report(issues, files=files, objects=rows, dropped_textures=dropped)
