"""Texture/material resolution check: do the scene's textures exist, load, have sane sizes, and actually show on what uses them?"""
from __future__ import annotations

TOOL = {
    "name": "texture_material_resolution_check",
    "title": "Texture and material resolution check",
    "type": "verifier",
    "category": "Scene Compile Checks",
    "summary": "Every texture file exists and decodes to the size MuJoCo compiled, texture sizes and total texture memory stay within "
               "budget, every material's texture resolves, and a 2D-textured material is not put on a mesh without texture "
               "coordinates (MuJoCo maps it automatically, but the Stage 4 exporter and any UV-based renderer drop it).",
    "description": (
        "Textures fail quietly: a renderer that cannot find a texture draws the base colour (a missing texture has killed a "
        "Cycles render without an error), a 2D texture on a mesh that has no UV coordinates is drawn by MuJoCo with automatic "
        "coordinates but dropped by the Stage 4 exporter (it writes UVs only when the mesh has them), so the photoreal render "
        "shows the base colour while the MuJoCo render shows the texture; and an 8k texture costs 256 MB of model memory per copy. The check reads every texture file reference of the MJCF "
        "(following includes, resolved as MuJoCo resolves them) and fails a missing file or one that does not decode (PNG/JPG "
        "via OpenCV, KTX skipped); it compiles the scene (a missing file or unknown material is MuJoCo's compile error, "
        "reported as such) and compares each file texture's decoded size with the compiled one; it fails a texture whose "
        "face exceeds `max_texture_px` per side or is under 2 px, total texture data above `max_total_mb`, a material whose "
        "texture index does not exist, and a mesh geom whose material carries a 2D texture while the mesh has no texture "
        "coordinates. 2D textures no material uses and repeated skyboxes (only the first is drawn) are notes."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF file, XML string, or compiled model (a model skips the file tests)."},
        {"name": "max_texture_px", "type": "int", "description": "Largest texture face side, pixels. Default 4096."},
        {"name": "max_total_mb", "type": "float", "description": "Largest total texture data, MB. Default 256."},
    ],
    "outputs": [
        {"name": "report", "type": "dict",
         "description": "ok; issues; notes; textures: [{index, name, type, width, height, face_px, file, decoded_size, problem}]; "
                        "untextured_meshes: [{geom, mesh, material}]; total_texture_mb; worst."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "opencv-python"],
    "simulators": ["MuJoCo"],
    "tags": ["compile", "texture", "material", "uv", "render"],
    "status": "tested",
    "entrypoint": "texture_material_resolution_check",
    "cli": "python -m tool_library.verifiers.texture_material_resolution_check scene.xml",
}

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

from tool_library.common import compile_checks as C

_TYPE = {0: "2d", 1: "cube", 2: "skybox"}


def _decode(path):
    """(width, height) of an image file, None when it does not decode, 'skip' for formats not decoded here."""
    p = Path(path)
    if p.suffix.lower() in (".ktx", ".ktx2"):
        return "skip"
    try:
        import cv2
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        return int(img.shape[1]), int(img.shape[0])
    except ImportError:
        try:
            from PIL import Image
            with Image.open(str(p)) as im:
                return int(im.size[0]), int(im.size[1])
        except Exception:
            return None


def texture_material_resolution_check(scene, max_texture_px=4096, max_total_mb=256.0):
    C.require_scene(scene)
    issues, notes = [], []
    p = C.scene_path(scene)
    file_refs = []
    if p is not None and p.exists():
        file_refs = [r for r in C.mjcf_file_refs(p) if r["tag"] == "texture"]
        for r in file_refs:
            res = Path(r["resolved"])
            if not res.exists():
                issues.append(f"texture '{r['name']}' file=\"{r['file']}\" does not exist (looked at {res}): the renderer draws the base "
                              "colour instead; ship the image next to the scene and reference it relatively")
                r["decoded"] = None
                continue
            dec = _decode(res)
            r["decoded"] = dec
            if dec is None:
                issues.append(f"texture '{r['name']}' file \"{r['file']}\" exists but does not decode as an image: re-export it as PNG")
    model, err = C.compile_scene(scene)
    if model is None:
        return C.compile_failure("a texture file MuJoCo cannot load or a material/texture name that does not exist: fix the asset named "
                                 "in the message", err, textures=[], file_issues=issues)
    by_name = {}
    for r in file_refs:
        if r.get("name"):
            by_name.setdefault(r["name"], r)
    rows, total = [], 0
    nskybox = 0
    for t in range(model.ntex):
        ttype = int(model.tex_type[t])
        w, h = int(model.tex_width[t]), int(model.tex_height[t])
        nch = int(model.tex_nchannel[t]) if hasattr(model, "tex_nchannel") else 3
        total += w * h * nch
        face_h = h // 6 if ttype in (1, 2) and h >= 6 * w - 6 else h
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TEXTURE, t) or ""
        ref = by_name.get(name)
        probs = []
        if max(w, face_h) > max_texture_px:
            probs.append(f"{w} x {face_h} px per face exceeds {max_texture_px} px: {w * h * nch / 2 ** 20:.0f} MB in the model; downsample it")
        if min(w, face_h) < 2:
            probs.append(f"{w} x {face_h} px is degenerate")
        if ref is not None and isinstance(ref.get("decoded"), tuple) and ttype == 0 and ref["decoded"] != (w, h):
            probs.append(f"the file decodes to {ref['decoded'][0]} x {ref['decoded'][1]} px but MuJoCo compiled {w} x {h}")
        if ttype == 2:
            nskybox += 1
        rows.append({"index": t, "name": name or f"texture{t}", "type": _TYPE.get(ttype, str(ttype)), "width": w, "height": h,
                     "face_px": [w, face_h], "file": ref["file"] if ref else None,
                     "decoded_size": list(ref["decoded"]) if ref and isinstance(ref.get("decoded"), tuple) else None,
                     "problem": "; ".join(probs) or None})
        for pr in probs:
            issues.append(f"texture '{rows[-1]['name']}': {pr}")
    total_mb = total / 2 ** 20
    if total_mb > max_total_mb:
        issues.append(f"textures hold {total_mb:.0f} MB of data (> {max_total_mb} MB): every MjModel copy and renderer pays it; downsample")
    if nskybox > 1:
        notes.append(f"{nskybox} skybox textures (each hand asset brings one); MuJoCo draws only the first")
    # material -> texture, and 2D textures on meshes without UVs
    texid = np.asarray(model.mat_texid)
    rgb_col = int(mujoco.mjtTextureRole.mjTEXROLE_RGB) if texid.ndim == 2 else None
    used_tex = set()
    for mi in range(model.nmat):
        ids = texid[mi] if texid.ndim == 2 else [texid[mi]]
        for tid in np.atleast_1d(ids):
            tid = int(tid)
            if tid >= model.ntex:
                issues.append(f"material '{C.name_of(model, mujoco.mjtObj.mjOBJ_MATERIAL, mi, 'material')}' points at texture {tid}, "
                              f"which does not exist (ntex = {model.ntex})")
            elif tid >= 0:
                used_tex.add(tid)
    untextured = []
    for g in range(model.ngeom):
        mi = int(model.geom_matid[g])
        if mi < 0:
            continue
        tid = int(texid[mi][rgb_col]) if rgb_col is not None else int(texid[mi])
        if tid < 0 or int(model.tex_type[tid]) != 0 or int(model.geom_type[g]) != C.MESH:
            continue
        k = int(model.geom_dataid[g])
        if int(model.mesh_texcoordnum[k]) <= 0 and float(model.geom_rgba[g][3]) > 0:
            untextured.append({"geom": C.geom_name(model, g), "mesh": C.name_of(model, mujoco.mjtObj.mjOBJ_MESH, k, "mesh"),
                               "material": C.name_of(model, mujoco.mjtObj.mjOBJ_MATERIAL, mi, "material")})
    if untextured:
        u = untextured[0]
        issues.append(f"{len(untextured)} mesh geom(s) carry a 2D-textured material on a mesh without texture coordinates (first "
                      f"'{u['geom']}', mesh '{u['mesh']}', material '{u['material']}'): MuJoCo maps the texture automatically but "
                      "Stage 4's exporter writes no UVs for it and drops the texture, so the photoreal render shows only the base "
                      "colour; export the mesh with UVs (OBJ vt) or use an untextured material")
    unused_t = [rows[t]["name"] for t in range(model.ntex) if t not in used_tex and int(model.tex_type[t]) == 0]
    if unused_t:
        notes.append(f"{len(unused_t)} 2D textures are not used by any material ({unused_t[:4]})")
    rows.sort(key=lambda r: r["problem"] is None)
    return C.jsonable({"ok": not issues, "compiled": True, "issues": issues, "notes": notes, "textures": rows[:40], "ntex": int(model.ntex),
                       "nmat": int(model.nmat), "untextured_meshes": untextured[:20], "total_texture_mb": total_mb,
                       "worst": rows[0] if rows and rows[0]["problem"] else None})


def main(argv=None):
    rep = texture_material_resolution_check((sys.argv[1:] if argv is None else argv)[0])
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
