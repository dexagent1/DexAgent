"""URDF articulation check: a constructed articulated asset, not hand-typed boxes."""
from __future__ import annotations

TOOL = {
    "name": "urdf_articulation_check",
    "title": "URDF articulation check",
    "type": "verifier",
    "category": "Scene Compile Checks",
    "summary": "A URDF describes a real mechanism: valid joint types with limits, geometry in its "
               "links, and mesh-based geometry rather than a few primitive boxes typed by hand.",
    "description": (
        "When a construction tool keeps rejecting its inputs, the easy escape is to type a "
        "URDF by hand: three boxes and an invalid joint type that count as 'articulated' while "
        "looking nothing like the object. This verifier parses the URDF and decides from its "
        "structure: every joint type must be one URDF defines; at least `min_moving_joints` "
        "joints must be prismatic, revolute or continuous, with finite limits where the type "
        "needs them; the links must carry some geometry (a link without any, such as a root "
        "frame, is allowed); and, unless `allow_primitives` is set, the visual and collision "
        "geometry must include meshes, because a measured reconstruction is never only "
        "primitives. Mesh files referenced by the URDF must exist next to it. The report "
        "lists joints with their types and limits and the reasons for any failure."
    ),
    "inputs": [
        {"name": "urdf_path", "type": "str", "description": "The URDF file."},
        {"name": "min_moving_joints", "type": "int", "description": "Prismatic/revolute/continuous joints required. Default 1."},
        {"name": "allow_primitives", "type": "bool", "description": "Accept a URDF whose visuals are only boxes, cylinders and spheres. Default False."},
    ],
    "outputs": [
        {"name": "report", "type": "dict",
         "description": "ok; joints: [{name, type, parent, child, limits}]; links: count; mesh_files: "
                        "found and missing; reasons: why it failed."},
    ],
    "requirements": [],
    "simulators": ["MuJoCo", "Isaac Sim"],
    "tags": ["urdf", "articulated", "articraft", "mechanism"],
    "used_in": ["Place Toy in Drawer"],
    "status": "tested",
    "entrypoint": "urdf_articulation_check",
    "cli": "python -m tool_library.verifiers.urdf_articulation_check model.urdf",
}

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

VALID = {"revolute", "continuous", "prismatic", "fixed", "floating", "planar"}
MOVING = {"revolute", "continuous", "prismatic"}


def check_urdf_articulation(urdf_path, min_moving_joints=1, allow_primitives=False):
    p = Path(urdf_path)
    reasons = []
    try:
        root = ET.parse(p).getroot()
    except Exception as exc:
        return {"ok": False, "reasons": [f"not parseable: {exc}"], "joints": [], "links": 0}
    joints = []
    for j in root.findall("joint"):
        jt = j.get("type", "")
        lim = j.find("limit")
        limits = ({k: float(lim.get(k)) for k in ("lower", "upper") if lim.get(k) is not None}
                  if lim is not None else {})
        parent = j.find("parent").get("link") if j.find("parent") is not None else None
        child = j.find("child").get("link") if j.find("child") is not None else None
        joints.append({"name": j.get("name"), "type": jt, "parent": parent, "child": child,
                       "limits": limits})
        if jt not in VALID:
            reasons.append(f"joint '{j.get('name')}' has invalid type '{jt}'")
        elif jt in ("revolute", "prismatic") and not {"lower", "upper"} <= set(limits):
            reasons.append(f"joint '{j.get('name')}' ({jt}) has no lower/upper limits")
        elif jt in ("revolute", "prismatic") and limits["upper"] - limits["lower"] <= 0:
            reasons.append(f"joint '{j.get('name')}' has an empty range")
    moving = [j for j in joints if j["type"] in MOVING]
    if len(moving) < min_moving_joints:
        reasons.append(f"{len(moving)} moving joints, need {min_moving_joints}")
    links = root.findall("link")
    link_names = {l.get("name") for l in links}
    for j in joints:
        for side in ("parent", "child"):
            if j[side] not in link_names:
                reasons.append(f"joint '{j['name']}' refers to unknown link '{j[side]}'")
    n_mesh, n_prim, missing, found = 0, 0, [], []
    for l in links:
        geoms = l.findall("visual/geometry") + l.findall("collision/geometry")
        for g in geoms:
            m = g.find("mesh")
            if m is not None:
                n_mesh += 1
                f = m.get("filename", "")
                f = f.replace("package://", "").replace("file://", "")
                cand = p.parent / f
                (found if cand.exists() or Path(f).exists() else missing).append(f)
            elif any(g.find(t) is not None for t in ("box", "cylinder", "sphere")):
                n_prim += 1
    if n_mesh == 0 and n_prim == 0:
        reasons.append("no geometry in any link")
    if n_mesh == 0 and not allow_primitives:
        reasons.append("visuals are only primitives (box/cylinder/sphere): this looks hand-typed, "
                       "not reconstructed")
    if missing:
        reasons.append(f"{len(missing)} mesh files are missing next to the URDF")
    return {"ok": not reasons, "reasons": reasons, "joints": joints, "moving_joints": len(moving),
            "links": len(links), "mesh_files": {"found": found, "missing": missing},
            "primitive_geoms": n_prim}


def main(argv=None):
    rep = check_urdf_articulation(sys.argv[1] if argv is None else argv[0])
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())


urdf_articulation_check = check_urdf_articulation   # the library calls every tool by its own name
