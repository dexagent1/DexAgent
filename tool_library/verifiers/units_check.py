"""Units check: is the scene authored in metres, or in pixels?"""
from __future__ import annotations

TOOL = {
    "name": "units_check",
    "title": "Units check",
    "type": "verifier",
    "category": "Scene Compile Checks",
    "summary": "No geom is positioned or sized at a scale that only pixel coordinates could "
               "produce; the scene fits inside a table-sized world.",
    "description": (
        "A reconstruction that copies image coordinates into the scene puts a 500 m table a "
        "kilometre from the origin; every object then free-falls and the physics checks "
        "report instability, which sends the author chasing the wrong fault. This verifier "
        "looks at the geometry alone: the position of every collidable geom and its largest "
        "extent must stay under a limit (50 m by default, generous for any tabletop or room "
        "scene). It names the geoms that break it, so the units mistake is reported as a "
        "units mistake."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF file, XML string, or compiled model."},
        {"name": "max_extent_m", "type": "float", "description": "Largest allowed geom half-size or distance from the origin. Default 50."},
    ],
    "outputs": [
        {"name": "report", "type": "dict",
         "description": "ok; offenders: [{geom, body, position, size}]; scene_radius_m: how far the farthest "
                        "geom sits from the origin."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["compile", "units", "scale"],
    "used_in": ["Place Cup in Bowl", "Place Toy in Drawer"],
    "status": "tested",
    "entrypoint": "units_check",
    "cli": "python -m tool_library.verifiers.units_check scene.xml",
}

import json
import sys

import numpy as np

from tool_library.common import scene as S


def check_units(scene, max_extent_m=50.0):
    model = S.load_model(scene)
    data = S.forward(model)
    offenders, radius = [], 0.0
    for g in range(model.ngeom):
        if int(model.geom_type[g]) == int(S.mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        p = np.asarray(data.geom_xpos[g], float)
        size = float(np.max(model.geom_size[g]))
        radius = max(radius, float(np.linalg.norm(p)))
        if np.linalg.norm(p) > max_extent_m or size > max_extent_m:
            offenders.append({"geom": model.geom(g).name or f"geom{g}",
                              "body": model.body(int(model.geom_bodyid[g])).name,
                              "position": np.round(p, 2).tolist(), "size": round(size, 2)})
    return {"ok": not offenders, "offenders": offenders, "scene_radius_m": round(radius, 2),
            "max_extent_m": max_extent_m}


def main(argv=None):
    rep = check_units(sys.argv[1] if argv is None else argv[0])
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())


units_check = check_units   # the library calls every tool by its own name
