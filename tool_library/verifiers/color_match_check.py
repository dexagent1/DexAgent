"""Colour match check: a constructed object wears the colours measured on the real one."""
from __future__ import annotations

TOOL = {
    "name": "color_match_check",
    "title": "Colour match check",
    "type": "verifier",
    "category": "Scene Assertions",
    "summary": "The dominant surface colour of a constructed object is within a distance of the "
               "colour palette measured from the video.",
    "description": (
        "Constructed objects are painted with measured colours, never invented ones. The "
        "verifier gathers every visible geom of the object (the body and the bodies named "
        "with its prefix), weights each colour by the geom's volume so a large carcass counts "
        "and a small handle does not, takes the dominant colour, and compares it with the "
        "closest entry of the measured palette in RGB. A distance above `max_dist` (0.30 in "
        "0..1 RGB) fails. Only the dominant surface is judged: a palette measured over an "
        "object's whole mask says nothing about small hardware such as a pull."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF file, XML string, or compiled model."},
        {"name": "object", "type": "str", "description": "Body name or name prefix of the constructed object."},
        {"name": "palette", "type": "list[dict | list]", "description": "Measured colours: [{rgb: [r,g,b], weight}] or plain [r,g,b] triples in 0..1."},
        {"name": "max_dist", "type": "float", "description": "Largest accepted RGB distance. Default 0.30."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; dominant_rgb; closest_palette_rgb; dist; colours: every colour with its volume share."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["assertion", "colour", "appearance"],
    "used_in": ["Place Toy in Drawer"],
    "status": "tested",
    "entrypoint": "color_match_check",
}

import numpy as np

from tool_library.common import scene as S


def check_color_match(scene, object, palette, max_dist=0.30):
    model = S.load_model(scene)
    s = S.slug(object)
    bodies = {b for b in range(1, model.nbody)
              if model.body(b).name in (object, s) or model.body(b).name.startswith(f"{s}_")
              or model.body(b).name.startswith(f"{object}_")}
    bid = S.resolve_body(model, object)
    if bid >= 0:
        bodies |= S.subtree(model, bid)
    if not bodies:
        raise ValueError(f"no body named or prefixed '{object}' in the scene")
    weights = {}
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) not in bodies:
            continue
        rgba = model.geom_rgba[g]
        mid = int(model.geom_matid[g])
        if mid >= 0 and np.allclose(rgba, [0.5, 0.5, 0.5, 1.0]):
            rgba = model.mat_rgba[mid]
        if rgba[3] < 0.5:                 # invisible or translucent parts (a clear straw) do not define the colour
            continue
        key = tuple(round(float(x), 3) for x in rgba[:3])
        vol = S.geom_volume(model, g)
        weights[key] = weights.get(key, 0.0) + vol
    total = sum(weights.values()) or 1.0
    ranked = sorted(weights.items(), key=lambda kv: -kv[1])
    dominant = np.array(ranked[0][0])
    pal = [np.asarray(p["rgb"] if isinstance(p, dict) else p, float)[:3] for p in palette]
    dists = [float(np.linalg.norm(dominant - p)) for p in pal]
    i = int(np.argmin(dists))
    return {"ok": dists[i] <= max_dist, "dominant_rgb": dominant.tolist(),
            "closest_palette_rgb": pal[i].tolist(), "dist": round(dists[i], 3),
            "colours": [{"rgb": list(k), "share": round(v / total, 3)} for k, v in ranked[:6]]}


color_match_check = check_color_match   # the library calls every tool by its own name
