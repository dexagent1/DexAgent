"""No-float check: a free body is supported after the scene settles."""
from __future__ import annotations

TOOL = {
    "name": "no_float_check",
    "title": "No-float check",
    "type": "verifier",
    "category": "Scene Assertions",
    "summary": "A single free body touches something after settling; welded or jointed bodies "
               "pass trivially because they are held kinematically.",
    "description": (
        "The per-object form of the support check, for asserting one placement. Only a body "
        "with a free joint can float; a welded body or a part on a slide is reported as held. "
        "The scene is settled for `seconds` and the body must then have at least one contact."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF file, XML string, or compiled model."},
        {"name": "body", "type": "str", "description": "The body to check."},
        {"name": "seconds", "type": "float", "description": "Settle time. Default 2.0."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; free (has a free joint); touching: bodies it is in contact with; fell_m: how far it dropped."},
    ],
    "requirements": ["mujoco>=3.2"],
    "simulators": ["MuJoCo"],
    "tags": ["assertion", "support", "floating"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping"],
    "status": "tested",
    "entrypoint": "no_float_check",
}

from tool_library.common import scene as S


def check_no_float(scene, body, seconds=2.0):
    model = S.load_model(scene)
    bid = S.body_id(model, body)
    name = model.body(bid).name
    if not S.has_free_joint(model, bid):
        return {"ok": True, "free": False, "touching": [], "fell_m": 0.0}
    data = S.forward(model)
    z0 = float(data.xpos[bid][2])
    S.settle(model, data, seconds)
    touching = sorted({b if a == name else a for a, b, _ in S.contact_pairs(model, data) if name in (a, b)})
    return {"ok": bool(touching), "free": True, "touching": touching,
            "fell_m": round(z0 - float(data.xpos[bid][2]), 4)}


no_float_check = check_no_float   # the library calls every tool by its own name
