"""Height check: a body's centre sits in a given band of world z."""
from __future__ import annotations

TOOL = {
    "name": "height_check",
    "title": "Height check",
    "type": "verifier",
    "category": "Scene Assertions",
    "summary": "The world-z of a body's centre lies within [lo, hi], with half a millimetre of slack.",
    "description": (
        "Measures the body's height for you and compares it with the band. The band should "
        "be stated relative to the scene's own table top, which is rarely z = 0; pass "
        "`relative_to` (a body name) to express lo and hi as offsets from that body's "
        "centre instead of absolute heights. Half a millimetre of slack keeps a band copied "
        "from a printed measurement from failing on rounding."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF file, XML string, or compiled model."},
        {"name": "body", "type": "str", "description": "The body to measure."},
        {"name": "lo", "type": "float, metres", "description": "Lower bound."},
        {"name": "hi", "type": "float, metres", "description": "Upper bound."},
        {"name": "relative_to", "type": "str | None", "description": "Measure relative to this body's centre height."},
        {"name": "settled", "type": "bool", "description": "Judge after settling (default) or at the initial state."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; z (measured value compared with the band); reference_z."},
    ],
    "requirements": ["mujoco>=3.2"],
    "simulators": ["MuJoCo"],
    "tags": ["assertion", "height"],
    "used_in": ["Place Cup in Bowl", "Place Toy in Drawer"],
    "status": "tested",
    "entrypoint": "height_check",
}

from tool_library.common import scene as S


def check_height(scene, body, lo, hi, relative_to=None, settled=True, seconds=2.0):
    model = S.load_model(scene)
    bid = S.body_id(model, body)
    data = S.forward(model)
    if settled:
        S.settle(model, data, seconds)
    ref = float(data.xpos[S.body_id(model, relative_to)][2]) if relative_to else 0.0
    z = float(data.xpos[bid][2]) - ref
    eps = 5e-4
    return {"ok": (lo - eps) <= z <= (hi + eps), "z": round(z, 4), "reference_z": round(ref, 4),
            "band": [lo, hi]}


height_check = check_height   # the library calls every tool by its own name
