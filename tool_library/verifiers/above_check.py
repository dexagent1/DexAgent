"""Above check: one body's centre is higher than another's."""
from __future__ import annotations

TOOL = {
    "name": "above_check",
    "title": "Above check",
    "type": "verifier",
    "category": "Scene Assertions",
    "summary": "The centre of one body is higher (world z) than the centre of another, optionally "
               "by at least a margin.",
    "description": (
        "A cheap relation for 'the lid is on the box' or 'the cup sits on the tray' that does "
        "not need the two to touch. Compares world-frame body positions, after settling by "
        "default. `min_gap` demands a height difference (metres); zero accepts any ordering."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF file, XML string, or compiled model."},
        {"name": "top", "type": "str", "description": "The body expected higher."},
        {"name": "bottom", "type": "str", "description": "The body expected lower."},
        {"name": "min_gap", "type": "float, metres", "description": "Required height difference. Default 0."},
        {"name": "settled", "type": "bool", "description": "Judge after settling (default) or at the initial state."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; z_top; z_bottom; gap_m."},
    ],
    "requirements": ["mujoco>=3.2"],
    "simulators": ["MuJoCo"],
    "tags": ["assertion", "height"],
    "used_in": ["Place Cup in Bowl", "Giftbox Packing"],
    "status": "tested",
    "entrypoint": "above_check",
}

from tool_library.common import scene as S


def check_above(scene, top, bottom, min_gap=0.0, settled=True, seconds=2.0):
    model = S.load_model(scene)
    bt, bb = S.body_id(model, top), S.body_id(model, bottom)
    data = S.forward(model)
    if settled:
        S.settle(model, data, seconds)
    zt, zb = float(data.xpos[bt][2]), float(data.xpos[bb][2])
    return {"ok": zt - zb >= min_gap, "z_top": round(zt, 4), "z_bottom": round(zb, 4),
            "gap_m": round(zt - zb, 4)}


above_check = check_above   # the library calls every tool by its own name
