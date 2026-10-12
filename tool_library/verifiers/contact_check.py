"""Contact check: two named bodies touch."""
from __future__ import annotations

TOOL = {
    "name": "contact_check",
    "title": "Contact check",
    "type": "verifier",
    "category": "Scene Assertions",
    "summary": "Two bodies are in contact, at the initial state or after the scene settles.",
    "description": (
        "Asserts a task relation such as 'the cup rests on the tray'. Contacts are read from "
        "MuJoCo's own collision detection, so the answer is what the physics sees, not a "
        "distance heuristic. By default the scene is settled first (2 s) because that is the "
        "state the robot will meet; `settled=False` judges the authored initial state. Two "
        "bodies that are both fixed to the world never produce contacts in MuJoCo, so that "
        "case is reported as such instead of as a false failure. Names resolve by exact "
        "name, slug, or unique assembly-part suffix."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF file, XML string, or compiled model."},
        {"name": "a", "type": "str", "description": "First body."},
        {"name": "b", "type": "str", "description": "Second body."},
        {"name": "settled", "type": "bool", "description": "Judge after settling (default) or at the initial state."},
        {"name": "seconds", "type": "float", "description": "Settle time. Default 2.0."},
    ],
    "outputs": [
        {"name": "report", "type": "dict",
         "description": "ok; in_contact; both_static (contact impossible); contacts: all pairs touching "
                        "either body."},
    ],
    "requirements": ["mujoco>=3.2"],
    "simulators": ["MuJoCo"],
    "tags": ["assertion", "contact"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Computer Installation"],
    "status": "tested",
    "entrypoint": "contact_check",
}

from tool_library.common import scene as S


def _state(model, settled, seconds):
    data = S.forward(model)
    if settled:
        S.settle(model, data, seconds)
    return data


def check_contact(scene, a, b, settled=True, seconds=2.0):
    model = S.load_model(scene)
    ba, bb = S.body_id(model, a), S.body_id(model, b)
    both_static = int(model.body_weldid[ba]) == int(model.body_weldid[bb])
    data = _state(model, settled, seconds)
    hit = S.in_contact(model, data, ba, bb)
    names = {model.body(ba).name, model.body(bb).name}
    rel = [(x, y, round(d, 5)) for x, y, d in S.contact_pairs(model, data) if names & {x, y}]
    return {"ok": bool(hit), "in_contact": bool(hit), "both_static": both_static,
            "contacts": rel, "settled": settled}


contact_check = check_contact   # the library calls every tool by its own name
