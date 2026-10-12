"""No-contact check: two named bodies do not touch."""
from __future__ import annotations

TOOL = {
    "name": "no_contact_check",
    "title": "No-contact check",
    "type": "verifier",
    "category": "Scene Assertions",
    "summary": "Two bodies are not in contact, at the initial state or after the scene settles.",
    "description": (
        "The complement of contact_check, for relations such as 'the cup is not yet in the "
        "bowl': the scene is the task's starting state, and asserting the goal relation on "
        "it is a common mistake this pair of checks makes explicit. Contacts come from "
        "MuJoCo's collision detection on the settled scene by default."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF file, XML string, or compiled model."},
        {"name": "a", "type": "str", "description": "First body."},
        {"name": "b", "type": "str", "description": "Second body."},
        {"name": "settled", "type": "bool", "description": "Judge after settling (default) or at the initial state."},
        {"name": "seconds", "type": "float", "description": "Settle time. Default 2.0."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; in_contact; contacts touching either body."},
    ],
    "requirements": ["mujoco>=3.2"],
    "simulators": ["MuJoCo"],
    "tags": ["assertion", "contact"],
    "used_in": ["Place Cup in Bowl", "Place Toy in Drawer"],
    "status": "tested",
    "entrypoint": "no_contact_check",
}

from tool_library.verifiers.contact_check import check_contact


def check_no_contact(scene, a, b, settled=True, seconds=2.0):
    rep = check_contact(scene, a, b, settled=settled, seconds=seconds)
    rep["ok"] = not rep["in_contact"]
    return rep


no_contact_check = check_no_contact   # the library calls every tool by its own name
