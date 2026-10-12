"""Required bodies check: is every object the task names actually in the scene?"""
from __future__ import annotations

TOOL = {
    "name": "required_bodies_check",
    "title": "Required bodies check",
    "type": "verifier",
    "category": "Scene Compile Checks",
    "summary": "Every object named by the task (the stage-1 object list) resolves to a body in "
               "the compiled scene.",
    "description": (
        "The scene has to contain exactly the objects the task analysis named; an object that "
        "was dropped during reconstruction makes every later stage silently work on a "
        "different task. Names are matched the way the rest of the library matches them: an "
        "exact body name, its slug ('yellow cup' is 'yellow_cup'), or a body whose name "
        "starts or ends with the slug as whole '_'-separated parts (a constructed object's "
        "'cup_body', an assembly part 'cabinet_drawer'). A slug inside a longer word does not "
        "count ('cup' is not 'cupboard'), and a robot body (robot_ prefix) never stands in for "
        "a task object unless named exactly. Missing names are listed with the bodies that do "
        "exist, so the mismatch is easy to see."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF file, XML string, or compiled model."},
        {"name": "required", "type": "list[str]", "description": "Object names the scene must contain."},
    ],
    "outputs": [
        {"name": "report", "type": "dict",
         "description": "ok; found: {name: body}; missing: [names]; bodies: every body in the scene."},
    ],
    "requirements": ["mujoco>=3.2"],
    "simulators": ["MuJoCo"],
    "tags": ["compile", "contract", "objects"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer"],
    "status": "tested",
    "entrypoint": "required_bodies_check",
}

from tool_library.common import scene as S


def check_required_bodies(scene, required):
    model = S.load_model(scene)
    names = S.body_names(model)[1:]
    scene_names = [n for n in names if not n.startswith("robot_")]   # a robot link is never a task object
    found, missing = {}, []
    for req in required:
        bid = S.resolve_body(model, req)
        if bid >= 0 and (model.body(bid).name == str(req) or not model.body(bid).name.startswith("robot_")):
            found[req] = model.body(bid).name
            continue
        s = S.slug(req)
        cands = [n for n in scene_names if n.startswith(f"{s}_") or n.endswith(f"_{s}")]
        if cands:
            found[req] = cands[0]
        else:
            missing.append(req)
    return {"ok": not missing, "found": found, "missing": missing, "bodies": names}


required_bodies_check = check_required_bodies   # the library calls every tool by its own name
