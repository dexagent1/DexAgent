"""Joint range check: a joint was built with the intended limits."""
from __future__ import annotations

TOOL = {
    "name": "joint_range_check",
    "title": "Joint range check",
    "type": "verifier",
    "category": "Scene Assertions",
    "summary": "A named joint exists, is limited, and its built range equals the expected one "
               "within a tolerance.",
    "description": (
        "Checks the DEFINED limits of a joint, not its current position: a scene is one "
        "instant, and what can be asserted about a drawer slide or a lid hinge is the travel "
        "it was built with. Fails when the joint does not exist (and lists the ones that do), "
        "when it is unlimited, or when either limit differs from the expectation by more than "
        "`tol`."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF file, XML string, or compiled model."},
        {"name": "joint", "type": "str", "description": "Joint name."},
        {"name": "lo", "type": "float", "description": "Expected lower limit (m or rad)."},
        {"name": "hi", "type": "float", "description": "Expected upper limit (m or rad)."},
        {"name": "tol", "type": "float", "description": "Allowed difference per limit. Default 1e-6."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; range: the built limits; type; limited; current: the joint's value at the initial state."},
    ],
    "requirements": ["mujoco>=3.2"],
    "simulators": ["MuJoCo"],
    "tags": ["assertion", "joint", "range"],
    "used_in": ["Place Toy in Drawer", "Giftbox Packing", "Scissor Cutting"],
    "status": "tested",
    "entrypoint": "joint_range_check",
}

from tool_library.common import scene as S


def check_joint_range(scene, joint, lo, hi, tol=1e-6):
    model = S.load_model(scene)
    jid = S.joint_id(model, joint)
    r = [float(x) for x in model.jnt_range[jid]]
    limited = bool(model.jnt_limited[jid])
    kind = {S.FREE: "free", S.HINGE: "hinge", S.SLIDE: "slide"}.get(int(model.jnt_type[jid]), "ball")
    ok = limited and abs(r[0] - lo) <= tol and abs(r[1] - hi) <= tol
    return {"ok": ok, "range": r, "limited": limited, "type": kind,
            "current": float(model.qpos0[int(model.jnt_qposadr[jid])]), "expected": [lo, hi]}


joint_range_check = check_joint_range   # the library calls every tool by its own name
