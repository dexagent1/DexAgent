"""Probe scene: where everything is, in numbers."""
from __future__ import annotations

TOOL = {
    "name": "probe_scene",
    "title": "Probe scene",
    "type": "skill",
    "category": "Scene Runtime & Rendering",
    "summary": "Read-only inspection of a compiled scene: every body's world position and bounds, "
               "the joints with their ranges and current values, and the contacts, at the initial "
               "or a given state.",
    "description": (
        "What an author needs to see before writing an assertion or a placement: the world "
        "position and axis-aligned bounds of each body, which bodies are free, fixed or "
        "jointed, each joint's type, range and value, and which bodies touch. Works at the "
        "initial state or at any qpos. The bounds come from the geoms' bounding radii, so they "
        "are quick and slightly loose."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF file, XML string, or compiled model."},
        {"name": "qpos", "type": "ndarray | None", "description": "State to inspect; the initial state when omitted."},
        {"name": "bodies", "type": "list[str] | None", "description": "Restrict to these bodies."},
    ],
    "outputs": [
        {"name": "report", "type": "dict",
         "description": "bodies: {name: {pos, bounds, kind in free|fixed|jointed, mass}}; joints: [{name, type, "
                        "range, value, body}]; contacts: [[a, b, dist]]."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["scene", "inspect", "debug"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing"],
    "status": "tested",
    "entrypoint": "probe_scene",
    "cli": "python -m tool_library.skills.probe_scene scene.xml",
}

import json
import sys

import numpy as np

from tool_library.common import scene as S
from tool_library.skills.compile_scene import KIND


def probe_scene(scene, qpos=None, bodies=None):
    model = S.load_model(scene)
    data = S.forward(model, qpos)
    want = None if bodies is None else {S.body_id(model, b) for b in bodies}
    out_b = {}
    for b in range(1, model.nbody):
        if want is not None and b not in want:
            continue
        gids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == b]
        bounds = None
        if gids:
            lo = np.min([data.geom_xpos[g] - model.geom_rbound[g] for g in gids], 0)
            hi = np.max([data.geom_xpos[g] + model.geom_rbound[g] for g in gids], 0)
            bounds = [np.round(lo, 3).tolist(), np.round(hi, 3).tolist()]
        kind = "free" if S.has_free_joint(model, b) else (
            "jointed" if model.body_jntnum[b] > 0 or int(model.body_weldid[b]) != 0 else "fixed")
        out_b[model.body(b).name] = {"pos": np.round(data.xpos[b], 4).tolist(), "bounds": bounds,
                                     "kind": kind, "mass": round(float(model.body_mass[b]), 4)}
    joints = []
    for j in range(model.njnt):
        q = int(model.jnt_qposadr[j])
        kind = KIND.get(int(model.jnt_type[j]), "?")
        joints.append({"name": model.joint(j).name, "type": kind,
                       "range": [float(x) for x in model.jnt_range[j]] if model.jnt_limited[j] else None,
                       "value": None if kind == "free" else round(float(data.qpos[q]), 5),
                       "body": model.body(int(model.jnt_bodyid[j])).name})
    contacts = [[a, b, round(d, 5)] for a, b, d in S.contact_pairs(model, data)]
    return {"bodies": out_b, "joints": joints, "contacts": contacts, "nq": int(model.nq)}


def main(argv=None):
    rep = probe_scene(sys.argv[1] if argv is None else argv[0])
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
