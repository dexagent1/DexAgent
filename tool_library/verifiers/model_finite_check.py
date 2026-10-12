"""Model finite check: does the scene compile into sane physics at all?"""
from __future__ import annotations

TOOL = {
    "name": "model_finite_check",
    "title": "Model finite check",
    "type": "verifier",
    "category": "Scene Compile Checks",
    "summary": "Every body has a finite, non-negative mass and inertia, and the forward "
               "kinematics of the initial state are finite.",
    "description": (
        "The first thing to ask of a reconstructed scene, before any physics is read: does "
        "MuJoCo compile it into numbers that mean anything? A mesh with a degenerate hull, a "
        "zero-density body or a geom placed at infinity produces a model whose masses or "
        "state are NaN or negative, and every later check would then fail for a reason that "
        "has nothing to do with the scene. This verifier compiles the MJCF, runs one forward "
        "pass, and reports the bodies whose mass or diagonal inertia is not finite or is "
        "negative, and whether the state itself is finite. Works on any scene."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF file, XML string, or compiled model."},
    ],
    "outputs": [
        {"name": "report", "type": "dict",
         "description": "ok; state_finite; bad_bodies: [{body, mass, inertia}] with the offending values; "
                        "n_bodies."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["compile", "mass", "inertia", "nan"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing"],
    "status": "tested",
    "entrypoint": "model_finite_check",
    "cli": "python -m tool_library.verifiers.model_finite_check scene.xml",
}

import json
import sys

import numpy as np

from tool_library.common import scene as S


def check_model_finite(scene):
    model = S.load_model(scene)
    bad = []
    for b in range(1, model.nbody):
        m = float(model.body_mass[b])
        I = np.asarray(model.body_inertia[b], float)
        if not np.isfinite(m) or m < 0 or not np.isfinite(I).all() or (I < 0).any():
            bad.append({"body": model.body(b).name, "mass": m, "inertia": I.tolist()})
    data = S.forward(model)
    finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.xpos).all())
    return {"ok": finite and not bad, "state_finite": finite, "bad_bodies": bad,
            "n_bodies": int(model.nbody) - 1}


def main(argv=None):
    rep = check_model_finite(sys.argv[1] if argv is None else argv[0])
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())


model_finite_check = check_model_finite   # the library calls every tool by its own name
