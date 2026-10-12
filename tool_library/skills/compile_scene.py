"""Compile scene: build an MJCF into a MuJoCo model and say what came out."""
from __future__ import annotations

TOOL = {
    "name": "compile_scene",
    "title": "Compile scene",
    "type": "skill",
    "category": "Scene Runtime & Rendering",
    "summary": "Compiles an MJCF scene, reports its contents (bodies, joints, actuators, cameras, "
               "degrees of freedom) and, when asked, runs the default QC on it.",
    "description": (
        "The step between authoring a scene and using it: MuJoCo's compiler is the first "
        "verifier, and its error message is the most useful thing a broken file can produce. "
        "The skill compiles the file, returns a summary of what the model contains, optionally "
        "saves the compiled model as .mjb at `save_mjb`, and with `qc=True` runs "
        "scene_qc and includes its report. It never edits the scene."
    ),
    "inputs": [
        {"name": "scene_xml", "type": "str", "description": "MJCF file or XML string."},
        {"name": "qc", "type": "bool", "description": "Also run scene_qc. Default False."},
        {"name": "save_mjb", "type": "str | None", "description": "Write the compiled binary model here."},
        {"name": "qc_kwargs", "type": "dict", "description": "Extra arguments for scene_qc (required, articulated, masks, camera)."},
    ],
    "outputs": [
        {"name": "result", "type": "dict",
         "description": "compiled (bool); error (compiler message when it failed); counts: bodies, geoms, joints, "
                        "dofs, actuators, cameras; joints: [{name, type, range}]; cameras; qc: the scene_qc report."},
    ],
    "requirements": ["mujoco>=3.2"],
    "simulators": ["MuJoCo"],
    "tags": ["compile", "scene", "mjcf"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope", "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting", "Biology Experiment"],
    "status": "tested",
    "entrypoint": "compile_scene",
    "cli": "python -m tool_library.skills.compile_scene scene.xml",
}

import json
import sys

import mujoco

from tool_library.common import scene as S

KIND = {S.FREE: "free", S.HINGE: "hinge", S.SLIDE: "slide", int(mujoco.mjtJoint.mjJNT_BALL): "ball"}


def compile_scene(scene_xml, qc=False, save_mjb=None, qc_kwargs=None):
    try:
        model = S.load_model(scene_xml)
    except Exception as exc:
        return {"compiled": False, "error": str(exc)}
    out = {
        "compiled": True,
        "counts": {"bodies": model.nbody - 1, "geoms": model.ngeom, "joints": model.njnt,
                   "dofs": model.nv, "actuators": model.nu, "cameras": model.ncam,
                   "equality_constraints": model.neq},
        "joints": [{"name": model.joint(j).name, "type": KIND.get(int(model.jnt_type[j]), "?"),
                    "range": [float(x) for x in model.jnt_range[j]] if model.jnt_limited[j] else None}
                   for j in range(model.njnt)],
        "cameras": [model.camera(i).name for i in range(model.ncam)],
        "timestep": float(model.opt.timestep),
    }
    if save_mjb:
        mujoco.mj_saveModel(model, str(save_mjb), None)
        out["mjb"] = str(save_mjb)
    if qc:
        from tool_library.verifiers.scene_qc import run_scene_qc
        out["qc"] = run_scene_qc(model, **(qc_kwargs or {}))
    return out


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    rep = compile_scene(args[0], qc="--qc" in args)
    print(json.dumps(rep, indent=1))
    return 0 if rep["compiled"] else 1


if __name__ == "__main__":
    sys.exit(main())
