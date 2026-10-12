"""Scene QC: the default sensor stack every compiled scene must pass."""
from __future__ import annotations

TOOL = {
    "name": "scene_qc",
    "title": "Scene QC",
    "type": "verifier",
    "category": "Scene Compile Checks",
    "summary": "Runs the whole compile-time battery on a scene: finite model, units, initial "
               "penetration, settle stability, support, and optionally required objects, "
               "articulation and silhouette IoU, and returns one verdict with every measurement.",
    "description": (
        "The checks a reconstructed scene must pass before any task-specific assertion is "
        "worth running, in the order a failure explains the next: a model that does not "
        "compile into finite physics, geometry in the wrong units, objects that start inside "
        "each other, objects that fall or tip when physics runs, and objects left hanging in "
        "the air. With the optional inputs it also checks that the task's objects are all "
        "present, that the ones declared articulated have a mechanism, and that the render "
        "matches the video masks. Each sub-check is the standalone verifier of the same name, "
        "so a failing item can be rerun alone. The report keeps every number and lists the "
        "failures by check name."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF file, XML string, or compiled model."},
        {"name": "required", "type": "list[str] | None", "description": "Objects that must exist (required_bodies_check)."},
        {"name": "articulated", "type": "list[str] | None", "description": "Objects that must have a joint (articulation_check)."},
        {"name": "masks", "type": "dict | None", "description": "Object -> video mask for silhouette_iou_check; needs `camera`."},
        {"name": "ignore_mask", "type": "str | None", "description": "Pixels left out of every silhouette and mask (the demonstrator's hands): a PNG path sized like the masks."},
        {"name": "camera", "type": "str | None", "description": "Video-registered camera name for the silhouette check."},
        {"name": "groups", "type": "dict[str, list[str]] | None", "description": "Mask key -> the piece root bodies of a group object (scene_recon `parts`): scored by the union silhouette of its pieces."},
        {"name": "hidden", "type": "list[str] | None", "description": "Mask keys of objects declared hidden at frame 0 (scene_recon `hidden_at_frame0`): they must render (near) invisible through the camera instead of matching an IoU."},
        {"name": "penetration_tol", "type": "float, metres", "description": "Initial interpenetration allowed. Default 0.01."},
        {"name": "settle_seconds", "type": "float", "description": "Settle time for the stability and support checks. Default 2.0."},
    ],
    "outputs": [
        {"name": "report", "type": "dict",
         "description": "ok; failed: names of the sub-checks that failed; checks: {name: that check's own report}."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["compile", "qc", "scene"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Battery Insertion", "Computer Installation", "Open the Bottle", "Biology Experiment"],
    "status": "tested",
    "entrypoint": "scene_qc",
    "cli": "python -m tool_library.verifiers.scene_qc scene.xml",
}

import json
from pathlib import Path
import sys

from tool_library.common import scene as S
from tool_library.verifiers.articulation_check import check_articulation
from tool_library.verifiers.model_finite_check import check_model_finite
from tool_library.verifiers.penetration_check import check_penetration
from tool_library.verifiers.required_bodies_check import check_required_bodies
from tool_library.verifiers.settle_stability_check import check_settle_stability
from tool_library.verifiers.silhouette_iou_check import check_silhouette_iou
from tool_library.verifiers.support_check import check_support
from tool_library.verifiers.units_check import check_units


def run_scene_qc(scene, required=None, articulated=None, masks=None, camera=None, ignore_mask=None,
                 penetration_tol=0.01, settle_seconds=2.0, groups=None, hidden=None):
    model = S.load_model(scene)
    checks = {}
    checks["model_finite"] = check_model_finite(model)
    if not checks["model_finite"]["ok"]:
        return {"ok": False, "failed": ["model_finite"], "checks": checks}
    checks["units"] = check_units(model)
    # initial-state penetration on the scene's own collision geometry; a
    # compiled model is passed so visual-only parts are not promoted here
    pen = check_penetration(model, model.qpos0, tolerance=penetration_tol)
    pen.pop("n_geoms_made_collidable", None)
    checks["penetration"] = pen
    checks["settle_stability"] = check_settle_stability(model, seconds=settle_seconds)
    checks["settle_stability"].pop("settled_qpos", None)
    checks["support"] = check_support(model, seconds=settle_seconds)
    if required:
        checks["required_bodies"] = check_required_bodies(model, required)
    if articulated:
        checks["articulation"] = check_articulation(model, articulated)
    if masks:
        if not camera:
            raise ValueError("silhouette check needs the video camera name")
        checks["silhouette_iou"] = check_silhouette_iou(model, masks, camera, ignore_mask=ignore_mask, groups=groups, hidden=hidden)
    failed = [k for k, v in checks.items() if not v["ok"]]
    # advisory: objects an arm cannot reach at all (the scene may be right and the bench too small; Stage 3 will
    # not be able to act on them, so it is said here, where the placement is made)
    warnings = []
    try:
        if isinstance(scene, (str, Path)):
            from tool_library.verifiers.object_reach_check import object_reach_check
            reach = object_reach_check(str(scene))
            checks["object_reach"] = {"ok": True, "advisory": True, "reachable": reach["ok"], "unreachable": reach["unreachable"], "verdict": reach["verdict"]}
            if not reach["ok"]:
                warnings.append(reach["verdict"])
    except Exception as e:
        checks["object_reach"] = {"ok": True, "advisory": True, "error": f"{type(e).__name__}: {e}"}
    return {"ok": not failed, "failed": failed, "checks": checks, "warnings": warnings}


def main(argv=None):
    rep = run_scene_qc(sys.argv[1] if argv is None else argv[0])
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())


scene_qc = run_scene_qc   # the library calls every tool by its own name
