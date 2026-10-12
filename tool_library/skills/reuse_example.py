"""Copy a verified example's stage outputs into a run's workspace."""
from __future__ import annotations

TOOL = {
    "name": "reuse_example",
    "title": "Reuse a verified example",
    "type": "skill",
    "category": "Perception & Grounding",
    "summary": "Copies one stage's verified outputs of an example (stage 1: scene_semantics.json; stage 2: scene.xml with its "
               "assets, scene_recon.json, camera.json, the per-object assets; stage 3: trajectory.npz, phases.json, "
               "trajectory_manifest.json and a ready Stage 4 task folder with the run's own video) into the workspace.",
    "description": (
        "For a video find_example calls `identical`, the example IS the answer: its scene was reconstructed and checked "
        "and its trajectory was proved, so the stage copies it instead of solving again, and the run reproduces the "
        "ground truth exactly. Scene files are rewritten with absolute paths into the shared asset pool so they "
        "compile from the workspace. Stage 2 also copies the per-object assets (objects/<id>/<id>.xml), which a "
        "same-task run uses to re-place the same objects in a new layout (stage_on_bench or fit_object_placement on "
        "the copied scene)."
    ),
    "inputs": [
        {"name": "example", "type": "str", "description": "Example folder (from find_example)."},
        {"name": "stage", "type": "int", "description": "1, 2 or 3."},
        {"name": "workspace", "type": "str", "description": "The stage's workspace."},
        {"name": "video", "type": "str | None", "description": "The run's video (stage 3: becomes the Stage 4 task's human video)."},
    ],
    "outputs": [
        {"name": "result", "type": "dict", "description": "stage, example, files written."},
    ],
    "requirements": ["numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["examples", "reuse", "asset reuse"],
    "used_in": ["Battery Insertion", "Open the Bottle", "Giftbox Packing", "Computer Installation", "Drawing and Wiping", "Knot the Rope",
                "Place Cup in Bowl", "Place Toy in Drawer", "Scissor Cutting", "Biology Experiment", "Multi-Cup Grasping", "Meal Box (extra sample)"],
    "status": "tested",
    "entrypoint": "reuse_example",
}

from tool_library.common import examples as EX


def reuse_example(example, stage, workspace, video=None):
    return EX.reuse_example(example, stage, workspace, video=video)
