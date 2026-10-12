"""Find the verified example a new video (and its Stage 1 result) matches."""
from __future__ import annotations

TOOL = {
    "name": "find_example",
    "title": "Find a verified example",
    "type": "skill",
    "category": "Perception & Grounding",
    "summary": "Compares a video's perceptual fingerprint (and, when given, its Stage 1 result) with every verified example "
               "(examples/default and examples/asset_reuse) and says whether it is the SAME recording (reuse the example's "
               "outputs verbatim), the SAME TASK in another take or layout (reuse the example's objects and adapt its "
               "trajectory), or neither.",
    "description": (
        "Every example stores a fingerprint of its video (24 frames at equal fractions of the clip, grey 32x24, "
        "normalised), its Stage 1 result, its scene and its verified trajectory, but no video. A re-encoded or resized "
        "copy of the same clip scores a mean normalised cross-correlation above 0.99; two takes of one task on the same "
        "bench score 0.8-0.92. `identical` needs mean >= 0.97, every frame >= 0.93 and durations within 5%. `same_task` "
        "needs the Stage 1 result to list the same objects (ids and types) with a similar prompt, or a close video with "
        "the same object words. Call it first in every stage: an identical match is reused with reuse_example (the "
        "stage is then done); a same-task match lets Stage 2 start from the example's objects and Stage 3 adapt the "
        "example's trajectory to the new layout (adapt_example_trajectory) instead of solving from scratch."
    ),
    "inputs": [
        {"name": "video", "type": "str", "description": "The run's video."},
        {"name": "scene_semantic", "type": "str | dict | None", "description": "The run's Stage 1 result (for the same-task test). Default none."},
        {"name": "examples_dir", "type": "str | None", "description": "Folder with default/ and asset_reuse/. Default the repository's examples/."},
    ],
    "outputs": [
        {"name": "result", "type": "dict", "description": "match ('identical' | 'same_task' | 'none'), example (folder), task, best {video similarity, semantic similarity}, candidates (top 5)."},
    ],
    "requirements": ["numpy", "opencv-python"],
    "simulators": [],
    "tags": ["examples", "reuse", "fingerprint", "asset reuse"],
    "used_in": ["Battery Insertion", "Open the Bottle", "Giftbox Packing", "Computer Installation", "Drawing and Wiping", "Knot the Rope",
                "Place Cup in Bowl", "Place Toy in Drawer", "Scissor Cutting", "Biology Experiment", "Multi-Cup Grasping", "Meal Box (extra sample)"],
    "status": "tested",
    "entrypoint": "find_example",
}

from tool_library.common import examples as EX


def find_example(video, scene_semantic=None, examples_dir=None):
    return EX.find_example(video=video, scene_semantic=scene_semantic, examples_dir=examples_dir)
