"""Carry a verified example's trajectory onto a new layout of the same task."""
from __future__ import annotations

TOOL = {
    "name": "adapt_example_trajectory",
    "title": "Adapt an example's trajectory",
    "type": "skill",
    "category": "Trajectory & Execution",
    "summary": "For the same task with the same objects in a new initial layout: re-anchors the example's verified trajectory "
               "to where each object now starts (phase by phase, by the example's anchor schedule), re-solves the arms by "
               "IK and accepts it only if reach, joint-speed, penetration and object-continuity checks pass.",
    "description": (
        "The example's trajectory proved a set of hand-object relative motions; they stay valid when an object starts "
        "a few centimetres or degrees elsewhere, as long as each hand follows the object it works on. The layout is "
        "per movable object (dx, dy, dyaw_deg) relative to the example's frame-0 placement: give it, or give the run's "
        "Stage 2 scene_recon.json and it is computed from the objects' poses (matched by Stage 1 id). This is the Stage "
        "4 randomization machinery applied to one chosen layout, so it takes seconds, not a solve. On success it writes "
        "trajectory.npz, the adapted scene (objects fixed to the world moved) and adapt_report.json; on failure the "
        "report names the failed checks (reach, speed, penetration, continuity) and the frames, and the subgoals are "
        "solved normally."
    ),
    "inputs": [
        {"name": "example", "type": "str", "description": "Example folder (from find_example)."},
        {"name": "workspace", "type": "str", "description": "Where to write the adapted trajectory."},
        {"name": "layout", "type": "dict | None", "description": "{body: [dx, dy, dyaw_deg]} for the example's movable objects."},
        {"name": "scene_recon_json", "type": "str | None", "description": "The run's Stage 2 manifest, to compute the layout from."},
        {"name": "checks", "type": "dict | None", "description": "Override the check tolerances (ik_pos_tol_m, ik_rot_tol_deg, penetration_tol_m, ...)."},
    ],
    "outputs": [
        {"name": "result", "type": "dict", "description": "ok, failed checks, layout, report, trajectory path, adapted scene path."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["examples", "reuse", "adaptation", "layout", "stage3"],
    "used_in": ["Battery Insertion", "Open the Bottle", "Place Toy in Drawer", "Place Cup in Bowl", "Computer Installation"],
    "status": "tested",
    "entrypoint": "adapt_example_trajectory",
}

from tool_library.common import examples as EX


def adapt_example_trajectory(example, workspace, layout=None, scene_recon_json=None, checks=None):
    if layout is None:
        if scene_recon_json is None:
            raise ValueError("give the layout or the run's scene_recon.json")
        lay = EX.layout_from_recon(example, scene_recon_json)
        if lay["unmatched"]:
            raise ValueError(f"objects of the example not found in the run's reconstruction: {lay['unmatched']}")
        layout = lay["objects"]
    return EX.adapt_example(example, {"objects": layout}, workspace, checks=checks)
