"""Bimanual sync gate: two-hand Stage 1 subgoals are executed with both hands acting together."""
from __future__ import annotations

TOOL = {
    "name": "bimanual_sync_gate",
    "title": "Bimanual sync gate",
    "type": "verifier",
    "category": "Task & Rollout Gates",
    "summary": "For every Stage 1 subgoal whose acting_hand is 'both', both hands act inside overlapping time windows (not "
               "one after the other, not one hand alone).",
    "description": (
        "A demonstration in which two hands press a lid, close two flaps or carry two bricks at once is not reproduced by "
        "one hand doing it and then the other. For each Stage 1 subgoal with acting_hand 'both', the manifest entries "
        "that cover it give the window (their frame span); a hand is active when it actually touches one of the "
        "subgoal's objects (Stage 1 object_refs resolved through scene_recon; 1 mm contact margin), so a left hand that keeps "
        "holding the battery holder while the right hand presses counts. A simultaneous two-hand entry {motion: 'both', "
        "hands: {left: {...}, right: {...}}, sync} is subject to the same physical contact evidence. The subgoal passes "
        "when both hands are active together for at least min_overlap_s. A sequential deviation cannot waive that "
        "requirement. The issue names the two activity spans, or the hand that never acts."
    ),
    "inputs": [
        {"name": "scene_semantic_json", "type": "str", "description": "Stage 1 scene_semantics.json."},
        {"name": "trajectory_manifest_json", "type": "str", "description": "Stage 3 trajectory_manifest.json."},
        {"name": "scene", "type": "str", "description": "MJCF the trajectory was executed in."},
        {"name": "trajectory", "type": "str", "description": "The delivered trajectory npz."},
        {"name": "scene_recon_json", "type": "str | None", "description": "Stage 2 scene_recon.json; default the manifest's scene_recon."},
        {"name": "min_overlap_s", "type": "float", "description": "Time both hands must be active together, s. Default 0.2."},
        {"name": "out_json", "type": "str | None", "description": "Optional report path."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; subgoals [{index, goal, entries, window, active {left, right}, overlap_s, sequential_deviation, ok, note}]."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "stage1", "bimanual", "manifest", "task gate"],
    "status": "tested",
    "entrypoint": "bimanual_sync_gate",
}

from pathlib import Path

from tool_library.common import task_gates as G


def bimanual_sync_gate(scene_semantic_json, trajectory_manifest_json, scene, trajectory, scene_recon_json=None,
                       min_overlap_s=0.2, out_json=None):
    if float(min_overlap_s) < 0:
        raise ValueError("min_overlap_s: must not be negative")
    sem = G.load_json(scene_semantic_json, "scene_semantic_json")
    man = G.load_json(trajectory_manifest_json, "trajectory_manifest_json")
    if not isinstance(man.get("subgoals"), list):
        man = dict(man, subgoals=[])
    recon = None
    rp = Path(scene_recon_json) if scene_recon_json else None
    if rp is None and man.get("scene_recon"):
        q = Path(man["scene_recon"])
        rp = q if q.is_absolute() else Path(trajectory_manifest_json).resolve().parent / q
    if rp is not None and rp.exists():
        recon = G.load_json(rp, "scene_recon_json")
    elif scene_recon_json:
        raise ValueError(f"scene_recon_json: {scene_recon_json} does not exist")
    ep = G.Episode(scene, trajectory)
    rep = G.gate_bimanual_sync(ep, sem, man, recon, min_overlap_s=float(min_overlap_s))
    return G.finish(rep, out_json)


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(bimanual_sync_gate(*sys.argv[1:5]), indent=1))
