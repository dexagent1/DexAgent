"""Stage 1 goal gate: every executed Stage 1 subgoal achieved what its goal sentence says, measured on the trajectory."""
from __future__ import annotations

TOOL = {
    "name": "stage1_goal_gate",
    "title": "Stage 1 goal gate",
    "type": "verifier",
    "category": "Task & Rollout Gates",
    "summary": "Independently measure every Stage 1 subgoal and the task_success final scene; generated expectations cannot redefine success.",
    "description": (
        "Uses the trusted Stage 1 goal sentences, object identities and reviewed declarative predicates. Each "
        "subgoal is checked at its own execution boundary; its temporary grasp, lift or opening requirement is not "
        "automatically imposed on the final scene. The separate task_success sentence is checked at the actual end "
        "of the episode, with a final stability interval when release or holding is required. Clauses whose requirements "
        "are not covered by the supported language/predicate vocabulary fail closed. The trusted Stage 1 input can "
        "add task_success_predicates using {class, bodies: {A, B}, params, side}; they cannot waive unknown prose. Final predicates cannot provide "
        "earlier frame windows. A requirement that an object has not moved is checked throughout the episode against "
        "its initial pose. Object and hand roles for the final scene come from Stage 1/2, not the generated manifest. "
        "Generated goal_predicates only add checks; deviation predicates cannot replace independently inferred goals. "
        "Subgoal state sentences also require complete supported clause coverage; a recognised motion or a reviewed "
        "predicate cannot waive an unknown clause. Unsupported event language needs an implemented measurement. "
        "Unknown prose, unresolved objects, omitted final success, unsupported final-state classes and missing evidence "
        "block acceptance. This is a conservative supported-language verifier, not an unrestricted natural-language oracle."
    ),
    "inputs": [
        {"name": "scene_semantic_json", "type": "str", "description": "Stage 1 scene_semantics.json."},
        {"name": "trajectory_manifest_json", "type": "str", "description": "Stage 3 trajectory_manifest.json (subgoals with stage1_indices, object, side, frames; optional goal_predicates, deviation)."},
        {"name": "scene", "type": "str", "description": "MJCF the trajectory was executed in."},
        {"name": "trajectory", "type": "str", "description": "The delivered trajectory npz (qpos, fps, labels, subgoal)."},
        {"name": "scene_recon_json", "type": "str | None", "description": "Stage 2 scene_recon.json; default the manifest's scene_recon."},
        {"name": "subgoals", "type": "list[int] | None", "description": "Stage 1 indices to diagnose; default all. A proper subset cannot return full-task ok."},
        {"name": "stride", "type": "int", "description": "Frame step for per-frame predicates. Default 1."},
        {"name": "out_json", "type": "str | None", "description": "Optional report path."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; task_success {goal, predicates, coverage, issues, ok}; subgoals [{index, goal, motion, entries, frames, required_classes, predicates [{class, A, B, joint, frames, source, ok, measured, issues, note}], deviation {reason, classes, accepted}, notes, errors, ok}]; counts {subgoals, judged, passed}."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "stage1", "goal", "manifest", "task gate"],
    "status": "tested",
    "entrypoint": "stage1_goal_gate",
}

from pathlib import Path

from tool_library.common import task_gates as G


def stage1_goal_gate(scene_semantic_json, trajectory_manifest_json, scene, trajectory, scene_recon_json=None, subgoals=None,
                     stride=1, out_json=None):
    sem = G.load_json(scene_semantic_json, "scene_semantic_json")
    man = G.load_json(trajectory_manifest_json, "trajectory_manifest_json")
    if not isinstance(sem.get("subgoals"), list):
        raise ValueError("scene_semantic_json: has no 'subgoals' list")
    if not isinstance(man.get("subgoals"), list):
        return G.finish({"ok": False, "issues": ["the manifest has no 'subgoals' list: each executed Stage 1 subgoal needs an entry "
                                                 "(stage1_indices, motion, side, object, frames) before its goal can be checked"],
                         "subgoals": [], "counts": {"subgoals": len(sem["subgoals"]), "judged": 0, "passed": 0}}, out_json)
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
    rep = G.stage1_goal_report(ep, sem, man, recon, only=subgoals, stride=int(stride))
    return G.finish(rep, out_json)


if __name__ == "__main__":
    import json
    import sys
    r = stage1_goal_gate(*sys.argv[1:5])
    print(json.dumps({"ok": r["ok"], "issues": r["issues"], "counts": r["counts"]}, indent=1))
