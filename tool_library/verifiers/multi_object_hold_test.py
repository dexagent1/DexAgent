"""Stress test of one hand holding two (or more) objects taken one after another with disjoint fingers."""
from __future__ import annotations

TOOL = {
    "name": "multi_object_hold_test",
    "title": "Multi-object hold test",
    "type": "verifier",
    "category": "Grasp Stress Tests",
    "summary": "Drives one hand through taking two (or more) objects one after another with disjoint fingers (the first cup "
               "between thumb and index, the second on the other three) and then shakes, lifts and turns them together; "
               "passes when every object rose with the hand and stayed in it through every phase.",
    "description": (
        "A second grasp can undo the first: the hand turns and travels with the first object hanging in it, the free "
        "fingers close next to the held ones, and the combined load changes how both grips hold. The test runs, through "
        "the actuators with the executor's clamps and timing (hand.multi_object_stress_rollout): the first grasp (open "
        "approach along its approach, close to its closed targets, the clamp), a lift of lift1_m; then for each next "
        "pose, holding what it has, the move to that pose's pre-grasp, the descent and the close of ONLY that pose's "
        "grip_fingers (the fingers already holding keep their press; overlapping allocations are refused); then every "
        "object together: lifts of lift_heights_m, shakes of shake_m in five directions, turns of turn_deg about the "
        "wrist's three axes and a hold. Every phase is judged on every object held by then: it rose with the hand "
        "(at least half of each lift), it drifted less than drift_max_m in the wrist frame on every integration step "
        "since that object's seating, turned no more than 15 degrees relative to the wrist for a rigid held part, "
        "and it still touches the "
        "hand; the seat of each grasp also needs one of its allocated fingers on its object. This is the stress test a "
        "candidate pair can use before execution (with the first grasp it was designed with: second_grasp_pose). "
        "For final grasp_more acceptance, run combined_hold_state_test on the actual executed segment-end state; "
        "a restart before the first grasp cannot certify the current retained-object state."
    ),
    "inputs": [
        {"name": "scene", "type": "dict | str", "description": "Hand scene (hand_scene_from_recon) holding every object as it lies before the first grasp."},
        {"name": "poses", "type": "list", "description": "Grasp poses in grasp order (dicts or JSON paths), each with its grip_fingers."},
        {"name": "objects", "type": "list", "description": "The bodies the poses take, in the same order."},
        {"name": "lift1_m", "type": "float", "description": "Lift after the first grasp. Default 0.04."},
        {"name": "lift_heights_m", "type": "list", "description": "Lifts with every object held. Default [0.05, 0.10]."},
        {"name": "shake_m", "type": "float", "description": "Default 0.03."},
        {"name": "turn_deg", "type": "float", "description": "Default 15."},
        {"name": "drift_max_m", "type": "float", "description": "Largest drift of an object in the wrist frame. Default 0.03."},
    ],
    "outputs": [
        {"name": "ok", "type": "bool", "description": "Every phase held every object."},
        {"name": "first_fail", "type": "str | None", "description": "The first phase that failed."},
        {"name": "phases", "type": "list", "description": "[{name, ok, objects: {object_i: {drift_mm, touching, rose_mm?}}, ...}]."},
        {"name": "allocation", "type": "list", "description": "The fingers each grasp uses."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["grasp", "stress", "finger allocation", "two objects"],
    "used_in": [],
    "status": "tested",
    "entrypoint": "multi_object_hold_test",
}

import mujoco


def multi_object_hold_test(scene, poses, objects, lift1_m=0.04, lift_heights_m=(0.05, 0.10), shake_m=0.03, turn_deg=15.0, drift_max_m=0.03):
    from tool_library.common import hand as H
    from tool_library.common.subgoal import load_pose
    # every object gets hand contact pairs (a hand scene pairs the hand only with the geoms near its target by default,
    # and the first object would be untouchable)
    model, data0, rig, ids = H.open_scene(scene, pair_region=None)
    if len(poses) != len(objects):
        raise ValueError("give one pose per object")
    bids, gsets = [], []
    for name in objects:
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if b < 0:
            raise ValueError(f"no body {name!r} in the hand scene")
        root = int(model.body_rootid[b])
        sub, st = {root}, [root]
        while st:
            x = st.pop()
            for c in range(model.nbody):
                if int(model.body_parentid[c]) == x and c not in sub:
                    sub.add(c); st.append(c)
        bids.append(int(b))
        gsets.append({g for g in ids["obj_gids"] if int(model.geom_bodyid[g]) in sub})
    r = H.multi_object_stress_rollout(model, data0, rig, [dict(load_pose(p)) for p in poses], bids, gsets, lift1_m=lift1_m,
                                      lift_heights=tuple(lift_heights_m), shake_m=shake_m, turn_deg=turn_deg, drift_max_m=drift_max_m, ids=ids)
    r.pop("data", None)
    r["objects"] = list(objects)
    return r
