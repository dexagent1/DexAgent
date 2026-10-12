"""Table clearance check: a carried object clears the table and the things on it.

    from tool_library.verifiers.table_clearance_check import check_table_clearance
    check_table_clearance("scene.xml", "trajectory.npz", body="cup", table_top_z=0.0,
                          frames=["*carry*"])

While an object travels, its lowest point must stay above the tabletop and
above the tops of the bodies it passes over; a plan that drags the object
through the table reads fine in a kinematic replay and fails in physics. The
lowest point is taken from the object's geoms (box, cylinder, capsule,
sphere and mesh bounds), so a tilted object is judged by its real corner.
"""
from __future__ import annotations

TOOL = {
    "name": "table_clearance_check",
    "title": "Table clearance check",
    "type": "verifier",
    "category": "Task & Rollout Gates",
    "summary": "Lowest point of a carried body stays above the table top and above other bodies it passes over during the carry.",
    "description": (
        "Over the carry window (frames, phase patterns, or the object's own motion window), "
        "replays the trajectory, computes the world-space lowest point of the body's geoms "
        "(from the oriented bounds of each geom, meshes included) and reports the minimum "
        "clearance to the table top. It fails below min_clearance (default -12 mm: a sliver of "
        "soft-contact penetration is how a resting object is carried, more is a collision). "
        "With obstacles (body names, or every non-robot body by default) it also requires "
        "that whenever the carried body's footprint overlaps an obstacle's footprint in XY, "
        "its lowest point is above that obstacle's top. The table top height can be given or "
        "read from a body named table."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF scene."},
        {"name": "trajectory", "type": "str | ndarray [T, nq]", "description": "trajectory.npz or qpos array."},
        {"name": "body", "type": "str", "description": "The carried object."},
        {"name": "table_top_z", "type": "float | None", "description": "Table surface height; default from the body named 'table'."},
        {"name": "frames", "type": "iterable[int] | list[str] | None", "description": "Carry window; default the object's motion window."},
        {"name": "obstacles", "type": "list[str] | None", "description": "Bodies to clear when passing over; default all non-robot bodies."},
        {"name": "min_clearance_m", "type": "float", "description": "Default -0.012."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; min clearance to the table (m) with frame and label; per obstacle the worst overlap; window used."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["carry", "clearance", "table", "collision"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Computer Installation", "Open the Bottle"],
    "status": "tested",
    "entrypoint": "table_clearance_check",
}

import json
import sys

import mujoco
import numpy as np

from tool_library.common import trajectory as TJ


def check_table_clearance(scene, trajectory, body, table_top_z=None, frames=None, obstacles=None,
                          min_clearance_m=None, robot_prefix="robot_"):
    g = TJ.GATES
    min_clearance_m = g["table_clearance_min_mm"] / 1000.0 if min_clearance_m is None else min_clearance_m
    model = TJ.load_model(scene)
    Q, labels, fps = TJ.as_trajectory(trajectory)
    bid = TJ.body_id(model, body)
    mesh_cache = TJ.mesh_bounds_cache(model)
    if table_top_z is None:
        table_top_z = TJ.table_top_from_scene(model)
    if frames is None:
        P, _ = TJ.pose_track(model, Q, bid)
        w = TJ.motion_window(P)
        if w is None:
            return {"ok": True, "note": f"{body} never moves; nothing carried", "body": body}
        frames = range(w[0], w[1] + 1)
    elif len(frames) and isinstance(next(iter(frames)), str):
        frames = TJ.frames_matching(labels, list(frames))
        if not frames:
            raise ValueError("the phase patterns match no frame (does the trajectory carry labels?)")
    frames = list(frames)
    ob_ids = TJ.subtree(model, bid)
    if obstacles is None:
        obstacles = [model.body(b).name for b in range(1, model.nbody)
                     if b not in ob_ids and not model.body(b).name.startswith(robot_prefix)
                     and model.body(b).name != "table" and model.body_geomnum[b] > 0]
    obs = {n: TJ.subtree(model, TJ.body_id(model, n)) for n in obstacles}
    data = mujoco.MjData(model)
    clear = np.zeros(len(frames))
    worst_obs = {n: None for n in obs}
    for k, t in enumerate(frames):
        data.qpos[:] = Q[t]
        mujoco.mj_kinematics(model, data)
        lo, hi = TJ.body_bounds(model, data, ob_ids, mesh_cache)
        clear[k] = lo[2] - table_top_z
        for n, ids in obs.items():
            olo, ohi = TJ.body_bounds(model, data, ids, mesh_cache)
            if np.all(lo[:2] < ohi[:2]) and np.all(hi[:2] > olo[:2]):   # footprints overlap
                gap = lo[2] - ohi[2]
                if worst_obs[n] is None or gap < worst_obs[n]["gap_m"]:
                    worst_obs[n] = {"gap_m": round(float(gap), 4), "frame": t, "label": labels[t]}
    k = int(np.argmin(clear))
    rep = {"body": body, "table_top_z": table_top_z, "window": [frames[0], frames[-1]],
           "min_clearance_m": round(float(clear[k]), 4), "at": frames[k], "label": labels[frames[k]],
           "limit_m": min_clearance_m,
           "obstacles": {n: v for n, v in worst_obs.items() if v is not None}}
    bad = [n for n, v in rep["obstacles"].items() if v["gap_m"] < min_clearance_m]
    rep["passes_through"] = bad
    rep["ok"] = bool(clear[k] >= min_clearance_m and not bad)
    return rep


if __name__ == "__main__":
    rep = check_table_clearance(sys.argv[1], sys.argv[2], sys.argv[3])
    print(json.dumps(rep, indent=1, default=str)); sys.exit(0 if rep["ok"] else 1)


table_clearance_check = check_table_clearance   # the library calls every tool by its own name
