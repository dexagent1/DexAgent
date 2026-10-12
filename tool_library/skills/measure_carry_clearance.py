"""Measure how much room a carried object has: above the table, around the other objects.

    from tool_library.skills.measure_carry_clearance import measure_carry_clearance
    m = measure_carry_clearance("scene.xml", "trajectory.npz", body="cup", table_top_z=0.0)
    m["min_table_clearance_m"], m["nearest"]

Over the carry window, the lowest point of the object's geometry against the
tabletop on every frame, the signed distance to every other body's geometry
(from MuJoCo's own distance queries, so meshes and primitives alike), and
the frame where each is tightest. The planning counterpart of the
table_clearance_check gate: it says how high a carry has to go and what it
nearly hits.
"""
from __future__ import annotations

TOOL = {
    "name": "measure_carry_clearance",
    "title": "Measure carry clearance",
    "type": "skill",
    "category": "Measurements",
    "summary": "Per-frame clearance of a carried body to the table top and signed distance to every other body over the carry window.",
    "description": (
        "Replays the carry window (given as frames, phase patterns, or the body's own motion "
        "window) and on each frame computes the lowest point of the body's geoms above the "
        "table top and the minimum signed distance between the body's geoms and each other "
        "body's geoms using mj_geomDistance (negative means penetration). Robot bodies are "
        "included under their own names, so the clearance to the other hand is reported too. "
        "Returns the per-frame table clearance profile, the tightest frame, the nearest "
        "bodies sorted by their closest approach, and the height of the carry at its peak. "
        "The table top can be given or read from a body named table."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF scene with collision geometry (distance queries use it)."},
        {"name": "trajectory", "type": "str | ndarray [T, nq]", "description": "trajectory.npz or qpos array."},
        {"name": "body", "type": "str", "description": "The carried body."},
        {"name": "table_top_z", "type": "float | None", "description": "Table surface height; default from the 'table' body."},
        {"name": "frames", "type": "iterable[int] | list[str] | None", "description": "Carry window; default the motion window."},
        {"name": "others", "type": "list[str] | None", "description": "Bodies to measure distance to; default every other body with geoms."},
        {"name": "stride", "type": "int", "description": "Frame stride for the distance queries. Default 1."},
    ],
    "outputs": [
        {"name": "measurement", "type": "dict", "description": "window, min_table_clearance_m with frame/label, peak_height_m, table_clearance [per frame], nearest: {body: {min_dist_m, frame, label}} sorted."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["measurement", "carry", "clearance", "distance"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Giftbox Packing", "Computer Installation", "Open the Bottle"],
    "status": "tested",
    "entrypoint": "measure_carry_clearance",
}

import json
import sys

import mujoco
import numpy as np

from tool_library.common import trajectory as TJ


def geom_pair_distance(model, data, ga, gb, maxdist=1.0):
    frm = np.zeros(6)
    return float(mujoco.mj_geomDistance(model, data, ga, gb, maxdist, frm))


def measure_carry_clearance(scene, trajectory, body, table_top_z=None, frames=None, others=None, stride=1):
    model = TJ.load_model(scene)
    Q, labels, fps = TJ.as_trajectory(trajectory)
    bid = TJ.body_id(model, body)
    if table_top_z is None:
        table_top_z = TJ.table_top_from_scene(model)
    if frames is None:
        P, _ = TJ.pose_track(model, Q, bid)
        w = TJ.motion_window(P)
        frames = range(w[0], w[1] + 1) if w else range(len(Q))
    elif len(frames) and isinstance(next(iter(frames)), str):
        frames = TJ.frames_matching(labels, list(frames))
        if not frames:
            raise ValueError("the phase patterns match no frame (does the trajectory carry labels?)")
    frames = list(frames)[::max(1, stride)]
    ids = TJ.subtree(model, bid)
    my_geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in ids]
    if others is None:
        others = [model.body(b).name for b in range(1, model.nbody) if b not in ids and model.body_geomnum[b] > 0]
    other_geoms = {n: [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in TJ.subtree(model, TJ.body_id(model, n))]
                   for n in others}
    mesh_cache = TJ.mesh_bounds_cache(model)
    data = mujoco.MjData(model)
    clear, height = np.zeros(len(frames)), np.zeros(len(frames))
    nearest = {n: {"min_dist_m": np.inf, "frame": None, "label": None} for n in others}
    for k, t in enumerate(frames):
        data.qpos[:] = Q[t]
        mujoco.mj_kinematics(model, data)
        lo, hi = TJ.body_bounds(model, data, ids, mesh_cache)
        clear[k], height[k] = lo[2] - table_top_z, hi[2] - table_top_z
        for n, gl in other_geoms.items():
            if not gl:
                continue
            olo, ohi = TJ.body_bounds(model, data, TJ.subtree(model, TJ.body_id(model, n)), mesh_cache)
            gap_aabb = np.max(np.maximum(olo - hi, lo - ohi))
            if gap_aabb > 0.25:                   # far apart: no need for exact distances
                d = float(gap_aabb)
            else:
                d = min(geom_pair_distance(model, data, ga, gb, 0.3) for ga in my_geoms for gb in gl)
            if d < nearest[n]["min_dist_m"]:
                nearest[n] = {"min_dist_m": round(d, 4), "frame": t, "label": labels[t]}
    k = int(np.argmin(clear))
    nearest = dict(sorted(((n, v) for n, v in nearest.items() if v["frame"] is not None),
                          key=lambda kv: kv[1]["min_dist_m"]))
    return {"body": body, "table_top_z": table_top_z, "window": [frames[0], frames[-1]], "n_frames": len(frames),
            "min_table_clearance_m": round(float(clear[k]), 4), "tightest": {"frame": frames[k], "label": labels[frames[k]]},
            "peak_height_m": round(float(height.max()), 4), "peak_at": {"frame": frames[int(np.argmax(height))]},
            "table_clearance": [round(float(c), 4) for c in clear], "frames": frames, "nearest": nearest}


if __name__ == "__main__":
    m = measure_carry_clearance(sys.argv[1], sys.argv[2], sys.argv[3])
    m.pop("table_clearance"); m.pop("frames")
    print(json.dumps(m, indent=1))
