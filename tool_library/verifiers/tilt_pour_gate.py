"""Tilt / pour gate: a container tilted past an angle with its mouth over a target for a minimum time."""
from __future__ import annotations

TOOL = {
    "name": "tilt_pour_gate",
    "title": "Tilt and pour gate",
    "type": "verifier",
    "category": "Task & Rollout Gates",
    "summary": "A container is tilted past min_tilt_deg with its mouth over a target body's footprint (or a table region) "
               "for a contiguous min_time_s inside a window -- a pour; without a target, a tilt (optionally one kept to "
               "the window's end).",
    "description": (
        "The goal predicate behind 'water transfers from the bottle into the mug' (whose liquid the simulation does not "
        "carry: the gate checks the motion that would pour it) and 'the bottle's long axis is roughly horizontal'. The "
        "tilt is the angle between the container's up axis (up_local, default the body axis that pointed up on frame 0) "
        "and the vertical. The mouth is mouth_local in the body frame, by default the point of the container's collision "
        "geoms farthest along its up axis on frame 0 (the top centre). A frame pours when the tilt reaches min_tilt_deg "
        "and the mouth's x, y lie over the target's collision footprint expanded by target_margin_m (or inside `region`, "
        "table frame); the longest contiguous pouring run must last min_time_s. With neither target nor region only the "
        "tilt counts; at_end requires the qualifying run to reach the window's last frame (a tilt that must be kept). "
        "The issue says whether the tilt was too small, the mouth never over the target, or the pour too short."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the trajectory was executed in."},
        {"name": "trajectory", "type": "str", "description": "Episode npz."},
        {"name": "body", "type": "str", "description": "The tilted container."},
        {"name": "target", "type": "str | None", "description": "Body the mouth must be over (the mug)."},
        {"name": "region", "type": "dict | None", "description": "Table-frame region instead of a target body (see target_region_gate)."},
        {"name": "min_tilt_deg", "type": "float", "description": "Tilt to reach, deg. Default 45."},
        {"name": "min_time_s", "type": "float", "description": "Contiguous pouring time, s. Default 0.5."},
        {"name": "mouth_local", "type": "list[float] | None", "description": "Mouth point in the body frame, m."},
        {"name": "up_local", "type": "list[float] | None", "description": "Container up axis in the body frame."},
        {"name": "target_margin_m", "type": "float", "description": "Footprint margin around the target, m. Default 0.01."},
        {"name": "at_end", "type": "bool", "description": "The tilted run must reach the window's end. Default false."},
        {"name": "frames", "type": "list[int] | None", "description": "[first, last] window."},
        {"name": "subgoal", "type": "int | None", "description": "Use the frames of this npz subgoal index."},
        {"name": "out_json", "type": "str | None", "description": "Optional report path."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; peak_tilt_deg; end_tilt_deg; required_tilt_deg; longest_pour_s; pour_from_frame; required_s; at_end."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "goal", "pour", "tilt", "task gate"],
    "used_in": ["Open the Bottle"],
    "status": "tested",
    "entrypoint": "tilt_pour_gate",
}

from tool_library.common import task_gates as G


def tilt_pour_gate(scene, trajectory, body, target=None, region=None, min_tilt_deg=45.0, min_time_s=0.5, mouth_local=None,
                   up_local=None, target_margin_m=0.01, at_end=False, frames=None, subgoal=None, out_json=None):
    if target is not None and region is not None:
        raise ValueError("target / region: give one or neither")
    if not 0.0 < float(min_tilt_deg) <= 180.0:
        raise ValueError("min_tilt_deg: in (0, 180]")
    if region is not None:
        G.region_polygon(region)
    ep, t0, t1 = G.open_window(scene, trajectory, frames, subgoal)
    A = ep.bodies(body, "body")
    T = ep.bodies(target, "target") - A if target is not None else None
    rep = G.gate_tilt_pour(ep, A, t0, t1, min_tilt_deg=float(min_tilt_deg), target=T, region=region, mouth_local=mouth_local,
                           up_local=up_local, min_time_s=float(min_time_s), target_margin_m=float(target_margin_m),
                           at_end=bool(at_end), names=(str(body), str(target or "the region")))
    rep["frames"] = [t0, t1]
    return G.finish(rep, out_json)


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(tilt_pour_gate(sys.argv[1], sys.argv[2], sys.argv[3], target=sys.argv[4] if len(sys.argv) > 4 else None), indent=1))
