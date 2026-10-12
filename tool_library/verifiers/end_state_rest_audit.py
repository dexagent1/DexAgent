"""End state rest audit: at the end, is every object at rest and supported by the scene?

    from tool_library.verifiers.end_state_rest_audit import end_state_rest_audit
    end_state_rest_audit("scene.xml", "trajectory.npz")

The success check reads the last frame. If an object is still sliding, rocking on a rim, or
only stays put because a finger still props it, the last frame is not the state the task
leaves behind: one more second of physics, or the hand moving away, and it changes.
episode_physics_check only asks whether something touches each object at the last frame.
"""
from __future__ import annotations

TOOL = {
    "name": "end_state_rest_audit",
    "title": "End state rest audit",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "At the end every object is at rest (still over the last frames), stays put when the scene is simulated "
               "on from the last frame, and is supported by the scene rather than propped by the robot.",
    "description": (
        "episode_physics_check fails an object that touches nothing at the last frame; this audit checks what that "
        "misses. (1) Rest: over the last rest_window_s (default 0.3 s) each free object's body moves slower than "
        "rest_speed_m_s (default 2 cm/s) and turns slower than rest_deg_s (default 15 deg/s). (2) Stable: the scene "
        "as the bench executor runs it is simulated for settle_s (default 1 s) from the last frame with every robot "
        "joint held; each object's peak motion must stay under settle_m (default 3 mm) and settle_deg (default 2 "
        "deg): an object balanced on a rim or sliding down a slope moves. (3) Released: objects the robot still "
        "touches at the last frame (within 1 mm) are simulated again with every robot contact removed; one that "
        "then moves more than settle_m / settle_deg was propped by the hand, unless it is listed in allow_held "
        "(a task that ends holding). Reports per object the final speed, the settle motion and whether the robot "
        "props it."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the episode was executed in."},
        {"name": "trajectory", "type": "str", "description": "trajectory.npz (qpos [T, nq], fps, labels)."},
        {"name": "frames", "type": "[first, last] | None", "description": "Window whose last frame is judged as the end. Default all."},
        {"name": "rest_window_s", "type": "float", "description": "Final stretch over which speed is measured. Default 0.3."},
        {"name": "rest_speed_m_s, rest_deg_s", "type": "float", "description": "Speed limits at rest. Defaults 0.02, 15."},
        {"name": "settle_s", "type": "float", "description": "Simulated time after the last frame. Default 1.0."},
        {"name": "settle_m, settle_deg", "type": "float", "description": "Allowed peak motion while simulating on. Defaults 0.003, 2."},
        {"name": "allow_held", "type": "list[str]", "description": "Objects allowed to end in the hand (fnmatch). Default none."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; worst; objects {name: final speed m/s and deg/s, settle peak mm / deg, touched_by_robot, propped peak mm / deg}; notes (the propped-by-robot rule skipped when the scene without robot contact cannot be built)."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "audit", "final state", "rest", "stability", "release"],
    "status": "tested",
    "entrypoint": "end_state_rest_audit",
}

import numpy as np

from tool_library.common import episode_audit as EA


def end_state_rest_audit(scene, trajectory, frames=None, rest_window_s=0.3, rest_speed_m_s=0.02, rest_deg_s=15.0, settle_s=1.0,
                         settle_m=0.003, settle_deg=2.0, allow_held=()):
    EA.check_positive(rest_window_s=rest_window_s, rest_speed_m_s=rest_speed_m_s, rest_deg_s=rest_deg_s, settle_s=settle_s,
                      settle_m=settle_m, settle_deg=settle_deg)
    ep = EA.Episode(scene, trajectory, frames)
    if not ep.objects:
        return EA.finish([], None, objects={}, note="no free objects")
    held_ok = EA.match_fn(allow_held)
    tE = ep.b
    n_w = max(2, int(round(rest_window_s * ep.fps)))
    a = max(ep.a, tE - n_w)
    issues, rows, worst = [], {}, None
    for n in ep.objects:
        P, q = EA.object_track(ep, n)
        if tE - a >= 1:
            v = float(np.linalg.norm(P[tE] - P[a])) / ((tE - a) / ep.fps)
            w = float(np.degrees(EA.quat_angle(q[a], q[tE]))) / ((tE - a) / ep.fps)
            vmax = float(np.max(np.linalg.norm(np.diff(P[a:tE + 1], axis=0), axis=1))) * ep.fps
        else:
            v = w = vmax = 0.0
        rows[n] = {"final_speed_m_s": round(v, 4), "final_turn_deg_s": round(w, 2), "final_peak_speed_m_s": round(vmax, 4)}
        if v > rest_speed_m_s or w > rest_deg_s:
            issues.append(f"{n} is still moving at the end ({v:.3f} m/s, {w:.1f} deg/s over the last {tE - a} frames, "
                          f"{ep.labels[tE]}): extend the episode until it settles, or place it so it does not slide or rock")
            worst = worst or {"body": n, "kind": "moving", "speed_m_s": round(v, 4), "turn_deg_s": round(w, 2)}
    res, _, bad = EA.settle(ep, ep.Q[tE], seconds=settle_s)
    for n, r in res.items():
        rows[n].update({"settle_peak_mm": EA.r3(1000 * r["peak_moved_m"], 2), "settle_peak_deg": EA.r3(r["peak_rotated_deg"], 2)})
        if bad is not None or r["peak_moved_m"] > settle_m or r["peak_rotated_deg"] > settle_deg:
            issues.append(f"{n} does not stay where the episode leaves it: simulated {settle_s:g} s on from frame {tE} with the robot "
                          f"still it moves {1000 * r['peak_moved_m']:.1f} mm / {r['peak_rotated_deg']:.1f} deg (to "
                          f"{np.round(r['end'], 3).tolist()}): it is balanced, sliding or interpenetrating; place it fully on its "
                          f"support")
            if worst is None or worst.get("kind") != "moving":
                worst = {"body": n, "kind": "unstable", "peak_moved_m": round(r["peak_moved_m"], 4), "peak_rotated_deg": round(r["peak_rotated_deg"], 2)}
    tl = EA.object_contact_timeline(ep, margin=0.001)
    touched = [n for n in ep.objects if any(v <= 0.001 for v in tl[tE][n]["robot"].values())]
    for n in ep.objects:
        rows[n]["touched_by_robot"] = n in touched
    notes = []
    bad2 = None
    if touched:
        try:
            m2 = EA.no_robot_contact_model(ep)
            res2, _, bad2 = EA.settle(ep, ep.Q[tE], seconds=settle_s, model=m2)
        except ValueError as e:
            res2 = {}
            notes.append(f"propped-by-robot rule skipped for {', '.join(touched)}: the scene without robot contact could not "
                         f"be built ({e})")
        for n in touched:
            if n not in res2:
                continue
            r = res2[n]
            rows[n].update({"propped_peak_mm": EA.r3(1000 * r["peak_moved_m"], 2), "propped_peak_deg": EA.r3(r["peak_rotated_deg"], 2)})
            if (r["peak_moved_m"] > settle_m or r["peak_rotated_deg"] > settle_deg) and not held_ok(n):
                sides = sorted(tl[tE][n]["robot"])
                issues.append(f"{n} is propped by the {'/'.join(sides)} hand at the end: without the hand it moves "
                              f"{1000 * r['peak_moved_m']:.1f} mm / {r['peak_rotated_deg']:.1f} deg; set it down on its support and "
                              f"let go before the episode ends (or list it in allow_held if the task ends holding it)")
                worst = worst or {"body": n, "kind": "propped", "peak_moved_m": round(r["peak_moved_m"], 4)}
    if bad is not None:
        issues.append(f"the simulation from the last frame became unstable after {bad} steps")
    if bad2 is not None:
        issues.append(f"the simulation from the last frame without robot contact became unstable after {bad2} steps")
    return EA.finish(issues, worst, objects=rows, end_frame=tE, settle_s=settle_s, notes=notes)


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(end_state_rest_audit(sys.argv[1], sys.argv[2]), indent=1))
