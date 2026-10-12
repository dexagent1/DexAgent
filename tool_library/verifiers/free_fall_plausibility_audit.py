"""Free fall plausibility audit: an object that touches nothing must fall at g.

    from tool_library.verifiers.free_fall_plausibility_audit import free_fall_plausibility_audit
    free_fall_plausibility_audit("scene.xml", "trajectory.npz")

An object that is not in contact with anything (and not held by a constraint) has exactly one
force on it: gravity. If, on such frames, it hovers, sinks slowly or drifts sideways, something
the episode does not show is holding it: an air grasp, a kinematic attach, an edited pose.
episode_physics_check only asks this of the LAST frame (nothing left floating at the end).
"""
from __future__ import annotations

TOOL = {
    "name": "free_fall_plausibility_audit",
    "title": "Free fall plausibility audit",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "On every frame where a free object touches nothing and no constraint holds it, its acceleration must be "
               "gravity's: hovering, falling too slowly or drifting sideways fails.",
    "description": (
        "episode_physics_check checks only the final frame for floating objects; this audit checks every frame. For "
        "each free object it marks the frames where the object's subtree has no contact within margin_m (default "
        "3 mm, plus the distance it travels in a frame) on that frame and both neighbours, on the scene as the bench "
        "executor runs it (hand-object pairs included), and no equality constraint holds it to another object or the "
        "world (constraints are taken as engaged from a recorded eq_active, else when their residual is under 3 mm / 3 "
        "deg). On those free frames the centre-of-mass acceleration (central second difference) must equal gravity "
        "within acc_tol (default 3 m/s^2). Runs of at least min_frames such frames that fail are reported as "
        "hovering (speed under 2 cm/s), falling too slowly, or accelerating sideways, with their frames, phase labels "
        "and mean acceleration. A constraint from the robot to the object is not support: the hand must hold by contact."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the episode was executed in."},
        {"name": "trajectory", "type": "str", "description": "trajectory.npz (qpos [T, nq], fps, labels, optional eq_active)."},
        {"name": "frames", "type": "[first, last] | None", "description": "Inclusive frame window. Default all."},
        {"name": "margin_m", "type": "float", "description": "Contact distance under which the object counts as touching. Default 0.003."},
        {"name": "acc_tol", "type": "float", "description": "Allowed |a - g| on a free frame, m/s^2. Default 3."},
        {"name": "min_frames", "type": "int", "description": "Shortest failing run reported. Default 2."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; worst {body, frames, labels, kind, mean_acc}; objects {name: free frames, worst |a - g|}."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "audit", "physics", "gravity", "hover", "air grasp"],
    "status": "tested",
    "entrypoint": "free_fall_plausibility_audit",
}

import numpy as np

from tool_library.common import episode_audit as EA


def free_fall_plausibility_audit(scene, trajectory, frames=None, margin_m=0.003, acc_tol=3.0, min_frames=2):
    EA.check_positive(margin_m=margin_m, acc_tol=acc_tol, min_frames=min_frames)
    ep = EA.Episode(scene, trajectory, frames)
    if not ep.objects:
        return EA.finish([], None, objects={}, note="no free objects")
    g = ep.gravity
    T = ep.T
    coms = {n: ep.com_track(o["bodies"]) for n, o in ep.objects.items()}
    # contacts are read with a 2 cm reach; a frame counts as touching within margin_m plus the object's own travel
    # over that frame (a fast object meets its contact between two frames)
    tl = EA.object_contact_timeline(ep, margin=margin_m + 0.02)
    links = EA.object_links(ep)
    events, objs = [], {}
    for n, o in ep.objects.items():
        if o["mass"] <= 0:
            continue
        P = coms[n]
        acc = np.zeros((T, 3))
        if T >= 3:
            acc[1:-1] = EA.central_accel(P, ep.fps)
        vel = np.zeros(T)
        if T >= 2:
            vel[1:] = np.linalg.norm(np.diff(P, axis=0), axis=1) * ep.fps
        held = np.zeros(T, bool)
        for mask, partner, _ in links.get(n, []):
            if partner != "robot":
                held |= np.asarray(mask, bool)
        step = np.r_[0.0, np.linalg.norm(np.diff(P, axis=0), axis=1)] if T > 1 else np.zeros(T)
        free = np.zeros(T, bool)
        for t in range(max(ep.a, 1), min(ep.b, T - 2) + 1):
            tol = margin_m + min(0.02, float(max(step[t], step[t + 1])))
            if any(EA.contact_within(tl[k][n], tol) or held[k] for k in (t - 1, t, t + 1)):
                continue
            free[t] = True
        err = np.linalg.norm(acc - g, axis=1)
        bad = free & (err > acc_tol)
        objs[n] = {"free_frames": int(free.sum()), "worst_acc_error": round(float(err[free].max()), 2) if free.any() else 0.0}
        for a, b in EA.runs_of(bad):
            if b - a + 1 < min_frames:
                continue
            ma = acc[a:b + 1].mean(axis=0)
            v = float(vel[a:b + 1].mean())
            if v < 0.02:
                kind = "hovers"
            elif abs(ma[2] - g[2]) > acc_tol and np.linalg.norm(ma[:2]) <= acc_tol:
                kind = "falls too slowly" if ma[2] > g[2] else "falls faster than gravity"
            else:
                kind = "accelerates sideways"
            events.append({"body": n, "frames": [a, b], "labels": [ep.labels[a], ep.labels[b]], "kind": kind,
                           "mean_acc": [round(float(x), 2) for x in ma], "mean_speed_m_s": round(v, 3), "n": b - a + 1,
                           "acc_error": round(float(err[a:b + 1].mean()), 2)})
    issues = []
    for n in sorted({e["body"] for e in events}):
        evs = [e for e in events if e["body"] == n]
        w = max(evs, key=lambda e: e["n"] * e["acc_error"])
        tot = sum(e["n"] for e in evs)
        issues.append(f"{n} touches nothing on frames {w['frames'][0]}-{w['frames'][1]} ({w['labels'][0]}) yet {w['kind']} "
                      f"(mean acceleration {w['mean_acc']} m/s^2 instead of gravity; {tot} such frames in {len(evs)} runs): "
                      f"something invisible holds it; make the hand hold it by contact (close the grasp onto it) or let it "
                      f"fall and settle")
    worst = max(events, key=lambda e: e["n"] * e["acc_error"]) if events else None
    return EA.finish(issues, worst, objects=objs, n_runs=len(events), frames=[ep.a, ep.b])


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(free_fall_plausibility_audit(sys.argv[1], sys.argv[2]), indent=1))
