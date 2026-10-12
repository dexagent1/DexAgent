"""Swept collision audit: collisions that happen BETWEEN frames.

    from tool_library.verifiers.swept_collision_audit import swept_collision_audit
    swept_collision_audit("scene.xml", "trajectory.npz")

Every collision check of an episode looks at frames (robot_self_collision_check at every second
frame). A fingertip moving 1 cm per frame passes through a 3 mm lid wall without being inside
it on either frame; an object dropping fast skips through a thin shelf; a forearm sweeps through
the table edge between two samples. Rendered at a higher rate, or replayed by a controller that
interpolates, the robot visibly passes through the part.
"""
from __future__ import annotations

TOOL = {
    "name": "swept_collision_audit",
    "title": "Swept collision audit",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "Check every recorded physics state and interpolated configurations for robot-robot, robot-scene "
               "and object-scene penetration, including tunnelling between samples.",
    "description": (
        "Per-frame checks (robot_self_collision_check, episode_physics_check, penetration_check) cannot see a part "
        "that is outside an obstacle on frame t and on frame t+1 but passed through it in between. This audit "
        "compiles a copy of the scene in which every visible robot geom collides as its convex hull (as "
        "robot_self_collision_check does) and the objects and the table keep their collision geometry, then for each "
        "pair of consecutive frames bounds how far any colliding body moves (its origin's travel plus its radius "
        "times its rotation). When that exceeds sub_m (default 3 mm, thinner than a lid wall), it inserts evenly "
        "spaced sub-steps (joints interpolated linearly, free-body and ball rotations by slerp; at most max_sub) and "
        "runs collision detection on each. Absolute penetration deeper than tol_m (default 3 mm) fails, including "
        "on the recorded states themselves. Incomplete subdivision coverage fails. Only mechanically related "
        "robot bodies, redundant hand proxies, static-static pairs, and parts within one object are excluded. "
        "Reports each event's frames, sub-step fraction, geoms, bodies and depth."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the episode was executed in (a path: a collidable copy is compiled from it)."},
        {"name": "trajectory", "type": "str", "description": "trajectory.npz (qpos [T, nq], fps, labels)."},
        {"name": "frames", "type": "[first, last] | None", "description": "Inclusive frame window. Default all."},
        {"name": "sub_m", "type": "float", "description": "Largest motion between checked configurations. Default 0.003."},
        {"name": "tol_m", "type": "float", "description": "Maximum absolute penetration. Default 0.003."},
        {"name": "max_sub", "type": "int", "description": "Most sub-steps per frame pair. Default 12."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; worst {frames, fraction, geoms, bodies, kind, depth_mm, frame_depth_mm}; events (up to 20); frame_pairs_subdivided, substeps."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "audit", "collision", "tunnelling", "continuous", "thin parts"],
    "status": "tested",
    "entrypoint": "swept_collision_audit",
}

import mujoco
import numpy as np

from tool_library.common import episode_audit as EA


def swept_collision_audit(scene, trajectory, frames=None, sub_m=0.003, tol_m=0.003, max_sub=12):
    EA.check_positive(sub_m=sub_m, tol_m=tol_m, max_sub=max_sub)
    from tool_library.verifiers._recording import collision_recording
    ep = EA.Episode(scene, collision_recording(trajectory) if frames is None else trajectory, frames)
    m = EA.visual_robot_model(ep)
    if m.nq != ep.model.nq:
        raise ValueError("scene: the collidable copy changed the joint layout")
    d = mujoco.MjData(m)
    kind = {b: ep.kind(b) for b in range(m.nbody)}
    owner = {}
    for n, o in ep.objects.items():
        for b in o["bodies"]:
            owner[b] = n
    collide = [g for g in range(m.ngeom) if int(m.geom_contype[g]) or int(m.geom_conaffinity[g])]
    radius = np.zeros(m.nbody)
    moving = set()
    for g in collide:
        b = int(m.geom_bodyid[g])
        radius[b] = max(radius[b], float(np.linalg.norm(m.geom_pos[g])) + float(m.geom_rbound[g]))
        if kind[b] != "static":
            moving.add(b)
    moving = sorted(moving)
    from tool_library.verifiers.robot_self_collision_check import related_robot_bodies, redundant_robot_proxy_contact
    coverage_issues = []

    def pairs_at(q):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        out = {}
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist >= 0:
                continue
            g1, g2 = int(c.geom1), int(c.geom2)
            b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
            k1, k2 = kind[b1], kind[b2]
            if k1 == "static" and k2 == "static":
                continue
            if k1 == k2 == "robot" and related_robot_bodies(m, b1, b2):
                continue
            if k1 == k2 == "robot" and redundant_robot_proxy_contact(m, g1, g2):
                continue
            if owner.get(b1) is not None and owner.get(b1) == owner.get(b2):
                continue
            key = (min(g1, g2), max(g1, g2))
            out[key] = max(out.get(key, 0.0), -float(c.dist))
        return out

    xp, xq = ep.kin()
    events, n_sub_pairs, n_steps = [], 0, 0
    def record(cur, t, fraction, baseline=None):
        for key, depth in cur.items():
            if depth <= tol_m:
                continue
            g1, g2 = key
            b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
            events.append({"frames": [t, min(t + 1, ep.b)], "fraction": round(fraction, 3), "label": ep.labels[t],
                           "geoms": [m.geom(g1).name or f"geom{g1}", m.geom(g2).name or f"geom{g2}"],
                           "bodies": [ep.body_name(b1), ep.body_name(b2)],
                           "kind": "-".join(sorted([kind[b1], kind[b2]])),
                           "depth_mm": round(1000 * depth, 1),
                           "frame_depth_mm": round(1000 * (baseline or cur).get(key, 0.), 1)})
    prev = pairs_at(ep.Q[ep.a])
    record(prev, ep.a, 0.)
    for t in range(ep.a, ep.b):
        nxt = pairs_at(ep.Q[t + 1])
        record(nxt, t + 1, 0.)
        if moving:
            dp = np.linalg.norm(xp[t + 1, moving] - xp[t, moving], axis=1)
            da = EA.quat_angle(xq[t, moving], xq[t + 1, moving])
            reach = float(np.max(dp + radius[moving] * da))
        else:
            reach = 0.0
        needed = max(0, int(np.ceil(reach / sub_m)) - 1)
        n = min(int(max_sub), needed)
        if needed > int(max_sub):
            coverage_issues.append(f"frames {t}-{t + 1}: motion needs {needed} collision samples, exceeds max_sub={max_sub}; slow or subdivide the trajectory")
        if n >= 1:
            n_sub_pairs += 1
            for k in range(1, n + 1):
                u = k / (n + 1)
                cur = pairs_at(EA.interp_qpos(m, ep.Q[t], ep.Q[t + 1], u))
                n_steps += 1
                record(cur, t, u, {key: max(prev.get(key, 0.), nxt.get(key, 0.)) for key in cur})
        prev = nxt
    issues = list(coverage_issues[:10])
    seen = {}
    for e in sorted(events, key=lambda e: -e["depth_mm"]):
        key = tuple(sorted(e["bodies"]))
        seen.setdefault(key, []).append(e)
    for key, evs in list(seen.items())[:6]:
        w = evs[0]
        issues.append(f"{w['bodies'][0]} penetrates {w['depth_mm']} mm into {w['bodies'][1]} at/between states {w['frames'][0]} and "
                      f"{w['frames'][1]} ({w['label']}; {w['frame_depth_mm']} mm on the frames themselves; {len(evs)} sub-steps): "
                      f"the motion tunnels through it; route that stretch around the part (a via point clear of it) or slow it "
                      f"so no part moves more than its thickness per frame")
    worst = max(events, key=lambda e: e["depth_mm"]) if events else None
    return EA.finish(issues, worst, events=sorted(events, key=lambda e: -e["depth_mm"])[:20], n_events=len(events),
                     frame_pairs_subdivided=n_sub_pairs, substeps=n_steps, frames=[ep.a, ep.b])


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(swept_collision_audit(sys.argv[1], sys.argv[2]), indent=1))
