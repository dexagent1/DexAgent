"""Contactless motion audit: an object moves while nothing of the robot touches it.

    from tool_library.verifiers.contactless_motion_audit import contactless_motion_audit
    contactless_motion_audit("scene.xml", "trajectory.npz")

In a physics-executed episode an object moves because the robot pushes it, directly or through
another object, or because it falls. An object that travels with no robot contact in that chain
and is not falling was moved by something else: an attach that holds it a centimetre off the
fingers (an air grasp), a ghost carry, an edited pose, an object sliding away on its own.
"""
from __future__ import annotations

TOOL = {
    "name": "contactless_motion_audit",
    "title": "Contactless motion audit",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "An object travels or turns beyond settling while no robot part touches it, directly or through a chain "
               "of touching objects, and it is not in free fall: air grasps and ghost carries.",
    "description": (
        "No existing audit relates object motion to robot contact over a whole episode (contact_force_check judges one "
        "hold window given by hand). On the scene as the bench executor runs it (hand-object pairs), each frame of "
        "each free object is classified as robot-connected when the robot touches it within margin_m (default 3 mm) "
        "or touches an object that is, recursively, or an engaged object-object constraint links it to one (a toy "
        "in a drawer the hand pulls, a cap on a bottle the hand carries). A constraint between the robot and the "
        "object does NOT connect: holding must be by contact. Frames in free fall (no contact at all, acceleration "
        "within 3 m/s^2 of gravity) are excluded; free_fall_plausibility_audit judges those. On the remaining "
        "unconnected frames, per run of consecutive moving frames (speed above 1 cm/s or 5 deg/s), the object body's "
        "net travel and rotation from its pose before the run are measured (a body rocking in place gets nowhere; parts "
        "hinged on it, a lid falling shut, are not the object moving). A run that starts within settle_s (default 1 s) of the last robot "
        "connection is settling (a released object dropping into place, a cylinder rolling to rest) and may travel up "
        "to settle_m / settle_deg (defaults 40 mm, 90 deg); any other run may travel up to move_m / move_deg (defaults "
        "10 mm, 10 deg). Reports the worst run per object with its frames, phase labels, travel and frames since "
        "the robot last touched it."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the episode was executed in."},
        {"name": "trajectory", "type": "str", "description": "trajectory.npz (qpos [T, nq], fps, labels, optional eq_active)."},
        {"name": "frames", "type": "[first, last] | None", "description": "Inclusive frame window. Default all."},
        {"name": "margin_m", "type": "float", "description": "Contact distance that counts as touching. Default 0.003."},
        {"name": "settle_s", "type": "float", "description": "Time after the last robot contact during which motion is settling. Default 1.0."},
        {"name": "settle_m, settle_deg", "type": "float", "description": "Allowed net travel / rotation of a settling run. Defaults 0.04, 90."},
        {"name": "move_m, move_deg", "type": "float", "description": "Allowed net travel / rotation of any other contactless run. Defaults 0.01, 10."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; worst {body, frames, labels, travel_mm, rotation_deg, since_robot_frames, settling}; objects {name: worst run}."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "audit", "physics", "contact", "air grasp", "ghost carry"],
    "status": "tested",
    "entrypoint": "contactless_motion_audit",
}

import numpy as np

from tool_library.common import episode_audit as EA


def contactless_motion_audit(scene, trajectory, frames=None, margin_m=0.003, settle_s=1.0, settle_m=0.04, settle_deg=90.0,
                             move_m=0.01, move_deg=10.0):
    EA.check_positive(margin_m=margin_m, settle_s=settle_s, settle_m=settle_m, settle_deg=settle_deg, move_m=move_m,
                      move_deg=move_deg)
    ep = EA.Episode(scene, trajectory, frames)
    if not ep.objects:
        return EA.finish([], None, objects={}, note="no free objects")
    T, fps, g = ep.T, ep.fps, ep.gravity
    tl = EA.object_contact_timeline(ep, margin=margin_m)
    links = {n: [x for x in v if x[1] != "robot"] for n, v in EA.object_links(ep).items()}
    settle_n = int(round(settle_s * fps))
    runs_all, objs = [], {}
    for n, o in ep.objects.items():
        P, q = EA.object_track(ep, n)               # the object's own body: a lid swinging on it is not the object moving
        conn = np.array([EA.robot_connected(tl, t, n, links) for t in range(T)])
        step = np.r_[0.0, np.linalg.norm(np.diff(P, axis=0), axis=1)] if T > 1 else np.zeros(T)     # step into frame t
        rot = np.r_[0.0, np.degrees(EA.quat_angle(q[:-1], q[1:]))] if T > 1 else np.zeros(T)
        acc = np.zeros((T, 3))
        if T >= 3:
            acc[1:-1] = EA.central_accel(P, fps)
        freefall = np.array([not EA.any_contact(tl[t][n]) and np.linalg.norm(acc[t] - g) < 3.0 for t in range(T)])
        moving = (step * fps > 0.01) | (rot * fps > 5.0)
        mask = moving & ~conn & ~freefall
        mask[:ep.a] = False
        mask[ep.b + 1:] = False
        last_conn = -10 ** 9
        lc = np.full(T, -10 ** 9)
        for t in range(T):
            if conn[t]:
                last_conn = t
            lc[t] = last_conn
        best = None
        for a, b in EA.runs_of(mask):
            # net motion from the pose before the run (a body vibrating in place travels far and gets nowhere)
            p0, q0 = P[max(a - 1, 0)], q[max(a - 1, 0)]
            travel = float(np.max(np.linalg.norm(P[a:b + 1] - p0, axis=1)))
            turn = float(np.degrees(np.max(EA.quat_angle(np.repeat(q0[None], b - a + 1, 0), q[a:b + 1]))))
            since = a - lc[a]
            settling = since <= settle_n
            lim_m, lim_d = (settle_m, settle_deg) if settling else (move_m, move_deg)
            r = {"body": n, "frames": [a, b], "labels": [ep.labels[a], ep.labels[b]], "travel_mm": round(1000 * travel, 1),
                 "rotation_deg": round(turn, 1), "since_robot_frames": int(since) if lc[a] >= 0 else None, "settling": bool(settling),
                 "limit_mm": 1000 * lim_m, "limit_deg": lim_d, "fail": bool(travel > lim_m or turn > lim_d)}
            if best is None or travel / lim_m + turn / lim_d > best["travel_mm"] / best["limit_mm"] + best["rotation_deg"] / best["limit_deg"]:
                best = r
            if r["fail"]:
                runs_all.append(r)
        objs[n] = best or {"travel_mm": 0.0}
        objs[n]["robot_connected_frames"] = int(conn.sum())
    issues = []
    for n in sorted({r["body"] for r in runs_all}):
        rs = [r for r in runs_all if r["body"] == n]
        w = max(rs, key=lambda r: r["travel_mm"] / r["limit_mm"] + r["rotation_deg"] / r["limit_deg"])
        since = (f"{w['since_robot_frames']} frames after the robot last touched it" if w["since_robot_frames"] is not None
                 else "and the robot never touched it")
        issues.append(f"{n} moves {w['travel_mm']} mm / {w['rotation_deg']} deg on frames {w['frames'][0]}-{w['frames'][1]} "
                      f"({w['labels'][0]}) with no robot contact, {since} (limit {w['limit_mm']:.0f} mm / {w['limit_deg']:.0f} deg"
                      f"{' while settling' if w['settling'] else ''}; {len(rs)} such runs): an air grasp or ghost carry; close the "
                      f"fingers onto the object (contact, not a constraint) or find what else moves it")
    worst = max(runs_all, key=lambda r: r["travel_mm"] / r["limit_mm"] + r["rotation_deg"] / r["limit_deg"]) if runs_all else None
    return EA.finish(issues, worst, objects=objs, n_runs=len(runs_all), frames=[ep.a, ep.b])


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(contactless_motion_audit(sys.argv[1], sys.argv[2]), indent=1))
