"""Object teleport audit: an object (or a mechanism joint) whose pose jumps in a way no motion explains.

    from tool_library.verifiers.object_teleport_audit import object_teleport_audit
    object_teleport_audit("scene.xml", "trajectory.npz")

object_disturbance_check and episode_physics_check already fail any free body that
translates more than 3 cm in one frame. What they cannot see is a glitch below that
step (one frame displaced and the next back), an orientation that flips between two
frames, an object snapping back to its start pose (a reset), a drawer or lid joint that
jumps, or a quaternion that is no longer unit length (an edited trajectory).
"""
from __future__ import annotations

TOOL = {
    "name": "object_teleport_audit",
    "title": "Object teleport audit",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "Per-frame object pose jumps the 3 cm step rule misses: one-frame glitches, orientation flips, snaps back "
               "to the start pose, jumping mechanism joints and non-unit quaternions.",
    "description": (
        "Covers what object_disturbance_check and episode_physics_check (translation step above 3 cm per frame) "
        "miss. For every free object outside the robot it measures, frame by frame: (1) the glitch offset, how far "
        "frame t sits off the midpoint of frames t-1 and t+1, in position and in orientation (slerp midpoint); a "
        "frame that sits farther off than glitch_m / glitch_deg AND more than glitch_ratio times the half-distance "
        "between its neighbours was displaced and put back, which neither a fall (the neighbours are far apart) nor "
        "an impact (the object does not come back) produces; (2) a single-frame rotation above rot_step_deg (at 30 "
        "fps) that is more than three times the rotation of both neighbouring frames, an orientation flip; (3) a "
        "snap back to the frame-0 pose (within reset_tol_m and 1 degree) from more than reset_from_m away in one "
        "frame, a reset; (4) a non-unit free-joint quaternion (an edited state); (5) non-finite values. Every hinge "
        "or slide joint outside the robot (drawers, lids, caps on threads) gets the glitch rule and a per-frame "
        "step limit (joint_step_deg / joint_step_m at 30 fps). Thresholds scale with the frame rate where they "
        "are speeds. Reports the worst event with its frame, phase label and body, and per-object maxima."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the episode was executed in."},
        {"name": "trajectory", "type": "str", "description": "trajectory.npz (qpos [T, nq], fps, labels)."},
        {"name": "frames", "type": "[first, last] | None", "description": "Inclusive frame window to judge. Default all."},
        {"name": "glitch_m, glitch_deg", "type": "float", "description": "Smallest glitch offset that counts. Defaults 0.005 m, 20 deg (a released cylinder rattling in its holder swings about 12 deg frame to frame)."},
        {"name": "glitch_ratio", "type": "float", "description": "Offset / neighbours' half-distance above which a frame is a glitch. Default 2."},
        {"name": "rot_step_deg", "type": "float", "description": "Single-frame rotation (at 30 fps) above which an isolated turn is a flip. Default 60."},
        {"name": "reset_tol_m, reset_from_m", "type": "float", "description": "A snap back to the start pose: within 1 mm of it, from 20 mm away. Defaults 0.001, 0.02."},
        {"name": "joint_step_deg, joint_step_m", "type": "float", "description": "Per-frame step limit of mechanism hinges / slides at 30 fps. Defaults 45 deg, 0.03 m."},
        {"name": "quat_norm_tol", "type": "float", "description": "Allowed |norm - 1| of a free-joint quaternion. Default 1e-3."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; worst {frame, label, body, kind, value}; objects {name: max glitch mm/deg, max rotation step deg}; joints {name: max step, max glitch}; events (up to 20)."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "audit", "teleport", "continuity", "reset", "glitch"],
    "status": "tested",
    "entrypoint": "object_teleport_audit",
}

import numpy as np

from tool_library.common import episode_audit as EA


def _slerp_mid(qa, qb):
    if np.dot(qa, qb) < 0:
        qb = -qb
    q = qa + qb
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else qa


def object_teleport_audit(scene, trajectory, frames=None, glitch_m=0.005, glitch_deg=20.0, glitch_ratio=2.0,
                          rot_step_deg=60.0, reset_tol_m=0.001, reset_from_m=0.02, joint_step_deg=45.0,
                          joint_step_m=0.03, quat_norm_tol=1e-3):
    EA.check_positive(glitch_m=glitch_m, glitch_deg=glitch_deg, glitch_ratio=glitch_ratio, rot_step_deg=rot_step_deg,
                      reset_tol_m=reset_tol_m, reset_from_m=reset_from_m, joint_step_deg=joint_step_deg,
                      joint_step_m=joint_step_m, quat_norm_tol=quat_norm_tol)
    ep = EA.Episode(scene, trajectory, frames)
    Q, fps = ep.Q, ep.fps
    sc = 30.0 / fps
    a, b = ep.a, ep.b
    events, objs, joints = [], {}, {}

    def add(t, body, kind, value, unit, why):
        events.append({"frame": int(t), "label": ep.labels[int(t)], "body": body, "kind": kind,
                       "value": EA.r3(value, 2), "unit": unit, "why": why})

    for name, o in ep.objects.items():
        qa = o["qadr"]
        X = Q[:, qa:qa + 7]
        q = X[:, 3:7]
        rec = {"max_glitch_mm": 0.0, "max_glitch_deg": 0.0, "max_rot_step_deg": 0.0}
        bad = ~np.isfinite(X).all(axis=1)
        if bad[a:b + 1].any():
            t = a + int(np.argmax(bad[a:b + 1]))
            add(t, name, "non_finite", float("nan"), "", "the object's state is not finite")
            objs[name] = rec
            continue
        nrm = np.abs(np.linalg.norm(q, axis=1) - 1.0)
        if nrm[a:b + 1].max() > quat_norm_tol:
            t = a + int(np.argmax(nrm[a:b + 1]))
            add(t, name, "quat_norm", nrm[t], "", "the free-joint quaternion is not unit length (an edited state)")
        P, qn = EA.object_track(ep, name)                               # world pose of the body
        rs = np.degrees(EA.quat_angle(qn[:-1], qn[1:]))                     # rotation of step t -> t+1
        for t in range(max(a, 1), min(b, ep.T - 2) + 1):
            mid = 0.5 * (P[t - 1] + P[t + 1])
            dev = np.linalg.norm(P[t] - mid)
            half = 0.5 * np.linalg.norm(P[t + 1] - P[t - 1])
            if dev > rec["max_glitch_mm"] / 1000:
                rec["max_glitch_mm"] = round(1000 * dev, 2)
            if dev >= glitch_m and dev >= glitch_ratio * half:
                add(t, name, "glitch", 1000 * dev, "mm", f"frame sits {1000 * dev:.1f} mm off its neighbours, which are "
                    f"{2000 * half:.1f} mm apart: displaced for one frame and put back")
            qm = _slerp_mid(qn[t - 1], qn[t + 1])
            adev = float(np.degrees(EA.quat_angle(qn[t], qm)))
            ahalf = 0.5 * float(np.degrees(EA.quat_angle(qn[t - 1], qn[t + 1])))
            rec["max_glitch_deg"] = max(rec["max_glitch_deg"], round(adev, 2))
            if adev >= glitch_deg and adev >= glitch_ratio * ahalf:
                add(t, name, "rotation_glitch", adev, "deg", f"orientation sits {adev:.0f} deg off its neighbours "
                    f"({2 * ahalf:.0f} deg apart): turned for one frame and turned back")
        for t in range(a, min(b, ep.T - 2) + 1):
            st = rs[t]
            rec["max_rot_step_deg"] = max(rec["max_rot_step_deg"], round(float(st), 2))
            nb = max(rs[t - 1] if t >= 1 else 0.0, rs[t + 1] if t + 1 < len(rs) else 0.0)
            if st >= rot_step_deg * sc and st >= 3.0 * nb:
                add(t + 1, name, "flip", st, "deg", f"turned {st:.0f} deg in one frame while the frames around turned "
                    f"{nb:.0f} deg: an orientation flip")
        d0 = np.linalg.norm(P - P[0], axis=1)
        a0 = np.degrees(EA.quat_angle(np.repeat(qn[:1], ep.T, 0), qn))
        for t in range(max(a, 1), b + 1):
            if d0[t] <= reset_tol_m and a0[t] <= 1.0 and d0[t - 1] >= reset_from_m:
                add(t, name, "reset", 1000 * d0[t - 1], "mm", f"snapped back to its frame-0 pose from {1000 * d0[t - 1]:.0f} mm "
                    "away in one frame (a reset)")
        objs[name] = rec

    for jn, (j, qadr, jt) in ep.mech_joints.items():
        x = Q[:, qadr]
        hinge = jt == EA.HINGE
        k = np.degrees(1.0) if hinge else 1000.0
        unit = "deg" if hinge else "mm"
        lim = (joint_step_deg if hinge else 1000 * joint_step_m) * sc
        gl = glitch_deg if hinge else 1000 * glitch_m
        st = np.abs(np.diff(x)) * k
        rec = {"max_step": 0.0, "max_glitch": 0.0, "unit": unit}
        for t in range(a, min(b, ep.T - 2) + 1):
            rec["max_step"] = max(rec["max_step"], round(float(st[t]), 2))
            if st[t] > lim:
                add(t + 1, jn, "joint_step", st[t], unit, f"mechanism joint moved {st[t]:.1f} {unit} in one frame "
                    f"(limit {lim:.1f} {unit} per frame)")
        for t in range(max(a, 1), min(b, ep.T - 2) + 1):
            dev = abs(x[t] - 0.5 * (x[t - 1] + x[t + 1])) * k
            half = 0.5 * abs(x[t + 1] - x[t - 1]) * k
            rec["max_glitch"] = max(rec["max_glitch"], round(float(dev), 2))
            if dev >= gl and dev >= glitch_ratio * half:
                add(t, jn, "joint_glitch", dev, unit, f"mechanism joint sits {dev:.1f} {unit} off its neighbours for one frame")
        joints[jn] = rec

    issues = []
    order = {"non_finite": 0, "reset": 1, "flip": 2, "glitch": 3, "rotation_glitch": 4, "joint_step": 5, "joint_glitch": 6, "quat_norm": 7}
    events.sort(key=lambda e: (order.get(e["kind"], 9), -(e["value"] or 0)))
    seen = set()
    for e in events:
        key = (e["body"], e["kind"])
        if key in seen:
            continue
        seen.add(key)
        n_same = sum(1 for x in events if (x["body"], x["kind"]) == key)
        issues.append(f"{e['body']} at frame {e['frame']} ({e['label']}): {e['why']}"
                      + (f" [{n_same} such frames]" if n_same > 1 else "")
                      + "; re-execute that segment from the last good state instead of editing poses, and check the "
                        "executor did not reset or re-seat the object")
    worst = events[0] if events else None
    return EA.finish(issues, worst, objects=objs, joints=joints, events=events[:20], n_events=len(events),
                     frames=[ep.a, ep.b], fps=fps)


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(object_teleport_audit(sys.argv[1], sys.argv[2]), indent=1))
