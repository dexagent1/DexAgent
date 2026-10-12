"""Object spin rate audit: an object that keeps spinning or flying fast over several frames.

    from tool_library.verifiers.object_spin_rate_audit import object_spin_rate_audit
    object_spin_rate_audit("scene.xml", "trajectory.npz")

A flung object, or one the solver is pumping energy into (an exploding contact), turns or
travels fast for many consecutive frames. Single-frame jumps are object_teleport_audit's and
the 4 m/s single-frame peak is episode_physics_check's; this audit asks for SUSTAINED speed,
which a tabletop manipulation never needs.
"""
from __future__ import annotations

TOOL = {
    "name": "object_spin_rate_audit",
    "title": "Object spin rate audit",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "Sustained object angular speed (and linear speed) over several consecutive frames: flung, tumbling or "
               "numerically exploding objects.",
    "description": (
        "Distinct from object_teleport_audit (single-frame jumps) and episode_physics_check (a single-frame peak "
        "speed of 4 m/s). For every free object outside the robot it computes the angular speed between consecutive "
        "frames (deg/s, from the body's world orientation) and the linear speed of its origin, and takes the lowest "
        "value inside every run of min_frames consecutive frames (at 30 fps; scaled to the frame rate): that is the "
        "speed the object kept for the whole run. It fails when the sustained angular speed reaches max_deg_s "
        "(default 720 deg/s, two turns per second, far above any in-hand turn or a cap unscrewed by a robot hand) or "
        "the sustained linear speed reaches max_m_s (default 1.5 m/s). Mechanism joints outside the robot (lids, "
        "drawers, caps on threads) get the same rule on their own speed. Reports the worst run per object with its "
        "frames and phase labels."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the episode was executed in."},
        {"name": "trajectory", "type": "str", "description": "trajectory.npz (qpos [T, nq], fps, labels)."},
        {"name": "frames", "type": "[first, last] | None", "description": "Inclusive frame window. Default all."},
        {"name": "max_deg_s", "type": "float", "description": "Sustained angular speed that fails. Default 720."},
        {"name": "max_m_s", "type": "float", "description": "Sustained linear speed that fails. Default 1.5."},
        {"name": "min_frames", "type": "int", "description": "Run length (at 30 fps) over which the speed must be kept. Default 6 (0.2 s)."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; worst {body, kind, frames, labels, sustained}; objects {name: sustained deg/s and m/s, peak deg/s}; joints {name: sustained speed}."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "audit", "spin", "flung", "explosion", "physics"],
    "status": "tested",
    "entrypoint": "object_spin_rate_audit",
}

import numpy as np

from tool_library.common import episode_audit as EA


def _sustained(v, n):
    """(value, start) of the largest minimum over runs of n consecutive entries."""
    if len(v) < n:
        return 0.0, 0
    win = np.lib.stride_tricks.sliding_window_view(v, n).min(axis=1)
    k = int(np.argmax(win))
    return float(win[k]), k


def object_spin_rate_audit(scene, trajectory, frames=None, max_deg_s=720.0, max_m_s=1.5, min_frames=6):
    EA.check_positive(max_deg_s=max_deg_s, max_m_s=max_m_s, min_frames=min_frames)
    ep = EA.Episode(scene, trajectory, frames)
    n = max(2, int(round(min_frames * ep.fps / 30.0)))
    a, b = ep.a, ep.b
    objs, joints, events = {}, {}, []
    for name in ep.objects:
        P, q = EA.object_track(ep, name)
        w = np.degrees(EA.quat_angle(q[a:b], q[a + 1:b + 1])) * ep.fps
        v = np.linalg.norm(np.diff(P[a:b + 1], axis=0), axis=1) * ep.fps
        sw, kw = _sustained(w, n)
        sv, kv = _sustained(v, n)
        objs[name] = {"sustained_deg_s": round(sw, 1), "sustained_m_s": round(sv, 3),
                      "peak_deg_s": round(float(w.max()) if len(w) else 0.0, 1)}
        if sw >= max_deg_s:
            events.append({"body": name, "kind": "spin", "frames": [a + kw, a + kw + n], "sustained": round(sw, 1), "unit": "deg/s",
                           "labels": [ep.labels[a + kw], ep.labels[min(a + kw + n, ep.T - 1)]]})
        if sv >= max_m_s:
            events.append({"body": name, "kind": "flight", "frames": [a + kv, a + kv + n], "sustained": round(sv, 3), "unit": "m/s",
                           "labels": [ep.labels[a + kv], ep.labels[min(a + kv + n, ep.T - 1)]]})
    for jn, (j, qadr, jt) in ep.mech_joints.items():
        x = ep.Q[a:b + 1, qadr]
        hinge = jt == EA.HINGE
        sp = np.abs(np.diff(x)) * ep.fps * (np.degrees(1.0) if hinge else 1.0)
        s_, k_ = _sustained(sp, n)
        lim = max_deg_s if hinge else max_m_s
        joints[jn] = {"sustained": round(s_, 3), "unit": "deg/s" if hinge else "m/s"}
        if s_ >= lim:
            events.append({"body": jn, "kind": "joint_spin" if hinge else "joint_flight", "frames": [a + k_, a + k_ + n],
                           "sustained": round(s_, 3), "unit": "deg/s" if hinge else "m/s",
                           "labels": [ep.labels[a + k_], ep.labels[min(a + k_ + n, ep.T - 1)]]})
    issues = [f"{e['body']} kept {e['sustained']} {e['unit']} for {n} frames from frame {e['frames'][0]} ({e['labels'][0]}): "
              f"{'flung or exploding' if e['kind'] in ('spin', 'flight') else 'a mechanism joint thrown'}; find the contact that "
              f"launched it (a finger driven into it, a release at speed, a stiff constraint) and re-execute"
              for e in events]
    worst = max(events, key=lambda e: e["sustained"] / (max_deg_s if e["unit"] == "deg/s" else max_m_s)) if events else None
    return EA.finish(issues, worst, objects=objs, joints=joints, run_frames=n, frames=[ep.a, ep.b])


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(object_spin_rate_audit(sys.argv[1], sys.argv[2]), indent=1))
