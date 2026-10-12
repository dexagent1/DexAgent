"""BADQACC reset audit: did the simulation silently reset (or freeze) somewhere in the episode?

    from tool_library.verifiers.badqacc_reset_audit import badqacc_reset_audit
    badqacc_reset_audit("scene.xml", "trajectory.npz")

When MuJoCo's constraint solver produces a bad acceleration it prints a warning and resets the
state to qpos0. A recorded episode then shows many joints snapping back to qpos0 (or to the
keyframe the executor restarts from) in the same frame. The Stage 3 executor guards against it
by ending the segment and copying the last good state forward, which leaves a frozen stretch
where the plan moved on and the state did not.
"""
from __future__ import annotations

TOOL = {
    "name": "badqacc_reset_audit",
    "title": "BADQACC reset audit",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "Frames where many degrees of freedom snap back to qpos0 or a keyframe at once (MuJoCo's silent reset after "
               "a bad acceleration), and frozen stretches where the whole state stops changing across phases.",
    "description": (
        "No other audit looks for this. For every frame t it compares the state with qpos0 and with every keyframe: a "
        "qpos entry 'snaps' when it was more than from_tol away from the reference on frame t-1 and within eps of "
        "it on frame t. A frame where at least min_dofs entries snap and they are at least min_frac of the entries "
        "that were away, and they jump at least five times farther than they moved the frame before, is a reset (a "
        "physics run never lands thirty joints exactly on their initial values by chance, and an interpolation that "
        "ends on qpos0 arrives with an ordinary step). It also measures runs of frames whose entire state is bit-identical to the previous frame: a "
        "physics-executed episode never repeats exactly, so a frozen run that crosses a phase boundary (the plan "
        "moved on, the state did not) or lasts longer than freeze_max_s is the executor's guard copying the last good "
        "frame forward after an instability. Reports each reset frame with the reference it snapped to and how many "
        "entries snapped (robot joints and objects separately), and the frozen runs."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the episode was executed in (qpos0 and keyframes come from it)."},
        {"name": "trajectory", "type": "str", "description": "trajectory.npz (qpos [T, nq], fps, labels)."},
        {"name": "frames", "type": "[first, last] | None", "description": "Inclusive frame window. Default all."},
        {"name": "min_dofs", "type": "int", "description": "Fewest snapped qpos entries that make a reset. Default 6."},
        {"name": "min_frac", "type": "float", "description": "Fewest snapped entries as a share of those that were away from the reference. Default 0.3."},
        {"name": "eps", "type": "float", "description": "How close to the reference counts as on it. Default 1e-6."},
        {"name": "from_tol", "type": "float", "description": "How far from the reference an entry must have been the frame before. Default 1e-3."},
        {"name": "freeze_min_frames", "type": "int", "description": "Shortest frozen run that counts when it crosses a phase boundary (at 30 fps). Default 3."},
        {"name": "freeze_max_s", "type": "float", "description": "A frozen run longer than this fails anywhere. Default 2.0."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; worst {frame, label, reference, snapped, robot_snapped, object_snapped}; resets (list); frozen_runs (list of {first, last, labels})."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "audit", "instability", "badqacc", "reset", "physics"],
    "status": "tested",
    "entrypoint": "badqacc_reset_audit",
}

import numpy as np

from tool_library.common import episode_audit as EA


def badqacc_reset_audit(scene, trajectory, frames=None, min_dofs=6, min_frac=0.3, eps=1e-6, from_tol=1e-3,
                        freeze_min_frames=3, freeze_max_s=2.0):
    EA.check_positive(min_dofs=min_dofs, min_frac=min_frac, eps=eps, from_tol=from_tol,
                      freeze_min_frames=freeze_min_frames, freeze_max_s=freeze_max_s)
    if min_frac > 1:
        raise ValueError("min_frac must lie in (0, 1]")
    ep = EA.Episode(scene, trajectory, frames)
    m, Q = ep.model, ep.Q
    refs = [("qpos0", np.asarray(m.qpos0, float))]
    for k in range(m.nkey):
        refs.append((f"keyframe {m.key(k).name or k}", np.asarray(m.key_qpos[k], float)))
    rq, _ = EA.robot_dofs(ep)
    is_robot = np.zeros(m.nq, bool)
    is_robot[rq] = True
    resets = []
    for name, ref in refs:
        dn = np.abs(Q - ref)
        for t in range(max(ep.a, 1), ep.b + 1):
            away = dn[t - 1] > from_tol
            snap = away & (dn[t] <= eps)
            n = int(snap.sum())
            if n >= min_dofs and n >= min_frac * max(int(away.sum()), 1):
                # a reset is a discontinuity: the snapped entries jump much farther than they moved the frame before
                # (a trajectory that interpolates onto qpos0 - fingers opened to zero - lands with an ordinary step)
                jump = np.abs(Q[t, snap] - Q[t - 1, snap])
                before = np.abs(Q[t - 1, snap] - Q[t - 2, snap]) if t >= 2 else np.zeros(n)
                if np.median(jump / (before + 1e-4)) < 5.0:
                    continue
                resets.append({"frame": t, "label": ep.labels[t], "reference": name, "snapped": n, "away_before": int(away.sum()),
                               "robot_snapped": int((snap & is_robot).sum()), "object_snapped": int((snap & ~is_robot).sum())})
    same = np.all(Q[1:] == Q[:-1], axis=1)          # same[t]: frame t+1 equals frame t
    frozen = []
    nmin = max(2, int(round(freeze_min_frames * ep.fps / 30.0)))
    for s, e in EA.runs_of(same):
        first, last = s, e + 1                       # frames first..last are identical
        if last < ep.a or first > ep.b:
            continue
        labs = sorted(set(ep.labels[first:last + 1]))
        n = last - first + 1
        crosses = len(labs) > 1
        if (crosses and n >= nmin) or n / ep.fps > freeze_max_s:
            frozen.append({"first": first, "last": last, "frames": n, "labels": labs[:4], "crosses_phases": crosses})
    issues = []
    for r in resets[:5]:
        issues.append(f"frame {r['frame']} ({r['label']}): {r['snapped']} qpos entries ({r['robot_snapped']} robot, "
                      f"{r['object_snapped']} object) snapped back to {r['reference']} at once: the simulation reset after a "
                      f"bad acceleration; find the contact that blew up before this frame (a finger driven deep into a "
                      f"part, a stiff constraint) and re-execute the segment with it fixed")
    for f in frozen[:3]:
        issues.append(f"frames {f['first']}-{f['last']}: the whole state is bit-identical for {f['frames']} frames"
                      + (f" across phases {f['labels']}" if f["crosses_phases"] else "")
                      + ": the executor stopped the segment after an instability and copied the last state forward; "
                        "re-execute that segment")
    worst = (max(resets, key=lambda r: r["snapped"]) if resets else
             ({"frame": frozen[0]["first"], "label": ep.labels[frozen[0]["first"]], "frozen_frames": frozen[0]["frames"]} if frozen else None))
    return EA.finish(issues, worst, resets=resets[:20], n_resets=len(resets), frozen_runs=frozen[:10],
                     references=[r[0] for r in refs], frames=[ep.a, ep.b])


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(badqacc_reset_audit(sys.argv[1], sys.argv[2]), indent=1))
