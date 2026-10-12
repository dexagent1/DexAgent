"""Redundant motion audit: the hand turns away and back inside a phase for no reason.

    from tool_library.verifiers.redundant_motion_audit import redundant_motion_audit
    redundant_motion_audit("scene.xml", "trajectory.npz")

wrist_spin_check counts reversals of the arm's wrist ROLL joint (joints named roll / j6 /
wrist) inside one phase. A hand also turns about its other axes, a floating hand turns through
its virtual wrist joints, and an arm can turn the hand through several joints at once; none of
that is seen by a check on one named joint. This audit watches the hand's orientation itself.
"""
from __future__ import annotations

TOOL = {
    "name": "redundant_motion_audit",
    "title": "Redundant motion audit",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "Inside each phase, the hand's orientation (all three axes) turning away and back: the rotation path is "
               "longer than the net rotation the phase needs by an out-and-back excursion.",
    "description": (
        "Covers what wrist_spin_check misses: it reads one named roll joint; this audit reads the hand's world "
        "orientation (the body its wrist joints move), whatever joints produce it, about any axis. For every phase "
        "run of a hand (its own part of a '<side>: <phase>' label, a 'both' part, or its half of a two-hand label "
        "'left: X | right: Y') it computes the angle from the "
        "phase's start orientation and the angle to its end orientation on every frame. The out-and-back excursion "
        "is the larger of (a) how far the hand came back toward its start after its farthest point and (b) how far "
        "it moved away from its end after its closest approach; a geodesic turn and a detour through a via "
        "orientation both score near zero, turning 40 degrees and back scores 40. It fails at excursion_deg "
        "(default 20). It also reports, per phase, the rotation path "
        "length against the net rotation. Phases whose label names a rotation task (turn, twist, unscrew, pour, "
        "flip, wipe...) are judged too: inside one such phase the turn should still be one way."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the episode was executed in."},
        {"name": "trajectory", "type": "str", "description": "trajectory.npz (qpos [T, nq], fps, labels)."},
        {"name": "frames", "type": "[first, last] | None", "description": "Inclusive frame window. Default all."},
        {"name": "excursion_deg", "type": "float", "description": "Out-and-back excursion that fails. Default 20."},
        {"name": "min_frames", "type": "int", "description": "Shortest phase run judged. Default 5."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; worst {side, frames, label, excursion_deg, path_deg, net_deg, turn_frame}; phases (top 10 by excursion)."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "audit", "wrist", "rotation", "redundancy", "deployability"],
    "status": "tested",
    "entrypoint": "redundant_motion_audit",
}

import numpy as np

from tool_library.common import episode_audit as EA


def _excursion(q):
    n = len(q)
    fs = np.degrees(EA.quat_angle(np.repeat(q[:1], n, 0), q))
    te = np.degrees(EA.quat_angle(q, np.repeat(q[-1:], n, 0)))
    back = np.maximum.accumulate(fs) - fs              # came back toward the start after the farthest point
    away = te - np.minimum.accumulate(te)              # moved away from the end after the closest approach
    kb, ka = int(np.argmax(back)), int(np.argmax(away))
    if back[kb] >= away[ka]:
        return float(back[kb]), int(np.argmax(fs[:kb + 1])), fs, te
    return float(away[ka]), int(np.argmin(te[:ka + 1])), fs, te


def redundant_motion_audit(scene, trajectory, frames=None, excursion_deg=20.0, min_frames=5):
    EA.check_positive(excursion_deg=excursion_deg, min_frames=min_frames)
    ep = EA.Episode(scene, trajectory, frames)
    if not ep.hands:
        return EA.finish([], None, phases=[], note="no robot hand in the scene")
    rows = []
    for s in ep.hands:
        for lab, a, b in EA.side_runs(ep.labels, s):
            a, b = max(a, ep.a), min(b, ep.b)
            if b - a + 1 < min_frames:
                continue
            q = ep.body_quat(ep.hands[s]["root"])[a:b + 1]
            exc, turn, fs, te = _excursion(q)
            path = float(np.degrees(np.sum(EA.quat_angle(q[:-1], q[1:]))))
            net = float(fs[-1])
            rows.append({"side": s, "frames": [a, b], "label": lab, "excursion_deg": round(exc, 1), "path_deg": round(path, 1),
                         "net_deg": round(net, 1), "turn_frame": a + turn, "rotation_task": EA.has_word(lab, EA.ROT_TASK_WORDS)})
    rows.sort(key=lambda r: -r["excursion_deg"])
    bad = [r for r in rows if r["excursion_deg"] >= excursion_deg]
    issues = [f"{r['side']} hand, frames {r['frames'][0]}-{r['frames'][1]} ({r['label']}): turns {r['excursion_deg']} deg away "
              f"and back (path {r['path_deg']} deg for a net {r['net_deg']} deg, turning point frame {r['turn_frame']}); plan the "
              f"phase's orientation as one geodesic turn (slerp between its start and end) or remove the intermediate "
              f"orientation target" for r in bad[:6]]
    if len(bad) > 6:
        issues.append(f"... {len(bad) - 6} more phases turn away and back")
    return EA.finish(issues, bad[0] if bad else None, phases=rows[:10], n_flagged=len(bad), frames=[ep.a, ep.b])


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(redundant_motion_audit(sys.argv[1], sys.argv[2]), indent=1))
