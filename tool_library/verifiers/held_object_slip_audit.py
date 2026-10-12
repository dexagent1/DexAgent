"""Held object slip audit: does the carried object stay put in the hand?

    from tool_library.verifiers.held_object_slip_audit import held_object_slip_audit
    held_object_slip_audit("scene.xml", "trajectory.npz")

A grasp that holds in the stress tests can still slip during the episode's real carry: the
object slides along the fingers, twists in the grip, and arrives at the placement somewhere
the plan did not put it. The executor measures slip per subgoal at execution time; this audit
re-measures it on the delivered episode, over the whole carry.
"""
from __future__ import annotations

TOOL = {
    "name": "held_object_slip_audit",
    "title": "Held object slip audit",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "During carry phases the held object's pose relative to the hand drifts no more than a tolerance from "
               "where it sat when the carry began.",
    "description": (
        "No audit measures in-hand slip on the delivered episode (the grasp stress tests run on a hand scene, the "
        "executor's per-subgoal outcome is written at execution time). For every run of phases whose label says the "
        "side carries (carry, transport; consecutive runs of the same hand and object, e.g. carry up / across / "
        "down, are joined), with the object named in the label or, when it names none, the object touching that hand "
        "on most of the run, the object's pose is expressed in the frame of the hand's wrist body on every frame. An "
        "articulated object (a cable or ribbon of linked segments, a box with a hinged lid) is judged at the link the hand "
        "holds, the body of the object its contacts touch most on the carry's first frame that has any (the deepest "
        "contact breaks a tie), and at the point those contacts touch, not at its root: a fingertip dragging the end link "
        "of a ribbon moves that link, while the ribbon's junction 10 cm away swings with the rest of the strand (a false "
        "108 mm). The turn of a cable link (a segment on a ball joint) is reported but not judged: it is the cable's local "
        "bend and twist under the pad, which no fingertip or pinch holds. The drift is the largest distance and rotation "
        "from that relative pose at the carry's first frame. It fails "
        "above slip_m (default 20 mm) or slip_deg (default 15 deg). Lift phases are left out by default: the grip "
        "seats while the object leaves its support (the paper's executed episodes seat by up to 13 mm there); pass "
        "words to include them. Reports the worst window per hand and object with the frame where the drift peaks."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the episode was executed in."},
        {"name": "trajectory", "type": "str", "description": "trajectory.npz (qpos [T, nq], fps, labels)."},
        {"name": "frames", "type": "[first, last] | None", "description": "Inclusive frame window. Default all."},
        {"name": "slip_m, slip_deg", "type": "float", "description": "Allowed drift of the relative pose. Defaults 0.02 m, 15 deg."},
        {"name": "words", "type": "list[str]", "description": "Phase words that mark a carry. Default ['carry', 'transport']."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; worst {side, object, body (the link judged), frames, labels, drift_mm, drift_deg, turn_judged, peak_frame}; windows (all, with drift)."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "audit", "grasp", "slip", "carry"],
    "status": "tested",
    "entrypoint": "held_object_slip_audit",
}

import numpy as np

from tool_library.common import episode_audit as EA
from tool_library.common import trajectory as TJ
from tool_library.common import held_contact as HC

HELD_SEARCH_S = HC.HELD_SEARCH_S


def _articulated(m, bodies, root):
    """Whether any body of the object below its root moves on its own joints (cable links, a lid on its hinge)."""
    return HC.articulated(m, bodies, root)


def _held_link(ep, scan, side, ob, a, b):
    """(body id, contact point in that body's frame or None): the link of an articulated object the hand holds, i.e. the
    body its contacts touch most on the first frame of the carry with any contact (ties: the deepest contact), with the
    mean point of those contacts; the object's root when the hand touches nothing in the first HELD_SEARCH_S."""
    o = ep.objects[ob]
    hand = ep.hands[side]["bodies"]
    for t in range(a, min(b, a + max(1, int(round(HELD_SEARCH_S * ep.fps)))) + 1):
        scan.at(t)
        frame = HC.contacted_frame(ep.model, scan.d, hand, o["bodies"])
        if frame is not None:
            return frame.body, frame.point
    return o["body"], None


def _cable_link(m, b):
    """A link of a cable: a body on a ball joint (cable composites and construct_multi_strand_cable link their segments
    so). Its orientation is the cable's local bend and twist, which a fingertip or a pinch does not hold."""
    return HC.ball_link(m, b)


def held_object_slip_audit(scene, trajectory, frames=None, slip_m=0.02, slip_deg=HC.RIGID_SLIP_DEG, words=("carry", "transport")):
    EA.check_positive(slip_m=slip_m, slip_deg=slip_deg)
    if not words:
        raise ValueError("words: give at least one phase word that marks a carry")
    ep = EA.Episode(scene, trajectory, frames)
    W = EA.held_windows(ep, words=tuple(words), merge=True)
    rows, scan = [], None
    m = ep.model
    for side, ob, a, b in W:
        o = ep.objects[ob]
        body, local = o["body"], None
        if _articulated(m, o["bodies"], body):
            # an articulated object is judged where the hand holds it: the touched link, at the touched point
            scan = scan or EA.ContactScan(ep, 0.003)
            body, local = _held_link(ep, scan, side, ob, a, b)
        p, R = EA.relative_pose(ep, body, ep.hands[side]["root"])
        if local is not None:
            p = p + np.einsum("tij,j->ti", R, local)       # the held point on the link, in the wrist frame
        d = np.linalg.norm(p[a:b + 1] - p[a], axis=1)
        ang = np.array([np.degrees(TJ.angle_between(R[a], R[t])) for t in range(a, b + 1)])
        turn_judged = not _cable_link(m, body)
        score = d / slip_m + (ang / slip_deg if turn_judged else 0.0)
        k = int(np.argmax(score))
        rows.append({"side": side, "object": ob, "body": ep.body_name(body), "frames": [a, b], "labels": [ep.labels[a], ep.labels[b]],
                     "drift_mm": round(1000 * float(d.max()), 1), "drift_deg": round(float(ang.max()), 1), "turn_judged": turn_judged,
                     "peak_frame": a + k, "fail": bool(d.max() > slip_m or (turn_judged and ang.max() > slip_deg))})
    bad = [r for r in rows if r["fail"]]
    issues = [f"{r['object']}{'' if r['body'] == r['object'] else ' (held at ' + r['body'] + ')'} slips {r['drift_mm']} mm / {r['drift_deg']} deg in the {r['side']} hand during frames {r['frames'][0]}-"
              f"{r['frames'][1]} ({r['labels'][0]}), peak at frame {r['peak_frame']}: choose a grasp with more enclosure or "
              f"contact area (or a slower carry); the grip that seated must hold through the whole transfer" for r in bad]
    worst = max(bad, key=lambda r: r["drift_mm"] / (1000 * slip_m) + r["drift_deg"] / slip_deg) if bad else None
    return EA.finish(issues, worst, windows=rows, n_windows=len(rows),
                     note=None if rows else "no carry phase naming (or touching) a free object", frames=[ep.a, ep.b])


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(held_object_slip_audit(sys.argv[1], sys.argv[2]), indent=1))
