"""Phase label continuity audit: are the labels a consistent timeline of who does what?

    from tool_library.verifiers.phase_label_continuity_audit import phase_label_continuity_audit
    phase_label_continuity_audit("scene.xml", "trajectory.npz")

Every later stage reads the phase labels: Stage 4 anchors randomization on them, the
finish gate finds the let-go and the carry by them, the audits look up the acting hand.
uncovered_frames_check already fails empty labels, phases shorter than three frames and a
trajectory shorter than the video; this audit checks the labels' CONTENT.
"""
from __future__ import annotations

TOOL = {
    "name": "phase_label_continuity_audit",
    "title": "Phase label continuity audit",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "Labels are '<side>: <phase>' with a side the scene has, the '#k' tag agrees with the subgoal array, each "
               "hand's subgoal index never goes back, and the labelled hand is the one that moves.",
    "description": (
        "Empty labels, sub-three-frame phases and coverage of the video are uncovered_frames_check's; this audit covers "
        "what it misses. (1) Format: every label parses as '<side>: <phase text>' with an optional ' #k' / '#k.m' "
        "subgoal tag, or as several such parts joined by ' | ' when both hands act at once ('left: X | right: Y', "
        "each hand at most once); side is left, right or both, and a named side must be a hand the scene has. (2) Tag agreement: "
        "with a subgoal array in the trajectory, the integer part of a '#k' tag of the label equals subgoal[t]. (3) "
        "Order: per hand (its own part of each label, or a 'both' part), the subgoal index never decreases (two hands may interleave their "
        "branches, e.g. #14.1 left then #14.2 right). Repeated cycles (regrasp, turn back, regrasp) are legal. "
        "(4) Acting hand: over a phase run of at least 0.3 s labelled with one side only, that hand's wrist travels less than "
        "still_m (default 2 mm) and turns less than 2 degrees while the other hand travels more than moving_m "
        "(default 30 mm): the motion belongs to the other hand. Reports the offending frames and labels."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the episode was executed in (for the hands)."},
        {"name": "trajectory", "type": "str", "description": "trajectory.npz (qpos, fps, labels, optional subgoal)."},
        {"name": "frames", "type": "[first, last] | None", "description": "Inclusive frame window. Default all."},
        {"name": "still_m, moving_m", "type": "float", "description": "Acting-hand rule (4): labelled hand under still_m while the other moves over moving_m. Defaults 0.002, 0.03."},
        {"name": "check_acting_hand", "type": "bool", "description": "Run rule (4). Default True."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; worst; malformed [labels]; unknown_sides; tag_mismatches [frames]; order_violations; mislabelled [runs]; phases n."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "audit", "labels", "phases", "timeline", "subgoal"],
    "status": "tested",
    "entrypoint": "phase_label_continuity_audit",
}

import numpy as np

from tool_library.common import episode_audit as EA


def phase_label_continuity_audit(scene, trajectory, frames=None, still_m=0.002, moving_m=0.03, check_acting_hand=True):
    EA.check_positive(still_m=still_m, moving_m=moving_m)
    ep = EA.Episode(scene, trajectory, frames)
    L = ep.labels
    win = list(ep.window)
    issues = []
    parsed = {t: EA.label_parts(L[t]) for t in win}           # one part per hand ('left: X | right: Y' has two)
    malformed = sorted({L[t] for t in win if not parsed[t] and L[t].strip()})
    if malformed:
        fr = [t for t in win if not parsed[t] and L[t].strip()]
        issues.append(f"{len(fr)} frames carry labels that are not '<side>: <phase>' (first at frame {fr[0]}: {L[fr[0]]!r}); "
                      f"write every label as '<left|right|both>: <phase> #<subgoal>' (two hands at once: "
                      f"'left: <phase> | right: <phase>')")
    sides_present = set(ep.hands)
    allp = [q for t in win for q in parsed[t]]
    unknown = sorted({q["side"] for q in allp if q["side"] not in EA.LABEL_SIDES})
    absent = sorted({q["side"] for q in allp if q["side"] in ("left", "right") and sides_present and q["side"] not in sides_present})
    dup = [t for t in win if len({q["side"] for q in parsed[t]}) < len(parsed[t])]
    if unknown:
        issues.append(f"labels name sides {unknown}; use left, right or both")
    if absent:
        issues.append(f"labels name the {absent} hand(s) but the scene has only {sorted(sides_present)}")
    if dup:
        issues.append(f"{len(dup)} frames name the same hand twice in one label (first frame {dup[0]}: {L[dup[0]]!r}); give each "
                      f"hand one part")
    mism = []
    if ep.subgoal is not None:
        for t in win:
            tags = [int(q["tag"].split(".")[0]) for q in parsed[t] if q["tag"] is not None]
            if tags and int(ep.subgoal[t]) not in tags:
                mism.append(t)
        if mism:
            issues.append(f"{len(mism)} frames whose '#k' tag disagrees with the subgoal array (first frame {mism[0]}: "
                          f"{L[mism[0]]!r} vs subgoal {int(ep.subgoal[mism[0]])}); regenerate the labels from the plan")
    order = []
    for s in (sorted(sides_present) or ["left", "right"]):
        last, last_t = None, None
        for t in win:
            p = EA.side_part(L[t], s)
            if p is None:
                continue
            if p["tag"] is not None:
                k = int(p["tag"].split(".")[0])
            else:
                k = int(ep.subgoal[t]) if ep.subgoal is not None else None
            if k is None:
                continue
            if last is not None and k < last:
                order.append({"side": s, "frame": t, "label": L[t], "subgoal": k, "after": last, "after_frame": last_t})
            if last is None or k >= last:
                last, last_t = k, t
    for o in order[:3]:
        issues.append(f"{o['side']} hand goes back from subgoal {o['after']} (frame {o['after_frame']}) to subgoal {o['subgoal']} at "
                      f"frame {o['frame']} ({o['label']}); order the segments by subgoal per hand")
    runs = [(lab, a, b) for lab, a, b in EA.phase_runs(L) if b >= ep.a and a <= ep.b]
    mis = []
    if check_acting_hand and len(sides_present) == 2:
        for lab, a, b in runs:
            parts = EA.label_parts(lab)
            if len(parts) != 1 or parts[0]["side"] not in sides_present or (b - a + 1) < 0.3 * ep.fps:
                continue                               # two-hand and 'both' labels name every hand that may move
            me_side = parts[0]["side"]
            other = [s for s in sides_present if s != me_side][0]
            mv = {}
            for s in sides_present:
                P = ep.body_pos(ep.hands[s]["root"])[a:b + 1]
                q = ep.body_quat(ep.hands[s]["root"])[a:b + 1]
                fq = ep.Q[a:b + 1][:, ep.hands[s]["finger_qadr"]] if ep.hands[s]["finger_qadr"] else np.zeros((b - a + 1, 1))
                mv[s] = (float(np.max(np.linalg.norm(P - P[0], axis=1))),
                         float(np.degrees(np.max(EA.quat_angle(np.repeat(q[:1], len(q), 0), q)))),
                         float(np.max(np.abs(fq - fq[0]))))
            me, ot = mv[me_side], mv[other]
            if me[0] < still_m and me[1] < 2.0 and me[2] < 0.05 and ot[0] > moving_m:
                mis.append({"frames": [a, b], "label": lab, "labelled_hand_mm": round(1000 * me[0], 1), "other": other,
                            "other_hand_mm": round(1000 * ot[0], 1)})
    for x in mis[:3]:
        issues.append(f"frames {x['frames'][0]}-{x['frames'][1]} are labelled {x['label']!r} but that hand moves "
                      f"{x['labelled_hand_mm']} mm while the {x['other']} hand moves {x['other_hand_mm']} mm; label the phase "
                      f"with the hand that acts")
    worst = None
    if order:
        worst = order[0]
    elif mism:
        worst = {"frame": mism[0], "label": L[mism[0]]}
    elif mis:
        worst = mis[0]
    elif malformed:
        worst = {"label": malformed[0]}
    return EA.finish(issues, worst, malformed=malformed[:10], unknown_sides=unknown, absent_sides=absent,
                     tag_mismatches=mism[:20], n_tag_mismatches=len(mism), order_violations=order[:10],
                     mislabelled=mis[:10], n_phases=len(runs), has_subgoal_array=ep.subgoal is not None, frames=[ep.a, ep.b])


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(phase_label_continuity_audit(sys.argv[1], sys.argv[2]), indent=1))
