"""Released contact audit: after the hand lets go, does it really stop touching the object?

    from tool_library.verifiers.released_contact_audit import released_contact_audit
    released_contact_audit("scene.xml", "trajectory.npz")

A phase labelled 'let go' that ends with a finger still on the object, or a retreat that drags
the palm across it, leaves the object resting on the hand: Stage 4 then randomizes a scene in
which the object is not actually placed, and the success check reads a state the hand props up.
"""
from __future__ import annotations

TOOL = {
    "name": "released_contact_audit",
    "title": "Released contact audit",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "After every phase whose label says the hand lets go, the hand stops touching the released object soon, "
               "does not drag it while withdrawing, and does not sink into any object while releasing.",
    "description": (
        "No other audit ties contact to the release phases. For every run of a let-go phase (labels with let go, "
        "release, open the hand, drop) of a side ('both' counts for each hand), the released objects are the free "
        "objects named in the label, else the objects that hand touches on the run's first frame. The frames checked "
        "run from grace_frames (default 3) before the end of the let-go run through the following away phases of the "
        "same hand (retreat, withdraw, back off, return home, hold clear...), at most after_s (default 1.5 s) past "
        "the let-go. On the scene as the bench executor runs it, the hand touches an object when their smallest "
        "distance is at most touch_m (default 0.5 mm). Three failures: (1) drag: while the hand touches a released "
        "object, the object moves more than drag_m (default 10 mm) or drag_deg (default 15 deg): the retreat pulls it "
        "off its placement (the spring-back of soft contacts as fingers separate nudges an object a few millimetres "
        "or degrees, which the defaults allow); (2) linger: the hand still touches a released object more than linger_s "
        "(default 1 s) after the let-go ends, or on the last frame checked (a fingertip sliding off a surface for a "
        "few frames as the hand lifts is normal); (3) sink: the hand's collision geometry is more than pen_m "
        "(default 5 mm) inside any object during the let-go and away phases (fingers buried in a container wall "
        "while opening). Reports per release the touching frames, the object's motion while touched and the "
        "deepest penetration."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the episode was executed in."},
        {"name": "trajectory", "type": "str", "description": "trajectory.npz (qpos [T, nq], fps, labels)."},
        {"name": "frames", "type": "[first, last] | None", "description": "Inclusive frame window. Default all."},
        {"name": "touch_m", "type": "float", "description": "Hand-object distance at or under which they touch. Default 0.0005."},
        {"name": "drag_m, drag_deg", "type": "float", "description": "Allowed motion of a released object while the hand touches it. Defaults 0.01, 15."},
        {"name": "linger_s", "type": "float", "description": "Longest time after the let-go the hand may still touch. Default 1.0."},
        {"name": "pen_m", "type": "float", "description": "Allowed hand-object penetration while releasing. Default 0.005."},
        {"name": "grace_frames", "type": "int", "description": "Frames before the end of the let-go run where checking starts. Default 3."},
        {"name": "after_s", "type": "float", "description": "Longest time after the let-go run that is checked. Default 1.5."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; worst {side, object, kind, frames, value}; releases (all, with touching frames, drag mm / deg, last touch, deepest penetration)."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "audit", "release", "contact", "placement", "drag"],
    "status": "tested",
    "entrypoint": "released_contact_audit",
}

import numpy as np

from tool_library.common import episode_audit as EA


def released_contact_audit(scene, trajectory, frames=None, touch_m=0.0005, drag_m=0.01, drag_deg=15.0, linger_s=1.0, pen_m=0.005,
                           grace_frames=3, after_s=1.5):
    EA.check_positive(touch_m=touch_m, drag_m=drag_m, drag_deg=drag_deg, linger_s=linger_s, pen_m=pen_m, after_s=after_s)
    if int(grace_frames) < 0:
        raise ValueError("grace_frames must be >= 0")
    ep = EA.Episode(scene, trajectory, frames)
    if not ep.hands or not ep.objects:
        return EA.finish([], None, releases=[], note="no robot hand or no free object")
    scan = EA.ContactScan(ep, margin=max(touch_m, 0.002))
    names = list(ep.objects)
    rel, events = [], []
    linger_n = int(round(linger_s * ep.fps))
    for side in ep.hands:
        sruns = [r for r in EA.side_runs(ep.labels, side) if r[2] >= ep.a and r[1] <= ep.b]
        hb = ep.hands[side]["bodies"]
        for i, (lab, a, b) in enumerate(sruns):
            if not EA.has_word(lab, EA.RELEASE_WORDS):
                continue
            a, b = max(a, ep.a), min(b, ep.b)               # the part of the let-go run inside the judged window
            if b < a:
                continue
            objs = EA.objects_named(lab, names)
            if not objs:
                objs = [n for n in names if scan.touching(a, hb, ep.objects[n]["bodies"]) < np.inf]
            end = b
            for lab2, a2, b2 in sruns[i + 1:]:
                if a2 > end + 1:
                    break
                if EA.has_word(lab2, EA.AWAY_WORDS):
                    end = b2
                else:
                    break
            end = min(end, b + int(round(after_s * ep.fps)), ep.b)
            start = max(a, b - int(grace_frames))
            dist = {ob: np.full(end - start + 1, np.inf) for ob in objs}
            deepest, deep_at = 0.0, None
            for t in range(a, end + 1):
                for b1, b2, d_, g1, g2, k in scan.at(t):
                    inh1, inh2 = b1 in hb, b2 in hb
                    if inh1 == inh2:
                        continue
                    other = b2 if inh1 else b1
                    if ep.kind(other) == "object" and -d_ > deepest:
                        deepest, deep_at = -d_, (t, ep.body_name(other))
                    if t >= start:
                        for ob in objs:
                            if other in ep.objects[ob]["bodies"]:
                                dist[ob][t - start] = min(dist[ob][t - start], d_)
            if deepest > pen_m:
                events.append({"side": side, "object": deep_at[1], "kind": "sink", "frames": [deep_at[0], deep_at[0]], "label": ep.labels[deep_at[0]],
                               "value": round(1000 * deepest, 1), "unit": "mm", "release": lab})
            for ob in objs:
                touch = dist[ob] <= touch_m
                P, q = EA.object_track(ep, ob)
                drag_mm, drag_dg, drag_fr = 0.0, 0.0, None
                for r0, r1 in EA.runs_of(touch):
                    f0, f1 = start + r0, min(start + r1 + 1, ep.T - 1)
                    dp = float(np.max(np.linalg.norm(P[f0:f1 + 1] - P[f0], axis=1)))
                    dg = float(np.degrees(np.max(EA.quat_angle(np.repeat(q[f0:f0 + 1], f1 - f0 + 1, 0), q[f0:f1 + 1]))))
                    if dp / drag_m + dg / drag_deg > drag_mm / (1000 * drag_m) + drag_dg / drag_deg:
                        drag_mm, drag_dg, drag_fr = 1000 * dp, dg, [f0, f1]
                tf = [start + k for k in np.where(touch)[0]]
                last = tf[-1] if tf else None
                rec = {"side": side, "object": ob, "label": lab, "release_frames": [a, b], "checked": [start, end],
                       "frames_touching": len(tf), "first_touch_frame": tf[0] if tf else None, "last_touch_frame": last,
                       "drag_mm": round(drag_mm, 1), "drag_deg": round(drag_dg, 1), "drag_frames": drag_fr,
                       "deepest_penetration_mm": round(1000 * deepest, 1)}
                rel.append(rec)
                if drag_fr is not None and (drag_mm > 1000 * drag_m or drag_dg > drag_deg):
                    events.append({"side": side, "object": ob, "kind": "drag", "frames": drag_fr, "label": ep.labels[drag_fr[0]],
                                   "value": round(drag_mm, 1), "deg": round(drag_dg, 1), "unit": "mm", "release": lab})
                if last is not None and (last - b > linger_n or (last == end and end > b)):
                    events.append({"side": side, "object": ob, "kind": "linger", "frames": [tf[0], last], "label": ep.labels[last],
                                   "value": round((last - b) / ep.fps, 2), "unit": "s", "release": lab})
    issues = []
    for e in events:
        if e["kind"] == "drag":
            issues.append(f"the {e['side']} hand drags {e['object']} {e['value']} mm / {e['deg']} deg on frames {e['frames'][0]}-{e['frames'][1]} "
                          f"({e['label']}) after '{e['release']}': open the fingers fully and retreat along the approach direction, "
                          f"clear of the object, before moving sideways")
        elif e["kind"] == "linger":
            issues.append(f"the {e['side']} hand still touches {e['object']} {e['value']} s after '{e['release']}' (frames {e['frames'][0]}-"
                          f"{e['frames'][1]}, {e['label']}): the release never separates the hand; open wider and retreat away from it")
        else:
            issues.append(f"the {e['side']} hand is {e['value']} mm inside {e['object']} at frame {e['frames'][0]} ({e['label']}) while "
                          f"releasing ('{e['release']}'): the fingers are buried in it; lift the hand so the fingers open clear of it")
    worst = max(events, key=lambda e: {"sink": 3, "drag": 2, "linger": 1}[e["kind"]]) if events else None
    return EA.finish(issues, worst, releases=rel, n_releases=len(rel), frames=[ep.a, ep.b])


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(released_contact_audit(sys.argv[1], sys.argv[2]), indent=1))
