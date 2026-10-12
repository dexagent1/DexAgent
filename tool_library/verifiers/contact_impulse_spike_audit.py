"""Contact impulse spike audit: slams and servo hits.

    from tool_library.verifiers.contact_impulse_spike_audit import contact_impulse_spike_audit
    contact_impulse_spike_audit("scene.xml", "trajectory.npz")

An object set down from a height, dropped into a bowl, or struck by a hand stops (or starts)
within a frame: the contact delivered an impulse many times the object's weight. A hand driven
into the table or into an object by its servo decelerates the same way. Both look harsh on video,
break real objects and are where a physics replay diverges.
"""
from __future__ import annotations

TOOL = {
    "name": "contact_impulse_spike_audit",
    "title": "Contact impulse spike audit",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "Contact force spikes from inverse dynamics (objects stopped or launched by a contact at many times their "
               "weight, hands decelerated hard on contact) and from the executor's force logs when the episode carries them.",
    "description": (
        "contact_force_check bounds the holding force in a window and episode_physics_check the object speed; neither "
        "looks for impulses. Objects: on frames where the object's subtree has a contact within margin_m plus its "
        "travel over the frame (the executor's contact model), the contact force it needed, M |a_com - g| by central "
        "difference, must stay under max_force_ratio times its weight (default 4: a drop of a few centimetres onto a "
        "hard surface reads 2, a slam 5 and more). Hands: the wrist body's linear acceleration (central difference) on "
        "frames where the hand touches an object or the table (or did one frame before) must stay under "
        "max_hand_acc (default 8 m/s^2; a hand stopping from 0.3 m/s within a frame reads 9). Logs: when the "
        "trajectory carries the executor's 'wrist_force_n' (or 'hand_object_force_n'), a frame whose value exceeds "
        "log_ratio (default 4) times the median of the surrounding second and log_floor_n (default 15 N) is a servo "
        "hit. Reports the worst spike per body with its frame and phase label."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the episode was executed in."},
        {"name": "trajectory", "type": "str", "description": "trajectory.npz (qpos, fps, labels; optional wrist_force_n, hand_object_force_n)."},
        {"name": "frames", "type": "[first, last] | None", "description": "Inclusive frame window. Default all."},
        {"name": "margin_m", "type": "float", "description": "Contact distance that counts (plus the frame's travel). Default 0.003."},
        {"name": "max_force_ratio", "type": "float", "description": "Allowed contact force on an object, in multiples of its weight. Default 4."},
        {"name": "max_hand_acc", "type": "float", "description": "Allowed wrist acceleration on contact frames, m/s^2. Default 8."},
        {"name": "log_ratio, log_floor_n", "type": "float", "description": "Spike rule on logged forces: above ratio x local median and above the floor. Defaults 4, 15 N."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; worst {body, frame, label, value, limit, unit}; objects {name: peak force ratio, frame}; hands {side: peak contact acceleration, frame}; logs {key: peak, frame}."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "audit", "physics", "impulse", "slam", "servo"],
    "status": "tested",
    "entrypoint": "contact_impulse_spike_audit",
}

import numpy as np

from tool_library.common import episode_audit as EA


def contact_impulse_spike_audit(scene, trajectory, frames=None, margin_m=0.003, max_force_ratio=4.0, max_hand_acc=8.0,
                                log_ratio=4.0, log_floor_n=15.0):
    EA.check_positive(margin_m=margin_m, max_force_ratio=max_force_ratio, max_hand_acc=max_hand_acc, log_ratio=log_ratio,
                      log_floor_n=log_floor_n)
    ep = EA.Episode(scene, trajectory, frames)
    T, g = ep.T, ep.gravity
    gn = float(np.linalg.norm(g)) or 9.81
    events, objs, hands, logs = [], {}, {}, {}
    tl = EA.object_contact_timeline(ep, margin=margin_m + 0.02) if ep.objects else None
    for n, o in ep.objects.items():
        if o["mass"] <= 0 or T < 3:
            continue
        P = ep.com_track(o["bodies"])
        acc = np.zeros((T, 3)); acc[1:-1] = EA.central_accel(P, ep.fps)
        step = np.r_[0.0, np.linalg.norm(np.diff(P, axis=0), axis=1)]
        ratio = np.linalg.norm(acc - g, axis=1) / gn
        ratio[0] = ratio[-1] = 0.0
        best = (0.0, None)
        for t in range(max(ep.a, 1), min(ep.b, T - 2) + 1):
            tol = margin_m + min(0.02, float(max(step[t], step[t + 1])))
            if not any(EA.contact_within(tl[k][n], tol) for k in (t - 1, t, t + 1)):
                continue
            if ratio[t] > best[0]:
                best = (float(ratio[t]), t)
            if ratio[t] > max_force_ratio:
                events.append({"body": n, "kind": "object", "frame": t, "label": ep.labels[t], "value": round(float(ratio[t]), 2),
                               "limit": max_force_ratio, "unit": "x weight"})
        objs[n] = {"peak_force_ratio": round(best[0], 2), "frame": best[1]}
    if ep.hands and T >= 3:
        from tool_library.common.episode_audit import ContactScan
        scan = ContactScan(ep, margin_m)
        touch = {s: np.zeros(T, bool) for s in ep.hands}
        side_of = {b: s for s, h in ep.hands.items() for b in h["bodies"]}
        for t in range(max(ep.a - 1, 0), min(ep.b + 1, T - 1) + 1):
            for b1, b2, dist, *_ in scan.at(t):
                for bh, bo in ((b1, b2), (b2, b1)):
                    s = side_of.get(bh)
                    if s is not None and bo not in ep.hands[s]["bodies"] and ep.kind(bo) != "robot":
                        touch[s][t] = True
        for s, h in ep.hands.items():
            P = ep.body_pos(h["root"])
            acc = np.zeros(T); acc[1:-1] = np.linalg.norm(EA.central_accel(P, ep.fps), axis=1)
            best = (0.0, None)
            for t in range(max(ep.a, 1), min(ep.b, T - 2) + 1):
                if not (touch[s][t] or touch[s][t - 1] or touch[s][t + 1]):
                    continue
                if acc[t] > best[0]:
                    best = (float(acc[t]), t)
                if acc[t] > max_hand_acc:
                    events.append({"body": f"{s} hand", "kind": "hand", "frame": t, "label": ep.labels[t], "value": round(float(acc[t]), 2),
                                   "limit": max_hand_acc, "unit": "m/s^2"})
            hands[s] = {"peak_contact_acc_m_s2": round(best[0], 2), "frame": best[1]}
    for key in ("wrist_force_n", "hand_object_force_n"):
        if key not in ep.extras:
            continue
        x = np.asarray(ep.extras[key], float)
        if x.ndim == 1:
            x = x[:, None]
        if len(x) != T:
            continue
        W = max(3, int(round(ep.fps)))
        for c in range(x.shape[1]):
            v = x[:, c]
            med = EA.moving_median(v, W // 2)
            k = ep.a + int(np.argmax(v[ep.a:ep.b + 1]))
            logs[f"{key}[{c}]"] = {"peak_n": round(float(v[k]), 2), "frame": k}
            for t in range(ep.a, ep.b + 1):
                if v[t] > log_floor_n and v[t] > log_ratio * max(med[t], 1e-6):
                    events.append({"body": f"{key}[{c}]", "kind": "log", "frame": t, "label": ep.labels[t], "value": round(float(v[t]), 2),
                                   "limit": round(float(max(log_floor_n, log_ratio * med[t])), 2), "unit": "N"})
    issues = []
    advice = {"object": "lower it onto its support at the speed of a placement (finish the descent before the let-go) instead of "
                        "dropping or striking it",
              "hand": "slow the approach in the last centimetres before contact (a guarded move) so the servo does not hit",
              "log": "the executor logged a force spike: slow the motion into that contact"}
    for body in sorted({e["body"] for e in events}):
        evs = [e for e in events if e["body"] == body]
        w = max(evs, key=lambda e: e["value"] / e["limit"])
        issues.append(f"{body}: contact spike {w['value']} {w['unit']} at frame {w['frame']} ({w['label']}), limit {w['limit']} "
                      f"({len(evs)} frames); {advice[w['kind']]}")
    worst = max(events, key=lambda e: e["value"] / e["limit"]) if events else None
    return EA.finish(issues, worst, objects=objs, hands=hands, logs=logs, n_events=len(events), frames=[ep.a, ep.b])


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(contact_impulse_spike_audit(sys.argv[1], sys.argv[2]), indent=1))
