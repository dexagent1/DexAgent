"""Dynamics consistency audit: can the contacts an object has explain how it accelerates?

    from tool_library.verifiers.dynamics_consistency_audit import dynamics_consistency_audit
    dynamics_consistency_audit("scene.xml", "trajectory.npz")

Inverse dynamics on every free object: the external force it needs at a frame is its mass
times (its acceleration minus gravity). Contacts can only push along their normals, within
their friction cones. An object that accelerates toward the finger that touches it (pulled), or
sideways faster than friction on its support allows, or that a hand constraint carries, was
moved kinematically: an attach cheat, an edited trajectory.
"""
from __future__ import annotations

TOOL = {
    "name": "dynamics_consistency_audit",
    "title": "Dynamics consistency audit",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "Inverse dynamics on free objects: on frames with contacts, the force each object needs (mass x "
               "(acceleration - gravity)) must lie in the cone its contacts can push with; a hand constraint is no support.",
    "description": (
        "No other audit relates an object's acceleration to its contacts (contact_force_check sums normal force in a "
        "hold window, episode_physics_check bounds speed and jumps). For each free object and frame, the external "
        "force it needs is F = M (a_com - g) (subtree mass M, centre-of-mass acceleration by central difference). "
        "The contacts of its subtree within margin_m (default 3 mm) on that frame and its two neighbours (an impact "
        "between frames counts), on the scene as the bench executor runs it, each contribute a friction pyramid "
        "(normal plus or minus mu along two tangents, mu from the contact, capped at mu_cap), and a contact of a body with "
        "an adhesion actuator (a magnet) may also pull, up to that actuator's force. Non-negative least "
        "squares finds the closest force the contacts can produce; the remainder divided by M is the unexplained "
        "acceleration. Frames without any contact are left to free_fall_plausibility_audit; frames where an "
        "object-object or object-world constraint (a cap on its thread) is engaged are skipped, because a constraint "
        "pushes and pulls. It fails when the unexplained acceleration exceeds acc_tol (default 4 m/s^2) on at least "
        "min_frames consecutive frames (default 2), and when a robot-object constraint is engaged for more than "
        "attach_frames frames (the object rides on a constraint to the hand: an attach). Reports the worst run per "
        "object with frames, phase labels, the needed force and the contacts present."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the episode was executed in."},
        {"name": "trajectory", "type": "str", "description": "trajectory.npz (qpos [T, nq], fps, labels, optional eq_active)."},
        {"name": "frames", "type": "[first, last] | None", "description": "Inclusive frame window. Default all."},
        {"name": "margin_m", "type": "float", "description": "Contact distance that counts. Default 0.003."},
        {"name": "acc_tol", "type": "float", "description": "Unexplained acceleration that fails, m/s^2. Default 4."},
        {"name": "min_frames", "type": "int", "description": "Consecutive failing frames that make a run. Default 2."},
        {"name": "mu_cap", "type": "float", "description": "Largest friction coefficient credited to a contact. Default 1.5."},
        {"name": "attach_frames", "type": "int", "description": "Frames a robot-object constraint may be engaged before it counts as an attach. Default 3."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; worst {body, frames, labels, unexplained_m_s2, needed_force_n, contacts}; objects {name: frames judged, worst unexplained acceleration, frame}; attaches."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "audit", "physics", "inverse dynamics", "attach", "kinematic cheat"],
    "status": "tested",
    "entrypoint": "dynamics_consistency_audit",
}

import numpy as np
from scipy.optimize import lsq_linear, nnls

from tool_library.common import episode_audit as EA


def _generators(normals, mu_cap):
    cols = []
    for n, mu, _k in normals:
        n = n / (np.linalg.norm(n) + 1e-12)
        a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
        t1 = np.cross(n, a); t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        mu = min(max(mu, 0.0), mu_cap)
        if mu <= 0:
            cols.append(n)
        else:
            cols += [n + mu * t1, n - mu * t1, n + mu * t2, n - mu * t2]
    return np.array(cols).T if cols else np.zeros((3, 0))


def _inside_one_cone(F, normals, mu_cap):
    """True when F lies inside the friction cone of a single contact (the common case: an object resting on its
    support, a held object pressed into one pad); saves the least-squares solve."""
    fn = float(np.linalg.norm(F))
    if fn < 1e-9:
        return True
    for n, mu, _k in normals:
        c = float(np.dot(F, n)) / (fn * (np.linalg.norm(n) + 1e-12))
        if c > 0 and np.sqrt(max(0.0, 1 - c * c)) <= min(max(mu, 0.0), mu_cap) * c:
            return True
    return False


def _dedupe(normals, step=0.1):
    """One contact per distinct normal direction (rounded to `step`), keeping the largest friction."""
    best = {}
    for n, mu, k in normals:
        u = n / (np.linalg.norm(n) + 1e-12)
        key = tuple(np.round(u / step).astype(int))
        if key not in best or mu > best[key][1]:
            best[key] = (u, mu, k)
    return list(best.values())


def dynamics_consistency_audit(scene, trajectory, frames=None, margin_m=0.003, acc_tol=4.0, min_frames=2, mu_cap=1.5,
                               attach_frames=3):
    EA.check_positive(margin_m=margin_m, acc_tol=acc_tol, min_frames=min_frames, mu_cap=mu_cap, attach_frames=attach_frames)
    ep = EA.Episode(scene, trajectory, frames)
    if not ep.objects:
        return EA.finish([], None, objects={}, note="no free objects")
    T, g = ep.T, ep.gravity
    tl = EA.object_contact_timeline(ep, margin=margin_m, normals=True)
    links = EA.object_links(ep)
    events, objs, attaches = [], {}, []
    for n, o in ep.objects.items():
        M = o["mass"]
        if M <= 0 or T < 3:
            continue
        P = ep.com_track(o["bodies"])
        acc = np.zeros((T, 3)); acc[1:-1] = EA.central_accel(P, ep.fps)
        mech = np.zeros(T, bool); rob = np.zeros(T, bool)
        for mask, partner, name in links.get(n, []):
            if partner == "robot":
                rob |= np.asarray(mask, bool)
            else:
                mech |= np.asarray(mask, bool)
        ra = rob.copy(); ra[:ep.a] = False; ra[ep.b + 1:] = False
        for a, b in EA.runs_of(ra):
            if b - a + 1 > attach_frames:
                attaches.append({"body": n, "frames": [a, b], "labels": [ep.labels[a], ep.labels[b]], "n": b - a + 1})
        un = np.zeros(T); judged = np.zeros(T, bool); info = {}
        for t in range(max(ep.a, 1), min(ep.b, T - 2) + 1):
            if mech[t]:
                continue
            normals = tl[t - 1][n]["normals"] + tl[t][n]["normals"] + tl[t + 1][n]["normals"]
            if not normals:
                continue
            adh = [(nv, f) for tt in (t - 1, t, t + 1) for (nv, _mu, _k), f in zip(tl[tt][n]["normals"], tl[tt][n].get("adhesion") or []) if f > 0]
            F = M * (acc[t] - g)
            if _inside_one_cone(F, normals, mu_cap):
                res = 0.0
            else:
                A = _generators(_dedupe(normals), mu_cap)
                lam, res = nnls(A, F)
                if adh and res / M > 1e-6:
                    # an adhesion actuator (a magnet) also pulls through its contacts, in all at most its own force (the same
                    # contact seen on the neighbouring frames is one contact): the pull is either spread evenly over the
                    # magnet's contacts or all on one of them, so no solution uses more than the magnet has
                    pulls = _dedupe([(nv, f, None) for nv, f in adh])
                    P = np.array([-u for u, _f, _k in pulls]).T
                    fmax = max(f for _u, f, _k in pulls)
                    k = P.shape[1]
                    caps = [np.full(k, fmax / k)] + [np.where(np.arange(k) == i, fmax, 0.0) for i in range(k)] if k > 1 else [np.array([fmax])]
                    AP = np.hstack([A, P])
                    for cap in caps:
                        sol = lsq_linear(AP, F, bounds=(np.zeros(AP.shape[1]), np.r_[np.full(A.shape[1], np.inf), cap]))
                        res = min(res, float(np.linalg.norm(AP @ sol.x - F)))
            un[t] = res / M
            judged[t] = True
            info[t] = (F, sorted({k for _, _, k in normals}))
        bad = judged & (un > acc_tol)
        k = int(np.argmax(un)) if judged.any() else None
        objs[n] = {"frames_judged": int(judged.sum()), "worst_unexplained_m_s2": round(float(un.max()), 2),
                   "worst_frame": k, "mechanism_frames_skipped": int(mech[ep.a:ep.b + 1].sum())}
        for a, b in EA.runs_of(bad):
            if b - a + 1 < min_frames:
                continue
            t = a + int(np.argmax(un[a:b + 1]))
            F, kinds = info[t]
            events.append({"body": n, "frames": [a, b], "labels": [ep.labels[a], ep.labels[b]], "frame": t,
                           "unexplained_m_s2": round(float(un[t]), 2), "needed_force_n": [round(float(x), 3) for x in F],
                           "weight_n": round(M * float(np.linalg.norm(g)), 3), "contacts": kinds, "n": b - a + 1})
    issues = []
    for n in sorted({e["body"] for e in events}):
        evs = [e for e in events if e["body"] == n]
        w = max(evs, key=lambda e: e["unexplained_m_s2"])
        issues.append(f"{n} at frame {w['frame']} ({w['labels'][0]}) needs a force {w['needed_force_n']} N (weight {w['weight_n']} N) "
                      f"that its contacts ({', '.join(w['contacts'])}) cannot push: {w['unexplained_m_s2']} m/s^2 unexplained over "
                      f"frames {w['frames'][0]}-{w['frames'][1]} ({len(evs)} runs): the object was moved kinematically; "
                      f"re-execute the segment in physics so only the fingers move it")
    for x in attaches[:3]:
        issues.append(f"{x['body']} rides on a constraint to the robot for {x['n']} frames ({x['frames'][0]}-{x['frames'][1]}, "
                      f"{x['labels'][0]}): an attach, not a grasp; remove the constraint and hold the object by contact")
    worst = max(events, key=lambda e: e["unexplained_m_s2"]) if events else (attaches[0] if attaches else None)
    return EA.finish(issues, worst, objects=objs, attaches=attaches[:10], n_runs=len(events), frames=[ep.a, ep.b])


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(dynamics_consistency_audit(sys.argv[1], sys.argv[2]), indent=1))
