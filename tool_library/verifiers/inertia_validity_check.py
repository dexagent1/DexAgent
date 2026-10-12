"""Inertia validity check: does every moving body carry a mass and an inertia a real rigid body could have?"""
from __future__ import annotations

TOOL = {
    "name": "inertia_validity_check",
    "title": "Inertia validity check",
    "type": "verifier",
    "category": "Scene Compile Checks",
    "summary": "Every body's inertia is positive (semi-)definite and satisfies the triangle inequality, no moving body has zero or "
               "negative mass, and each moving rigid part's radius of gyration fits its own geometry (no placeholder inertias).",
    "description": (
        "MuJoCo refuses some impossible inertias at compile time, but a model edited at run time, an inertia clamped by "
        "boundmass / boundinertia / balanceinertia, or a placeholder copied from a URDF (diaginertia 1e-6 on a 300 g box, "
        "or 1 kg m^2 on a cup) compiles and then spins too freely or not at all. The check works on rigid parts: bodies "
        "welded together without a joint count as one part (the part of a body is its weld root). For every body it "
        "checks finite non-negative mass, non-negative principal inertia and the triangle inequality I1 + I2 >= I3. For "
        "every part that moves (its root has joints), it checks that the part, with what it carries, weighs more than "
        "`min_mass_kg`, that its composite inertia about its centre of mass is positive definite when it can rotate, and "
        "and, for a part that turns freely (free or ball joint, where only its own inertia resists turning), that its largest "
        "radius of gyration sqrt(I_max / m) lies between `rg_ratio_min` and `rg_ratio_max` times the half-extent of the "
        "part's geometry at the initial state (a solid sphere gives 0.63, a rod 0.58, a cube 0.82). "
        "Across each joint the mass ratio of the two parts must stay under `max_mass_ratio`, beyond which the solver is "
        "ill-conditioned. Parts without geometry (wrist slide/hinge carriers) are only checked for mass and definiteness, and an "
        "intermediate link lighter than 1% of what it carries (the massless 'virtual link' between two axes of a finger "
        "joint) only for the mass it carries."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF file, XML string, or compiled model."},
        {"name": "min_mass_kg", "type": "float", "description": "Smallest mass of a moving part. Default 1e-6."},
        {"name": "rg_ratio_min", "type": "float", "description": "Smallest radius of gyration / part half-extent. Default 0.1."},
        {"name": "rg_ratio_max", "type": "float", "description": "Largest radius of gyration / part half-extent. Default 2.0."},
        {"name": "max_mass_ratio", "type": "float", "description": "Largest mass ratio between the two parts a joint connects. Default 1e6."},
        {"name": "key", "type": "int | str", "description": "Keyframe giving the pose the part extents are measured at. Default 0."},
    ],
    "outputs": [
        {"name": "report", "type": "dict",
         "description": "ok; issues; bad_bodies: [{body, mass, inertia, problem}]; parts: [{part, bodies, mass_kg, principal_inertia, "
                        "half_extent_m, rg_ratio, problem}] (problems first); worst; n_moving_parts."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["compile", "mass", "inertia", "radius of gyration", "placeholder"],
    "status": "tested",
    "entrypoint": "inertia_validity_check",
    "cli": "python -m tool_library.verifiers.inertia_validity_check scene.xml",
}

import json
import sys

import numpy as np

from tool_library.common import compile_checks as C


def _part_inertia(model, data, bodies):
    """Mass, COM and 3x3 inertia about the COM (world axes) of a set of bodies at `data`."""
    m = float(sum(model.body_mass[b] for b in bodies))
    if m <= 0:
        return 0.0, np.zeros(3), np.zeros((3, 3))
    com = sum(float(model.body_mass[b]) * data.xipos[b] for b in bodies) / m
    I = np.zeros((3, 3))
    for b in bodies:
        R = data.ximat[b].reshape(3, 3)
        Ib = R @ np.diag(model.body_inertia[b]) @ R.T
        r = data.xipos[b] - com
        I += Ib + float(model.body_mass[b]) * (float(r @ r) * np.eye(3) - np.outer(r, r))
    return m, com, I


def inertia_validity_check(scene, min_mass_kg=1e-6, rg_ratio_min=0.1, rg_ratio_max=2.0, max_mass_ratio=1e6, key=0):
    C.require_scene(scene)
    model, err = C.compile_scene(scene)
    if model is None:
        return C.compile_failure("an inertia MuJoCo rejects (mass/inertia of a moving body below mjMINVAL, or A + B < C on an "
                                 "explicit inertial): fix the body named in the message", err, bad_bodies=[], parts=[])
    data = C.key_data(model, key)
    issues, bad = [], []
    for b in range(1, model.nbody):
        m = float(model.body_mass[b])
        I = np.asarray(model.body_inertia[b], float)
        prob = None
        if not np.isfinite(m) or m < 0:
            prob = f"mass {m}"
        elif not np.isfinite(I).all() or (I < 0).any():
            prob = f"principal inertia {I.tolist()} is negative or not finite"
        elif I.max() > 0:
            s = np.sort(I)
            if s[0] + s[1] < s[2] * (1 - 1e-6) - 1e-15:
                prob = f"principal inertia {np.round(I, 9).tolist()} breaks the triangle inequality ({s[0]:.3g} + {s[1]:.3g} < {s[2]:.3g})"
        if prob:
            bad.append({"body": C.body_name(model, b), "mass": m, "inertia": I.tolist(), "problem": prob})
            issues.append(f"body '{C.body_name(model, b)}': {prob}; no rigid body has this inertia: recompute it from the geometry "
                          "(drop the explicit <inertial>) or give real principal moments")
    # rigid parts: bodies sharing a weld root
    parts = {}
    for b in range(1, model.nbody):
        parts.setdefault(int(model.body_weldid[b]), []).append(b)
    rows = []
    moving_roots = [r for r in parts if r > 0 and int(model.body_jntnum[r]) > 0]
    rotational = {r: any(int(model.jnt_type[j]) in (C.FREE, C.BALL, C.HINGE)
                         for j in range(int(model.body_jntadr[r]), int(model.body_jntadr[r]) + int(model.body_jntnum[r])))
                  for r in moving_roots}
    # free rotation (free or ball joint): the part's own inertia is all that resists turning, so a placeholder shows;
    # about a hinge the joint's armature and the parts it carries dominate, so only definiteness is asked there
    unconstrained = {r: any(int(model.jnt_type[j]) in (C.FREE, C.BALL)
                            for j in range(int(model.body_jntadr[r]), int(model.body_jntadr[r]) + int(model.body_jntnum[r])))
                     for r in moving_roots}
    part_mass = {}
    for r in moving_roots:
        bodies = parts[r]
        m, com, I = _part_inertia(model, data, bodies)
        part_mass[r] = m
        ev = np.linalg.eigvalsh(I) if m > 0 else np.zeros(3)
        gids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in set(bodies) and int(model.geom_type[g]) != C.PLANE]
        half = 0.0
        if gids:
            lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
            for g in gids:
                c, h = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
                R = data.geom_xmat[g].reshape(3, 3)
                ext = np.abs(R) @ h
                ctr = data.geom_xpos[g] + R @ c
                lo, hi = np.minimum(lo, ctr - ext), np.maximum(hi, ctr + ext)
            half = float(np.max(hi - lo)) / 2.0
        rg = float(np.sqrt(max(ev.max(), 0.0) / m)) if m > 0 else 0.0
        ratio = rg / half if half > 0 else None
        prob = None
        name = C.body_name(model, r)
        sub = float(sum(model.body_mass[b] for b in C.subtree_ids(model, r)))    # from body_mass: a run-time edit counts
        # an intermediate link of a multi-axis joint (a hand's 'virtual link' between two finger axes) is massless by
        # design: the bodies it carries hold the mass, so only the subtree's mass is judged for it
        virtual = m < 0.01 * sub
        if sub <= min_mass_kg:
            prob = f"moving part has mass {m:.3g} kg and carries {sub:.3g} kg in total (<= {min_mass_kg})"
            fix = "give it a real mass (geom density or mass, or an <inertial>)"
        elif virtual:
            prob = None
        elif rotational[r] and ev.min() <= 1e-15:
            prob = f"its inertia about the centre of mass is not positive definite (principal {np.round(ev, 12).tolist()})"
            fix = "every rotating part needs inertia about all three axes: give it volume or an explicit full inertia"
        elif ratio is not None and unconstrained[r] and ratio < rg_ratio_min:
            prob = (f"radius of gyration {1000 * rg:.2f} mm is {ratio:.3f} of its half-extent {1000 * half:.1f} mm "
                    f"(< {rg_ratio_min}): a placeholder inertia far too small for its size; it will spin like a point")
            fix = "drop the explicit <inertial> so MuJoCo computes it from the geoms, or compute real moments (m*L^2/12 for a box side L)"
        elif ratio is not None and unconstrained[r] and ratio > rg_ratio_max:
            prob = (f"radius of gyration {1000 * rg:.1f} mm is {ratio:.2f} times its half-extent {1000 * half:.1f} mm "
                    f"(> {rg_ratio_max}): a placeholder inertia far too large for its size; it will refuse to turn")
            fix = "drop the explicit <inertial> so MuJoCo computes it from the geoms, or compute real moments"
        row = {"part": name, "bodies": [C.body_name(model, b) for b in bodies], "mass_kg": m, "subtree_mass_kg": sub,
               "virtual_link": bool(virtual), "free_rotation": bool(unconstrained[r]), "principal_inertia": ev.tolist(),
               "half_extent_m": half, "rg_m": rg, "rg_ratio": ratio, "problem": prob}
        rows.append(row)
        if prob:
            issues.append(f"part '{name}': {prob}; {fix}")
    # mass ratio across joints
    ratio_rows = []
    for r in moving_roots:
        parent = int(model.body_weldid[int(model.body_parentid[r])])
        if parent <= 0 or parent not in part_mass:
            continue
        a, b = part_mass[r], part_mass[parent]
        # a massless intermediate link is judged by what it carries
        sa = float(sum(model.body_mass[x] for x in C.subtree_ids(model, r)))
        sb = float(sum(model.body_mass[x] for x in C.subtree_ids(model, parent)))
        if a < 0.01 * sa:
            a = sa
        if b < 0.01 * sb:
            continue
        if min(a, b) <= 0:
            continue
        q = max(a, b) / min(a, b)
        if q > max_mass_ratio:
            ratio_rows.append({"child": C.body_name(model, r), "parent": C.body_name(model, parent), "ratio": q})
            issues.append(f"joint between part '{C.body_name(model, r)}' ({a:.3g} kg) and its parent '{C.body_name(model, parent)}' "
                          f"({b:.3g} kg): mass ratio {q:.3g} > {max_mass_ratio:g} makes the solver ill-conditioned; bring the masses closer")
    rows.sort(key=lambda x: x["problem"] is None)
    return C.jsonable({"ok": not issues, "compiled": True, "issues": issues, "bad_bodies": bad, "parts": rows[:80],
                       "mass_ratio_violations": ratio_rows, "n_moving_parts": len(rows),
                       "worst": bad[0] if bad else (rows[0] if rows and rows[0]["problem"] else None)})


def main(argv=None):
    rep = inertia_validity_check((sys.argv[1:] if argv is None else argv)[0])
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
