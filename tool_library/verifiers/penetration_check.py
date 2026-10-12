"""Penetration check: does anything in a MuJoCo scene interpenetrate?

Works on any MJCF scene and any robot. It recompiles the scene with every
visible geom collidable, so visual-only parts such as arm links are checked
too, then measures the deepest penetration of every geom pair over a state or
a whole trajectory. With a reference trajectory it reports only penetration
that is NEW relative to the reference, which is how a randomized episode is
judged against the verified one it was derived from.

    from tool_library.verifiers.penetration_check import check_penetration
    report = check_penetration("scene.xml", qpos, baseline_qpos=qpos_ref)
    report["ok"], report["violations"]

    python -m tool_library.verifiers.penetration_check scene.xml trajectory.npz \\
        [--baseline reference.npz] [--tolerance 0.003]
"""
from __future__ import annotations

TOOL = {
    "name": "penetration_check",
    "title": "Penetration check",
    "type": "verifier",
    "category": "Scene Compile Checks",
    "summary": "Deepest interpenetration of every geom pair over a state or trajectory, "
               "on all visible geometry, optionally relative to a reference trajectory.",
    "description": (
        "Loads an MJCF scene and recompiles it with every visible geom collidable, so that "
        "parts a scene normally draws but does not collide, such as arm links, are checked "
        "as well. For each frame it runs MuJoCo's collision detection and keeps, per geom "
        "pair, the deepest penetration seen. A pair fails when its depth exceeds the "
        "tolerance; with a reference trajectory it fails only when it exceeds the depth the "
        "same pair reaches in the reference plus the tolerance, so contacts the verified "
        "episode already had (a grasp, a seated object) are not counted as new collisions. "
        "Bodies joined by an equality constraint and bodies of one limb are never checked "
        "against each other, and MuJoCo itself never collides two bodies that are both fixed "
        "to the world. Any robot and any environment work; nothing is task specific."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel",
         "description": "MJCF file of the scene. A compiled model also works, but then only "
                        "geoms that already collide are checked, because mesh hulls are built "
                        "at compile time."},
        {"name": "qpos", "type": "ndarray [T, nq] or [nq] | None",
         "description": "The state or trajectory to check; the model's initial state when omitted."},
        {"name": "baseline_qpos", "type": "ndarray [T, nq] | None",
         "description": "Reference trajectory. Each pair may penetrate as deep as it does here, "
                        "plus the tolerance."},
        {"name": "tolerance", "type": "float, metres",
         "description": "Penetration allowed beyond the reference, or in absolute terms without "
                        "one. Default 0.003."},
        {"name": "limbs", "type": "list[list[str]] | None",
         "description": "Groups of body names whose members are not checked against each other, "
                        "for example one arm with its hand. Child bodies are included."},
        {"name": "ignore", "type": "list[str]",
         "description": "Regular expressions; a geom pair is skipped when either geom's "
                        "'body/geom' name matches one of them."},
        {"name": "visual_groups", "type": "list[int]",
         "description": "MuJoCo geom groups that count as visible. Default [0, 1, 2]."},
        {"name": "stride", "type": "int", "description": "Check every n-th frame. Default 1."},
    ],
    "outputs": [
        {"name": "report", "type": "dict",
         "description": "ok (bool); violations: the failing pairs, worst first, each with the "
                        "two 'body/geom' names, depth, excess over the reference and the frame; "
                        "worst: the single worst pair; n_pairs_touching; n_frames; "
                        "n_geoms_made_collidable; tolerance."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["collision", "penetration", "contact", "trajectory", "scene"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope", "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting", "Biology Experiment"],
    "status": "tested",
    "entrypoint": "penetration_check",
    "cli": "python -m tool_library.verifiers.penetration_check scene.xml trajectory.npz --baseline reference.npz",
}

import json
import re
import sys
from pathlib import Path

import mujoco
import numpy as np

_PLANE = int(mujoco.mjtGeom.mjGEOM_PLANE)
_JOINED = (int(mujoco.mjtEq.mjEQ_WELD), int(mujoco.mjtEq.mjEQ_CONNECT))


def collidable_model(scene, visual_groups=(0, 1, 2)):
    """The scene with every visible geom able to collide.

    Switching collisions on must happen before compiling: that is when MuJoCo
    builds the convex hull a mesh needs in order to collide at all. A compiled
    model is returned unchanged with a count of 0."""
    if isinstance(scene, mujoco.MjModel):
        return scene, 0
    spec = mujoco.MjSpec.from_file(str(scene))
    # a body with collision geoms of its own (a bowl built from wall pieces) is measured by them: the convex hull of
    # its visual mesh would fill the bowl and read a cup resting inside it as 5 cm deep. Bodies without (arm links,
    # the hands' visual shells) get their visible meshes switched on.
    has_col = {g.parent.name for g in spec.geoms if (g.contype or g.conaffinity) and g.parent is not None
               and not (g.parent.name or "").startswith("robot_")}
    n = 0
    for g in spec.geoms:
        if g.parent is not None and g.parent.name in has_col:
            continue
        if (int(g.group) in visual_groups and g.rgba[3] > 0.0 and g.contype == 0
                and g.conaffinity == 0 and int(g.type) != _PLANE):
            g.contype = g.conaffinity = 1
            n += 1
    return spec.compile(), n


def _subtree(model, bid):
    out, stack = set(), [bid]
    while stack:
        b = stack.pop()
        out.add(b)
        stack += [c for c in range(model.nbody) if model.body_parentid[c] == b and c != b]
    return out


def _limb_sets(model, limbs):
    """Body id sets from body-name lists, each closed under children."""
    out = []
    for names in limbs or ():
        ids = set()
        for n in names:
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
            if b < 0:
                raise ValueError(f"limb body '{n}' is not in the scene")
            ids |= _subtree(model, b)
        out.append(ids)
    return out


def _joined_pairs(model):
    out = set()
    for e in range(model.neq):
        if model.eq_active0[e] and int(model.eq_type[e]) in _JOINED:
            b1, b2 = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
            out.add((min(b1, b2), max(b1, b2)))
    return out


def geom_label(model, g):
    return f"{model.body(model.geom_bodyid[g]).name}/{model.geom(g).name or f'geom{g}'}"


def penetration_profile(model, qpos, limbs=(), ignore=(), stride=1):
    """{(geom a, geom b): (deepest penetration in metres, frame)} over qpos."""
    Q = np.atleast_2d(np.asarray(qpos, float))
    if Q.shape[1] != model.nq:
        raise ValueError(f"qpos has {Q.shape[1]} columns, the scene has nq={model.nq}")
    data = mujoco.MjData(model)
    joined = _joined_pairs(model)
    skip = [re.compile(p) for p in ignore]
    labels = [geom_label(model, g) for g in range(model.ngeom)]
    skipped = {g for g in range(model.ngeom) if any(r.search(labels[g]) for r in skip)}
    out = {}
    for t in range(0, len(Q), max(1, int(stride))):
        data.qpos[:] = Q[t]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)
        for i in range(data.ncon):
            c = data.contact[i]
            if c.dist >= 0 or c.geom1 in skipped or c.geom2 in skipped:
                continue
            b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            if (min(b1, b2), max(b1, b2)) in joined:
                continue
            if any(b1 in limb and b2 in limb for limb in limbs):
                continue
            key = (min(c.geom1, c.geom2), max(c.geom1, c.geom2))
            if -c.dist > out.get(key, (0.0, 0))[0]:
                out[key] = (float(-c.dist), t)
    return out


def check_penetration(scene, qpos=None, baseline_qpos=None, tolerance=0.003, limbs=None,
                      ignore=(), visual_groups=(0, 1, 2), stride=1, max_violations=20):
    """Verify that a state or trajectory adds no interpenetration beyond
    `tolerance`, measured on every visible geom. See the module docstring."""
    model, n_on = collidable_model(scene, visual_groups)
    if qpos is None:
        qpos = model.qpos0.copy()
    limb_sets = _limb_sets(model, limbs)
    prof = penetration_profile(model, qpos, limb_sets, ignore, stride)
    base = {}
    if baseline_qpos is not None:
        base = penetration_profile(model, baseline_qpos, limb_sets, ignore, stride)
    rows = []
    for key, (depth, t) in prof.items():
        excess = depth - base.get(key, (0.0, 0))[0]
        rows.append({"geoms": [geom_label(model, key[0]), geom_label(model, key[1])],
                     "depth_m": round(depth, 5), "excess_m": round(excess, 5), "frame": int(t)})
    rows.sort(key=lambda r: -r["excess_m"])
    violations = [r for r in rows if r["excess_m"] > tolerance]
    return {
        "ok": not violations,
        "tolerance_m": tolerance,
        "relative_to_baseline": baseline_qpos is not None,
        "violations": violations[:max_violations],
        "n_violations": len(violations),
        "worst": rows[0] if rows else None,
        "n_pairs_touching": len(rows),
        "n_frames": int(np.atleast_2d(qpos).shape[0]),
        "n_geoms_made_collidable": n_on,
    }


def _load_qpos(path):
    p = Path(path)
    if p.suffix == ".npz":
        z = np.load(p)
        return z["qpos"] if "qpos" in z.files else z[z.files[0]]
    return np.load(p)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scene", help="MJCF scene file")
    ap.add_argument("qpos", nargs="?", help=".npz with 'qpos' [T, nq] (or .npy); default: the scene's initial state")
    ap.add_argument("--baseline", help="reference trajectory (.npz/.npy)")
    ap.add_argument("--tolerance", type=float, default=0.003)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limb", action="append", default=[],
                    help="comma-separated body names forming one limb; repeatable")
    ap.add_argument("--ignore", action="append", default=[], help="regex on 'body/geom'; repeatable")
    a = ap.parse_args(argv)
    rep = check_penetration(
        a.scene, _load_qpos(a.qpos) if a.qpos else None,
        baseline_qpos=_load_qpos(a.baseline) if a.baseline else None,
        tolerance=a.tolerance, stride=a.stride, ignore=a.ignore,
        limbs=[[s.strip() for s in l.split(",") if s.strip()] for l in a.limb] or None)
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())


penetration_check = check_penetration   # the library calls every tool by its own name
