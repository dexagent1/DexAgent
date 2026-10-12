TOOL = {
    "name": "rope_knot_check",
    "title": "Rope knot check",
    "type": "verifier",
    "category": "Task & Rollout Gates",
    "summary": "Is the rope knotted? Reads a cable's centreline on a frame of an executed trajectory (or a given polyline) "
               "and checks closure polynomials plus diagram certificates: unknot, trefoil "
               "(an overhand knot), figure-eight or other.",
    "description": (
        "A knot is a topological property, so crossings are not counted: a rope piled on itself, a rope through the "
        "table and a loose tangle all have crossings and none is a knot. The cable's segment bodies (construct_rope "
        "names them <prefix>B_first, <prefix>B_<i>, <prefix>B_last) give the centreline on the requested frame; the "
        "open curve is closed far away four ways (over / under the rope, radially / along the end tangents), each "
        "closure's planar diagram gives signed over/under crossings and an exact Alexander polynomial. "
        "A nontrivial polynomial proves nontriviality; polynomial matching alone cannot name a knot or prove the unknot. "
        "Independent 3-D segment checks reject self-intersections, overlapping backtracks and invalid artificial closures. "
        "A bounded deterministic search selects a regular projection before computing its invariant; no usable view means unverified. "
        "Named types require topology-preserving Reidemeister-I/II reduction to a certified small diagram for every closure. "
        "Inconclusive reduction fails a named-type request. The four normalized polynomials must agree exactly, "
        "including for knots named 'other'; zero polynomials and unresolved over/under crossings are rejected "
        "as ambiguous. ends_free is an informational projected-extension diagnostic, not a constraint on whether "
        "a hand or support holds an end. `expect` is 'trefoil' (an overhand knot, the default), "
        "'knot' (any non-trivial agreement), 'figure-eight' or 'unknot'. `also_frames` checks the same verdict on "
        "further frames (a knot that survives the release and settling). A rope that dips more than 5 mm under the "
        "table top (table_z) is reported, because a strand through the table makes crossings no real rope has."
    ),
    "inputs": [
        {"name": "scene", "type": "str | None", "description": "MJCF the trajectory was executed in (the scene that holds the cable)."},
        {"name": "trajectory", "type": "str | None", "description": "trajectory.npz (qpos [T, nq]). With scene and no trajectory the scene's home keyframe is read."},
        {"name": "rope", "type": "str | None", "description": "Name prefix of the cable's bodies (e.g. 'red_rope'). Default: the only cable in the scene."},
        {"name": "frame", "type": "int", "description": "Frame to classify. Default -1 (the last)."},
        {"name": "also_frames", "type": "list[int] | None", "description": "More frames that must give the same verdict."},
        {"name": "expect", "type": "str", "description": "'trefoil' | 'knot' | 'figure-eight' | 'unknot'. Default 'trefoil'."},
        {"name": "points", "type": "list | None", "description": "A centreline polyline (k, 3) to classify instead of a scene."},
        {"name": "table_z", "type": "float", "description": "Table top height for the through-the-table check. Default 0.0."},
    ],
    "outputs": [
        {"name": "ok", "type": "bool", "description": "Every checked frame gives the expected verdict and the rope stays above the table."},
        {"name": "type", "type": "str", "description": "unknot | trefoil | figure-eight | other | ambiguous, on `frame`."},
        {"name": "frames", "type": "list[dict]", "description": "Per checked frame: frame, type, agree, polynomial_agreement, diagram_certificates, embedding_issues, topology_issues, ends_free, n_self_crossings, min_z_m."},
        {"name": "closures", "type": "dict", "description": "The four closures' polynomials on `frame`."},
        {"name": "issues", "type": "list[str]", "description": "What failed."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "sympy"],
    "simulators": ["MuJoCo"],
    "tags": ["rope", "cable", "knot", "topology", "verifier", "deformable"],
    "used_in": ["Knot the Rope"],
    "status": "tested",
    "entrypoint": "rope_knot_check",
}

import re

import numpy as np

from tool_library.common import knot as K

_EXPECT = ("trefoil", "knot", "figure-eight", "unknot")


def _cable_bodies(model, rope=None):
    """Segment bodies of a construct_rope cable in order (first, 1..n, last), by name prefix."""
    pat = re.compile(r"^(?P<pre>.*)B_(?P<k>first|last|\d+)$")
    groups = {}
    for b in range(model.nbody):
        m = pat.match(model.body(b).name or "")
        if m and (rope is None or m.group("pre").startswith(rope)):
            groups.setdefault(m.group("pre"), []).append((m.group("k"), b))
    groups = {k: v for k, v in groups.items() if any(x[0] == "first" for x in v) and any(x[0] == "last" for x in v)}
    if not groups:
        raise ValueError("no cable bodies (<prefix>B_first .. B_last) found" +(f" for rope '{rope}'" if rope else ""))
    if len(groups) > 1:
        raise ValueError(f"several cables in the scene {sorted(groups)}: pass rope=<prefix>")
    (pre, items), = groups.items()
    order = lambda k: -1 if k == "first" else (10 ** 9 if k == "last" else int(k))
    return pre, [b for k, b in sorted(items, key=lambda x: order(x[0]))]


def _centreline(model, data, bodies, pre):
    import mujoco
    P = [data.xpos[b].copy() for b in bodies]
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, pre + "S_last")
    if sid >= 0:
        P.append(data.site_xpos[sid].copy())
    return np.asarray(P, float)


def _verdict(P, expect):
    v = K.check(np.asarray(P, float))
    t = v["type"]
    if expect == "knot":
        hit = bool(v["knot"])
    else:
        hit = bool(v["agree"] and t == expect)
    return v, hit


def rope_knot_check(scene=None, trajectory=None, rope=None, frame=-1, also_frames=None, expect="trefoil", points=None, table_z=0.0):
    if expect not in _EXPECT:
        raise ValueError(f"expect must be one of {_EXPECT}, got {expect!r}")
    issues, rows = [], []
    if points is not None:
        P = np.asarray(points, float)
        if P.ndim != 2 or P.shape[1] != 3 or len(P) < 4:
            raise ValueError("points must be a (k>=4, 3) polyline")
        v, hit = _verdict(P, expect)
        rows.append({"frame": None, "type": v["type"], "agree": v["agree"], "ends_free": v["ends_free"],
                     "polynomial_agreement": v["polynomial_agreement"], "diagram_certificates": v["diagram_certificates"], "topology_issues": v["issues"],
                     "embedding_issues": v["embedding_issues"],
                     "diagram_projections": v["diagram_projections"], "self_crossings_scope": v["self_crossings_scope"],
                     "n_self_crossings": v["n_self_crossings"], "min_z_m": round(float(P[:, 2].min()), 4), "hit": hit})
        closures = v["closures"]
    else:
        import mujoco
        if scene is None:
            raise ValueError("give scene (+ trajectory) or points")
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        if trajectory is not None:
            from tool_library.common.trajectory import load_trajectory
            Q, _, _ = load_trajectory(trajectory)
        else:
            if model.nkey:
                mujoco.mj_resetDataKeyframe(model, data, 0)
            Q = data.qpos[None].copy()
        if Q.shape[1] != model.nq:
            raise ValueError(f"trajectory nq {Q.shape[1]} != scene nq {model.nq}: pass the scene it was executed in")
        pre, bodies = _cable_bodies(model, rope)
        frames = [int(frame)] + [int(f) for f in (also_frames or [])]
        closures = None
        for f in frames:
            fi = f if f >= 0 else len(Q) + f
            if not 0 <= fi < len(Q):
                raise ValueError(f"frame {f} outside the trajectory (T={len(Q)})")
            data.qpos[:] = Q[fi]
            mujoco.mj_kinematics(model, data)
            P = _centreline(model, data, bodies, pre)
            v, hit = _verdict(P, expect)
            if closures is None:
                closures = v["closures"]
            rows.append({"frame": fi, "type": v["type"], "agree": v["agree"], "ends_free": v["ends_free"],
                         "polynomial_agreement": v["polynomial_agreement"], "diagram_certificates": v["diagram_certificates"], "topology_issues": v["issues"],
                         "embedding_issues": v["embedding_issues"],
                         "diagram_projections": v["diagram_projections"], "self_crossings_scope": v["self_crossings_scope"],
                         "n_self_crossings": v["n_self_crossings"], "min_z_m": round(float(P[:, 2].min()), 4), "hit": hit})
    for r in rows:
        at = "" if r["frame"] is None else f" on frame {r['frame']}"
        if not r["hit"]:
            issues.append(f"the rope is '{r['type']}'{at} (expected {expect}; closures agree: {r['agree']}, "
                          f"ends free: {r['ends_free']}, {r['n_self_crossings']} self-crossings)")
            issues.extend(f"rope topology{at}: {issue}" for issue in r["topology_issues"])
        if r["min_z_m"] < float(table_z) - 0.005:
            issues.append(f"the rope dips {1000 * (float(table_z) - r['min_z_m']):.0f} mm under the table{at}: crossings "
                          f"through the table are not a knot")
    return {"ok": not issues, "type": rows[0]["type"], "frames": rows, "closures": closures, "issues": issues}
