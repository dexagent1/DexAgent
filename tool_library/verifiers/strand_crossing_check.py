TOOL = {
    "name": "strand_crossing_check",
    "title": "Strand crossing check",
    "type": "verifier",
    "category": "Scene Assertions",
    "summary": "Counts the crossings between different strands of a multi-strand rope, cable or ribbon seen from above, records which strand lies on top at each, orders them along the braid by their mean arc length from the strands' first links (so a three-strand plait reads as alternating pairs), and optionally settles the scene and checks every crossing survives with the same strand on top.",
    "description": (
        "Each strand is an ordered list of link bodies "
        "(from a constructor's build dict, a scene_recon entry with 'strands', or a plain {label: [body, ...]}), "
        "listed from the joined end. At keyframe 0 (or a given qpos) every strand becomes the polyline of its link "
        "positions; segments of different strands are intersected in the horizontal plane and the height of both "
        "strands at the intersection decides which is on top. Each crossing gets the arc length from the first link "
        "along both strands (along_a, along_b) and is ordered by their mean (along_mean): strands that start together "
        "at a junction reach a given crossing at nearly the same arc length, so this is the order along the braid. "
        "Ordering along one strand alone would put every crossing that strand is not part of at the end, and a real "
        "three-strand plait (where each strand takes part in two crossings out of three) would read as repeating "
        "pairs. A three-strand plait shows crossings whose consecutive strand "
        "pairs differ; `expect_alternating` requires that. With settle_s > 0 the scene is simulated under gravity "
        "(robot actuators hold their keyframe controls) and the crossings are counted again: each initial crossing "
        "must be found again within match_m with the same strand on top, else the strands slid apart or passed "
        "through each other. Strands that share their first point (a fan of strands leaving one knot) touch there; an "
        "intersection lying within junction_m of both strands' first links is the joined end, not a crossing. Passes when the initial count is at least min_crossings (and at most max_crossings when "
        "given), the pairs alternate when asked, and, when settled, no crossing was lost or flipped. Assumes the "
        "strands lie roughly horizontal (crossings are judged in the x-y plane)."
    ),
    "inputs": [
        {"name": "scene", "type": "path", "description": "MJCF scene or asset."},
        {"name": "strands", "type": "dict | path", "description": "{label: [link body names in order from the joined end]} or a build dict / JSON with strands {label: {links: [...]}}."},
        {"name": "qpos", "type": "list | None", "description": "State to judge; default keyframe 0 (or qpos0)."},
        {"name": "min_crossings", "type": "int", "description": "Least number of crossings required. Default 1."},
        {"name": "max_crossings", "type": "int | None", "description": "Most crossings allowed (e.g. 0 for an unbraided goal). Default None."},
        {"name": "expect_alternating", "type": "bool", "description": "Consecutive crossings along the braid must be between different strand pairs (a plait). Default False."},
        {"name": "settle_s", "type": "float", "description": "Seconds of passive simulation before the persistence check; 0 skips it. Default 0."},
        {"name": "match_m", "type": "float", "description": "How far a crossing may move during settling and still count as the same. Default 0.03."},
        {"name": "junction_m", "type": "float", "description": "Intersections within this distance of both strands' first links are the shared joined end, not crossings. Default 0.005."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; n_crossings; crossings [{a, b, over, dz_m, xy, along_a, along_b, along_mean}] in braid order; pairs_sequence; alternating; order_rule; after_settle {n_crossings, lost, flipped} when settled."},
    ],
    "status": "tested",
    "entrypoint": "strand_crossing_check",
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["verifier", "deformable", "rope", "cable", "ribbon", "braid", "plait", "crossings", "topology", "unbraid"],
    "used_in": ["EgoDex braid_unbraid"],
    "cli": "python -m tool_library.verifiers.strand_crossing_check scene.xml strands.json",
}

import json
from pathlib import Path

import numpy as np


def _strand_map(strands):
    if isinstance(strands, (str, Path)):
        strands = json.loads(Path(strands).read_text())
    if not isinstance(strands, dict):
        raise ValueError("strands must be a dict {label: [bodies]} or a build dict with 'strands'")
    if "strands" in strands and isinstance(strands["strands"], dict):
        strands = strands["strands"]
    out = {}
    for lab, v in strands.items():
        links = v.get("links") if isinstance(v, dict) else v
        if not isinstance(links, (list, tuple)) or len(links) < 2:
            raise ValueError(f"strand {lab!r} needs an ordered list of at least two link bodies")
        out[str(lab)] = [str(b) for b in links]
    if len(out) < 2:
        raise ValueError("strands: at least two strands are needed to have crossings")
    return out


def _arc(P):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1))])


def _crossings(P, junction_m=0.005):
    labs = list(P)
    out = []
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            A, B = P[labs[i]], P[labs[j]]
            sA, sB = _arc(A), _arc(B)
            for a in range(len(A) - 1):
                p, q = A[a, :2], A[a + 1, :2]
                d1 = q - p
                for b in range(len(B) - 1):
                    r, s = B[b, :2], B[b + 1, :2]
                    d2 = s - r
                    den = d1[0] * d2[1] - d1[1] * d2[0]
                    if abs(den) < 1e-12:
                        continue
                    w = r - p
                    t = (w[0] * d2[1] - w[1] * d2[0]) / den
                    u = (w[0] * d1[1] - w[1] * d1[0]) / den
                    if 0 <= t < 1 and 0 <= u < 1:
                        za = A[a, 2] + t * (A[a + 1, 2] - A[a, 2])
                        zb = B[b, 2] + u * (B[b + 1, 2] - B[b, 2])
                        xy = p + t * d1
                        if (np.linalg.norm(xy - A[0, :2]) < junction_m and np.linalg.norm(xy - B[0, :2]) < junction_m):
                            continue                                   # the strands' shared joined end
                        la = float(sA[a] + t * (sA[a + 1] - sA[a]))
                        lb = float(sB[b] + u * (sB[b + 1] - sB[b]))
                        out.append({"a": labs[i], "b": labs[j], "over": labs[i] if za >= zb else labs[j],
                                    "dz_m": round(float(abs(za - zb)), 5), "xy": [round(float(xy[0]), 4), round(float(xy[1]), 4)],
                                    "along_a": round(la, 4), "along_b": round(lb, 4), "along_mean": round(0.5 * (la + lb), 4)})
    out.sort(key=lambda c: c["along_mean"])
    return out


def _positions(model, data, smap):
    return {lab: data.xpos[[model.body(b).id for b in links]].copy() for lab, links in smap.items()}


def strand_crossing_check(scene, strands, qpos=None, min_crossings=1, max_crossings=None, expect_alternating=False,
                          settle_s=0.0, match_m=0.03, junction_m=0.005):
    import mujoco
    smap = _strand_map(strands)
    if float(junction_m) < 0:
        raise ValueError("junction_m must be >= 0")
    if int(min_crossings) < 0:
        raise ValueError("min_crossings must be >= 0")
    if max_crossings is not None and int(max_crossings) < int(min_crossings):
        raise ValueError("max_crossings must be >= min_crossings")
    try:
        model = mujoco.MjModel.from_xml_path(str(scene))
    except Exception as e:                                   # a scene that does not compile is a verdict, not a crash
        return {"ok": False, "issues": [f"the scene does not compile: {e}"], "n_crossings": 0, "crossings": []}
    missing = [b for links in smap.values() for b in links if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) < 0]
    if missing:
        raise ValueError(f"strands name bodies the scene does not have: {missing[:5]} (check the strand link names)")
    data = mujoco.MjData(model)
    if qpos is not None:
        q = np.asarray(qpos, float)
        if q.shape != (model.nq,):
            raise ValueError(f"qpos has {q.size} values, the scene has nq = {model.nq}")
        data.qpos[:] = q
    elif model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    xs = _crossings(_positions(model, data, smap), float(junction_m))
    pseq = [tuple(sorted((c["a"], c["b"]))) for c in xs]
    alternating = all(pseq[i] != pseq[i + 1] for i in range(len(pseq) - 1)) if len(pseq) > 1 else True
    issues = []
    if len(xs) < int(min_crossings):
        issues.append(f"{len(xs)} strand crossing(s) at the judged state, at least {int(min_crossings)} required: lay the strands across "
                      f"each other (lift the upper strand at each crossing) or rebuild the braid with more crossings")
    if max_crossings is not None and len(xs) > int(max_crossings):
        issues.append(f"{len(xs)} strand crossing(s), at most {int(max_crossings)} allowed")
    if expect_alternating and not alternating:
        issues.append(f"consecutive crossings along the braid repeat the same strand pair ({pseq}): not a plait; alternate "
                      f"left-over-middle and right-over-middle")
    rep = {"n_crossings": len(xs), "crossings": xs[:40], "pairs_sequence": [list(p) for p in pseq], "alternating": bool(alternating),
           "order_rule": "mean arc length from the strands' first links (along_mean)"}
    if settle_s and settle_s > 0:
        if model.nu and model.nkey:
            data.ctrl[:] = model.key_ctrl[0]
        for _ in range(int(float(settle_s) / model.opt.timestep)):
            mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all():
            issues.append("the simulation diverged while settling (non-finite state)")
            rep["after_settle"] = {"diverged": True}
        else:
            ys = _crossings(_positions(model, data, smap), float(junction_m))
            lost, flipped = [], []
            used = set()
            for c in xs:
                best, bd = None, 1e9
                for k, e in enumerate(ys):
                    if k in used or {e["a"], e["b"]} != {c["a"], c["b"]}:
                        continue
                    d = float(np.hypot(e["xy"][0] - c["xy"][0], e["xy"][1] - c["xy"][1]))
                    if d < bd:
                        best, bd = k, d
                if best is None or bd > float(match_m):
                    lost.append(c)
                else:
                    used.add(best)
                    if ys[best]["over"] != c["over"]:
                        flipped.append({"before": c, "after": ys[best]})
            rep["after_settle"] = {"n_crossings": len(ys), "lost": lost[:10], "flipped": flipped[:10], "settle_s": float(settle_s)}
            if lost:
                issues.append(f"{len(lost)} crossing(s) disappeared while settling for {settle_s} s (strands slid apart or passed through "
                              f"each other): check self-collision between strands and the lift at the crossings")
            if flipped:
                issues.append(f"{len(flipped)} crossing(s) changed which strand is on top while settling: a strand passed through "
                              f"another; enable contact between the strands")
    return {"ok": not issues, "issues": issues, **rep}


if __name__ == "__main__":
    import sys
    print(json.dumps(strand_crossing_check(sys.argv[1], sys.argv[2]), indent=1))
