"""Body contact window check: chosen body pairs stay touching (and others end apart) over frames of an executed episode."""
from __future__ import annotations

TOOL = {
    "name": "body_contact_window_check",
    "title": "Body contact window check",
    "type": "verifier",
    "category": "Task & Rollout Gates",
    "summary": "On the frames of an executed episode, named body pairs must stay in contact (a carried ring kept on two held rings, a part kept on its support) and other pairs must be apart at the end or at some frame (two held parts pulled apart).",
    "description": (
        "Replays the qpos of an executed trajectory (npz with 'qpos', optional 'subgoal') in the scene it was executed in "
        "and, on every evaluated frame, runs forward kinematics and collision detection and lists the contacts between the "
        "subtrees of each named pair (a body and every body rigidly or jointedly below it, so a ring's magnet child counts "
        "as the ring). A pair is 'touching' on a frame when some contact between the two subtrees has distance <= tol_m "
        "(contacts inside a geom margin count, so attraction zones of magnets register). touching pairs must touch on at "
        "least touch_frac of the frames (default all); apart pairs must have no contact closer than apart_m at the frame "
        "'end' (default), on 'any' frame or on 'all' frames. Frames are a [start, end] window, or the frames whose "
        "subgoal index matches (the trajectory must then carry a 'subgoal' array), or all frames. Returns ok, issues "
        "naming the pair, the frame and the distance, and per pair the touching fraction, the "
        "first frame it lost contact and the closest distance."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "Scene MJCF the trajectory was executed in."},
        {"name": "trajectory", "type": "str", "description": "Episode npz with qpos [T, nq] (and optionally subgoal [T])."},
        {"name": "touching", "type": "list[list[str]]", "description": "Body pairs that must stay in contact. Default []"},
        {"name": "apart", "type": "list[list[str]]", "description": "Body pairs that must not be in contact (see apart_at). Default []"},
        {"name": "frames", "type": "list[int]", "description": "[start, end] frame window (inclusive), or None."},
        {"name": "subgoal", "type": "int", "description": "Evaluate the frames whose subgoal index equals this, or None. Raises when the trajectory has no 'subgoal' array."},
        {"name": "tol_m", "type": "float", "description": "Contact distance below which a pair touches. Default 0.002"},
        {"name": "touch_frac", "type": "float", "description": "Fraction of frames a touching pair must touch. Default 1.0"},
        {"name": "apart_m", "type": "float", "description": "Smallest contact distance an apart pair may have. Default 0.0"},
        {"name": "apart_at", "type": "str", "description": "'end', 'any' or 'all' frames for the apart pairs. Default end"},
        {"name": "stride", "type": "int", "description": "Evaluate every stride-th frame (the last frame always). Default 1"},
        {"name": "out_json", "type": "str", "description": "Optional report path."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; frames [start, end]; n_evaluated; touching {pair: {frac, first_lost_frame, max_gap_frame, min_dist_m}}; apart {pair: {apart_ok, closest_m, frame}}."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["contact", "trajectory", "chain", "magnet", "outcome", "verifier", "stage3"],
    "status": "tested",
    "entrypoint": "body_contact_window_check",
}

import json
from pathlib import Path

import numpy as np

_PARAMS = {"touching": None, "apart": None, "frames": None, "subgoal": None, "tol_m": 0.002, "touch_frac": 1.0,
           "apart_m": 0.0, "apart_at": "end", "stride": 1, "out_json": None}


def _subtree(m, root):
    out = {root}
    for b in range(m.nbody):
        p = b
        while p > 0:
            if p == root:
                out.add(b)
                break
            p = int(m.body_parentid[p])
    return out


def body_contact_window_check(scene, trajectory, **params):
    import mujoco
    unknown = sorted(set(params) - set(_PARAMS))
    if unknown:
        raise ValueError(f"body_contact_window_check: unknown parameter(s) {unknown}; allowed: {sorted(_PARAMS)}")
    P = dict(_PARAMS)
    P.update(params)
    touching = [list(p) for p in (P["touching"] or [])]
    apart = [list(p) for p in (P["apart"] or [])]
    if not touching and not apart:
        raise ValueError("body_contact_window_check: give at least one pair in touching or apart")
    if P["apart_at"] not in ("end", "any", "all"):
        raise ValueError("body_contact_window_check: apart_at must be 'end', 'any' or 'all'")
    if not Path(scene).exists() or not Path(trajectory).exists():
        raise ValueError("body_contact_window_check: scene or trajectory path does not exist")
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    z = np.load(trajectory, allow_pickle=True)
    if P["subgoal"] is not None and "subgoal" not in z.files:
        raise ValueError("body_contact_window_check: subgoal was given but the trajectory has no 'subgoal' array; "
                         "pass frames=[start, end] instead")
    Q = np.asarray(z["qpos"])
    issues = []
    report = {"ok": False, "issues": issues}
    if Q.ndim != 2 or Q.shape[1] != m.nq:
        issues.append(f"trajectory qpos width {Q.shape[-1]} does not match the scene nq {m.nq}: pass the scene it was executed in")
        return _finish(report, P)
    T = Q.shape[0]
    if P["frames"] is not None:
        f0, f1 = int(P["frames"][0]), int(P["frames"][1])
        idx = list(range(max(0, f0), min(T - 1, f1) + 1))
    elif P["subgoal"] is not None:
        idx = [int(i) for i in np.nonzero(np.asarray(z["subgoal"]) == int(P["subgoal"]))[0]]
    else:
        idx = list(range(T))
    if not idx:
        issues.append("no frames to evaluate: check frames or subgoal")
        return _finish(report, P)
    st = max(1, int(P["stride"]))
    ev = idx[::st]
    if ev[-1] != idx[-1]:
        ev.append(idx[-1])
    report["frames"] = [idx[0], idx[-1]]
    report["n_evaluated"] = len(ev)
    names = {}
    for pair in touching + apart:
        for nm in pair:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm)
            if bid < 0:
                issues.append(f"body '{nm}' is not in the scene: use the compiled body names")
                return _finish(report, P)
            names[nm] = _subtree(m, bid)
    gb = np.asarray(m.geom_bodyid)

    def pair_dist():
        best = {}
        for c in d.contact[:d.ncon]:
            b1, b2 = int(gb[c.geom1]), int(gb[c.geom2])
            for key in keys:
                A, B = names[key[0]], names[key[1]]
                if (b1 in A and b2 in B) or (b1 in B and b2 in A):
                    best[key] = min(best.get(key, np.inf), float(c.dist))
        return best

    keys = [tuple(p) for p in touching + apart]
    tstat = {tuple(p): {"n_touch": 0, "first_lost_frame": None, "min_dist_m": np.inf, "worst": None} for p in touching}
    astat = {tuple(p): {"closest_m": np.inf, "frame": None, "per_frame_ok": []} for p in apart}
    for f in ev:
        d.qpos[:] = Q[f]
        mujoco.mj_forward(m, d)
        bd = pair_dist()
        for key, s in tstat.items():
            dist = bd.get(key, np.inf)
            if dist <= float(P["tol_m"]):
                s["n_touch"] += 1
            elif s["first_lost_frame"] is None:
                s["first_lost_frame"] = int(f)
            s["min_dist_m"] = min(s["min_dist_m"], dist)
            if s["worst"] is None or dist > s["worst"][1]:
                s["worst"] = (int(f), dist)
        for key, s in astat.items():
            dist = bd.get(key, np.inf)
            if dist < s["closest_m"]:
                s["closest_m"], s["frame"] = dist, int(f)
            s["per_frame_ok"].append((int(f), bool(dist > max(float(P["apart_m"]), float(P["tol_m"])))))
    ok = True
    rep_t, rep_a = {}, {}
    for key, s in tstat.items():
        frac = s["n_touch"] / len(ev)
        good = frac >= float(P["touch_frac"]) - 1e-9
        ok &= good
        rep_t[" - ".join(key)] = {"ok": bool(good), "frac": frac, "first_lost_frame": s["first_lost_frame"],
                                  "max_gap_frame": s["worst"][0] if s["worst"] else None,
                                  "max_gap_m": None if not s["worst"] or s["worst"][1] == np.inf else s["worst"][1],
                                  "min_dist_m": None if s["min_dist_m"] == np.inf else s["min_dist_m"]}
        if not good:
            issues.append(f"{key[0]} and {key[1]} touched on {frac:.0%} of the frames (needed {float(P['touch_frac']):.0%}); "
                          f"first lost at frame {s['first_lost_frame']}: keep them in contact through the motion")
    for key, s in astat.items():
        flags = [g for _, g in s["per_frame_ok"]]
        if P["apart_at"] == "end":
            good = flags[-1]
        elif P["apart_at"] == "any":
            good = any(flags)
        else:
            good = all(flags)
        ok &= good
        rep_a[" - ".join(key)] = {"ok": bool(good), "closest_m": None if s["closest_m"] == np.inf else s["closest_m"],
                                  "frame": s["frame"], "end_apart": bool(flags[-1])}
        if not good:
            issues.append(f"{key[0]} and {key[1]} are still in contact ({P['apart_at']} frame rule): move them apart further")
    report.update(ok=bool(ok), touching=rep_t, apart=rep_a)
    return _finish(report, P)


def _finish(report, P):
    if P.get("out_json"):
        Path(P["out_json"]).parent.mkdir(parents=True, exist_ok=True)
        Path(P["out_json"]).write_text(json.dumps(report, indent=1, default=float))
    return report


if __name__ == "__main__":
    import sys as _sys
    print(json.dumps(body_contact_window_check(_sys.argv[1], _sys.argv[2], touching=[_sys.argv[3:5]]), indent=1, default=float))
