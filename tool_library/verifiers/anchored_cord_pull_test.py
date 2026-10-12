"""Verifier: a cord whose one end is held by a fixed object (thread under a tape dispenser, rope tied to a post) stays
held when its free end is pulled, and the pull straightens it between the anchor and the pulling hand.
"""
from __future__ import annotations

TOOL = {
    "name": "anchored_cord_pull_test",
    "title": "Anchored cord pull test",
    "type": "verifier",
    "category": "Reconstruction Mechanism Tests",
    "summary": "Force-only trial on a scene: the free end of a cord whose other end is anchored to a fixed object is pulled horizontally away from the anchor with a capped, ramped force; the anchored end must stay put, the anchor body must not move, and the cord must end taut (end-to-end distance close to its length).",
    "description": (
        "Loads the scene at keyframe `key` (gravity on, the robot held by its own actuators at the keyframe ctrl), "
        "finds the cord's links as the chain of bodies between anchored_body and free_body (nested bodies, as "
        "MuJoCo cables build them; or the given chain_bodies) and its length as the summed distance between "
        "consecutive link origins at the keyframe. It checks that an active weld or connect ties anchored_body to a "
        "body that cannot move (not the world), then applies a force on free_body along the horizontal direction "
        "from the anchored end to the free end at the keyframe (or `direction`), ramped to pull_force_N over "
        "ramp_s and held for hold_s. Verdict: the anchored end moves at most tol_anchor_m, the anchor body does not "
        "move, the state stays finite, and at the end the end-to-end distance is at least taut_frac of the cord's "
        "length while the free end has travelled (the pull acted). Assumes the cord lies free to straighten (nothing "
        "else holds it)."
    ),
    "inputs": [
        {"name": "scene", "type": "path", "description": "Scene MJCF with the anchored cord."},
        {"name": "anchored_body", "type": "str", "description": "The cord's end link held by the anchor."},
        {"name": "free_body", "type": "str", "description": "The cord's free end link (the one a hand pulls)."},
        {"name": "chain_bodies", "type": "list[str] | None", "description": "The cord's links in order, when they are not nested bodies. Default None"},
        {"name": "pull_force_N", "type": "float", "description": "Peak pulling force, newtons. Default 0.5"},
        {"name": "ramp_s", "type": "float", "description": "Ramp time to the peak force, seconds. Default 0.5"},
        {"name": "hold_s", "type": "float", "description": "Time the peak force is held, seconds. Default 1.5"},
        {"name": "direction", "type": "list[3] | None", "description": "Pull direction (normalised); default horizontal from the anchored end to the free end. Default None"},
        {"name": "tol_anchor_m", "type": "float", "description": "Largest allowed motion of the anchored end, metres. Default 0.005"},
        {"name": "taut_frac", "type": "float", "description": "End-to-end distance / cord length required at the end. Default 0.9"},
        {"name": "key", "type": "int", "description": "Keyframe to start from. Default 0"},
        {"name": "out_json", "type": "path | None", "description": "Write the report here as JSON. Default None"},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; anchor {equality, type, other_body}; length_m; end_to_end_start_m; end_to_end_end_m; taut_ratio; anchored_end_max_motion_m; anchor_body_motion_m; free_end_travel_m; finite; params."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["rope", "cord", "thread", "anchor", "taut", "pull", "weld", "deformable", "mechanism test"],
    "used_in": ["EgoDex thread_unthread_bead_necklace"],
    "status": "tested",
    "entrypoint": "anchored_cord_pull_test",
    "cli": "python -m tool_library.verifiers.anchored_cord_pull_test scene.xml anchored_body free_body",
}

import json
from pathlib import Path

import numpy as np
import mujoco

_ALLOWED = {"chain_bodies", "pull_force_N", "ramp_s", "hold_s", "direction", "tol_anchor_m", "taut_frac", "key", "out_json"}


def _chain(model, a, b):
    """Bodies from a to b along the nesting (either may be the ancestor), or None."""
    def up(x):
        path = [x]
        while x != 0:
            x = int(model.body_parentid[x]); path.append(x)
        return path
    pa, pb = up(a), up(b)
    if b in pa:
        return pa[:pa.index(b) + 1]
    if a in pb:
        return list(reversed(pb[:pb.index(a) + 1]))
    return None


def anchored_cord_pull_test(scene, anchored_body, free_body, **kw):
    unknown = sorted(set(kw) - _ALLOWED)
    if unknown:
        raise ValueError(f"anchored_cord_pull_test: unknown parameter(s) {unknown}; allowed: {sorted(_ALLOWED)}")
    F = float(kw.get("pull_force_N", 0.5)); ramp = float(kw.get("ramp_s", 0.5)); hold = float(kw.get("hold_s", 1.5))
    tol = float(kw.get("tol_anchor_m", 0.005)); taut = float(kw.get("taut_frac", 0.9)); key = int(kw.get("key", 0))
    if F <= 0 or ramp < 0 or hold <= 0:
        raise ValueError("anchored_cord_pull_test: pull_force_N and hold_s must be > 0, ramp_s >= 0")
    issues = []
    rep = {"ok": False, "issues": issues, "params": {"pull_force_N": F, "ramp_s": ramp, "hold_s": hold, "tol_anchor_m": tol,
                                                     "taut_frac": taut, "key": key}}

    def done():
        rep["ok"] = not issues
        if kw.get("out_json"):
            Path(kw["out_json"]).parent.mkdir(parents=True, exist_ok=True)
            Path(kw["out_json"]).write_text(json.dumps(rep, indent=1, default=float))
        return rep

    model = mujoco.MjModel.from_xml_path(str(scene))
    a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, anchored_body)
    f = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, free_body)
    if a < 0 or f < 0:
        issues.append(f"anchored_body {anchored_body!r} or free_body {free_body!r} is not a body of the scene: pass the "
                      "cord's end link names")
        return done()
    if kw.get("chain_bodies"):
        chain = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in kw["chain_bodies"]]
        if min(chain) < 0:
            issues.append("a chain_bodies name is not a body of the scene")
            return done()
    else:
        chain = _chain(model, a, f)
        if chain is None:
            issues.append("the two end links are not one nested chain: pass chain_bodies in order")
            return done()
    data = mujoco.MjData(model)
    if model.nkey > key:
        mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    # the anchor: an active weld/connect between the anchored link and a body that cannot move
    anchor = None
    for e in range(model.neq):
        if model.eq_type[e] not in (mujoco.mjtEq.mjEQ_WELD, mujoco.mjtEq.mjEQ_CONNECT) or not data.eq_active[e]:
            continue
        if model.eq_objtype[e] != mujoco.mjtObj.mjOBJ_BODY:
            continue
        b1, b2 = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
        if a in (b1, b2):
            other = b2 if b1 == a else b1
            anchor = {"equality": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, e),
                      "type": "weld" if model.eq_type[e] == mujoco.mjtEq.mjEQ_WELD else "connect",
                      "other_body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other), "_id": other}
            break
    if anchor is None:
        issues.append(f"no active weld or connect holds {anchored_body!r}: anchor the cord end to the fixed object "
                      "(anchor_cord_end_to_body) before testing it")
    elif anchor["_id"] == 0:
        issues.append("the cord end is welded to the world: anchor it to the fixed object's body instead")
    elif model.body_weldid[anchor["_id"]] != 0:
        issues.append(f"the anchor body {anchor['other_body']!r} can move: a cord end needs a fixed object to hold it")
    if anchor:
        rep["anchor"] = {k: v for k, v in anchor.items() if k != "_id"}
    pts = data.xpos[chain].copy()
    length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    rep["length_m"] = length
    e2e0 = float(np.linalg.norm(data.xpos[f] - data.xpos[a]))
    rep["end_to_end_start_m"] = e2e0
    if kw.get("direction") is not None:
        u = np.asarray(kw["direction"], float)
    else:
        u = data.xpos[f] - data.xpos[a]; u[2] = 0.0
    if np.linalg.norm(u) < 1e-9:
        issues.append("the free end sits above the anchored end: give a pull direction")
        return done()
    u = u / np.linalg.norm(u)
    a0 = data.xpos[a].copy(); f0 = data.xpos[f].copy()
    ob = anchor["_id"] if anchor else 0
    o0 = data.xpos[ob].copy()
    ctrl0 = data.ctrl.copy()
    dt = float(model.opt.timestep)
    worst, finite = 0.0, True
    for k in range(int((ramp + hold) / dt)):
        t = k * dt
        s = min(1.0, t / ramp) if ramp > 0 else 1.0
        data.ctrl[:] = ctrl0
        data.xfrc_applied[f, :3] = F * s * u
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            finite = False
            break
        worst = max(worst, float(np.linalg.norm(data.xpos[a] - a0)))
    rep["finite"] = finite
    if not finite:
        issues.append("the simulation diverged during the pull: soften the anchor's solref or check the cord's masses")
        return done()
    e2e1 = float(np.linalg.norm(data.xpos[f] - data.xpos[a]))
    rep.update({"end_to_end_end_m": e2e1, "taut_ratio": e2e1 / max(length, 1e-9), "anchored_end_max_motion_m": worst,
                "anchor_body_motion_m": float(np.linalg.norm(data.xpos[ob] - o0)),
                "free_end_travel_m": float(np.linalg.norm(data.xpos[f] - f0))})
    if worst > tol:
        issues.append(f"the anchored end moved {worst * 1000:.1f} mm under a {F:.2f} N pull (allowed {tol * 1000:.1f} mm): "
                      "the anchor does not hold; stiffen its solref/solimp or anchor the right link")
    if rep["anchor_body_motion_m"] > 1e-4:
        issues.append(f"the anchor body moved {rep['anchor_body_motion_m'] * 1000:.1f} mm: it must be fixed")
    if rep["taut_ratio"] < taut:
        issues.append(f"the cord did not go taut: end-to-end {e2e1:.3f} m of {length:.3f} m ({rep['taut_ratio']:.2f} < "
                      f"{taut}); something else holds it, or raise hold_s / pull_force_N")
    if rep["free_end_travel_m"] < 1e-3 and e2e0 < taut * length:
        issues.append("the free end did not move: the pull did not act on the cord")
    return done()


if __name__ == "__main__":
    import sys as _sys
    a = _sys.argv[1:]
    print(json.dumps(anchored_cord_pull_test(a[0], a[1], a[2]), indent=1, default=float))
