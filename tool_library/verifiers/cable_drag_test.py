"""Cable drag test: does a rope behave like a rope when dragged?

Puts a `construct_rope` cable on trial with forces only. One end is lifted
by a capped spring-damper pull (0.9 N at most), swung in a circle, lowered
and let go; both ends are then pulled apart to tension it and crossed into
a loop. Measured over the whole trial: stretch of the centreline, the
largest end-to-end span against the nominal length, and the number of
non-neighbour self-contacts. A real cable stretches under 5%, never spans
more than its own length, and touches itself when heaped or looped.

    from tool_library.verifiers.cable_drag_test import test_cable_drag
    r = test_cable_drag(built)                 # built = construct_rope(...)
"""
from __future__ import annotations

TOOL = {
    "name": "cable_drag_test",
    "title": "Cable drag test",
    "type": "verifier",
    "category": "Reconstruction Mechanism Tests",
    "summary": "Drags a cable by capped end forces (lift, swing, drop, tension, loop) and "
               "checks inextensibility, span and self-collision.",
    "description": (
        "Loads a cable built by construct_rope. First it drops the cable as a vertical coil "
        "(turns 1.96 radii apart, i.e. just touching, 1 cm up) and counts the contacts between non-neighbouring "
        "segments as the turns rest on each other: a cable that cannot touch itself falls "
        "through its own turns and cannot be knotted. Then it resets the cable straight and "
        "runs the thread driver: settle, lift one "
        "end 22 cm with a <= 0.9 N spring-damper pull, swing it in a circle whose radius "
        "scales with the span, lower it, let go, pull both ends apart to 92% of the nominal "
        "length, then cross the ends into a loop. Every step records the centreline length "
        "(stretch), the end-to-end span and the count of self-contacts between non-adjacent "
        "segments. CABLE OK requires max stretch under 5%, max span at most 102% of the "
        "nominal length and at least one self-contact in the coil drop or the drag. A cable "
        "that stretches is under-constrained; one that never touches itself cannot be knotted. "
        "The pull is a force on the end link's centre of mass, damped with that point's own velocity "
        "(MuJoCo's cvel is the velocity at the cable's overall centre of mass, which on a thin light "
        "cord turned the damping into a spin feedback that went unstable within seconds), so cords "
        "down to 1.25 mm radius run at the asset's timestep. Every phase, the coil drop included, "
        "fails loudly on a MuJoCo bad-acceleration reset."
    ),
    "inputs": [
        {"name": "built", "type": "dict | str",
         "description": "The dict construct_rope returned, the path of its <name>_params.json, or any MJCF holding a "
                        "cable (a path, or a dict with just 'xml'): the cable's links are found in the model (the longest "
                        "chain of ball-jointed bodies under a free root, or the one named after built['name']) when the "
                        "dict does not list bodies.first / segments / last, and its length, radius and start are read "
                        "from the model when there is no params_json."},
        {"name": "height", "type": "float, metres",
         "description": "z of the cable's resting centreline; the lift is measured from it. "
                        "Default: one radius above the asset's own floor (z = 0)."},
        {"name": "settle_s", "type": "float, seconds",
         "description": "Time the cable is left to land on the floor before the trial. Default 0.6."},
        {"name": "span", "type": "float, metres",
         "description": "Working span for the swing radius (0.35 x span). Default: the cable length."},
        {"name": "out_mp4", "type": "str | None", "description": "Also render the trial to this video."},
        {"name": "out_json", "type": "str | None", "description": "Write the report here."},
    ],
    "outputs": [
        {"name": "ok", "type": "bool", "description": "True when the verdict is CABLE OK."},
        {"name": "verdict", "type": "str", "description": "CABLE OK / CABLE SUSPECT with the numbers."},
        {"name": "measured", "type": "dict",
         "description": "max_force_N, max_stretch_pct, max_span_m, self_contacts, nominal_length_m, "
                        "drag_self_contacts, coil_drop {coil_radius_m, coil_turns, max_self_contacts}."},
        {"name": "video", "type": "str | None", "description": "The rendered mp4 when out_mp4 was given."},
    ],
    "status": "tested",
    "entrypoint": "cable_drag_test",
    "requirements": ["mujoco", "numpy"],
    "simulators": ["mujoco"],
    "tags": ["mechanism", "rope", "cable", "deformable", "force-driven"],
    "used_in": ["Knot the Rope", "Scissor Cutting"],
    "cli": "python -m tool_library.verifiers.cable_drag_test <rope_params.json>",
}


def cable_links(model, built: dict) -> dict:
    """{first, segments, last}: the cable's link bodies in order. Taken from built['bodies'] when it lists them
    (construct_rope's dict); otherwise found in the model: every chain that starts at a free-jointed body and continues
    through children carrying one ball joint each, the one whose root is named after built['name'] (B_first) if any,
    else the longest. Raises ValueError when the model holds no such chain."""
    import mujoco
    b = built.get("bodies")
    if isinstance(b, dict) and b.get("first") and b.get("last") and isinstance(b.get("segments"), list):
        return {"first": b["first"], "last": b["last"], "segments": list(b["segments"])}
    FREE, BALL = int(mujoco.mjtJoint.mjJNT_FREE), int(mujoco.mjtJoint.mjJNT_BALL)
    kids = {i: [] for i in range(model.nbody)}
    for i in range(1, model.nbody):
        kids[int(model.body_parentid[i])].append(i)

    def jtypes(i):
        a, n = int(model.body_jntadr[i]), int(model.body_jntnum[i])
        return [int(model.jnt_type[j]) for j in range(a, a + n)]

    chains, branched = [], []
    for i in range(1, model.nbody):
        if jtypes(i) != [FREE]:
            continue
        if sum(1 for c in kids[i] if jtypes(c) == [BALL]) >= 2:
            branched.append(model.body(i).name)
        ch = [i]
        while True:
            nxt = [c for c in kids[ch[-1]] if jtypes(c) == [BALL]]
            if len(nxt) != 1:
                break
            ch.append(nxt[0])
        if len(ch) >= 3:
            chains.append(ch)
    if not chains and branched:
        raise ValueError(f"cable_drag_test: {branched[0]} joins several strands (a branched cable, e.g. construct_multi_strand_cable): "
                         f"the drag test needs one chain; prove a branched cable with deformable_topology_check (declared topology) "
                         f"and strand_crossing_check")
    if not chains:
        raise ValueError("cable_drag_test: no cable in the model (a chain of >= 3 bodies on ball joints under a free root); "
                         "pass the dict construct_rope returned")
    name = str(built.get("name") or "")
    named = [c for c in chains if name and model.body(c[0]).name.endswith(f"{name}B_first")]
    ch = max(named or chains, key=len)
    names = [model.body(i).name for i in ch]
    return {"first": names[0], "last": names[-1], "segments": names[1:-1]}


def cable_params(model, links: dict) -> dict:
    """length, radius, pos (first vertex), yaw_deg 0 of a cable read from the model at qpos0 (no params sheet)."""
    import mujoco
    import numpy as np
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    ids = [model.body(n).id for n in [links["first"]] + links["segments"] + [links["last"]]]
    P = d.xpos[ids]
    last_geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == ids[-1]]
    g0 = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == ids[0]][0]
    tail = 2.0 * float(model.geom_size[last_geoms[0]][1]) if last_geoms else 0.0
    return {"length": float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) + tail,
            "radius": float(model.geom_size[g0][0]), "pos": [float(v) for v in P[0]], "yaw_deg": 0.0}


def _step_checked(model, data, phase):
    import mujoco
    t = float(data.time)                         # MuJoCo's reset on a bad acceleration sets the time back to 0
    mujoco.mj_step(model, data)
    if int(data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number):
        dof = int(data.warning[mujoco.mjtWarning.mjWARN_BADQACC].lastinfo)
        raise RuntimeError(f"simulation went unstable in the {phase} at t={t:.3f}s (mjWARN_BADQACC at dof {dof}): MuJoCo reset the state")


def self_collision_probe(model, data, built: dict, params: dict,
                         seconds: float = 1.2) -> dict:
    """Drop the cable as a vertical coil and count the non-neighbour
    self-contacts as the turns land on each other. A cable that cannot touch
    itself collapses through its own turns into a flat line with none; one
    that can (the property a knot needs) piles up with dozens."""
    import numpy as np
    L = float(params["length"])
    r = float(params["radius"])
    rad = max(4.0 * r, min(0.04, L / 8.0))           # coil radius
    # TURNS TOUCHING (1.96 radii apart: 2% overlap) and dropped from 1 cm. A
    # rope with dry friction in its joints holds the helix almost rigidly, so
    # turns three radii apart never meet in the trial (measured: 0 contacts in
    # 1.2 s), and on a thin light cord (0.4 m x 2 mm) not even 2.1 radii
    # closed: gravity on its links is too weak against the joint friction.
    # Turns that start in contact are counted at once when the capsules
    # collide and never when they cannot, which is the property under test.
    pitch = 1.96 * r
    turns = L / (2 * np.pi * rad)
    n = 24 * max(2, int(np.ceil(turns)))
    th = np.linspace(0.0, 2 * np.pi * turns, n)
    p0 = np.asarray(params["pos"], float)
    pts = np.stack([p0[0] + rad * np.cos(th), p0[1] + rad * np.sin(th),
                    r + 0.01 + pitch * th / (2 * np.pi)], axis=1)
    from tool_library.common import cable as C
    C.set_centreline(model, data, built, pts)
    ids = [model.body(built["bodies"]["first"]).id] + [
        model.body(b).id for b in built["bodies"]["segments"]] + [
        model.body(built["bodies"]["last"]).id]
    idset = set(ids)
    worst = 0
    for _ in range(int(float(seconds) / model.opt.timestep)):
        _step_checked(model, data, "coil drop")
        sc = 0
        for c in range(data.ncon):
            b1 = model.geom_bodyid[data.contact.geom1[c]]
            b2 = model.geom_bodyid[data.contact.geom2[c]]
            if b1 in idset and b2 in idset and abs(b1 - b2) > 1:
                sc += 1
        worst = max(worst, sc)
    return {"coil_radius_m": round(rad, 4), "coil_turns": round(turns, 2),
            "max_self_contacts": int(worst)}


def test_cable_drag(built, height=None, span=None, settle_s: float = 0.6,
                    out_mp4=None, out_json=None) -> dict:
    import json
    from pathlib import Path
    import mujoco
    from tool_library.common.mjcf import load, clear_forces
    from tool_library.common.drivers import ThreadDriver
    from tool_library.common.mechtest import resolve_built, run_headless, report
    from tool_library.common import cable as C

    if isinstance(built, (str, Path)) and str(built).endswith(".xml"):
        built = {"xml": str(built)}
    built = dict(resolve_built(built, C.rebuild))
    m = load(built["xml"])
    d = mujoco.MjData(m)
    built["bodies"] = cable_links(m, built)        # a dict without bodies.segments used to raise KeyError('segments')
    if built.get("params_json") and Path(built["params_json"]).exists():
        p = json.loads(Path(built["params_json"]).read_text())
    else:
        p = cable_params(m, built["bodies"])
    # 1. self-collision: a dropped coil must land on itself
    probe = self_collision_probe(m, d, built, p)
    # 2. the drag trial from the straight pose
    mujoco.mj_resetData(m, d)
    C.reset_straight(m, d, built, p)
    # let it land on the surface first: the asset's start height is where the
    # video's rope was, not where it rests
    run_headless(m, d, None, settle_s)
    if height is None:
        height = float(p["radius"])          # floor at z = 0 in the asset's own scene
    if span is None:
        span = float(p["length"])
    drv = ThreadDriver(m, d, built, height=float(height), span=float(span))

    def control(t, model, data):
        clear_forces(data)
        return drv.step(t)

    run_headless(m, d, control, drv.T["loop"])
    meas = dict(drv.meas)
    meas["drag_self_contacts"] = meas["self_contacts"]
    meas["coil_drop"] = probe
    # the drag's own loop phase rarely brings the cord onto itself; the coil
    # drop is the self-collision measurement, the drag the inextensibility one
    meas["self_contacts"] = max(meas["self_contacts"], probe["max_self_contacts"])
    drv.meas["self_contacts"] = meas["self_contacts"]
    verdict = drv.verdict()
    video = None
    if out_mp4:
        video = C.demo(built, out_mp4).get("video", str(out_mp4))
    return report(verdict.startswith("CABLE OK"), verdict, meas,
                  out_json=out_json, video=video)


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(test_cable_drag(sys.argv[1]), indent=1))


cable_drag_test = test_cable_drag   # the library calls every tool by its own name
