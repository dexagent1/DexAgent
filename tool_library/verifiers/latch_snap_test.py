"""Latch snap test: do the straps of a latch box snap shut, carry the box
by the lid, survive a shake and let go when prised?

A headless version of the proof `construct_latch_box` ships with. Five
phases, all forces and gravity, no actuator: the lid is dropped onto the
body (it seats, and nothing holds it); every strap is pinched shut at its
tip along its own outward normal; the whole box is lifted BY THE LID with a
capped gravity-compensated PD force (a latched box comes up with it, an
unlatched one leaves its body behind); the hanging body is shaken sideways;
the pinch is reversed and the latches must let go. The latch state is not
a strap angle: it is whether the wall's nub is inside the strap's window,
catch bar under the nub and still overlapping it radially.

    from tool_library.verifiers.latch_snap_test import test_latch_snap
    r = test_latch_snap(built)                     # built = construct_latch_box(...)
"""
from __future__ import annotations

TOOL = {
    "name": "latch_snap_test",
    "title": "Latch snap test",
    "type": "verifier",
    "category": "Reconstruction Mechanism Tests",
    "summary": "Pinches a latch box's straps shut, lifts the box by its lid, shakes it and "
               "prises the latches open; passes only if the nubs sit in the windows throughout.",
    "description": (
        "Loads a box built by construct_latch_box in its 'seated' layout with the lid 12 mm "
        "above the rim, lets it drop, then runs the five measured phases of the asset's own "
        "proof without rendering: seat (n_latched must read 0, nothing holds the lid), close "
        "(a `close_force_N` pinch at each strap tip along the strap's rotating normal, with "
        "the equal and opposite force on the body; the strap must bow out over the nub and "
        "spring back), lift by the lid (capped PD force; the body must rise more than 40 mm "
        "with at least n-1 latches still engaged), shake (+-0.45 g sideways while hanging), "
        "and open (the reversed pinch; every latch must release). Latched means the catch "
        "bar's top is under the nub's bottom and the bar still overlaps the nub radially by "
        "more than 0.2 mm, measured in the body's own frame."
    ),
    "inputs": [
        {"name": "built", "type": "dict | str",
         "description": "The dict construct_latch_box returned, or the path of its <name>_params.json."},
        {"name": "close_force_N", "type": "float",
         "description": "The pinch at each strap tip. Default 10 N (a thumb). Compare with built['snap_force_N']."},
        {"name": "seconds", "type": "tuple[float x5]",
         "description": "Durations of seat, close, lift, shake, open. Default (1.4, 3.0, 2.6, 2.0, 2.2)."},
        {"name": "out_mp4", "type": "str | None", "description": "Also render the trial to this video."},
        {"name": "out_json", "type": "str | None", "description": "Write the report here."},
    ],
    "outputs": [
        {"name": "ok", "type": "bool", "description": "All four checks passed: latch, carry, hold through the shake, release."},
        {"name": "verdict", "type": "str", "description": "One sentence, or MECHANISM FAILED: <which phases>."},
        {"name": "measured", "type": "dict",
         "description": "after_seat, after_close (n_latched, peak strap bow per latch), after_lift "
                        "(body_rose_mm), after_shake, after_prise_open (fold_deg per latch), snap_force_N."},
        {"name": "video", "type": "str | None", "description": "The rendered mp4 when out_mp4 was given."},
    ],
    "status": "tested",
    "entrypoint": "latch_snap_test",
    "requirements": ["mujoco", "numpy"],
    "simulators": ["mujoco"],
    "tags": ["mechanism", "latch", "snap", "lid", "box", "force-driven"],
    "used_in": [],
    "cli": "python -m tool_library.verifiers.latch_snap_test <latchbox_params.json>",
}


def test_latch_snap(built, close_force_N: float = 10.0,
                    seconds=(1.4, 3.0, 2.6, 2.0, 2.2), out_mp4=None,
                    out_json=None) -> dict:
    import mujoco
    import numpy as np
    from tool_library.common.mjcf import load, clear_forces, apply_wrench
    from tool_library.common.mechtest import resolve_built, run_headless, report
    from tool_library.skills import construct_latch_box as LB

    built = resolve_built(built, LB.rebuild)
    m = load(built["xml"])
    d = mujoco.MjData(m)
    LB.set_layout(m, d, built, "seated")
    lj = LB._free_adr(m, built["lid_body"])
    d.qpos[lj + 2] += 0.012                  # its own weight does the seating
    mujoco.mj_forward(m, d)
    drivers = {k: LB.LatchDriver(m, d, built, k, force_N=float(close_force_N))
               for k in built["latch_edges"]}
    lid_id = m.body(LB._bname(m, built["lid_body"])).id
    body_id = m.body(LB._bname(m, built["body_body"])).id
    z_start = float(d.xpos[body_id][2])
    total_kg = built["body_mass_kg"] + built["lid_mass_kg"]
    t0, t1, t2, t3, t4 = np.cumsum([float(s) for s in seconds])
    z_lid0 = float(d.xpos[lid_id][2])
    lift_to = z_lid0 + 0.10

    def control(t, mm, dd):
        clear_forces(dd)
        if t < t0:
            return "seat"
        if t < t1:
            for k in built["latch_edges"]:
                drivers[k].push(dd, -1.0)
            return "close"
        if t < t2:
            u = float(np.clip((t - t1) / max(t2 - t1, 1e-6), 0.0, 1.0))
            LB._pd_lift(dd, lid_id, z_lid0 + (lift_to - z_lid0)
                        * (10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5),
                        mass_hint=total_kg)
            return "lift"
        if t < t3:
            LB._pd_lift(dd, lid_id, lift_to, mass_hint=total_kg)
            g = 9.81 * total_kg
            apply_wrench(dd, body_id, force=[0.45 * g * np.sin(5.0 * t),
                                             0.35 * g * np.cos(4.0 * t), 0.0])
            return "shake"
        for k in built["latch_edges"]:
            drivers[k].push(dd, +1.0)
        return "open"

    def snap():
        return {**LB.latch_state(m, d, built), **LB.lid_seating(m, d, built),
                "body_z_m": round(float(d.xpos[body_id][2]), 4)}

    marks = {}
    phase_end = {"seated": t0, "closed": t1, "lifted": t2, "shaken": t3}

    def on_tick(t, mm, dd):
        for tag, tt in phase_end.items():
            if tag not in marks and t >= tt - 0.02:
                marks[tag] = snap()

    run_headless(m, d, control, t4, on_tick=on_tick, tick_every=0.01)
    end = snap()
    n_edges = len(built["latch_edges"])
    rose = round(1000 * (marks.get("lifted", {}).get("body_z_m", z_start) - z_start), 1)
    meas = {
        "after_seat": {k: marks.get("seated", {}).get(k)
                       for k in ("seated", "lid_proud_mm", "n_latched")},
        "after_close": {**{k: marks.get("closed", {}).get(k)
                           for k in ("n_latched", "all_latched", "seated", "lid_proud_mm")},
                        "peak_strap_bow_mm": {k: round(1000 * v.peak_bow, 2)
                                              for k, v in drivers.items()}},
        "after_lift": {"n_latched": marks.get("lifted", {}).get("n_latched"),
                       "body_rose_mm": rose,
                       "lid_proud_mm": marks.get("lifted", {}).get("lid_proud_mm")},
        "after_shake": {k: marks.get("shaken", {}).get(k)
                        for k in ("n_latched", "all_latched", "lid_proud_mm")},
        "after_prise_open": {"n_latched": end["n_latched"],
                             "fold_deg": {k: v["fold_deg"] for k, v in end["latches"].items()}},
        "snap_force_N": built["snap_force_N"],
        "close_force_N": float(close_force_N),
    }
    ok_close = marks.get("closed", {}).get("n_latched", 0) == n_edges
    ok_lift = (marks.get("lifted", {}).get("n_latched", 0) >= n_edges - 1 and rose > 40)
    ok_hold = marks.get("shaken", {}).get("n_latched", 0) >= n_edges - 1
    ok_open = end["n_latched"] == 0
    fails = [t for t, ok in (("did not latch", ok_close),
                             ("the box did not come up with the lid", ok_lift),
                             ("shook loose", ok_hold),
                             ("did not release when prised", ok_open)) if not ok]
    verdict = (f"LATCHES OK — all {n_edges} latch under a {close_force_N:.0f} N pinch "
               f"(snap {built['snap_force_N']:.1f} N), carry the {total_kg:.2f} kg box "
               f"{rose:.0f} mm off the table by the lid alone, survive a shake, and "
               f"release when prised" if not fails
               else "MECHANISM FAILED: " + ", ".join(fails))
    video = None
    if out_mp4:
        video = LB.demo(built, out_mp4, seconds=tuple(seconds),
                        close_force_N=float(close_force_N)).get("video", str(out_mp4))
    return report(not fails, verdict, meas, out_json=out_json, video=video)


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(test_latch_snap(sys.argv[1]), indent=1))


latch_snap_test = test_latch_snap   # the library calls every tool by its own name
