"""Plunger two-stage test: does a micropipette's plunger have a first stop
and a blow-out stage, and does it spring back?

Holds a `construct_micropipette` upright with a force grip (the hand) and
presses the plunger with a thumb force along the barrel: a ramp to 4 N must
take it to the first stop (the sprung stroke, ~12 mm) and no further, because
the blow-out pin's 5 N preload has to yield first; a ramp on to 22 N must
push it through the second stage (a further ~5.5 mm); easing the thumb off
must let the springs return it to within 2 mm of the top. Forces only; the
plunger joint is never written.

    from tool_library.verifiers.plunger_two_stage_test import test_plunger_two_stage
    r = test_plunger_two_stage(built)              # built = construct_micropipette(...)
"""
from __future__ import annotations

TOOL = {
    "name": "plunger_two_stage_test",
    "title": "Plunger two-stage test",
    "type": "verifier",
    "category": "Reconstruction Mechanism Tests",
    "summary": "Thumb-force ramp on a micropipette plunger: first stop at the sprung stroke, "
               "blow-out past it under more force, spring return on release.",
    "description": (
        "Rebuilds the pipette upright on a floor, holds its body in place with a sized "
        "force grip, and applies a thumb force on the plunger body along the barrel axis in "
        "three phases: ramp to 4 N over 2.5 s and hold (the plunger must reach at least 70% "
        "of stroke1 and at most stroke1 + 1 mm: the sprung stage), ramp to 22 N and hold "
        "(the plunger must pass stroke1 + 2 mm: the blow-out stage behind the preloaded "
        "pin), then ease off over 0.8 s (the plunger must return to within 2 mm of the top). "
        "Reports the depth reached at each stage and the designed stroke numbers from the "
        "built dict."
    ),
    "inputs": [
        {"name": "built", "type": "dict | str",
         "description": "The dict construct_micropipette returned, or the path of its <name>_params.json."},
        {"name": "first_stage_N", "type": "float", "description": "Thumb force for the first stage. Default 4."},
        {"name": "second_stage_N", "type": "float", "description": "Thumb force for the blow-out. Default 22."},
        {"name": "out_json", "type": "str | None", "description": "Write the report here."},
    ],
    "outputs": [
        {"name": "ok", "type": "bool", "description": "First stop, blow-out and return all within their bands."},
        {"name": "verdict", "type": "str", "description": "One sentence with the three depths."},
        {"name": "measured", "type": "dict",
         "description": "first_stop {N, mm}, second_stop {N, mm}, released_mm, stroke1_mm, stroke2_mm, "
                        "first_stop_N (designed), second_stage_step_N (designed)."},
    ],
    "status": "tested",
    "entrypoint": "plunger_two_stage_test",
    "requirements": ["mujoco", "numpy"],
    "simulators": ["mujoco"],
    "tags": ["mechanism", "micropipette", "plunger", "spring", "biology", "force-driven"],
    "used_in": ["Biology Experiment"],
    "cli": "python -m tool_library.verifiers.plunger_two_stage_test <pipette_params.json>",
}


def test_plunger_two_stage(built, first_stage_N: float = 4.0,
                           second_stage_N: float = 22.0, out_json=None) -> dict:
    import json
    from pathlib import Path
    import mujoco
    from tool_library.common import pipettekit as PK
    from tool_library.common.mjcf import load, clear_forces, apply_wrench, ForceGrip
    from tool_library.common.mechtest import resolve_built, run_headless, report

    built = resolve_built(built, PK.rebuild)
    pp = json.loads(Path(built["params_json"]).read_text())
    up = PK.build_micropipette(dict(pp, layout="upright", pos=[0.0, 0.0, 0.05],
                                    include_floor=True),
                               Path(built["xml"]).parent / "plunger_test")
    m = load(up["xml"])
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    name = up["name"]
    pb = m.body(name).id
    plb = m.body(f"{name}_plunger").id
    ja = m.joint(f"{name}_plunge").qposadr[0]
    grip = ForceGrip.sized(m, pb, stiff=2.0, max_force=28.0, max_torque=1.2)
    p0 = d.xpos[pb].copy()
    q0 = d.xquat[pb].copy()
    F1, F2 = float(first_stage_N), float(second_stage_N)
    T1, T2, T3 = 3.5, 6.5, 8.0
    log = {"first_stop": None, "second_stop": None, "released_mm": None,
           "max_mm_stage1": 0.0}

    def control(t, mm, dd):
        clear_forces(dd)
        grip.hold(dd, target_pos=p0, target_quat=q0)
        axis_dn = -dd.xmat[pb].reshape(3, 3)[:, 2]
        q = float(dd.qpos[ja])
        if t < T1:
            f = min(F1, F1 * t / 2.5)
            apply_wrench(dd, plb, force=f * axis_dn)
            if t > 1.0:
                log["max_mm_stage1"] = max(log["max_mm_stage1"], 1000 * q)
            if t > T1 - 0.05:
                log["first_stop"] = {"N": round(f, 2), "mm": round(1000 * q, 2)}
            return "stage 1"
        if t < T2:
            f = min(F2, F1 + (F2 - F1) * (t - T1) / 2.0)
            apply_wrench(dd, plb, force=f * axis_dn)
            if t > T2 - 0.05:
                log["second_stop"] = {"N": round(f, 2), "mm": round(1000 * q, 2)}
            return "stage 2"
        f = max(0.0, F2 * (1.0 - (t - T2) / 0.8))
        if f > 0:
            apply_wrench(dd, plb, force=f * axis_dn)
        if t > T3 - 0.05:
            log["released_mm"] = round(1000 * q, 2)
        return "release"

    run_headless(m, d, control, T3)
    s1, s2 = float(built["stroke1_mm"]), float(built["stroke2_mm"])
    fs, ss = log["first_stop"], log["second_stop"]
    ok_first = fs is not None and 0.7 * s1 <= fs["mm"] <= s1 + 1.0
    ok_second = ss is not None and ss["mm"] > s1 + 2.0
    ok_return = log["released_mm"] is not None and log["released_mm"] < 2.0
    ok = bool(ok_first and ok_second and ok_return)
    meas = {**log, "stroke1_mm": s1, "stroke2_mm": s2,
            "first_stop_N_designed": built["first_stop_N"],
            "second_stage_step_N_designed": built["second_stage_step_N"],
            "checks": {"first_stop": ok_first, "blow_out": ok_second, "spring_return": ok_return}}
    verdict = (f"PLUNGER OK — {F1:.0f} N took it {fs['mm']:.1f} mm (first stop at {s1:.1f}), "
               f"{F2:.0f} N took it {ss['mm']:.1f} mm (blow-out {s2:.1f} further), and it "
               f"returned to {log['released_mm']:.1f} mm" if ok else
               f"PLUNGER FAILED — first stop {fs}, second stop {ss}, released {log['released_mm']} "
               f"(designed {s1:.1f} + {s2:.1f} mm)")
    return report(ok, verdict, meas, out_json=out_json)


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(test_plunger_two_stage(sys.argv[1]), indent=1))


plunger_two_stage_test = test_plunger_two_stage   # the library calls every tool by its own name
