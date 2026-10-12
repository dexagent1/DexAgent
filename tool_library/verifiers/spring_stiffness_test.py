"""Spring stiffness test: does a built spring have the stiffness it promises?

A quasi-static force ramp on the tip plate of a `construct_spring` asset,
with every force held until fully settled, gives a force-deflection line
whose slope is the measured N/m; a 10x overload then checks that the
coil-bound stop really stops it. Joint limits, armature and damping are all
in the loop, so this is the compiled model's stiffness, not the algebra's.

    from tool_library.verifiers.spring_stiffness_test import test_spring_stiffness
    r = test_spring_stiffness(built, tolerance_pct=15)
"""
from __future__ import annotations

TOOL = {
    "name": "spring_stiffness_test",
    "title": "Spring stiffness test",
    "type": "verifier",
    "category": "Reconstruction Mechanism Tests",
    "summary": "Quasi-static force ramp on a built spring: measured N/m against the requested "
               "stiffness, plus a 10x overload check of the solid stop.",
    "description": (
        "Loads a spring built by construct_spring, presses its tip plate with each force in "
        "`forces` for 6 s of sim time, reads the compression, fits a line through the points "
        "still on the spring (under 85% of the travel) and reports the slope as the measured "
        "stiffness with its error against the requested value. It then presses with ten times "
        "the largest force and requires the compression to stay within 0.3 mm of the design "
        "travel, which is what a coil-bound stop does and a soft joint limit does not. Passes "
        "when the stiffness error is within `tolerance_pct` and the stop holds."
    ),
    "inputs": [
        {"name": "built", "type": "dict | str",
         "description": "The dict construct_spring returned, or the path of its <name>_params.json."},
        {"name": "forces", "type": "list[float], newtons",
         "description": "The ramp. Default (0.5, 1.0, 1.5, 2.0); scale it to the spring."},
        {"name": "tolerance_pct", "type": "float",
         "description": "Allowed |measured - requested| / requested in percent. Default 15."},
        {"name": "out_json", "type": "str | None", "description": "Write the report here."},
    ],
    "outputs": [
        {"name": "ok", "type": "bool", "description": "Stiffness within tolerance and the solid stop holds."},
        {"name": "verdict", "type": "str", "description": "One sentence with the measured and requested N/m."},
        {"name": "measured", "type": "dict",
         "description": "requested_N_per_m, measured_N_per_m, error_pct, force_deflection [(N, mm)], "
                        "travel_m, solid_stop_m, solid_stop_ok."},
    ],
    "status": "tested",
    "entrypoint": "spring_stiffness_test",
    "requirements": ["mujoco", "numpy"],
    "simulators": ["mujoco"],
    "tags": ["mechanism", "spring", "stiffness", "battery", "force-driven"],
    "used_in": ["Battery Insertion"],
    "cli": "python -m tool_library.verifiers.spring_stiffness_test <spring_params.json>",
}


def test_spring_stiffness(built, forces=(0.5, 1.0, 1.5, 2.0),
                          tolerance_pct: float = 15.0, out_json=None) -> dict:
    import math
    from tool_library.common import spring as S
    from tool_library.common.mechtest import resolve_built, report

    built = resolve_built(built, S.rebuild)
    meas = S.probe(built, forces=tuple(float(f) for f in forces))
    err = meas["error_pct"]
    stiff_ok = (err is not None and not (isinstance(err, float) and math.isnan(err))
                and abs(err) <= float(tolerance_pct))
    ok = bool(stiff_ok and meas["solid_stop_ok"])
    verdict = (f"SPRING OK — measured {meas['measured_N_per_m']:.0f} N/m against "
               f"{meas['requested_N_per_m']:.0f} requested ({err:+.1f}%); the solid "
               f"stop held at {1000*meas['solid_stop_m']:.2f} mm of "
               f"{1000*meas['travel_m']:.2f} mm travel under 10x load"
               if ok else
               f"SPRING SUSPECT — measured {meas['measured_N_per_m']} N/m against "
               f"{meas['requested_N_per_m']:.0f} requested ({err}%), solid stop "
               f"{'held' if meas['solid_stop_ok'] else 'FAILED'} "
               f"({1000*meas['solid_stop_m']:.2f} mm vs {1000*meas['travel_m']:.2f} mm)")
    return report(ok, verdict, meas, out_json=out_json,
                  tolerance_pct=float(tolerance_pct))


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(test_spring_stiffness(sys.argv[1]), indent=1))


spring_stiffness_test = test_spring_stiffness   # the library calls every tool by its own name
