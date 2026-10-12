"""Thread self-locking test: is a screw cap really a thread?

Puts a `construct_bottle` asset on trial with forces only. A 20 N straight
push and a 20 N straight pull on the cap must NOT move it along the neck; a
0.1 N*m twist must drive it the whole thread length, with lift and turn
coupled by the pitch; when the thread runs out the cap must come free. A cap
that slides under a straight pull, or that only turns part way under the
twist, fails. Nothing is posed: the driver only writes `xfrc_applied`.

    from tool_library.verifiers.thread_self_locking_test import test_thread_self_locking
    r = test_thread_self_locking(built)          # built = construct_bottle(...)
    r["ok"], r["verdict"], r["measured"]
"""
from __future__ import annotations

TOOL = {
    "name": "thread_self_locking_test",
    "title": "Thread self-locking test",
    "type": "verifier",
    "category": "Reconstruction Mechanism Tests",
    "summary": "Force-only trial of a screw cap: straight push/pull must not move it, "
               "a twist must unscrew it along the full thread, then it must come free.",
    "description": (
        "Loads a bottle built by construct_bottle, seats the cap at the top of the thread "
        "(or leaves it as compiled with start='seated'), and runs the bottle driver: settle, "
        "push 20 N straight down, twist 0.1 N*m closed, pull 20 N straight up, twist open "
        "until the thread runs out, lift the free cap away, drop it. Only xfrc_applied is "
        "written. It measures the cap travel without turning, the pitch coupling error "
        "between lift and yaw, the turns applied, and how far the twist actually got before "
        "the thread released. THREAD OK requires under 2 mm of travel under the straight "
        "forces, under 2 mm of coupling error and the twist reaching at least 90% of the "
        "thread. TWIST STALLED means the cap jammed (contact interference or friction); "
        "NOT A THREAD means straight force alone moved the cap. The same trial runs on any "
        "asset that exposes the bottle contract: bodies cap/bottle/carrier, joints yaw/lift, "
        "equalities thread/engaged and total_lift_m."
    ),
    "inputs": [
        {"name": "built", "type": "dict | str",
         "description": "The dict construct_bottle returned, or the path of its "
                        "<name>_params.json (the asset is rebuilt beside it)."},
        {"name": "start", "type": "str",
         "description": "'top' seats the cap at the top of the thread so the closing twist is "
                        "exercised too; 'seated' tests the compiled state (open only). Default 'top'."},
        {"name": "out_mp4", "type": "str | None",
         "description": "Also render the trial to this video with captions (slower)."},
        {"name": "out_json", "type": "str | None", "description": "Write the report here."},
    ],
    "outputs": [
        {"name": "ok", "type": "bool", "description": "True when the verdict is THREAD OK."},
        {"name": "verdict", "type": "str", "description": "THREAD OK / NOT A THREAD / TWIST STALLED with the numbers."},
        {"name": "measured", "type": "dict",
         "description": "push_N, twist_Nm, travel_without_turning_mm, thread_travel_mm, "
                        "turns_applied, coupling_error_mm, lift_at_release_mm, turns_at_release."},
        {"name": "video", "type": "str | None", "description": "The rendered mp4 when out_mp4 was given."},
    ],
    "status": "tested",
    "entrypoint": "thread_self_locking_test",
    "requirements": ["mujoco", "numpy"],
    "simulators": ["mujoco"],
    "tags": ["mechanism", "thread", "bottle", "force-driven", "proof"],
    "used_in": ["Open the Bottle"],
    "cli": "python -m tool_library.verifiers.thread_self_locking_test <bottle_params.json>",
}


def test_thread_self_locking(built, start: str = "top", out_mp4=None,
                             out_json=None, hold_body: bool = True) -> dict:
    """`hold_body`: test a copy with the body fixed when the asset is a free body (default True)."""
    import mujoco
    from pathlib import Path
    from tool_library.common.mjcf import load, clear_forces
    from tool_library.common.drivers import BottleDriver
    from tool_library.common.mechtest import resolve_built, run_headless, report
    from tool_library.skills import construct_bottle as B

    built = resolve_built(built, B.rebuild)
    params = built.get("params")
    if params is None and built.get("params_json") and Path(built["params_json"]).exists():
        import json as _json
        params = _json.loads(Path(built["params_json"]).read_text())
    if hold_body and params is not None and params.get("bottle_fixed") is False:
        # the test applies forces to the cap only and assumes the body is held, as a hand holds it in the video:
        # a free body topples under the twist; rebuild a held copy of the same parameters beside the asset
        p = dict(params); p["bottle_fixed"] = True; p["include_floor"] = True
        built = B.build(p, Path(built["xml"]).parent / "mechtest_held")
    m = load(built["xml"])
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    if start == "top":
        B.seat_cap(m, d, built, float(built["total_lift_m"]))
    elif start != "seated":
        raise ValueError("start must be 'top' or 'seated'")
    drv = BottleDriver(m, d, built)

    def control(t, model, data):
        clear_forces(data)
        return drv.step(t)

    run_headless(m, d, control, drv.T["drop"])
    verdict = drv.verdict()
    video = None
    if out_mp4:
        video = B.demo(built, out_mp4).get("video", str(out_mp4))
    return report(verdict.startswith("THREAD OK"), verdict, drv.meas,
                  out_json=out_json, video=video, start=start)


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(test_thread_self_locking(sys.argv[1]), indent=1))


thread_self_locking_test = test_thread_self_locking   # the library calls every tool by its own name
