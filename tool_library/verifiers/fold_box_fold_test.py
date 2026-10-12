"""Fold-box fold test: do the creases fold under a push and open under gravity?

Puts a `construct_fold_box` asset on trial with forces only. Each wall and
flap is pushed at its far edge along its own normal (1.2 N, easing off near
the stop), in the order walls, inner flaps, outer flaps; the box is held
closed, then every force is removed and gravity is left to unfold it. A
style="lid" box has one lid hinge, which is swung open by the same push. The
measurement is how far the panels went, the worst panel interpenetration,
and how far the box fell back open with the hands off.

    from tool_library.verifiers.fold_box_fold_test import test_fold_box_fold
    r = test_fold_box_fold(built)              # built = construct_fold_box(...)
"""
from __future__ import annotations

TOOL = {
    "name": "fold_box_fold_test",
    "title": "Fold-box fold test",
    "type": "verifier",
    "category": "Reconstruction Mechanism Tests",
    "summary": "Pushes every crease of a folding carton shut at its panel edge, measures the "
               "fold angles and panel interpenetration, then lets gravity reopen it.",
    "description": (
        "Loads a carton built by construct_fold_box and runs the fold-box driver: rest, push "
        "the walls up, push the inner flaps over, push the outer flaps over, hold, then all "
        "forces off. Pushes are 1.2 N at each panel's far edge along the panel's own normal "
        "and ease off over the last 25 degrees so the joint limit is not slammed. For a "
        "style='lid' box the single lid hinge is swung open instead. Measured: max wall angle, "
        "max flap angle, the worst interpenetration among the box's own panels (contacts of "
        "other objects are ignored), and the angle the walls fell back open by with the hands "
        "off. Passes when the walls (or the lid) reach more than 85 degrees under the push "
        "and the worst panel interpenetration stays under 1.5 mm."
    ),
    "inputs": [
        {"name": "built", "type": "dict | str",
         "description": "The dict construct_fold_box returned, or the path of its <name>_params.json."},
        {"name": "out_mp4", "type": "str | None", "description": "Also render the trial to this video."},
        {"name": "out_json", "type": "str | None", "description": "Write the report here."},
    ],
    "outputs": [
        {"name": "ok", "type": "bool", "description": "True when the panels fold past 85 degrees without interpenetrating."},
        {"name": "verdict", "type": "str", "description": "FOLDS UNDER FORCE / LID OPENS UNDER FORCE / FOLD FAILED with numbers."},
        {"name": "measured", "type": "dict",
         "description": "push_N, max_wall_deg, max_flap_deg, max_penetration_mm, opened_back_deg."},
        {"name": "video", "type": "str | None", "description": "The rendered mp4 when out_mp4 was given."},
    ],
    "status": "tested",
    "entrypoint": "fold_box_fold_test",
    "requirements": ["mujoco", "numpy"],
    "simulators": ["mujoco"],
    "tags": ["mechanism", "fold", "crease", "carton", "giftbox", "force-driven"],
    "used_in": ["Giftbox Packing"],
    "cli": "python -m tool_library.verifiers.fold_box_fold_test <box_params.json>",
}


def test_fold_box_fold(built, out_mp4=None, out_json=None) -> dict:
    import mujoco
    from tool_library.common.mjcf import load, clear_forces
    from tool_library.common.drivers import FoldBoxDriver
    from tool_library.common.mechtest import resolve_built, run_headless, report
    from tool_library.skills import construct_fold_box as FB

    built = resolve_built(built, FB.rebuild)
    m = load(built["xml"])
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    drv = FoldBoxDriver(m, d, built)

    def control(t, model, data):
        clear_forces(data)
        return drv.step(t)

    run_headless(m, d, control, drv.T["open"])
    verdict = drv.verdict()
    video = None
    if out_mp4:
        video = FB.demo(built, out_mp4).get("video", str(out_mp4))
    ok = verdict.startswith(("FOLDS UNDER FORCE", "LID OPENS UNDER FORCE"))
    return report(ok, verdict, drv.meas, out_json=out_json, video=video)


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(test_fold_box_fold(sys.argv[1]), indent=1))


fold_box_fold_test = test_fold_box_fold   # the library calls every tool by its own name
