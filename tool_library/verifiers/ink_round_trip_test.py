"""Ink round-trip test: does a whiteboard take ink only where a nib touches
it, and give it up where a pad wipes?

Exercises a `construct_whiteboard` asset's live texture through its own
InkSurface: a nib dragged ACROSS the face in contact must leave a mark; the
same stroke commanded 5 mm above the face must leave nothing (a plan that
says "write" with the nib in the air must not draw); a second stroke that
jumps across the board must not be joined by a line; a pad wiped over the
mark must clear it; `darkness()` reads each of these off the texture as a
number. The test also checks the texture round-trips into a rendered frame
when a renderer is available.

    from tool_library.verifiers.ink_round_trip_test import test_ink_round_trip
    r = test_ink_round_trip(built)                 # built = construct_whiteboard(...)
"""
from __future__ import annotations

TOOL = {
    "name": "ink_round_trip_test",
    "title": "Ink round-trip test",
    "type": "verifier",
    "category": "Reconstruction Mechanism Tests",
    "summary": "A nib in contact marks the board, a nib in the air does not, a jump is not "
               "joined, and a pad wipe clears the mark; all read off the live texture.",
    "description": (
        "Loads a board built by construct_whiteboard and drives its InkSurface with world "
        "points: a 60% width stroke on the face (gap 0) must raise the darkness of its "
        "bounding rectangle above min_ink; the same stroke at `air_gap_m` above the face "
        "must leave the darkness under max_ghost; two dabs far apart on the board must not "
        "produce a connecting line (the midpoint rectangle stays clean); a pad wipe over "
        "the first stroke must bring its rectangle back under max_ghost. With render=True "
        "the texture is uploaded to a MuJoCo renderer and the frame's dark pixel count over "
        "the board must change between the inked and wiped states. Works on any asset that "
        "exposes surface_geom and texture names in its built dict."
    ),
    "inputs": [
        {"name": "built", "type": "dict | str",
         "description": "The dict construct_whiteboard returned, or the path of its <name>_params.json."},
        {"name": "air_gap_m", "type": "float", "description": "Height of the no-contact stroke. Default 0.005."},
        {"name": "min_ink", "type": "float", "description": "Darkness a contact stroke must reach. Default 0.05."},
        {"name": "max_ghost", "type": "float", "description": "Darkness allowed after an air stroke or a wipe. Default 0.005."},
        {"name": "wipe_passes", "type": "int",
         "description": "Back-and-forth passes of the pad over the stroke. Default 20 (one pass takes 16% of the ink)."},
        {"name": "render", "type": "bool", "description": "Also check the texture reaches a rendered frame. Default True."},
        {"name": "out_json", "type": "str | None", "description": "Write the report here."},
    ],
    "outputs": [
        {"name": "ok", "type": "bool", "description": "All four texture checks (and the render check) passed."},
        {"name": "verdict", "type": "str", "description": "One sentence with the darkness numbers."},
        {"name": "measured", "type": "dict",
         "description": "darkness after contact stroke / air stroke / jump / wipe, and the rendered dark pixel counts."},
    ],
    "status": "tested",
    "entrypoint": "ink_round_trip_test",
    "requirements": ["mujoco", "numpy"],
    "simulators": ["mujoco"],
    "tags": ["mechanism", "whiteboard", "ink", "texture", "drawing", "wipe"],
    "used_in": ["Drawing and Wiping"],
    "cli": "python -m tool_library.verifiers.ink_round_trip_test <board_params.json>",
}


def test_ink_round_trip(built, air_gap_m: float = 0.005, min_ink: float = 0.05,
                        max_ghost: float = 0.005, wipe_passes: int = 20,
                        render: bool = True, out_json=None) -> dict:
    import mujoco
    import numpy as np
    from tool_library.common.ink import InkSurface
    from tool_library.common.mjcf import load
    from tool_library.common.mechtest import resolve_built, report
    from tool_library.skills import construct_whiteboard as WB

    built = resolve_built(built, WB.rebuild)
    m = load(built["xml"])
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    ink = InkSurface(m, built)
    hx, hy = ink.hx, ink.hy
    R = np.asarray(d.geom_xmat[ink.gid], float).reshape(3, 3)
    c = np.asarray(d.geom_xpos[ink.gid], float)

    def world(x, y, gap=0.0):
        return c + R @ np.array([x, y, ink.half_t + gap])

    rad = 0.0022
    xs = np.linspace(-0.6 * hx, 0.6 * hx, 60)
    y0 = 0.2 * hy
    box = (-0.6 * hx - 2 * rad, 0.6 * hx + 2 * rad, y0 - 2 * rad, y0 + 2 * rad)
    meas = {}
    # 1. contact stroke marks
    ink.lift("pen")
    for x in xs:
        ink.stroke(d, world(x, y0), radius_m=rad)
    meas["after_contact_stroke"] = ink.darkness(*box)
    # render check: the ink must reach the frame
    if render:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = c
        cam.distance = max(hx, hy) * 2.6
        cam.azimuth, cam.elevation = 90.0, -89.0
        with mujoco.Renderer(m, height=240, width=320) as r:
            ink.upload(r)
            r.update_scene(d, camera=cam)
            inked = int((r.render().mean(2) < 100).sum())
            meas["rendered_dark_px_inked"] = inked
    # 2. the same stroke in the air marks nothing new
    ink.reset()
    ink.lift("pen")
    for x in xs:
        ink.stroke(d, world(x, y0, float(air_gap_m)), radius_m=rad)
    meas["after_air_stroke"] = ink.darkness(*box)
    # 3. two dabs far apart are not joined
    ink.reset()
    ink.lift("pen")
    ink.stroke(d, world(-0.7 * hx, -0.5 * hy), radius_m=rad)
    ink.stroke(d, world(0.7 * hx, -0.5 * hy), radius_m=rad)
    meas["after_jump_midline"] = ink.darkness(-0.4 * hx, 0.4 * hx,
                                              -0.5 * hy - 2 * rad, -0.5 * hy + 2 * rad)
    # 4. a wipe clears a stroke
    ink.reset()
    ink.lift("pen")
    for x in xs:
        ink.stroke(d, world(x, y0), radius_m=rad)
    before = ink.darkness(*box)
    # a pad wiped BACK AND FORTH: one pass takes 16% of the ink (the asset's
    # own efficiency, matched to the footage), so the mark fades over passes
    for sweep in range(int(wipe_passes)):
        xs_w = np.linspace(-0.66 * hx, 0.66 * hx, 12)
        for x in (xs_w if sweep % 2 == 0 else xs_w[::-1]):
            ink.wipe(d, world(x, y0), R, 0.09 * hx, 4 * rad)
    meas["before_wipe"] = before
    meas["after_wipe"] = ink.darkness(*box)
    if render:
        with mujoco.Renderer(m, height=240, width=320) as r:
            ink.upload(r)
            r.update_scene(d, camera=cam)
            meas["rendered_dark_px_wiped"] = int((r.render().mean(2) < 100).sum())
    checks = {
        "contact_marks": meas["after_contact_stroke"] >= float(min_ink),
        "air_does_not_mark": meas["after_air_stroke"] <= float(max_ghost),
        "jump_not_joined": meas["after_jump_midline"] <= float(max_ghost),
        "wipe_clears": meas["after_wipe"] <= float(max_ghost) and before >= float(min_ink),
    }
    if render:
        checks["render_round_trip"] = meas["rendered_dark_px_inked"] > meas["rendered_dark_px_wiped"]
    meas["checks"] = checks
    ok = all(checks.values())
    verdict = (f"INK OK — a contact stroke darkened its band to {meas['after_contact_stroke']:.3f}, "
               f"the same stroke {1000*air_gap_m:.0f} mm up left {meas['after_air_stroke']:.3f}, a jump "
               f"left {meas['after_jump_midline']:.3f} between the dabs, and a wipe took the band from "
               f"{before:.3f} to {meas['after_wipe']:.3f}" if ok else
               "INK FAILED: " + ", ".join(k for k, v in checks.items() if not v))
    return report(ok, verdict, meas, out_json=out_json)


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(test_ink_round_trip(sys.argv[1]), indent=1))


ink_round_trip_test = test_ink_round_trip   # the library calls every tool by its own name
