"""Tube seat test: do the tubes in a rack stay seated when the rack is
shaken and tilted, and does a dropped tube find its well?

A headless version of the proof `construct_tube_rack` ships with. The loaded
rack is held by a force grip, shaken sideways at 1.5 Hz with growing
amplitude, tilted 15 degrees and back, and set down; a spare tube is dropped
from 30 mm above an empty socket well at the same time. Seated is measured
in the rack's own frame: lateral offset from the well under 4 mm and the
tube upright, so a rack that slid during the shake does not fail its tubes.

    from tool_library.verifiers.tube_seat_test import test_tube_seat
    r = test_tube_seat(built)                      # built = construct_tube_rack(... tubes=[...])
"""
from __future__ import annotations

TOOL = {
    "name": "tube_seat_test",
    "title": "Tube seat test",
    "type": "verifier",
    "category": "Reconstruction Mechanism Tests",
    "summary": "Shakes and tilts a loaded tube rack by force and drops a spare tube into an empty "
               "well; every tube must end seated in its own well.",
    "description": (
        "Rebuilds the rack with a floor, adds one spare tube 30 mm above an empty socket well, "
        "and holds the rack body with a spring-damper force grip: settle 1 s, shake laterally "
        "at 1.5 Hz up to 9 mm for 4 s, tilt 15 degrees and back over 2 s, then hold still. "
        "Afterwards every seated tube and the dropped tube are checked with tube_seated(): "
        "lateral offset from their well centre under 4 mm and upright (cosine over 0.9), "
        "measured in the rack frame. Requires at least one seated tube in the built dict."
    ),
    "inputs": [
        {"name": "built", "type": "dict | str",
         "description": "The dict construct_tube_rack returned (with tubes), or the path of its <name>_params.json."},
        {"name": "seconds", "type": "float", "description": "Trial length. Default 10."},
        {"name": "out_mp4", "type": "str | None", "description": "Also render the trial to this video."},
        {"name": "out_json", "type": "str | None", "description": "Write the report here."},
    ],
    "outputs": [
        {"name": "ok", "type": "bool", "description": "Every tube, including the dropped one, is seated at the end."},
        {"name": "verdict", "type": "str", "description": "PASS/FAIL sentence."},
        {"name": "measured", "type": "dict",
         "description": "tubes_after_shake [{lateral_off_mm, upright_cos, seated}], dropped_tube {...}."},
        {"name": "video", "type": "str | None", "description": "The rendered mp4 when out_mp4 was given."},
    ],
    "status": "tested",
    "entrypoint": "tube_seat_test",
    "requirements": ["mujoco", "numpy"],
    "simulators": ["mujoco"],
    "tags": ["mechanism", "rack", "microtube", "seat", "biology", "force-driven"],
    "used_in": ["Biology Experiment"],
    "cli": "python -m tool_library.verifiers.tube_seat_test <rack_params.json>",
}


def test_tube_seat(built, seconds: float = 10.0, out_mp4=None, out_json=None) -> dict:
    import json
    from pathlib import Path
    import mujoco
    import numpy as np
    from tool_library.common import labtube as LT
    from tool_library.common.mjcf import load, clear_forces, ForceGrip, merge_params
    from tool_library.common.mechtest import resolve_built, run_headless, report

    built = resolve_built(built, LT.rebuild)
    p = json.loads(Path(built["params_json"]).read_text())
    if not built["tubes"]:
        raise ValueError("tube_seat_test needs a rack built with at least one seated tube")
    occupied = {(t["row"], t["col"]) for t in built["tubes"]}
    empty = [(r, c) for r, c in built["socket_wells"] if (r, c) not in occupied]
    if empty:
        drop_rc = empty[0]
        p2 = dict(p, include_floor=True)
    else:
        drop_rc = next((r, c) for r in range(int(p["rows"])) for c in range(int(p["cols"]))
                       if (r, c) not in occupied)
        p2 = dict(p, include_floor=True,
                  open_sockets=list(p["open_sockets"]) + [list(drop_rc)])
    x, y = LT.rack_well_xy(p, *drop_rc)
    tmp = Path(built["xml"]).parent / "seat_test"
    rk = LT.build_tube_rack(p2, tmp)
    tp = merge_params(LT.TUBE_DEFAULTS, dict(p["tube_params"], include_floor=False))
    drop = LT._tube_body_xml(tp, f"{p['name']}_drop", [x, y, rk["plate_top_m"] + 0.030],
                             0.0, 0.0, f"{p['name']}_tubelathe_m")
    xml_p = Path(rk["xml"])
    xml_p.write_text(xml_p.read_text().replace("  </worldbody>", f"{drop}\n  </worldbody>"))
    m = load(xml_p)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    rb = m.body(rk["rack_body"]).id
    grip = ForceGrip(m, rb, kp=260.0, kd=22.0, max_force=60.0, kr=4.0, kdr=0.35, max_torque=4.0)
    p0 = d.xpos[rb].copy()
    q0 = d.xquat[rb].copy()

    def control(t, mm, dd):
        clear_forces(dd)
        if t < 1.0:
            return "settle"
        if t < 5.0:
            a = 0.009 * min(1.0, (t - 1.0) / 1.0)
            dx = a * np.sin(2 * np.pi * 1.5 * (t - 1.0))
            grip.hold(dd, target_pos=p0 + [dx, 0, 0], target_quat=q0)
            return "shake"
        if t < 7.0:
            u = np.sin(np.pi * (t - 5.0) / 2.0)
            th = np.radians(15.0 * u)
            qt = np.array([np.cos(th / 2), 0, np.sin(th / 2), 0])
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, qt, q0)
            grip.hold(dd, target_pos=p0 + [0, 0, 0.004 * u], target_quat=q)
            return "tilt"
        grip.hold(dd, target_pos=p0, target_quat=q0)
        return "set down"

    run_headless(m, d, control, float(seconds))
    seats = [LT.tube_seated(m, d, rk, t) for t in rk["tubes"]]
    dseat = LT.tube_seated(m, d, rk, {"body": f"{p['name']}_drop",
                                      "row": drop_rc[0], "col": drop_rc[1]})
    ok = all(s["seated"] for s in seats) and dseat["seated"]
    meas = {"tubes_after_shake": seats, "dropped_tube": dseat,
            "n_tubes": len(seats), "drop_well": list(drop_rc)}
    verdict = (f"RACK OK — all {len(seats)} tubes rode out a 9 mm / 1.5 Hz shake and a 15 deg "
               f"tilt (worst offset {max(s['lateral_off_mm'] for s in seats):.1f} mm) and the "
               f"dropped tube self-seated {dseat['lateral_off_mm']:.1f} mm off its well centre"
               if ok else
               f"RACK FAILED — {sum(not s['seated'] for s in seats)} of {len(seats)} tubes unseated"
               f"{'' if dseat['seated'] else ', the dropped tube did not seat'}")
    video = None
    if out_mp4:
        video = LT.demo_rack(built, out_mp4, seconds=float(seconds)).get("video", str(out_mp4))
    return report(ok, verdict, meas, out_json=out_json, video=video)


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(test_tube_seat(sys.argv[1]), indent=1))


tube_seat_test = test_tube_seat   # the library calls every tool by its own name
