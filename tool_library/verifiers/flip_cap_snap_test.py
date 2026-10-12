"""Flip-cap snap test: does a microtube's hinged cap pop open under a pry and
snap shut again under a press?

A headless version of the proof `construct_microtube` ships with. The tube
stands in a one-well rack (the holder a real tube needs); a constant 10 N
press pins it, as a thumb does. A pry force ramps up under the cap's tab
until the cap pops past its bead (the force at which it popped is recorded),
the cap is left to fall open, then an 8 N press on the tab closes it and the
bead must catch again. Forces only; no joint is written.

    from tool_library.verifiers.flip_cap_snap_test import test_flip_cap_snap
    r = test_flip_cap_snap(built)                  # built = construct_microtube(...)
"""
from __future__ import annotations

TOOL = {
    "name": "flip_cap_snap_test",
    "title": "Flip-cap snap test",
    "type": "verifier",
    "category": "Reconstruction Mechanism Tests",
    "summary": "Pries a microtube's flip cap open with a ramped force under its tab and presses "
               "it shut again; measures the pop force, the opening and the reclosed angle.",
    "description": (
        "Rebuilds the tube from its parameters into a one-well rack so it has a holder, pins "
        "the tube with a constant 10 N downward force, and ramps a pry force under the cap "
        "tab (3 N/s up to 7 N) until the cap fold angle passes 6 degrees; the force at that "
        "moment is the measured pop force (compare with built['snap_force_N']). The cap is "
        "left to fall open, then a press_force_N press at the tab, where a thumb lands, closes it. "
        "Passes when the cap opened past 80 degrees and reclosed to within 8 degrees of shut. "
        "The same trial works on any cap that exposes <tube>_fold and <tube>_flex joints, a "
        "<tube>_tab geom and a <tube>_capface body."
    ),
    "inputs": [
        {"name": "built", "type": "dict | str",
         "description": "The dict construct_microtube returned, or the path of its <name>_params.json."},
        {"name": "pry_force_N", "type": "float", "description": "Peak of the pry ramp under the tab. Default 7 (a thumbnail)."},
        {"name": "press_force_N", "type": "float", "description": "The closing press on the tab. Default 8."},
        {"name": "seconds", "type": "float", "description": "Trial length. Default 9."},
        {"name": "out_mp4", "type": "str | None", "description": "Also render the trial to this video."},
        {"name": "out_json", "type": "str | None", "description": "Write the report here."},
    ],
    "outputs": [
        {"name": "ok", "type": "bool", "description": "Opened past 80 degrees and reclosed within 8 degrees."},
        {"name": "verdict", "type": "str", "description": "PASS/FAIL with the angles."},
        {"name": "measured", "type": "dict",
         "description": "popped_at_N, max_open_deg, reclosed_deg, end_state {fold_deg, flex_mm}, snap_force_designed_N."},
        {"name": "video", "type": "str | None", "description": "The rendered mp4 when out_mp4 was given."},
    ],
    "status": "tested",
    "entrypoint": "flip_cap_snap_test",
    "requirements": ["mujoco", "numpy"],
    "simulators": ["mujoco"],
    "tags": ["mechanism", "microtube", "flip cap", "snap", "biology", "force-driven"],
    "used_in": ["Biology Experiment"],
    "cli": "python -m tool_library.verifiers.flip_cap_snap_test <tube_params.json>",
}


def test_flip_cap_snap(built, pry_force_N: float = 7.0, press_force_N: float = 8.0,
                       seconds: float = 9.0, out_mp4=None, out_json=None) -> dict:
    import json
    from pathlib import Path
    import mujoco
    import numpy as np
    from tool_library.common import labtube as LT
    from tool_library.common.mjcf import load, clear_forces, apply_wrench
    from tool_library.common.mechtest import resolve_built, run_headless, report

    built = resolve_built(built, LT.rebuild)
    tp = json.loads(Path(built["params_json"]).read_text())
    keep = {"total_len", "body_r", "cone_len", "tip_r", "rim_r", "rim_t", "bore_r",
            "cap_r", "cap_t", "plug_r", "plug_depth", "bead_out", "nub_in",
            "flex_travel", "flex_stiffness", "tube_rgba", "cap_rgba"}
    rp = {"name": "captest1w", "rows": 1, "cols": 1, "pitch": 0.016,
          "length": 0.05, "width": 0.05, "height": 0.045,
          "tubes": [{"row": 0, "col": 0, "cap_deg": 0.0}],
          "tube_params": {k: v for k, v in tp.items() if k in keep},
          "rack_free": False}
    rk = LT.build_tube_rack(rp, Path(built["xml"]).parent / "cap_test")
    m = load(rk["xml"])
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    tube = rk["tubes"][0]
    tb = m.body(tube["body"]).id
    cb = m.body(f"{tube['body']}_capface").id
    tab_g = m.geom(f"{tube['body']}_tab").id
    fold_a = LT._jadr(m, f"{tube['body']}_fold")
    log = {"popped_at_N": None, "max_open_deg": 0.0, "reclosed_deg": None}

    def control(t, mm, dd):
        clear_forces(dd)
        fold = float(np.degrees(dd.qpos[fold_a]))
        log["max_open_deg"] = max(log["max_open_deg"], fold)
        apply_wrench(dd, tb, force=[0.0, 0.0, -10.0])        # the holding thumb
        cz = dd.xmat[cb].reshape(3, 3)[:, 2]
        if t < 4.0:
            f = min(float(pry_force_N), float(pry_force_N) * t / 2.33)   # ramp over ~2.3 s
            if fold < 100.0:
                apply_wrench(dd, cb, force=f * cz, point=dd.geom_xpos[tab_g],
                             com=dd.xpos[cb])
                if log["popped_at_N"] is None and fold > 6.0:
                    log["popped_at_N"] = round(f, 2)
            return "pry"
        if t < 5.5:
            return "released"
        if t < 8.2:
            apply_wrench(dd, cb, force=-float(press_force_N) * cz,
                         point=dd.geom_xpos[tab_g], com=dd.xpos[cb])
            return "press shut"
        log["reclosed_deg"] = round(fold, 1)
        return "closed"

    run_headless(m, d, control, float(seconds))
    end = LT.cap_state(m, d, tube["body"])
    ok = (log["max_open_deg"] > 80.0 and log["reclosed_deg"] is not None
          and abs(log["reclosed_deg"]) < 8.0)
    meas = {**log, "end_state": end, "snap_force_designed_N": built["snap_force_N"]}
    verdict = (f"CAP OK — popped at {log['popped_at_N']} N (designed "
               f"{built['snap_force_N']:.1f} N), opened to {log['max_open_deg']:.0f} deg "
               f"and snapped shut again at {log['reclosed_deg']} deg" if ok else
               f"CAP FAILED — popped_at {log['popped_at_N']} N, max_open "
               f"{log['max_open_deg']:.0f} deg, reclosed {log['reclosed_deg']}")
    video = None
    if out_mp4:
        video = LT.demo_microtube(built, out_mp4, seconds=float(seconds)).get("video", str(out_mp4))
    return report(ok, verdict, meas, out_json=out_json, video=video)


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(test_flip_cap_snap(sys.argv[1]), indent=1))


flip_cap_snap_test = test_flip_cap_snap   # the library calls every tool by its own name
