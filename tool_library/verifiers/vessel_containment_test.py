"""Vessel containment test: is a cup or bowl really hollow, and does it hold
what is put in it?

Three gravity-only trials on a `construct_cup` / `construct_bowl` asset (or
any body whose built dict names a rim site and a floor site). Drop: the
vessel is dropped from 2 cm and must land upright on its base. Fill: a ball
narrower than the mouth is released above the rim and must come to rest
inside, below the rim and above the inner floor, within the inner radius.
Nest (optional): a second vessel released above the first must end with
its base below the first's rim, i.e. inside it, not perched on a sealed
hull. A convex-hull cup fails the fill trial at once: the ball rolls off
the invisible lid.

    from tool_library.verifiers.vessel_containment_test import test_vessel_containment
    r = test_vessel_containment(bowl_built, inner_built=cup_built)
"""
from __future__ import annotations

TOOL = {
    "name": "vessel_containment_test",
    "title": "Vessel containment test",
    "type": "verifier",
    "category": "Reconstruction Mechanism Tests",
    "summary": "Drops the vessel (lands upright), drops a ball into it (stays inside, below the "
               "rim) and optionally a second vessel (nests inside); gravity only.",
    "description": (
        "Loads the vessel asset and attaches a ball (radius 0.4 x the inner rim radius by "
        "default, capped at 2 cm) and, with inner_built, the second vessel's XML under a "
        "prefix. Trial 1 lifts the vessel 2 cm and lets it fall for 1.5 s: its z axis must stay "
        "within 15 degrees of vertical and its base within 5 mm of the floor. Trial 2 releases "
        "the ball 3 cm above the rim centre and after 2 s requires it inside: horizontal "
        "distance from the vessel axis under the inner rim radius minus the ball radius, "
        "centre below the rim and above the inner floor. Trial 3 lowers the inner vessel in "
        "(its body origin 1 cm above the outer vessel's floor, on the axis, as a hand sets a cup "
        "into a bowl; dropped from the rim it would tunnel through the base) and after 2 s "
        "requires its body origin more than 5 mm below the outer rim and both vessels within "
        "15 degrees of upright. Reports every number; passes when all requested trials pass."
    ),
    "inputs": [
        {"name": "built", "type": "dict | str",
         "description": "The dict construct_cup/construct_bowl returned, or the path of its <name>_params.json."},
        {"name": "inner_built", "type": "dict | str | None",
         "description": "A second vessel to nest inside (its rim must fit through this one's mouth)."},
        {"name": "ball_radius", "type": "float | None", "description": "Test ball radius; default 0.4 x inner rim radius, max 2 cm."},
        {"name": "out_json", "type": "str | None", "description": "Write the report here."},
    ],
    "outputs": [
        {"name": "ok", "type": "bool", "description": "All trials passed."},
        {"name": "verdict", "type": "str", "description": "One sentence with the measured numbers."},
        {"name": "measured", "type": "dict",
         "description": "drop {tilt_deg, base_height_mm, ok}, fill {ball_off_axis_mm, ball_z_mm, below_rim, above_floor, ok}, "
                        "nest {inner_base_below_rim_mm, tilt_deg, ok} when requested."},
    ],
    "status": "tested",
    "entrypoint": "vessel_containment_test",
    "requirements": ["mujoco", "numpy"],
    "simulators": ["mujoco"],
    "tags": ["mechanism", "cup", "bowl", "hollow", "containment", "pick-and-place"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping"],
    "cli": "python -m tool_library.verifiers.vessel_containment_test <vessel_params.json> [inner_params.json]",
}


def _free_adr(model, body_name):
    import mujoco
    bid = model.body(body_name).id
    for j in range(model.njnt):
        if model.jnt_bodyid[j] == bid and int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
            return int(model.jnt_qposadr[j])
    return None


def test_vessel_containment(built, inner_built=None, ball_radius=None, out_json=None) -> dict:
    import mujoco
    import numpy as np
    from tool_library.common import vessel as V
    from tool_library.common.mechtest import resolve_built, run_headless, report

    built = resolve_built(built, V.rebuild)
    inner = resolve_built(inner_built, V.rebuild) if inner_built is not None else None
    r_in = float(built["inner_rim_radius_m"])
    rb = float(ball_radius) if ball_radius else min(0.02, 0.4 * r_in)
    if rb >= r_in:
        raise ValueError("ball_radius must be smaller than the inner rim radius")
    spec = mujoco.MjSpec.from_file(built["xml"])
    if not any(g.type == mujoco.mjtGeom.mjGEOM_PLANE for g in spec.worldbody.geoms):
        # an asset built for the bench carries no floor; the drop test needs one
        spec.worldbody.add_geom(name="test_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1], rgba=[0.8, 0.8, 0.8, 1])
    ball = spec.worldbody.add_body(name="test_ball", pos=[0, 0, 1.0])
    ball.add_freejoint(name="test_ball_free")
    ball.add_geom(name="test_ball", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[rb, 0, 0],
                  rgba=[0.2, 0.4, 0.9, 1], density=300.0)
    if inner is not None:
        child = mujoco.MjSpec.from_file(inner["xml"])
        for g in list(child.worldbody.geoms):
            if g.name == "floor":
                child.delete(g)
        for lt in list(child.worldbody.lights):
            child.delete(lt)
        fr = spec.worldbody.add_frame()
        fr.pos = [0.4, 0.0, 0.0]            # parked beside, out of the way
        spec.attach(child, prefix="inner_", frame=fr)
    m = spec.compile()
    d = mujoco.MjData(m)
    vb = m.body(built["body"]).id
    vadr = _free_adr(m, built["body"])
    badr = _free_adr(m, "test_ball")
    rim = m.site(built["sites"]["rim"]).id
    floor_site = m.site(built["sites"]["floor"]).id

    def park_ball(z=1.0):
        d.qpos[badr:badr + 3] = [0.5, 0.5, z]
        d.qpos[badr + 3:badr + 7] = [1, 0, 0, 0]

    def tilt_deg(bid):
        return float(np.degrees(np.arccos(np.clip(d.xmat[bid].reshape(3, 3)[2, 2], -1, 1))))

    meas = {}
    # --- 1. drop: lands upright ---
    mujoco.mj_resetData(m, d)
    park_ball()
    if vadr is not None:
        d.qpos[vadr + 2] += 0.02
    mujoco.mj_forward(m, d)
    run_headless(m, d, None, 1.5)
    base_h = float(d.xpos[vb][2])
    meas["drop"] = {"tilt_deg": round(tilt_deg(vb), 2), "base_height_mm": round(1000 * base_h, 2)}
    meas["drop"]["ok"] = bool(meas["drop"]["tilt_deg"] < 15.0 and abs(base_h) < 0.005)
    # --- 2. fill: a ball dropped in stays in ---
    rim_p = d.site_xpos[rim].copy()
    d.qpos[badr:badr + 3] = [rim_p[0], rim_p[1], rim_p[2] + 0.03 + rb]
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    run_headless(m, d, None, 2.0)
    bp = d.xpos[m.body("test_ball").id]
    axis_p, floor_z, rim_z = d.xpos[vb], float(d.site_xpos[floor_site][2]), float(d.site_xpos[rim].copy()[2])
    off = float(np.hypot(bp[0] - axis_p[0], bp[1] - axis_p[1]))
    meas["fill"] = {"ball_radius_mm": round(1000 * rb, 1), "ball_off_axis_mm": round(1000 * off, 2),
                    "ball_z_mm": round(1000 * float(bp[2]), 2), "rim_z_mm": round(1000 * rim_z, 2),
                    "floor_z_mm": round(1000 * floor_z, 2),
                    "below_rim": bool(bp[2] + rb < rim_z + 0.002),
                    "above_floor": bool(bp[2] > floor_z - 0.002),
                    "inside_radius": bool(off < r_in - rb + 0.002),
                    "vessel_tilt_deg": round(tilt_deg(vb), 2)}
    meas["fill"]["ok"] = bool(meas["fill"]["below_rim"] and meas["fill"]["above_floor"]
                              and meas["fill"]["inside_radius"] and meas["fill"]["vessel_tilt_deg"] < 15.0)
    checks = [meas["drop"]["ok"], meas["fill"]["ok"]]
    # --- 3. nest: a second vessel goes in ---
    if inner is not None:
        park_ball()
        ib = m.body("inner_" + inner["body"]).id
        iadr = _free_adr(m, "inner_" + inner["body"])
        if iadr is None:
            raise ValueError("the inner vessel must have a freejoint")
        # LOWERED IN, NOT DROPPED FROM THE RIM: a hand lowers a cup into a
        # bowl. Released at the rim it falls 10 cm, hits the 3 mm base disk at
        # 1.5 m/s (= 3 mm per 2 ms step) and tunnels straight through it
        # (measured: the inner cup ended on the floor under the outer one).
        rim_p = d.site_xpos[rim].copy()
        fl = d.site_xpos[floor_site].copy()
        d.qpos[iadr:iadr + 3] = [rim_p[0], rim_p[1], fl[2] + 0.01]
        d.qpos[iadr + 3:iadr + 7] = [1, 0, 0, 0]
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        run_headless(m, d, None, 2.0)
        rim_z = float(d.site_xpos[rim][2])
        in_base = float(d.xpos[ib][2])
        meas["nest"] = {"inner_base_below_rim_mm": round(1000 * (rim_z - in_base), 2),
                        "inner_tilt_deg": round(tilt_deg(ib), 2),
                        "outer_tilt_deg": round(tilt_deg(vb), 2)}
        meas["nest"]["ok"] = bool(rim_z - in_base > 0.005 and meas["nest"]["inner_tilt_deg"] < 15.0
                                  and meas["nest"]["outer_tilt_deg"] < 15.0)
        checks.append(meas["nest"]["ok"])
    ok = all(checks)
    f = meas["fill"]
    verdict = (f"VESSEL OK — landed upright ({meas['drop']['tilt_deg']:.1f} deg), a "
               f"{f['ball_radius_mm']:.0f} mm ball came to rest {f['ball_off_axis_mm']:.1f} mm off axis "
               f"at z {f['ball_z_mm']:.1f} mm, under the {f['rim_z_mm']:.0f} mm rim"
               + (f", and the inner vessel's base sits {meas['nest']['inner_base_below_rim_mm']:.1f} mm "
                  f"below the rim" if inner is not None else "") if ok else
               "VESSEL FAILED: " + ", ".join(k for k in ("drop", "fill", "nest") if k in meas and not meas[k]["ok"]))
    return report(ok, verdict, meas, out_json=out_json)


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(test_vessel_containment(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None), indent=1))


vessel_containment_test = test_vessel_containment   # the library calls every tool by its own name
