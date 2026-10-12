"""Scissors cut test: do the blades open on a finger force and part a cord?

Assembles a `construct_scissors` pair with a cuttable cord laid across the
open blades and drives the pair with forces on the rings only (0.7 N, then
1.4 and 2.8 N if the cord has not parted). The cord is one rope authored as two cables tied by a CONNECT
equality, which `try_sever` deactivates only when six measured conditions
hold at once: the join is inside the blades' span, inside the wedge, in the
blade plane, on the blade line, the blades are closing, and both blades are
biting into it with the edges closed to within the cord's own diameter.
Without a cord the test checks that the pair opens to at least 60% of its
range on the ring force and shuts again.

    from tool_library.verifiers.scissors_cut_test import test_scissors_cut
    r = test_scissors_cut(built)                   # built = construct_scissors(...)
    r = test_scissors_cut(built, cord=False)       # open/shut only
"""
from __future__ import annotations

TOOL = {
    "name": "scissors_cut_test",
    "title": "Scissors cut test",
    "type": "verifier",
    "category": "Reconstruction Mechanism Tests",
    "summary": "Pushes the rings of a built pair of scissors with a capped force and checks "
               "that the blades open, shut, and really sever a cord laid in the wedge.",
    "description": (
        "Builds a trial scene from the scissors asset: the pair on the floor with its blades "
        "set open, and (with cord=True) a cuttable rope laid across the wedge perpendicular "
        "to its bisector at `cut_frac` of the blade length, the rope's cut join at the "
        "crossing. The scissors driver then holds the tool in place with a gentle in-plane "
        "grip and squeezes the upper ring with 0.7 N, retrying at 1.4 and 2.8 N while the cord "
        "has not parted (ring_force_N fixes one force); nothing writes the hinge. Every "
        "step try_sever measures the join against the blades and deactivates the join's "
        "equality only when the join is in the blades' span, in the wedge, in their plane, "
        "within cut_catch_m of the blade line, the hinge is closing, and both blades bite "
        "at least cut_pinch_mm with the edges closed to within the cord's diameter. Reports "
        "the opening reached, the smallest edge gap at the cord, and the cut record (blade "
        "gap and distance from the pivot at the moment it parted). With cord=False the pair "
        "starts shut, is opened by the ring force and must reach 60% of its range."
    ),
    "inputs": [
        {"name": "built", "type": "dict | str",
         "description": "The dict construct_scissors returned, or the path of its <name>_params.json."},
        {"name": "cord", "type": "bool",
         "description": "Lay a cuttable cord in the wedge and require the cut. Default True."},
        {"name": "cord_params", "type": "dict | None",
         "description": "construct_rope parameters for the cord (radius, rgba, stiffness...). "
                        "Default: a 20 cm cord of radius 2.5 mm."},
        {"name": "cut_frac", "type": "float",
         "description": "Where along the blade (0 pivot, 1 tip) the cord crosses. Default 0.55."},
        {"name": "ring_force_N", "type": "float | None",
         "description": "Cap on the finger force at the ring. Default: escalate 0.7, 1.4, 2.8 N "
                        "until the cord parts (longer blades put less force at the cord; a 7 mm "
                        "cord jams plate blades at 0.7 N). The force it took is reported."},
        {"name": "out_mp4", "type": "str | None", "description": "Also render the trial to this video."},
        {"name": "out_json", "type": "str | None", "description": "Write the report here."},
    ],
    "outputs": [
        {"name": "ok", "type": "bool", "description": "CUT with a cord, SCISSORS OK without one."},
        {"name": "verdict", "type": "str", "description": "CUT / NO CUT / SCISSORS OK / SCISSORS SUSPECT with numbers."},
        {"name": "measured", "type": "dict",
         "description": "max_open_deg, min_gap_at_join_mm, ring_force_N (the force that cut), "
                        "closed_deg, cut [records], attempts, and the per-gate sever_report at the "
                        "end when nothing was cut."},
        {"name": "scene_xml", "type": "str", "description": "The assembled trial scene, for inspection."},
    ],
    "status": "tested",
    "entrypoint": "scissors_cut_test",
    "requirements": ["mujoco", "numpy"],
    "simulators": ["mujoco"],
    "tags": ["mechanism", "scissors", "cut", "rope", "force-driven"],
    "used_in": ["Scissor Cutting"],
    "cli": "python -m tool_library.verifiers.scissors_cut_test <scissors_params.json>",
}

_CORD = {"name": "cord", "length": 0.20, "segments": 16, "radius": 0.0025,
         "bend_stiffness": 1e5, "grips": False}


def _assemble(built, cord: bool, cord_params, cut_frac: float, out_dir):
    """The trial scene. Without a cord: the pair on the floor. With one: the
    pair held in the air by a weld (the hand that holds the tool), blades
    open, and the cord hanging THROUGH the wedge plane, both ends held by
    mocap welds (the hand that holds the rope) and its cut join exactly in
    the blade plane at `cut_frac` of the blade length. A cord is cut where it
    pierces the blade plane; one lying IN the plane would cross both blades.
    Returns (model, data, joins, scene_xml)."""
    import json
    from pathlib import Path
    import mujoco
    import numpy as np
    from tool_library.common import scissors as SC
    from tool_library.common.mjcf import mjcf_header, FLOOR_XML, LIGHTS_XML

    p = json.loads(Path(built["params_json"]).read_text())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = mujoco.MjSpec.from_string(
        mjcf_header("scissors_cut_test", timestep=p["timestep"])
        + f"<worldbody>{LIGHTS_XML}{FLOOR_XML}</worldbody></mujoco>")
    child = mujoco.MjSpec.from_file(built["xml"])
    # the asset ships with its own floor and lights; the trial scene has them
    for g in list(child.worldbody.geoms):
        if g.name == "floor":
            child.delete(g)
    for lt in list(child.worldbody.lights):
        child.delete(lt)
    z_tool = 0.15 if cord else 0.0
    tool_frame = spec.worldbody.add_frame()
    tool_frame.pos = [0.0, 0.0, z_tool]
    spec.attach(child, prefix="", frame=tool_frame)
    joins, held = [], []
    if cord:
        # the hand that holds the tool: a weld of the lower half to the world
        # at its compiled pose (zero relpose = "where it starts")
        eq = spec.add_equality()
        eq.type = mujoco.mjtEq.mjEQ_WELD
        eq.objtype = mujoco.mjtObj.mjOBJ_BODY
        eq.name = "tool_hold"
        eq.name1 = built["bodies"]["lower"]
        eq.active = True
        # anchor at the body origin, relpose quat all-zero = "take the pose
        # the model compiles in", torquescale 1 (the spec's own default data
        # is not that)
        eq.data = [0.0] * 10 + [1.0]
        # a HAND holds a tool stiffly: with the stock 20 ms weld the 45 g
        # tool sagged 7 mm and then spun bodily under the ring torque
        eq.solref = [0.002, 1.0]
        eq.solimp = [0.99, 0.999, 0.001, 0.5, 2]
        cp = dict(_CORD, **(cord_params or {}))
        open_deg = float(built["joint_range_deg"][1])
        half = np.radians(open_deg / 2.0)
        L = float(built["blade_length_m"])
        s = float(cut_frac) * L
        yaw0 = np.radians(float(p["yaw_deg"]))
        pivot = np.asarray(p["pos"], float) + [0.0, 0.0, z_tool + float(built["pivot_height_m"])]
        bis = yaw0 + half                       # the wedge bisector, world yaw
        cross = pivot + s * np.array([np.cos(bis), np.sin(bis), 0.0])
        Lc = float(cp["length"])
        # the cord runs straight DOWN through the plane; half a is the upper
        # part, so its far end (the join) lands on the plane at the crossing
        top = cross + np.array([0.0, 0.0, 0.5 * Lc])
        cp = dict(cp, pos=[0.0, 0.0, 0.0], yaw_deg=0.0, include_floor=False,
                  grips=True)
        frame = spec.worldbody.add_frame()
        frame.pos = [float(v) for v in top]
        frame.quat = [np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0]   # +x -> -z
        res = SC.attach_cuttable_thread(spec, cp, out_dir / "cord", prefix="cord_",
                                        cut_frac=0.5, frame=frame,
                                        join_name="cord_cut")
        joins = [res["join"]]
        held = [(f"cord_a_{res['halves']['a']['grips']['a']}",
                 f"cord_a_{res['halves']['a']['welds']['a']}",
                 f"cord_a_{res['halves']['a']['bodies']['first']}"),
                (f"cord_b_{res['halves']['b']['grips']['b']}",
                 f"cord_b_{res['halves']['b']['welds']['b']}",
                 f"cord_b_{res['halves']['b']['bodies']['last']}")]
    scene_xml = out_dir / "scissors_cut_test.xml"
    model = spec.compile()
    scene_xml.write_text(spec.to_xml())
    data = mujoco.MjData(model)
    if cord:
        jid = model.joint(built["joint"]).id
        data.qpos[model.jnt_qposadr[jid]] = np.radians(float(built["joint_range_deg"][1]))
    mujoco.mj_forward(model, data)
    if cord:
        # THE CORD IS HELD AT BOTH ENDS, taut through the wedge, as the other
        # hand holds it close to the cut in the task: closing blades push a
        # loose cord along the wedge toward the tips. The holds are welds to
        # mocap bodies parked at the ends' own poses; the join itself is free.
        for grip, weld, body in held:
            gb = model.body(grip).id
            data.mocap_pos[model.body_mocapid[gb]] = data.xpos[model.body(body).id]
            data.mocap_quat[model.body_mocapid[gb]] = data.xquat[model.body(body).id]
            data.eq_active[model.equality(weld).id] = 1
        mujoco.mj_forward(model, data)
    return model, data, joins, str(scene_xml)


def _trial(built, cord, cord_params, cut_frac, ring_force_N, out_dir):
    from tool_library.common import scissors as SC
    from tool_library.common.mjcf import clear_forces
    from tool_library.common.mechtest import run_headless
    m, d, joins, scene_xml = _assemble(built, cord, cord_params, cut_frac, out_dir)
    if cord:
        run_headless(m, d, None, 0.5)        # the hanging cord settles
    drv = SC.ScissorsDriver(m, d, built, joins=joins, hold_root=not cord)
    drv.MAX_N = float(ring_force_N)
    drv.meas["ring_force_N"] = float(ring_force_N)
    b = built
    if cord:
        # the blades START open with the cord in the wedge: skip the opening
        # stroke, which would only sweep the cord aside
        drv.T["open"] = drv.T["settle"]
        drv.meas["max_open_deg"] = SC.opening_deg(m, d, built)
        # the cut gates, scaled to the cord as the cutting task used them: a
        # cord thicker than the blade overlap is sheared by ONE blade bearing
        # on it while the other backs it, and it sits one radius off the
        # blade plane by construction
        b, drv.sever_kwargs = SC.cord_gates(built, joins[0]["diameter_m"])
        drv.b = b

    def control(t, model, data):
        clear_forces(data)
        return drv.step(t)

    run_headless(m, d, control, drv.T["end"])
    meas = dict(drv.meas)
    if cord and not meas["cut"]:
        meas["sever_report"] = SC.sever_report(m, d, b, joins)
    if meas.get("min_gap_at_join_mm", 0) == 1e9:
        meas["min_gap_at_join_mm"] = None
    return drv.verdict(), meas, scene_xml


def test_scissors_cut(built, cord: bool = True, cord_params=None,
                      cut_frac: float = 0.55, ring_force_N=None,
                      out_mp4=None, out_json=None) -> dict:
    from pathlib import Path
    from tool_library.common import scissors as SC
    from tool_library.common.mechtest import resolve_built, report

    built = resolve_built(built, SC.rebuild)
    out_dir = Path(built["xml"]).parent / "cut_test"
    # A HAND SQUEEZES HARDER WHEN THE CORD RESISTS. The driver's 0.7 N shuts
    # 5 cm blades on a 5 mm cord; longer blades put less force at the cord
    # and a fat cord jams plate blades, so the squeeze escalates and the
    # force it took is part of the measurement.
    forces = ([float(ring_force_N)] if ring_force_N is not None
              else ([0.7, 1.4, 2.8] if cord else [0.7]))
    attempts = []
    for F in forces:
        verdict, meas, scene_xml = _trial(built, cord, cord_params, cut_frac, F, out_dir)
        ok = verdict.startswith("CUT") if cord else verdict.startswith("SCISSORS OK")
        attempts.append({"ring_force_N": F, "verdict": verdict})
        if ok:
            break
    meas["attempts"] = attempts
    video = None
    if out_mp4:
        video = SC.demo(built, out_mp4).get("video", str(out_mp4))
    return report(ok, verdict, meas, out_json=out_json, scene_xml=scene_xml,
                  video=video, cord=bool(cord))


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(test_scissors_cut(sys.argv[1]), indent=1))


scissors_cut_test = test_scissors_cut   # the library calls every tool by its own name
