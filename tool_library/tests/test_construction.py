"""Smoke tests for the construction skills and the reconstruction mechanism
tests.

    MUJOCO_GL=egl python tool_library/tests/test_construction.py [quick]

Every constructor is built with its defaults and with changed parameters and
compiled in MuJoCo; every mechanism verifier is run on a case that must pass
and on one that must fail. No external assets are needed; the ArtiCraft
backend of construct_articulated runs only when ARTICRAFT_DIR points at a
checkout (the builtin backend always runs). `quick` skips the slow cable
drag (about 20 s per rope).
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402

QUICK = "quick" in sys.argv[1:]
OUT = Path(tempfile.mkdtemp(prefix="dexagent_construction_", dir=os.environ.get("TMPDIR")))
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, note: str = ""):
    RESULTS.append((name, bool(ok), note))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {note}", flush=True)


def compiles(xml: str) -> bool:
    m = mujoco.MjModel.from_xml_path(xml)
    return m.nbody > 1


def timed(fn, *a, **k):
    t = time.time()
    r = fn(*a, **k)
    return r, round(time.time() - t, 1)


# ---------------------------------------------------------------------------
# 1. constructors: defaults + changed parameters
# ---------------------------------------------------------------------------
from tool_library.skills.construct_bottle import construct_bottle  # noqa: E402
from tool_library.skills.construct_fold_box import construct_fold_box  # noqa: E402
from tool_library.skills.construct_latch_box import construct_latch_box  # noqa: E402
from tool_library.skills.construct_whiteboard import construct_whiteboard  # noqa: E402
from tool_library.skills.construct_marker import construct_marker  # noqa: E402
from tool_library.skills.construct_eraser import construct_eraser  # noqa: E402
from tool_library.skills.construct_rope import construct_rope  # noqa: E402
from tool_library.skills.construct_scissors import construct_scissors  # noqa: E402
from tool_library.skills.construct_spring import construct_spring  # noqa: E402
from tool_library.skills.construct_battery import construct_battery  # noqa: E402
from tool_library.skills.construct_battery_case import construct_battery_case  # noqa: E402
from tool_library.skills.construct_microtube import construct_microtube  # noqa: E402
from tool_library.skills.construct_tube_rack import construct_tube_rack  # noqa: E402
from tool_library.skills.construct_tube_opener import construct_tube_opener  # noqa: E402
from tool_library.skills.construct_conical_tube_rack import construct_conical_tube_rack  # noqa: E402
from tool_library.skills.construct_micropipette import construct_micropipette  # noqa: E402
from tool_library.skills.construct_pipette_tip import construct_pipette_tip  # noqa: E402
from tool_library.skills.construct_tip_box import construct_tip_box  # noqa: E402
from tool_library.skills.construct_hard_disk import construct_hard_disk  # noqa: E402
from tool_library.skills.construct_disk_rack import construct_disk_rack  # noqa: E402
from tool_library.skills.construct_pc_case import construct_pc_case  # noqa: E402
from tool_library.skills.construct_cubes import construct_cubes  # noqa: E402
from tool_library.skills.construct_mesh_asset import construct_mesh_asset  # noqa: E402
from tool_library.skills.construct_cup import construct_cup  # noqa: E402
from tool_library.skills.construct_bowl import construct_bowl  # noqa: E402
from tool_library.skills.construct_articulated import construct_articulated  # noqa: E402
from tool_library.skills.port_urdf_asset import port_urdf_asset  # noqa: E402

CASES = {
    "construct_bottle": (construct_bottle, [{}, dict(body_radius=0.04, body_height=0.2, neck_radius=0.014,
                                                      cap_radius=0.016, thread_pitch=0.003, thread_turns=3,
                                                      body_rgba=[0.2, 0.4, 0.9, 1], cap_rgba=[0.9, 0.9, 0.9, 1])]),
    "construct_fold_box": (construct_fold_box, [{}, dict(style="lid", length=0.2, width=0.15, height=0.08),
                                                dict(style="tray")]),
    "construct_latch_box": (construct_latch_box, [{}]),
    "construct_whiteboard": (construct_whiteboard, [{}, dict(surface_w=0.6, surface_d=0.4)]),
    "construct_marker": (construct_marker, [{}]),
    "construct_eraser": (construct_eraser, [{}]),
    "construct_rope": (construct_rope, [{}, dict(length=0.4, radius=0.004, segments=24, rgba=[0.1, 0.1, 0.9, 1])]),
    "construct_scissors": (construct_scissors, [{}, dict(blade_length=0.09)]),
    "construct_spring": (construct_spring, [{}, dict(stiffness=600.0)]),
    "construct_battery": (construct_battery, [{}]),
    "construct_battery_case": (construct_battery_case, [{}]),
    "construct_microtube": (construct_microtube, [{}]),
    "construct_tube_rack": (construct_tube_rack, [{}, dict(rows=3, cols=4, tubes=[{"row": 0, "col": 0, "cap_deg": 0.0},
                                                                                   {"row": 2, "col": 3, "cap_deg": 90.0}])]),
    "construct_tube_opener": (construct_tube_opener, [{}]),
    "construct_conical_tube_rack": (construct_conical_tube_rack, [{}]),
    "construct_micropipette": (construct_micropipette, [{}]),
    "construct_pipette_tip": (construct_pipette_tip, [{}]),
    "construct_tip_box": (construct_tip_box, [{}]),
    "construct_hard_disk": (construct_hard_disk, [{}]),
    "construct_disk_rack": (construct_disk_rack, [{}, dict(n_slots=3)]),
    "construct_pc_case": (construct_pc_case, [{}]),
    "construct_cubes": (construct_cubes, [{}]),
    "construct_cup": (construct_cup, [{}, dict(name="mug", base_radius=0.04, rim_radius=0.04, height=0.09,
                                               wall=0.004, handle={"radius": 0.02, "tube": 0.005})]),
    "construct_bowl": (construct_bowl, [{}, dict(rim_radius=0.11, base_radius=0.05, height=0.07, curvature=0.4)]),
}
BUILT: dict[str, list[dict]] = {}
for name, (fn, variants) in CASES.items():
    BUILT[name] = []
    for i, kw in enumerate(variants):
        try:
            b, dt = timed(fn, OUT / f"{name}_{i}", **kw)
            ok = compiles(b["xml"]) and Path(b["params_json"]).exists()
            BUILT[name].append(b)
            check(f"{name}[{i}] builds+compiles", ok, f"({dt}s, {list(kw)[:3]})")
        except Exception as exc:
            check(f"{name}[{i}] builds+compiles", False, repr(exc)[:200])

# unknown parameter names are refused (not silently ignored)
try:
    construct_bottle(OUT / "bad", not_a_parameter=1)
    check("construct_bottle refuses unknown keys", False)
except Exception as exc:
    check("construct_bottle refuses unknown keys", "not_a_parameter" in str(exc) or "unknown" in str(exc).lower(), "")

# a mesh asset from a generated mesh
try:
    import trimesh
    mesh = trimesh.creation.box(extents=[0.06, 0.04, 0.03])
    mp = OUT / "block.obj"
    mesh.export(mp)
    b, dt = timed(construct_mesh_asset, OUT / "mesh_asset", mesh_obj=str(mp), up="z")
    check("construct_mesh_asset builds+compiles", compiles(b["xml"]), f"({dt}s)")
except Exception as exc:
    check("construct_mesh_asset builds+compiles", False, repr(exc)[:200])

# the articulated route: builtin always, ArtiCraft when a checkout is configured
ROWS = [{"cols": 2, "height_frac": 0.35}, {"cols": 1, "height_frac": 0.33}, {"cols": 1, "height_frac": 0.32}]
art, dt = timed(construct_articulated, OUT / "cabinet_builtin", width=0.471, depth=0.454, height=0.487,
                rows=ROWS, moving={"row": 1, "col": 0}, body_rgba=[0.58, 0.44, 0.25, 1],
                handle_rgba=[0.05, 0.05, 0.05, 1], backend="builtin")
check("construct_articulated[builtin] writes a URDF", Path(art["urdf"]).exists() and len(art["joints"]) == 4, f"({dt}s)")
check("construct_articulated previews rendered", all(p.endswith(".png") for p in art["previews"]), str(art["previews"])[:120])
try:
    construct_articulated(OUT / "cabinet_bad", width=0.4, depth=0.4, height=0.4, rows=ROWS,
                          body_rgba=[0.5, 0.5, 0.5, 1], handle_rgba=[0.5, 0.5, 0.5, 1], backend="builtin")
    check("construct_articulated refuses a handle painted the body colour", False)
except ValueError:
    check("construct_articulated refuses a handle painted the body colour", True)
port, dt = timed(port_urdf_asset, art["urdf"], OUT / "cabinet_mjcf", name="cabinet")
check("port_urdf_asset ports the cabinet", compiles(port["xml"]) and len(port["joints"]) == 4
      and len(port["handles"]) == 4 and len(port["drawers"]) == 4, f"({dt}s)")
port_free = port_urdf_asset(art["urdf"], OUT / "cabinet_mjcf_free", name="cabfree", scale=0.5,
                            fixed_base=False, include_floor=True)
check("port_urdf_asset scales a free-based copy", abs(port_free["joints"][0]["range"][1] - 0.5 * port["joints"][0]["range"][1]) < 1e-6)
if os.environ.get("ARTICRAFT_DIR") and (Path(os.environ["ARTICRAFT_DIR"]) / "cli" / "compile_record.py").exists() and not QUICK:
    try:
        art2, dt = timed(construct_articulated, OUT / "cabinet_articraft", width=0.40, depth=0.35, height=0.30,
                         rows=[{"cols": 1}, {"cols": 2}], body_rgba=[0.6, 0.45, 0.3, 1],
                         handle_rgba=[0.05, 0.05, 0.05, 1], backend="articraft", preview=False)
        check("construct_articulated[articraft] compiles through ArtiCraft QC",
              Path(art2["urdf"]).exists() and "failures=0" in art2["qc"].get("summary", ""), f"({dt}s)")
    except Exception as exc:
        check("construct_articulated[articraft] compiles through ArtiCraft QC", False, repr(exc)[:300])
else:
    print("[SKIP] construct_articulated[articraft]: ARTICRAFT_DIR not set (or quick)")

# ---------------------------------------------------------------------------
# 2. mechanism verifiers: a pass case and a fail case each
# ---------------------------------------------------------------------------
from tool_library.verifiers.thread_self_locking_test import test_thread_self_locking  # noqa: E402
from tool_library.verifiers.fold_box_fold_test import test_fold_box_fold  # noqa: E402
from tool_library.verifiers.cable_drag_test import test_cable_drag  # noqa: E402
from tool_library.verifiers.scissors_cut_test import test_scissors_cut  # noqa: E402
from tool_library.verifiers.spring_stiffness_test import test_spring_stiffness  # noqa: E402
from tool_library.verifiers.battery_seat_test import test_battery_seat  # noqa: E402
from tool_library.verifiers.latch_snap_test import test_latch_snap  # noqa: E402
from tool_library.verifiers.flip_cap_snap_test import test_flip_cap_snap  # noqa: E402
from tool_library.verifiers.ink_round_trip_test import test_ink_round_trip  # noqa: E402
from tool_library.verifiers.tube_seat_test import test_tube_seat  # noqa: E402
from tool_library.verifiers.plunger_two_stage_test import test_plunger_two_stage  # noqa: E402
from tool_library.verifiers.articulated_joint_travel_test import test_articulated_joint_travel  # noqa: E402
from tool_library.verifiers.vessel_containment_test import test_vessel_containment  # noqa: E402


def run(name, fn, *a, expect=True, **k):
    try:
        r, dt = timed(fn, *a, **k)
        check(f"{name} -> {'pass' if expect else 'fail'} case", r["ok"] == expect, f"({dt}s) {r['verdict'][:110]}")
        return r
    except Exception as exc:
        check(f"{name} -> {'pass' if expect else 'fail'} case", False, repr(exc)[:200])


# thread: both bottles pass; a bottle whose thread equality is switched off is NOT A THREAD
for i, b in enumerate(BUILT["construct_bottle"]):
    run(f"thread_self_locking_test[{i}]", test_thread_self_locking, b)
bad = dict(BUILT["construct_bottle"][0])
bad_xml = Path(bad["xml"]).with_name("bottle_nothread.xml")
bad_xml.write_text(re.sub(r'(<joint name="bottle_thread"[^>]*?)/>', r'\1 active="false"/>', Path(bad["xml"]).read_text()))
bad["xml"] = str(bad_xml)
run("thread_self_locking_test", test_thread_self_locking, bad, expect=False)

# fold box: carton, lid and tray pass; creases with 0.5 N*m of friction do not fold under 1.2 N
for i, b in enumerate(BUILT["construct_fold_box"]):
    run(f"fold_box_fold_test[{i}]", test_fold_box_fold, b)
stiff = construct_fold_box(OUT / "fold_stiff", hinge_frictionloss=0.5)
run("fold_box_fold_test", test_fold_box_fold, stiff, expect=False)

# cable: the default rope passes; capsules that cannot collide have no self-contacts
if not QUICK:
    run("cable_drag_test[1]", test_cable_drag, BUILT["construct_rope"][1])
    nocol = dict(BUILT["construct_rope"][1])
    nx = Path(nocol["xml"]).with_name("thread_nocollide.xml")
    nx.write_text(Path(nocol["xml"]).read_text().replace('<geom type="capsule"', '<geom type="capsule" contype="0" conaffinity="0"', 1))
    nocol["xml"] = str(nx)
    run("cable_drag_test", test_cable_drag, nocol, expect=False)
else:
    print("[SKIP] cable_drag_test (quick)")

# scissors: open/shut without a cord, a cut with one; a 0.05 N finger cuts nothing
sc = BUILT["construct_scissors"][0]
run("scissors_cut_test[no cord]", test_scissors_cut, sc, cord=False)
run("scissors_cut_test[cord]", test_scissors_cut, sc)
run("scissors_cut_test[long blades]", test_scissors_cut, BUILT["construct_scissors"][1])
run("scissors_cut_test", test_scissors_cut, sc, ring_force_N=0.05, expect=False)

# spring: both pass; a spring whose joints are twice as stiff as requested fails the tolerance
for i, b in enumerate(BUILT["construct_spring"]):
    run(f"spring_stiffness_test[{i}]", test_spring_stiffness, b)
wrong = dict(BUILT["construct_spring"][0])
wx = Path(wrong["xml"]).with_name("spring_wrong.xml")
wx.write_text(re.sub(r'stiffness="([0-9.]+)"', lambda mm: f'stiffness="{2*float(mm.group(1)):.3f}"', Path(wrong["xml"]).read_text()))
wrong["xml"] = str(wx)
run("spring_stiffness_test", test_spring_stiffness, wrong, expect=False)

# battery: the default pair passes; a cell 12 mm short of the slot is not held
run("battery_seat_test", test_battery_seat, BUILT["construct_battery_case"][0], BUILT["construct_battery"][0])
short = construct_battery(OUT / "battery_short", length=0.0505 - 0.012)
run("battery_seat_test", test_battery_seat, BUILT["construct_battery_case"][0], short, expect=False)

# latch: 10 N latches; 0.2 N at the strap tip cannot bow the strap over the nub
run("latch_snap_test", test_latch_snap, BUILT["construct_latch_box"][0])
run("latch_snap_test", test_latch_snap, BUILT["construct_latch_box"][0], close_force_N=0.2, expect=False)

# flip cap: the default tube pops and recloses; a 0.5 N pry does not pop the bead
run("flip_cap_snap_test", test_flip_cap_snap, BUILT["construct_microtube"][0])
run("flip_cap_snap_test", test_flip_cap_snap, BUILT["construct_microtube"][0], pry_force_N=0.5, expect=False)

# ink: the board passes; demanding 90% darkness from one stroke fails
run("ink_round_trip_test", test_ink_round_trip, BUILT["construct_whiteboard"][0])
run("ink_round_trip_test", test_ink_round_trip, BUILT["construct_whiteboard"][0], min_ink=0.9, expect=False)

# tube rack: the loaded rack passes; the empty default rack is refused (nothing to test)
run("tube_seat_test", test_tube_seat, BUILT["construct_tube_rack"][1])
try:
    test_tube_seat(BUILT["construct_tube_rack"][0])
    check("tube_seat_test refuses an empty rack", False)
except ValueError:
    check("tube_seat_test refuses an empty rack", True)

# plunger: passes; a 0.3 N thumb never reaches the first stop
run("plunger_two_stage_test", test_plunger_two_stage, BUILT["construct_micropipette"][0])
run("plunger_two_stage_test", test_plunger_two_stage, BUILT["construct_micropipette"][0], first_stage_N=0.3, expect=False)

# articulated travel: the fixed cabinet passes; a 0.2 N pull does not open a damped drawer
run("articulated_joint_travel_test", test_articulated_joint_travel, port)
run("articulated_joint_travel_test", test_articulated_joint_travel, port, force_N=0.2, expect=False)
# a free-standing cabinet (free root) is held where it stands while its drawer is pulled, not yanked away
FREE_CAB = OUT / "free_cabinet.xml"
FREE_CAB.write_text('<mujoco><option timestep="0.002"/><worldbody><body name="cab" pos="0.3 0.2 0.15"><freejoint/>'
                    '<geom type="box" size="0.12 0.1 0.15" mass="2"/><body name="cab_drawer" pos="0 -0.1 0">'
                    '<joint name="cab_slide" type="slide" axis="0 -1 0" range="0 0.15" damping="2"/>'
                    '<geom type="box" size="0.1 0.01 0.05" mass="0.2"/></body></body></worldbody></mujoco>')
run("articulated_joint_travel_test[free root held]", test_articulated_joint_travel, str(FREE_CAB))

# vessels: cup, mug, bowl pass, cup nests in bowl; a cup sealed by a convex hull fails the fill
run("vessel_containment_test[cup]", test_vessel_containment, BUILT["construct_cup"][0])
run("vessel_containment_test[mug]", test_vessel_containment, BUILT["construct_cup"][1])
run("vessel_containment_test[bowl+cup]", test_vessel_containment, BUILT["construct_bowl"][0],
    inner_built=BUILT["construct_cup"][0])
sealed = dict(BUILT["construct_cup"][0])
sx = Path(sealed["xml"]).with_name("cup_sealed.xml")
txt = Path(sealed["xml"]).read_text()
txt = re.sub(r'<geom name="cup_wedge\d+"[^>]*/>\n?', "", txt)
txt = txt.replace('mesh="cup_shell" material="cup_mat" contype="0" conaffinity="0" mass="0"',
                  'mesh="cup_shell" material="cup_mat" density="250"')
sx.write_text(txt)
sealed["xml"] = str(sx)
run("vessel_containment_test", test_vessel_containment, sealed, expect=False)

# ---------------------------------------------------------------------------
n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed; outputs in {OUT}")
if n_fail:
    for name, ok, note in RESULTS:
        if not ok:
            print("  FAILED:", name, note)
    sys.exit(1)
shutil.rmtree(OUT, ignore_errors=True)
