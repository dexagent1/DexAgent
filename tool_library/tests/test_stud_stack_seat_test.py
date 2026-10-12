import sys, tempfile, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root: runnable from anywhere
from tool_library.skills.construct_stud_brick import construct_stud_brick
from tool_library.verifiers.stud_stack_seat_test import stud_stack_seat_test

fails = 0
def case(name, fn):
    global fails
    try:
        fn(); print(f"[PASS] {name}")
    except Exception as e:
        fails += 1; print(f"[FAIL] {name}: {e!r}")

LARGE = dict(pitch=0.029, body_height=0.026, stud_radius=0.0082, stud_height=0.0065, top_thickness=0.0025,
             clearance=0.0006, clutch_margin=0.0008, lead_in_frac=0.6, cap_friction=0.05, mass=0.02)

def duplo_default():
    b = construct_stud_brick(tempfile.mkdtemp(), name="duplo", clutch_margin=0.00042, lead_in_frac=0.6)
    r = stud_stack_seat_test(b)
    assert r["ok"], r["issues"]
    assert len(r["trials"]) == 3 and r["shove"]["ok"]

def large_overhang_both_axes():
    b = construct_stud_brick(tempfile.mkdtemp(), name="big", **LARGE)
    r = stud_stack_seat_test(b, offsets_m=[[0, 0], [0.0021, -0.0021], [0.0035, 0.0], [0.029, 0.0], [0.0, -0.029]])
    assert r["ok"], r["issues"]

def long_on_square():
    lo = construct_stud_brick(tempfile.mkdtemp(), name="sq", **LARGE)
    up = construct_stud_brick(tempfile.mkdtemp(), name="long", studs_x=4, **dict(LARGE, mass=0.04))
    r = stud_stack_seat_test(lo, up, offsets_m=[[0, 0], [0.029, 0.0]])
    assert r["ok"], r["issues"]

def plain_blocks_fail():
    d = tempfile.mkdtemp()
    xml = os.path.join(d, "block.xml")
    open(xml, "w").write('<mujoco><worldbody><body name="blk" pos="0 0 0"><freejoint/>'
                         '<geom type="box" size="0.02 0.02 0.0125" pos="0 0 0.0125" mass="0.01" friction="0.6 0.01 0.001"/>'
                         '</body></worldbody></mujoco>')
    r = stud_stack_seat_test({"xml": xml, "top_z_m": 0.025}, shove_N=0.3)
    assert not r["ok"] and r["shove"] and not r["shove"]["ok"], r
    assert any("shove" in i for i in r["issues"])

def perched_brick_fails():
    # studs taller than the clearance allows a seat with no lead-in and a tight clutch: it must be reported as perched
    b = construct_stud_brick(tempfile.mkdtemp(), name="tight", **dict(LARGE, lead_in_frac=0.0, clutch_margin=0.0026, clearance=0.0006))
    r = stud_stack_seat_test(b, offsets_m=[[0.004, 0.004]], press_N=0.5, shove_N=0)
    assert not r["ok"] and not r["trials"][0]["seated"], r

case("duplo 2x2 light clutch: aligned, lead-in and one-pitch offset seat; shove held", duplo_default)
case("large 2x2 with clutch: lead-in and half overhang on both axes", large_overhang_both_axes)
case("2x4 brick seated on a 2x2, centred and overhanging", long_on_square)
case("plain blocks without studs fail the shove", plain_blocks_fail)
case("tight brick without lead-in is reported perched", perched_brick_fails)
sys.exit(1 if fails else 0)
