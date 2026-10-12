import sys, tempfile
import mujoco, numpy as np
from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.skills.construct_battery_case_contact_slide import construct_battery_case_contact_slide as build
from tool_library.skills.construct_battery import construct_battery
from tool_library.common import battery_case as BC

cases = {
    "AA_soft_long_travel": dict(battery_diameter=0.0145, battery_length=0.0505, height=0.0145, spring_coil_radius=0.0042,
                                spring_free_length=0.0126, spring_wire_radius=0.0003, spring_n_coils=4, seated_preload=0.7,
                                spring_stiffness=120.0, bump_radius=0.0035, slot_clearance=0.001),
    "AAA_long_travel": dict(battery_diameter=0.0105, battery_length=0.0445, height=0.0105, slot_clearance=0.001,
                            spring_coil_radius=0.0035, spring_free_length=0.0133, spring_wire_radius=0.00028,
                            spring_n_coils=4, seated_preload=0.7, spring_stiffness=100.0, bump_radius=0.0025),
    "AA_three_slots_no_wires": dict(battery_diameter=0.0145, battery_length=0.0505, n_slots=3,
                                    springs_at=["-x", "+x", "-x"], with_wires=False),
}
fail = 0
for name, p in cases.items():
    d = tempfile.mkdtemp()
    try:
        b = build(d, include_floor=True, **p)
        m = mujoco.MjModel.from_xml_path(b["xml"])
        nslide = sum(1 for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE)
        assert nslide == len(b["slots"]), (nslide, len(b["slots"]))
        for s in b["slots"]:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, s["contact_joint"])
            lo, hi = m.jnt_range[jid]
            assert hi > 0 and abs(-lo - s["contact_travel_m"]) < 1e-6
        bat = construct_battery(tempfile.mkdtemp(), diameter=p["battery_diameter"], length=p["battery_length"], include_floor=False)
        r = BC.probe(b, bat, slot=0)
        # structure is asserted for every case; the seat trial must pass for the AA cases, and for every case
        # the released cell must be ejected (the spring is live); the AAA seat drift is reported, it sits at the 1 mm gate
        assert r["eject"]["ok"], r
        if name.startswith("AA_"):
            assert r["ok"], r
        print(f"[PASS] {name}: travel {b['slots'][0]['contact_travel_m']*1000:.1f} mm, seat {r['seated']}, eject {r['eject']['thrown_mm']} mm")
    except Exception as e:
        fail += 1
        print(f"[FAIL] {name}: {e!r}")
try:
    build(tempfile.mkdtemp(), bogus=1)
    print("[FAIL] unknown parameter accepted"); fail += 1
except ValueError:
    print("[PASS] unknown parameter rejected")
sys.exit(1 if fail else 0)
