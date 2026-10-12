import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root: runnable from anywhere
from tool_library.skills.construct_hinged_lid_box import construct_hinged_lid_box
from tool_library.verifiers.hinged_lid_closure_test import hinged_lid_closure_test
fails = 0
cases = [
    ("gift_box_rest_skirt", dict(length=0.25, width=0.23, height=0.045, lid_init_deg="rest"), True),
    ("chest_rest_no_skirt", dict(length=0.40, width=0.30, height=0.20, lid_flap=False, lid_init_deg="rest"), True),
    ("small_rest_skirt60", dict(length=0.12, width=0.10, height=0.05, lid_init_deg="rest", skirt_init_deg=60.0), True),
    ("open_120_lid_falls_back", dict(length=0.30, width=0.20, height=0.08, lid_init_deg=120.0), False),
    ("blocked_lid_cannot_shut", dict(length=0.25, width=0.20, height=0.05, lid_init_deg="rest", lid_range_deg=[30.0, 200.0]), False),
]
for name, kw, want in cases:
    td = tempfile.mkdtemp()
    b = construct_hinged_lid_box(td, **kw)
    r = hinged_lid_closure_test(b["xml"], b["lid_joint"], b["skirt_joint"])
    good = (r["ok"] == want) and (want or r["issues"])
    print(("[PASS] " if good else "[FAIL] ") + f"{name}: ok={r['ok']} close={r['close']} hold={r['hold']} issues={r['issues']}")
    fails += (not good)
r = hinged_lid_closure_test(b["xml"], "no_such_joint")
good = (not r["ok"]) and r["issues"]
print(("[PASS] " if good else "[FAIL] ") + "missing joint reported")
fails += (not good)
sys.exit(1 if fails else 0)
