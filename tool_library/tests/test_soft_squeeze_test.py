import os, sys, tempfile
os.environ.setdefault("MUJOCO_GL", "egl")
from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.verifiers.soft_squeeze_test import soft_squeeze_test

SOFT = """<mujoco><option timestep="0.002"/><worldbody>{floor}<body name="obj" pos="0 0 {z}"><freejoint/>
<geom type="{t}" size="{s}" mass="{mass}" solref="{tc} 1" solimp="{dmin} 0.95 {w}"/></body></worldbody></mujoco>"""
cases = [
    ("soft_ellipsoid_passes", dict(t="ellipsoid", s="0.04 0.03 0.02", z=0.02, mass=0.03, tc=0.03, dmin=0.5, w=0.012, floor=""), True),
    ("soft_box_with_own_floor_passes", dict(t="box", s="0.03 0.03 0.02", z=0.02, mass=0.05, tc=0.04, dmin=0.4, w=0.015,
                                             floor='<geom type="plane" size="1 1 0.01"/>'), True),
    ("rigid_box_fails", dict(t="box", s="0.03 0.03 0.02", z=0.02, mass=0.05, tc=0.004, dmin=0.95, w=0.001, floor=""), False),
    ("too_soft_fails", dict(t="sphere", s="0.03", z=0.03, mass=0.02, tc=0.3, dmin=0.01, w=0.08, floor=""), False),
]
fails = 0
with tempfile.TemporaryDirectory() as td:
    for name, kw, expect in cases:
        p = os.path.join(td, name + ".xml"); open(p, "w").write(SOFT.format(**kw))
        r = soft_squeeze_test(p, body="obj")
        good = (r["ok"] == expect) and isinstance(r["issues"], list) and "compression_mm" in r
        print(("[PASS] " if good else "[FAIL] ") + name, r["compression_mm"], r["recovered_mm"], r["issues"][:1])
        fails += 0 if good else 1
    p = os.path.join(td, "nojoint.xml"); open(p, "w").write('<mujoco><worldbody><body name="obj"><geom size="0.02"/></body></worldbody></mujoco>')
    r = soft_squeeze_test(p, body="obj")
    good = r["ok"] is False and r["issues"]
    print(("[PASS] " if good else "[FAIL] ") + "fixed_body_reported"); fails += 0 if good else 1
sys.exit(1 if fails else 0)
