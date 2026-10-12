import os, sys, tempfile
os.environ.setdefault("MUJOCO_GL", "egl")
from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.verifiers.drop_rest_test import drop_rest_test

BOX = """<mujoco><option timestep="0.002"/><worldbody><body name="cube" pos="0 0 0.1"><freejoint/>
<geom name="cube_g" type="box" size="0.03 0.03 0.03" mass="0.1"/></body></worldbody></mujoco>"""
ELL = """<mujoco><option timestep="0.002"/><worldbody><body name="egg" pos="0 0 0.1"><freejoint/>
<geom name="egg_g" type="ellipsoid" size="0.035 0.025 0.018" mass="0.03" friction="1 0.01 {roll}" condim="{cd}" solref="0.03 1" solimp="0.5 0.95 0.012"/>
</body></worldbody></mujoco>"""
FIXED = """<mujoco><worldbody><body name="slab" pos="0 0 0.1"><geom type="box" size="0.03 0.03 0.03"/></body></worldbody></mujoco>"""
fails = 0
with tempfile.TemporaryDirectory() as td:
    def w(name, txt):
        p = os.path.join(td, name); open(p, "w").write(txt); return p
    cases = [
        ("box_comes_to_rest", w("box.xml", BOX), True),
        ("ellipsoid_condim4_rocks_and_fails", w("ell4.xml", ELL.format(roll=0.001, cd=4)), False),
        ("ellipsoid_condim6_rolling_rests", w("ell6.xml", ELL.format(roll=0.005, cd=6)), True),
        ("ellipsoid_condim6_no_rolling_rocks_and_fails", w("ell60.xml", ELL.format(roll=0.0, cd=6)), False),
    ]
    for name, path, expect in cases:
        try:
            r = drop_rest_test(path)
            assert r["ok"] == expect, f"ok={r['ok']} expected {expect}: {r['issues'][:2]} {r['drops']}"
            assert len(r["drops"]) == 5 and (expect or r["issues"]), "report shape"
            print(f"[PASS] {name} worst={r['worst']}")
        except Exception as e:
            fails += 1; print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    try:
        r = drop_rest_test(cases[0][1], drops=[[0, 0, 0], [45, 0, 0]], drop_height=0.05)
        assert r["ok"] and len(r["drops"]) == 2
        print("[PASS] custom_drops_and_height")
    except Exception as e:
        fails += 1; print(f"[FAIL] custom_drops_and_height: {e}")
    for name, kw in (("no_free_body_rejected", dict(asset_xml=w("fixed.xml", FIXED))), ("bad_threshold_rejected", dict(asset_xml=cases[0][1], settle_m=0))):
        try:
            drop_rest_test(**kw); fails += 1; print(f"[FAIL] {name}: accepted")
        except ValueError:
            print(f"[PASS] {name}")
sys.exit(1 if fails else 0)
