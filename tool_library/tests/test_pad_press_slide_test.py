import sys, tempfile, pathlib
from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.skills.construct_sponge_pad import construct_sponge_pad
from tool_library.verifiers.pad_press_slide_test import pad_press_slide_test
fails = 0
t = tempfile.mkdtemp()
for name, kw in {"default": {}, "small_thick_soft": dict(length=0.10, width=0.06, thickness=0.03, nx=3, ny=2, stiffness_N_per_m=1200),
                 "large_stiff_heavy": dict(length=0.15, width=0.10, nx=5, ny=4, stiffness_N_per_m=4000, mass=0.03)}.items():
    r = pad_press_slide_test(construct_sponge_pad(f"{t}/{name}", name=name, **kw))
    if r["ok"]:
        print(f"[PASS] {name}: squash {r['compression_m']*1000:.2f} mm, k {r['measured_stiffness_N_per_m']:.0f} N/m, slide {r['slide_m']*1000:.0f} mm")
    else:
        fails += 1; print(f"[FAIL] {name}: {r['issues']}")
# negative: a rigid block is not compliant
x = pathlib.Path(t) / "rigid.xml"
x.write_text('<mujoco><worldbody><body name="b" pos="0 0 0.01"><freejoint/><geom type="box" size="0.06 0.04 0.01" density="300"/></body></worldbody></mujoco>')
r = pad_press_slide_test(str(x))
if not r["ok"] and any("no compliant joints" in s for s in r["issues"]):
    print("[PASS] rigid block is reported as not compliant")
else:
    fails += 1; print(f"[FAIL] rigid block: {r.get('issues')}")
# negative: an ice-slick floor does not hold the pad
r = pad_press_slide_test(construct_sponge_pad(f"{t}/slick", name="slick", friction=[0.02, 0.01, 0.001]), floor_friction=0.02, hold_frac=3.0, slide_frac=4.0)
if not r["ok"]:
    print("[PASS] slick pad is reported:", r["issues"][0][:60])
else:
    fails += 1; print("[FAIL] slick pad passed")
sys.exit(1 if fails else 0)
