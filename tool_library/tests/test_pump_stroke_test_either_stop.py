import sys, tempfile
from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.skills.construct_pump_sprayer import construct_pump_sprayer as upper_rest
from tool_library.skills.construct_pump_sprayer_down_axis import construct_pump_sprayer_down_axis as construct_pump_sprayer
from tool_library.verifiers.pump_stroke_test_either_stop import pump_stroke_test_either_stop as pump_stroke_test
fails = 0
t = tempfile.mkdtemp()
for name, kw in {"default": {}, "stiff": dict(stroke=0.004, spring_stiffness=900, preload_N=3.0),
                 "soft_heavy_head": dict(body_radius=0.03, stroke=0.008, stem_height=0.012, spring_stiffness=300, preload_N=1.0, head_mass=0.01)}.items():
    r = pump_stroke_test(construct_pump_sprayer(f"{t}/{name}", name=name, **kw))
    if r["ok"]:
        print(f"[PASS] {name}: measured {r['measured_stiffness_N_per_m']:.0f} N/m, full {max(r['displacements_m']['full'])*1000:.2f} mm")
    else:
        fails += 1; print(f"[FAIL] {name}: {r['issues']}")
# negative: a pump whose head cannot reach the bottom under a weak press must fail
b = construct_pump_sprayer(f"{t}/weak", name="weak")
r = pump_stroke_test(b, press_force_N=2.5)
if not r["ok"] and any("reaches only" in s for s in r["issues"]):
    print("[PASS] weak press is reported")
else:
    fails += 1; print(f"[FAIL] weak press not reported: {r}")
r = pump_stroke_test(upper_rest(f"{t}/upper", name="upper"))
if r["ok"] and r["rest_stop"] == "upper":
    print("[PASS] upper-rest convention also passes")
else:
    fails += 1; print(f"[FAIL] upper-rest convention: {r['issues']}")
# negative: a rigid bottle without a slide joint
import pathlib
x = pathlib.Path(t) / "rigid.xml"
x.write_text('<mujoco><worldbody><body name="b"><freejoint/><geom type="cylinder" size="0.02 0.05"/></body></worldbody></mujoco>')
r = pump_stroke_test(str(x))
if not r["ok"] and r["issues"]:
    print("[PASS] rigid bottle is reported:", r["issues"][0][:60])
else:
    fails += 1; print("[FAIL] rigid bottle passed")
sys.exit(1 if fails else 0)
