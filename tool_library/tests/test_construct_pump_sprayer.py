import sys, tempfile
import mujoco
from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.skills.construct_pump_sprayer import construct_pump_sprayer
fails = 0
t = tempfile.mkdtemp()
cases = {
    "default": {},
    "short_stiff": dict(stroke=0.004, spring_stiffness=900, preload_N=3.0, nozzle_azimuth_deg=90),
    "wide_long_stroke": dict(body_radius=0.03, body_height=0.12, stroke=0.008, stem_height=0.012, spring_stiffness=300, preload_N=1.0, band_height=0.0),
}
for name, kw in cases.items():
    try:
        b = construct_pump_sprayer(f"{t}/{name}", name=name, **kw)
        m = mujoco.MjModel.from_xml_path(b["xml"])
        j = m.joint(b["joint"])
        assert j.type[0] == mujoco.mjtJoint.mjJNT_SLIDE
        assert abs(float(j.range[0]) + b["stroke_m"]) < 1e-9 and float(j.range[1]) == 0.0
        assert abs(float(m.qpos_spring[j.qposadr[0]]) * b["spring_stiffness"] - b["preload_N"]) < 1e-3
        assert abs(b["mass_kg"] - (kw.get("mass", 0.08) + 0.005)) < 1e-3, b["mass_kg"]
        d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        assert m.nexclude == 1
        print(f"[PASS] {name}: height {b['height_m']:.3f} m, full stroke {b['full_stroke_force_N']:.2f} N")
    except Exception as e:
        fails += 1; print(f"[FAIL] {name}: {e}")
for bad in (dict(stroke=0.02), dict(colour=1), dict(spring_stiffness=0)):
    try:
        construct_pump_sprayer(f"{t}/bad", **bad); fails += 1; print(f"[FAIL] accepted {bad}")
    except ValueError:
        print(f"[PASS] refuses {bad}")
sys.exit(1 if fails else 0)
