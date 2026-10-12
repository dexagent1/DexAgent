import sys, tempfile, os
from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.verifiers.rest_inside_container_test import rest_inside_container_test

def scene(inner_half, inner_h, obj_half, obj_h, euler="0 0 0"):
    t = inner_half + 0.004
    xml = f"""<mujoco><option timestep="0.002"/><worldbody>
<geom type="plane" size="1 1 0.1"/>
<body name="bin" pos="0.2 0.1 0"><freejoint/>
 <geom type="box" size="{t} {t} 0.002" pos="0 0 0.002" mass="0.2"/>
 <geom type="box" size="0.002 {t} {inner_h/2}" pos="{t-0.002} 0 {inner_h/2}" mass="0.05"/>
 <geom type="box" size="0.002 {t} {inner_h/2}" pos="{-t+0.002} 0 {inner_h/2}" mass="0.05"/>
 <geom type="box" size="{t} 0.002 {inner_h/2}" pos="0 {t-0.002} {inner_h/2}" mass="0.05"/>
 <geom type="box" size="{t} 0.002 {inner_h/2}" pos="0 {-t+0.002} {inner_h/2}" mass="0.05"/>
</body>
<body name="obj" pos="-0.2 0 {obj_h}" euler="{euler}"><freejoint/>
 <geom type="box" size="{obj_half} {obj_half} {obj_h}" mass="0.02"/>
</body></worldbody></mujoco>"""
    p = os.path.join(tempfile.mkdtemp(), "s.xml"); open(p, "w").write(xml); return p

fails = 0
r = rest_inside_container_test(scene(0.05, 0.04, 0.015, 0.02), "obj", "bin", floor_z=0.004)
print("[PASS] small block rests on the floor" if r["ok"] else f"[FAIL] small block {r}"); fails += not r["ok"]
r = rest_inside_container_test(scene(0.05, 0.04, 0.015, 0.02, euler="180 0 0"), "obj", "bin", floor_z=0.004)
print("[PASS] inverted block keeps its orientation" if r["ok"] else f"[FAIL] inverted {r}"); fails += not r["ok"]
r = rest_inside_container_test(scene(0.02, 0.04, 0.03, 0.02), "obj", "bin", floor_z=0.004)
ok = (not r["ok"]) and r["issues"]
print("[PASS] oversized block refused" if ok else f"[FAIL] oversized accepted {r}"); fails += not ok
r = rest_inside_container_test(scene(0.05, 0.04, 0.015, 0.02), "obj", "bin", floor_z=0.004, xy_offset=[0.04, 0.0])
ok = (not r["ok"]) and any("wall" in i or "inside" in i or "floor" in i for i in r["issues"])
print("[PASS] off-centre drop against the wall flagged" if ok else f"[FAIL] off-centre {r}"); fails += not ok
r = rest_inside_container_test(scene(0.05, 0.04, 0.015, 0.02), "nope", "bin")
ok = (not r["ok"]) and r["issues"]
print("[PASS] missing body reported" if ok else f"[FAIL] missing {r}"); fails += not ok
sys.exit(1 if fails else 0)
