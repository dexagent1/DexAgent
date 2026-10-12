import sys
import tempfile
from pathlib import Path

from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.skills.construct_magnetic_ring import construct_magnetic_ring
from tool_library.verifiers.magnetic_coupling_test import magnetic_coupling_test

fails = 0


def check(name, cond, msg=""):
    global fails
    if cond:
        print(f"[PASS] {name}")
    else:
        fails += 1
        print(f"[FAIL] {name} {msg}")


tmp = Path(tempfile.mkdtemp())

# 1. a fingertip-sized fidget ring: hangs, pulls off at a hand-sized force, keeps the third ring in a chain, rejoins
r = magnetic_coupling_test([construct_magnetic_ring(tmp / "a", name="a")["xml"]], out_json=str(tmp / "a.json"))
check("31 mm ring, 2 N magnets: hang, pull-off, chain and rejoin all pass", r["ok"] and (tmp / "a.json").exists(),
      str(r.get("issues")))

# 2. a larger, heavier ring with stronger magnets, third ring starting on top, three distinct assets
xs = [construct_magnetic_ring(tmp / f"b{i}", name=f"b{i}", outer_radius=0.02, inner_radius=0.014, height=0.01,
                              mass=0.02, magnet_force_N=3.0)["xml"] for i in range(3)]
r2 = magnetic_coupling_test(xs, third_start="above")
check("40 mm ring, 3 N magnets, third ring on top, three assets: passes", r2["ok"], str(r2.get("issues")))

# 3. no magnetism: a ring built with magnet_force_N 0 has no magnet at all, and the verdict says to build one
r3 = magnetic_coupling_test([construct_magnetic_ring(tmp / "c", name="c", magnet_force_N=0.0)["xml"]])
check("zero magnet force is rejected: the ring has no adhesion actuator and the issue names magnet_force_N",
      (not r3["ok"]) and any("magnet_force_N > 0" in s for s in r3["issues"]), str(r3.get("issues")))

# 4. over-strong magnets (25 N on a 12 g ring) are rejected with an actionable issue
r4 = magnetic_coupling_test([construct_magnetic_ring(tmp / "d", name="d", magnet_force_N=25.0)["xml"]])
check("25 N magnets on a 12 g ring are rejected", (not r4["ok"]) and len(r4["issues"]) > 0, str(r4.get("issues")))

# 5. an asset without a magnet actuator is reported, not raised
plain = tmp / "plain.xml"
plain.write_text('<mujoco><worldbody><body name="disc"><freejoint/><geom type="cylinder" size="0.015 0.006" '
                 'mass="0.01"/></body></worldbody></mujoco>')
r5 = magnetic_coupling_test([str(plain)])
check("asset without an adhesion actuator is reported with an actionable issue",
      (not r5["ok"]) and any("adhesion" in s for s in r5["issues"]), str(r5.get("issues")))

# 6. misuse raises
try:
    magnetic_coupling_test([str(plain)], chain_frac=2.5)
    check("chain_frac outside (1, 2) raises ValueError", False)
except ValueError:
    check("chain_frac outside (1, 2) raises ValueError", True)

sys.exit(1 if fails else 0)
