"""geom_count_budget_check: objects whose collision geoms are spread out pass; three 25-geom magnetic rings in one cluster
(the case that made each hand-scene step 15 ms) fail, as a bench scene and as a hand scene, and pass when simplified.

    python tool_library/tests/test_geom_count_budget_check.py
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tool_library.skills.construct_magnetic_ring import construct_magnetic_ring  # noqa: E402
from tool_library.verifiers.geom_count_budget_check import geom_count_budget_check  # noqa: E402

fails = 0


def check(name, cond, msg=""):
    global fails
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + ("" if cond else f"  {msg}"))
    fails += 0 if cond else 1


tmp = Path(tempfile.mkdtemp(prefix="geom_budget_"))


def rings(n_segments, spacing, tag):
    """Three rings in one MJCF: built by the library's constructor, merged into one worldbody."""
    bodies, acts = [], []
    for k, (x, y) in enumerate([(0, 0), (spacing, 0), (spacing / 2, 0.87 * spacing)]):
        b = construct_magnetic_ring(str(tmp / f"{tag}{k}"), name=f"ring{k}", n_segments=n_segments, pos=[x, y, 0])
        txt = Path(b["xml"]).read_text()
        bodies.append(txt[txt.index("<worldbody>") + 11:txt.index("</worldbody>")])
        acts.append(txt[txt.index("<actuator>") + 10:txt.index("</actuator>")])
    xml = ('<mujoco><compiler angle="radian"/><worldbody><geom name="floor" type="plane" size="1 1 .1"/>'
           + "".join(bodies) + "</worldbody><actuator>" + "".join(acts) + "</actuator></mujoco>")
    p = tmp / f"{tag}.xml"
    p.write_text(xml)
    return str(p)


# 1. the setup and three 8-segment rings pass; three 24-segment rings touching each other (75 geoms in 6 cm) fail
r0 = geom_count_budget_check(str(ROOT / "setup" / "scene.xml"))
r1 = geom_count_budget_check(rings(8, 0.031, "coarse"))
r2 = geom_count_budget_check(rings(24, 0.031, "dense"))
check("setup and 8-segment rings pass; three touching 24-segment rings fail with a simplification",
      r0["ok"] and r1["ok"] and not r2["ok"] and r2["densest"]["count"] == 75 and "fewer, larger segments" in r2["issues"][0],
      f"{r1['issues']} {r2['issues']} {r2['densest']}")

# 2. the same dense rings spread 30 cm apart: each cluster is one ring (25 geoms) and passes; an object cap still applies
r3 = geom_count_budget_check(rings(24, 0.30, "spread"))
r4 = geom_count_budget_check(rings(24, 0.30, "spread2"), max_object_geoms=16)
check("spread rings pass the reach budget; max_object_geoms=16 still fails each 25-geom ring",
      r3["ok"] and r3["densest"]["count"] == 25 and not r4["ok"], f"{r3['issues']} {r4['issues']}")

# 3. as a hand scene (the way the grasp tools compile it), with the forward time measured and reported
hand = str(ROOT / "setup" / "robot" / "sharpa_right.xml")
objs = []
for k, (x, y) in enumerate([(0, 0), (0.031, 0), (0.0155, 0.027)]):
    b = construct_magnetic_ring(str(tmp / f"hs{k}"), name=f"ring{k}", n_segments=24)
    objs.append({"name": f"ring{k}", "xml": b["xml"], "pos": [x, y, 0.0]})
r5 = geom_count_budget_check({"hand_xml": hand, "side": "right", "objects": objs}, measure=True)
check("the dense rings in a hand scene fail and report the measured forward time",
      not r5["ok"] and r5["kind"] == "hand_scene" and r5["timing_ms"]["forward_ms"] > 0, f"{r5['issues']} {r5['timing_ms']}")

# 4. misuse
try:
    geom_count_budget_check(str(ROOT / "setup" / "scene.xml"), reach_m=0)
    check("reach_m <= 0 raises ValueError", False)
except ValueError:
    check("reach_m <= 0 raises ValueError", True)

sys.exit(1 if fails else 0)
