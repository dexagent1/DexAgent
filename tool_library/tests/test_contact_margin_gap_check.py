"""contact_margin_gap_check: margins judged under the measured MuJoCo semantics. A 'margin == gap' meant to detect without
force (it repels), a large margin and a magnet without margin fail; gap-only detection, small clutch margins and a magnet
with its range pass.

    python tool_library/tests/test_contact_margin_gap_check.py
"""
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tool_library.skills.construct_magnetic_ring import construct_magnetic_ring  # noqa: E402
from tool_library.verifiers.contact_margin_gap_check import contact_margin_gap_check  # noqa: E402

fails = 0


def check(name, cond, msg=""):
    global fails
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + ("" if cond else f"  {msg}"))
    fails += 0 if cond else 1


def box(margin, gap):
    return ('<mujoco><worldbody><geom name="floor" type="plane" size="1 1 .1"/>'
            f'<body name="box" pos="0 0 .03"><freejoint/><geom name="box_g" type="box" size=".03 .03 .03" mass=".2" '
            f'margin="{margin}" gap="{gap}"/></body></worldbody></mujoco>')


# 1. the setup, gap-only early detection, and a 0.5 mm clutch margin pass; the measured semantics are reported
r0 = contact_margin_gap_check(str(ROOT / "setup" / "scene.xml"))
r1 = contact_margin_gap_check(box(0, 0.003))
r2 = contact_margin_gap_check(box(0.0005, 0))
check("setup, gap-only detection and a 0.5 mm margin pass", r0["ok"] and r1["ok"] and r2["ok"], f"{r1['issues']} {r2['issues']}")
check("the probe measures MuJoCo's semantics (margins add, gap is an inactive band in this version)",
      r1["margin_semantics"]["combine"] in ("sum", "max") and isinstance(r1["margin_semantics"]["gap_band"], bool), str(r1["margin_semantics"]))

# 2. margin == gap = 3 mm, written to detect early without force: under MuJoCo 3.x the box is pushed from 3 mm, and the
#    box already hovers at its keyframe (evidence)
r3 = contact_margin_gap_check(box(0.003, 0.003))
check("margin == gap (3 mm) fails as a repelling contact", not r3["ok"] and "push from 3.00 mm" in r3["issues"][0] and
      ("margin == gap" in r3["issues"][0] or not r3["margin_semantics"]["gap_band"]), str(r3["issues"]))

# 3. a 2.5 mm margin alone also pushes the box off the floor; an explicit pair with a 2 mm margin as well
r4 = contact_margin_gap_check(box(0.0025, 0))
pair = ('<mujoco><worldbody><geom name="floor" type="plane" size="1 1 .1"/><body name="box" pos="0 0 .03"><freejoint/>'
        '<geom name="box_g" type="box" size=".03 .03 .03" mass=".2"/></body></worldbody>'
        '<contact><pair geom1="floor" geom2="box_g" margin="0.002"/></contact></mujoco>')
r5 = contact_margin_gap_check(pair)
check("a 2.5 mm geom margin and a 2 mm pair margin fail", not r4["ok"] and not r5["ok"] and "pair" in r5["issues"][0],
      f"{r4['issues']} {r5['issues']}")

# 4. a magnetic ring: its magnet shell has margin = gap = range and is pulled by an adhesion actuator (allowed); with the
#    margin removed it cannot attract from a distance
tmp = Path(tempfile.mkdtemp(prefix="margin_gap_"))
built = construct_magnetic_ring(str(tmp), name="ring", include_floor=True)
r6 = contact_margin_gap_check(built["xml"])
xml0 = re.sub(r'margin="[^"]+" gap="[^"]+"', 'margin="0" gap="0.0003"', Path(built["xml"]).read_text())
(tmp / "ring_nomargin.xml").write_text(xml0)
r7 = contact_margin_gap_check(str(tmp / "ring_nomargin.xml"))
check("a magnet with margin = gap = range passes; without a margin it fails (no attraction before touching)",
      r6["ok"] and not r7["ok"] and "adhesion" in r7["issues"][0], f"{r6['issues']} {r7['issues']}")

sys.exit(1 if fails else 0)
