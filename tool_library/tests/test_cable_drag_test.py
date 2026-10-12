"""cable_drag_test on thin light cords and on cables it has to find in the model itself.

    python tool_library/tests/test_cable_drag_test.py

Two failures from the runs: a built dict without bodies.segments raised KeyError('segments'), and every 1.25 mm
cord went unstable (mjWARN_BADQACC) because the end grip damped MuJoCo's cvel, which is the velocity of the point
at the cable's overall centre of mass rather than of the gripped link, so the damping fed the link's spin back as
force. The cord is not changed; the harness damps the link's own centre-of-mass velocity.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.common.mjcf import ForceGrip  # noqa: E402
from tool_library.skills.construct_rope import construct_rope  # noqa: E402
from tool_library.verifiers.cable_drag_test import cable_drag_test, cable_links  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dexagent_drag_", dir=os.environ.get("TMPDIR")))
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    FAILS += 0 if ok else 1


# 1. the grip's damping velocity is the gripped link's centre-of-mass velocity (what mj_objectVelocity reports)
thin = construct_rope(TMP / "thin", radius=0.00125, length=1.0, segments=60)
m = mujoco.MjModel.from_xml_path(thin["xml"])
d = mujoco.MjData(m)
d.qvel[:] = np.random.default_rng(0).normal(0, 0.5, m.nv)
mujoco.mj_forward(m, d)
g = ForceGrip(m, thin["bodies"]["first"])
ref = np.zeros(6)
mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, g.bid, ref, 0)
check("ForceGrip damps the link's own COM velocity, not cvel's subtree-COM point",
      np.allclose(g.com_velocity(d), ref[3:], atol=1e-9) and not np.allclose(d.cvel[g.bid, 3:], ref[3:], atol=1e-3),
      f"grip {g.com_velocity(d).round(3)} objvel {ref[3:].round(3)} cvel {d.cvel[g.bid, 3:].round(3)}")

# 2. a 1 m x 1.25 mm cord (0.09 g links): went unstable at t=3.25 s of the swing, now passes as a cable
try:
    r = cable_drag_test(thin)
    check("1 m x 1.25 mm cord: the drag runs stably and the cord passes", r["ok"], r["verdict"][:140])
except Exception as e:                                       # noqa: BLE001
    check("1 m x 1.25 mm cord: the drag runs stably and the cord passes", False, repr(e)[:200])

# 3. a cable given only as an MJCF: its links (and length, radius) are found in the model
short = construct_rope(TMP / "short", radius=0.002, length=0.4, segments=24, name="cord")
links = cable_links(mujoco.MjModel.from_xml_path(short["xml"]), {"xml": short["xml"]})
check("cable_links finds the 23 links in order from the model alone", links["first"] == "cordB_first" and links["last"] == "cordB_last"
      and len(links["segments"]) == 21, f"{links['first']} .. {links['last']} ({len(links['segments'])} between)")
try:
    r = cable_drag_test({"xml": short["xml"]})
    check("a dict with only 'xml' (no bodies.segments, no params): runs and passes", r["ok"] and abs(r["measured"]["nominal_length_m"] - 0.4) < 0.03,
          r["verdict"][:120])
except Exception as e:                                       # noqa: BLE001
    check("a dict with only 'xml' (no bodies.segments, no params): runs and passes", False, repr(e)[:200])

# 4. the verdict still fails a cord that cannot touch itself
nocol = TMP / "short" / "cord_nocollide.xml"
nocol.write_text(Path(short["xml"]).read_text().replace('<geom type="capsule"', '<geom type="capsule" contype="0" conaffinity="0"', 1))
try:
    r = cable_drag_test(dict(short, xml=str(nocol)))
    check("a cord whose capsules cannot collide still fails (no self-contacts)", not r["ok"], r["verdict"][:120])
except Exception as e:                                       # noqa: BLE001
    check("a cord whose capsules cannot collide still fails (no self-contacts)", False, repr(e)[:200])
try:
    cable_drag_test({"xml": str(Path(__file__).resolve().parents[2] / "setup" / "scene.xml")})
    check("a model without a cable is refused with a ValueError", False)
except ValueError as e:
    check("a model without a cable is refused with a ValueError", "no cable" in str(e))
# a branched cable (strands joined at one body) is refused with the checks that do apply to it
link = lambda n, k: "".join(f'<body name="{n}{i}" pos="0.02 0 0"><joint type="ball"/><geom type="capsule" size="0.002 0.01" quat="0.7071 0 0.7071 0" mass="0.001"/>' for i in range(k)) + "</body>" * k
tree = Path(tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))) / "tree.xml"
tree.write_text(f'<mujoco><worldbody><body name="knotB" pos="0 0.5 0.01"><freejoint/><geom type="sphere" size="0.004" mass="0.002"/>'
                f'{link("a", 5)}{link("b", 5)}</body></worldbody></mujoco>')
try:
    cable_drag_test({"xml": str(tree)})
    check("a branched cable is refused with the checks that apply to it", False)
except ValueError as e:
    check("a branched cable is refused with the checks that apply to it", "branched" in str(e) and "strand_crossing_check" in str(e), str(e)[:150])

print("ALL PASS" if not FAILS else f"{FAILS} FAILED")
sys.exit(1 if FAILS else 0)
