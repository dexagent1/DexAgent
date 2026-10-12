"""The planner carries a free object with the hand that holds it, never a part that turns or slides on one of its
joints: a hand still holding a box's hinged lid after a hold must not drag the whole box with its next move.

    python tool_library/tests/test_planner_attach.py
"""
import os
import re
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tool_library.common import plan as P  # noqa: E402
from tool_library.common.subgoal import SubgoalPlanner  # noqa: E402

SETUP = ROOT / "setup"
TMP = Path(tempfile.mkdtemp(prefix="dexagent_attach_", dir=os.environ.get("TMPDIR")))
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


BOX = ('<body name="box" pos="0 0.5 0.03"><freejoint name="box_free"/><geom name="box_tray" type="box" size="0.06 0.05 0.03" mass="0.2"/>'
       '<body name="box_lid" pos="0 0.05 0.03"><joint name="box_hinge" type="hinge" axis="1 0 0" range="0 3.1"/>'
       '<geom name="box_lid_g" type="box" size="0.06 0.05 0.003" pos="0 -0.05 0" mass="0.03"/></body></body>'
       '<body name="cube" pos="0.15 0.5 0.025"><freejoint name="cube_free"/><geom name="cube_g" type="box" size="0.025 0.025 0.025" mass="0.1"/></body>')
xml = (SETUP / "scene.xml").read_text()
xml = xml.replace('<compiler angle="radian" />', f'<compiler angle="radian" meshdir="{SETUP}" texturedir="{SETUP}" />', 1)
xml = xml.replace("</worldbody>", BOX + "\n  </worldbody>", 1)
xml = re.sub(r'(<key name="home" qpos=")([^"]*)', lambda m: m.group(1) + m.group(2) + " 0 0.5 0.03 1 0 0 0 0 0.15 0.5 0.025 1 0 0 0", xml, count=1)
(TMP / "scene.xml").write_text(xml)
model = mujoco.MjModel.from_xml_path(str(TMP / "scene.xml"))
d = mujoco.MjData(model); mujoco.mj_resetDataKeyframe(model, d, 0)
q0 = d.qpos.copy()

pl = P.Planner(model, qpos0=q0)
try:
    pl.attach("right", "box_lid")
    check("attaching a hinged lid is refused", False, "attached")
except ValueError as e:
    check("attaching a hinged lid is refused", "move_joint" in str(e), str(e)[:120])
pl.attach("right", "cube")
check("a free cube is carried", pl.attached.get("right", {}).get("root_body") == "cube")

sp = SubgoalPlanner(model, q0, state={"held": {"left": "box_lid", "right": "cube"}})
check("a held lid in the state is not re-attached (the box stays put); a held cube is",
      "left" not in sp.pl.attached and sp.pl.attached.get("right", {}).get("root_body") == "cube", str(sp.pl.attached))

try:
    SubgoalPlanner(model, q0).run({"index": 3, "motion": "wait", "side": "right", "with": ["cube", "no_such_body"], "duration_s": 0.2})
    check("`with` naming a body the scene lacks is refused up front", False, "accepted")
except ValueError as e:
    check("`with` naming a body the scene lacks is refused up front", "no_such_body" in str(e), str(e)[:120])
r = SubgoalPlanner(model, q0).run({"index": 3, "motion": "wait", "side": "right", "with": ["cube", "box"], "duration_s": 0.2})
check("`with` takes a list of bodies", {"cube", "box"} <= set(r["expect"]["involved"]), str(r["expect"]["involved"]))

print("ALL PASS" if not FAILS else f"{FAILS} FAILED")
sys.exit(1 if FAILS else 0)
