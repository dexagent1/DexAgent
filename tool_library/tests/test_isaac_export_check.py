"""isaac_export_check: an export that matches its scene passes (and loads in Isaac Sim when DEXAGENT_ISAAC_PYTHON is set);
a scene re-placed after the export, and a stage whose prims are not where the scene has them, fail. Without Isaac the load
test is skipped with ok and a note.

    python tool_library/tests/test_isaac_export_check.py        # set DEXAGENT_ISAAC_PYTHON (+ DEXAGENT_ISAAC_HOME) for the Isaac cases
"""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.verifiers.isaac_export_check import isaac_export_check  # noqa: E402

fails = 0


def check(name, cond, msg=""):
    global fails
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + ("" if cond else f"  {msg}"))
    fails += 0 if cond else 1


tmp = Path(tempfile.mkdtemp(prefix="isaac_export_"))
SCENE = ('<mujoco><compiler angle="radian"/><worldbody><body name="table"><geom name="g_table" type="box" size=".5 .5 .02" pos="0 0 -.02"/></body>'
         '<body name="box_box" pos="{x} 0.1 .03" euler="0 0 0.4"><freejoint/><geom name="box_box_g" type="box" size=".03 .03 .03"/></body>'
         '</worldbody></mujoco>')
(tmp / "scene.xml").write_text(SCENE.format(x="0.05"))
(tmp / "scene_moved.xml").write_text(SCENE.format(x="0.08"))


def export_records(scene):
    """The geoms build_isaac_scene exports, at the scene's initial state."""
    m = mujoco.MjModel.from_xml_path(scene)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    out = []
    for g in range(m.ngeom):
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, d.geom_xmat[g])
        out.append({"name": m.geom(g).name, "body": m.body(int(m.geom_bodyid[g])).name, "type": "box",
                    "size": m.geom_size[g].tolist(), "pos": d.geom_xpos[g].tolist(), "quat": q.tolist()})
    return out


def usda(records, path, shift=(0, 0, 0)):
    prims = []
    for k, r in enumerate(records):
        p = np.asarray(r["pos"]) + (np.asarray(shift) if r["body"] != "table" else 0)
        q = r["quat"]
        prims.append(f'        def Xform "g{k:03d}"\n        {{\n'
                     f'            double3 xformOp:translate = ({p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f})\n'
                     f'            quatd xformOp:orient = ({q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f})\n'
                     '            double3 xformOp:scale = (1, 1, 1)\n'
                     '            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]\n'
                     f'            def Cube "box"\n            {{\n                double size = 2\n            }}\n        }}\n')
    path.write_text('#usda 1.0\n(\n    defaultPrim = "World"\n    metersPerUnit = 1\n    upAxis = "Z"\n)\n\n'
                    'def Xform "World"\n{\n    def Xform "static"\n    {\n' + "".join(prims) + "    }\n}\n")


recs = export_records(str(tmp / "scene.xml"))
for d in ("good", "stale"):
    (tmp / d).mkdir()
    (tmp / d / "export.json").write_text(json.dumps({"geoms": recs, "deformables": [], "scene_xml": "scene.xml"}))
usda(recs, tmp / "good" / "isaac_scene.usda")
usda(recs, tmp / "stale" / "isaac_scene.usda", shift=(0.0, 0.04, 0.0))
isaac = os.environ.get("DEXAGENT_ISAAC_PYTHON")
have_isaac = bool(isaac) and Path(isaac).exists()

# 1. a matching export: export.json agrees with the scene; Isaac (when configured) opens the stage and agrees too
r1 = isaac_export_check(str(tmp / "good" / "isaac_scene.usda"), str(tmp / "scene.xml"))
check("a matching export passes" + (" in Isaac Sim" if have_isaac else " (Isaac load test skipped with a note)"),
      r1["ok"] and r1["export_vs_scene"]["n"] == 2 and (r1["skipped_isaac"] != have_isaac) and
      (not have_isaac or (r1["usd_vs_scene"]["opened"] and r1["usd_vs_scene"]["n_bad"] == 0)), str(r1)[:400])

# 2. the scene re-placed after the export (box moved 3 cm): the export no longer matches, no Isaac needed
r2 = isaac_export_check(str(tmp / "good" / "isaac_scene.usda"), str(tmp / "scene_moved.xml"), isaac_python="/nonexistent/python")
check("a scene re-placed after the export fails without Isaac",
      not r2["ok"] and r2["skipped_isaac"] and "no longer matches" in r2["issues"][0], str(r2["issues"]))

# 3. a stage whose box prim sits 4 cm from where its export (and the scene) put it: only Isaac's reading of the stage sees it
if have_isaac:
    r3 = isaac_export_check(str(tmp / "stale" / "isaac_scene.usda"), str(tmp / "scene.xml"))
    check("a stage with a misplaced prim fails in Isaac Sim", not r3["ok"] and r3["usd_vs_scene"]["n_bad"] == 1, str(r3["issues"]))
else:
    print("[SKIP] misplaced-prim case (DEXAGENT_ISAAC_PYTHON not set)")

# 4. misuse
try:
    isaac_export_check(str(tmp / "nope.usda"), str(tmp / "scene.xml"))
    check("a missing USD raises ValueError", False)
except ValueError:
    check("a missing USD raises ValueError", True)

sys.exit(1 if fails else 0)
