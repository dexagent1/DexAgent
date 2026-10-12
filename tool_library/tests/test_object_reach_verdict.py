"""object_reach_check / scene_qc reach verdicts never print '(None mm short)'.

On a scene whose stations have hands but no arms (a hand-built twin before attach_arms) every object came back
'OUT OF REACH ... (None mm short)', and scene_qc repeated it as a warning. Without arms the check is now skipped and
says so; with arms an unreachable object is reported with its real shortfall.

    python tool_library/tests/test_object_reach_verdict.py
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402

from tool_library.verifiers.object_reach_check import object_reach_check  # noqa: E402
from tool_library.verifiers.scene_qc import scene_qc  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dexagent_reach_", dir=os.environ.get("TMPDIR")))
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def absolute_assets(spec, xml_path):
    """Mesh and texture files as absolute paths, so the spec can be written anywhere."""
    base = Path(xml_path).resolve().parent
    for m in spec.meshes:
        if m.file and not Path(m.file).is_absolute():
            m.file = str(base / (spec.meshdir or "") / m.file)
    for t in spec.textures:
        if t.file and not Path(t.file).is_absolute():
            t.file = str(base / (spec.texturedir or "") / t.file)
    spec.meshdir = ""; spec.texturedir = ""
    return spec


def box_body(spec, name, pos):
    b = spec.worldbody.add_body(); b.name = name; b.pos = list(pos)
    b.add_freejoint()
    g = b.add_geom(); g.name = f"{name}_g"; g.type = mujoco.mjtGeom.mjGEOM_BOX; g.size = [0.03, 0.03, 0.03]; g.mass = 0.1


# 1. hands on their stations, no arms: skipped, not 'None mm short'
sp = mujoco.MjSpec.from_string('<mujoco><worldbody><geom name="floor" type="plane" size="1 1 .1"/></worldbody></mujoco>')
box_body(sp, "cup", [0.1, 0.4, 0.03])
hand = absolute_assets(mujoco.MjSpec.from_file(str(ROOT / "setup" / "robot" / "sharpa_right.xml")), ROOT / "setup" / "robot" / "sharpa_right.xml")
f = sp.worldbody.add_frame(); f.pos = [0.3, 0.0, 0.3]
sp.attach(hand, prefix="robot_right_", frame=f)
noarm = TMP / "noarm.xml"; noarm.write_text(sp.to_xml())
r = object_reach_check(str(noarm))
check("no arm stations: unchecked cannot certify reach", not r["ok"] and r.get("skipped") and "None" not in r["verdict"] and "NOT CHECKED" in r["verdict"], r["verdict"])
q = scene_qc(str(noarm), settle_seconds=0.3)
check("scene_qc reports the unchecked reach", bool(q["warnings"]) and "None" not in str(q["checks"].get("object_reach")), str(q["warnings"]))

# 2. the setup bench with one object in reach and one 1.2 m out: the far one is short by a real number of mm
sp = absolute_assets(mujoco.MjSpec.from_file(str(ROOT / "setup" / "scene.xml")), ROOT / "setup" / "scene.xml")
box_body(sp, "near_box", [0.15, 0.40, 0.03])
box_body(sp, "far_box", [0.15, 1.20, 0.03])
for k in list(sp.keys):                         # the keyframe no longer matches nq: let the scene start from qpos0
    sp.delete(k)
bench = TMP / "bench.xml"; bench.write_text(sp.to_xml())
r = object_reach_check(str(bench), objects=["near_box", "far_box"])
far = r["objects"]["far_box"]
check("bench: the near box is reachable, the far one is not", r["objects"]["near_box"]["reachable"] and not far["reachable"] and r["unreachable"] == ["far_box"],
      r["verdict"][:120])
check("bench: the verdict gives the far box's shortfall in mm, never None",
      "None" not in r["verdict"] and f"{far['residual_mm']:.0f} mm short" in r["verdict"] and far["residual_mm"] > 20, r["verdict"][:160])

# Gate logic must retain a feasible candidate even when an infeasible candidate
# has a smaller weighted residual, and must respect the Stage 1 acting arm.
from unittest.mock import patch
import numpy as np
def solver(model, side):
    values = iter([(.001, np.radians(21), []), (.019, 0., [])]) if side == 'right' else iter([(.2, 0., []), (.2, 0., [])])
    return lambda p, R: next(values)
with patch('tool_library.skills.optimize_pose_for_reach._reach_solver', solver):
    r = object_reach_check(str(bench), objects=['near_box'])
check('feasible candidate wins over lower weighted but invalid residual', r['ok'] and r['objects']['near_box']['best_side'] == 'right')
with patch('tool_library.skills.optimize_pose_for_reach._reach_solver', solver):
    r = object_reach_check(str(bench), objects=['near_box'], required_sides={'near_box': ['left']})
check('wrong arm reach cannot satisfy acting-hand prerequisite', not r['ok'] and r['objects']['near_box']['required_sides'] == ['left'])

print("ALL PASS" if not FAILS else f"{FAILS} FAILED")
sys.exit(1 if FAILS else 0)
