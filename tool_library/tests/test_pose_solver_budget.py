"""pose_from_points / generate_flat_pinches on a scene of many small parts: fast, bounded, and saying how far they got.

On three EgoDex magnetic rings (75 collision geoms) one pose_from_points call took 15 minutes and generate_flat_pinches
25 minutes without a result: every fingertip-IK iteration ran a full mj_forward whose constraint solver (370
contacts) was 15 of its 15.2 ms. The IK now evaluates kinematics and collisions only (identical numbers), builds hand
pairs only near the contacts, takes a time budget and reports its progress. These cases check the equivalence, the
progress counts and that a budget cuts a call short with a note instead of running on.

    python tool_library/tests/test_pose_solver_budget.py
"""
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.common import hand as H  # noqa: E402
from tool_library.skills.construct_magnetic_ring import construct_magnetic_ring  # noqa: E402
from tool_library.skills.generate_flat_pinches import generate_flat_pinches  # noqa: E402
from tool_library.skills.pose_from_points import pose_from_points  # noqa: E402

HAND = os.environ.get("DEXAGENT_HAND_XML") or str(ROOT / "setup" / "robot" / "sharpa_right.xml")
TMP = Path(tempfile.mkdtemp(prefix="dexagent_budget_", dir=os.environ.get("TMPDIR")))
FAILS = 0
T0 = time.time()


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


# three rings touching in a triangle (the EgoDex fidget-spinner layout), the last one grasped
ring = construct_magnetic_ring(str(TMP / "ring"), name="ring")
centres = {"ring_a": [0.0999, 0.47523, 0.0], "ring_b": [0.08388, 0.45034, 0.0], "ring_c": [0.07033, 0.47666, 0.0]}
scene = {"hand_xml": HAND, "side": "right", "floor": True,
         "objects": [{"name": n, "xml": ring["xml"], "pos": p} for n, p in centres.items()]}
c = np.array(centres["ring_c"])
# the wall pinch the EgoDex run asked for: pads on the ring's top edge, normals 30 deg above the horizontal
points = {"index": [0.0725406408777524, 0.4891971674197009, 0.01295], "thumb": [0.0681193591222476, 0.46412283258029907, 0.01295]}
normals = {"index": [0.15038373318043535, 0.8528685319524433, 0.5], "thumb": [-0.1503837331804353, -0.8528685319524433, 0.5]}

# 1. the kinematic forward gives exactly what mj_forward gives (sites, Jacobians, contacts)
model, data0, rig, ids = H.open_scene(scene, settle_s=0.2)
d1, d2 = mujoco.MjData(model), mujoco.MjData(model)
for d in (d1, d2):
    d.qpos[:] = data0.qpos
    rig.teleport(d, c + [0.0, 0.0, 0.08], [np.pi, 0.0, 0.0])
mujoco.mj_forward(model, d1); H.kinematic_forward(model, d2, contacts=True)
J1, J2 = np.zeros((3, model.nv)), np.zeros((3, model.nv))
s = rig.fingertip_sites["index"]
mujoco.mj_jacSite(model, d1, J1, None, s); mujoco.mj_jacSite(model, d2, J2, None, s)
t = time.time()
for _ in range(50):
    mujoco.mj_forward(model, d1)
t_full = (time.time() - t) / 50
t = time.time()
for _ in range(50):
    H.kinematic_forward(model, d2, contacts=True)
t_kin = (time.time() - t) / 50
check("kinematic_forward == mj_forward for sites, Jacobians and contacts, and much cheaper",
      np.array_equal(d1.site_xpos, d2.site_xpos) and np.array_equal(J1, J2) and d1.ncon == d2.ncon
      and all(abs(d1.contact[i].dist - d2.contact[i].dist) < 1e-12 for i in range(d1.ncon)) and t_kin < 0.5 * t_full,
      f"{d1.ncon} contacts, {1000 * t_full:.2f} ms -> {1000 * t_kin:.3f} ms")

# 2. the EgoDex call's seed sweep (24 seeds; 140 s before) runs to the end and reports it
kw = dict(approach_dirs=[[0.0, -0.3, 0.95]], yaw_step_deg=45, pitches_deg=(-30, 0, 30), curl_free=0.8)     # the run's own sweep
t = time.time()
pose = pose_from_points(scene, points, normals, **kw)
dt = time.time() - t
pr = pose["progress"]
check("pose_from_points on three rings: every seed tried, counted, fast", pr["seeds_tried"] == pr["seeds_total"] == 24 and pr["stopped"] is None
      and "note" not in pose and pose["worst_residual_mm"] < 15.0 and dt < 60, f"{dt:.1f} s, residual {pose['worst_residual_mm']} mm, {pr}")

# 3. pairs everywhere or only near the contacts: the same pose (every ring is within the hand's span here)
pose_all = pose_from_points(scene, points, normals, pair_region=None, **kw)
check("pair_region 'auto' and None give the same pose here", np.allclose(pose["pos"], pose_all["pos"], atol=1e-9)
      and np.allclose(pose["euler"], pose_all["euler"], atol=1e-9))

# 4. a zero budget: one seed, the best pose so far, and a note that says what to change
pose0 = pose_from_points(scene, points, normals, time_budget_s=0.0, **kw)
check("time_budget_s=0: stops after the first seed with a note", pose0["progress"]["seeds_tried"] == 1 and pose0["progress"]["seeds_total"] == 24 and pose0["progress"]["stopped"] == "time_budget"
      and "time budget" in pose0.get("note", "") and len(pose0["pos"]) == 3, pose0.get("note", "")[:90])

# 5. generate_flat_pinches under a tight budget returns early with its progress and a note
t = time.time()
r = generate_flat_pinches(scene, verify_top=1, time_budget_s=8.0)
dt = time.time() - t
p = r["progress"]
check("generate_flat_pinches with an 8 s budget: stops, says how far it got", p["stopped"] == "time_budget" and "note" in r
      and p["candidates_solved"] < p["candidates_total"] and dt < 60, f"{dt:.1f} s, {p}")

print(f"{'ALL PASS' if not FAILS else f'{FAILS} FAILED'} ({time.time() - T0:.0f}s)")
sys.exit(1 if FAILS else 0)
