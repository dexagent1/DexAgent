import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np

from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.verifiers.body_contact_window_check import body_contact_window_check

fails = 0


def check(name, cond, msg=""):
    global fails
    if cond:
        print(f"[PASS] {name}")
    else:
        fails += 1
        print(f"[FAIL] {name} {msg}")


tmp = Path(tempfile.mkdtemp())
xml = tmp / "scene.xml"
xml.write_text("""
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.01"/>
    <body name="a" pos="0 0 0.02"><freejoint/><geom type="box" size="0.02 0.02 0.02" mass="0.05"/></body>
    <body name="b" pos="0.04 0 0.02"><freejoint/><geom type="box" size="0.02 0.02 0.02" mass="0.05"/>
      <body name="b_shell"><geom type="sphere" size="0.021" contype="2" conaffinity="2" margin="0.001" gap="0.001"/></body>
    </body>
    <body name="c" pos="-0.06 0 0.021"><freejoint/><geom type="sphere" size="0.021" contype="2" conaffinity="2"
      margin="0.001" gap="0.001" mass="0.05"/></body>
  </worldbody>
</mujoco>""")
m = mujoco.MjModel.from_xml_path(str(xml))
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
q0 = d.qpos.copy()

# 1. static: a and b touch on every frame
static = tmp / "static.npz"
np.savez(static, qpos=np.repeat(q0[None], 10, 0), subgoal=np.ones(10, int))
r = body_contact_window_check(str(xml), str(static), touching=[["a", "b"]])
check("two touching boxes stay in contact on all frames", r["ok"] and r["touching"]["a - b"]["frac"] == 1.0, str(r))

# 2. b slides away from frame 5: touching fails with the frame, apart at the end passes
Q = np.repeat(q0[None], 10, 0)
for f in range(5, 10):
    Q[f, 7:10] = q0[7:10] + [0.01 * (f - 4), 0, 0]
moving = tmp / "moving.npz"
np.savez(moving, qpos=Q, subgoal=np.array([1] * 5 + [2] * 5))
r2 = body_contact_window_check(str(xml), str(moving), touching=[["a", "b"]])
r3 = body_contact_window_check(str(xml), str(moving), apart=[["a", "b"]], subgoal=2)
r4 = body_contact_window_check(str(xml), str(moving), apart=[["a", "b"]], apart_at="all")
check("a body pulled away loses contact at the right frame; apart at the end passes, apart on all frames fails",
      (not r2["ok"]) and r2["touching"]["a - b"]["first_lost_frame"] == 5 and r3["ok"] and not r4["ok"],
      f"{r2['touching']} {r3.get('apart')} {r4.get('apart')}")

# 3. a jointless child (a magnet shell) counts for its parent, margin contacts count as touching
Qc = np.repeat(q0[None], 4, 0)
for f in range(4):
    Qc[f, 14:17] = [0.04 - 0.042 - 0.0005 * f - 0.0002, 0, 0.021]  # c's sphere 0.2-1.7 mm from b's shell
near = tmp / "near.npz"
np.savez(near, qpos=Qc)
r5 = body_contact_window_check(str(xml), str(near), touching=[["b", "c"]], frames=[0, 1], tol_m=0.0015)
r6 = body_contact_window_check(str(xml), str(near), touching=[["b", "c"]], frames=[0, 3], tol_m=0.0015)
check("a child body's geoms and margin contacts count; a widening gap fails once beyond tol",
      r5["ok"] and not r6["ok"], f"{r5.get('touching')} {r6.get('touching')}")

# 4. bad input is reported or raised
r7 = body_contact_window_check(str(xml), str(static), touching=[["a", "nope"]])
raised = False
try:
    body_contact_window_check(str(xml), str(static), touching=[["a", "b"]], window=3)
except ValueError:
    raised = True
check("an unknown body is reported, an unknown parameter raises", (not r7["ok"]) and r7["issues"] and raised, str(r7))

sys.exit(1 if fails else 0)
