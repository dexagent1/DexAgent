"""Hand scenes keep the joint state they are given: a lid set open by joint_init stays open through the settle (it
used to fall back onto its rest), and the planner re-anchors poses to the grasp part as the hand scene compiles, not
to the pose recorded in the file (stale once joint_init is edited). Three lid angles."""
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

from tool_library.common import hand as H  # noqa: E402
from tool_library.common.subgoal import pose_anchor  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="hs_joint_", dir=os.environ.get("TMPDIR")))
HAND = str(ROOT / "setup" / "robot" / "sharpa_right.xml")
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def lid_box():
    p = TMP / "lidbox.xml"
    p.write_text("""<mujoco model="lidbox"><worldbody><body name="lidbox" pos="0 0 0">
<geom name="lidbox_base" type="box" size="0.06 0.06 0.03" pos="0 0 0.03" mass="0.3"/>
<body name="lidbox_lid" pos="0 0.06 0.06"><joint name="lidbox_hinge" type="hinge" axis="1 0 0" range="0 3.3" damping="0.01"/>
<geom name="lidbox_lid_g" type="box" size="0.06 0.06 0.004" pos="0 -0.06 0.004" mass="0.05"/></body>
</body></worldbody></mujoco>""")
    return str(p)


def quat_angle(a, b):
    ra = Rot.from_quat([a[1], a[2], a[3], a[0]]); rb = Rot.from_quat([b[1], b[2], b[3], b[0]])
    return float(np.degrees((ra.inv() * rb).magnitude()))


def main():
    xml = lid_box()
    rec = None
    for ang in (0.6, 1.2, 2.0):
        hs = {"hand_xml": HAND, "side": "right", "floor": True, "grasp_body": "lidbox_lidbox_lid",
              "objects": [{"name": "lidbox", "xml": xml, "pos": [0.1, 0.45, 0.0], "quat": [1, 0, 0, 0]}],
              "joint_init": {"lidbox_lidbox_hinge": ang}}
        model, data, rig, ids = H.open_scene(hs)
        q = float(data.qpos[model.jnt_qposadr[model.joint("lidbox_lidbox_hinge").id]])
        check(f"lid set to {ang} rad stays there through the settle", abs(q - ang) < 1e-6, f"{q:.4f}")
        b = H.grasp_body(model, hs, ids)
        qb = Rot.from_matrix(data.xmat[b].reshape(3, 3)).as_quat()
        live = [float(qb[3]), float(qb[0]), float(qb[1]), float(qb[2])]
        if rec is None:                                   # the record written when the scene was at the first angle
            rec = {"body": "lidbox_lidbox_lid", "pos": data.xpos[b].tolist(), "quat_wxyz": live}
        hs["grasp_body_pose"] = rec                       # an edited joint_init with the old record
        f = TMP / f"hs_{ang}.json"; f.write_text(json.dumps(hs))
        a = pose_anchor({"hand_scene": str(f)}, {})
        check(f"re-anchoring reads the lid as compiled at {ang} rad", quat_angle(a["quat_wxyz"], live) < 0.5
              and abs(quat_angle(a["quat_wxyz"], rec["quat_wxyz"]) - np.degrees(ang - 0.6)) < 1.0,
              f"{quat_angle(a['quat_wxyz'], rec['quat_wxyz']):.1f} deg from the record")
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
