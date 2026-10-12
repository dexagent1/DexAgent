"""A joint that did not reach its target names what its part pressed against where it stopped (another hand, a wall,
a bystander), and names nothing when the way is clear."""
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.common import outcome as O  # noqa: E402

FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def scene(block):
    blk = '<body name="stopper" pos="0 -0.005 0.1"><geom type="box" size="0.03 0.01 0.03"/></body>' if block else ""
    return mujoco.MjModel.from_xml_string(f"""<mujoco><worldbody>
<body name="box" pos="0 0 0"><geom type="box" size="0.05 0.05 0.02" pos="0 0 0.02"/>
<body name="lid" pos="0 0.05 0.04"><joint name="hinge" type="hinge" axis="-1 0 0" range="0 3"/>
<geom type="box" size="0.05 0.05 0.003" pos="0 -0.05 0.003"/></body></body>{blk}</worldbody></mujoco>""")


def main():
    for block in (True, False):
        m = scene(block)
        a = m.jnt_qposadr[m.joint("hinge").id]
        T = 30
        Q = np.zeros((T, m.nq)); Q[:, a] = np.concatenate([np.linspace(0, 0.45, 15), np.linspace(0.45, 0.0, 15)])
        r = O.evaluate(m, Q, 30.0, list(range(T)), {"joint": "hinge", "joint_target": 1.5})
        blk = r["measured"].get("joint_blocked_by")
        if block:
            check("a lid stopped by a body names it", not r["ok"] and blk and "stopper" in blk["bodies"], str(blk))
        else:
            check("a lid that falls back with nothing in its way names nothing", not r["ok"] and blk is None, str(blk))
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
