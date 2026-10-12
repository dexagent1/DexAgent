"""A screw bite during which the thread lets the cap go is judged by the release: the cap stops turning once free, so
`turned` passes when the lift joint reached the release travel (released_by), and fails without it or when the lift
fell short. Three bites: released, short of the release, a full plain bite."""
import os
import sys
import tempfile
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


XML = """<mujoco><worldbody><body name="bottle"><geom type="cylinder" size="0.03 0.05" pos="0 0 0.05"/>
<body name="carrier" pos="0 0 0.1"><joint name="yaw" type="hinge" axis="0 0 1"/><joint name="lift" type="slide" axis="0 0 1" range="0 0.03"/>
<body name="cap"><geom type="cylinder" size="0.02 0.01"/></body></body></body></worldbody></mujoco>"""


def traj(model, yaw_deg, lift_m, n=30):
    Q = np.zeros((n, model.nq))
    Q[:, model.jnt_qposadr[model.joint("yaw").id]] = np.linspace(0, np.radians(yaw_deg), n)
    Q[:, model.jnt_qposadr[model.joint("lift").id]] = np.linspace(0, lift_m, n)
    return Q


def main():
    model = mujoco.MjModel.from_xml_string(XML)
    exp = {"joint": "yaw", "turned_rad": np.pi / 2, "object": "cap"}
    rel = {"released_by": {"joint": "lift", "travel_m": 0.0158}}
    Q = traj(model, 7.0, 0.0159)
    a = O.evaluate(model, Q, 30.0, list(range(len(Q))), dict(exp))
    b = O.evaluate(model, Q, 30.0, list(range(len(Q))), {**exp, **rel})
    check("a releasing bite fails as a plain bite", not a["ok"] and a["failed"] == ["turned"], str(a["measured"].get("turned_deg")))
    check("a releasing bite passes judged by the release", b["ok"] and b["measured"].get("turned_until_release"), str(b["measured"].get("release_travel_mm")))
    Qs = traj(model, 7.0, 0.012)
    c = O.evaluate(model, Qs, 30.0, list(range(len(Qs))), {**exp, **rel})
    check("a bite short of the release still fails", not c["ok"] and "thread_released" in c["failed"] and "turned" in c["failed"], str(c["failed"]))
    Qf = traj(model, 88.0, 0.0014)
    d = O.evaluate(model, Qf, 30.0, list(range(len(Qf))), dict(exp))
    check("a full plain bite passes", d["ok"], str(d["measured"].get("turned_deg")))
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
