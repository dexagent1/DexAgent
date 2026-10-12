"""stacked_on_gate: a brick on a brick, bridging two supports, a brick beside its support, a tilted brick, yaw symmetry,
a hinged lid on its own box (collision excluded), misuse."""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

from tool_library.verifiers.stacked_on_gate import stacked_on_gate  # noqa: E402

FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f" {detail}"))
    FAILS += 0 if ok else 1


XML = """<mujoco><worldbody>
<body name="table" pos="0 0 -0.02"><geom type="box" size="0.4 0.4 0.02"/></body>
<body name="b1" pos="0 0 0.02"><freejoint/><geom type="box" size="0.02 0.02 0.02" mass="0.02"/></body>
<body name="b2" pos="0.04 0 0.02"><freejoint/><geom type="box" size="0.02 0.02 0.02" mass="0.02"/></body>
<body name="top" pos="0.2 0 0.02"><freejoint/><geom type="box" size="0.02 0.02 0.02" mass="0.02"/></body>
<body name="wide" pos="0.2 0.2 0.01"><freejoint/><geom type="box" size="0.04 0.02 0.01" mass="0.02"/></body>
<body name="box" pos="-0.2 0 0">
  <geom type="box" size="0.05 0.05 0.003" pos="0 0 0.003"/>
  <geom type="box" size="0.05 0.003 0.03" pos="0 0.047 0.03"/><geom type="box" size="0.05 0.003 0.03" pos="0 -0.047 0.03"/>
  <body name="lid" pos="0 0.05 0.06"><joint name="hinge" type="hinge" axis="1 0 0" range="0 3"/>
    <geom type="box" size="0.05 0.05 0.002" pos="0 -0.05 0.002"/></body>
</body>
</worldbody>
<contact><exclude body1="box" body2="lid"/></contact></mujoco>"""


def main():
    tmp = Path(tempfile.mkdtemp())
    scene = tmp / "scene.xml"
    scene.write_text(XML)
    m = mujoco.MjModel.from_xml_path(str(scene))
    q0 = m.qpos0.copy()
    adr = lambda b: int(m.jnt_qposadr[m.body(b).jntadr[0]])

    def save(name, edits, n=5):
        Q = np.repeat(q0[None], n, 0)
        for b, (pos, euler) in edits.items():
            a = adr(b)
            Q[:, a:a + 3] = pos
            w = Rot.from_euler("xyz", euler, degrees=True).as_quat()
            Q[:, a + 3:a + 7] = [w[3], w[0], w[1], w[2]]
        p = tmp / f"{name}.npz"
        np.savez(p, qpos=Q, fps=30.0)
        return str(p)

    on = save("on", {"top": ([0, 0, 0.06], [0, 0, 0])})
    r = stacked_on_gate(str(scene), on, "top", "b1")
    check("a brick resting on top of another passes", r["ok"] and r["supports"]["b1"]["top_contacts"] > 0, str(r))
    bridge = save("bridge", {"wide": ([0.02, 0, 0.05], [0, 0, 0])})
    rb = stacked_on_gate(str(scene), bridge, "wide", ["b1", "b2"])
    half = save("half", {"wide": ([0.06, 0, 0.05], [0, 0, 0]), "b1": ([-0.1, 0, 0.02], [0, 0, 0])})
    rh = stacked_on_gate(str(scene), half, "wide", ["b1", "b2"])
    check("a plate bridging two bricks passes; with one brick gone it fails naming it",
          rb["ok"] and (not rh["ok"]) and any("b1" in i for i in rh["issues"]), f"{rb} {rh}")
    beside = save("beside", {"top": ([0.08, 0, 0.02], [0, 0, 0])})
    r2 = stacked_on_gate(str(scene), beside, "top", "b2")
    check("a brick standing beside its support fails (side contact only, on the table)",
          (not r2["ok"]) and r2["touches_table"] and r2["supports"]["b2"]["top_contacts"] == 0, str(r2))
    # tilted 20 deg about y, one bottom edge on b1's top
    h = 0.02 * (np.cos(np.radians(20)) + np.sin(np.radians(20)))
    tilt = save("tilt", {"top": ([0, 0, 0.04 + h - 0.0005], [0, 20, 0])})
    r3 = stacked_on_gate(str(scene), tilt, "top", "b1")
    check("a brick balanced 20 deg on an edge fails level", (not r3["ok"]) and r3["level_err_deg"] > 15, str(r3))
    yaw45 = save("yaw45", {"top": ([0, 0, 0.06], [0, 0, 45])})
    yaw90 = save("yaw90", {"top": ([0, 0, 0.06], [0, 0, 90])})
    r4 = stacked_on_gate(str(scene), yaw45, "top", "b1", yaw_tol_deg=10)
    r5 = stacked_on_gate(str(scene), yaw90, "top", "b1", yaw_tol_deg=10, yaw_sym_deg=90)
    check("yaw: 45 deg off fails, 90 deg off passes for a square brick", (not r4["ok"]) and r5["ok"], f"{r4} {r5}")
    # a hinged lid closed on its own box: collision is excluded in the scene, judged by the geometric gap
    lid_adr = int(m.jnt_qposadr[m.joint("hinge").id])
    Q = np.repeat(q0[None], 3, 0)
    closed = tmp / "closed.npz"; np.savez(closed, qpos=Q, fps=30.0)
    Q2 = Q.copy(); Q2[:, lid_adr] = 1.5
    opened = tmp / "open.npz"; np.savez(opened, qpos=Q2, fps=30.0)
    r6 = stacked_on_gate(str(scene), str(closed), "lid", "box")
    r7 = stacked_on_gate(str(scene), str(opened), "lid", "box")
    check("a lid excluded from collision with its box: closed passes, opened fails",
          r6["ok"] and (not r7["ok"]) and "geom distance" in str(r6["supports"]), f"{r6} {r7}")
    raised = []
    for kw in ({"support": "nope"}, {"support": []}, {"frame": 99}):
        try:
            stacked_on_gate(str(scene), on, "top", **{"support": "b1", **kw})
        except ValueError as e:
            raised.append(str(e).split(":")[0])
    check("unknown support, no support and a frame outside the trajectory raise", raised == ["support", "support", "frame"], str(raised))
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
