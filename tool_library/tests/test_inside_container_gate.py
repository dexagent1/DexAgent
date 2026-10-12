"""inside_container_gate: a cube on a box's inner floor, a cube balanced on the rim, a cube beside the box, a cube
sticking out over the edge (entirely), a container that moved (judged in its own frame), misuse."""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

from tool_library.verifiers.inside_container_gate import inside_container_gate  # noqa: E402

FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f" {detail}"))
    FAILS += 0 if ok else 1


# an open box 12 x 12 cm, walls 5 cm high and 4 mm thick, floor 4 mm
XML = """<mujoco><worldbody>
<body name="table" pos="0 0 -0.02"><geom type="box" size="0.4 0.4 0.02"/></body>
<body name="tray" pos="0 0 0"><freejoint/>
  <geom type="box" size="0.06 0.06 0.002" pos="0 0 0.002" mass="0.1"/>
  <geom type="box" size="0.06 0.002 0.025" pos="0 0.058 0.025" mass="0.02"/><geom type="box" size="0.06 0.002 0.025" pos="0 -0.058 0.025" mass="0.02"/>
  <geom type="box" size="0.002 0.06 0.025" pos="0.058 0 0.025" mass="0.02"/><geom type="box" size="0.002 0.06 0.025" pos="-0.058 0 0.025" mass="0.02"/>
</body>
<body name="cube" pos="0.3 0 0.015"><freejoint/><geom type="box" size="0.015 0.015 0.015" mass="0.02"/></body>
</worldbody></mujoco>"""


def main():
    tmp = Path(tempfile.mkdtemp())
    scene = tmp / "scene.xml"
    scene.write_text(XML)
    m = mujoco.MjModel.from_xml_path(str(scene))
    q0 = m.qpos0.copy()
    ca = int(m.jnt_qposadr[m.body("cube").jntadr[0]])
    ta = int(m.jnt_qposadr[m.body("tray").jntadr[0]])

    def save(name, cube_pos, tray_pos=(0, 0, 0), tray_yaw=0.0, n=4):
        Q = np.repeat(q0[None], n, 0)
        Q[:, ca:ca + 3] = cube_pos
        Q[:, ta:ta + 3] = tray_pos
        w = Rot.from_euler("z", tray_yaw, degrees=True).as_quat()
        Q[:, ta + 3:ta + 7] = [w[3], w[0], w[1], w[2]]
        p = tmp / f"{name}.npz"
        np.savez(p, qpos=Q, fps=30.0)
        return str(p)

    good = save("good", [0.01, -0.01, 0.004 + 0.015])
    r = inside_container_gate(str(scene), good, "cube", "tray")
    check("a cube resting on the inner floor passes", r["ok"] and r["contacts_below_rim"] > 0, str(r))
    rim = save("rim", [0.058, 0, 0.05 + 0.015])
    r2 = inside_container_gate(str(scene), rim, "cube", "tray")
    check("a cube balanced on the rim fails (only rim contacts, centre above the rim)",
          (not r2["ok"]) and r2["contacts_below_rim"] == 0 and not r2["com_inside_height"], str(r2))
    out = save("out", [0.1, 0, 0.015])
    r3 = inside_container_gate(str(scene), out, "cube", "tray")
    check("a cube beside the box fails (outside the footprint, on the table)",
          (not r3["ok"]) and (not r3["com_inside_xy"]) and r3["touches_table"], str(r3))
    edge = save("edge", [0.05, 0, 0.004 + 0.015])
    r4a = inside_container_gate(str(scene), edge, "cube", "tray")
    r4b = inside_container_gate(str(scene), edge, "cube", "tray", entirely=True)
    check("a cube through the wall: inside by its centre, not entirely", r4a["ok"] and not r4b["ok"], f"{r4a} {r4b}")
    moved = save("moved", [0.2 + 0.02 * np.cos(np.radians(30)), 0.1 + 0.02 * np.sin(np.radians(30)), 0.019], tray_pos=(0.2, 0.1, 0), tray_yaw=30)
    r5 = inside_container_gate(str(scene), moved, "cube", "tray")
    check("a container moved and turned on the table is judged in its own frame", r5["ok"], str(r5))
    raised = []
    for args in (("cube", "nope"), ("nope", "tray")):
        try:
            inside_container_gate(str(scene), good, *args)
        except ValueError as e:
            raised.append(str(e).split(":")[0])
    check("unknown bodies raise naming the parameter", raised == ["container", "body"], str(raised))
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
