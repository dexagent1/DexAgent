"""insertion_depth_gate: a drive slid fully into a cage, half in, a fifth in; a flush check on the trailing face; the
envelope test without an axis (seated vs resting on top); a receiver that moved; misuse."""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

from tool_library.verifiers.insertion_depth_gate import insertion_depth_gate  # noqa: E402

FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f" {detail}"))
    FAILS += 0 if ok else 1


# a drive cage open toward -y (its mouth at y = -0.06 in its frame), a 10 cm long drive sliding along +y
XML = """<mujoco><worldbody>
<body name="table" pos="0 0 -0.02"><geom type="box" size="0.4 0.4 0.02"/></body>
<body name="cage" pos="0 0.2 0"><freejoint/>
  <geom type="box" size="0.05 0.06 0.002" pos="0 0 0.002" mass="0.1"/><geom type="box" size="0.05 0.06 0.002" pos="0 0 0.032" mass="0.1"/>
  <geom type="box" size="0.002 0.06 0.016" pos="0.052 0 0.017" mass="0.02"/><geom type="box" size="0.002 0.06 0.016" pos="-0.052 0 0.017" mass="0.02"/>
  <geom type="box" size="0.05 0.002 0.016" pos="0 0.058 0.017" mass="0.02"/>
</body>
<body name="drive" pos="0 0 0.017"><freejoint/><geom type="box" size="0.045 0.05 0.012" mass="0.1"/></body>
</worldbody></mujoco>"""


def main():
    tmp = Path(tempfile.mkdtemp())
    scene = tmp / "scene.xml"
    scene.write_text(XML)
    m = mujoco.MjModel.from_xml_path(str(scene))
    q0 = m.qpos0.copy()
    ca, da = (int(m.jnt_qposadr[m.body(b).jntadr[0]]) for b in ("cage", "drive"))

    def save(name, inside_m, cage_pos=(0, 0.2, 0), yaw=0.0):
        """inside_m: how much of the drive's 10 cm length is past the cage mouth."""
        Q = np.repeat(q0[None], 3, 0)
        R = Rot.from_euler("z", yaw, degrees=True)
        Q[:, ca:ca + 3] = cage_pos
        w = R.as_quat()
        Q[:, ca + 3:ca + 7] = [w[3], w[0], w[1], w[2]]
        Q[:, da + 3:da + 7] = [w[3], w[0], w[1], w[2]]
        y_local = -0.06 + inside_m - 0.05            # drive centre in the cage frame
        Q[:, da:da + 3] = np.asarray(cage_pos) + R.apply([0, y_local, 0.017])
        p = tmp / f"{name}.npz"
        np.savez(p, qpos=Q, fps=30.0)
        return str(p)

    full, half, fifth = save("full", 0.1), save("half", 0.055), save("fifth", 0.02)
    r1 = insertion_depth_gate(str(scene), full, "drive", "cage", axis=[0, 1, 0])
    r2 = insertion_depth_gate(str(scene), half, "drive", "cage", axis=[0, 1, 0])
    r3 = insertion_depth_gate(str(scene), fifth, "drive", "cage", axis=[0, 1, 0])
    check("fully in and 55% in pass at half the length; a fifth fails",
          r1["ok"] and r2["ok"] and (not r3["ok"]) and abs(r3["inserted_fraction"] - 0.2) < 0.02, f"{r1} {r2} {r3}")
    r4 = insertion_depth_gate(str(scene), full, "drive", "cage", axis=[0, 1, 0], point="trailing", min_depth_m=-0.008)
    r5 = insertion_depth_gate(str(scene), save("almost", 0.085), "drive", "cage", axis=[0, 1, 0], point="trailing", min_depth_m=-0.008)
    check("flush: a trailing face at the mouth passes, 15 mm proud fails", r4["ok"] and not r5["ok"], f"{r4} {r5}")
    r6 = insertion_depth_gate(str(scene), full, "drive", "cage")
    Q = np.load(full)["qpos"]
    Q[:, da:da + 3] = [0, 0.2, 0.046]
    np.savez(tmp / "on_top.npz", qpos=Q, fps=30.0)
    r7 = insertion_depth_gate(str(scene), str(tmp / "on_top.npz"), "drive", "cage")
    check("without an axis: seated inside the envelope passes, resting on top of the cage fails", r6["ok"] and not r7["ok"], f"{r6} {r7}")
    r8 = insertion_depth_gate(str(scene), save("moved", 0.1, cage_pos=(0.15, 0.25, 0), yaw=35), "drive", "cage", axis=[0, 1, 0])
    check("a cage that moved and turned: the axis is in the receiver's frame", r8["ok"], str(r8))
    raised = []
    for kw in ({"axis": [0, 0]}, {"point": "tip"}, {"entry": [0, 0, 0]}, {"min_fraction": 1.5}):
        try:
            insertion_depth_gate(str(scene), full, "drive", "cage", **kw)
        except ValueError as e:
            raised.append(str(e).split(":")[0])
    check("a short axis, a bad point, an entry without an axis and a fraction above 1 raise",
          raised == ["axis", "point", "entry", "min_fraction"], str(raised))
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
