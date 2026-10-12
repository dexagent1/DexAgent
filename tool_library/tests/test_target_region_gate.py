"""target_region_gate: inside a box region, outside it, a polygon, a disc with a height band, yaw and upright tolerances,
an object that leaves the region during the hold time, misuse."""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

from tool_library.verifiers.target_region_gate import target_region_gate  # noqa: E402

FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f" {detail}"))
    FAILS += 0 if ok else 1


XML = """<mujoco><worldbody>
<body name="table" pos="0 0 -0.02"><geom type="box" size="0.4 0.4 0.02"/></body>
<body name="cap" pos="0 0 0.01"><freejoint/><geom type="cylinder" size="0.015 0.01" mass="0.01"/></body>
</worldbody></mujoco>"""


def main():
    tmp = Path(tempfile.mkdtemp())
    scene = tmp / "scene.xml"
    scene.write_text(XML)
    m = mujoco.MjModel.from_xml_path(str(scene))
    q0 = m.qpos0.copy()
    a = int(m.jnt_qposadr[m.body("cap").jntadr[0]])

    def save(name, path, eulers=None):
        Q = np.repeat(q0[None], len(path), 0)
        Q[:, a:a + 3] = path
        if eulers is not None:
            for i, e in enumerate(eulers):
                w = Rot.from_euler("xyz", e, degrees=True).as_quat()
                Q[i, a + 3:a + 7] = [w[3], w[0], w[1], w[2]]
        p = tmp / f"{name}.npz"
        np.savez(p, qpos=Q, fps=30.0)
        return str(p)

    path = np.linspace([0.0, 0.0, 0.01], [0.15, 0.05, 0.01], 30)
    t = save("to_right", path)
    box = {"box": [[0.10, 0.0], [0.20, 0.10]]}
    r1 = target_region_gate(str(scene), t, "cap", box)
    r2 = target_region_gate(str(scene), t, "cap", {"box": [[-0.2, 0.0], [-0.1, 0.1]]})
    check("ending inside the box region passes, a region on the other side fails with the distance",
          r1["ok"] and (not r2["ok"]) and "cm toward the region" in r2["issues"][0], f"{r1} {r2}")
    tri = {"polygon": [[0.12, 0.0], [0.2, 0.0], [0.16, 0.12]]}
    disc = {"center": [0.15, 0.05], "radius": 0.02}
    r3 = target_region_gate(str(scene), t, "cap", tri)
    r4 = target_region_gate(str(scene), t, "cap", disc, z_range=[0.0, 0.02])
    r5 = target_region_gate(str(scene), t, "cap", disc, z_range=[0.03, 0.1])
    check("a polygon and a disc with a height band; the wrong band fails", r3["ok"] and r4["ok"] and not r5["ok"], f"{r3} {r4} {r5}")
    yawed = save("yawed", path, [[0, 0, 0]] * 29 + [[0, 0, 40]])
    r6 = target_region_gate(str(scene), yawed, "cap", box, yaw_target_deg=0, yaw_tol_deg=10)
    r7 = target_region_gate(str(scene), yawed, "cap", box, yaw_target_deg=0, yaw_tol_deg=10, yaw_sym_deg=40)
    tipped = save("tipped", path, [[0, 0, 0]] * 29 + [[80, 0, 0]])
    r8 = target_region_gate(str(scene), tipped, "cap", box, upright_tol_deg=15)
    check("yaw 40 deg off fails, passes with a 40 deg symmetry; a cap on its side fails upright",
          (not r6["ok"]) and r7["ok"] and not r8["ok"], f"{r6} {r7} {r8}")
    leave = np.r_[path, np.repeat(path[-1:], 10, 0)]
    leave[-6:-2] = [0.3, 0.05, 0.01]
    lv = save("leave", leave)
    r9 = target_region_gate(str(scene), lv, "cap", box)
    r10 = target_region_gate(str(scene), lv, "cap", box, hold_s=0.3)
    check("an object that leaves and returns: the end passes, the hold time fails", r9["ok"] and not r10["ok"], f"{r9} {r10}")
    raised = []
    for kw in ({"region": {"circle": 1}}, {"region": box, "point": "top"}, {"region": box, "z_range": [0.1, 0.0]}):
        try:
            target_region_gate(str(scene), t, "cap", **kw)
        except ValueError as e:
            raised.append(str(e).split(":")[0])
    check("a malformed region, an unknown point and an inverted height band raise", raised == ["region", "point", "z_range"], str(raised))
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
