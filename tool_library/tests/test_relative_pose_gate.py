"""relative_pose_gate: a part seated on its base wherever the pair ends, a 2 cm offset, a rotated part with and without
symmetry, an axis mask, the target from a reference frame, misuse."""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

from tool_library.verifiers.relative_pose_gate import relative_pose_gate  # noqa: E402

FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f" {detail}"))
    FAILS += 0 if ok else 1


XML = """<mujoco><worldbody>
<body name="base" pos="0 0 0.02"><freejoint/><geom type="box" size="0.03 0.03 0.02" mass="0.1"/></body>
<body name="peg" pos="0.2 0 0.02"><freejoint/><geom type="box" size="0.01 0.01 0.02" mass="0.01"/></body>
</worldbody></mujoco>"""


def quat(e):
    w = Rot.from_euler("xyz", e, degrees=True).as_quat()
    return [w[3], w[0], w[1], w[2]]


def main():
    tmp = Path(tempfile.mkdtemp())
    scene = tmp / "scene.xml"
    scene.write_text(XML)
    m = mujoco.MjModel.from_xml_path(str(scene))
    q0 = m.qpos0.copy()
    ab, ap = (int(m.jnt_qposadr[m.body(b).jntadr[0]]) for b in ("base", "peg"))

    def save(name, base_pos, base_yaw, rel, rel_yaw, n=3):
        Q = np.repeat(q0[None], n, 0)
        Rb = Rot.from_euler("z", base_yaw, degrees=True)
        Q[:, ab:ab + 3] = base_pos
        Q[:, ab + 3:ab + 7] = quat([0, 0, base_yaw])
        Q[-1, ap:ap + 3] = np.asarray(base_pos) + Rb.apply(rel)
        Q[-1, ap + 3:ap + 7] = quat([0, 0, base_yaw + rel_yaw])
        p = tmp / f"{name}.npz"
        np.savez(p, qpos=Q, fps=30.0)
        return str(p)

    tgt = [0.0, 0.0, 0.04]
    s1 = save("seated", [0.1, 0.2, 0.02], 70, tgt, 0)
    r1 = relative_pose_gate(str(scene), s1, "peg", "base", rel_pos=tgt, rel_euler_deg=[0, 0, 0])
    s2 = save("offset", [0.1, 0.2, 0.02], 70, [0.02, 0, 0.04], 0)
    r2 = relative_pose_gate(str(scene), s2, "peg", "base", rel_pos=tgt, rel_euler_deg=[0, 0, 0])
    check("seated on a base that moved and turned passes; a 2 cm offset fails", r1["ok"] and (not r2["ok"]) and abs(r2["pos_err_m"] - 0.02) < 1e-3,
          f"{r1} {r2}")
    s3 = save("turned", [0, 0, 0.02], 0, tgt, 90)
    r3 = relative_pose_gate(str(scene), s3, "peg", "base", rel_pos=tgt, rel_quat=[1, 0, 0, 0])
    r4 = relative_pose_gate(str(scene), s3, "peg", "base", rel_pos=tgt, rel_quat=[1, 0, 0, 0], sym_axis=[0, 0, 1], sym_deg=90)
    check("a square peg turned 90 deg fails without and passes with a 90 deg symmetry", (not r3["ok"]) and r4["ok"], f"{r3} {r4}")
    r5 = relative_pose_gate(str(scene), s2, "peg", "base", rel_pos=tgt, pos_axes="z")
    check("an axis mask judges only the named axes", r5["ok"], str(r5))
    Q = np.load(s1)["qpos"]
    Q = np.r_[Q, Q[-1:]]
    Q[0] = Q[-1]
    Q[-1, ab:ab + 3] += [0.05, 0, 0]
    ref = tmp / "ref.npz"
    np.savez(ref, qpos=Q, fps=30.0)
    r6 = relative_pose_gate(str(scene), str(ref), "peg", "base", reference_frame=0)
    check("the target taken from a reference frame catches a base that moved away", (not r6["ok"]) and r6["pos_err_m"] > 0.04, str(r6))
    raised = []
    for kw in ({}, {"rel_pos": tgt, "pos_axes": "xw"}, {"rel_quat": [1, 0, 0]}):
        try:
            relative_pose_gate(str(scene), s1, "peg", "base", **kw)
        except ValueError as e:
            raised.append(str(e).split(":")[0])
    check("no target, a bad axis mask and a short quaternion raise", len(raised) == 3 and raised[1] == "pos_axes" and raised[2] == "rel_quat", str(raised))
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
