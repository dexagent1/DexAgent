"""deformable_shape_gate: a rope laid straight across the table vs a straight target (either direction), a rope bent into
a U, ordered matching, end positions (and swapped ends), planar targets, misuse."""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.verifiers.deformable_shape_gate import deformable_shape_gate  # noqa: E402

FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f" {detail}"))
    FAILS += 0 if ok else 1


N = 12
names = ["ropeB_first"] + [f"ropeB_{i}" for i in range(1, N - 1)] + ["ropeB_last"]
BODIES = "".join(f'<body name="{nm}" pos="{0.03 * i:.3f} 0 0.005"><freejoint/><geom type="capsule" size="0.005 0.012" '
                 f'euler="0 90 0" mass="0.002"/></body>' for i, nm in enumerate(names))
XML = f"""<mujoco><worldbody><body name="table" pos="0 0 -0.02"><geom type="box" size="0.5 0.5 0.02"/></body>{BODIES}</worldbody></mujoco>"""


def main():
    tmp = Path(tempfile.mkdtemp())
    scene = tmp / "scene.xml"
    scene.write_text(XML)
    m = mujoco.MjModel.from_xml_path(str(scene))
    q0 = m.qpos0.copy()
    adr = [int(m.jnt_qposadr[m.body(nm).jntadr[0]]) for nm in names]

    def save(name, pts):
        Q = np.repeat(q0[None], 3, 0)
        for a, p in zip(adr, pts):
            Q[-1, a:a + 3] = p
        f = tmp / f"{name}.npz"
        np.savez(f, qpos=Q, fps=30.0)
        return str(f)

    xs = np.linspace(-0.15, 0.15, N)
    straight = save("straight", np.c_[xs, np.full(N, 0.1), np.full(N, 0.005)])
    target = [[-0.15, 0.1, 0.005], [0.15, 0.1, 0.005]]
    r1 = deformable_shape_gate(str(scene), straight, target_points=target)
    r2 = deformable_shape_gate(str(scene), straight, target_points=target[::-1])
    r3 = deformable_shape_gate(str(scene), straight, target_points=target[::-1], ordered=True)
    check("a straight rope matches a straight target either way unless ordered", r1["ok"] and r2["ok"] and not r3["ok"], f"{r1} {r2} {r3}")
    ang = np.linspace(0, np.pi, N)
    u = save("u", np.c_[0.08 * np.cos(ang), 0.1 + 0.08 * np.sin(ang), np.full(N, 0.005)])
    r4 = deformable_shape_gate(str(scene), u, target_points=target)
    check("a rope bent into a U does not match the straight target (Frechet distance reported)", (not r4["ok"]) and r4["frechet_m"] > 0.05, str(r4))
    r5 = deformable_shape_gate(str(scene), straight, target_ends={"first": [-0.15, 0.1, 0.005], "last": [0.15, 0.1, 0.005]})
    r6 = deformable_shape_gate(str(scene), straight, target_ends={"first": [0.15, 0.1, 0.005], "last": [-0.15, 0.1, 0.005]})
    r7 = deformable_shape_gate(str(scene), straight, target_ends={"first": [0.15, 0.1, 0.005]}, ordered=True)
    check("end targets pass, swapped ends pass unordered (and are reported), an ordered swapped end fails",
          r5["ok"] and r6["ok"] and r6["ends_swapped"] and not r7["ok"], f"{r5} {r6} {r7}")
    r8 = deformable_shape_gate(str(scene), straight, target_points=[[-0.15, 0.1], [0.15, 0.1]])
    check("a planar [k, 2] target compares x, y only", r8["ok"], str(r8))
    raised = []
    for kw in ({}, {"target_points": [[0, 0, 0]]}, {"target_ends": {"middle": [0, 0, 0]}}, {"rope": "cable", "target_points": target}):
        try:
            deformable_shape_gate(str(scene), straight, **kw)
        except ValueError as e:
            raised.append(str(e).split(":")[0])
    check("no target, a one-point target, a bad end key and an unknown rope raise",
          raised == ["target_points / target_ends", "target_points", "target_ends", "rope"], str(raised))
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
