"""rope_pinch_pose.rope_segments orders a rope's capsules along it: one chain by its tree, a rope built in two chains
(the halves of a cuttable rope) chain after chain without interleaving, the second entered from its end nearest the
first (a half whose root is at the far end is walked backwards). Arc lengths then land on the right link."""
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.skills.rope_pinch_pose import rope_segments  # noqa: E402

FAILS = 0
SEG = 0.02


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def chain(prefix, n, x0, step):
    """n capsules along x from x0, each nested in the previous (step +SEG or -SEG)."""
    xml = f'<body name="{prefix}B0" pos="{x0} 0 0.01"><freejoint/><geom name="{prefix}G0" type="capsule" size="0.004 {SEG / 2 - 0.001}" euler="0 90 0"/>'
    for k in range(1, n):
        xml += f'<body name="{prefix}B{k}" pos="{step} 0 0"><joint type="ball"/><geom name="{prefix}G{k}" type="capsule" size="0.004 {SEG / 2 - 0.001}" euler="0 90 0"/>'
    return xml + "</body>" * n


def xs(model, prefix):
    d = mujoco.MjData(model); mujoco.mj_forward(model, d)
    segs = rope_segments(model, d, prefix)
    return [float(0.5 * (s["p0"][0] + s["p1"][0])) for s in segs], [s["geom"] for s in segs]


def main():
    one = mujoco.MjModel.from_xml_string(f"<mujoco><worldbody>{chain('rope_a', 12, 0.0, SEG)}</worldbody></mujoco>")
    x, g = xs(one, "rope_")
    check("one chain runs along the rope", np.all(np.diff(x) > 0) and g[0] == "rope_aG0", str(g[:3]))
    # two halves: a runs +x from 0, b's root sits at the FAR end and its links come back toward the join
    two = mujoco.MjModel.from_xml_string(f"<mujoco><worldbody>{chain('rope_a', 12, 0.0, SEG)}{chain('rope_b', 12, 24 * SEG - SEG, -SEG)}"
                                         "</worldbody></mujoco>")
    x, g = xs(two, "rope_")
    check("a rope in two chains is walked chain after chain, never interleaved", np.all(np.abs(np.diff(x)) < 1.5 * SEG) and len(set(g)) == 24,
          str(g[10:15]))
    check("the second chain is entered from its end nearest the first (walked backwards)",
          g[11] == "rope_aG11" and g[12] == "rope_bG11" and g[-1] == "rope_bG0" and np.all(np.diff(x) > 0), str(g[11:14]) + " ... " + g[-1])
    print("ALL PASS" if not FAILS else f"{FAILS} FAILED")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
