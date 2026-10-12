"""What a hand holding a body may touch (episode.held_bodies, used by the in-hand outcome check and the plan collision
sweep): a link of a ball-jointed cable counts through its two parent links (a pinch straddles neighbours as soon as the
cable bends), a hinged lid or a drawer only through its own subtree, never the box it hangs on. Two cables (a free root
and a mid-span link) and a lid."""
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402

from tool_library.common import episode as E  # noqa: E402

FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def cable(prefix, n, x0):
    xml = f'<body name="{prefix}0" pos="{x0} 0 0.02"><freejoint/><geom type="capsule" size="0.005 0.01" euler="0 90 0"/>'
    for k in range(1, n):
        xml += f'<body name="{prefix}{k}" pos="0.02 0 0"><joint type="ball"/><geom type="capsule" size="0.005 0.01" euler="0 90 0"/>'
    return xml + "</body>" * n


XML = f"""<mujoco><worldbody>{cable("rope", 6, 0.0)}{cable("cord", 3, 0.3)}
<body name="box" pos="0 0.3 0.05"><freejoint/><geom type="box" size="0.05 0.05 0.05"/>
<body name="lid" pos="0 0.05 0.05"><joint name="hinge" type="hinge" axis="1 0 0"/><geom type="box" size="0.05 0.05 0.003" pos="0 -0.05 0"/>
<body name="knob" pos="0 -0.05 0.01"><geom type="sphere" size="0.01"/></body></body></body></worldbody></mujoco>"""


def names(m, ids):
    return sorted(m.body(int(b)).name for b in ids)


def main():
    m = mujoco.MjModel.from_xml_string(XML)
    got = names(m, E.held_bodies(m, m.body("rope3").id))
    check("a mid-span cable link counts through its two parent links and everything after it",
          got == ["rope1", "rope2", "rope3", "rope4", "rope5"], str(got))
    got = names(m, E.held_bodies(m, m.body("cord1").id))
    check("a link next to the cable's free root counts through the root", got == ["cord0", "cord1", "cord2"], str(got))
    got = names(m, E.held_bodies(m, m.body("lid").id))
    check("a hinged lid counts through its own subtree only, never the box", got == ["knob", "lid"], str(got))
    got = names(m, E.held_bodies(m, m.body("rope3").id, span=0))
    check("span 0 is the plain subtree", got == names(m, E.subtree_bodies(m, m.body("rope3").id)), str(got))
    print("ALL PASS" if not FAILS else f"{FAILS} FAILED")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
