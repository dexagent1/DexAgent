"""scene_size_budget_check: scenes whose peak arena use fits pass; a contact-dense scene in a too-small arena, a contact cap
below the contacts needed and a trajectory frame that buries a body fail.

    python tool_library/tests/test_scene_size_budget_check.py
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.verifiers.scene_size_budget_check import scene_size_budget_check  # noqa: E402

fails = 0


def check(name, cond, msg=""):
    global fails
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + ("" if cond else f"  {msg}"))
    fails += 0 if cond else 1


tmp = Path(tempfile.mkdtemp(prefix="arena_"))


def dense(memory="", size_extra="", n=8, z=0.004):
    """A free plate of n x n small boxes lying on a floor: every box touches the floor (4 contacts each)."""
    geoms = "".join(f'<geom type="box" size=".004 .004 .004" pos="{(i - n / 2) * .009:.4f} {(j - n / 2) * .009:.4f} 0" mass=".001"/>'
                    for i in range(n) for j in range(n))
    sz = f'<size {memory} {size_extra}/>' if (memory or size_extra) else ""
    p = tmp / f"dense_{n}_{len(list(tmp.iterdir()))}.xml"
    p.write_text(f'<mujoco>{sz}<worldbody><geom name="floor" type="plane" size="1 1 .1"/>'
                 f'<body name="plate" pos="0 0 {z}"><freejoint/>{geoms}</body></worldbody></mujoco>')
    return str(p)


# 1. the setup at home and a hand scene (256 MB) pass
r0 = scene_size_budget_check(str(ROOT / "setup" / "scene.xml"))
r1 = scene_size_budget_check({"hand_xml": str(ROOT / "setup" / "robot" / "sharpa_right.xml"), "side": "right",
                              "objects": [{"name": "plate", "xml": dense(n=4), "pos": [0, 0, 0]}]})
check("setup and a hand scene with its 256 MB arena pass", r0["ok"] and r1["ok"] and r1["arena_mb"] >= 256,
      f"{r0['issues']} {r1['issues']} {r1.get('arena_mb')}")

# 2. 196 floor-touching boxes in a 100 KB arena: exhausted or over the headroom; with a 64 MB arena (as the consumer
#    compiles it) the same scene passes
small = dense(memory='memory="100K"', n=14)
r2 = scene_size_budget_check(small, seconds=0.2)
r3 = scene_size_budget_check(small, seconds=0.2, arena_mb=64)
check("a contact-dense scene in a 100 KB arena fails; compiled with 64 MB it passes",
      not r2["ok"] and r3["ok"] and r3["max_contacts"] > 100, f"{r2['issues']} | {r3['issues']} {r3['max_contacts']}")

# 3. a scene capping contacts below what it needs
capped = dense(size_extra='nconmax="20"', n=8)
r4 = scene_size_budget_check(capped, seconds=0.2)
ok_cap = dense(size_extra='nconmax="2000"', n=8)
r5 = scene_size_budget_check(ok_cap, seconds=0.2)
check("nconmax below the contacts needed fails; a generous cap passes", not r4["ok"] and r5["ok"], f"{r4['issues']} {r5['issues']}")

# 4. a trajectory whose last frame sinks the plate into the floor: the dense frame is found
scene = dense(memory='memory="300K"', n=10, z=0.05)
m = mujoco.MjModel.from_xml_path(scene)
Q = np.repeat(mujoco.MjData(m).qpos[None], 20, axis=0)
Q[-1, 2] = 0.002
np.savez(tmp / "sink.npz", qpos=Q, fps=30.0)
r6 = scene_size_budget_check(scene, seconds=0, trajectory=str(tmp / "sink.npz"), stride=5)
r7 = scene_size_budget_check(scene, seconds=0)
check("a trajectory frame that buries the plate exceeds the arena; the scene's own resting state does not",
      not r6["ok"] and r6["peak_at"] in ("frame 19", None) and r7["ok"], f"{r6['issues']} {r6['peak_at']} | {r7['issues']}")

sys.exit(1 if fails else 0)
