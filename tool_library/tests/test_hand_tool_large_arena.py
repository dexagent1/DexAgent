import os, sys, tempfile
import mujoco
from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.common import hand as H
from tool_library.skills.hand_tool_large_arena import hand_tool_large_arena, large_arena

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(H.__file__))))
HAND = os.path.join(ROOT, "setup", "robot", "sharpa_right.xml")
fails = 0
tmp = tempfile.mkdtemp()


def grid_xml(n, size=0.004, gap=0.009, z=0.03):
    geoms = "".join(f'<geom type="box" size="{size} {size} {size}" pos="{(i - n / 2) * gap} {(j - n / 2) * gap} {z}" contype="1" conaffinity="1"/>'
                    for i in range(n) for j in range(n))
    p = os.path.join(tmp, f"grid{n}.xml")
    open(p, "w").write(f'<mujoco><worldbody><body name="g"><freejoint/>{geoms}</body></worldbody></mujoco>')
    return p


def check(name, cond):
    global fails
    print(("[PASS] " if cond else "[FAIL] ") + name); fails += 0 if cond else 1


orig = H._SCENE_XML
# 1. the arena reaches the compiled hand scene and is restored afterwards, at two sizes
for mb in (16, 96):
    with large_arena(mb) as built:
        m, d, rig, ids = H.open_scene({"hand_xml": HAND, "side": "right", "objects": [{"name": "g", "xml": grid_xml(3), "pos": [0, 0, 0]}]})
    check(f"arena {mb} MB applied ({m.narena} bytes) and base restored", m.narena >= mb * 2**20 and H._SCENE_XML == orig and len(built) == 1)

# 2. a contact-dense object with the hand buried in it: default arena overflows or not, the large one never does
xml = grid_xml(14)
m, d, rig, ids = H.open_scene({"hand_xml": HAND, "side": "right", "objects": [{"name": "g", "xml": xml, "pos": [0, 0, 0]}]})
default_bytes = m.narena
with large_arena(256):
    m2, d2, rig2, ids2 = H.open_scene({"hand_xml": HAND, "side": "right", "objects": [{"name": "g", "xml": xml, "pos": [0, 0, 0]}]})
rig2.teleport(d2, [0, 0, 0.03], [0, 0, 0])
ok = True
try:
    mujoco.mj_forward(m2, d2)
except mujoco.FatalError:
    ok = False
check(f"dense grid ({m2.ngeom} geoms, {d2.ncon} contacts) forward runs in 256 MB arena (default {default_bytes} B)", ok and m2.narena >= default_bytes)

# 3. a real tool through the wrapper: pose_from_points pinching a grid block, result passed back verbatim
r = hand_tool_large_arena("pose_from_points", {"scene": {"hand_xml": HAND, "side": "right", "objects": [{"name": "g", "xml": grid_xml(4), "pos": [0, 0, 0]}]},
                                               "points": {"thumb": [0, -0.022, 0.03], "index": [0, 0.022, 0.03]},
                                               "normals": {"thumb": [0, -1, 0], "index": [0, 1, 0]}, "iters": 80}, memory_mb=64)
check(f"pose_from_points via wrapper: {r['hand_scenes_built']} scene(s), worst residual {r['result'].get('worst_residual_mm')} mm",
      r["hand_scenes_built"] >= 1 and "finger_targets" in r["result"] and r["narena_bytes"] >= 64 * 2**20 and H._SCENE_XML == orig)

# 4. misuse fails loudly
for bad, kw in (("unknown tool", dict(tool="no_such_tool_xyz", args={})), ("tiny arena", dict(tool="pose_from_points", args={}, memory_mb=1)),
                ("args not a dict", dict(tool="pose_from_points", args=[1]))):
    try:
        hand_tool_large_arena(**kw); check(f"misuse '{bad}' raises ValueError", False)
    except ValueError:
        check(f"misuse '{bad}' raises ValueError", True)
check("base MJCF restored after failures", H._SCENE_XML == orig)
sys.exit(1 if fails else 0)
