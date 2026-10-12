# Three surfaces, two hands, both contact sides: the seated fingers must sit within a millimetre of the plane, never in it.
import sys, tempfile
from pathlib import Path
import numpy as np, mujoco
from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
import tool_library.common.hand as Hm
from tool_library.common import hand as H
from tool_library.skills.flat_hand_pose_seated import flat_hand_pose_seated

ROOT = Path(Hm.__file__).resolve().parents[2]
HAND = {s: ROOT / "setup" / "robot" / f"sharpa_{s}.xml" for s in ("left", "right")}
fails = 0


def obj(tmp, name, size, z):
    p = Path(tmp) / f"{name}.xml"
    p.write_text(f'<mujoco model="{name}"><worldbody><body name="{name}"><geom name="g" type="box" size="{size[0]} {size[1]} {size[2]}" '
                 f'pos="0 0 {z}" rgba="0.6 0.4 0.2 1"/></body></worldbody></mujoco>')
    return str(p)


def min_gap(scene, pose, objname):
    model, data0, rig, ids = H.open_scene(scene)
    d = mujoco.MjData(model); d.qpos[:] = data0.qpos
    rig.teleport(d, np.array(pose["pos"]), np.array(pose["euler"])); rig.set_fingers(d, pose["finger_targets"]); mujoco.mj_forward(model, d)
    og = [g for g in range(model.ngeom) if model.geom(g).name.startswith(objname)]
    fin = [g for g in rig.collision_gids if any(f in model.body(int(model.geom_bodyid[g])).name for f in ("index", "middle", "ring", "pinky"))]
    best = 1.0
    for g in fin:
        for o in og:
            fr = np.zeros(6); best = min(best, mujoco.mj_geomDistance(model, d, g, o, 0.2, fr))
    return best


def case(name, scene, kw, objname, lo=-0.0006, hi=0.0015):
    global fails
    try:
        p = flat_hand_pose_seated(scene, **kw)
        g = min_gap(scene, p, objname)
        ok = lo <= g <= hi and abs(p["gap_m"] - kw.get("clearance_m", 0.0005)) < 1e-4
        print(("[PASS] " if ok else "[FAIL] ") + f"{name}: measured min finger gap {g*1000:.2f} mm (gap_before {p['gap_before_m']*1000:.1f} mm, shift {p['seated_shift_m']*1000:.1f} mm)")
        fails += 0 if ok else 1
    except Exception as e:
        print(f"[FAIL] {name}: {e!r}"); fails += 1


with tempfile.TemporaryDirectory() as tmp:
    top = {"hand_xml": str(HAND["right"]), "side": "right", "floor": True,
           "objects": [{"name": "slab", "xml": obj(tmp, "slab", (0.20, 0.20, 0.02), 0.02), "pos": [0.0, 0.0, 0.0], "quat": [1, 0, 0, 0]}]}
    case("right pad on a slab top", top, dict(point=[0.0, 0.0, 0.04], normal=[0, 0, 1], finger_dir=[0, 1, 0], contact="pad"), "slab")
    wall = {"hand_xml": str(HAND["left"]), "side": "left", "floor": True,
            "objects": [{"name": "block", "xml": obj(tmp, "block", (0.03, 0.12, 0.09), 0.09), "pos": [0.10, 0.0, 0.0], "quat": [1, 0, 0, 0]}]}
    case("left pad on a vertical face", wall, dict(point=[0.07, 0.0, 0.11], normal=[-1, 0, 0], finger_dir=[0, 1, 0], contact="pad", clearance_m=0.001), "block")
    under = {"hand_xml": str(HAND["right"]), "side": "right", "floor": True,
             "objects": [{"name": "plate", "xml": obj(tmp, "plate", (0.12, 0.12, 0.0015), 0.10), "pos": [0.0, 0.0, 0.0], "quat": [1, 0, 0, 0]}]}
    case("right back under a raised plate", under, dict(point=[0.0, 0.0, 0.0985], normal=[0, 0, -1], finger_dir=[1, 0, 0], contact="back"), "plate")
    try:
        flat_hand_pose_seated(top, point=[0, 0, 0.04], normal=[0, 0, 1], finger_dir=[0, 0, 1])
        print("[FAIL] finger_dir along the normal was accepted"); fails += 1
    except ValueError:
        print("[PASS] finger_dir along the normal refused")
    try:
        flat_hand_pose_seated(top, point=[0, 0, 0.04], normal=[0, 0, 1], finger_dir=[0, 1, 0], contact="side")
        print("[FAIL] unknown contact accepted"); fails += 1
    except ValueError:
        print("[PASS] unknown contact refused")
sys.exit(1 if fails else 0)
