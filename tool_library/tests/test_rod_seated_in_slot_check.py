import os, sys, tempfile
import numpy as np, mujoco
from scipy.spatial.transform import Rotation as Rot
from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.verifiers.rod_seated_in_slot_check import rod_seated_in_slot_check

XML = """<mujoco><worldbody><geom type="plane" size="1 1 0.1"/>
<body name="tray" pos="0 0 0"><freejoint/><geom type="box" size="0.03 0.015 0.001" pos="0 0 0.001"/></body>
<body name="rod" pos="0 0 0.1"><freejoint/><geom type="cylinder" size="0.005 {h}" pos="0 0 {h}"/></body>
</worldbody></mujoco>"""
fails = 0
def case(name, cond):
    global fails
    print(("[PASS] " if cond else "[FAIL] ") + name); fails += 0 if cond else 1

def state(model, tray_xy=(0, 0), tray_yaw=0.0, rod_tilt=0.0, rod_off=(0.0, 0.0, 0.0), rod_len=0.044, slot_c=(0.0, -0.007, 0.006)):
    q = model.qpos0.copy()
    Rt = Rot.from_euler('z', tray_yaw, degrees=True)
    q[0:3] = [tray_xy[0], tray_xy[1], 0.0]; w = Rt.as_quat(); q[3:7] = [w[3], w[0], w[1], w[2]]
    # rod axis (local z) along the tray's +x, tilted up by rod_tilt about the tray's y, root at the slot's +x end
    Rr_local = Rot.from_euler('y', -90.0 + rod_tilt, degrees=True)
    centre_local = np.array(slot_c) + np.array(rod_off)
    root_local = centre_local - Rr_local.apply([0, 0, rod_len / 2])
    q[7:10] = np.array(tray_xy + (0.0,)) + Rt.apply(root_local)
    w = (Rt * Rr_local).as_quat(); q[10:14] = [w[3], w[0], w[1], w[2]]
    return q

for h, label in ((0.022, "AAA-length rod"), (0.0125, "short dowel")):
    L = 2 * h
    m = mujoco.MjModel.from_xml_path(os.path.join(tempfile.mkdtemp(), "x.xml")) if False else mujoco.MjModel.from_xml_string(XML.format(h=h))
    kw = dict(body="rod", holder="tray", slot_center=[0.0, -0.007, 0.006], rod_center=[0, 0, h], rod_half_length=h, stops=[-h - 0.002, h + 0.002])
    r = rod_seated_in_slot_check(m, state(m, rod_len=L), **kw)
    case(f"{label}: seated in an unturned tray passes", r["ok"] and r["angle_deg"] < 0.5 and r["radial_m"] < 1e-4)
    r = rod_seated_in_slot_check(m, state(m, tray_xy=(0.05, -0.03), tray_yaw=-133.0, rod_len=L), **kw)
    case(f"{label}: seated in a tray that slid and turned 133 deg passes (holder frame)", r["ok"])
    r = rod_seated_in_slot_check(m, state(m, rod_tilt=12.0, rod_len=L), **kw)
    case(f"{label}: resting 12 deg up on its end fails with an angle issue", (not r["ok"]) and abs(r["angle_deg"] - 12.0) < 0.5 and any("deg off" in i for i in r["issues"]))
    r = rod_seated_in_slot_check(m, state(m, rod_off=(0.0, 0.0, 0.004), rod_len=L), **kw)
    case(f"{label}: lying 4 mm above the channel fails with a radial issue", (not r["ok"]) and any("off the slot's axis line" in i for i in r["issues"]))
    r = rod_seated_in_slot_check(m, state(m, rod_off=(-0.006, 0.0, 0.0), rod_len=L), **kw)
    case(f"{label}: slid 6 mm past an end stop fails with a stop issue", (not r["ok"]) and any("outside the stops" in i for i in r["issues"]))
    # a trajectory: tilted, tilted, seated, seated -> seated from frame 2
    Q = np.stack([state(m, rod_tilt=20, rod_len=L), state(m, rod_tilt=8, rod_len=L), state(m, rod_len=L), state(m, rod_len=L)])
    path = os.path.join(tempfile.mkdtemp(), "traj.npz"); np.savez(path, qpos=Q)
    r = rod_seated_in_slot_check(m, path, settle_frames=2, **kw)
    case(f"{label}: npz trajectory reports seated_from_frame 2", r["ok"] and r["seated_from_frame"] == 2 and r["n_frames"] == 4)
m = mujoco.MjModel.from_xml_string(XML.format(h=0.02))
try:
    rod_seated_in_slot_check(m, None, body="rod", holder="tray", slot_center=[0, 0, 0], bogus=1)
    case("unknown parameter raises ValueError", False)
except ValueError:
    case("unknown parameter raises ValueError", True)
try:
    rod_seated_in_slot_check(m, np.zeros((3, m.nq + 1)), body="rod", holder="tray", slot_center=[0, 0, 0])
    case("wrong qpos width raises ValueError", False)
except ValueError:
    case("wrong qpos width raises ValueError", True)
sys.exit(1 if fails else 0)
