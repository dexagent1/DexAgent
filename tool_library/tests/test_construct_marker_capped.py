import sys, tempfile
from pathlib import Path
import numpy as np
import mujoco
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root: runnable from anywhere
from tool_library.skills.construct_marker_capped import construct_marker_capped


def trial(built, pull_N, upright_nib_down=True, seconds=1.0):
    """Hold the pen body fixed in space (nib down, gravity pulling the cap off), apply
    pull_N along the axis to the cap, return the cap slide position."""
    m = mujoco.MjModel.from_xml_path(built["xml"])
    d = mujoco.MjData(m)
    bid = m.body(built["base_body"]).id
    cid = m.body(built["cap_body"]).id
    jadr = m.jnt_qposadr[m.joint(built["cap_joint"]).id]
    fj = m.body_jntadr[bid]
    fadr = m.jnt_qposadr[fj]; fdof = m.jnt_dofadr[fj]
    # nib down: rotate local +x to world -z (pitch +90 deg about y)
    q = np.array([np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0])
    d.qpos[fadr:fadr + 3] = [0, 0, 0.3]
    d.qpos[fadr + 3:fadr + 7] = q
    mujoco.mj_forward(m, d)
    for _ in range(int(seconds / m.opt.timestep)):
        d.qpos[fadr:fadr + 3] = [0, 0, 0.3]; d.qpos[fadr + 3:fadr + 7] = q
        d.qvel[fdof:fdof + 6] = 0
        axis = d.xmat[bid].reshape(3, 3) @ np.array([1.0, 0, 0])
        d.xfrc_applied[cid, :3] = pull_N * axis
        mujoco.mj_step(m, d)
    return float(d.qpos[jadr])


variants = {
    "default": {},
    "thin_long_firm": {"length": 0.150, "radius": 0.0065, "cap_radius": 0.0078, "cap_length": 0.040,
                       "hold_force_N": 3.0, "yaw_deg": 45.0},
    "fat_short_loose": {"length": 0.110, "radius": 0.009, "cap_radius": 0.0105, "cap_length": 0.030,
                        "nib_length": 0.008, "collar_length": 0.008, "hold_force_N": 0.8},
}
fails = 0
for tag, params in variants.items():
    out = tempfile.mkdtemp()
    b = construct_marker_capped(out, **params)
    held = trial(b, 0.0)                        # gravity only, nib down
    pulled = trial(b, b["hold_force_N"] + 2.0)  # a fingertip pull above the press fit
    ok = held < 0.001 and b["engaged_depth_m"] <= pulled <= b["off_q"] + 0.003 and abs(b["cap_range"][1] - b["off_q"]) < 1e-9
    print(f"{'[PASS]' if ok else '[FAIL]'} {tag}: gravity q={held*1000:.2f} mm, pull q={pulled*1000:.1f} mm "
          f"(engaged {b['engaged_depth_m']*1000:.1f} mm)")
    fails += not ok
try:
    construct_marker_capped(tempfile.mkdtemp(), wobble=1)
    print("[FAIL] unknown parameter accepted"); fails += 1
except ValueError:
    print("[PASS] unknown parameter refused")
try:
    construct_marker_capped(tempfile.mkdtemp(), cap_radius=0.005, radius=0.0075)
    print("[FAIL] cap narrower than barrel accepted"); fails += 1
except ValueError:
    print("[PASS] cap narrower than barrel refused")
sys.exit(1 if fails else 0)
