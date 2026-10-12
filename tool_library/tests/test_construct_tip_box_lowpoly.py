"""Test construct_tip_box_lowpoly on three box variations: the box settles with its free tips seated and its lid held,
the drawn collars lie within 5 mm of the physics, the lid inertia comes from its plates, the collision budget holds,
and a force-driven micropipette press snaps the free tip onto the nose and carries it up out of its socket."""
import json
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np

from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.common.mjcf import ForceGrip, clear_forces, vec
from tool_library.common.pipettekit import TIP_DEFAULTS, build_micropipette
from tool_library.skills.construct_tip_box_lowpoly import construct_tip_box_lowpoly

FAIL = []


def check(case, cond, msg):
    if not cond:
        FAIL.append(f"{case}: {msg}")
        print(f"[FAIL] {case}: {msg}")
    return cond


def settle(built, case):
    m = mujoco.MjModel.from_xml_path(built["xml"])
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    tips = [m.body(t["body"]).id for t in built["free_tips"]]
    p0 = [d.xpos[b].copy() for b in tips]
    lid_j = m.joint(built["lid_joint"]).id
    a0 = float(d.qpos[m.jnt_qposadr[lid_j]])
    for _ in range(int(1.5 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    drift = max([float(np.linalg.norm(d.xpos[b] - q)) for b, q in zip(tips, p0)] or [0.0])
    tilt = max([float(np.degrees(np.arccos(np.clip(d.xmat[b].reshape(3, 3)[2, 2], -1, 1)))) for b in tips] or [0.0])
    dlid = abs(np.degrees(float(d.qpos[m.jnt_qposadr[lid_j]]) - a0))
    check(case, np.all(np.isfinite(d.qpos)), "simulation diverged")
    check(case, drift < 0.0015, f"a free tip drifted {1000 * drift:.2f} mm while settling")
    check(case, tilt < 2.0, f"a free tip tilted {tilt:.2f} deg in its socket")
    check(case, dlid < 2.0, f"the lid moved {dlid:.2f} deg on its friction hinge")
    lid = m.body(built["lid_body"]).id
    check(case, float(m.body_inertia[lid].min()) > 1e-6, f"lid inertia {m.body_inertia[lid]} is not its plates'")
    # drawn collars: their top must stand at most 5 mm above the slab top
    expo = built["tip_collar_top_m"] - built["slab_top_m"]
    check(case, 0 < expo <= 0.0045, f"drawn collars stand {1000 * expo:.1f} mm above the slabs")
    return {"drift_mm": round(1000 * drift, 3), "tilt_deg": round(tilt, 3), "lid_moved_deg": round(dlid, 3)}


def mount(built, case, tip_index=0):
    """Press a micropipette (force grip, no robot) down onto a free tip, then lift it 7 cm: the tip must ride on the nose."""
    tmp = Path(built["xml"]).parent
    t = built["free_tips"][tip_index]
    m0 = mujoco.MjModel.from_xml_path(built["xml"])
    d0 = mujoco.MjData(m0)
    mujoco.mj_forward(m0, d0)
    px, py = (float(v) for v in d0.xpos[m0.body(t["body"]).id][:2])
    tip_top = float(built["tip_collar_top_m"])
    pip = build_micropipette({"name": "mpip", "layout": "upright", "pos": [0, 0, 0], "include_floor": False}, tmp / "pip")
    inner = Path(pip["xml"]).read_text().split("<worldbody>")[1].split("</worldbody>")[0]
    inner = "\n".join(ln for ln in inner.splitlines() if "<light" not in ln)
    hover0 = tip_top + 0.020
    inner = inner.replace('pos="0.000000 0.000000 0.000000"', f'pos="{vec((px, py, hover0))}"', 1)
    xml = Path(built["xml"]).read_text().replace("  </worldbody>", f"{inner}\n  </worldbody>")
    xml = xml.replace('timestep="0.001"', 'timestep="0.0005"')
    xp = tmp / "mount_world.xml"
    xp.write_text(xml)
    m = mujoco.MjModel.from_xml_path(str(xp))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    pb, tipb = m.body("mpip").id, m.body(t["body"]).id
    grip = ForceGrip.sized(m, pb, stiff=2.0, max_force=28.0, max_torque=1.2)
    seat = float(pip["nose_seat_z_m"])
    qup = np.array([1.0, 0, 0, 0])

    def state():
        R = d.xmat[pb].reshape(3, 3)
        top = d.xpos[tipb] + d.xmat[tipb].reshape(3, 3) @ np.array([0, 0, float(TIP_DEFAULTS["length"])])
        v = R.T @ (top - d.xpos[pb])
        return 1000 * (v[2] - seat), 1000 * np.hypot(v[0], v[1])

    dt = float(m.opt.timestep)
    z, ok, t_mount, tsec = hover0, 0, None, 0.0
    while tsec < 7.0:
        gap, lat = state()
        z = max(hover0 - 0.048, z - (0.012 if gap < -6.0 else 0.003) * dt)
        clear_forces(d)
        grip.hold(d, target_pos=[px, py, z], target_quat=qup)
        mujoco.mj_step(m, d)
        tsec += dt
        ok = ok + 1 if (gap > -0.8 and lat < 2.0) else 0
        if ok >= 120:
            t_mount = tsec
            break
    if not check(case, t_mount is not None, f"the nose never seated on the tip (gap {state()[0]:.1f} mm)"):
        return {"mounted": False}
    z0 = z
    for i in range(int(3.0 / dt)):
        u = min(1.0, (i * dt / 2.5) ** 2)
        clear_forces(d)
        grip.hold(d, target_pos=[px, py, z0 + 0.070 * u], target_quat=qup)
        mujoco.mj_step(m, d)
    gap, lat = state()
    rise = float(d.xpos[tipb][2] - built["tip_seat_z_m"])
    check(case, abs(gap) < 2.0 and lat < 2.0, f"after the lift the tip is {gap:.1f} mm / {lat:.1f} mm off its seat on the nose")
    check(case, rise > 0.05, f"the tip rose only {1000 * rise:.1f} mm with the pipette")
    return {"mounted": True, "t_mount_s": round(t_mount, 2), "carry_gap_mm": round(gap, 2), "tip_rise_mm": round(1000 * rise, 1)}


CASES = {
    "default_one_free_well_named_tip": {"free_wells": [[0, 11]], "tip_names": ["tip"]},
    "two_free_wells_6x10_end_walls_visual": {"rows": 6, "cols": 10, "length": 0.11, "width": 0.08, "free_wells": [[2, 3], [5, 8]],
                                             "lid_end_walls_collide": False, "lid_deg": 110.0},
    "sixteen_lip_tall_base_given_rise": {"free_wells": [[4, 6]], "lip_segments": 16, "base_h": 0.08, "slab_rise": 0.012},
}

with tempfile.TemporaryDirectory() as td:
    for case, prm in CASES.items():
        n0 = len(FAIL)
        b = construct_tip_box_lowpoly(Path(td) / case, name="tbx", **prm)
        cg = b["collision_geoms"]
        if "tip_names" in prm:
            check(case, [t["body"] for t in b["free_tips"]] == [f"tbx_{n}" for n in prm["tip_names"]], f"tip bodies {b['free_tips']}")
        check(case, cg["box"] + cg["lid"] <= 16 + 4 * (len(prm["free_wells"]) - 1), f"box collision geoms {cg}")
        check(case, cg["per_tip"] <= 32 + (prm.get("lip_segments", 12) - 12), f"tip collision geoms {cg['per_tip']}")
        s = settle(b, case)
        mt = mount(b, case, tip_index=len(b["free_tips"]) - 1)
        if len(FAIL) == n0:
            print(f"[PASS] {case}: {json.dumps({'geoms': cg, **s, **mt})}")
    try:
        construct_tip_box_lowpoly(Path(td) / "bad", lip_segments=8)
        check("misuse", False, "lip_segments 8 was accepted")
    except ValueError:
        pass
    try:
        construct_tip_box_lowpoly(Path(td) / "bad2", lidd_deg=10)
        check("misuse", False, "an unknown parameter was accepted")
    except ValueError:
        pass
    try:
        construct_tip_box_lowpoly(Path(td) / "bad3", free_wells=[[0, 0]], slab_rise=0.005)
        check("misuse", False, "a slab_rise that leaves the drawn collars 9 mm proud was accepted")
    except ValueError:
        pass
    try:
        construct_tip_box_lowpoly(Path(td) / "bad4", free_wells=[[0, 0], [1, 1]], tip_names=["a"])
        check("misuse", False, "tip_names shorter than free_wells was accepted")
    except ValueError:
        pass
    if not any(f.startswith("misuse") for f in FAIL):
        print("[PASS] misuse: lip_segments 8, an unknown parameter and a too-low slab_rise and short tip_names raise ValueError")

sys.exit(1 if FAIL else 0)
