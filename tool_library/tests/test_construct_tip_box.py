"""construct_tip_box: lid inertia from its own plates, drawn tip stubs inside collision; construct_conical_tube_rack
with several rows of holes.

    python tool_library/tests/test_construct_tip_box.py

The tip box hand-wrote the lid inertia at 1e-7 kg m^2 (38x too small for its plates) and drew the 95 moulded tips as
visual-only stubs standing 15.6 mm above any collision geometry; the conical tube rack cut top-plate openings for
the first row of holes only.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.skills.construct_conical_tube_rack import construct_conical_tube_rack  # noqa: E402
from tool_library.skills.construct_tip_box import construct_tip_box  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dexagent_tipbox_", dir=os.environ.get("TMPDIR")))
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def plate_inertia(m, body, mass):
    """The inertia (principal moments, ascending) of the body's plates as uniform solids carrying `mass` in total."""
    spec = mujoco.MjSpec()
    b = spec.worldbody.add_body(name="lid")
    b.add_freejoint()
    vols = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == body and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            vols.append((g, 8 * float(np.prod(m.geom_size[g]))))
    tot = sum(v for _, v in vols)
    for g, v in vols:
        x = b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=m.geom_size[g].tolist(), pos=m.geom_pos[g].tolist(),
                       quat=m.geom_quat[g].tolist())
        x.mass = mass * v / tot
    mm = spec.compile()
    return np.sort(mm.body_inertia[1])


for i, kw in enumerate([dict(free_wells=[[4, 6]]), dict(free_wells=[[0, 0], [2, 3]], lid_mass=0.025, lid_deg=30.0)]):
    b = construct_tip_box(TMP / f"box{i}", **kw)
    m = mujoco.MjModel.from_xml_path(b["xml"])
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    lid = m.body(b["lid_body"]).id
    want_mass = float(kw.get("lid_mass", 0.001))
    I_model, I_plates = np.sort(m.body_inertia[lid]), plate_inertia(m, lid, want_mass)

    # 1. the lid's mass and inertia are its plates'
    check(f"[{i}] lid mass {want_mass} kg and inertia from its own plates (was a hand-written 1e-7 kg m^2)",
          abs(m.body_mass[lid] - want_mass) < 1e-9 and np.allclose(I_model, I_plates, rtol=1e-6) and I_model.max() > 30e-7 * want_mass / 0.001,
          f"inertia {I_model} vs plates {I_plates}")

    # 2. every drawn stub stands inside a colliding cylinder of the same height in its well
    box = m.body(b["box_body"]).id
    vis = [g for g in range(m.ngeom) if m.geom(g).name.startswith(f"{b['name']}_stub") and not m.geom(g).name.startswith(f"{b['name']}_stubc")]
    col = {m.geom(g).name.replace("_stubc", "_stub"): g for g in range(m.ngeom) if m.geom(g).name.startswith(f"{b['name']}_stubc")}
    worst_top, worst_r, missing = 0.0, 0.0, []
    for g in vis:
        c = col.get(m.geom(g).name)
        if c is None or not (m.geom_contype[c] or m.geom_conaffinity[c]):
            missing.append(m.geom(g).name)
            continue
        top_v = d.geom_xpos[g][2] + m.geom_size[g][1]
        top_c = d.geom_xpos[c][2] + m.geom_size[c][1]
        worst_top = max(worst_top, top_v - top_c)
        worst_r = max(worst_r, m.geom_size[g][0] - m.geom_size[c][0])
    check(f"[{i}] all {len(vis)} drawn stubs collide to their full height (radius within 0.7 mm of the drawing)",
          not missing and worst_top <= 1e-9 and worst_r <= 0.0007 and len(vis) == 96 - len(kw["free_wells"]),
          f"missing {missing[:3]}, top excess {worst_top * 1000:.2f} mm, radius excess {worst_r * 1000:.2f} mm")

    # 3. the free tips still settle in their sockets with the colliding stubs beside them
    tips = [m.body(t["body"]).id for t in b["free_tips"]]
    z0 = d.xpos[tips].copy()
    for _ in range(int(1.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    moved = float(np.linalg.norm(d.xpos[tips] - z0, axis=1).max())
    tilt = max(float(np.degrees(np.arccos(np.clip(d.xmat[t].reshape(3, 3)[2, 2], -1, 1)))) for t in tips)
    check(f"[{i}] free tips stay seated for 1 s next to colliding stubs (moved <= 2 mm, tilt <= 3 deg)",
          moved < 0.002 and tilt < 3.0 and np.isfinite(d.qpos).all(), f"moved {moved * 1000:.2f} mm, tilt {tilt:.2f} deg")

# 4. a hand-sized probe pressed down onto a stub is stopped at the stub top, not at the tray
b = construct_tip_box(TMP / "probe", include_floor=True)
xml = Path(b["xml"]).read_text().replace(
    "</worldbody>", '<body name="probe" pos="0.0045 0.0045 0.11"><freejoint/><geom type="sphere" size="0.006" mass="0.02"/></body></worldbody>')
spec = mujoco.MjSpec.from_string(xml)
spec.meshdir = str(Path(b["xml"]).parent / "meshes")
mp = spec.compile()
dp = mujoco.MjData(mp)
mujoco.mj_forward(mp, dp)
pid = mp.body("probe").id
for _ in range(int(1.5 / mp.opt.timestep)):
    mujoco.mj_step(mp, dp)
check("a fingertip-sized sphere dropped on the standing tips rests on them, ~15 mm above the tray",
      dp.xpos[pid][2] - 0.006 > b["tray_top_m"] + 0.012, f"sphere bottom {1000 * (dp.xpos[pid][2] - 0.006):.1f} mm, tray {1000 * b['tray_top_m']:.1f} mm")

# 5. conical tube rack: every hole of a 2 x 3 grid is a real opening in the top plate
r = construct_conical_tube_rack(TMP / "rack", rows=2, cols=3, width=0.075, pitch_y=0.030, hole_r=0.0105)
m = mujoco.MjModel.from_xml_path(r["xml"])
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
H = r["hole_top_z_m"]
for g in range(m.ngeom):                                # rays see only what collides (the moulded rims are drawn only)
    if not (m.geom_contype[g] or m.geom_conaffinity[g]):
        m.geom_group[g] = 5
solid = np.array([1, 1, 1, 1, 1, 0], np.uint8)
open_ok, blocked_ok = True, True
for h in r["holes"]:
    x, y = h["xy"]
    gid = np.array([-1], np.int32)
    dist = mujoco.mj_ray(m, d, np.array([x, y, H + 0.05]), np.array([0, 0, -1.0]), solid, 1, -1, gid)
    open_ok &= dist > 0.05 + 0.01                      # passes the top plate, lands near the bottom
x0, y0 = r["holes"][0]["xy"]
gid = np.array([-1], np.int32)
dist = mujoco.mj_ray(m, d, np.array([x0 + 0.02, y0 + 0.015, H + 0.05]), np.array([0, 0, -1.0]), solid, 1, -1, gid)
blocked_ok = abs(dist - 0.05) < 0.002
check("2 x 3 conical rack: a ray down every hole centre passes the top plate, one between holes hits it",
      open_ok and blocked_ok and len(r["holes"]) == 6, f"between-holes ray {dist:.3f} m")
try:
    construct_conical_tube_rack(TMP / "rack_bad", rows=3, width=0.062, pitch_y=0.030)
    check("rows that do not fit between the side walls are refused", False)
except ValueError as e:
    check("rows that do not fit between the side walls are refused", "side walls" in str(e), str(e)[:90])

print("ALL PASS" if not FAILS else f"{FAILS} FAILED")
sys.exit(1 if FAILS else 0)
