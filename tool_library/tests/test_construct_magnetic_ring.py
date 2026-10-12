import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.skills.construct_magnetic_ring import construct_magnetic_ring

fails = 0


def check(name, cond, msg=""):
    global fails
    if cond:
        print(f"[PASS] {name}")
    else:
        fails += 1
        print(f"[FAIL] {name} {msg}")


def ray_hits(m, d, start, direction):
    gid = np.array([-1], dtype=np.int32)
    groups = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)  # visible groups only: the magnet (group 3) is ignored
    dist = mujoco.mj_ray(m, d, np.array(start, float), np.array(direction, float), groups, 1, -1, gid)
    return dist >= 0, int(gid[0])


def two_ring_world(xa, xb):
    ra, rb = ET.parse(xa).getroot(), ET.parse(xb).getroot()
    for sec in ("worldbody", "actuator"):
        if rb.find(sec) is None:                # a ring without a magnet has no actuator section
            continue
        if ra.find(sec) is None:
            ET.SubElement(ra, sec)
        for el in list(rb.find(sec)):
            ra.find(sec).append(el)
    return ET.tostring(ra, encoding="unicode")


tmp = Path(tempfile.mkdtemp())

# 1. default ring lying flat: compiles, mass, magnet actuator, hollow bore, wall where it should be
b = construct_magnetic_ring(tmp / "a")
m = mujoco.MjModel.from_xml_path(b["xml"])
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
act_ok = m.nu == 1 and m.actuator_trntype[0] == mujoco.mjtTrn.mjTRN_BODY and abs(m.actuator_biasprm[0, 0] - 2.0) < 1e-9
mass_ok = abs(m.body_subtreemass[1] - 0.012) < 1e-4
bore_hit, _ = ray_hits(m, d, [0, 0, 0.1], [0, 0, -1])
wall_hit, _ = ray_hits(m, d, [0.5 * (b["outer_radius_m"] + b["inner_radius_m"]), 0, 0.1], [0, 0, -1])
low = min(d.geom_xpos[g][2] - m.geom_size[g][2] for g in range(m.ngeom) if m.geom_group[g] < 3)
check("default ring: compiles, 12 g, body-transmission magnet actuator, hollow bore, wall, rests at z=0",
      act_ok and mass_ok and not bore_hit and wall_hit and abs(low) < 1e-6,
      f"act {act_ok} mass {m.body_subtreemass[1]} bore_hit {bore_hit} wall_hit {wall_hit} low {low}")

# 2. a bigger thin ring standing on its rim (axis horizontal), 16 segments, no accents
b2 = construct_magnetic_ring(tmp / "b", name="big", outer_radius=0.025, inner_radius=0.02, height=0.008,
                             n_segments=16, accent_every=0, axis="horizontal", mass=0.02, magnet_force_N=3.0)
m2 = mujoco.MjModel.from_xml_path(b2["xml"])
d2 = mujoco.MjData(m2)
mujoco.mj_forward(m2, d2)
c = np.array(b2["centre_local"])
bore2, _ = ray_hits(m2, d2, c + [0, 0.1, 0], [0, -1, 0])
rim2, _ = ray_hits(m2, d2, c + [0, 0.1, 0.0225], [0, -1, 0])
corners = []
for g in range(m2.ngeom):
    if m2.geom_group[g] >= 3:
        continue
    R = d2.geom_xmat[g].reshape(3, 3)
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                corners.append(d2.geom_xpos[g] + R @ (m2.geom_size[g] * [sx, sy, sz]))
zmin = min(p[2] for p in corners)
check("horizontal 50 mm ring: bore open along the axis, rim present, lowest point on z=0, 3 N magnet",
      (not bore2) and rim2 and abs(zmin) < 0.0015 and abs(m2.actuator_biasprm[0, 0] - 3.0) < 1e-9,
      f"bore {bore2} rim {rim2} zmin {zmin}")

# 3. physics: two rings lying flat side by side, a 0.15 N sideways pull on one for 0.5 s (more than one ring's floor
#    friction, less than two rings'):
#    with magnets they stay together, without magnets the pulled ring slides away
def pair_gap(force):
    r = 0.0155 + 0.0003
    a = construct_magnetic_ring(tmp / f"pa{force}", name="pa", magnet_force_N=force, include_floor=True)
    bb = construct_magnetic_ring(tmp / f"pb{force}", name="pb", magnet_force_N=force, pos=[2 * r + 0.0002, 0, 0])
    mm = mujoco.MjModel.from_xml_string(two_ring_world(a["xml"], bb["xml"]))
    dd = mujoco.MjData(mm)
    bid = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_BODY, "pb")
    for k in range(1000):
        dd.xfrc_applied[bid, 0] = 0.15 if k >= 300 and k < 800 else 0.0
        mujoco.mj_step(mm, dd)
    # ring centres from the bodies (a ring without a magnet has no shell to measure from)
    pa_, pb_ = (dd.xpos[mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_BODY, n)] for n in ("pa", "pb"))
    zs = [pa_[2], pb_[2]]
    return float(np.linalg.norm(pa_ - pb_) - 2 * r), zs


gap_on, z_on = pair_gap(2.0)
gap_off, z_off = pair_gap(0.0)
check("two flat rings under a 0.15 N pull: magnets keep them together on the floor, without magnets they part",
      gap_on < 0.001 and gap_off > 0.005 and max(abs(z) for z in z_on) < 0.001,
      f"gap_on {gap_on} gap_off {gap_off} z {z_on} {z_off}")

# 4. a ring with magnet_force_N 0 is a plain ring: no magnet body, shell or actuator for the scene gates to flag
#    (hand_contact_pair_mask_check saw a collision-only shell, actuator_mapping_check a zero-force actuator)
b0 = construct_magnetic_ring(tmp / "plain", name="bead", magnet_force_N=0.0)
m0 = mujoco.MjModel.from_xml_path(b0["xml"])
names0 = [m0.body(i).name for i in range(m0.nbody)] + [m0.geom(i).name for i in range(m0.ngeom)]
check("zero magnet force builds a plain ring: no actuator, no magnet body or shell, magnet fields None, same mass",
      m0.nu == 0 and not any("magnet" in n for n in names0) and b0["magnet_body"] is None and b0["magnet_geom"] is None
      and b0["magnet_actuator"] is None and abs(m0.body_subtreemass[1] - 0.012) < 1e-4 and len(b0["wall_geoms"]) == 24,
      f"nu {m0.nu} magnet names {[n for n in names0 if 'magnet' in n]} mass {m0.body_subtreemass[1]}")
check("a ring with a magnet keeps its shell and actuator", b["magnet_geom"] == "magnetic_ring_magnet_shell" and m.nu == 1)

# 5. misuse fails loudly
errs = 0
for bad in ({"radius": 0.01}, {"inner_radius": 0.02, "outer_radius": 0.015}, {"magnet_bit": 3}, {"magnet_shape": "cube"}):
    try:
        construct_magnetic_ring(tmp / "bad", **bad)
    except ValueError:
        errs += 1
check("unknown parameter and impossible geometry raise ValueError", errs == 4, f"{errs}/4 raised")

sys.exit(1 if fails else 0)
