"""Which object geoms a hand scene pairs with the hand (common/hand.build_hand_scene).

The hand assets ship with contacts off and touch only the object geoms they get explicit pairs with. A pair used to be
made for EVERY object geom, overriding the geom's own contype/conaffinity: an EgoDex magnetic ring's magnet sphere
(collision bit 4, meant only for ring-ring contact) became solid to the hand and filled the ring's bore. These cases
check that masks are respected, that a geom can opt out by class, name or list, that an object without any collision
bits is still paired whole, and that a pair region leaves far geoms out.

    python tool_library/tests/test_hand_scene_pairs.py
"""
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.common import hand as H  # noqa: E402
from tool_library.skills.construct_magnetic_ring import construct_magnetic_ring  # noqa: E402

HAND = os.environ.get("DEXAGENT_HAND_XML") or str(ROOT / "setup" / "robot" / "sharpa_right.xml")
TMP = Path(tempfile.mkdtemp(prefix="dexagent_pairs_", dir=os.environ.get("TMPDIR")))
FAILS = 0
T0 = time.time()


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def pair_names(model):
    return [model.pair(i).name for i in range(model.npair)]


def write(name, body):
    p = TMP / f"{name}.xml"
    p.write_text(f'<mujoco model="{name}"><default><default class="nohand"><geom contype="1" conaffinity="1"/></default></default>'
                 f'<worldbody>{body}</worldbody></mujoco>')
    return str(p)


# 1. the magnetic ring: walls paired, the magnet sphere (bit 4) not, and a fingertip in the bore meets no magnet
ring = construct_magnetic_ring(str(TMP / "ring"), name="ring")
model, data0, rig, ids = H.build_hand_scene([{"name": "ring", "xml": ring["xml"], "pos": [0.0, 0.3, 0.0]}], HAND, "right", settle_s=0.2)
names = pair_names(model)
magnet = [g for g in range(model.ngeom) if "magnet" in (model.geom(g).name or "")]
walls = [g for g in range(model.ngeom) if "_wall" in (model.geom(g).name or "")]
check("ring: walls paired, magnet sphere not",
      magnet and walls and not any("magnet" in n for n in names) and any("_wall" in n for n in names)
      and any("magnet" in k and "never meet" in v for k, v in ids["unpaired"].items()),
      f"{len(names)} pairs, unpaired {ids['unpaired']}")
check("ring: the magnet is not among the geoms clearances are measured against",
      not (set(magnet) & set(ids["obj_gids"])) and set(walls) <= set(ids["obj_gids"]))
# put the index tip on the ring's axis at mid-height, fingers down: the hand overlaps the magnet sphere, not the walls
d = mujoco.MjData(model); d.qpos[:] = data0.qpos
eul = np.array([np.pi, 0.0, 0.0])
rig.teleport(d, [0.0, 0.3, 0.2], eul); mujoco.mj_forward(model, d)
tip = d.site_xpos[rig.fingertip_sites["index"]].copy()
centre = np.array([0.0, 0.3, 0.5 * ring["height_m"]]) if "height_m" in ring else np.array([0.0, 0.3, 0.006])
rig.teleport(d, np.array([0.0, 0.3, 0.2]) + (centre - tip), eul); mujoco.mj_forward(model, d)
mag_dist = min(H.hand_geom_distance(model, d, g, magnet[0], 0.05) for g in rig.tip_geoms("index"))
touch_mag = [i for i in range(d.ncon) if magnet[0] in (int(d.contact[i].geom1), int(d.contact[i].geom2))
             and (int(d.contact[i].geom1) in rig.geom_ids or int(d.contact[i].geom2) in rig.geom_ids)]
check("ring: an index tip inside the bore overlaps the magnet geometrically but gets no contact from it",
      mag_dist < -0.002 and not touch_mag, f"tip-magnet distance {1000 * mag_dist:.1f} mm, contacts {len(touch_mag)}")

# 2. opt-outs: class ending in 'nohand', a name ending in '_nohand', and the no_hand_pairs list; visual geoms unpaired
box = write("box", '<body name="b" pos="0 0 0.02"><freejoint/>'
                   '<geom name="shell" type="box" size=".03 .03 .02" mass=".1"/>'
                   '<geom name="skin" type="box" size=".031 .031 .021" contype="0" conaffinity="0" mass="0"/>'
                   '<geom name="knob" class="nohand" type="sphere" size=".005" pos="0 0 .02" mass=".001"/>'
                   '<geom name="tab_nohand" type="box" size=".005 .005 .005" pos=".03 0 0" mass=".001"/>'
                   '<geom name="lug" type="box" size=".005 .005 .005" pos="-.03 0 0" mass=".001"/></body>')
model, data0, rig, ids = H.build_hand_scene([{"name": "bx", "xml": box, "pos": [0.0, 0.3, 0.0]}], HAND, "right", settle_s=0.1,
                                            no_hand_pairs=["bx_lug"])
names = pair_names(model)
un = ids["unpaired"]
check("opt-outs: class 'nohand', name '_nohand' and no_hand_pairs leave the geom unpaired; the visual skin too",
      un.get("bx_knob") == "opted out" and un.get("bx_tab_nohand") == "opted out" and un.get("bx_lug") == "opted out"
      and "visual" in un.get("bx_skin", "") and any("bx_shell" in n for n in names)
      and not any(k in n for n in names for k in ("bx_knob", "bx_tab_nohand", "bx_lug", "bx_skin")), str(un))
# the same through open_scene's scene description
model2, _, _, ids2 = H.open_scene({"hand_xml": HAND, "side": "right", "objects": [{"name": "bx", "xml": box, "pos": [0.0, 0.3, 0.0]}],
                                   "no_hand_pairs": ["bx_shell"]}, settle_s=0.1)
check("open_scene passes no_hand_pairs through", ids2["unpaired"].get("bx_shell") == "opted out"
      and not any("bx_shell" in n for n in pair_names(model2)))

# 3. an object with no collision bits at all (a mesh-only stand-in) is still paired whole
ghost = write("ghost", '<body name="g" pos="0 0 0.02"><geom name="blob" type="box" size=".02 .02 .02" contype="0" conaffinity="0"/></body>')
model, data0, rig, ids = H.build_hand_scene([{"name": "gh", "xml": ghost, "pos": [0.0, 0.3, 0.0]}], HAND, "right", settle_s=0.0)
check("an object without collision bits is paired whole (as before)", any("gh_blob" in n for n in pair_names(model)) and not ids["unpaired"])

# 4. hand_bits: a hand meant to meet bit 2 pairs with a bit-2 geom and not with a bit-1 one
two = write("two", '<body name="t" pos="0 0 0.02"><geom name="a" type="box" size=".02 .02 .02" contype="2" conaffinity="2"/>'
                   '<geom name="c" type="box" size=".01 .01 .01" pos="0 0 .03"/></body>')
model, data0, rig, ids = H.build_hand_scene([{"name": "tw", "xml": two, "pos": [0.0, 0.3, 0.0]}], HAND, "right", settle_s=0.0, hand_bits=2)
names = pair_names(model)
check("hand_bits=2 pairs the bit-2 geom only", any("tw_a" in n for n in names) and not any("tw_c" in n for n in names))

# 5. a pair region: the near cube is paired, the one a metre away is not, and the floor always is
near = write("near", '<body name="n" pos="0 0 0.02"><freejoint/><geom name="cube" type="box" size=".02 .02 .02"/></body>')
far = write("far", '<body name="f" pos="0 0 0.02"><freejoint/><geom name="cube" type="box" size=".02 .02 .02"/></body>')
objs = [{"name": "far", "xml": far, "pos": [1.0, 0.3, 0.0]}, {"name": "near", "xml": near, "pos": [0.0, 0.3, 0.0]}]
m_all, _, _, ids_all = H.build_hand_scene(objs, HAND, "right", settle_s=0.0)
m_reg, _, rig_r, ids_reg = H.build_hand_scene(objs, HAND, "right", settle_s=0.0, pair_region={"points": [[0.0, 0.3, 0.02]]})
# object-object contacts (contact_defaults' rigid_default__ pairs between unspecified rigid assets) are not hand pairs:
# the pair region bounds only what the hand is paired with
names = [n for n in pair_names(m_reg) if not n.startswith("rigid_default__")]
span = H.hand_span(m_reg, rig_r)
check("pair region: near cube and floor paired, the far cube left out, clearances still see both",
      any("near_cube" in n for n in names) and not any("far_cube" in n for n in names) and any(n.endswith("__floor") for n in names)
      and ids_reg["unpaired"].get("far_cube") == "outside the pair region" and ids_reg["n_pairs"] < ids_all["n_pairs"]
      and len(ids_reg["obj_gids"]) == len(ids_all["obj_gids"]) and 0.1 < span < 0.4,
      f"{ids_all['n_pairs']} -> {ids_reg['n_pairs']} pairs, hand span {span:.3f} m, region {ids_reg['pair_region']}")

print(f"{'ALL PASS' if not FAILS else f'{FAILS} FAILED'} ({time.time() - T0:.0f}s)")
sys.exit(1 if FAILS else 0)
