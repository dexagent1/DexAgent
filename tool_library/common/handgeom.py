"""Collision geometry of the Sharpa hand that matches what the camera sees.

The hand asset's collision primitives are smaller than its visual meshes: the palm box leaves 9 mm of the palm faces,
up to 17.5 mm at the edges, outside it, the thumb metacarpal capsule 13.5 mm and the back of the thumb tip 5.7 mm
(the finger capsules are within 2.4 mm). Physics and every check that measured the hand on its primitives therefore
let the VISIBLE palm and thumb sink a centimetre into objects and the table with no contact at all.

`add_hand_hulls` gives those three links a collision geom that is the convex hull of their visual mesh (MuJoCo
collides a mesh geom as its hull), placed exactly like the visual geom, named `<prefix>collision_hand_<side>_<link>_hull`
so every tool that pairs or measures the hand's `collision_hand_` geoms picks it up. It is idempotent and works on the
hand asset, on a hand scene (prefix `hand_`) and on a bench scene (prefix `robot_<side>_`). The pads of the fingertips
keep their capsules (`pad_radius` reads the capsule), so pinch standoffs are unchanged; the thumb tip's hull, half a
millimetre proud of its pad, counts as the pad for contact.
"""
from __future__ import annotations

import mujoco

HULL_LINKS = ("hand_C_MC", "thumb_MC", "thumb_DP")


def hull_geom_name(prefix: str, side: str, link: str) -> str:
    return f"{prefix}collision_hand_{side}_{link}_hull"


def is_hull(name: str) -> bool:
    return (name or "").endswith("_hull") and "collision_hand_" in (name or "")


def add_hand_hulls(spec) -> list:
    """Add the hull collision geoms to every Sharpa hand in `spec` (an MjSpec). Returns the names added."""
    existing = {g.name for g in spec.geoms if g.name}
    added = []
    for b in spec.bodies:
        nm = b.name or ""
        for side in ("left", "right"):
            for link in HULL_LINKS:
                tail = f"{side}_{link}"
                if not nm.endswith(tail):
                    continue
                prefix = nm[: -len(tail)]
                gname = hull_geom_name(prefix, side, link)
                if gname in existing:
                    continue
                vis = [g for g in b.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH and g.meshname
                       and "elastomer" not in g.meshname and int(g.group) <= 2]
                if not vis:
                    continue
                v = vis[0]
                g = b.add_geom()
                g.name = gname
                g.type = mujoco.mjtGeom.mjGEOM_MESH
                g.meshname = v.meshname
                g.pos = list(v.pos)
                g.quat = list(v.quat)
                g.contype = 0
                g.conaffinity = 0
                g.group = 3
                g.density = 0.0
                g.condim = 1
                g.rgba = [0.0, 1.0, 0.0, 1.0]
                existing.add(gname)
                added.append(gname)
    return added
