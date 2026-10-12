"""Numerical contact defaults for otherwise unspecified, rigid MJCF assets.

MuJoCo's 20 ms default contact response allows millimetres of interpenetration
even on a gentle rigid-object landing. Use the same 6 ms response as our hand
contacts, but only for pairs of colliders whose source provenance proves no
contact response was set. Per-geom overrides would also alter mixed contacts
with soft objects through parameter averaging, so geom parameters stay intact.
Measured/explicit compliance, mechanisms and deformables are left untouched.
This changes numerical contact response, never mass, friction or geometry.
"""
from pathlib import Path
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

SOLREF = (0.006, 1.)
SOLIMP = (0.99, 0.999, 0.001, 0.5, 2.)
_DEFAULT_REF = (0.02, 1.)
_DEFAULT_IMP = (0.9, 0.95, 0.001, 0.5, 2.)


def _unspecified_rigid_source(path, seen=None):
    """Conservative source check: uncertainty means preserve the asset."""
    path = Path(path).resolve()
    seen = set() if seen is None else seen
    if path in seen:
        return False
    seen.add(path)
    try:
        root = ET.parse(path).getroot()
        for node in root.iter():
            if node.tag in ("flex", "flexcomp", "composite", "plugin", "tendon", "equality"):
                return False
            if node.tag == "joint" and node.get("type", "hinge") != "free":
                return False
            if node.tag in ("geom", "pair") and {"solref", "solimp"} & node.attrib.keys():
                return False
            if node.tag == "include":
                if not node.get("file") or not _unspecified_rigid_source(path.parent / node.get("file"), seen):
                    return False
        return True
    except (OSError, ET.ParseError, ValueError):
        return False


def eligible_geoms(spec, source_xml):
    """Source-confirmed default-valued rigid colliders, before name prefixing."""
    if not _unspecified_rigid_source(source_xml):
        return []
    if any(j.type != mujoco.mjtJoint.mjJNT_FREE for j in spec.joints):
        return []
    return [g for g in spec.geoms
            if (g.contype or g.conaffinity) and g.type != mujoco.mjtGeom.mjGEOM_PLANE
            and np.array_equal(g.solref, _DEFAULT_REF)
            and np.array_equal(g.solimp, _DEFAULT_IMP)]


def asset_default_names(spec, source_xml, prefix=""):
    return [prefix + g.name for g in eligible_geoms(spec, source_xml) if g.name]


def add_rigid_pairs(spec, names):
    """Profile only eligible pairs which ordinary MuJoCo collision enables.

    Preserve friction, contact dimension, margin and gap mixing. Existing
    explicit pairs win, including custom parameters equal to engine defaults.
    """
    # Mixing/filtering below is verified against this engine's implementation.
    # On other engine versions preserve ordinary contacts until revalidated.
    if tuple(map(int, mujoco.__version__.split(".")[:2])) != (3, 10) or len(set(names)) < 2:
        return []
    model = spec.compile()
    if mujoco.get_mjcb_contactfilter() is not None:
        return []
    sleep_flag = getattr(mujoco.mjtEnableBit, "mjENBL_SLEEP", 0)
    if int(model.opt.enableflags) & int(sleep_flag):
        return []
    ids = sorted({model.geom(name).id for name in names})
    existing = {frozenset((pair.geomname1, pair.geomname2)) for pair in spec.pairs}
    excluded = set(map(int, model.exclude_signature))
    filter_parent = not (int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT))
    changed = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            an, bn = model.geom(a).name, model.geom(b).name
            key = frozenset((an, bn))
            if key in existing:
                continue
            ba, bb = int(model.geom_bodyid[a]), int(model.geom_bodyid[b])
            wa, wb = int(model.body_weldid[ba]), int(model.body_weldid[bb])
            if wa == wb:
                continue
            if filter_parent and wa and wb and (int(model.body_weldid[model.body_parentid[wa]]) == wb
                                               or int(model.body_weldid[model.body_parentid[wb]]) == wa):
                continue
            low, high = sorted((ba, bb))
            if (low << 16) + high in excluded:
                continue
            if not ((int(model.geom_contype[a]) & int(model.geom_conaffinity[b]))
                    or (int(model.geom_contype[b]) & int(model.geom_conaffinity[a]))):
                continue
            priority = model.geom_priority
            if priority[a] == priority[b]:
                friction = np.maximum(model.geom_friction[a], model.geom_friction[b])
                dim = max(int(model.geom_condim[a]), int(model.geom_condim[b]))
            else:
                winner = a if priority[a] > priority[b] else b
                friction, dim = model.geom_friction[winner], int(model.geom_condim[winner])
            pair = spec.add_pair()
            pair.name = f"rigid_default__{an}__{bn}"
            pair.geomname1, pair.geomname2 = an, bn
            pair.solref, pair.solimp = SOLREF, SOLIMP
            pair.solreffriction = [0., 0.]
            pair.condim = dim
            pair.friction = [friction[0], friction[0], friction[1], friction[2], friction[2]]
            # MuJoCo 3.10 collision-driver getMargin/getGap use sums.
            pair.margin = float(model.geom_margin[a] + model.geom_margin[b])
            pair.gap = float(model.geom_gap[a] + model.geom_gap[b])
            existing.add(key)
            changed.append(pair.name)
    return changed


def _same_asset_geom(source, target, source_name, target_name, prefix):
    """Check compiled geometry and the local fixed-body tree, not mesh labels."""
    a, b = source.geom(source_name).id, target.geom(target_name).id
    for key in ("geom_type", "geom_size", "geom_pos", "geom_quat", "geom_contype", "geom_conaffinity",
                "geom_condim", "geom_friction", "geom_solref", "geom_solimp", "geom_priority",
                "geom_solmix", "geom_margin", "geom_gap"):
        if not np.allclose(getattr(source, key)[a], getattr(target, key)[b], rtol=1e-5, atol=1e-7):
            return False
    kind = int(source.geom_type[a])
    if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
        ma, mb = int(source.geom_dataid[a]), int(target.geom_dataid[b])
        for values, address, count in (("mesh_vert", "mesh_vertadr", "mesh_vertnum"),
                                       ("mesh_face", "mesh_faceadr", "mesh_facenum")):
            aa, ab = int(getattr(source, address)[ma]), int(getattr(target, address)[mb])
            na, nb = int(getattr(source, count)[ma]), int(getattr(target, count)[mb])
            if na != nb or not np.allclose(getattr(source, values)[aa:aa + na],
                                          getattr(target, values)[ab:ab + nb], rtol=1e-5, atol=1e-7):
                return False
    elif kind >= int(mujoco.mjtGeom.mjGEOM_HFIELD) and kind not in (
            int(mujoco.mjtGeom.mjGEOM_SPHERE), int(mujoco.mjtGeom.mjGEOM_CAPSULE),
            int(mujoco.mjtGeom.mjGEOM_ELLIPSOID), int(mujoco.mjtGeom.mjGEOM_CYLINDER),
            int(mujoco.mjtGeom.mjGEOM_BOX)):
        return False
    ba, bb = int(source.geom_bodyid[a]), int(target.geom_bodyid[b])
    while ba:
        if target.body(bb).name != prefix + source.body(ba).name:
            return False
        parent = int(source.body_parentid[ba])
        if parent:
            for key in ("body_pos", "body_quat"):
                if not np.allclose(getattr(source, key)[ba], getattr(target, key)[bb], rtol=1e-5, atol=1e-7):
                    return False
        ba, bb = parent, int(target.body_parentid[bb])
    return True


def _same_world_but_stations(scene, source):
    """A scene derived from the manifest's scene by moving the robot stations (place_arm_stations' scene_stations.xml):
    every non-robot body and geom is the source's, at the same pose."""
    try:
        a, b = mujoco.MjModel.from_xml_path(str(scene)), mujoco.MjModel.from_xml_path(str(source))
    except Exception:
        return False
    def world(m):
        out = {}
        for i in range(1, m.nbody):
            name = m.body(i).name
            if name.startswith("robot_"):
                continue
            geoms = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == i]
            out[name] = (np.round(m.body_pos[i], 6).tolist(), np.round(m.body_quat[i], 6).tolist(), len(geoms),
                         [int(m.geom_type[g]) for g in geoms], [int(m.geom_contype[g]) for g in geoms])
        return out
    return world(a) == world(b)


def apply_scene_defaults(spec, scene_xml, scene_recon_json=None):
    """Legacy composed scenes need raw source provenance: serialization loses it.

    A sibling manifest is usable only if it identifies this exact scene. Source
    and composed colliders must agree before filling an unspecified response.
    A scene without verifiable provenance retains its original parameters.
    """
    scene = Path(scene_xml).resolve()
    manifest = Path(scene_recon_json).resolve() if scene_recon_json else scene.with_name("scene_recon.json")
    if not manifest.is_file():
        return []
    try:
        recon = json.loads(manifest.read_text())
        source_scene = (manifest.parent / recon.get("scene_xml", "scene.xml")).resolve()
        if source_scene != scene and not _same_world_but_stations(scene, source_scene):
            return []
        objects = recon.get("objects", [])
        if not isinstance(objects, list):
            return []
    except (OSError, ValueError, TypeError):
        return []
    targets = {g.name: g for g in spec.geoms if g.name}
    compiled_target = None
    changed = []
    for obj in objects:
        if not isinstance(obj, dict) or obj.get("type") != "rigid" or not obj.get("asset_xml") or not obj.get("id"):
            continue
        source = (manifest.parent / obj["asset_xml"]).resolve()
        if not _unspecified_rigid_source(source):
            continue
        child = mujoco.MjSpec.from_file(str(source))
        compiled_source = child.compile()
        if compiled_target is None:
            compiled_target = spec.compile()
        prefix = str(obj["id"]) + "_"
        for original in eligible_geoms(child, source):
            if not original.name:
                continue
            target = targets.get(prefix + original.name)
            if target is None or target.parent.name != prefix + original.parent.name:
                continue
            # A customized scene, including an explicitly changed contact
            # response, never gets silently replaced with the asset default.
            if not _same_asset_geom(compiled_source, compiled_target, original.name, target.name, prefix):
                continue
            changed.append(target.name)
    return add_rigid_pairs(spec, changed)
