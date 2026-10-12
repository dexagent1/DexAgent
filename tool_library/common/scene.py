"""MuJoCo scene helpers shared by the scene checks.

Everything here works on a plain MJCF file or a compiled `mujoco.MjModel`;
nothing assumes a particular robot, object naming scheme or workspace.
"""
from __future__ import annotations

import re
from pathlib import Path

import mujoco
import numpy as np

FREE = int(mujoco.mjtJoint.mjJNT_FREE)
HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
SLIDE = int(mujoco.mjtJoint.mjJNT_SLIDE)
BODY = mujoco.mjtObj.mjOBJ_BODY
JOINT = mujoco.mjtObj.mjOBJ_JOINT
CAMERA = mujoco.mjtObj.mjOBJ_CAMERA


def load_model(scene):
    """An MJCF path, an XML string or an MjModel -> MjModel."""
    if isinstance(scene, mujoco.MjModel):
        return scene
    s = str(scene)
    if s.lstrip().startswith("<"):
        return mujoco.MjModel.from_xml_string(s)
    p = Path(s)
    if p.suffix.lower() == ".mjb":
        return mujoco.MjModel.from_binary_path(str(p))
    return mujoco.MjModel.from_xml_path(str(p))


def forward(model, qpos=None):
    """MjData at the model's initial state (its first keyframe when it has one, else qpos0), or at `qpos`,
    kinematics done. A bench scene declares its home pose and the objects' initial joint states in keyframe 0."""
    data = mujoco.MjData(model)
    if qpos is not None:
        data.qpos[:] = np.asarray(qpos, float)
    elif model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return data


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def body_names(model):
    return [model.body(i).name for i in range(model.nbody)]


def resolve_body(model, name):
    """Body id for an exact name, its slug ('yellow cup' -> yellow_cup), or
    the one body that ends in '_<slug>' (an assembly part). -1 when none."""
    bid = mujoco.mj_name2id(model, BODY, str(name))
    if bid >= 0:
        return bid
    s = slug(name)
    bid = mujoco.mj_name2id(model, BODY, s)
    if bid >= 0:
        return bid
    cands = [i for i, n in enumerate(body_names(model)) if n.endswith(f"_{s}")]
    return cands[0] if len(cands) == 1 else -1


def body_id(model, name):
    bid = resolve_body(model, name)
    if bid < 0:
        raise ValueError(f"body '{name}' is not in the scene; bodies: {body_names(model)[1:]}")
    return bid


def joint_id(model, name):
    jid = mujoco.mj_name2id(model, JOINT, str(name))
    if jid < 0:
        names = [model.joint(j).name for j in range(model.njnt) if model.joint(j).name]
        raise ValueError(f"joint '{name}' is not in the scene; named joints: {names}")
    return jid


def free_bodies(model):
    """Body ids that carry a free joint."""
    return [int(model.jnt_bodyid[j]) for j in range(model.njnt)
            if int(model.jnt_type[j]) == FREE]


def has_free_joint(model, bid):
    adr, num = int(model.body_jntadr[bid]), int(model.body_jntnum[bid])
    return any(int(model.jnt_type[j]) == FREE for j in range(adr, adr + max(0, num)))


def subtree(model, bid):
    out, stack = set(), [bid]
    while stack:
        b = stack.pop()
        out.add(b)
        stack += [c for c in range(model.nbody) if model.body_parentid[c] == b and c != b]
    return out


def collidable_geoms(model, bids):
    bids = set(bids)
    return [g for g in range(model.ngeom)
            if int(model.geom_bodyid[g]) in bids
            and (model.geom_contype[g] != 0 or model.geom_conaffinity[g] != 0)]


def contact_pairs(model, data):
    """[(body a, body b, signed distance)] for the current contacts."""
    out = []
    for i in range(data.ncon):
        c = data.contact[i]
        a = model.body(int(model.geom_bodyid[c.geom1])).name
        b = model.body(int(model.geom_bodyid[c.geom2])).name
        out.append((a, b, float(c.dist)))
    return out


def in_contact(model, data, bid_a, bid_b):
    na, nb = model.body(bid_a).name, model.body(bid_b).name
    return any({a, b} == {na, nb} for a, b, _ in contact_pairs(model, data))


def settle(model, data, seconds=2.0):
    """Step physics for `seconds` in place. Raises on a diverging simulation."""
    steps = int(round(seconds / model.opt.timestep))
    for _ in range(steps):
        mujoco.mj_step(model, data)
    if not np.isfinite(data.qpos).all():
        raise FloatingPointError("the simulation diverged while settling")
    return data


def rotation_angle_deg(R_a, R_b):
    R = np.asarray(R_b).reshape(3, 3) @ np.asarray(R_a).reshape(3, 3).T
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def group_world_aabb(model, data, gids):
    """World AABB over geoms. Loose for rotated meshes; good enough to place
    probe points, never for a raycast start (see probe_below)."""
    corners = []
    for g in gids:
        c, h = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
        R = data.geom_xmat[g].reshape(3, 3)
        p = data.geom_xpos[g]
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    corners.append(R @ (c + h * np.array([sx, sy, sz])) + p)
    corners = np.asarray(corners)
    return corners.min(axis=0), corners.max(axis=0)


def probe_below(model, data, x, y, z_start, own_bodies, max_bounce=40):
    """Exact downward raycast from (x, y, z_start) that skips the group's own
    geometry. Returns (hit, distance from start, hit z)."""
    geomid = np.zeros(1, dtype=np.int32)
    total = 0.0
    for _ in range(max_bounce):
        pnt = np.array([x, y, z_start - total])
        d = mujoco.mj_ray(model, data, pnt, np.array([0.0, 0.0, -1.0]), None, 1, -1, geomid)
        if d < 0:
            return False, None, None
        if int(model.geom_bodyid[int(geomid[0])]) not in own_bodies:
            return True, total + float(d), float(z_start - total - d)
        total += float(d) + 1e-4
    return False, None, None


def support_of_group(model, data, gids, grid=5, edge_pad=0.08):
    """Ray-probe under a group of geoms: fraction of its footprint that has a
    surface below, and the median gap to it (m)."""
    lo, hi = group_world_aabb(model, data, gids)
    own = {int(model.geom_bodyid[g]) for g in gids}
    span = hi[:2] - lo[:2]
    pad = edge_pad * span
    xs = np.linspace(lo[0] + pad[0], hi[0] - pad[0], grid)
    ys = np.linspace(lo[1] + pad[1], hi[1] - pad[1], grid)
    z_start = float(hi[2]) + 0.05
    hits, gaps = 0, []
    for x in xs:
        for y in ys:
            found, dist, hit_z = probe_below(model, data, x, y, z_start, own)
            if found and dist <= (z_start - lo[2]) + 0.25:
                hits += 1
                gaps.append(max(0.0, float(lo[2]) - float(hit_z)))   # base above the surface
    frac = hits / float(grid * grid)
    return {"supported_frac": round(frac, 2),
            "gap_m": None if not gaps else round(float(np.median(gaps)), 4),
            "base_z": round(float(lo[2]), 4),
            "footprint_x": [round(float(lo[0]), 3), round(float(hi[0]), 3)],
            "footprint_y": [round(float(lo[1]), 3), round(float(hi[1]), 3)]}


# ------------------------------------------------------------- rendering --
def camera_arg(model, camera):
    """A camera name, id, or a free-camera dict {lookat, distance, azimuth,
    elevation} -> something mujoco.Renderer.update_scene accepts."""
    if isinstance(camera, dict):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = np.asarray(camera.get("lookat", (0, 0, 0)), float)
        cam.distance = float(camera.get("distance", 1.5))
        cam.azimuth = float(camera.get("azimuth", 135.0))
        cam.elevation = float(camera.get("elevation", -25.0))
        return cam
    if isinstance(camera, str):
        if mujoco.mj_name2id(model, CAMERA, camera) < 0:
            names = [model.camera(i).name for i in range(model.ncam)]
            raise ValueError(f"camera '{camera}' is not in the scene; cameras: {names}")
    return camera


def render_rgb(model, data, width=640, height=480, camera=-1):
    with mujoco.Renderer(model, height=height, width=width) as r:
        r.update_scene(data, camera=camera_arg(model, camera))
        return r.render().copy()


def render_body_masks(model, data, width=640, height=480, camera=-1, hide_bodies=()):
    """{body name: bool HxW} from a segmentation render; bodies in hide_bodies (ids) are not drawn."""
    from tool_library.common.render import segmentation_renderer
    hide = {int(b) for b in hide_bodies}
    moved = [(g, int(model.geom_group[g])) for g in range(model.ngeom) if int(model.geom_bodyid[g]) in hide]
    try:
        for g, _ in moved:
            model.geom_group[g] = 5                      # a group the default scene option does not draw
        with segmentation_renderer(model, height, width) as r:
            r.update_scene(data, camera=camera_arg(model, camera))
            seg = r.render().copy()
    finally:
        for g, grp in moved:
            model.geom_group[g] = grp
    gid = seg[..., 0].astype(int)
    gid[seg[..., 1] != int(mujoco.mjtObj.mjOBJ_GEOM)] = -1     # ids of other object types (a skin, a flex) are not geoms
    out = {}
    for g in np.unique(gid):
        if g < 0 or g >= model.ngeom:
            continue
        name = model.body(int(model.geom_bodyid[g])).name
        m = gid == g
        out[name] = out.get(name, np.zeros_like(m)) | m
    return out


def iou(a, b):
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def load_mask(mask, size=None):
    """A bool array, a uint8 array or a PNG path -> bool array, optionally
    resized (nearest) to (width, height)."""
    if isinstance(mask, (str, Path)):
        from PIL import Image
        img = Image.open(mask).convert("L")
        if size is not None:
            img = img.resize(tuple(size), Image.NEAREST)
        return np.asarray(img) > 127
    m = np.asarray(mask)
    m = m > 127 if m.dtype != bool else m
    if size is not None and (m.shape[1], m.shape[0]) != tuple(size):
        from PIL import Image
        m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(tuple(size), Image.NEAREST)) > 127
    return m


def geom_volume(model, g) -> float:
    """Approximate volume of one geom in m^3 (box, sphere, capsule, cylinder, ellipsoid; AABB for meshes)."""
    t = int(model.geom_type[g]); s = [float(x) for x in model.geom_size[g]]
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        return 8.0 * s[0] * s[1] * s[2]
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return 4.0 / 3.0 * np.pi * s[0] ** 3
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return np.pi * s[0] ** 2 * (2 * s[1]) + 4.0 / 3.0 * np.pi * s[0] ** 3
    if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return np.pi * s[0] ** 2 * (2 * s[1])
    if t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        return 4.0 / 3.0 * np.pi * s[0] * s[1] * s[2]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        r = float(model.geom_rbound[g])
        return max(1e-9, 0.5 * 4.0 / 3.0 * np.pi * r ** 3)
    return 1e-9
