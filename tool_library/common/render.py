"""MuJoCo rendering and camera maths shared by the scene tools.

Conventions (MuJoCo's, which Blender shares): a camera looks down its own -Z
with +Y up; `fovy` is the vertical field of view in degrees; quaternions are
(w, x, y, z).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

if sys.platform.startswith("linux"):  # headless offscreen rendering; MuJoCo rejects "egl" elsewhere
    os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np


def segmentation_renderer(model, height, width):
    """A mujoco.Renderer drawing segmentation ids, built WITHOUT multisampling.

    The segmentation image encodes each scene geom's id as a colour (red + 256 * green + ...). With the model's
    multisampled offscreen buffer (<quality offsamples="4"/>, MuJoCo's default) the edge pixels are averages of two
    ids' colours: below about 255 geoms that silently paints a third geom's id along every silhouette edge, and above
    it the average can exceed the scene's geom count and mujoco.Renderer.render() fails with an IndexError in
    segid2output (seen at 410 geoms). An offscreen buffer with offsamples 0 has no averaging; the model's own setting
    is restored as soon as the context exists, so RGB renders of the same model keep their anti-aliasing."""
    old = int(model.vis.quality.offsamples)
    model.vis.quality.offsamples = 0
    try:
        r = mujoco.Renderer(model, height=int(height), width=int(width))
    finally:
        model.vis.quality.offsamples = old
    r.enable_segmentation_rendering()
    return r


def load_model(scene, cameras=None):
    """An MjModel from an MJCF/MJB path or an existing model.

    `cameras` = {name: {'pos', 'quat' (wxyz) | 'target', 'fovy'}} adds fixed
    world cameras before compiling (exact pose incl. roll), which a free
    MjvCamera cannot express. Needs an MJCF path for that."""
    if cameras:
        assert not isinstance(scene, mujoco.MjModel), "injecting cameras needs an MJCF path"
        spec = mujoco.MjSpec.from_file(str(scene))
        for name, c in cameras.items():
            cam = spec.worldbody.add_camera()
            cam.name = name
            cam.pos = [float(v) for v in c["pos"]]
            q = c.get("quat", c.get("quat_wxyz"))
            cam.quat = [float(v) for v in (q if q is not None else lookat_quat(c["pos"], c["target"], c.get("up", (0, 0, 1))))]
            cam.fovy = float(c.get("fovy", 45.0))
        return spec.compile()
    if isinstance(scene, mujoco.MjModel):
        return scene
    p = Path(scene)
    if p.suffix.lower() == ".mjb":
        return mujoco.MjModel.from_binary_path(str(p))
    return mujoco.MjModel.from_xml_path(str(p))


def camera_dict(model, data, name):
    """World pose + fovy of a named camera as {'pos','quat','fovy'}."""
    cid = mujoco.mj_name2id(model, int(mujoco.mjtObj.mjOBJ_CAMERA), name)
    assert cid >= 0, f"no camera {name}"
    return {"pos": np.array(data.cam_xpos[cid]), "quat": mat_to_quat(np.array(data.cam_xmat[cid]).reshape(3, 3)),
            "fovy": float(model.cam_fovy[cid])}


def quat_to_mat(q):
    M = np.zeros(9)
    mujoco.mju_quat2Mat(M, np.asarray(q, float))
    return M.reshape(3, 3)


def mat_to_quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).flatten())
    return q


def lookat_quat(eye, target, up_hint=(0.0, 0.0, 1.0)):
    """Camera quaternion at `eye` looking at `target` with `up_hint` upward."""
    z = np.asarray(eye, float) - np.asarray(target, float)
    z /= np.linalg.norm(z)
    up = np.asarray(up_hint, float)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-8:
        x = np.cross(np.array([1.0, 0.0, 0.0]), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return mat_to_quat(np.column_stack((x, y, z)))


def focal_px(fovy_deg, height):
    return (height / 2.0) / np.tan(np.radians(fovy_deg) / 2.0)


def fovy_from_focal(focal, height):
    return float(np.degrees(2.0 * np.arctan((height / 2.0) / float(focal))))


def project(points_world, cam_pos, cam_quat, fovy_deg, width, height):
    """Pixel (u, v) and depth of world points through a MuJoCo camera."""
    R = quat_to_mat(cam_quat)
    q = (np.asarray(points_world, float) - np.asarray(cam_pos, float)) @ R
    z = -q[..., 2]
    f = focal_px(fovy_deg, height)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = width / 2.0 + f * q[..., 0] / z
        v = height / 2.0 - f * q[..., 1] / z
    return np.stack([u, v], -1), z


def camera_from_table_lens(fovy_deg, height_m, pitch_deg, yaw_deg=90.0, xy=(0.0, 0.0),
                           table_top_z=0.0):
    """A camera `height_m` above the table at `xy`, optical axis pitched
    `pitch_deg` BELOW the horizon (positive = looking down) and turned
    `yaw_deg` about +Z (90 = looking along +Y). Returns (pos, quat_wxyz)."""
    yaw, pit = np.radians(yaw_deg), np.radians(pitch_deg)
    fwd = np.array([np.cos(yaw) * np.cos(pit), np.sin(yaw) * np.cos(pit), -np.sin(pit)])
    pos = np.array([xy[0], xy[1], table_top_z + height_m], float)
    return pos, lookat_quat(pos, pos + fwd)


class Offscreen:
    """One renderer per (model, size); call .rgb / .segmentation at a state."""

    def __init__(self, model, width=640, height=480):
        self.model, self.width, self.height = model, int(width), int(height)
        self.r = mujoco.Renderer(model, self.height, self.width)
        self._seg = None                       # a second, non-multisampled context (segmentation_renderer)

    def _cam(self, camera):
        if camera is None or isinstance(camera, (str, int)):
            return camera if camera is not None else -1
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        if "lookat" in camera:
            cam.lookat[:] = camera["lookat"]
            cam.distance = float(camera.get("distance", 1.5))
            cam.azimuth = float(camera.get("azimuth", 90.0))
            cam.elevation = float(camera.get("elevation", -30.0))
        else:                                  # explicit pose: pos + quat
            R = quat_to_mat(camera.get("quat", camera.get("quat_wxyz")))
            fwd = -R[:, 2]
            d = float(camera.get("distance", 1.0))
            cam.lookat[:] = np.asarray(camera["pos"], float) + fwd * d
            cam.distance = d
            cam.azimuth = float(np.degrees(np.arctan2(fwd[1], fwd[0])))
            cam.elevation = float(np.degrees(np.arcsin(np.clip(fwd[2], -1, 1))))
        if "fovy" in camera:
            self.model.vis.global_.fovy = float(camera["fovy"])
        return cam

    def rgb(self, data, camera=None):
        self.r.update_scene(data, camera=self._cam(camera))
        return self.r.render().copy()

    def segmentation(self, data, camera=None, by="body"):
        """Per-pixel id map, -1 where nothing is drawn: body ids (default) or geom ids (by='geom')."""
        if self._seg is None:
            self._seg = segmentation_renderer(self.model, self.height, self.width)
        self._seg.update_scene(data, camera=self._cam(camera))
        seg = self._seg.render().copy()
        gid = seg[..., 0].astype(np.int32)
        is_geom = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
        gid[~is_geom] = -1
        if by == "geom":
            return gid
        body = np.full_like(gid, -1)
        m = gid >= 0
        body[m] = self.model.geom_bodyid[gid[m]]
        return body

    def depth(self, data, camera=None):
        """Metric depth (H, W) float32 in metres."""
        self.r.enable_depth_rendering()
        try:
            self.r.update_scene(data, camera=self._cam(camera))
            d = self.r.render().copy()
        finally:
            self.r.disable_depth_rendering()
        return d.astype(np.float32)

    def close(self):
        self.r.close()
        if self._seg is not None:
            self._seg.close()


def body_subtree(model, bid):
    out, stack = set(), [int(bid)]
    while stack:
        b = stack.pop()
        out.add(b)
        stack += [c for c in range(model.nbody) if model.body_parentid[c] == b and c != b]
    return out


def geoms_of_bodies(model, bodies):
    bodies = set(int(b) for b in bodies)
    return [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies]


def iou(a, b) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0
