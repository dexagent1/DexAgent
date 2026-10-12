"""MJCF building blocks shared by the construction skills.

Every constructor is a pure function of numbers: dimensions, colours, masses
and placements come in as one flat parameter dictionary, and the dictionary is
written next to the asset it produced. Fitting an asset to a clip is editing
numbers, never editing code. Units are SI; angles are degrees only where a key
says `_deg`; colours are rgba in 0..1.

The constructors emit MJCF rather than URDF because the mechanisms that matter
here (a thread's rotation-to-lift coupling, a compliant grasp weld) are
equality constraints, which URDF cannot express.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

# EGL before any renderer is constructed: these builders render offscreen on a
# headless node. Linux only, since MuJoCo rejects MUJOCO_GL=egl on other systems.
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np


# ------------------------------- parameters ----------------------------------
class UnknownParameter(KeyError, ValueError):
    """A misspelled or unknown parameter: a ValueError as the library's conventions ask, and a KeyError for older callers."""

    def __str__(self):
        return str(self.args[0]) if self.args else ""


def merge_params(defaults: dict, overrides: dict | None) -> dict:
    """defaults <- overrides, with an UNKNOWN-KEY ERROR.

    Silently ignoring a misspelled key is how a build ends up not reflecting
    the number someone just changed, and that failure is invisible: the asset
    still compiles, it is just the old shape. So a typo is an exception.
    """
    out = dict(defaults)
    for k, v in (overrides or {}).items():
        if k not in defaults:
            close = [d for d in defaults if d.startswith(k[:4])]
            raise UnknownParameter(f"unknown parameter {k!r}"
                           + (f" — did you mean {close}?" if close else "")
                           + f"\nvalid keys: {sorted(defaults)}")
        out[k] = v
    return out


def save_params(path: str | Path, params: dict) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=2, sort_keys=True))
    return str(path)


def load_params(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def rgba(v: Sequence[float]) -> str:
    v = list(v)
    if len(v) == 3:
        v = v + [1.0]
    return " ".join(f"{float(x):.4f}" for x in v)


def vec(v: Sequence[float]) -> str:
    return " ".join(f"{float(x):.6f}" for x in v)


# ------------------------------- geometry ------------------------------------
def revolve_profile(profile_xz: Sequence[Sequence[float]], sections: int = 64):
    """(radius, height) polyline -> a solid of revolution about +Z (trimesh).

    Normals are made consistently OUTWARD when the revolve closes into a
    solid. trimesh.revolve orients each band by the profile segment's own
    direction, so a non-convex profile (a hollow cap) gets faces wound both
    ways: measured, the cap's top disc pointed inward and CULLED from above —
    the closed cap rendered as an open ring you could see into, which reads
    as a missing lid. Invisible at a 2 mm ceiling, glaring at 25 mm.
    """
    import trimesh
    pts = np.asarray(profile_xz, float)
    mesh = trimesh.creation.revolve(pts, sections=int(sections))
    mesh.remove_unreferenced_vertices()
    if mesh.is_watertight:
        mesh.fix_normals()
    return mesh


def helix_tube(coil_radius: float, pitch: float, turns: float,
               tube_radius: float, z0: float = 0.0, right_handed: bool = True,
               steps_per_turn: int = 48, ring: int = 10):
    """A swept tube along a helix — the VISIBLE thread ridge.

    Visual only: MuJoCo collides meshes by their convex hull, so a helical
    ridge cannot be a collision geom. The thread's PHYSICS lives in the
    rotation-to-lift equality constraint; this mesh is what makes the asset
    read as threaded to a person and to a VLM looking at the render.
    """
    import trimesh
    n = max(8, int(steps_per_turn * max(turns, 0.05)))
    t = np.linspace(0.0, 2 * np.pi * turns, n)
    s = 1.0 if right_handed else -1.0
    path = np.stack([coil_radius * np.cos(s * t),
                     coil_radius * np.sin(s * t),
                     z0 + pitch * t / (2 * np.pi)], axis=1)
    # Frenet-ish frame: tangent + a stable normal (world +Z projected out)
    tan = np.gradient(path, axis=0)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-12
    up = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    nrm = np.cross(up, tan)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    bin_ = np.cross(tan, nrm)
    ang = np.linspace(0, 2 * np.pi, ring, endpoint=False)
    verts = (path[:, None, :]
             + tube_radius * (np.cos(ang)[None, :, None] * nrm[:, None, :]
                              + np.sin(ang)[None, :, None] * bin_[:, None, :]))
    verts = verts.reshape(-1, 3)
    faces = []
    for i in range(n - 1):
        for j in range(ring):
            a = i * ring + j
            b = i * ring + (j + 1) % ring
            c = (i + 1) * ring + (j + 1) % ring
            d = (i + 1) * ring + j
            faces += [[a, b, c], [a, c, d]]
    mesh = trimesh.Trimesh(vertices=verts, faces=np.asarray(faces), process=True)
    return mesh


def write_mesh(mesh, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    return str(path)


def shell_ring(name: str, r_in: float, t: float, z0: float, z1: float,
               rgba_s: str, gattr: str, n: int = 10,
               gap_deg: tuple[float, float] | None = None,
               center=(0.0, 0.0)) -> list[str]:
    """A HOLLOW cylinder wall as n box segments — the only way to give MuJoCo
    an opening something can enter. A mesh's convex hull is a plug and a
    cylinder geom is solid, so every bore in these builders (a tube's mouth,
    a decapper's skirt, a pipette tip's collar) is one of these rings.
    `gap_deg=(a, b)` leaves the segments whose centre angle falls in [a, b]
    out — a window in the wall, e.g. the slot a cap's tab enters through."""
    out = []
    rc = r_in + t / 2.0
    half_chord = rc * np.tan(np.pi / n) * 1.06   # slight overlap, no slits
    for i in range(int(n)):
        ang = 360.0 * (i + 0.5) / n
        if gap_deg is not None:
            a = (ang - gap_deg[0]) % 360.0
            if a <= (gap_deg[1] - gap_deg[0]) % 360.0:
                continue
        th = np.radians(ang)
        out.append(
            f'      <geom name="{name}_{i}" type="box" '
            f'size="{vec((t / 2, half_chord, (z1 - z0) / 2))}" '
            f'pos="{vec((center[0] + rc * np.cos(th), center[1] + rc * np.sin(th), (z0 + z1) / 2))}" '
            f'euler="0 0 {ang:.3f}" rgba="{rgba_s}" {gattr}/>')
    return out


# ------------------------------ external forces -------------------------------
# EVERYTHING THAT MOVES IN THESE DEMOS MOVES BECAUSE OF A FORCE.
#
# The alternative — a position servo on the mechanism's own joint, or a mocap
# body welded to the object and teleported along a path — LOOKS the same in a
# render and proves nothing: a prescribed pose is imposed on the solver, so
# the object tracks it whether or not the physics permits it, and a demo of a
# thread that "cannot be pushed on" is worthless if the driver could have
# pushed it on regardless. Applying a wrench to a body and letting the solver
# decide what happens is the only version where the mechanism is on trial.

def apply_wrench(data, body_id: int, force=None, torque=None,
                 point=None, com=None) -> None:
    """Accumulate an external wrench on a body (world frame).

    `point` applies the force at a world point rather than the centre of mass,
    which is what pushing on the top edge of a panel means — the moment arm is
    the whole reason the panel folds instead of sliding.
    """
    if force is not None:
        f = np.asarray(force, float)
        data.xfrc_applied[body_id, :3] += f
        if point is not None and com is not None:
            data.xfrc_applied[body_id, 3:] += np.cross(
                np.asarray(point, float) - np.asarray(com, float), f)
    if torque is not None:
        data.xfrc_applied[body_id, 3:] += np.asarray(torque, float)


def _quat_err(q_cur: np.ndarray, q_tgt: np.ndarray) -> np.ndarray:
    """Rotation vector taking q_cur to q_tgt (world frame)."""
    import mujoco
    qc = np.asarray(q_cur, float).copy()
    neg = np.zeros(4)
    mujoco.mju_negQuat(neg, qc)
    dq = np.zeros(4)
    mujoco.mju_mulQuat(dq, np.asarray(q_tgt, float), neg)
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, dq, 1.0)
    return v


class ForceGrip:
    """A hand modelled as a capped spring-damper wrench on a body.

    It never writes qpos, qvel or mocap targets — only `xfrc_applied`. If the
    object is bolted down, constrained, or simply too heavy, the grip pulls
    and nothing happens, exactly as a real hand would.
    """

    def __init__(self, model, body: str | int, kp=40.0, kd=3.0,
                 max_force=25.0, kr=0.6, kdr=0.02, max_torque=0.8,
                 spin_free_axis=None):
        """spin_free_axis: a BODY-LOCAL axis (e.g. (0,0,1)) about which the
        grip applies NO torque — a pinched cylinder spins freely between
        fingers. This is not just realism: a cylinder's spin inertia is
        ~m r^2/2 (an AA cell: 6e-7 kg m^2), so any kr/kdr sized for the
        tumble axes is 100x past the explicit-Euler stability bound on the
        spin axis, and holding it detonates |w| within 20 steps (measured).
        """
        self.bid = model.body(body).id if isinstance(body, str) else int(body)
        self.root = int(model.body_rootid[self.bid])
        self.kp, self.kd, self.max_force = kp, kd, max_force
        self.kr, self.kdr, self.max_torque = kr, kdr, max_torque
        self.spin_free_axis = (None if spin_free_axis is None
                               else np.asarray(spin_free_axis, float))
        self.last = np.zeros(6)

    @classmethod
    def sized(cls, model, body: str | int, timestep: float | None = None,
              stiff: float = 1.0, max_force: float | None = None,
              max_torque: float | None = None, spin_free_axis=None):
        """Gains derived from the body's own mass/inertia and the timestep.

        xfrc_applied is integrated EXPLICITLY (implicitfast only covers
        joint damping), so a grip tuned for a 1 kg box detonates a 3-gram
        tube: the stability bounds are kd/m < ~2/dt and sqrt(kp/m) < ~1/dt,
        and hand-picked constants sit 100x past them for small bodies
        (measured: NaN within 30 steps). This sizes every gain to a safe
        fraction of its own bound. The price is honest: a force-controlled
        grip on a light object holds position firmly but orientation only
        weakly — as a real pinch does.
        """
        bid = model.body(body).id if isinstance(body, str) else int(body)
        dt = float(timestep if timestep is not None else model.opt.timestep)
        m = max(1e-4, float(model.body_subtreemass[bid]))
        # root-body inertia only (no subtree accumulation): an UNDERestimate,
        # which errs toward smaller — stabler — rotational gains. The MIN
        # principal axis bounds kr; declaring it spin-free (physically apt
        # for anything pinched between fingers) lets kr be sized to the
        # MIDDLE axis instead — on a slender cylinder that is 10x the
        # wrist authority for the same stability margin.
        vals = np.sort(np.asarray(model.body_inertia[bid], float))
        inertia = max(1e-9, float(vals[0] if spin_free_axis is None
                                  else vals[1]))
        # NEAR-CRITICAL damping (zeta ~0.9), not the bare stability
        # minimum: at zeta 0.28 a 29 cm pipette swung upright waggled
        # through several visible cycles and every contact rang — the
        # single biggest source of on-camera shaking (measured). The
        # explicit-integration budget has room: kd*dt/m stays ~0.45.
        kp = 0.5 * m * (0.25 / dt) ** 2 * stiff
        kd = 0.32 * m / dt * float(np.sqrt(stiff))
        kr = 0.5 * inertia * (0.25 / dt) ** 2 * stiff
        kdr = 0.32 * inertia / dt * float(np.sqrt(stiff))
        return cls(model, bid, kp=kp, kd=kd,
                   max_force=(max_force if max_force is not None
                              else min(60.0, 400.0 * m)),
                   kr=kr, kdr=kdr,
                   max_torque=(max_torque if max_torque is not None
                               else min(2.0, 40.0 * np.sqrt(inertia * m))),
                   spin_free_axis=spin_free_axis)

    @staticmethod
    def _clip(v, lim):
        n = float(np.linalg.norm(v))
        return v if n <= lim or n < 1e-12 else v * (lim / n)

    def com_velocity(self, data) -> np.ndarray:
        """Linear velocity of the body's centre of mass, where xfrc_applied acts. cvel's linear part is the velocity
        of the point at its kinematic tree's subtree COM, not of the body: for a link deep in a chain (a cable's end
        segment, whose tree is the whole cable) that point is far away, so damping it fed the link's spin back as
        force (a 1 m x 1.25 mm cord dragged by its end went unstable in 3 s)."""
        w = data.cvel[self.bid, :3]
        return data.cvel[self.bid, 3:] + np.cross(w, data.xipos[self.bid] - data.subtree_com[self.root])

    def pos(self, data) -> np.ndarray:
        return data.xpos[self.bid].copy()

    def quat(self, data) -> np.ndarray:
        return data.xquat[self.bid].copy()

    def hold(self, data, target_pos=None, target_quat=None,
             axes=(1, 1, 1)) -> np.ndarray:
        """One control tick. Returns the wrench actually applied (N, N*m)."""
        w = np.zeros(6)
        if target_pos is not None:
            e = np.asarray(target_pos, float) - data.xpos[self.bid]
            e = e * np.asarray(axes, float)       # free axes: no force at all
            v = self.com_velocity(data) * np.asarray(axes, float)
            w[:3] = self._clip(self.kp * e - self.kd * v, self.max_force)
        if target_quat is not None:
            r = _quat_err(data.xquat[self.bid], np.asarray(target_quat, float))
            wv = data.cvel[self.bid, :3].copy()
            if self.spin_free_axis is not None:
                R = data.xmat[self.bid].reshape(3, 3)
                a = R @ self.spin_free_axis
                a = a / (np.linalg.norm(a) + 1e-12)
                r = r - (r @ a) * a
                wv = wv - (wv @ a) * a
            w[3:] = self._clip(self.kr * r - self.kdr * wv, self.max_torque)
        data.xfrc_applied[self.bid] += w
        self.last = w
        return w

    def twist(self, data, axis, torque: float) -> float:
        """Pure torque about a world axis — a twisting hand, no position hold.
        Nothing constrains where the object goes; the mechanism decides."""
        a = np.asarray(axis, float)
        a = a / (np.linalg.norm(a) + 1e-12)
        data.xfrc_applied[self.bid, 3:] += a * float(torque)
        self.last[3:] = a * float(torque)
        return float(torque)

    def pull(self, data, direction, force: float) -> float:
        """Pure force in a world direction — a straight tug, no twist."""
        d = np.asarray(direction, float)
        d = d / (np.linalg.norm(d) + 1e-12)
        data.xfrc_applied[self.bid, :3] += d * float(force)
        self.last[:3] = d * float(force)
        return float(force)


def clear_forces(data) -> None:
    """Call once per control tick BEFORE any grip acts: xfrc_applied
    persists across steps, so forgetting this silently integrates a force
    forever."""
    data.xfrc_applied[:] = 0.0


# ------------------------------- rendering -----------------------------------
def free_camera(lookat, distance: float, azimuth: float, elevation: float):
    import mujoco
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.asarray(lookat, float)
    cam.distance = float(distance)
    cam.azimuth = float(azimuth)
    cam.elevation = float(elevation)
    return cam


def _label(img: np.ndarray, text: str, max_lines: int = 2) -> np.ndarray:
    if not text:
        return img
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    per = max(30, im.width // 7)
    lines = [text[i:i + per] for i in range(0, len(text), per)][:max_lines]
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, im.width, 6 + 15 * len(lines)], fill=(15, 15, 18))
    for i, ln in enumerate(lines):
        dr.text((8, 3 + 15 * i), ln, fill=(245, 245, 245))
    return np.asarray(im)


def rollout(model, data, seconds: float, fps: int,
            control: Callable[[float, Any, Any], str | None] | None = None,
            camera=None, width: int = 640, height: int = 480,
            spin_deg_per_s: float = 0.0) -> list[np.ndarray]:
    """Step the sim and render one frame every 1/fps of SIM time.

    `control(t, model, data) -> label` is called once per control tick (every
    sim step) and may set ctrl / mocap / eq_active. Its return value, if any,
    is burnt into the frame as a caption — the phase labels are how a viewer
    tells "the thread is holding" from "the thread let go".
    """
    import mujoco
    dt = float(model.opt.timestep)
    steps_per_frame = max(1, int(round((1.0 / fps) / dt)))
    n_frames = int(round(seconds * fps))
    frames: list[np.ndarray] = []
    cam = camera if camera is not None else free_camera([0, 0, 0.1], 0.8, 90, -20)
    with mujoco.Renderer(model, height=height, width=width) as r:
        for i in range(n_frames):
            text = None
            for _ in range(steps_per_frame):
                if control is not None:
                    text = control(data.time, model, data) or text
                mujoco.mj_step(model, data)
            # A demo that blew up is worse than no demo: the video still looks
            # plausible for the first few seconds and the numbers printed
            # beside it are garbage. Fail loudly at the frame it happened.
            #
            # Checking qpos for NaN is NOT enough — on bad acceleration MuJoCo
            # resets the state itself, so by the time you look, qpos is finite
            # again and the only trace is the warning counter and a rope that
            # silently teleported back to its spawn pose.
            if int(data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number):
                raise RuntimeError(
                    f"simulation went unstable at t={data.time:.3f}s "
                    f"(frame {i}, mjWARN_BADQACC) — MuJoCo reset the state. "
                    f"Usual causes: a mocap/weld target that jumped between "
                    f"phases, or a timestep too large for the stiffness.")
            if not np.isfinite(data.qpos).all():
                raise RuntimeError(
                    f"simulation went non-finite at t={data.time:.3f}s "
                    f"(frame {i}) — usually a mocap/weld target that jumped "
                    f"discontinuously between phases")
            if spin_deg_per_s:
                cam.azimuth += spin_deg_per_s / fps
            r.update_scene(data, camera=cam)
            frames.append(_label(r.render().copy(), text or ""))
    return frames


def write_video(frames: Iterable[np.ndarray], out_mp4: str | Path,
                fps: int = 30, gif: bool = True, gif_stride: int = 3,
                gif_width: int = 380, gif_colors: int = 96) -> dict:
    import imageio.v2 as imageio
    from PIL import Image
    frames = list(frames)
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out_mp4, fps=fps, macro_block_size=1,
                            codec="libx264", quality=7) as w:
        for f in frames:
            w.append_data(f)
    out = {"mp4": str(out_mp4), "frames": len(frames)}
    if gif:
        g = out_mp4.with_suffix(".gif")
        # A full-rate, full-width, true-colour GIF of a 20 s demo is 12 MB
        # and unshareable. Stride + downscale + an adaptive palette keeps it
        # legible at roughly a tenth of that.
        small = []
        for f in frames[::max(1, gif_stride)]:
            im = Image.fromarray(f)
            h = int(im.height * gif_width / im.width)
            im = im.resize((gif_width, h), Image.LANCZOS).convert(
                "P", palette=Image.ADAPTIVE, colors=int(gif_colors))
            small.append(im.convert("RGB"))
        small[0].save(g, save_all=True, append_images=small[1:], loop=0,
                      duration=int(1000 * gif_stride / float(fps)),
                      optimize=True)
        out["gif"] = str(g)
    return out


def key_states(model, data, states, out_dir: str | Path, prefix: str,
               camera=None, width: int = 420, height: int = 480,
               settle_steps: int = 0) -> list[str]:
    """Render one still per named configuration — a build's cheapest proof.

    `states` is [(label, setter(model, data)), ...]. Each setter puts the
    model in a pose; the frame is rendered after mj_forward (plus optional
    settling). No stepping, no video: this is what a tool attaches so the
    agent can SEE the mechanism at its extremes for the cost of 3 renders.
    """
    import mujoco
    from PIL import Image
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cam = camera if camera is not None else free_camera([0, 0, 0.1], 0.8, 90, -20)
    paths = []
    with mujoco.Renderer(model, height=height, width=width) as r:
        for i, (label, setter) in enumerate(states):
            mujoco.mj_resetData(model, data)
            setter(model, data)
            mujoco.mj_forward(model, data)
            for _ in range(settle_steps):
                setter(model, data)
                mujoco.mj_step(model, data)
            r.update_scene(data, camera=cam)
            p = out_dir / f"{prefix}_{i}_{label}.png"
            Image.fromarray(_label(r.render().copy(), label)).save(p)
            paths.append(str(p))
    return paths


def save_stills(frames: Sequence[np.ndarray], out_dir: str | Path,
                prefix: str, n: int = 4) -> list[str]:
    """n evenly spaced stills — the images a tool attaches for the agent to
    LOOK at, and the cheapest way for a person to check a build."""
    from PIL import Image
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = np.linspace(0, len(frames) - 1, max(1, n)).round().astype(int)
    paths = []
    for k, i in enumerate(idx):
        p = out_dir / f"{prefix}_{k}.png"
        Image.fromarray(frames[int(i)]).save(p)
        paths.append(str(p))
    return paths


# ------------------------------- MJCF bits -----------------------------------
FLOOR_XML = """
    <geom name="floor" type="plane" size="2 2 .05" pos="0 0 0"
          rgba="0.62 0.60 0.57 1" friction="1 0.02 0.001"/>"""

LIGHTS_XML = """
    <light name="key" pos="0.4 -0.5 1.2" dir="-0.3 0.4 -1" diffuse=".7 .7 .7"/>
    <light name="fill" pos="-0.6 0.4 0.9" dir="0.5 -0.3 -1" diffuse=".3 .3 .3"/>"""


def mjcf_header(model_name: str, timestep: float = 0.002,
                meshdir: str | None = None) -> str:
    md = f' meshdir="{meshdir}"' if meshdir else ""
    return (f'<mujoco model="{model_name}">\n'
            f'  <compiler angle="degree" autolimits="true"{md}/>\n'
            f'  <option timestep="{timestep}" integrator="implicitfast" '
            f'cone="elliptic"/>\n'
            f'  <visual><global offwidth="1280" offheight="960"/>'
            f'<quality shadowsize="4096"/></visual>\n')


def load(xml_path: str | Path):
    """MjModel from a built asset file (kept here so callers never have to
    remember the EGL/env dance)."""
    import mujoco
    return mujoco.MjModel.from_xml_path(str(xml_path))
