"""Synthetic episodes for the Episode Audits tests: the repository's bench (setup/scene.xml) with primitive objects
added, physics rollouts with the robot driven kinematically, and helpers to write trajectory.npz files.

Every test builds its inputs here; nothing depends on examples/ or runs/.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np

SETUP = ROOT / "setup" / "scene.xml"
FPS = 30.0


def tmpdir():
    base = os.environ.get("TMPDIR") or None
    return Path(tempfile.mkdtemp(prefix="audit_", dir=base))


def bench(out_dir, objects=(), keyframe=True, key_overrides=None, edit=None, name="scene.xml"):
    """setup/scene.xml plus objects, written to out_dir/name. objects: dicts {name, type ('box' | 'cylinder' |
    'sphere' | 'capsule'), size, pos, quat (wxyz), mass, rgba, free (True), visual_mesh (optional (V, F) drawn
    instead of the collision geom)}. The 'home' keyframe keeps the robot's home pose and puts every object at its
    pos/quat (key_overrides: {object name: (pos, quat)} for a keyframe that differs). edit(spec) runs last."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = mujoco.MjSpec.from_file(str(SETUP))
    spec.meshdir = str(SETUP.parent)
    home = mujoco.MjModel.from_xml_path(str(SETUP)).key_qpos[0].copy()
    types = {"box": mujoco.mjtGeom.mjGEOM_BOX, "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
             "sphere": mujoco.mjtGeom.mjGEOM_SPHERE, "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE}
    for o in objects:
        b = spec.worldbody.add_body()
        b.name = o["name"]
        b.pos = list(o.get("pos", [0, 0.5, 0.05]))
        b.quat = list(o.get("quat", [1, 0, 0, 0]))
        if o.get("free", True):
            j = b.add_freejoint(); j.name = o["name"] + "_free"
        g = b.add_geom()
        g.name = o["name"] + "_geom"
        g.type = types[o.get("type", "box")]
        g.size = list(o.get("size", [0.02, 0.02, 0.02])) + [0] * (3 - len(o.get("size", [0.02, 0.02, 0.02])))
        g.mass = float(o.get("mass", 0.1))
        g.rgba = list(o.get("rgba", [0.8, 0.3, 0.2, 1]))
        g.condim = 4
        if o.get("visual_mesh") is not None:
            V, F = o["visual_mesh"]
            mesh = spec.add_mesh(); mesh.name = o["name"] + "_vmesh"
            mesh.uservert = np.asarray(V, float).flatten().tolist()
            mesh.userface = np.asarray(F, int).flatten().tolist()
            vg = b.add_geom()
            vg.name = o["name"] + "_visual"; vg.type = mujoco.mjtGeom.mjGEOM_MESH; vg.meshname = mesh.name
            vg.contype = 0; vg.conaffinity = 0; vg.group = 1; vg.mass = 0
            g.group = 3; g.rgba = [0.8, 0.2, 0.2, 0]
    if edit is not None:
        edit(spec)
    m = spec.compile()
    if keyframe:
        q = m.qpos0.copy()
        q[:len(home)] = home
        for o in objects:
            if not o.get("free", True):
                continue
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, o["name"] + "_free")
            a = int(m.jnt_qposadr[jid])
            pos, quat = (key_overrides or {}).get(o["name"], (o.get("pos", [0, 0.5, 0.05]), o.get("quat", [1, 0, 0, 0])))
            q[a:a + 3] = pos; q[a + 3:a + 7] = quat
        spec.keys[0].qpos = q.tolist()
    else:
        for k in list(spec.keys):
            spec.delete(k)
        spec.nkey = 0                                    # setup/scene.xml reserves one keyframe slot (<size nkey="1"/>)
    m = spec.compile()
    path = out_dir / name
    path.write_text(spec.to_xml())
    return str(path), mujoco.MjModel.from_xml_path(str(path))


def home(m):
    return (m.key_qpos[0] if m.nkey else m.qpos0).copy()


def free_adr(m, name):
    return int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name + "_free")])


def jadr(m, name):
    return int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)])


def wrist_adr(m, side):
    return [jadr(m, f"robot_{side}_{side}_{k}") for k in ("pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z")]


def save(path, Q, labels, fps=FPS, **extra):
    Q = np.asarray(Q, float)
    if isinstance(labels, str):
        labels = [labels] * len(Q)
    arrs = {"qpos": Q, "labels": np.array([str(x) for x in labels]), "fps": float(fps)}
    arrs.update({k: np.asarray(v) for k, v in extra.items()})
    np.savez(str(path), **arrs)
    return str(path)


def site_pos(m, q, site):
    d = mujoco.MjData(m)
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    return d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site)].copy()


def body_pos(m, q, body):
    d = mujoco.MjData(m)
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    return d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)].copy()


def rollout(scene, q0, n_frames, drive=None, fps=FPS, exec_model=True):
    """Physics at the scene's timestep from q0, recorded at fps. drive(frame index, qpos) -> qpos for the robot dofs
    (applied every step, interpolated between frames: the robot is kinematic); objects move by contact and gravity.
    Uses the executor's contact model (hand-object pairs) for bench scenes."""
    from tool_library.common import episode_audit as EA
    ep = EA.Episode(scene, {"qpos": np.asarray([q0]), "fps": fps, "labels": [""]})
    m = ep.contact_model(0.0) if exec_model else ep.model
    d = mujoco.MjData(m)
    d.qpos[:] = q0
    radr, rdadr = EA.robot_dofs(ep, m)
    for e in range(m.neq):
        b1, b2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
        if b1 in ep.robot_bodies and b2 in ep.robot_bodies:
            d.eq_active[e] = 0
    mujoco.mj_forward(m, d)
    n_sub = max(1, int(round(1.0 / (fps * m.opt.timestep))))
    Q = np.zeros((n_frames, m.nq))
    Q[0] = d.qpos
    tgt_prev = d.qpos[radr].copy()
    for t in range(1, n_frames):
        tgt = drive(t, d.qpos.copy())[radr] if drive is not None else tgt_prev
        for k in range(n_sub):
            u = (k + 1) / n_sub
            d.qpos[radr] = (1 - u) * tgt_prev + u * tgt
            d.qvel[rdadr] = (tgt - tgt_prev) * fps
            mujoco.mj_step(m, d)
        d.qpos[radr] = tgt
        d.qvel[rdadr] = 0.0
        mujoco.mj_forward(m, d)
        tgt_prev = tgt
        Q[t] = d.qpos
    return Q


def smooth(a, b, n):
    """n samples from a to b with a smoothstep profile (rows)."""
    u = np.linspace(0, 1, n)
    s = u * u * (3 - 2 * u)
    return np.asarray(a, float)[None] * (1 - s[:, None]) + np.asarray(b, float)[None] * s[:, None]


class Cases:
    """[PASS] / [FAIL] bookkeeping for a test script."""

    def __init__(self):
        self.fails = 0

    def run(self, name, fn):
        try:
            fn()
            print(f"[PASS] {name}", flush=True)
        except Exception as e:
            import traceback
            self.fails += 1
            print(f"[FAIL] {name}: {e!r}", flush=True)
            traceback.print_exc(limit=2)

    def exit(self):
        sys.exit(1 if self.fails else 0)


def expect_value_error(fn, word):
    try:
        fn()
    except ValueError as e:
        assert word in str(e), f"ValueError without {word!r}: {e}"
        return
    raise AssertionError(f"no ValueError naming {word!r}")


def gap(scene, q, side, obj):
    """Smallest distance between a hand's collision geometry and an object at qpos q (negative: inside)."""
    from tool_library.common import episode_audit as EA
    ep = EA.Episode(scene, {"qpos": np.asarray([q]), "fps": FPS, "labels": [""]})
    scan = EA.ContactScan(ep, margin=0.1)
    return float(scan.touching(0, ep.hands[side]["bodies"], ep.objects[obj]["bodies"], tol=0.1))


def touch_from_above(scene, m, q, side, obj, target=-0.0002, site="index_tip", half_height=None):
    """q with the object moved under the hand's `site` (its x-y) and raised until the hand rests on its top at
    `target` distance."""
    q = np.asarray(q, float).copy()
    a = free_adr(m, obj)
    p = site_pos(m, q, f"robot_{side}_{side}_{site}")
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, obj + "_geom")
    hh = float(m.geom_size[gid][2]) if half_height is None else half_height
    q[a:a + 2] = p[:2]
    q[a + 2] = p[2] - hh - 0.04
    for _ in range(6):
        g = gap(scene, q, side, obj)
        if not np.isfinite(g):
            raise RuntimeError("the hand is more than 10 cm from the object")
        q[a + 2] += g - target
        if abs(g - target) < 2e-5:
            break
    return q


def carry_with_hand(m, Q, side, obj, rel_pos=(0.0, 0.12, -0.06)):
    """Q with the object riding rigidly with the hand's wrist body at a fixed offset (in the wrist frame)."""
    d = mujoco.MjData(m)
    a = free_adr(m, obj)
    root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"robot_{side}_{side}_base_yaw")
    Q = np.asarray(Q, float).copy()
    for t in range(len(Q)):
        d.qpos[:] = Q[t]
        mujoco.mj_kinematics(m, d)
        R = d.xmat[root].reshape(3, 3)
        Q[t, a:a + 3] = d.xpos[root] + R @ np.asarray(rel_pos)
        Q[t, a + 3:a + 7] = d.xquat[root]
    return Q
