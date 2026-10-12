"""A cable on the table alone, simulated: what a lay really does to it (used by plan_knot_moves to check the moves its
fast follow-the-leader search proposes).

The bench scene's cable (its own asset, joints, stiffness, friction and self-collision) is put on a slab with the
table's contact parameters, under the bench's execution physics (episode.apply_execution_options), and the picked link
is welded to a mocap body that does what the executor's `lay` does with the hand: lift it, carry it along the path at
that height keeping its orientation, set it down, let go (the weld switched off) and let the rope settle. No hand, no
arm: what the rope does when one of its links is moved that way, in the same contact physics as the episode, in a
fraction of a second per move. Follow-the-leader alone is wrong for a sideways carry: it drags the far rope along the
hand's path (measured on the bench: a 40 cm carry of a rope end across the other leg, follow-the-leader 0.20 m mean
off the executed rope with the far leg pulled 22 cm, the simulated cable within centimetres)."""
from __future__ import annotations

import mujoco
import numpy as np

from tool_library.common import episode as E


class RopeSim:
    """The cable of a bench scene on its own. `scene` is the bench MJCF, `asset_xml` the cable's own MJCF (scene_recon's
    asset_xml, construct_rope's xml), `prefix` the name prefix the scene gave the cable's bodies and joints."""

    def __init__(self, scene, asset_xml, prefix, fast=True):
        bench = mujoco.MjModel.from_xml_path(str(scene))
        spec = mujoco.MjSpec.from_file(str(asset_xml))
        E.apply_execution_options(spec)
        spec.option.timestep = float(bench.opt.timestep)
        spec.option.integrator = bench.opt.integrator
        spec.option.impratio = max(float(spec.option.impratio), float(bench.opt.impratio))
        if fast:
            # a prediction, four times faster: the rope alone needs no hand no-slip and takes a 4 ms step (measured on a
            # 40 cm carry of a 3.2 cm, 0.86 kg cable: the same 10 mm mean from the executed rope as the full options)
            spec.option.noslip_iterations = 0
            spec.option.timestep = 2.0 * float(bench.opt.timestep)
        for g in list(spec.geoms):
            if g.type == mujoco.mjtGeom.mjGEOM_PLANE:
                spec.delete(g)
        tg = mujoco.mj_name2id(bench, mujoco.mjtObj.mjOBJ_GEOM, "g_table")
        tb = spec.worldbody.add_geom()
        tb.name = "ropesim_table"
        if tg >= 0:
            d0 = mujoco.MjData(bench)
            mujoco.mj_forward(bench, d0)
            tb.type = mujoco.mjtGeom(int(bench.geom_type[tg]))
            tb.size = bench.geom_size[tg].copy()
            tb.pos = d0.geom_xpos[tg].copy()
            tb.quat = np.array([1.0, 0.0, 0.0, 0.0])
            tb.friction = bench.geom_friction[tg].copy()
            tb.solref = bench.geom_solref[tg].copy()
            tb.solimp = bench.geom_solimp[tg].copy()
            tb.condim = int(bench.geom_condim[tg])
        else:
            tb.type = mujoco.mjtGeom.mjGEOM_PLANE
            tb.size = [2.0, 2.0, 0.1]
        mb = spec.worldbody.add_body()
        mb.name = "ropesim_hand"
        mb.mocap = True
        sg = mb.add_geom()
        sg.type = mujoco.mjtGeom.mjGEOM_SPHERE
        sg.size = [0.005, 0, 0]
        sg.contype = 0
        sg.conaffinity = 0
        first = next(b.name for b in spec.bodies if b.name.endswith("B_first"))
        eq = spec.add_equality()
        eq.name = "ropesim_grip"
        eq.type = mujoco.mjtEq.mjEQ_WELD
        eq.objtype = mujoco.mjtObj.mjOBJ_BODY
        eq.name1 = "ropesim_hand"
        eq.name2 = first
        eq.active = False
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.eq = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "ropesim_grip")
        self.mocap = int(m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ropesim_hand")])
        # the scene's qpos addresses of the cable's joints, by name (the scene adds its prefix)
        self.map = []
        for j in range(m.njnt):
            nm = m.joint(j).name
            sj = mujoco.mj_name2id(bench, mujoco.mjtObj.mjOBJ_JOINT, prefix + nm) if nm else -1
            if sj < 0:
                continue
            n = {0: 7, 1: 4, 2: 1, 3: 1}[int(m.jnt_type[j])]
            self.map.append((int(m.jnt_qposadr[j]), int(bench.jnt_qposadr[sj]), n))
        from tool_library.verifiers.rope_knot_check import _cable_bodies
        self.pre, self.bodies = _cable_bodies(m, None)

    def set_from_scene_qpos(self, q_scene):
        d = self.data
        mujoco.mj_resetData(self.model, d)
        for a, b, n in self.map:
            d.qpos[a:a + n] = np.asarray(q_scene, float)[b:b + n]
        d.qvel[:] = 0.0
        d.eq_active[self.eq] = 0
        mujoco.mj_forward(self.model, d)

    def centreline(self):
        from tool_library.verifiers.rope_knot_check import _centreline
        mujoco.mj_kinematics(self.model, self.data)
        return _centreline(self.model, self.data, self.bodies, self.pre)

    def lay(self, k, path, lift_m, z_land=None, speed=0.4, lift_s=0.4, set_s=0.4, settle_s=0.3, pins=()):
        """Lay link k (index along the cable, B_first = 0) along `path` (xy points of the link's body origin, the last is
        where it lands) carried `lift_m` above where it lies; `pins` (link indices) are held where they are. Returns the
        centreline after settling (n, 3), or None when the simulation diverged."""
        m, d = self.model, self.data
        b = int(self.bodies[k])
        p0 = d.xpos[b].copy()
        q0 = d.xquat[b].copy()
        z_land = float(p0[2]) if z_land is None else float(z_land)
        d.mocap_pos[self.mocap] = p0
        d.mocap_quat[self.mocap] = q0
        m.eq_obj2id[self.eq] = b
        m.eq_data[self.eq][:] = 0.0
        m.eq_data[self.eq][6] = 1.0                       # relpose quaternion = identity: the link stays on the hand
        m.eq_data[self.eq][10] = 1.0                      # torquescale
        d.eq_active[self.eq] = 1
        pinned = [(int(self.bodies[p]), d.xpos[int(self.bodies[p])].copy()) for p in pins]
        dt = float(m.opt.timestep)
        way = [p0, p0 + [0.0, 0.0, float(lift_m)]]
        for xy in path:
            way.append(np.array([xy[0], xy[1], p0[2] + float(lift_m)]))
        way.append(np.array([path[-1][0], path[-1][1], z_land]))
        durs = [lift_s] + [max(0.15, float(np.linalg.norm(b_ - a_)) / float(speed)) for a_, b_ in zip(way[1:-2], way[2:-1])] + [set_s]
        for (a_, b_), T in zip(zip(way[:-1], way[1:]), durs):
            n = max(1, int(round(T / dt)))
            for i in range(n):
                u = (i + 1) / n
                s = u * u * (3 - 2 * u)
                d.mocap_pos[self.mocap] = a_ + s * (b_ - a_)
                for pb, pp in pinned:
                    d.xfrc_applied[pb, :3] = 400.0 * (pp - d.xpos[pb]) - 20.0 * d.cvel[pb, 3:]
                mujoco.mj_step(m, d)
            if not np.isfinite(d.qpos).all():
                return None
        d.eq_active[self.eq] = 0
        d.xfrc_applied[:] = 0.0
        for _ in range(int(round(settle_s / dt))):
            mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all():
            return None
        return self.centreline()

    def state(self):
        return self.data.qpos.copy(), self.data.qvel.copy()

    def restore(self, st):
        self.data.qpos[:] = st[0]
        self.data.qvel[:] = st[1]
        self.data.eq_active[self.eq] = 0
        mujoco.mj_forward(self.model, self.data)
