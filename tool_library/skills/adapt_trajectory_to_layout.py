"""Move the objects, keep the task: adapt a verified trajectory to a new initial layout.

    from tool_library.skills.adapt_trajectory_to_layout import Episode, adapt_trajectory_to_layout
    ep = Episode("scene.xml", "trajectory.npz",
                 end_effectors={"right": "robot_right_right_base_yaw"},
                 objects={"cup": {"carrier": "right", "release": "*: let go"}},
                 segments={"right": [{"phases": ["*: reach"], "blend": ["world", "cup"]},
                                     {"phases": ["*: grasp", "*: lift", "*: carry"], "anchor": "cup"},
                                     {"phases": ["*: place", "*: retreat"], "blend": ["cup", "world"]}]})
    Q, overrides, report = ep.adapt({"objects": {"cup": [0.05, -0.03, 20.0]}, "robot": {}})
    # or draw layouts until n pass every check:
    samples = adapt_trajectory_to_layout(..., ranges={"cup": {"dx": [-.05, .05], "dy": [-.05, .05], "dyaw_deg": [-30, 30]}}, num_samples=5)

The verified trajectory proved a set of hand-object RELATIVE motions. Each
frame of an end effector is anchored to an object (the hand keeps its pose
relative to where that object started), to the world (home, goals that did
not move), or blends between two anchors (reaching, carrying). Moving an
object moves every frame anchored to it by the same rigid transform; the
arm is re-solved by IK; a carried object rides its hand; and the result has
to pass reach, joint speed, object continuity and all-geometry penetration
checks relative to the original, or the layout is discarded.
"""
from __future__ import annotations

TOOL = {
    "name": "adapt_trajectory_to_layout",
    "title": "Adapt trajectory to layout",
    "type": "skill",
    "category": "Trajectory & Execution",
    "summary": "Re-anchor a verified episode to displaced objects (object-anchored trajectory warping), re-solve the arms by IK and accept only layouts that pass the feasibility checks.",
    "description": (
        "Loads the scene and the verified qpos trajectory, discovers each end effector's chain "
        "and the arm bolted to it (a position-only weld or connect leaves the arm's "
        "orientation free), and builds a per-frame anchor schedule per side from segments "
        "given by phase-label patterns or frame ranges: anchor = one object or 'world', or a "
        "smoothstep blend from one anchor to another. Anchors may only change through a "
        "blend, otherwise the hand would jump; the schedule is validated. For a layout (dx, "
        "dy, dyaw per object, a joint-space start offset per arm that fades out before the "
        "first contact phase) it displaces every anchored end-effector frame rigidly, "
        "re-solves the end effector and the arm by IK with retries, moves a carried object "
        "with its carrier from the frame it first moves until its release phase (then it "
        "rests with whatever its carrier was anchored to), moves static objects in the model, "
        "and runs the checks: IK reach (2 mm, 1 deg), joint speed (1.5 x original + 0.5), "
        "object step continuity (10 mm), and penetration on all visible geometry relative to "
        "the original (3 mm). adapt_trajectory_to_layout draws layouts from ranges until "
        "num_samples pass and writes each as trajectory.npz + sample.json. A zero layout "
        "reproduces the input exactly."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF scene (.xml; the penetration check recompiles it)."},
        {"name": "trajectory", "type": "str", "description": "trajectory.npz with qpos, labels, fps."},
        {"name": "end_effectors", "type": "dict side -> body name", "description": "Hand root per side."},
        {"name": "objects", "type": "dict name -> {carrier?, release?, attached_to?}", "description": "Objects that may be displaced; carrier = side that moves it; release = phase pattern or frame where it is put down; attached_to = rides another object's start."},
        {"name": "segments", "type": "dict side -> list[{phases | frames, anchor | blend}]", "description": "Anchor schedule per side."},
        {"name": "ranges", "type": "dict name -> {dx, dy, dyaw_deg}", "description": "Sampling ranges (metres, degrees) for adapt_trajectory_to_layout."},
        {"name": "num_samples, max_attempts, seed, init_jitter_rad, init_blend_frames", "type": "int, int, int, float, int", "description": "Sampling controls. Defaults 1, 50, 0, 0.05, 45."},
        {"name": "checks", "type": "dict | None", "description": "Override ik_pos_tol_m, ik_rot_tol_deg, joint_speed_scale, joint_speed_floor, penetration_tol_m, object_step_tol_m."},
        {"name": "out_dir", "type": "str | None", "description": "Where samples are written."},
    ],
    "outputs": [
        {"name": "samples", "type": "list[dict]", "description": "Per accepted sample: layout, qpos [T, nq], body_overrides for static objects, checks report, path."},
        {"name": "summary", "type": "dict", "description": "accepted, attempts, rejected reasons per attempt."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["randomization", "augmentation", "anchoring", "ik", "stage 4"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope", "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting", "Biology Experiment"],
    "status": "tested",
    "entrypoint": "adapt_trajectory_to_layout",
}

import copy
import json
from pathlib import Path

import mujoco
import numpy as np

from tool_library.common import trajectory as TJ

IDENT = (np.eye(3), np.zeros(3))
MOVE_TOL = 0.002           # m: an object that moved less than this is still at its start
CHECKS = {"ik_pos_tol_m": 0.002, "ik_rot_tol_deg": 1.0, "joint_speed_scale": 1.5, "joint_speed_floor": 0.5,
          "penetration_tol_m": 0.003, "object_step_tol_m": 0.01}


def _is_identity(D):
    return np.allclose(D[0], np.eye(3), atol=1e-12) and np.allclose(D[1], 0.0, atol=1e-12)


class Episode:
    """The scene, the verified trajectory, and everything derived from them."""

    def __init__(self, scene, trajectory, end_effectors, objects, segments, checks=None,
                 init_jitter_rad=0.05, init_blend_frames=45, visual_groups=(0, 1, 2)):
        self.scene = str(scene)
        self.model = m = TJ.load_model(scene)
        self.Q, self.labels, self.fps = TJ.as_trajectory(trajectory)
        self.T = len(self.Q)
        if self.Q.shape[1] != m.nq:
            raise ValueError(f"trajectory has {self.Q.shape[1]} qpos columns, the scene has {m.nq}")
        self.checks = {**CHECKS, **(checks or {})}
        self.init_jitter_rad, self.init_blend_frames = init_jitter_rad, init_blend_frames
        self.specs = {k: dict(v or {}) for k, v in objects.items()}
        self.sides = TJ.discover_sides(m, end_effectors)
        self.objects = {}
        for name in self.specs:
            bid = TJ.body_id(m, name)
            adr = TJ.free_joint_qadr(m, bid)
            if adr is None and not (m.body_parentid[bid] == 0 and m.body_jntnum[bid] == 0):
                raise ValueError(f"'{name}' must have a free joint or be a fixed body on the world")
            self.objects[name] = {"bid": bid, "qadr": adr}
        # reference forward kinematics
        data = mujoco.MjData(m)
        track = {}
        for side, s in self.sides.items():
            track[("ee", side)] = s["ee"].bid
            if s["follower"]:
                track[("tip", side)] = s["follower"]["chain"].bid
        for name, o in self.objects.items():
            track[("obj", name)] = o["bid"]
        self.ref = {k: (np.zeros((self.T, 3)), np.zeros((self.T, 3, 3))) for k in track}
        for t in range(self.T):
            data.qpos[:] = self.Q[t]
            mujoco.mj_kinematics(m, data)
            for k, bid in track.items():
                p, R = data.xpos[bid].copy(), data.xmat[bid].reshape(3, 3).copy()
                if k[0] == "tip":
                    p = p + R @ self.sides[k[1]]["follower"]["point"]
                self.ref[k][0][t], self.ref[k][1][t] = p, R
        # a position-only weld in the scene does not mean the trajectory's hand turns on the flange: a Stage 3 episode
        # planned with the arm keeps the hand bolted (the relative rotation never changes). Then the arm follows the
        # FULL pose, or an adapted layout would twist the hand on its flange.
        self.rigid_mount = {}
        for side, s in self.sides.items():
            fol = s["follower"]
            if not fol or not fol["pos_only"]:
                continue
            Re, Rt = self.ref[("ee", side)][1], self.ref[("tip", side)][1]
            rel = np.einsum("tji,tjk->tik", Re, Rt)            # tip rotation in the hand frame, per frame
            dev = max(TJ.angle_between(rel[0], r) for r in rel[:: max(1, self.T // 300)])
            # the flange follows its ORIGINAL pose moved with the hand, whatever the scene's weld says: the hand keeps
            # exactly the relation to its flange the verified trajectory had (bolted, or the twin's own turn). A frame
            # where that full pose is out of reach falls back to the weld's position-only follow.
            fol["pos_only"] = False
            fol["scene_pos_only"] = True
            self.rigid_mount[side] = round(float(np.degrees(dev)), 3)
        self.schedule = {side: self._schedule(segments.get(side, [])) for side in self.sides}
        for side, sched in self.schedule.items():
            self._check_schedule(side, sched)
        for name, o in self.objects.items():
            p = self.ref[("obj", name)][0]
            moved = np.flatnonzero(np.linalg.norm(p - p[0], axis=1) > MOVE_TOL)
            o["t_move"] = int(moved[0]) if len(moved) else None
            spec = self.specs[name]
            carrier = spec.get("carrier")
            if carrier is not None and (o["qadr"] is None or carrier not in self.sides):
                raise ValueError(f"'{name}': carrier must be a side with an end effector, and the object free")
            o["carrier"] = carrier
            rel = spec.get("release")
            if isinstance(rel, str):
                hits = TJ.frames_matching(self.labels, rel)
                if not hits:
                    raise ValueError(f"release phase '{rel}' matches no frame")
                rel = hits[0]
            o["t_release"], o["rest_anchor"] = rel, None
            if rel is not None:
                if carrier is None:
                    raise ValueError(f"'{name}' has a release but no carrier")
                a, b, w = self.schedule[carrier]
                o["rest_anchor"] = b[rel] if w[rel] >= 0.5 else a[rel]
        self.cmodel, self.n_collision_on = TJ.collidable_model(self.scene, visual_groups)
        if (self.cmodel.nq, self.cmodel.ngeom, self.cmodel.nbody) != (m.nq, m.ngeom, m.nbody):
            raise ValueError("the collision copy of the scene does not match the scene")
        self.limbs = TJ.limb_bodies(self.cmodel, self.sides)
        self._baseline = None

    # -- anchors -----------------------------------------------------------
    def _schedule(self, segments):
        a, b, w = ["world"] * self.T, ["world"] * self.T, np.zeros(self.T)
        for seg in segments:
            if "frames" in seg:
                lo, hi = seg["frames"]
                idx = list(range(max(0, lo), min(self.T, hi + 1)))
            else:
                idx = TJ.frames_matching(self.labels, seg["phases"])
                if not idx:
                    raise ValueError(f"segment {seg['phases']} matches no frame of the trajectory")
            names = [seg["anchor"]] * 2 if "anchor" in seg else list(seg["blend"])
            for n in names:
                if n != "world" and n not in self.objects:
                    raise ValueError(f"segment anchor '{n}' is not 'world' or a listed object")
            runs, start = [], 0
            for i in range(1, len(idx) + 1):
                if i == len(idx) or idx[i] != idx[i - 1] + 1:
                    runs.append(idx[start:i]); start = i
            for run in runs:
                for k, t in enumerate(run):
                    a[t], b[t] = names
                    w[t] = 0.0 if "anchor" in seg else float(TJ.smoothstep(k / max(1, len(run) - 1)))
        return a, b, w

    def _check_schedule(self, side, sched):
        a, b, w = sched
        for t in range(1, self.T):
            if (a[t], b[t]) == (a[t - 1], b[t - 1]):
                continue
            end = b[t - 1] if w[t - 1] >= 0.999 else (a[t - 1] if w[t - 1] <= 0.001 else None)
            start = a[t] if w[t] <= 0.001 else None
            if end is not None and start is not None and end == start:
                continue
            raise ValueError(
                f"segments for '{side}' jump at frame {t} ('{self.labels[t - 1]}' -> '{self.labels[t]}'): "
                f"from {a[t - 1]}->{b[t - 1]} (w={w[t - 1]:.2f}) to {a[t]}->{b[t]} (w={w[t]:.2f}); "
                f"add a blend between the two anchors")

    def baseline(self):
        if self._baseline is None:
            self._baseline = self._profile(self.Q)
        return self._baseline

    def _profile(self, Q):
        prof = TJ.contact_profile(self.cmodel, Q)
        joined = TJ.joined_bodies(self.cmodel)
        out = {}
        for key, (depth, t, (b1, b2), geoms) in prof.items():
            if (b1, b2) in joined or any(b1 in L and b2 in L for L in self.limbs):
                continue
            label = tuple(f"{self.cmodel.body(b).name}/{g}" for b, g in zip((b1, b2), geoms))
            out[key] = (depth, t, label)
        return out

    # -- layout ------------------------------------------------------------
    def sample_layout(self, rng, ranges):
        objs = {}
        for name in self.specs:
            r = ranges.get(name, {})
            if self.specs[name].get("attached_to"):
                objs[name] = [0.0, 0.0, 0.0]; continue
            objs[name] = [float(rng.uniform(*r.get("dx", [0.0, 0.0]))), float(rng.uniform(*r.get("dy", [0.0, 0.0]))),
                          float(rng.uniform(*r.get("dyaw_deg", [0.0, 0.0])))]
        robot = {}
        for side, s in self.sides.items():
            n = len((s["follower"]["chain"] if s["follower"] else s["ee"]).qadr)
            robot[side] = [float(v) for v in rng.uniform(-self.init_jitter_rad, self.init_jitter_rad, n)]
        return {"objects": objs, "robot": robot}

    def _displacements(self, layout):
        D = {"world": IDENT}
        for name, (dx, dy, dyaw) in layout["objects"].items():
            if self.specs[name].get("attached_to"):
                continue
            p0 = self.ref[("obj", name)][0][0]
            dR = TJ.rot_z(np.radians(dyaw))
            D[name] = (dR, p0 + np.array([dx, dy, 0.0]) - dR @ p0)
        for name, spec in self.specs.items():
            if spec.get("attached_to"):
                if spec["attached_to"] not in D:
                    raise ValueError(f"'{name}' is attached_to '{spec['attached_to']}', which is not a listed object")
                D[name] = D[spec["attached_to"]]
        return D

    def _jitter_fade(self, side):
        P = self.ref[("ee", side)][0]
        moved = np.flatnonzero(np.linalg.norm(P - P[0], axis=1) > 1e-3)
        if not len(moved):
            return np.ones(self.T)
        t0, n = int(moved[0]), int(self.init_blend_frames)
        a, b, w = self.schedule[side]
        for t in range(t0, self.T):
            if (a[t] != "world" and w[t] == 0.0) or (b[t] != "world" and w[t] == 1.0):
                n = min(n, max(1, t - t0)); break
        fade = np.ones(self.T)
        fade[t0:] = 1.0 - TJ.smoothstep(np.arange(self.T - t0) / n)
        return fade

    # -- the adaptation ----------------------------------------------------
    def adapt(self, layout):
        """Returns (qpos trajectory, body overrides for static objects, report)."""
        m, ck = self.model, self.checks
        layout = {"objects": {n: layout.get("objects", {}).get(n, [0.0, 0.0, 0.0]) for n in self.specs},
                  "robot": layout.get("robot", {})}
        D = self._displacements(layout)
        Q = self.Q.copy()
        model, cmodel, overrides = m, self.cmodel, {}
        for name, o in self.objects.items():
            if o["qadr"] is None and not np.allclose(layout["objects"][name], 0.0):
                if model is m:
                    model, cmodel = copy.deepcopy(m), copy.deepcopy(self.cmodel)
                p, R = TJ.apply_rigid(D[name], m.body_pos[o["bid"]].copy(), TJ.quat2mat(m.body_quat[o["bid"]]))
                for mm in (model, cmodel):
                    mm.body_pos[o["bid"]], mm.body_quat[o["bid"]] = p, TJ.mat2quat(R)
                overrides[name] = {"pos": p.tolist(), "quat_wxyz": TJ.mat2quat(R).tolist()}
        data = mujoco.MjData(model)
        side_D = {}
        for side in self.sides:
            a, b, w = self.schedule[side]
            P, R = self.ref[("ee", side)]
            Dn = []
            for t in range(self.T):
                pa, Ra = TJ.apply_rigid(D[a[t]], P[t], R[t])
                if a[t] == b[t] or w[t] == 0.0:
                    p2, R2 = pa, Ra
                else:
                    pb, Rb = TJ.apply_rigid(D[b[t]], P[t], R[t])
                    p2, R2 = (1 - w[t]) * pa + w[t] * pb, TJ.slerp(Ra, Rb, w[t])
                dR = R2 @ R[t].T
                Dn.append((dR, p2 - dR @ P[t]))
            side_D[side] = Dn
        for name, o in self.objects.items():
            if o["qadr"] is None:
                continue
            P, R = self.ref[("obj", name)]
            adr = o["qadr"]
            for t in range(self.T):
                if o["carrier"] is None or o["t_move"] is None or t < o["t_move"]:
                    d = D[name]
                elif o["t_release"] is not None and t >= o["t_release"]:
                    d = D[o["rest_anchor"]]
                else:
                    d = side_D[o["carrier"]][t]
                if _is_identity(d):
                    continue
                p2, R2 = TJ.apply_rigid(d, P[t], R[t])
                q2 = TJ.mat2quat(R2)
                if np.dot(q2, self.Q[t, adr + 3:adr + 7]) < 0:
                    q2 = -q2
                Q[t, adr:adr + 3], Q[t, adr + 3:adr + 7] = p2, q2
        report = {"ik": {}}
        tol_p, tol_r = ck["ik_pos_tol_m"], np.radians(ck["ik_rot_tol_deg"])
        for side, s in self.sides.items():
            for mode in (("full", "pos") if (s["follower"] and s["follower"].get("scene_pos_only")) else ("as_is",)):
                fol0 = s["follower"]
                if mode == "pos":
                    fol0["pos_only"] = True                # the full pose could not be held on every frame
                cols = list(s["ee"].qadr) + (list(fol0["chain"].qadr) if fol0 else [])
                Q_save = Q[:, cols].copy()
                Dn, ee, fol = side_D[side], s["ee"], s["follower"]
                arm = fol["chain"] if fol else None
                driver = arm or ee
                jit = np.asarray(layout["robot"].get(side, np.zeros(len(driver.qadr))), float)
                fade = self._jitter_fade(side) if np.any(jit) else np.zeros(self.T)
                worst = {"pos_m": 0.0, "rot_deg": 0.0, "frame": 0}
                moved = 0
                rng = np.random.default_rng(0)

                def note(ep, er, t):
                    if ep > worst["pos_m"] or np.degrees(er) > worst["rot_deg"]:
                        worst.update(pos_m=max(ep, worst["pos_m"]), frame=t, rot_deg=max(float(np.degrees(er)), worst["rot_deg"]))

                def seed(ch, t):
                    redundant = ch is arm and fol["pos_only"]
                    if t > 0 and not redundant:
                        data.qpos[ch.qadr] = self.Q[t, ch.qadr] + (Q[t - 1, ch.qadr] - self.Q[t - 1, ch.qadr]
                                                                   - (jit * fade[t - 1] if ch is driver else 0.0))

                def solve(ch, t, p, R=None, point=None, rest=None):
                    start = data.qpos[ch.qadr].copy()
                    best = (ch.solve(data, p, R, point=point, rest=rest), data.qpos[ch.qadr].copy())
                    seeds = ([Q[t - 1, ch.qadr]] if t > 0 else []) + [start + rng.normal(0.0, 0.25, len(start)) for _ in range(10)]
                    for sd in seeds:
                        if best[0][0] <= tol_p and best[0][1] <= tol_r:
                            break
                        data.qpos[ch.qadr] = sd
                        err = ch.solve(data, p, R, point=point)
                        if err[0] + 0.1 * err[1] < best[0][0] + 0.1 * best[0][1]:
                            best = (err, data.qpos[ch.qadr].copy())
                    data.qpos[ch.qadr] = best[1]
                    return best[0]

                for t in range(self.T):
                    if _is_identity(Dn[t]) and fade[t] == 0.0:
                        continue
                    moved += 1
                    data.qpos[:] = Q[t]
                    if not _is_identity(Dn[t]):
                        seed(ee, t)
                        note(*solve(ee, t, *TJ.apply_rigid(Dn[t], *[x[t] for x in self.ref[("ee", side)]])), t)
                        Q[t, ee.qadr] = data.qpos[ee.qadr]
                        if arm:
                            seed(arm, t)
                            p, R = TJ.apply_rigid(Dn[t], *[x[t] for x in self.ref[("tip", side)]])
                            note(*solve(arm, t, p, None if fol["pos_only"] else R, point=fol["point"], rest=self.Q[t, arm.qadr]), t)
                            Q[t, arm.qadr] = data.qpos[arm.qadr]
                    if fade[t] > 0.0:
                        q = Q[t, driver.qadr] + jit * fade[t]
                        q = np.where(driver.limited, np.clip(q, driver.lo, driver.hi), q)
                        if arm:
                            mujoco.mj_kinematics(model, data)
                            p_tip, R_tip = data.xpos[arm.bid].copy(), data.xmat[arm.bid].reshape(3, 3).copy()
                            p_tip = p_tip + R_tip @ fol["point"]
                            p_ee, R_ee = data.xpos[ee.bid].copy(), data.xmat[ee.bid].reshape(3, 3).copy()
                            data.qpos[arm.qadr] = q
                            mujoco.mj_kinematics(model, data)
                            p2, R2 = data.xpos[arm.bid].copy(), data.xmat[arm.bid].reshape(3, 3).copy()
                            p2 = p2 + R2 @ fol["point"]
                            dR = np.eye(3) if fol["pos_only"] else R2 @ R_tip.T
                            note(*solve(ee, t, p2 + dR @ (p_ee - p_tip), dR @ R_ee), t)
                            Q[t, ee.qadr] = data.qpos[ee.qadr]
                        Q[t, driver.qadr] = q
                worst["frames_adapted"] = moved
                worst["arm_mode"] = mode
                if mode == "full" and (worst["pos_m"] > tol_p or np.radians(worst["rot_deg"]) > tol_r):
                    Q[:, cols] = Q_save                     # try the whole side again, position-only
                    continue
                break
            if s["follower"] and s["follower"].get("scene_pos_only"):
                s["follower"]["pos_only"] = False          # the next layout starts from the full pose again
            report["ik"][side] = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in worst.items()}
        report["ik"]["ok"] = all(r["pos_m"] <= ck["ik_pos_tol_m"] and r["rot_deg"] <= ck["ik_rot_tol_deg"]
                                 for r in report["ik"].values() if isinstance(r, dict))
        # joint speed vs the original
        chains = TJ.robot_chains(model, self.sides)
        js, ok = {}, True
        for name, qadr in chains.items():
            v = float(np.abs(np.diff(Q[:, qadr], axis=0)).max() * self.fps)
            v0 = float(np.abs(np.diff(self.Q[:, qadr], axis=0)).max() * self.fps)
            lim = ck["joint_speed_scale"] * v0 + ck["joint_speed_floor"]
            js[name] = {"max": round(v, 3), "original": round(v0, 3), "limit": round(lim, 3)}
            ok &= v <= lim
        js["ok"] = bool(ok); report["joint_speed"] = js
        st, ok = {}, True
        for name, o in self.objects.items():
            if o["qadr"] is None:
                continue
            a = o["qadr"]
            step = np.linalg.norm(np.diff(Q[:, a:a + 3], axis=0), axis=1)
            step0 = np.linalg.norm(np.diff(self.Q[:, a:a + 3], axis=0), axis=1)
            worst_s = float((step - step0).max()) if len(step) else 0.0
            st[name] = round(worst_s, 5); ok &= worst_s <= ck["object_step_tol_m"]
        st["ok"] = bool(ok); report["object_steps"] = st
        prof, base = self._profile(Q) if cmodel is self.cmodel else self._profile_with(cmodel, Q), self.baseline()
        worst_p, where = 0.0, None
        for key, (depth, t, geoms) in prof.items():
            excess = depth - base.get(key, (0.0, 0, None))[0]
            if excess > worst_p:
                worst_p, where = excess, (geoms, t, depth)
        pen = {"new_penetration_m": round(worst_p, 5), "ok": worst_p <= ck["penetration_tol_m"]}
        if where:
            pen.update(pair=list(where[0]), frame=int(where[1]), depth_m=round(where[2], 5))
        report["penetration"] = pen
        keys = ("ik", "joint_speed", "object_steps", "penetration")
        report["failed"] = [k for k in keys if not report[k]["ok"]]
        report["ok"] = not report["failed"]
        return Q, overrides, report

    def _profile_with(self, cmodel, Q):
        saved, self.cmodel = self.cmodel, cmodel
        try:
            return self._profile(Q)
        finally:
            self.cmodel = saved

    def joint_names(self):
        m, names = self.model, [""] * self.model.nq
        for j in range(m.njnt):
            n = {TJ.FREE: 7, TJ.BALL: 4}.get(int(m.jnt_type[j]), 1)
            for k in range(n):
                names[m.jnt_qposadr[j] + k] = m.joint(j).name + (f"[{k}]" if n > 1 else "")
        return names


def adapt_trajectory_to_layout(scene, trajectory, end_effectors, objects, segments, ranges, num_samples=1,
                               max_attempts=50, seed=0, out_dir=None, checks=None, init_jitter_rad=0.05,
                               init_blend_frames=45, include_original=False, log=print):
    ep = Episode(scene, trajectory, end_effectors, objects, segments, checks, init_jitter_rad, init_blend_frames)
    out = Path(out_dir) if out_dir else None
    samples, rows = [], []
    if include_original:
        zero = {"objects": {n: [0.0, 0.0, 0.0] for n in ep.specs}, "robot": {}}
        Q, ov, rep = ep.adapt(zero)
        if not np.array_equal(Q, ep.Q):
            raise RuntimeError("a zero layout must reproduce the trajectory exactly")
        samples.append(_save(ep, out, "original", Q, zero, ov, rep, None))
    for attempt in range(int(max_attempts)):
        if len([s for s in samples if s["name"] != "original"]) >= num_samples:
            break
        rng = np.random.default_rng([int(seed), attempt])
        layout = ep.sample_layout(rng, ranges)
        Q, ov, rep = ep.adapt(layout)
        rows.append({"attempt": attempt, "ok": rep["ok"], "failed": rep["failed"], "layout": layout["objects"]})
        if rep["ok"]:
            name = f"{len([s for s in samples if s['name'] != 'original']):04d}"
            samples.append(_save(ep, out, name, Q, layout, ov, rep, attempt))
        log(f"[adapt] attempt {attempt}: {'OK' if rep['ok'] else 'rejected (' + ', '.join(rep['failed']) + ')'}  "
            f"{json.dumps(layout['objects'])}")
    n_ok = len([s for s in samples if s["name"] != "original"])
    summary = {"accepted": n_ok, "attempts": len(rows), "wanted": num_samples, "rows": rows, "ok": n_ok >= num_samples}
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "adapt.json").write_text(json.dumps(summary, indent=1))
    return samples, summary


def _save(ep, out, name, Q, layout, overrides, report, seed):
    s = {"name": name, "layout": layout, "qpos": Q, "body_overrides": overrides, "checks": report, "seed": seed}
    if out:
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "trajectory.npz", qpos=Q, fps=ep.fps, labels=np.array(ep.labels),
                            joint_names=np.array(ep.joint_names()))
        (d / "sample.json").write_text(json.dumps({k: v for k, v in s.items() if k != "qpos"}, indent=1, default=str))
        s["path"] = str(d)
    return s
