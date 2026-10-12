"""Physics replay of an augmented episode (checks.physics_replay).

The kinematic checks prove that the robot can follow the adapted trajectory
without new collisions. They do not prove that the objects still do what the
task needs: that a brick still lifts when the hand closes on it at its new
place, or that it stays on the brick it was stacked on. So every layout that
passes the kinematic checks is run again in contact physics, the way Stage 3
executed the original:

  - the scene gets the execution model of Stage 3 (hand collision hulls paired
    with every object and the table, elliptic cones, no-slip iterations);
  - the hands' position servos follow the adapted COMMANDS (the original
    commands moved with the hands, see Episode.adapt_reference), segment by
    segment as Stage 3 ran its subgoals, each from the state the last one
    left; the objects move only by contact; the arms follow the adapted path;
  - every subgoal expectation of the Stage 3 manifest (rise, in hand, no
    slip, at target, released, at rest, supported on, untouched, penetration,
    servo tracking, ...) is re-measured on the replay with the library's own
    outcome evaluation, its targets moved with the objects they belong to;
  - every randomized object must also end where the adapted plan puts it,
    and the simulation must stay stable.

The original layout must pass its replay before any augmented episode can
be accepted. A failed original check is never removed from later judgements.

Deformable objects (deformables.json) are not in the MuJoCo scene: once the
robot's replay holds, the cords run in Isaac Sim driven by the replayed robot
(deformables.py), and the Stage 1 goals and the expectations that name a cord
are judged on the Isaac mirror with the same code.

This module uses the DexAgent tool library that ships next to Stage 4
(tool_library.common.episode and .outcome) read-only.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import mujoco
import numpy as np

IDENT = (np.eye(3), np.zeros(3))

def library():
    """tool_library.common.{episode, outcome, hand}, from the repository Stage 4 ships in."""
    try:
        import tool_library  # noqa: F401
    except ImportError:
        root = Path(__file__).resolve().parents[2]
        if not (root / "tool_library").is_dir():
            raise ImportError("checks.physics_replay needs the DexAgent tool library next to Stage 4 "
                              f"(expected {root / 'tool_library'})")
        sys.path.insert(0, str(root))
    from tool_library.common import episode as E
    from tool_library.common import hand as H
    from tool_library.common import outcome as O
    return E, O, H


def load_subgoals(path, labels, subgoal_ids=None):
    """The manifest's executed subgoals as [{name, frames [a, b], expect, plan, motion}], in frame order.

    An entry's window is its own `frames` when given, else the frames whose
    per-frame subgoal index (the trajectory's `subgoal` array) is its index."""
    if path is None or not Path(path).exists():
        return []
    man = json.loads(Path(path).read_text())
    entries = man.get("subgoals", man) if isinstance(man, dict) else man
    out = []
    for sg in entries or []:
        if not isinstance(sg, dict) or sg.get("skipped"):
            continue
        fr = sg.get("frames")
        if not fr and subgoal_ids is not None and sg.get("index") is not None:
            idx = np.flatnonzero(np.asarray(subgoal_ids) == int(sg["index"]))
            fr = [int(idx[0]), int(idx[-1])] if len(idx) else None
        if not fr:
            continue
        a, b = int(fr[0]), int(fr[-1])
        if not (0 <= a <= b < len(labels)):
            continue
        plan = sg.get("plan")
        if isinstance(plan, list):                     # a plan given as a list (bites, alternatives): its first entry
            plan = next((x for x in plan if isinstance(x, dict)), {})
        plan = plan if isinstance(plan, dict) else {}
        step = plan.get("step") or sg.get("step")
        name = f"subgoal {sg.get('index')}" + (f".{step}" if step else "") + f" {sg.get('motion', '')}".rstrip()
        out.append({"name": name, "frames": [a, b], "expect": sg.get("expect") if isinstance(sg.get("expect"), dict) else None, "plan": plan,
                    "motion": str(sg.get("motion") or plan.get("motion") or "")})
    out.sort(key=lambda r: r["frames"][0])
    return out


class Replay:
    """Replays layouts of one Episode in contact physics and judges them."""

    def __init__(self, ep, cfg, log=print):
        self.E, self.O, self.H = library()
        self.ep, self.cfg, self.log = ep, cfg, log
        t0 = time.time()
        from tool_library.common.stage4 import load_runtime_pair, physics_rules
        pc = cfg.get("physics") or {}
        self.scene_recon = cfg.path_of(pc.get("scene_recon_json"))
        if self.scene_recon is not None:
            if hashlib.sha256(self.scene_recon.read_bytes()).hexdigest() != pc.get("scene_recon_sha256"):
                raise ValueError("Stage 4 mechanism evidence does not match its trusted Stage 2 handoff hash")
            self.rules = physics_rules(self.scene_recon)
            if self.rules != list(pc.get("rules") or []):
                raise ValueError("Stage 4 mechanism rules differ from the supplied Stage 2 evidence")
        else:
            self.rules = []
            if pc.get("rules"):
                raise ValueError("Stage 4 mechanism rules require the trusted Stage 2 reconstruction")
        _scene, _trajectory, self.model, z = load_runtime_pair(cfg.path("scene"), cfg.path("trajectory"), self.scene_recon)
        self._scene_workspaces = []
        self.last_scene_path = cfg.path("scene").resolve()
        self.source_scene_sha256 = hashlib.sha256(self.last_scene_path.read_bytes()).hexdigest()
        # Reload the serialized runtime as executed. Rebuilding collision pairs
        # or solver options here would replay a different physical system.
        self.n_pairs = self.model.npair
        assert self.model.nq == ep.model.nq, "the execution model of the scene changed its qpos width"
        from tool_library.common import rigid_robot as R
        if not R.is_rigid(self.model):
            raise ValueError("Stage 4 requires the rigid runtime scene; floating wrists are planning proposals")
        self.parts = self.E.robot_parts(self.model)
        if str(np.asarray(z.get("robot_schema", "")).item()) != R.SCHEMA:
            raise ValueError("Stage 4 trajectory has no rigid runtime schema")
        if str(np.asarray(z.get("scene_sha256", "")).item()) != hashlib.sha256(cfg.path("scene").read_bytes()).hexdigest():
            raise ValueError("Stage 4 trajectory and runtime scene hashes do not match")
        ref = np.asarray(z["reference"], float) if "reference" in z else None
        self.has_reference = ref is not None and ref.shape == ep.Q.shape
        if not self.has_reference:
            raise ValueError("Stage 4 requires the runtime servo reference that produced Stage 3")
        self.Qref = ref
        for key, width in (("initial_qpos", self.model.nq), ("initial_qvel", self.model.nv)):
            if key not in z or np.asarray(z[key]).shape != (width,) or not np.isfinite(z[key]).all():
                raise ValueError(f"Stage 4 is missing a valid {key}")
        self.initial_state = {k: np.asarray(z[k]).copy() for k in ("initial_qpos", "initial_qvel")}
        self.initial_state["eq_active_start"] = np.asarray(z["eq_active_start"] if "eq_active_start" in z
                                                          else self.model.eq_active0, np.int8)
        for key in ("initial_state", "state_spec", "initial_act", "eq_data_start"):
            if key in z:
                self.initial_state[key] = np.asarray(z[key]).copy()
        if "mechanism_initial" in z:
            self.initial_state["mechanism_initial"] = json.loads(str(z["mechanism_initial"]))
        self.last_execution = None
        self.baseline_execution = None
        sub_ids = z["subgoal"] if "subgoal" in z else None
        man = cfg.path_of(cfg.get("manifest"))
        self.subgoals = load_subgoals(man, ep.labels, sub_ids)
        self.segments = self._segments(self.subgoals, ep.T)
        # constraints Stage 3's execution switched off, and when: every segment after that starts with them off
        self.releases = [r for r in (cfg.get("physics") or {}).get("releases") or [] if isinstance(r, dict)]
        self.end_tol = float(cfg["checks"]["physics_end_tol_m"])
        self.goal = self._goal_context(cfg, man)
        self.baseline = None
        self.baseline_qpos = None
        # the replay must use the MuJoCo the episode was executed with: contact settling, friction cones and solver
        # details change between releases (3.10 -> 3.15 moved battery eject distances), and a replay that differs for
        # that reason alone judges nothing
        want = (cfg.get("physics") or {}).get("mujoco_version")
        self.mujoco_version = {"replay": mujoco.__version__, "stage3": want}
        if want and str(want) != mujoco.__version__:
            log(f"[physics] warning: the Stage 3 episode was executed with MuJoCo {want}, this replay runs {mujoco.__version__}; "
                f"replayed outcomes can drift for that reason alone (pip install mujoco=={want})")
        # deformable objects: simulated in Isaac Sim after the robot's MuJoCo replay (stage4/deformables.py)
        self.deformables = getattr(ep, "deformables", None)
        self.last_deformables = None
        self.baseline_deformables = None
        self._isaac_dirs = []
        if self.deformables is not None:
            from . import deformables as DF
            self.isaac_python = DF.isaac_python(cfg)
            self.cord_names = sorted(self.deformables["cords"])
        n_exp = sum(1 for s in self.subgoals if s["expect"])
        log(f"[physics] execution model: {self.n_pairs} hand contact pairs; "
            f"{len(self.segments)} segments; {n_exp} subgoal expectations"
            + ("" if man and Path(man).exists() else " (no manifest: stability and end poses only)")
            + f" ({time.time() - t0:.1f}s)")

    # -- segments ------------------------------------------------------------
    def close(self):
        for workspace in getattr(self, "_scene_workspaces", []):
            workspace.cleanup()
        self._scene_workspaces = []
        import shutil
        for d in getattr(self, "_isaac_dirs", []):
            shutil.rmtree(d, ignore_errors=True)
        self._isaac_dirs = []

    def __del__(self):
        self.close()

    @staticmethod
    def _segments(subgoals, T):
        """[(first, last, servo limits)]: the manifest's subgoal windows, gaps filled, else one segment."""
        segs, t = [], 0
        for sg in subgoals:
            a, b = sg["frames"]
            if a < t:                                   # overlapping windows: keep the frame order
                a = t
            if a > b:
                continue
            if a > t:
                segs.append((t, a - 1, {}))
            p = sg["plan"]
            lim = {k: p[k] for k in ("finger_force_nm", "wrist_force_n", "wrist_torque_nm") if p.get(k) is not None}
            if sg["motion"].endswith("_bite") and "wrist_force_n" not in lim:
                lim["wrist_force_n"] = [10.0, 10.0, 40.0]    # what the Stage 3 executor gives a bite
            segs.append((a, b, lim))
            t = b + 1
        if t < T:
            segs.append((t, T - 1, {}))
        return segs

    # -- the replay ----------------------------------------------------------
    def model_for(self, overrides):
        """Compile the exact scene bytes that this replay will record and audit."""
        source = self.cfg.path("scene").resolve()
        if not overrides:
            self.last_scene_path = source
            return mujoco.MjModel.from_xml_path(str(source))
        # Canonicalize before simulation; the same file is later audited and
        # copied verbatim to the published sample. Assets are already portable.
        from tool_library.skills.write_stage4_config import _copy_scene
        import xml.etree.ElementTree as ET
        workspace = tempfile.TemporaryDirectory(prefix="stage4_runtime_scene_")
        self._scene_workspaces.append(workspace)
        directory = Path(workspace.name)
        _copy_scene(source, directory)
        scene = directory / "scene.xml"
        root = ET.fromstring(scene.read_bytes())
        for name, ov in overrides.items():
            b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if b < 1 or int(self.model.body_parentid[b]) != 0 or int(self.model.body_jntnum[b]) != 0:
                raise ValueError(f"layout override {name!r} is not a fixed world-root body")
            if name not in self.ep.objects or self.ep.objects[name]["qadr"] is not None:
                raise ValueError(f"layout override {name!r} is not a configured fixed object")
            body = next(node for node in root.findall(".//body") if node.get("name") == name)
            for key, values, width in (("pos", ov["pos"], 3), ("quat", ov["quat_wxyz"], 4)):
                values = np.asarray(values, float)
                if values.shape != (width,) or not np.isfinite(values).all():
                    raise ValueError(f"layout override {name!r} has invalid {key}")
                body.set(key, " ".join(repr(float(v)) for v in values))
        scene.write_text(ET.tostring(root, encoding="unicode"))
        self.last_scene_path = scene
        m = mujoco.MjModel.from_xml_path(str(scene))
        if (m.nq, m.nv, m.nu) != (self.model.nq, self.model.nv, self.model.nu):
            raise ValueError("canonical layout scene changed the runtime joint or actuator layout")
        return m

    def replay(self, Q, Qref, overrides=None, after_segment=None, segments=None, releases=None, initial_state=None):
        """Run the commands Qref segment by segment from Q[0]; the arms follow Q. after_segment(m, Qr, diag, last
        frame) may return True to stop early (a layout that already failed a check). segments / releases default to
        the original episode's (a retimed or re-executed episode brings its own)."""
        E, H = self.E, self.H
        m = self.model_for(overrides)
        T = len(Q)
        Qr = np.zeros_like(Q)
        diag = {k: np.zeros(T) for k in ("hand_pen_m", "object_pen_m", "track_pos_m", "track_rot_deg")}
        if initial_state is None:
            initial_state = (self.initial_state if np.array_equal(Q, self.ep.Q) else
                             {"initial_qpos": Q[0], "initial_qvel": np.zeros(m.nv),
                              "eq_active_start": self.initial_state["eq_active_start"]})
        q = np.asarray(initial_state["initial_qpos"], float).copy()
        velocity = np.asarray(initial_state["initial_qvel"], float).copy()
        equality = np.asarray(initial_state.get("eq_active_start", m.eq_active0), np.int8).copy()
        integration = initial_state.get("initial_state")
        mechanism = copy.deepcopy(initial_state.get("mechanism_initial"))
        first_mechanism = None
        eq_template = initial_state.get("eq_data_start", self.initial_state.get("eq_data_start"))
        if eq_template is not None:
            eq_data = np.asarray(eq_template, float)
            if eq_data.shape != m.eq_data.shape or not np.isfinite(eq_data).all():
                raise ValueError("initial equality parameters do not match the runtime scene")
            m.eq_data[:] = eq_data
        eq_off, unstable, stopped = [], None, None
        fields = {k: [] for k in ("qvel", "ctrl", "time_s", "eq_active", "substep_qpos", "substep_qvel",
                                  "substep_ctrl", "substep_time_s", "substep_eq_active")}
        optional_fields = ("act", "substep_act", "substep_eq_active_before", "substep_eq_data")
        first_state = {}
        offset = 0.0
        latched = {}                                   # connector latches engaged at a segment's end, as Stage 3 carried them
        segments = self.segments if segments is None else segments
        releases = self.releases if releases is None else releases
        for a, b, lim in segments:
            # Release times from the reference are observations, not commands
            # allowed to detach a part in a newly randomized physical scene.
            # The trusted contact mechanism must determine every transition.
            # Segment starts after frame zero are commands following the prior
            # frame. Include that boundary reference and drop its repeated state.
            prefix = int(a > 0)
            commands = Qref[a - prefix:b + 1]
            state_kw = {"state_start": integration} if integration is not None else {}
            r = E.servo_rollout(m, commands, self.ep.fps, qpos_start=q, qvel_start=velocity,
                                eq_active_start=equality,
                                mechanism_state_start=mechanism,
                                eq_inactive=eq_off, rules=self.rules, latched=latched, record_contacts=True,
                                finger_force_nm=lim.get("finger_force_nm", H.FINGER_FORCE_NM),
                                wrist_force_n=lim.get("wrist_force_n", H.WRIST_FORCE_N),
                                wrist_torque_nm=lim.get("wrist_torque_nm", H.WRIST_TORQUE_NM), **state_kw)
            count = len(r["qpos"]) - prefix
            if a == 0:
                first_state = {key: np.asarray(r[key]).copy() for key in
                               ("initial_state", "state_spec", "initial_act", "eq_active_start", "eq_data_start", "initial_time_s") if key in r}
                first_mechanism = r.get("mechanism_initial")
            end = a + count
            Qr[a:end] = r["qpos"][prefix:]
            for k in diag:
                diag[k][a:end] = np.asarray(r[k])[prefix:prefix + count]
            q, eq_off = np.asarray(r["qpos"][-1], float).copy(), list(r["eq_inactive"])
            velocity, equality = np.asarray(r["qvel"][-1], float).copy(), np.asarray(r["eq_active"][-1], np.int8).copy()
            integration = r.get("final_state")
            mechanism = r.get("mechanism_final")
            for key in optional_fields:
                if key in r:
                    fields.setdefault(key, [])
            for key in fields:
                values = np.asarray(r[key]).copy()
                if not key.startswith("substep_"):
                    values = values[prefix:]
                if key in ("time_s", "substep_time_s"):
                    values = values + offset
                fields[key].append(values)
            offset = float(fields["substep_time_s"][-1][-1]) if len(fields["substep_time_s"][-1]) else offset
            latched = dict(r.get("latched") or {})
            if r.get("unstable"):
                unstable = dict(r["unstable"], frame=a + int(r["unstable"]["frame"]))
            if count != b + 1 - a and unstable is None:
                unstable = {"frame": end - 1, "reason": "incomplete physics segment"}
            if unstable is not None or (after_segment is not None and after_segment(m, Qr, diag, end - 1)):
                stopped = end - 1
                Qr = Qr[:end]
                diag = {key: value[:end] for key, value in diag.items()}
                break
        self.last_execution = {key: np.concatenate(values, axis=0) for key, values in fields.items()}
        self.last_execution.update(qpos=Qr, reference=np.asarray(Qref[:len(Qr)], float),
                                   initial_qpos=np.asarray(initial_state["initial_qpos"], float),
                                   initial_qvel=np.asarray(initial_state["initial_qvel"], float),
                                   eq_active_start=np.asarray(initial_state.get("eq_active_start", m.eq_active0), np.int8),
                                   robot_schema=np.asarray("dexagent.rigid_robot.v1"),
                                   actuator_names=np.array([m.actuator(i).name for i in range(m.nu)]),
                                   eq_names=np.array(E.equality_names(m)),
                                   scene_sha256=np.asarray(hashlib.sha256(self.last_scene_path.read_bytes()).hexdigest()),
                                   source_scene_sha256=np.asarray(self.source_scene_sha256),
                                   _scene_path=str(self.last_scene_path))
        self.last_execution.update(first_state)
        if integration is not None:
            self.last_execution["final_state"] = np.asarray(integration).copy()
        if first_mechanism is not None:
            self.last_execution["mechanism_initial"] = np.asarray(json.dumps(first_mechanism, sort_keys=True, allow_nan=False))
            self.last_execution["mechanism_final"] = np.asarray(json.dumps(mechanism, sort_keys=True, allow_nan=False))
        if self.scene_recon is not None:
            self.last_execution["scene_recon_sha256"] = np.asarray(hashlib.sha256(self.scene_recon.read_bytes()).hexdigest())
            self.last_execution["_scene_recon_path"] = str(self.scene_recon)
        return m, Qr, diag, unstable, stopped

    # -- the judgement -------------------------------------------------------
    def _displacement_of(self, body, pl, t):
        """The adapted plan's rigid displacement of a body at frame t (through its randomized ancestor)."""
        m = self.ep.model
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body) if body else -1
        while b > 0:
            name = m.body(b).name
            if name in self.ep.objects:
                if self.ep.objects[name]["qadr"] is None:
                    return pl["D"][name]
                R, tt = pl["obj"][name]
                return R[t], tt[t]
            b = int(m.body_parentid[b])
        return IDENT

    def _expect_for(self, expect, pl, a, b):
        """A subgoal's expectation with its world-frame targets moved with their object."""
        if pl is None:
            return expect
        ex = copy.deepcopy(expect)
        if isinstance(ex.get("hands"), dict):
            ex["hands"] = {side: self._expect_for(value, pl, a, b) for side, value in ex["hands"].items()}
        obj = ex.get("object")
        if obj and ex.get("target_pos") is not None:
            R, t = self._displacement_of(obj, pl, b)
            ex["target_pos"] = (R @ np.asarray(ex["target_pos"], float) + t).tolist()
        if obj and isinstance(ex.get("displacement"), dict) and ex["displacement"].get("dir") is not None:
            R, _ = self._displacement_of(obj, pl, a)
            ex["displacement"]["dir"] = (R @ np.asarray(ex["displacement"]["dir"], float)).tolist()
        if obj and isinstance(ex.get("pads"), dict):
            R, t = self._displacement_of(obj, pl, a)
            for spec in ex["pads"].values():
                if isinstance(spec, dict) and "point" in spec:
                    spec["point"] = (R @ np.asarray(spec["point"], float) + t).tolist()
                    spec["normal"] = (R @ np.asarray(spec["normal"], float)).tolist()
        return ex

    @staticmethod
    def _retimed_expect(ex, src, a0, a1):
        """An expectation's frame references (relative to its subgoal's first frame a0, or absolute) moved to a
        retimed episode whose window starts at a1 (src: the source frame of every retimed frame)."""
        from . import retime as RT

        def new_abs(t):
            return RT.window(src, t, t)[0]
        ex = copy.deepcopy(ex)
        if isinstance(ex.get("hands"), dict):
            ex["hands"] = {k: Replay._retimed_expect(v, src, a0, a1) for k, v in ex["hands"].items()}
        for k in ("pads_at_rel", "slip_from_rel"):
            if ex.get(k) is not None:
                ex[k] = int(new_abs(a0 + int(ex[k])) - a1)
        for k in ("pads_at", "slip_from"):
            if isinstance(ex.get(k), (int, np.integer)):
                ex[k] = int(new_abs(int(ex[k])))
        st = ex.get("seated")
        if isinstance(st, dict) and st.get("frames_rel"):
            ex["seated"] = dict(st, frames_rel=[int(new_abs(a0 + int(f)) - a1) for f in st["frames_rel"]])
        return ex

    def _judge_subgoal(self, sg, m, Qr, diag, pl, checks, rows):
        a, b = sg["frames"]
        oa, ob = sg.get("orig_frames", sg["frames"])       # the original's frames: where the plan's targets are read
        ex = self._expect_for(sg["expect"], pl, oa, ob)
        if sg.get("src") is not None:
            ex = self._retimed_expect(ex, sg["src"], oa, a)
        try:
            ev = self.O.evaluate(m, Qr, self.ep.fps, list(range(a, b + 1)), ex, diagnostics=diag, parts=self.parts)
        except Exception as e:                         # an expectation this scene cannot measure
            rows.append({"subgoal": sg["name"], "ok": False, "failed": ["evaluation"], "error": f"{type(e).__name__}: {e}"})
            checks[f"{sg['name']}: evaluation"] = False
            return
        for k, v in ev["checks"].items():
            checks[f"{sg['name']}: {k}"] = bool(v)
        meas = {k: v for k, v in ev["measured"].items() if isinstance(v, (int, float, str, bool))}
        rows.append({"subgoal": sg["name"], "frames": [a, b], "ok": ev["ok"], "failed": ev["failed"], "measured": meas})

    def _names_cord(self, sg):
        """Whether a subgoal's expectation names a cord (judged on the Isaac mirror, not on the rigid replay)."""
        if self.deformables is None or not sg.get("expect"):
            return False
        import re
        text = json.dumps(sg["expect"], default=str)
        return any(re.search(r'(?<![A-Za-z0-9_])' + re.escape(n) + r'(?:B_|"|\b)', text) for n in self.cord_names)

    def _deformables_run(self, Qr, pl, labels, src, original):
        """The cords for a replayed robot trajectory: the Stage 3 Isaac recording for the original layout when the replay
        reproduced the recorded trajectory bit for bit, else a new Isaac Sim episode with the cords moved by the plan."""
        from . import deformables as DF
        if original:
            r = DF.stage3_recording(self.deformables, self.cfg.path("trajectory"), Qr, self.ep.Q)
            if r is not None:
                return r
        D = {n: (pl["D"][n] if pl is not None and n in pl.get("D", {}) else IDENT) for n in self.cord_names}
        sub = self.last_execution.get("subgoal") if self.last_execution else None
        r = DF.simulate(self.deformables, self.last_scene_path, Qr, self.ep.fps, labels, D=D, src=src,
                        python=self.isaac_python, subgoal=sub)
        self._isaac_dirs.append(r["dir"])
        return r

    def run(self, Q, Qref, overrides=None, pl=None, judged_out=None, subgoals=None, segments=None, labels=None,
            releases=None, initial_state=None, src=None):
        """Replay a plan Q (commands Qref) and judge it: {check name: bool}, per-subgoal rows, end errors.

        Each subgoal is judged as soon as its last frame is simulated; with judged_out (the checks not to judge),
        the replay stops at the first judged check that fails: the layout is rejected either way. subgoals,
        segments, labels and releases default to the original episode's. With deformables, the cords are then run in
        Isaac Sim (src: the source frame of every frame of a retimed episode, for the grasp schedule), and the Stage 1
        goals and the expectations that name a cord are judged on the Isaac mirror."""
        t0 = time.time()
        checks, rows = {}, []
        subgoals = self.subgoals if subgoals is None else subgoals
        pending = [sg for sg in subgoals if sg["expect"]]
        on_cords = [sg for sg in pending if self._names_cord(sg)]
        pending = [sg for sg in pending if sg not in on_cords]
        self.last_deformables = None

        def after(m, Qr, diag, last):
            while pending and pending[0]["frames"][1] <= last:
                self._judge_subgoal(pending.pop(0), m, Qr, diag, pl, checks, rows)
            return judged_out is not None and any(not v for v in checks.values())

        m, Qr, diag, unstable, stopped = self.replay(Q, Qref, overrides, after, segments=segments, releases=releases,
                                                   initial_state=initial_state)
        checks["simulation stable"] = unstable is None
        checks["finite replay state"] = bool(np.isfinite(Qr).all())
        checks["subgoal expectations present"] = bool(subgoals) and all(bool(sg.get("expect")) for sg in subgoals)
        checks["Stage 1 goals present"] = self.goal is not None
        checks["MuJoCo version matches Stage 3"] = not self.mujoco_version["stage3"] or str(self.mujoco_version["stage3"]) == mujoco.__version__
        audits = {}
        if stopped is None:
            after(m, Qr, diag, len(Q) - 1)
            # every randomized free object ends where the adapted plan puts it
            data, data_r = mujoco.MjData(m), mujoco.MjData(m)
            data.qpos[:], data_r.qpos[:] = Q[-1], Qr[-1]
            mujoco.mj_kinematics(m, data)
            mujoco.mj_kinematics(m, data_r)
            ends = {}
            for name, o in self.ep.objects.items():
                if o["qadr"] is None:
                    continue
                err = float(np.linalg.norm(data_r.xpos[o["bid"]] - data.xpos[o["bid"]]))
                ends[name] = round(err, 4)
                checks[f"{name}: ends where the plan puts it"] = err <= self.end_tol
            if self.goal is not None and self.deformables is None:
                checks.update(self._goal_checks(m, Qr, self.ep.labels if labels is None else labels))
            audits = self._physical_audits(overrides)
            checks.update({name: bool(result.get("ok")) for name, result in audits.items()})
            if self.deformables is not None:
                lab = self.ep.labels if labels is None else labels
                if any(not v for v in checks.values()) and pl is not None:
                    # the layout is rejected already: no Isaac episode is spent on it
                    checks["deformables: Isaac Sim episode"] = False
                    for sg in on_cords:
                        checks[f"{sg['name']}: evaluation"] = False
                else:
                    dr = self._deformables_run(Qr, pl, lab, src, original=pl is None and np.array_equal(Q, self.ep.Q))
                    self.last_deformables = dr
                    checks["deformables: Isaac Sim episode"] = bool(dr["ok"])
                    if dr.get("scene_mirror"):
                        from . import deformables as DF
                        mm, Qm, _lab = DF.mirror(dr)
                        for sg in on_cords:
                            self._judge_subgoal(sg, mm, Qm, diag, pl, checks, rows)
                        if self.goal is not None:
                            checks.update(self._goal_checks(mm, Qm, lab))
                    else:
                        for sg in on_cords:
                            checks[f"{sg['name']}: evaluation"] = False
                        if self.goal is not None:
                            checks["Stage 1 goals: evaluation"] = False
                    audits["deformables: Isaac Sim episode"] = {"ok": bool(dr["ok"]), "issues": list(dr.get("issues") or []),
                                                                "source": dr.get("source")}
        else:
            ends = {}
        return {"checks": checks, "subgoals": rows, "end_error_m": ends, "unstable": unstable, "physical_audits": audits,
                "stopped_at_frame": stopped, "frames_simulated": len(Q) if stopped is None else stopped + 1,
                "max_hand_pen_mm": round(1000 * float(diag["hand_pen_m"].max()), 2),
                "max_object_pen_mm": round(1000 * float(diag["object_pen_m"].max()), 2),
                "seconds": round(time.time() - t0, 1)}, Qr

    def _physical_audits(self, overrides):
        """Apply the release collision gates to every integrated state.

        Kinematic warp checks help reject proposals early; their relative
        baseline tolerances do not authorize a publishable physical episode.
        """
        from tool_library.verifiers.physics_recording_check import physics_recording_check
        from tool_library.verifiers.robot_self_collision_check import robot_self_collision_check
        from tool_library.verifiers.object_visual_penetration_audit import object_visual_penetration_audit
        from tool_library.verifiers.swept_collision_audit import swept_collision_audit
        from tool_library.verifiers.arm_reach_check import arm_reach_check
        with tempfile.TemporaryDirectory(prefix="stage4_physical_audit_") as directory:
            directory = Path(directory)
            scene = self.last_scene_path
            trajectory = directory / "trajectory.npz"
            arrays = dict({k: v for k, v in self.last_execution.items() if not k.startswith("_")},
                          fps=self.ep.fps, joint_names=np.array(self.ep.joint_names()),
                          labels=np.array(["both: replay"] * len(self.last_execution["qpos"])))
            np.savez_compressed(trajectory, **arrays)
            reports = {}
            for name, fn in (("complete dynamics recording", physics_recording_check),
                             ("physical arm speed and mount", arm_reach_check),
                             ("absolute robot collision", robot_self_collision_check),
                             ("absolute visible object collision", object_visual_penetration_audit),
                             ("swept collision", swept_collision_audit)):
                try:
                    kwargs = {"scene_recon_json": self.scene_recon} if name == "complete dynamics recording" else {}
                    reports[name] = fn(str(scene), str(trajectory), **kwargs)
                except Exception as exc:
                    reports[name] = {"ok": False, "issues": [f"{type(exc).__name__}: {exc}"]}
            return reports

    @staticmethod
    def _goal_context(cfg, man):
        """(scene_semantics, objects, manifest) when the hand-off carries Stage 1's goals, else None."""
        g = cfg.get("goal") or {}
        sem_p, obj_p = cfg.path_of(g.get("semantics")), cfg.path_of(g.get("objects"))
        if not (sem_p and obj_p and man and Path(sem_p).exists() and Path(obj_p).exists() and Path(man).exists()):
            return None
        return (json.loads(Path(sem_p).read_text()), json.loads(Path(obj_p).read_text()), json.loads(Path(man).read_text()))

    def _goal_checks(self, m, Qr, labels):
        """Recheck every Stage 1 goal predicate on actual physics states.

        Absolute world targets remain absolute. A layout that violates one is
        rejected; adapting an object's pose never authorizes dropping a goal.
        """
        from tool_library.common import task_gates as G
        sem, objs, man = self.goal
        try:
            ep = G.Episode(m, qpos=Qr, fps=self.ep.fps, labels=list(labels))
            rep = G.stage1_goal_report(ep, sem, man, objs, stride=2)
        except Exception as e:                         # a goal this replay cannot measure judges nothing
            self.log(f"[physics] Stage 1 goals not judged: {type(e).__name__}: {e}"[:300])
            return {"Stage 1 goals: evaluation": False}
        out = {}
        for row in rep.get("subgoals", []):
            if row.get("judged") is False:
                out[f"Stage 1 goal {row['index']}"] = False
                continue
            preds = row.get("predicates", [])
            out[f"Stage 1 goal {row['index']}"] = bool(preds) and all(p.get("ok") for p in preds) and not row.get("errors")
        out["Stage 1 goals complete"] = bool(out) and bool(rep.get("ok"))
        return out

    def original(self):
        """Replay the verified layout once: the baseline every layout is judged against."""
        if self.baseline is None:
            rep, self.baseline_qpos = self.run(self.ep.Q, self.Qref)
            self.baseline_execution = self.last_execution
            self.baseline_deformables = self.last_deformables
            rep["failed"] = sorted(k for k, v in rep["checks"].items() if not v)
            rep["not_judged"] = []
            rep["ok"] = not rep["failed"]
            self.baseline = rep
            n = len(rep["checks"])
            self.log(f"[physics] original layout replayed in {rep['seconds']}s: {n - len(rep['failed'])}/{n} "
                     f"checks hold" + (f"; rejected: {rep['failed'][:6]}" if rep["failed"] else ""))
        return self.baseline

    def _judged(self, rep):
        base = self.original()
        rep["failed"] = sorted(k for k, v in rep["checks"].items() if not v)
        rep["failed"].extend(f"original replay: {k}" for k in base["failed"])
        rep["not_judged"] = []
        rep["measured_only"] = []
        rep["ok"] = not rep["failed"]
        return rep

    def check(self, layout, Q, overrides, timing=None):
        """Replay one kinematically accepted layout; ok when every check that holds for the original holds.
        timing {src, warp}: Q is the warp `warp` retimed (src: the source frame of every frame of Q); the commands,
        the subgoal windows and the recorded releases are retimed with it, the targets still read at the original
        frames."""
        base = self.original()
        ep = self.ep
        pl = ep.plan(layout)
        if timing is None:
            Qref, ref_err = ep.adapt_reference(self.Qref, Q)
            rep, Qr = self.run(Q, Qref, overrides, pl, judged_out=set(base["not_judged"]))
        else:
            from . import retime as RT
            src = np.asarray(timing["src"], float)
            # the commands adapted to the warp at the original timing (timing["warp"]), then retimed like the warp
            Qref_w, ref_err = ep.adapt_reference(self.Qref, timing["warp"])
            Qref = RT.resample(Qref_w, src, RT.quaternion_slices(ep.model))
            segs = [(*RT.window(src, a, b), lim) for a, b, lim in self.segments]
            subs = [dict(sg, frames=list(RT.window(src, *sg["frames"])), orig_frames=list(sg["frames"]), src=src)
                    for sg in self.subgoals]
            rels = [dict(r, frame=RT.window(src, int(r["frame"]), int(r["frame"]))[0]) for r in self.releases]
            rep, Qr = self.run(Q, Qref, overrides, pl, judged_out=set(base["not_judged"]), subgoals=subs,
                               segments=segs, labels=RT.resample_labels(ep.labels, src), releases=rels, src=src)
            rep["retimed"] = {"frames": len(Q), "original_frames": ep.T}
        rep = self._judged(rep)
        rep["reference_ik_m"] = round(float(ref_err), 6)
        return rep, Qr

    def check_episode(self, Q, Qref, overrides, subgoals, labels, releases=(), initial_state=None):
        """Replay a re-executed episode (its own commands Qref, subgoal windows and expectations, labels, recorded
        releases) and judge it like a layout: every check that holds for the original must hold, and every
        randomized object must end where the episode put it."""
        base = self.original()
        segs = self._segments(subgoals, len(Q))
        rep, Qr = self.run(Q, Qref, overrides, None, judged_out=set(base["not_judged"]), subgoals=subgoals,
                           segments=segs, labels=labels, releases=list(releases), initial_state=initial_state)
        return self._judged(rep), Qr
