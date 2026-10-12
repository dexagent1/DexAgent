"""Re-execution of the Stage 3 plan on a drawn layout (the randomize step's fallback).

Unavailable on rigid runtime scenes, which are every Stage 4 input in this release: make() returns (None, why) for
them and the randomize step keeps the warp only. The code below drives the floating-wrist proposal scenes of earlier
releases.

The warp moves the verified trajectory rigidly with its objects and replays
the moved servo commands open-loop. It cannot re-plan: a longer reach, a
collision the moved arm now makes, a grip or a press that lands a millimetre
off and slips. Stage 3 hands off its executable plan instead (plan.json,
written by write_stage4_config: every executed plan entry with its hand
poses, each pose with the object pose it was solved against, and the run-time
rules), and for a layout the warp rejects the Stage 3 executor
(tool_library execute_subgoal_plan) runs that plan again on the randomized
scene:

  - the scene starts with every object at its drawn place (free objects
    through their joints, fixed bases through their body pose, a cable as one
    rigid piece), and settles before the first subgoal as in Stage 3;
  - every hand pose moves with the object it was solved against, every world
    quantity of an entry (a carry's target and orientation, a push direction,
    a pivot, an offset) moves the way the warp moves the hand acting in that
    entry, so targets on or in another object follow that object;
  - the executor plans each subgoal from the state the previous one really
    left, rolls it out in contact physics and checks its expectations: the
    result is closed-loop and physically valid by construction.

The episode it writes must then pass the same Stage 4 checks as a warped one
(joint speed, continuity, penetration, the arms holding their hands) and the
physics replay. Robot start jitter is not applied to a re-executed draw: it
starts from the home pose as Stage 3 did.

Runs in this interpreter when it can import the tool library's executor, else
through reexecute.python (an interpreter that can), like run_dexagent.py
launches Stage 4 in its own environment.
"""
from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from . import kinematics as K

IDENT = (np.eye(3), np.zeros(3))
WORKER_TIMEOUT_S = 6 * 3600


def _repo_root():
    return Path(__file__).resolve().parents[2]


def make(ep, cfg, log=print, physics=None, workdir=None):
    """(Reexecutor, None) when the task folder hands off a plan and an executor can run it, else (None, why)."""
    from tool_library.common.rigid_robot import is_rigid
    if is_rigid(ep.model):
        return None, "plan re-execution on rigid runtime scenes is unavailable; layouts use arm IK and verified servo replay"
    rx = cfg.get("reexecute") or {}
    p = cfg.path_of(rx.get("plan"))
    if p is None:
        return None, ("no plan handed off (reexecute.plan); regenerate the task folder with write_stage4_config "
                      "from the Stage 3 workspace (its trajectory_manifest.json and execution folder)")
    if not Path(p).exists():
        return None, f"reexecute.plan {p} does not exist"
    plan = json.loads(Path(p).read_text())
    if not plan.get("subgoals"):
        return None, f"{p} has no plan entries"
    python = rx.get("python")
    if not python:
        try:
            if str(_repo_root()) not in sys.path:
                sys.path.insert(0, str(_repo_root()))
            from tool_library.skills.execute_subgoal_plan import execute_subgoal_plan  # noqa: F401
        except Exception as e:                       # the executor's packages are missing here
            return None, (f"the Stage 3 executor does not import in this interpreter ({type(e).__name__}: {e}); "
                          "set reexecute.python to an interpreter that can")
    return Reexecutor(ep, cfg, plan, log, physics, workdir or (cfg.out / "reexecute")), None


# ----------------------------------------------------------------------------- moving a plan with a layout
def _pt(D, p):
    return (D[0] @ np.asarray(p, float) + D[1]).tolist()


def _vec(D, v):
    return (D[0] @ np.asarray(v, float)).tolist()


def _euler_xyz(R, near=None):
    from scipy.spatial.transform import Rotation as Rot
    e = Rot.from_matrix(R).as_euler("XYZ")
    if near is not None:                             # the triple nearest the old one (the wrist joints follow it)
        alt = np.array([e[0] + np.pi, np.pi - e[1], e[2] + np.pi])
        cands = [e, alt]
        cands = [c + 2 * np.pi * np.round((np.asarray(near) - c) / (2 * np.pi)) for c in cands]
        e = min(cands, key=lambda c: float(np.linalg.norm(c - np.asarray(near))))
    return e


def move_pose(D, pose):
    """A hand pose (pos, euler XYZ rad, approach, contact points and normals, the object pose it was solved against)
    moved rigidly by D."""
    if not isinstance(pose, dict) or is_ident(D):
        return pose
    from scipy.spatial.transform import Rotation as Rot
    out = copy.deepcopy(pose)
    if out.get("pos") is not None:
        out["pos"] = _pt(D, out["pos"])
    if out.get("euler") is not None:
        R = Rot.from_euler("XYZ", np.asarray(out["euler"], float)).as_matrix()
        out["euler"] = _euler_xyz(D[0] @ R, near=out["euler"]).tolist()
    for k in ("approach",):
        if out.get(k) is not None:
            out[k] = _vec(D, out[k])
    if isinstance(out.get("approach_waypoints"), list):
        out["approach_waypoints"] = [_vec(D, w) for w in out["approach_waypoints"]]
    if isinstance(out.get("points"), dict):
        out["points"] = {f: _pt(D, v) for f, v in out["points"].items()}
    if isinstance(out.get("normals"), dict):
        out["normals"] = {f: _vec(D, v) for f, v in out["normals"].items()}
    a = out.get("anchor")
    if isinstance(a, dict) and a.get("pos") is not None and a.get("quat_wxyz") is not None:
        a = dict(a)
        a["pos"] = _pt(D, a["pos"])
        a["quat_wxyz"] = K.mat2quat(D[0] @ K.quat2mat(a["quat_wxyz"])).tolist()
        out["anchor"] = a
    return out


def is_ident(D):
    return np.allclose(D[0], np.eye(3), atol=1e-12) and np.allclose(D[1], 0.0, atol=1e-12)


class Reexecutor:
    """Runs the handed-off Stage 3 plan on drawn layouts."""

    def __init__(self, ep, cfg, plan, log, physics, workdir):
        self.ep, self.cfg, self.plan, self.log, self.phys = ep, cfg, plan, log, physics
        self.workdir = Path(workdir)
        rx = cfg.get("reexecute") or {}
        self.python = rx.get("python")
        self.keep = bool(rx.get("keep_files", False))
        self.max_attempts = int(rx.get("max_attempts", 6))
        self.settle_s = rx.get("settle_s")
        self.entries = [e for e in plan["subgoals"] if isinstance(e, dict)]
        self.rules = list(plan.get("rules") if plan.get("rules") is not None else (cfg.get("physics") or {}).get("rules") or [])
        self._weld0 = None

    # -- the displacement of a body / a hand at a frame of the original episode ---------------------------------
    def body_D(self, pl, body, t):
        """Where the layout moves a body at frame t of the original episode: through its randomized ancestor (a
        free object as it moves then, a fixed base by its static displacement)."""
        m = self.ep.model
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, str(body)) if body else -1
        t = int(np.clip(t, 0, self.ep.T - 1))
        while b > 0:
            name = m.body(b).name
            if name in self.ep.objects:
                if self.ep.objects[name]["qadr"] is None:
                    return pl["D"][name]
                R, tt = pl["obj"][name]
                return R[t], tt[t]
            b = int(m.body_parentid[b])
        return IDENT

    def side_D(self, pl, side, t):
        if side not in pl["side"]:
            return IDENT
        return pl["side"][side][int(np.clip(t, 0, self.ep.T - 1))]

    def move_entry(self, e, pl, side=None):
        """One plan entry moved with the layout (see the module docstring)."""
        e = copy.deepcopy(e)
        fr = e.pop("stage3_frames", None)
        a, b = (int(fr[0]), int(fr[1])) if fr else (0, self.ep.T - 1)
        side = e.get("side") or side
        if e.get("motion") == "both" and isinstance(e.get("hands"), dict):
            for s_, h in e["hands"].items():
                e["hands"][s_] = self.move_entry(dict(h, stage3_frames=[a, b]), pl, side=s_)
            return e
        obj = e.get("object")
        D_obj = self.body_D(pl, obj, a) if obj else self.side_D(pl, side, a)     # what the entry acts on, at its start
        D_end = self.side_D(pl, side, b)                                          # where its hand ends up

        def pose_of(p):
            if not isinstance(p, dict):
                return p
            anc = p.get("anchor") or {}
            return move_pose(self.body_D(pl, anc.get("body"), a) if anc.get("body") else D_obj, p)

        poses = []
        for k in ("pose", "pose_candidates", "regrasp_pose"):
            v = e.get(k)
            if isinstance(v, list):
                e[k] = [pose_of(p) for p in v]
                poses += e[k]
            elif isinstance(v, dict):
                e[k] = pose_of(v)
                poses.append(e[k])
        if poses and all(isinstance(p, dict) and p.get("anchor") for p in poses):
            # the hand scene shows the objects where the poses were solved, not where this layout puts them: with every
            # pose anchored it is not needed, and the executor's reach re-solve then reads the pads off the moved scene
            e.pop("hand_scene", None)
        # where the moved part goes: with the hand that takes it there
        if e.get("target_pos") is not None:
            e["target_pos"] = _pt(D_end, e["target_pos"])
        if e.get("target_euler_deg") is not None:
            from scipy.spatial.transform import Rotation as Rot
            R = Rot.from_euler("XYZ", np.radians(np.asarray(e["target_euler_deg"], float))).as_matrix()
            e["target_euler_deg"] = np.degrees(_euler_xyz(D_end[0] @ R, near=np.radians(e["target_euler_deg"]))).tolist()
        if isinstance(e.get("turn"), dict) and e["turn"].get("axis") is not None:
            e["turn"] = dict(e["turn"], axis=_vec(D_end, e["turn"]["axis"]))
        for k in ("offset", "retreat_dir"):
            if e.get(k) is not None and len(e[k]) == 3:
                e[k] = _vec(D_end, e[k])
        ex = e.get("expect")
        if isinstance(ex, dict):                     # an expectation the plan overrides moves like the entry
            ex = copy.deepcopy(ex)
            if ex.get("target_pos") is not None:
                ex["target_pos"] = _pt(D_end, ex["target_pos"])
            if isinstance(ex.get("displacement"), dict) and ex["displacement"].get("dir") is not None:
                ex["displacement"]["dir"] = _vec(D_obj, ex["displacement"]["dir"])
            if isinstance(ex.get("pads"), dict):
                for spec in ex["pads"].values():
                    if isinstance(spec, dict) and spec.get("point") is not None:
                        spec["point"] = _pt(D_obj, spec["point"])
                        if spec.get("normal") is not None:
                            spec["normal"] = _vec(D_obj, spec["normal"])
            e["expect"] = ex
        # how the part is acted on: with the part
        for k in ("direction", "axis", "lift_dir"):
            if e.get(k) is not None and len(e[k]) == 3:
                e[k] = _vec(D_obj, e[k])
        if isinstance(e.get("lift_path"), list):
            e["lift_path"] = [_vec(D_obj, w) for w in e["lift_path"]]
        for k in ("pivot", "follow_point", "cut_point"):
            if e.get(k) is not None and len(e[k]) == 3:
                e[k] = _pt(D_obj, e[k])
        if isinstance(e.get("via"), list):
            e["via"] = [_pt(D_obj, w) for w in e["via"]]
        return e

    def plan_for(self, pl):
        return {"subgoals": [self.move_entry(e, pl) for e in self.entries], "rules": self.rules}

    # -- the randomized scene ------------------------------------------------------------------------------------
    def scene_for(self, layout, pl, out_dir):
        """The task scene with every object at its drawn start (written to out_dir/scene.xml, asset paths
        absolute). Free objects move in the model's start pose and in every keyframe; a fixed base moves its body;
        a cable moves as one rigid piece (its root joint; the chain's own joints are relative)."""
        ep, m = self.ep, self.ep.model
        src = Path(self.cfg.path("scene")).resolve()
        spec = mujoco.MjSpec.from_file(str(src))
        for attr in ("meshdir", "texturedir"):
            cur = getattr(spec, attr) or ""
            setattr(spec, attr, str((src.parent / cur).resolve()) if not Path(cur).is_absolute() else cur)
        D = pl["D"]
        moved = []
        for name, o in ep.objects.items():
            d = D.get(name, IDENT)
            if is_ident(d):
                continue
            moved.append(name)
            body = spec.body(name)
            p, R = np.asarray(m.body_pos[o["bid"]], float), K.quat2mat(m.body_quat[o["bid"]])
            p2, R2 = d[0] @ p + d[1], d[0] @ R
            body.pos, body.quat = p2.tolist(), K.mat2quat(R2).tolist()
            if o["qadr"] is not None:
                adr = o["qadr"]
                for key in spec.keys:
                    q = np.array(key.qpos, float)
                    if len(q) < adr + 7:             # a short keyframe takes the rest from the body pose
                        continue
                    pk, Rk = q[adr:adr + 3], K.quat2mat(q[adr + 3:adr + 7])
                    q[adr:adr + 3] = d[0] @ pk + d[1]
                    q[adr + 3:adr + 7] = K.mat2quat(d[0] @ Rk)
                    key.qpos = q.tolist()
        out_dir.mkdir(parents=True, exist_ok=True)
        model = spec.compile()
        assert (model.nq, model.nbody, model.ngeom) == (m.nq, m.nbody, m.ngeom), "the randomized scene changed shape"
        path = out_dir / "scene.xml"
        path.write_text(spec.to_xml())
        return path, moved

    # -- the run ---------------------------------------------------------------------------------------------------
    def _execute(self, scene, plan, out_dir, settle_s):
        args = {"scene": str(scene), "plan": plan, "out_dir": str(out_dir), "fps": float(self.ep.fps),
                "settle_s": float(settle_s), "max_attempts": self.max_attempts}
        if not self.python:
            if str(_repo_root()) not in sys.path:
                sys.path.insert(0, str(_repo_root()))
            from tool_library.skills.execute_subgoal_plan import execute_subgoal_plan
            return execute_subgoal_plan(args["scene"], args["plan"], args["out_dir"], fps=args["fps"],
                                        settle_s=args["settle_s"], max_attempts=args["max_attempts"], stop_on_fail=False)
        (out_dir / "reexecute_args.json").write_text(json.dumps(args, default=float))
        r = subprocess.run([str(self.python), "-m", "stage4.reexecute", str(out_dir / "reexecute_args.json")],
                           cwd=str(Path(__file__).resolve().parents[1]), capture_output=True, text=True,
                           timeout=WORKER_TIMEOUT_S)
        res = out_dir / "reexecute_summary.json"
        if r.returncode != 0 or not res.exists():
            raise RuntimeError(f"the re-execution worker failed ({r.returncode}): {(r.stderr or r.stdout)[-600:]}")
        return json.loads(res.read_text())

    def _reach(self, Q):
        """The arms holding their hands on the re-executed episode: the weld residual of every frame, no worse than
        the original episode's (or Stage 4's reach tolerance, whichever is larger)."""
        try:
            if str(_repo_root()) not in sys.path:
                sys.path.insert(0, str(_repo_root()))
            from tool_library.common import armhand as AH
        except Exception as e:
            return {"ok": True, "note": f"not measured ({type(e).__name__})"}
        ck = self.cfg["checks"]
        if self._weld0 is None:
            self._weld0 = {s: (float(pe.max()), float(re.max())) for s, (pe, re, _src) in AH.weld_residuals(self.ep.model, self.ep.Q).items()}
        rep, ok = {}, True
        for s, (pe, re, _src) in AH.weld_residuals(self.ep.model, Q).items():
            p0, r0 = self._weld0.get(s, (0.0, 0.0))
            lim_p, lim_r = max(1000 * ck["ik_pos_tol_m"], p0), max(ck["ik_rot_tol_deg"], r0)
            rep[s] = {"pos_m": round(float(pe.max()) / 1000, 6), "rot_deg": round(float(re.max()), 4), "frame": int(np.argmax(pe)),
                      "limit_mm": round(lim_p, 3), "limit_deg": round(lim_r, 3)}
            ok &= float(pe.max()) <= lim_p + 1e-9 and float(re.max()) <= lim_r + 1e-9
        rep["ok"] = bool(ok)
        return rep

    def run(self, layout, pl, tag):
        """Re-execute the plan on one layout and judge the episode. Returns {ok, row, Q, labels, overrides, report,
        replay, physics}."""
        from . import physics as PH
        ep = self.ep
        t0 = time.time()
        out = self.workdir / tag
        shutil.rmtree(out, ignore_errors=True)
        row = {"ok": False}
        res = {"ok": False, "row": row}
        try:
            scene, moved = self.scene_for(layout, pl, out)
            deform = any(ep.objects[n]["deformable"] for n in moved)
            settle = float(self.settle_s) if self.settle_s is not None else (1.0 if deform else 0.3)
            plan = self.plan_for(pl)
            (out / "plan.json").write_text(json.dumps(plan, indent=1, default=float))
            self.log(f"[reexecute] {tag}: running the Stage 3 plan ({len(plan['subgoals'])} entries) on the layout "
                     f"{json.dumps(layout['objects'])}")
            summary = self._execute(scene, plan, out, settle)
        except Exception as e:                        # a plan that cannot run on this layout is a rejection
            row.update(why=f"{type(e).__name__}: {e}"[:400], seconds=round(time.time() - t0, 1))
            self._cleanup(out)
            return res
        row["executor_seconds"] = round(time.time() - t0, 1)
        subs = summary.get("subgoals") or []
        executed = bool(subs) and len(subs) == int(summary.get("n_subgoals") or len(subs)) and all(r.get("frames") for r in subs)
        if not summary.get("ok") or not executed or any(not s.get("ok") for s in subs):
            ff = summary.get("first_failure") or {}
            row.update(why=f"subgoal {ff.get('index')} {ff.get('motion')} failed {ff.get('failed')}",
                       first_failure={k: ff.get(k) for k in ("index", "motion", "side", "object", "failed", "error")},
                       seconds=round(time.time() - t0, 1))
            self._cleanup(out)
            return res
        z = np.load(out / "trajectory.npz", allow_pickle=False)
        Q = np.asarray(z["qpos"], float)
        labels = [str(x) for x in z["labels"]]
        if str(np.asarray(z.get("robot_schema", "")).item()) != "dexagent.rigid_robot.v1" or "reference" not in z:
            raise ValueError("re-execution did not produce a rigid runtime trajectory and commands")
        Qref = np.asarray(z["reference"], float)
        initial_state = {key: z[key] for key in ("initial_qpos", "initial_qvel", "eq_active_start", "initial_state", "state_spec", "initial_act") if key in z}
        outcomes = json.loads((out / "outcomes.json").read_text())
        # the episode's own subgoal windows and expectations, named as the original manifest names them
        entries, man, rels = list(plan["subgoals"]), [], []
        for i, r in enumerate(outcomes):
            if not r.get("frames"):
                continue
            if len(entries) == len(outcomes):
                pe = entries[i]
            else:                                     # entries expanded by the executor (screw bites): match them
                pe = next((x for x in entries if (x.get("index"), x.get("motion")) == (r.get("index"), r.get("motion"))), {})
            limits = {k: pe[k] for k in ("finger_force_nm", "wrist_force_n", "wrist_torque_nm", "step", "motion") if pe.get(k) is not None}
            man.append({"index": r.get("index"), "step": r.get("step"), "motion": r.get("motion"), "frames": r["frames"],
                        "expect": r.get("expect"), "plan": limits})
            for e in r.get("released") or []:
                if isinstance(e, dict) and e.get("equality") is not None and e.get("frame") is not None:
                    rels.append({"equality": str(e["equality"]), "frame": int(r["frames"][0]) + int(e["frame"])})
        (out / "manifest.json").write_text(json.dumps({"subgoals": man}, indent=1, default=float))
        subgoals = PH.load_subgoals(out / "manifest.json", labels)
        _model, cmodel, ov = ep.layout_models(layout, pl["D"])
        rep = ep.check_trajectory(Q, cmodel, reach=self._reach(Q))
        rep["reexecuted"] = {"frames": len(Q), "original_frames": ep.T, "executor_seconds": row["executor_seconds"],
                             "settle_s": settle, "start_jitter": "not applied (re-executed from the home pose)",
                             "variants": [s_.get("variant") for s_ in summary.get("subgoals", []) if s_.get("variant") not in (None, "as planned")]}
        row.update(frames=len(Q), failed=list(rep["failed"]))
        prep, replay = None, None
        if rep["ok"] and self.phys is not None:
            t1 = time.time()
            prep, replay = self.phys.check_episode(Q, Qref, ov, subgoals, labels, releases=rels, initial_state=initial_state)
            row["physics_seconds"] = round(time.time() - t1, 1)
            from .randomize import physics_summary
            rep["physics"] = physics_summary(prep)
            if prep["ok"]:
                executed_checks = ep.check_trajectory(replay, cmodel, reach=self._reach(replay))
                rep["executed_trajectory"] = executed_checks
                if not executed_checks["ok"]:
                    prep["ok"] = False
                    prep["failed"].extend(f"executed trajectory: {k}" for k in executed_checks["failed"])
                    rep["physics"] = physics_summary(prep)
            if not prep["ok"]:
                rep["ok"] = False
                rep["failed"].append("physics")
                row["physics_failed"] = prep["failed"][:12]
        if not rep["ok"]:
            row["why"] = "Stage 4 checks failed on the re-executed episode: " + ", ".join(rep["failed"])
        row.update(ok=rep["ok"], seconds=round(time.time() - t0, 1))
        self._cleanup(out)
        res.update(ok=rep["ok"], Q=Q, labels=labels, overrides=ov, report=rep, replay=replay, physics=prep,
                   execution=self.phys.last_execution if prep is not None else None)
        return res

    def _cleanup(self, out):
        if not self.keep:
            shutil.rmtree(out, ignore_errors=True)


def _worker(args_path):
    """Run the executor in this interpreter for a Stage 4 process that cannot import it (reexecute.python)."""
    args = json.loads(Path(args_path).read_text())
    sys.path.insert(0, str(_repo_root()))
    from tool_library.skills.execute_subgoal_plan import execute_subgoal_plan
    s = execute_subgoal_plan(args["scene"], args["plan"], args["out_dir"], fps=args["fps"], settle_s=args["settle_s"],
                             max_attempts=args["max_attempts"], stop_on_fail=False)
    (Path(args["out_dir"]) / "reexecute_summary.json").write_text(json.dumps(s, indent=1, default=float))


if __name__ == "__main__":
    _worker(sys.argv[1])
