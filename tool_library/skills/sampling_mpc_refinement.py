"""Sampling-based MPC refinement: predictive-sampling model-predictive control on the bench's own CPU physics refines a
tracking reference closed-loop, so the objects follow it under the executor's exact contact dynamics."""
from __future__ import annotations

TOOL = {
    "name": "sampling_mpc_refinement",
    "title": "Sampling-based MPC refinement",
    "type": "skill",
    "category": "Trajectory & Execution",
    "summary": "Stage 3 method module (the second after the agent): predictive-sampling model-predictive control on the "
               "bench's own rigid runtime (CPU MuJoCo, the arms and hands, the executor's servo physics) re-plans a "
               "residual on a tracking reference every replan_s: candidate residuals over a short horizon are rolled out in "
               "parallel from the current physics state, scored by how closely every part the reference moves follows it "
               "(position and orientation, so in-hand turns count), and the best is executed; writes the refined tracking "
               "reference and its track_reference plan.",
    "description": (
        "Predictive sampling (Howell et al., 'Predictive Sampling: Real-time Behaviour Synthesis with MuJoCo', 2022; the "
        "planner MuJoCo MPC runs) on the same simulator the executor and the gates use, so there is no gap between the "
        "model the plan is optimized in and the one it is judged in. Reference: `reference` (an earlier method's tracking "
        "reference), else the agent's best physical attempt in the workspace (tracking.reference_from_attempt). It is put "
        "on the bench as the executor will run it (tracking.BenchTracker: from the Stage 2 home state, the joint-rate "
        "ceilings, the arms coupled with the executor's settings and arm planner). The decision variable is a residual "
        "on the commands, a piecewise-linear spline with a knot every knot_s: per acting station six arm-joint offsets "
        "(which move the hand rigidly) and one grip offset (added to every flexion joint of that hand's fingers, clipped "
        "to their ranges). At every replan_s the knots inside the horizon are sampled around the current nominal "
        "(samples - 1 Gaussian perturbations plus the nominal itself), each candidate is rolled out over horizon_s from "
        "the complete integration state the previous step left (rigid_robot.servo_rollout, in `workers` processes), and "
        "scored per frame by 10 x position error (m) + 0.5 x rotation error (rad) of every free part the reference moves, "
        "2 x the error of every articulated joint it moves, hand penetration past 3 mm, hand-object force past 12 N (the "
        "gates' contact spike audit allows 15 N) and the residual's size; the best "
        "candidate's first replan_s is executed and becomes the start of the next step. The refined commands are rolled "
        "out once from the start and compared with the unrefined ones by the same cost over the whole episode; the better "
        "of the two is written (`improved` says which) as reference.npz, the track_reference plan.json and report.json. The harness executes the plan with "
        "execute_subgoal_plan, packages it and runs finish's verifiers, as for an agent's plan; the agent may call it the "
        "same way. CPU only. Limits: the residual corrects a reference whose contacts are roughly right (it does not "
        "invent a new grasp), and the work grows with the reference's length (max_minutes stops refining; the rest of "
        "the episode then keeps the reference's own commands)."
    ),
    "inputs": [
        {"name": "workspace", "type": "str", "description": "Stage 3 workspace (the agent's attempts are read from it)."},
        {"name": "scene_semantic_json", "type": "str", "description": "Stage 1 scene_semantics.json."},
        {"name": "scene_recon_json", "type": "str", "description": "Stage 2 scene_recon.json."},
        {"name": "reference", "type": "str | None", "description": "A tracking reference to refine (an earlier method's); default the agent's best attempt."},
        {"name": "scene", "type": "str | None", "description": "Bench scene; default the stations scene of the workspace, else Stage 2's."},
        {"name": "out_dir", "type": "str | None", "description": "Default <workspace>/methods/sampling_mpc_refinement."},
        {"name": "horizon_s, replan_s, knot_s", "type": "float", "description": "Planning horizon, re-planning interval and spline knot spacing. Defaults 0.4, 0.2, 0.2."},
        {"name": "samples", "type": "int", "description": "Candidates per re-plan, the nominal included. Default 16."},
        {"name": "sigma_arm_rad, sigma_grip_rad", "type": "float", "description": "Sampling noise of the arm-joint and grip knots. Defaults 0.01, 0.05."},
        {"name": "max_arm_rad, max_grip_rad", "type": "float", "description": "Bounds of the residual. Defaults 0.15, 0.4."},
        {"name": "window", "type": "list[int] | None", "description": "[first, last] command frames to refine (default all); outside it the reference's own commands run."},
        {"name": "workers", "type": "int | None", "description": "Rollout processes; default dexagent.yaml compute.workers, else the CPUs - 1 (at most 8)."},
        {"name": "max_minutes", "type": "float", "description": "Refinement time budget. Default 30."},
        {"name": "seed", "type": "int", "description": "Default 0."},
    ],
    "outputs": [
        {"name": "result", "type": "dict", "description": "ok, method, reference (tracking reference npz), plan (plan.json for execute_subgoal_plan), scene, fps, improved, cost_before / cost_after (the whole episode's cost), before / after (object tracking on the bench's runtime: mean / max / end error per part), replans, rollouts, coverage, seconds, next_step."},
    ],
    "status": "tested",
    "entrypoint": "sampling_mpc_refinement",
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "method-module", "physics", "mpc", "sampling", "in-hand"],
    "used_in": [],
}

import json
import time
from pathlib import Path

import numpy as np

from tool_library.common import tracking as TR

NAME = TOOL["name"]
W_RESIDUAL = 0.5


def _knots(T, knot_frames):
    k = list(range(0, T - 1, max(1, int(knot_frames)))) + [T - 1]
    return np.array(sorted(set(k)), int)


def _residual(U, knots, f0, f1):
    """The spline's residual on frames f0..f1 (rows)."""
    f = np.arange(f0, f1 + 1)
    return np.stack([np.interp(f, knots, U[:, j]) for j in range(U.shape[1])], axis=1)


def _bounds(sim, max_arm, max_grip):
    return np.tile(np.r_[np.full(6, max_arm), max_grip], len(sim.sides))


def _evaluate(task):
    """Worker: roll one candidate out, return its cost."""
    t0, t1, commands, state, mechanism, residual = task
    sim = TR.worker_sim()
    roll = sim.rollout(t0, t1, commands, state, mechanism)
    return sim.cost(t0, roll, residual, W_RESIDUAL)


def sampling_mpc_refinement(workspace, scene_semantic_json, scene_recon_json, reference=None, scene=None, out_dir=None, horizon_s=0.4,
                            replan_s=0.2, knot_s=0.2, samples=16, sigma_arm_rad=0.01, sigma_grip_rad=0.05, max_arm_rad=0.15,
                            max_grip_rad=0.4, window=None, workers=None, max_minutes=30.0, seed=0):
    t_all = time.time()
    ws = Path(workspace).resolve()
    sem_p, recon_p = Path(scene_semantic_json).resolve(), Path(scene_recon_json).resolve()
    sem, recon = json.loads(sem_p.read_text()), json.loads(recon_p.read_text())
    out = Path(out_dir or ws / "methods" / NAME).resolve()
    out.mkdir(parents=True, exist_ok=True)
    scene_p = TR.scene_for(ws, recon_p, scene)
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(scene_p))
    ref_in, info = TR.best_reference(ws, model, out / "reference_in.npz", given=reference, sem=sem)
    bt = TR.BenchTracker(scene_p, ref_in, out / "bench", scene_recon_json=recon_p)
    sim = bt.sim
    T, fps = sim.T, sim.fps
    H, step, kf = (max(2, int(round(x * fps))) for x in (horizon_s, replan_s, knot_s))
    knots = _knots(T, kf)
    U = np.zeros((len(knots), sim.dim))
    lim = _bounds(sim, max_arm_rad, max_grip_rad)
    sigma = np.tile(np.r_[np.full(6, sigma_arm_rad), sigma_grip_rad], len(sim.sides))
    rng = np.random.default_rng(seed)
    a, b = (int(window[0]), int(window[1])) if window else (0, T - 1)
    rep = {"ok": False, "method": NAME, "out_dir": str(out), "scene": str(scene_p), "reference_in": {"path": ref_in, **info},
           "frames": int(T), "fps": fps, "tracked": {"bodies": list(sim.bodies), "joints": list(sim.joints)},
           "arm_coupling": bt.couple_report, "settings": {"horizon_frames": H, "replan_frames": step, "knot_frames": kf, "samples": int(samples)}}
    base = sim.base
    before = sim.rollout(0, T - 1, base)
    rep["before"] = dict(sim.summary(before["qpos"]), unstable=before["unstable"])
    Qopt = base.copy()
    pool = TR.make_pool(bt.spec, TR.default_workers(workers))
    evaluate = (lambda tasks: [_evaluate(x) for x in tasks]) if pool is None else (lambda tasks: list(pool.map(_evaluate, tasks)))
    state, mech, t = None, None, 0
    n_replans = n_rollouts = 0
    stopped = None
    try:
        if a > 0:                                     # the unrefined prefix: physics up to the window's first frame
            r = sim.rollout(0, a, base[:a + 1])
            state, mech, t = r["final_state"], r["mechanism_final"], a
        while t < min(b, T - 1):
            if TR.elapsed_min(t_all) > float(max_minutes):
                stopped = f"time budget ({max_minutes} min) spent at frame {t} of {T}"
                break
            h1 = min(T - 1, t + H)
            free = np.where((knots > t) & (np.r_[-1, knots[:-1]] < h1))[0]
            nominal = U[free].copy()
            cands = [nominal] + [np.clip(nominal + rng.normal(0.0, 1.0, nominal.shape) * sigma, -lim, lim) for _ in range(int(samples) - 1)]
            tasks = []
            for c in cands:
                Uc = U.copy(); Uc[free] = c
                R = _residual(Uc, knots, t, h1)
                cmd = sim.apply(base[t:h1 + 1], R)
                cmd[0] = Qopt[t]                      # frame t's command was already sent
                tasks.append((t, h1, cmd, state, mech, R[1:]))
            costs = evaluate(tasks)
            n_replans += 1; n_rollouts += len(tasks)
            best = int(np.argmin(costs))
            U[free] = cands[best]
            t1 = min(T - 1, t + step)
            cmd = tasks[best][2][:t1 - t + 1]
            r = sim.rollout(t, t1, cmd, state, mech)
            Qopt[t + 1:t1 + 1] = cmd[1:]
            if r["unstable"]:
                stopped = f"the integration became unstable at frame {t1}"
                break
            state, mech, t = r["final_state"], r["mechanism_final"], t1
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)
    if t < T - 1:                                     # the rest of the episode: the reference's commands plus the residual reached
        R = _residual(U, knots, t, T - 1)
        Qopt[t + 1:] = sim.apply(base[t:], R)[1:]
    after = sim.rollout(0, T - 1, Qopt)
    rep["after"] = dict(sim.summary(after["qpos"]), unstable=after["unstable"])
    # the refinement is kept when the whole episode's cost (the one optimized: tracking, penetration, contact force) drops
    rep["cost_before"], rep["cost_after"] = round(sim.cost(0, before), 5), round(sim.cost(0, after), 5)
    improved = rep["cost_after"] < rep["cost_before"]
    rep.update(improved=bool(improved), replans=n_replans, rollouts=n_rollouts, stopped=stopped,
               residual_max={"arm_rad": round(float(np.abs(U[:, [i for i in range(sim.dim) if i % 7 != 6]]).max()) if U.size else 0.0, 4),
                             "grip_rad": round(float(np.abs(U[:, [i for i in range(sim.dim) if i % 7 == 6]]).max()) if U.size else 0.0, 4)})
    path = bt.export(out / "reference.npz", Qopt if improved else base, source=f"{NAME} from {bt.ref['source']}" + ("" if improved else " (unrefined: no improvement)"))
    plan = TR.write_plan(out / "plan.json", path, bt.model, sem, recon, scene_recon_json=recon_p, method=NAME)
    rep.update(ok=True, reference=path, plan=str(out / "plan.json"), n_entries=len(plan["subgoals"]), coverage=TR.coverage(path, bt.model, sem),
               next_step=f"execute_subgoal_plan(scene={str(scene_p)!r}, plan={str(out / 'plan.json')!r}, out_dir={str(out / 'execution')!r}) "
                         "then package_stage3_episode and finish")
    rep["seconds"] = round(time.time() - t_all, 1)
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep
