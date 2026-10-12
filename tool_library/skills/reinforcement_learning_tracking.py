"""Reinforcement-learning tracking: a residual policy trained with PPO on the bench's own physics makes the objects
follow a tracking reference closed-loop (in-hand turns included), and its deterministic rollout is written as a new
tracking reference the executor follows."""
from __future__ import annotations

TOOL = {
    "name": "reinforcement_learning_tracking",
    "title": "Reinforcement-learning residual tracking",
    "type": "skill",
    "category": "Trajectory & Execution",
    "summary": "Stage 3 method module (the third after the agent): a residual tracking policy is trained with proximal "
               "policy optimization on the bench's own rigid runtime (CPU MuJoCo, the executor's servo physics) to keep "
               "every part the reference moves on its reference pose (position and orientation, so in-hand rotations "
               "count), episodes starting where the reference first loses the object and moving earlier (a reset "
               "curriculum); the trained policy's deterministic rollout over the whole episode is written as a tracking "
               "reference and its track_reference plan.",
    "description": (
        "Residual reinforcement learning for demonstration tracking (as in ManipTrans, Li et al. 2025, and DexMachina, "
        "Mandi et al. 2025: a policy that corrects a kinematic reference under contact physics), with a reset "
        "curriculum from the demonstration's own states (DemoStart, Bauza et al. 2024), trained in this library's numpy "
        "PPO (common/ppo.py: Gaussian tanh-MLP policy and value function, clipped surrogate, GAE, Adam, observation "
        "normalization). Reference: `reference` (an earlier method's tracking reference), else the agent's best physical "
        "attempt (tracking.reference_from_attempt); it is put on the bench as the executor will run it "
        "(tracking.BenchTracker). Action every act_s: per acting station six arm-joint offsets and one grip offset on the "
        "reference's commands (scaled by action_scale_arm_rad / action_scale_grip_rad), interpolated linearly from the "
        "previous action across the interval. Observation: the episode phase, every tracked part's position error, "
        "rotation error (rotation vector) and velocity, every tracked joint's error, the previous action. Reward per "
        "interval: the mean over its frames of exp(-50 x position error) + 0.5 x exp(-3 x rotation error) per part (and "
        "exp(-10 x joint error) per joint), minus the action's size, hand penetration past 3 mm and hand-object force past "
        "12 N (the gates' contact spike audit allows 15 N); an episode ends with "
        "-2 when a part is more than 10 cm off its reference. Episodes of episode_s start from the reference's own "
        "integration states at action boundaries, first just before the frame where the unrefined rollout first loses "
        "a part (2 cm / 20 deg) and widening toward the start as the policy succeeds. Every eval_every iterations the "
        "deterministic policy runs the whole episode from the start; the best such rollout (by its worst mean and end "
        "error) is kept, and written only when its whole-episode cost beats the unrefined reference's (`improved`), as reference.npz, "
        "plan.json (track_reference entries) and report.json with the training curve. The harness executes the plan with "
        "execute_subgoal_plan, packages it and runs finish's verifiers, as for an agent's plan; the agent may call it the "
        "same way. CPU only, in `workers` processes. Limits: a residual policy corrects a reference whose contacts are "
        "roughly right; training time grows with the episode's length (max_minutes ends training with the best policy so far)."
    ),
    "inputs": [
        {"name": "workspace", "type": "str", "description": "Stage 3 workspace (the agent's attempts are read from it)."},
        {"name": "scene_semantic_json", "type": "str", "description": "Stage 1 scene_semantics.json."},
        {"name": "scene_recon_json", "type": "str", "description": "Stage 2 scene_recon.json."},
        {"name": "reference", "type": "str | None", "description": "A tracking reference to track (an earlier method's); default the agent's best attempt."},
        {"name": "scene", "type": "str | None", "description": "Bench scene; default the stations scene of the workspace, else Stage 2's."},
        {"name": "out_dir", "type": "str | None", "description": "Default <workspace>/methods/reinforcement_learning_tracking."},
        {"name": "act_s, episode_s", "type": "float", "description": "Control interval and training episode length. Defaults 0.1, 2.0."},
        {"name": "iterations, episodes_per_iteration", "type": "int", "description": "PPO iterations (each one batch of episodes) and episodes per batch. Defaults 300, 16."},
        {"name": "hidden, lr, gamma, lam, clip, epochs, minibatch", "type": "float | int", "description": "PPO settings. Defaults 64, 3e-4, 0.98, 0.95, 0.2, 4, 256."},
        {"name": "action_scale_arm_rad, action_scale_grip_rad", "type": "float", "description": "Residual per unit action. Defaults 0.05, 0.3."},
        {"name": "eval_every", "type": "int", "description": "Iterations between whole-episode evaluations. Default 10."},
        {"name": "workers", "type": "int | None", "description": "Rollout processes; default dexagent.yaml compute.workers, else the CPUs - 1 (at most 8)."},
        {"name": "max_minutes", "type": "float", "description": "Training time budget. Default 60."},
        {"name": "seed", "type": "int", "description": "Default 0."},
    ],
    "outputs": [
        {"name": "result", "type": "dict", "description": "ok, method, reference (tracking reference npz), plan (plan.json for execute_subgoal_plan), scene, fps, improved, before / after (object tracking on the bench's runtime: mean / max / end error per part), iterations, curve [{iteration, return, episode_frames, eval}], coverage, seconds, next_step."},
    ],
    "status": "tested",
    "entrypoint": "reinforcement_learning_tracking",
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "method-module", "physics", "reinforcement-learning", "ppo", "in-hand"],
    "used_in": [],
}

import json
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from tool_library.common import ppo as PP
from tool_library.common import tracking as TR

NAME = TOOL["name"]
LOST_M = 0.10                       # a part this far off its reference ends the episode
SLIPPING = (0.02, 20.0)             # where the unrefined rollout first loses a part (m, deg): the curriculum starts there


def _scales(sim, arm, grip):
    return np.tile(np.r_[np.full(6, arm), grip], len(sim.sides))


def _observe(sim, t, qpos, qvel, prev):
    """Phase, per tracked part: position error (dm), rotation-vector error, linear velocity; joint errors; previous action."""
    d, m = sim.data, sim.model
    d.qpos[:] = qpos; mujoco.mj_kinematics(m, d)
    t = min(int(t), sim.T - 1)
    obs = [t / max(1, sim.T - 1)]
    for k, (name, bid) in enumerate(sim.bodies.items()):
        obs += list(10.0 * (d.xpos[bid] - sim.tpos[t, k]))
        q, qt = d.xquat[bid], sim.tquat[t, k]
        rv = (Rot.from_quat(qt[[1, 2, 3, 0]]).inv() * Rot.from_quat(q[[1, 2, 3, 0]])).as_rotvec()
        obs += list(rv)
        dof = sim.body_dof[name]
        obs += list((np.asarray(qvel[dof:dof + 3]) if dof is not None else np.zeros(3)) * 2.0)
    for k, (name, adr) in enumerate(sim.joints.items()):
        obs.append(5.0 * (float(qpos[adr]) - sim.tjoint[t, k]))
    obs += list(prev)
    return np.asarray(obs, float)


def _reward(sim, t0, roll, action):
    ep, er, ej = sim.errors(t0 + 1, roll["qpos"][1:]) if len(roll["qpos"]) > 1 else sim.errors(t0, roll["qpos"])
    r = np.exp(-50.0 * ep).mean(axis=0).sum() + 0.5 * np.exp(-3.0 * er).mean(axis=0).sum()
    if ej.size:
        r += np.exp(-10.0 * ej).mean(axis=0).sum()
    n = max(1, len(sim.bodies) + len(sim.joints))
    r = (r / n - 0.01 * float(np.square(action).sum()) - 100.0 * float(np.clip(np.asarray(roll["hand_pen_m"]) - 0.003, 0, None).mean())
         - 0.02 * float(np.clip(np.asarray(roll["hand_object_force_n"]) - TR.FORCE_SOFT_N, 0, None).mean()))
    lost = bool(ep.size and ep.max() > LOST_M) or bool(roll["unstable"])
    return float(r - (2.0 if lost else 0.0)), lost


def _state_obs(sim, state, t, prev):
    """qpos, qvel at an integration state (the base's first row at rest when None) and the observation there."""
    if state is None:
        qpos, qvel = sim.base[0].copy(), np.zeros(sim.model.nv)
    else:
        mujoco.mj_setState(sim.model, sim.data, np.asarray(state, float), mujoco.mjtState.mjSTATE_INTEGRATION)
        qpos, qvel = sim.data.qpos.copy(), sim.data.qvel.copy()
    return _observe(sim, t, qpos, qvel, prev)


def _episode(task):
    """Worker: one episode of the policy from a reset state. Returns transitions (and the commands sent)."""
    params, t, state, mech, n_steps, act_frames, scales, seed, deterministic = task
    sim = TR.worker_sim()
    pol = PP.Policy(params)
    rng = np.random.default_rng(seed)
    dim = sim.dim
    prev = np.zeros(dim)
    obs = _state_obs(sim, state, t, prev)
    out = {"obs": [], "act": [], "logp": [], "rew": [], "done": [], "start": int(t), "commands": [], "qpos": [], "lost": False}
    for _ in range(int(n_steps)):
        if t >= sim.T - 1:
            break
        a, logp = pol.act(obs, rng, deterministic)
        ac = np.clip(a, -1.0, 1.0)
        t1 = min(sim.T - 1, t + int(act_frames))
        w = np.linspace(0.0, 1.0, t1 - t + 1)[:, None]
        R = ((1 - w) * prev + w * ac) * scales
        cmd = sim.apply(sim.base[t:t1 + 1], R)
        roll = sim.rollout(t, t1, cmd, state, mech)
        r, lost = _reward(sim, t, roll, ac)
        out["obs"].append(obs); out["act"].append(a); out["logp"].append(logp); out["rew"].append(r)
        out["commands"].append(cmd[1:])
        out["qpos"].append(roll["qpos"] if not out["qpos"] else roll["qpos"][1:])      # the start frame once
        state, mech, t, prev = roll["final_state"], roll["mechanism_final"], t1, ac
        done = lost or t >= sim.T - 1
        out["done"].append(done)
        if done:
            out["lost"] = lost
            break
        obs = _observe(sim, t, roll["qpos"][-1], roll["qvel"][-1], prev)
    out["last_obs"] = obs
    out["end"] = int(t)
    out["commands"] = np.concatenate(out["commands"]) if out["commands"] else np.zeros((0, sim.model.nq))
    out["qpos"] = np.concatenate(out["qpos"]) if out["qpos"] else np.zeros((0, sim.model.nq))
    return out


def _base_states(sim, act_frames):
    """The unrefined commands rolled out chunk by chunk: the integration state and mechanism at every action boundary,
    and the whole rollout."""
    states, t, state, mech, Q = {0: (None, None)}, 0, None, None, [sim.base[0][None]]
    while t < sim.T - 1:
        t1 = min(sim.T - 1, t + act_frames)
        r = sim.rollout(t, t1, sim.base[t:t1 + 1], state, mech)
        Q.append(r["qpos"][1:])
        if r["unstable"]:
            break
        state, mech, t = r["final_state"], r["mechanism_final"], t1
        states[t] = (state, mech)
    return states, np.concatenate(Q)


def reinforcement_learning_tracking(workspace, scene_semantic_json, scene_recon_json, reference=None, scene=None, out_dir=None, act_s=0.1,
                                    episode_s=2.0, iterations=300, episodes_per_iteration=16, hidden=64, lr=3e-4, gamma=0.98, lam=0.95,
                                    clip=0.2, epochs=4, minibatch=256, action_scale_arm_rad=0.05, action_scale_grip_rad=0.3, eval_every=10,
                                    workers=None, max_minutes=60.0, seed=0):
    t_all = time.time()
    ws = Path(workspace).resolve()
    sem_p, recon_p = Path(scene_semantic_json).resolve(), Path(scene_recon_json).resolve()
    sem, recon = json.loads(sem_p.read_text()), json.loads(recon_p.read_text())
    out = Path(out_dir or ws / "methods" / NAME).resolve()
    out.mkdir(parents=True, exist_ok=True)
    scene_p = TR.scene_for(ws, recon_p, scene)
    model = mujoco.MjModel.from_xml_path(str(scene_p))
    ref_in, info = TR.best_reference(ws, model, out / "reference_in.npz", given=reference, sem=sem)
    bt = TR.BenchTracker(scene_p, ref_in, out / "bench", scene_recon_json=recon_p)
    TR.worker_init(bt.spec)                           # this process acts as a worker for the base rollout and its own checks
    sim = TR.worker_sim()
    T, fps = sim.T, sim.fps
    act = max(1, int(round(act_s * fps)))
    n_steps = max(1, int(round(episode_s / act_s)))
    scales = _scales(sim, action_scale_arm_rad, action_scale_grip_rad)
    rep = {"ok": False, "method": NAME, "out_dir": str(out), "scene": str(scene_p), "reference_in": {"path": ref_in, **info},
           "frames": int(T), "fps": fps, "tracked": {"bodies": list(sim.bodies), "joints": list(sim.joints)}, "arm_coupling": bt.couple_report,
           "settings": {"act_frames": act, "episode_steps": n_steps, "episodes_per_iteration": int(episodes_per_iteration)}}
    states, Q0 = _base_states(sim, act)
    rep["before"] = sim.summary(Q0)
    ep, er, _ = sim.errors(0, Q0)
    bad = np.where((ep > SLIPPING[0]).any(axis=1) | (np.degrees(er) > SLIPPING[1]).any(axis=1))[0] if ep.size else np.array([], int)
    f_lost = int(bad[0]) if len(bad) else T - 1
    boundaries = np.array(sorted(states), int)
    rep["first_lost_frame"] = int(f_lost) if len(bad) else None
    obs_dim = len(_state_obs(sim, None, 0, np.zeros(sim.dim)))
    agent = PP.PPO(obs_dim, sim.dim, hidden=int(hidden), lr=float(lr), seed=int(seed), clip=float(clip))
    rng = np.random.default_rng(seed)
    pool = TR.make_pool(bt.spec, TR.default_workers(workers))
    run = (lambda tasks: [_episode(x) for x in tasks]) if pool is None else (lambda tasks: list(pool.map(_episode, tasks)))
    span = act * n_steps                              # the curriculum: reset frames from [lo, f_lost], lo moving earlier
    lo = max(0, f_lost - span)
    best = {"score": None, "commands": None, "summary": None, "iteration": None}
    curve, stopped, it = [], None, 0
    try:
        for it in range(1, int(iterations) + 1):
            if TR.elapsed_min(t_all) > float(max_minutes):
                stopped = f"time budget ({max_minutes} min) spent after {it - 1} iterations"
                break
            cand = boundaries[(boundaries >= lo) & (boundaries <= max(lo, f_lost))]
            if not len(cand):
                cand = boundaries[:1]
            starts = rng.choice(cand, int(episodes_per_iteration))
            params = agent.params()
            tasks = [(params, int(s), states[int(s)][0], states[int(s)][1], n_steps, act, scales, int(rng.integers(1 << 30)), False)
                     for s in starts]
            eps = run(tasks)
            obs, acts, logps, advs, rets, ret_ep, lens = [], [], [], [], [], [], []
            collected = PP.Policy(params)             # the normalization the episodes were collected with
            raw = []
            for e in eps:
                if not e["obs"]:
                    continue
                x = collected.norm(np.asarray(e["obs"]))
                v = agent.value(x, normalized=True)
                last = 0.0 if e["done"][-1] else float(agent.value(collected.norm(e["last_obs"][None]), normalized=True)[0])
                adv, ret = PP.gae(np.asarray(e["rew"]), v, np.asarray(e["done"]), last, gamma, lam)
                obs += list(x); acts += e["act"]; logps += e["logp"]; advs += list(adv); rets += list(ret); raw += e["obs"]
                ret_ep.append(float(np.sum(e["rew"]))); lens.append(e["end"] - e["start"])
            if not obs:
                continue
            upd = agent.update(np.asarray(obs), np.asarray(acts), np.asarray(logps), np.asarray(advs), np.asarray(rets),
                               epochs=int(epochs), minibatch=int(minibatch), rng=rng, normalized=True)
            agent.norm.update(np.asarray(raw))
            # the curriculum widens toward the start once episodes keep the parts on the reference
            kept = np.mean([not e["lost"] for e in eps]) if eps else 0.0
            if kept >= 0.7:
                lo = max(0, lo - span // 2)
            row = {"iteration": it, "return": round(float(np.mean(ret_ep)), 4), "episode_frames": round(float(np.mean(lens)), 1),
                   "kept": round(float(kept), 2), "curriculum_from": int(lo), **upd}
            if it % int(eval_every) == 0 or it == int(iterations):
                ev = _episode((agent.params(), 0, None, None, int(np.ceil((T - 1) / act)), act, scales, 0, True))
                if len(ev["commands"]) == T - 1:
                    cmds = np.vstack([sim.base[:1], ev["commands"]])
                    s = sim.summary(ev["qpos"])
                    score = (s["worst_end_mm"] + s["worst_mean_mm"])
                    row["eval"] = {"worst_mean_mm": s["worst_mean_mm"], "worst_end_mm": s["worst_end_mm"]}
                    if best["score"] is None or score < best["score"]:
                        best = {"score": score, "commands": cmds, "summary": s, "iteration": it}
                else:
                    row["eval"] = {"lost_at_frame": int(ev["end"])}
            curve.append(row)
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)
    # the trained policy is kept when its whole-episode rollout costs less than the unrefined commands' (tracking,
    # penetration, contact force: what the reward pays for)
    base_cost = sim.cost(0, sim.rollout(0, T - 1, sim.base))
    best_cost = sim.cost(0, sim.rollout(0, T - 1, best["commands"])) if best["commands"] is not None else None
    rep["cost_before"], rep["cost_after"] = round(base_cost, 5), (round(best_cost, 5) if best_cost is not None else None)
    improved = best_cost is not None and best_cost < base_cost
    rep.update(iterations=it if not stopped else it - 1, stopped=stopped, curve=curve, improved=bool(improved),
               best_iteration=best["iteration"], after=best["summary"] if improved else rep["before"])
    np.savez(out / "policy.npz", **{f"W{i}": w for i, w in enumerate(agent.actor.W)}, **{f"b{i}": b for i, b in enumerate(agent.actor.b)},
             log_std=agent.log_std, obs_mean=agent.norm.mean, obs_var=agent.norm.var)
    path = bt.export(out / "reference.npz", best["commands"] if improved else sim.base,
                     source=f"{NAME} from {bt.ref['source']}" + ("" if improved else " (untrained: no improvement)"))
    plan = TR.write_plan(out / "plan.json", path, bt.model, sem, recon, scene_recon_json=recon_p, method=NAME)
    rep.update(ok=True, reference=path, plan=str(out / "plan.json"), n_entries=len(plan["subgoals"]), coverage=TR.coverage(path, bt.model, sem),
               policy=str(out / "policy.npz"),
               next_step=f"execute_subgoal_plan(scene={str(scene_p)!r}, plan={str(out / 'plan.json')!r}, out_dir={str(out / 'execution')!r}) "
                         "then package_stage3_episode and finish")
    rep["seconds"] = round(time.time() - t_all, 1)
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep
