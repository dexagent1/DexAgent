"""Cross-entropy optimization over a trajectory's parameters with hard gates first, reward second.

    from tool_library.skills.optimize_reward_cem import optimize_reward_cem
    best = optimize_reward_cem(evaluate, bounds={"lift_h": (0.03, 0.12), "approach_deg": (-30, 30)},
                               iterations=6, population=16, elite=4)
    best["params"], best["gates_failed"], best["reward"]

`evaluate(params) -> {"gates": {name: bool}, "terms": {name: float}}` builds
and verifies a trajectory for one parameter set (a plan's free numbers: a
lift height, an approach angle, a grasp offset). Candidates are ranked
lexicographically: fewer failing gates first, then a weighted sum of the
reward terms. The elite set refits a Gaussian per parameter, with a
variance floor so the search never collapses before the gates are met, and
the best candidate ever seen is returned with its full evaluation.
"""
from __future__ import annotations

TOOL = {
    "name": "optimize_reward_cem",
    "title": "Optimize reward (CEM)",
    "type": "skill",
    "category": "Trajectory & Execution",
    "summary": "Cross-entropy method over named bounded parameters, ranking by (number of failed gates, weighted reward); returns the best candidate and its evaluation.",
    "description": (
        "Samples parameter vectors from a Gaussian clipped to the bounds (initial mean at the "
        "centre or a given start, initial std a quarter of the range), evaluates each with the "
        "user's function, which returns boolean gates and real-valued terms, ranks by the "
        "number of failed gates and then by the weighted sum of terms (weights default to 1; "
        "give negative weights to penalize), and refits mean and std to the elite fraction "
        "with a std floor (2% of the range) and optional smoothing. Supports integer "
        "parameters (rounded), a deterministic seed, early stop when every elite passes all "
        "gates and the reward has stopped improving, an evaluation budget, and a history of "
        "every candidate for inspection. The paper's trajectory optimizer used this with "
        "terms such as object tracking error, penetration, smoothness and clearance."
    ),
    "inputs": [
        {"name": "evaluate", "type": "callable(params: dict) -> {gates: dict[str, bool], terms: dict[str, float], ...}", "description": "Builds and verifies one candidate."},
        {"name": "bounds", "type": "dict name -> (lo, hi)", "description": "Parameter ranges."},
        {"name": "weights", "type": "dict name -> float | None", "description": "Reward weights per term (higher reward is better)."},
        {"name": "integers", "type": "list[str]", "description": "Parameters rounded to integers."},
        {"name": "start", "type": "dict | None", "description": "Initial mean."},
        {"name": "iterations, population, elite, seed, max_evals", "type": "int", "description": "Defaults 8, 16, 4, 0, unlimited."},
        {"name": "std_floor_frac, smoothing, patience", "type": "float, float, int", "description": "Defaults 0.02, 0.1 and 2 (iterations without improvement, all gates met, before stopping early)."},
        {"name": "log", "type": "callable | None", "description": "Progress printer."},
    ],
    "outputs": [
        {"name": "best", "type": "dict", "description": "params, gates, gates_failed, terms, reward, evaluation (the full return of evaluate), iteration, n_evals, history (every candidate)."},
    ],
    "requirements": ["numpy"],
    "tags": ["optimization", "cem", "gates", "reward", "stage 3"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Battery Insertion"],
    "status": "tested",
    "entrypoint": "optimize_reward_cem",
}

import numpy as np


def reward_of(terms, weights=None):
    weights = weights or {}
    return float(sum((weights.get(k, 1.0)) * float(v) for k, v in terms.items()))


def optimize_reward_cem(evaluate, bounds, weights=None, integers=(), start=None, iterations=8, population=16,
                        elite=4, seed=0, max_evals=None, std_floor_frac=0.02, smoothing=0.1, patience=2, log=None):
    names = list(bounds)
    lo = np.array([bounds[n][0] for n in names], float)
    hi = np.array([bounds[n][1] for n in names], float)
    mean = np.array([start[n] if start and n in start else 0.5 * (bounds[n][0] + bounds[n][1]) for n in names], float)
    std = 0.25 * (hi - lo)
    floor = std_floor_frac * (hi - lo)
    rng = np.random.default_rng(seed)
    best, history, n_evals = None, [], 0

    def to_params(x):
        p = {}
        for k, n in enumerate(names):
            v = float(np.clip(x[k], lo[k], hi[k]))
            p[n] = int(round(v)) if n in integers else v
        return p

    last_best_reward, stale = None, 0
    for it in range(int(iterations)):
        X = rng.normal(mean, std, size=(int(population), len(names)))
        if it == 0:
            X[0] = mean                              # always evaluate the start
        scored = []
        for x in X:
            if max_evals is not None and n_evals >= max_evals:
                break
            params = to_params(x)
            ev = evaluate(params)
            n_evals += 1
            gates = {k: bool(v) for k, v in (ev.get("gates") or {}).items()}
            terms = {k: float(v) for k, v in (ev.get("terms") or {}).items()}
            failed = [k for k, v in gates.items() if not v]
            rew = reward_of(terms, weights)
            row = {"params": params, "gates": gates, "gates_failed": failed, "terms": terms, "reward": rew,
                   "evaluation": ev, "iteration": it}
            history.append(row)
            scored.append((len(failed), -rew, np.array([params[n] for n in names], float), row))
            if best is None or (len(failed), -rew) < (len(best["gates_failed"]), -best["reward"]):
                best = row
        if not scored:
            break
        scored.sort(key=lambda s: (s[0], s[1]))
        E = np.stack([s[2] for s in scored[:int(elite)]])
        mean = (1 - smoothing) * E.mean(axis=0) + smoothing * mean
        # the spread may at most halve per iteration: an elite set that happens
        # to cluster must not collapse the search before the mean has caught up
        std = np.maximum.reduce([(1 - smoothing) * E.std(axis=0) + smoothing * std, floor, 0.5 * std])
        if log:
            log(f"[cem] iter {it}: best gates failed {len(best['gates_failed'])} {best['gates_failed']} "
                f"reward {best['reward']:.4f}  elite mean {dict(zip(names, np.round(mean, 4)))}")
        all_pass = all(s[0] == 0 for s in scored[:int(elite)])
        improved = last_best_reward is None or best["reward"] > last_best_reward + 1e-6
        stale = 0 if improved else stale + 1
        converged = bool(np.all(std <= 1.05 * floor))
        if all_pass and stale >= patience and converged:
            break
        last_best_reward = best["reward"]
    best = dict(best)
    best.update(n_evals=n_evals, history=history)
    return best
