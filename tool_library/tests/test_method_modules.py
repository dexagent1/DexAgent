"""Smoke tests of Stage 3's CPU method modules on a synthetic bench (a box on setup/'s bench, a reach-close-lift reference
as the agent's attempt): sampling_mpc_refinement and reinforcement_learning_tracking each start from the workspace,
run a tiny budget (in this process and in worker processes) and write a tracking reference and a plan of
track_reference entries; the bench tracker's frames keep their reference rows' labels and segments, with the arms as
given or coupled; execute_subgoal_plan accepts the methods' plan (a dry run: planning and the arms as given, no physics);
the numpy PPO learns a one-step problem. Budgets are tiny: these check the plumbing, not the methods' results.

    python tool_library/tests/test_method_modules.py
"""
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import test_physics_informed_retargeting as F  # noqa: E402  (the synthetic bench, reference and attempt workspace)
from tool_library.common import ppo as PP  # noqa: E402
from tool_library.common import tracking as TR  # noqa: E402

FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def fixture(tag):
    model, scene = F.make_bench(tag, (0.02, 0.02, 0.03), 0.12, 0.42, 0.0, second=False)
    Q, labels = F.lift_reference(model, T=45)
    ws = F._attempt_workspace(tag, model, Q, labels)
    sem_p = F.TMP / tag / "sem.json"; F._semantics(sem_p, model)
    recon_p = F.TMP / tag / "scene_recon.json"; recon_p.write_text(json.dumps({"scene_xml": "scene.xml", "objects": []}))
    return model, scene, ws, sem_p, recon_p


def outputs_ok(r, model):
    if not (r.get("ok") and Path(r["reference"]).is_file() and Path(r["plan"]).is_file()):
        return False
    ref = TR.load_reference(r["reference"], model)
    plan = json.loads(Path(r["plan"]).read_text())
    return bool(plan["subgoals"]) and all(e["motion"] == "track_reference" for e in plan["subgoals"]) and len(ref["qpos"]) > 0


def test_mpc():
    from tool_library.skills.sampling_mpc_refinement import sampling_mpc_refinement
    model, scene, ws, sem_p, recon_p = fixture("mpc")
    for workers in (1, 2):
        r = sampling_mpc_refinement(ws, sem_p, recon_p, scene=str(scene), out_dir=str(ws / f"mpc_{workers}"), window=[0, 12], samples=3,
                                    horizon_s=0.2, replan_s=0.1, knot_s=0.1, workers=workers, max_minutes=10)
        check(f"sampling MPC ({workers} worker{'s' if workers > 1 else ''}): re-plans from the attempt, rolls candidates out, "
              "writes the reference and its track_reference plan", outputs_ok(r, model) and r["replans"] >= 3 and r["rollouts"] >= 9
              and "worst_mean_mm" in r["before"] and "worst_mean_mm" in r["after"],
              json.dumps({k: r.get(k) for k in ("replans", "rollouts", "improved", "stopped", "seconds")}))
    return scene, r


def test_rl():
    from tool_library.skills.reinforcement_learning_tracking import reinforcement_learning_tracking
    model, scene, ws, sem_p, recon_p = fixture("rl")
    r = reinforcement_learning_tracking(ws, sem_p, recon_p, scene=str(scene), out_dir=str(ws / "rl"), iterations=2, episodes_per_iteration=2,
                                        episode_s=0.2, act_s=0.1, eval_every=2, workers=2, max_minutes=10)
    check("RL tracking: PPO iterations from the reference's own states, a whole-episode evaluation, the policy, reference and plan written",
          outputs_ok(r, model) and len(r["curve"]) == 2 and Path(r["policy"]).is_file() and "eval" in r["curve"][-1],
          json.dumps({k: r.get(k) for k in ("iterations", "improved", "first_lost_frame", "seconds")}) + f" curve={r['curve'][-1]}")


def test_bench_tracker_labels():
    """Every command frame's segment is the one of the reference row its label came from (no one-frame phases where
    the retimed coupling and the segment map disagree), and a reference whose arms hold its hands runs them as given."""
    model, scene, ws, sem_p, recon_p = fixture("bt")
    path, _ = TR.reference_from_attempt(ws, model, F.TMP / "bt" / "ref.npz")
    for given in (True, False):
        if not given:                                 # the same rows, the arms left to the coupling
            ref = TR.load_reference(path, model)
            path = TR.save_reference(F.TMP / "bt" / "ref_free.npz", model, ref["qpos"], ref["labels"], ref["fps"], ref["segment"],
                                     ref["segments"], "test", arms_coupled=False)
        bt = TR.BenchTracker(scene, path, F.TMP / "bt" / f"bench_{given}")
        rows = bt.ref_row
        ok = all(bt.labels[t] == bt.ref["labels"][rows[t]] for t in range(len(rows)))
        check(f"bench tracker ({'arms as given' if given else 'arms coupled'}): every frame's label is its reference row's, so "
              "segments and labels agree at every boundary", ok and bt.couple_report["arms"] == ("as given" if given else "coupled"),
              json.dumps(bt.couple_report, default=str)[:200])


def test_executor_accepts_plan(scene, r):
    from tool_library.skills.execute_subgoal_plan import execute_subgoal_plan
    out = Path(r["plan"]).parent / "dry"
    res = execute_subgoal_plan(str(scene), r["plan"], str(out), dry_run=True)
    rows = res["subgoals"]
    planned = [g for g in rows if "planning" not in (g.get("failed") or []) and not str(g.get("error") or "").startswith(("ValueError", "KeyError"))]
    check("execute_subgoal_plan takes the method's plan: every track_reference entry is planned and its arms coupled (dry run)",
          len(rows) >= 1 and len(planned) == len(rows) and all(g.get("motion") == "track_reference" for g in rows),
          json.dumps([{k: g.get(k) for k in ("index", "ok", "failed", "error")} for g in rows], default=str)[:400])


def test_ppo():
    rng = np.random.default_rng(0)
    agent = PP.PPO(2, 1, hidden=16, lr=3e-3, seed=0)
    for _ in range(60):                               # one-step episodes: reward -(a - 0.5)^2
        obs = rng.normal(size=(64, 2))
        agent.norm.update(obs)
        acts, logps = zip(*[agent.act(o, rng) for o in obs])
        acts = np.array(acts)
        rew = -(acts[:, 0] - 0.5) ** 2
        v = agent.value(obs)
        adv, ret = rew - v, rew
        agent.update(obs, acts, np.array(logps), adv, ret, epochs=4, minibatch=32, rng=rng)
    mu = PP.Policy(agent.params()).mean(rng.normal(size=(200, 2)))[:, 0].mean()
    check("numpy PPO: the policy's mean moves to the rewarded action", abs(mu - 0.5) < 0.1, f"(mean {mu:.3f})")


if __name__ == "__main__":
    scene, r = None, None
    for fn in (test_ppo, test_bench_tracker_labels, test_mpc, test_rl):
        try:
            got = fn()
            if fn is test_mpc:
                scene, r = got
        except Exception as e:
            import traceback
            traceback.print_exc()
            check(fn.__name__, False, f"crashed: {type(e).__name__}: {e}")
    if r is not None:
        try:
            test_executor_accepts_plan(scene, r)
        except Exception as e:
            import traceback
            traceback.print_exc()
            check("test_executor_accepts_plan", False, f"crashed: {type(e).__name__}: {e}")
    shutil.rmtree(F.TMP, ignore_errors=True)
    print(f"\n{'ALL PASS' if FAILS == 0 else f'{FAILS} FAILED'}")
    sys.exit(1 if FAILS else 0)
