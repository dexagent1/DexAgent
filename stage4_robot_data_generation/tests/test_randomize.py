"""Quick tests of Stage 4's randomize step: relational goals and the physics replay of accepted layouts.

Runs in the Stage 4 environment in a few seconds, on the synthetic task folders of synthetic.py:

    cd stage4_robot_data_generation && python tests/test_randomize.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import synthetic  # noqa: E402
from stage4 import randomize  # noqa: E402
from stage4.config import DEFAULTS, load_config  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="stage4_rand_", dir=os.environ.get("TMPDIR")))
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def diagnostic_run(cfg):
    """Exercise the sampler/warper without publishing any robot episode.

    Private fixtures are saved under diagnostics for assertions; they carry no
    physics success claim and cannot pass export.validate_sample.
    """
    ep = randomize.Episode(cfg)
    rz = randomize.Randomizer(ep, cfg, log=lambda *_: None, physics_replay=False, reexecute=False)
    rows, accepted = [], 0
    dst = cfg.out / "diagnostics"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "original").mkdir(exist_ok=True)
    np.savez_compressed(dst / "original/trajectory.npz", qpos=ep.Q)
    for i in range(int(cfg["max_attempts"])):
        if accepted >= cfg["num_samples"]:
            break
        res = rz.attempt(i)
        rows.append(res["row"])
        if res["ok"]:
            out = dst / f"{accepted:04d}"
            out.mkdir(exist_ok=True)
            np.savez_compressed(out / "trajectory.npz", qpos=res["Q"], labels=res.get("labels", ep.labels))
            (out / "sample.json").write_text(json.dumps({"layout": res["layout"],
                "checks": dict(res["report"], method=res["method"]), "diagnostic_only": True}, default=float))
            accepted += 1
    summary = {"accepted": accepted, "attempts": len(rows), "rows": rows,
        "relations": {n: f"{o['relation'] or 'rests with'} {o['rest']}" for n, o in ep.objects.items() if o["rests_from"] is not None}}
    (cfg.out / "diagnostics.json").write_text(json.dumps(summary, default=float))
    return summary


def run(cfg_path, physics=None, attempts=None):
    cfg = load_config(cfg_path)
    if physics is not None:
        cfg["checks"]["physics_replay"] = physics
    if attempts is not None:
        cfg["max_attempts"] = attempts
    summary = diagnostic_run(cfg)
    return cfg, summary, None if summary["accepted"] >= cfg["num_samples"] else "not enough diagnostic draws"


def main():
    check("physics replay is on by default", DEFAULTS["checks"]["physics_replay"] is True and DEFAULTS["checks"]["physics_end_tol_m"] > 0)

    # 1. a stacked tower: every kept sample ends with C on A on B, wherever B went
    cfg_p, _, _ = synthetic.stack_task(TMP / "stack")
    cfg, s, err = run(cfg_p)
    ok_end = True
    for k in range(s["accepted"]):
        smp = json.loads((cfg.out / "diagnostics" / f"{k:04d}" / "sample.json").read_text())
        Q = np.load(cfg.out / "diagnostics" / f"{k:04d}" / "trajectory.npz")["qpos"]
        z = Q[-1]
        qa = {n: 7 * i + 6 for i, n in enumerate("ABC")}           # hand: 6 joints, then three free bodies
        dB = np.array(smp["layout"]["objects"]["B"][:2])
        z0 = np.load(cfg.path("trajectory"))["qpos"][-1]
        for n in "AC":
            ok_end &= np.linalg.norm((z[qa[n]:qa[n] + 2] - z0[qa[n]:qa[n] + 2]) - (z[qa["B"]:qa["B"] + 2] - z0[qa["B"]:qa["B"] + 2])) < 0.03
            ok_end &= np.linalg.norm(z[qa["B"]:qa["B"] + 2] - z0[qa["B"]:qa["B"] + 2] - dB) < 0.03
    check("stack: samples kept, relations recorded, tower follows B", err is None and s["accepted"] == 2 and ok_end
          and s["relations"] == {"A": "rests_on B", "C": "rests_on A"} and (cfg.out / "diagnostics" / "original" / "trajectory.npz").exists(),
          f"{s['accepted']}/{s['attempts']} {s['relations']}")

    # Legacy floating-wrist recordings are no longer replayable robot episodes.
    # Real rigid replay/state continuity is covered by test_runtime_replay.py.
    cfg_p = synthetic.bench_task(TMP / "legacy_bench", attached=True)
    try:
        randomize.run(load_config(cfg_p), log=lambda *_: None)
        check("legacy floating wrist episode refused", False)
    except ValueError as exc:
        check("legacy floating wrist episode refused", "rigid" in str(exc) or "proposal" in str(exc), str(exc))

    # 5. retiming: a layout that makes the carry faster than the joint-speed limit is slowed down where it is too fast
    #    instead of rejected; without retiming the same layouts are rejected on joint speed
    objs = {"A": {"carrier": "right", "release": "right: let go of A #2", "rests_on": "B"},
            "B": {"dx": [0.06, 0.08], "dy": [0.0, 0.0]},
            "C": {"carrier": "right", "release": "right: let go of C #4", "rests_on": "A"}}
    tight = {"joint_speed_scale": 1.0, "joint_speed_floor": 0.02}
    cfg_p, _, _ = synthetic.stack_task(TMP / "retime", objects=objs, checks=tight)
    cfg, s_rt, err = run(cfg_p)
    ok_rt = err is None and s_rt["accepted"] == 2 and all(r["method"] == "warp+retime" for r in s_rt["rows"] if r["ok"])
    for k in range(s_rt["accepted"]):
        z = np.load(cfg.out / "diagnostics" / f"{k:04d}" / "trajectory.npz")
        smp = json.loads((cfg.out / "diagnostics" / f"{k:04d}" / "sample.json").read_text())
        T0 = len(np.load(cfg.path("trajectory"))["qpos"])
        sp = smp["checks"]["joint_speed"]
        ok_rt &= len(z["qpos"]) > T0 and len(z["labels"]) == len(z["qpos"]) and all(
            v["max"] <= v["limit"] for kk, v in sp.items() if isinstance(v, dict)) and smp["checks"]["method"] == "warp+retime"
    cfg2 = load_config(cfg_p)
    cfg2["retime"]["enabled"] = False
    cfg2["max_attempts"] = 2
    try:
        diagnostic_run(cfg2)
    except AssertionError:
        pass
    s_no = json.loads((cfg2.out / "diagnostics.json").read_text())
    check("retime: too fast -> slowed down and kept; off -> rejected on joint speed", ok_rt and s_no["accepted"] == 0
          and all(r["failed"] == ["joint_speed"] for r in s_no["rows"]),
          f"{s_rt['accepted']}/{s_rt['attempts']} retimed {[r.get('retime', {}).get('added_s') for r in s_rt['rows']]}")

    # 6. reach pre-screen and adaptive sampling: the hand cannot go past x = 0.31 (B sits at x = 0.30, A and C at
    #    0.10), so a B moved toward +x cannot be reached; such draws are dropped by the pre-screen before the full
    #    adaptation, and the sampler steers later draws away from them (fewer rejections than the uniform draws of the
    #    same seeds)
    objs = {"A": {"carrier": "right", "release": "right: let go of A #2", "rests_on": "B"},
            "B": {"dx": [-0.02, 0.06], "dy": [-0.01, 0.01]},
            "C": {"carrier": "right", "release": "right: let go of C #4", "rests_on": "A"}}
    cfg_p, _, _ = synthetic.stack_task(TMP / "reach", objects=objs)
    xml = (TMP / "reach" / "scene.xml").read_text().replace('name="hx" type="slide" axis="1 0 0" range="-2 2"',
                                                          'name="hx" type="slide" axis="1 0 0" range="-2 0.313"')
    (TMP / "reach" / "scene.xml").write_text(xml)
    fails = {}
    for adaptive in (False, True):
        cfg = load_config(cfg_p)
        cfg["sampling"]["adaptive"] = adaptive
        cfg["sampling"]["prescreen_tol_scale"] = 1.0
        cfg["num_samples"], cfg["max_attempts"] = 30, 12
        try:
            diagnostic_run(cfg)
        except AssertionError:
            pass
        sr = json.loads((cfg.out / "diagnostics.json").read_text())
        fails[adaptive] = sr["rows"]
    pre = [r for r in fails[False] if r["failed"] == ["reach"]]
    n_fail = {k: sum(1 for r in v if not r["ok"]) for k, v in fails.items()}
    check("pre-screen drops unreachable layouts before the full adaptation", len(pre) >= 3 and all(
        r["prescreen"]["right"]["pos_m"] > 0.002 and r["layout"]["B"][0] > 0.0 for r in pre)
          and all(r["layout"]["B"][0] < 0.02 for r in fails[False] if r["ok"]), f"{len(pre)} of {len(fails[False])} pre-screened")
    same_first = fails[True][0]["layout"] == fails[False][0]["layout"]
    picked = [r["sampling"] for r in fails[True] if r["sampling"]["candidates"] > 1]
    check("adaptive sampling: same first draw, then fewer rejections inside the same ranges, picks recorded",
          same_first and n_fail[True] < n_fail[False] and picked and all(-0.02 <= r["layout"]["B"][0] <= 0.06 for r in fails[True]),
          f"rejected uniform {n_fail[False]}/12, adaptive {n_fail[True]}/12")

    # 7. several carriers: the left hand puts A down in the middle and the right hand stacks it on B (two carries, one
    #    resting place between them), or hands it over in the air; a hand reaching for a part the other hand holds
    #    follows the part where it is
    import mujoco
    for handover in (False, True):
        cfg_p, labels = synthetic.relay_task(TMP / f"relay_{handover}", handover=handover)
        cfg = load_config(cfg_p)
        ep = randomize.Episode(cfg)
        lay = {"objects": {"A": [0.015, -0.01, 0.0], "B": [0.02, 0.015, 8.0]}, "robot": {}}
        pl = ep.plan(lay)
        Q, ov, rep = ep.adapt(lay, pl)
        o, a = ep.objects["A"], ep.objects["A"]["qadr"]
        r1, r2 = o["rides"][0]["t_release"], o["rides"][1]["t_ride"]
        D = pl["D"]["B"]
        grip = [t for t, lab in enumerate(ep.labels) if lab == "right: grip A #3"]
        hand_on_part = all(np.allclose(pl["side"]["right"][t][1], pl["obj"]["A"][1][t], atol=1e-9) for t in grip)
        ok = (rep["ok"] and [r["side"] for r in o["rides"]] == ["left", "right"] and o["rest_steps"][-1][1] == "B"
              and np.allclose(Q[r1, a:a + 3], ep.Q[r1, a:a + 3]) and np.allclose(Q[r2, a:a + 3], ep.Q[r2, a:a + 3])
              and np.allclose(Q[0, a:a + 3] - ep.Q[0, a:a + 3], [0.015, -0.01, 0.0])
              and np.allclose(Q[-1, a:a + 3], D[0] @ ep.Q[-1, a:a + 3] + D[1], atol=1e-9) and hand_on_part)
        check(f"two carriers ({'handover' if handover else 'put down, picked up'}): A starts moved, passes where it "
              f"was passed, ends on B", ok, f"failed {rep['failed']} rides {[(r['side'], r['t_ride'], r['t_release']) for r in o['rides']]}")

    # 8. bottom-up starts: an object attached_to another with ranges of its own starts where its support starts, moved
    #    by its own draw in the support's frame
    objs = {"A": {"dx": [-0.01, 0.01], "dy": [-0.01, 0.01], "dyaw_deg": [-10, 10]},
            "B": {"dx": [0.02, 0.04], "dy": [-0.02, 0.02], "dyaw_deg": [-10, 10]},
            "C": {"attached_to": "A", "dx": [-0.005, 0.005], "dy": [-0.005, 0.005]}}
    cfg_p, _, _ = synthetic.stack_task(TMP / "bottomup", objects=objs, segments={"right": []})
    ep = randomize.Episode(load_config(cfg_p))
    lay = ep.sample_layout(np.random.default_rng([0, 3]))
    D = ep._displacements(lay)
    cA, cC = ep.objects["A"]["center"], ep.objects["C"]["center"]
    loc = cC + np.array(lay["objects"]["C"][:2] + [0.0])
    want = D["A"][0] @ loc + D["A"][1]
    check("bottom-up start: C follows A's draw, plus its own offset in A's frame", np.allclose(D["C"][0] @ cC + D["C"][1], want)
          and np.allclose(D["C"][0], D["A"][0]) and lay["objects"]["C"][0] != 0.0, str(np.round(want, 4)))

    # 9. a part put down right beside another (next_to) follows that part where it was when it was put down; the
    #    neighbour carried away later does not drag it along, and placing A beside B before B moves is no cycle
    cfg_p, _, _ = synthetic.beside_task(TMP / "beside")
    ep = randomize.Episode(load_config(cfg_p))
    lay = {"objects": {"A": [0.005, -0.004, 0.0], "B": [0.03, 0.01, 6.0]}, "robot": {}}
    pl = ep.plan(lay)
    Q, _, rep = ep.adapt(lay, pl)
    a, b = ep.objects["A"]["qadr"], ep.objects["B"]["qadr"]
    DB = pl["D"]["B"]
    a_err = float(np.linalg.norm(Q[-1, a:a + 3] - (DB[0] @ ep.Q[-1, a:a + 3] + DB[1])))
    check("next_to: A keeps its place beside B as B was placed, B's later carry leaves it", rep["ok"] and a_err < 1e-9
          and np.allclose(Q[-1, b:b + 3], ep.Q[-1, b:b + 3]) and ep.objects["A"]["frozen_from"] is not None,
          f"failed {rep['failed']} A end off {a_err:.1e} m")

    # 10. a base fixed to the world with a hinged part (a cabinet with a door): the warp moves the base through its
    #     body pose, and the scene a re-execution starts from has the base moved, the door on its hinge with it
    from stage4 import reexecute
    cab = ("<body name='cab' pos='0.45 0.50 0.05'><geom name='cab' type='box' size='0.05 0.05 0.05' rgba='0.5 0.4 0.3 1'/>"
           "<body name='door' pos='0.05 0.0 0.0'><joint name='hinge' type='hinge' axis='0 0 1' range='-1.5 0'/>"
           "<geom name='door' type='box' size='0.004 0.05 0.05' pos='0 0.05 0'/></body></body>")
    objs = {"cab": {"dx": [0.02, 0.03], "dy": [-0.01, 0.01], "dyaw_deg": [5, 8]}}
    cfg_p, _, _ = synthetic.stack_task(TMP / "cabinet", objects=objs, segments={"right": []}, extra_xml=cab)
    cfg = load_config(cfg_p)
    ep = randomize.Episode(cfg)
    lay = ep.sample_layout(np.random.default_rng([0, 0]))
    pl = ep.plan(lay)
    Q, ov, rep = ep.adapt(lay, pl)
    rx = reexecute.Reexecutor(ep, cfg, {"subgoals": []}, lambda *_: None, None, TMP / "cabinet_rx")
    scene, moved = rx.scene_for(lay, pl, TMP / "cabinet_rx" / "draw")
    import mujoco
    m2 = mujoco.MjModel.from_xml_path(str(scene))
    d2 = mujoco.MjData(m2)
    mujoco.mj_forward(m2, d2)
    D = pl["D"]["cab"]
    m0 = ep.model
    d0 = mujoco.MjData(m0)
    mujoco.mj_forward(m0, d0)
    door_err = float(np.linalg.norm(d2.xpos[m2.body("door").id] - (D[0] @ d0.xpos[m0.body("door").id] + D[1])))
    check("a fixed base with a hinged part moves as a whole (warp overrides, re-execution scene)", ep.objects["cab"]["qadr"] is None
          and "cab" in ov and moved == ["cab"] and door_err < 1e-6 and np.allclose(ov["cab"]["pos"], m2.body("cab").pos), f"door off {door_err:.1e} m")

    # 11. a config error in the relations stops the run before any draw
    objs = json.loads((TMP / "stack" / "stage4.json").read_text())["randomize"]["objects"]
    objs["A"]["rests_on"], objs["C"]["rests_on"] = "C", "A"
    cfg_p, _, _ = synthetic.stack_task(TMP / "cycle", objects=objs)
    try:
        randomize.Episode(load_config(cfg_p))
        check("cycle refused at load", False)
    except ValueError as e:
        check("cycle refused at load", "cycle" in str(e), str(e)[:80])
    print(f"{'ALL PASS' if FAILS == 0 else f'{FAILS} FAILED'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
