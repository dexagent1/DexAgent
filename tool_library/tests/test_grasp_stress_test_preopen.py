"""Three variations: a block on the floor gripped from above at two openings, a hand posed far above the block
(must fail at 'seated'), and misuse (bad open_frac / phases) that must raise ValueError."""
import json, os, sys, tempfile
from pathlib import Path
import numpy as np

from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.verifiers.grasp_stress_test_preopen import grasp_stress_test_preopen


def find_hand(side="right"):
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        c = p / "setup" / "robot" / f"sharpa_{side}.xml"
        if c.exists():
            return str(c)
    env = os.environ.get("DEXAGENT_SETUP")
    if env and (Path(env) / "robot" / f"sharpa_{side}.xml").exists():
        return str(Path(env) / "robot" / f"sharpa_{side}.xml")
    raise SystemExit("no setup/robot hand model found")


def block_xml(d, sx, sy, sz):
    p = Path(d) / f"block_{sx}_{sy}_{sz}.xml"
    p.write_text(f"""<mujoco><worldbody><body name="block" pos="0 0 {sz}"><freejoint name="block_free"/>
<geom name="block_geom" type="box" size="{sx} {sy} {sz}" mass="0.05" friction="1 0.02 0.002" condim="4"/></body></worldbody></mujoco>""")
    return str(p)


def main():
    from tool_library.skills.generate_rigid_grasps import generate_rigid_grasps
    hand = find_hand("right")
    fails = 0
    with tempfile.TemporaryDirectory() as d:
        for case, (sx, sy, sz) in {"block_A": (0.02, 0.04, 0.025), "block_B": (0.025, 0.05, 0.03)}.items():
            scene = {"hand_xml": hand, "side": "right", "floor": True, "grasp_body": "block_block",
                     "objects": [{"name": "block", "xml": block_xml(d, sx, sy, sz), "pos": [0.0, 0.0, 0.0], "quat": [1, 0, 0, 0]}]}
            g = generate_rigid_grasps(scene, verify=False)
            cands = g.get("grasps") or []
            if not cands:
                print(f"[FAIL] {case}: no candidate grasp to test"); fails += 1; continue
            pose = cands[0]
            reps = {of: grasp_stress_test_preopen(scene, pose, open_frac=of, phases=["approach", "close", "lift"], lift_heights_m=[0.05])
                    for of in (0.1, 0.55)}
            good = all(isinstance(r["ok"], bool) and r["open_frac"] == of and any(p["name"] == "seated" for p in r["phases"])
                       and isinstance(r["issues"], list) and (r["ok"] or r["issues"]) for of, r in reps.items())
            json.dumps(reps)
            if good:
                print(f"[PASS] {case}: report well formed at open_frac 0.1 ({reps[0.1]['ok']}) and 0.55 ({reps[0.55]['ok']})")
            else:
                print(f"[FAIL] {case}: {reps}"); fails += 1
            if case == "block_A":
                far = dict(pose); far["pos"] = (np.asarray(pose["pos"]) + [0, 0, 0.25]).tolist()
                r = grasp_stress_test_preopen(scene, far, phases=["close", "lift"], lift_heights_m=[0.05])
                if (not r["ok"]) and r["first_fail"] is not None and r["issues"]:
                    print(f"[PASS] hand_far_above: fails at {r['first_fail']} with an issue")
                else:
                    print(f"[FAIL] hand_far_above: {r}"); fails += 1
                bad = 0
                for kw in ({"open_frac": 1.5}, {"phases": ["lift"]}, {"phases": ["close", "wiggle"]}):
                    try:
                        grasp_stress_test_preopen(scene, pose, **kw)
                    except ValueError:
                        bad += 1
                if bad == 3:
                    print("[PASS] misuse: bad open_frac and phases raise ValueError")
                else:
                    print(f"[FAIL] misuse: only {bad}/3 raised"); fails += 1
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
