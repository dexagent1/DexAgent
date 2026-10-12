"""One rope over-move in contact physics on the setup bench (common/ropemoves.py `lay`, planned by plan_knot_moves):

a cable lying in an arch (construct_rope staged on the bench, its right end bending in toward the middle as the knot
example's rope does), the left hand's thumb-index pinch on a cable link
(fixtures/knot_pinch_left.json: optimize_grasp_pose family 'bar', lift and hold tests passed, as plan_knot_moves solves
it), plan_knot_moves with the move given (a link where the right leg bends in, laid by the left hand across the left
leg; straight down, the same reach folds the left forearm onto its own hand and the arm screen refuses it)
writes the `lay` entry with its
predicted and simulated crossing order, and execute_subgoal_plan runs it with the hand's servos in contact physics:
the link lands at its target, the fingers open with the wrist still, the executed crossing order is the planned one
(a loop, the carried end on top) and rope_knot_check reads the executed rope as unknotted with one crossing.
About eight minutes.

    MUJOCO_GL=egl python tool_library/tests/test_knot_lay_physics.py
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

R = 0.0156                     # the fixture pinch's cable
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    FAILS += 0 if ok else 1


def u_points(L, w=0.40, y0=0.30):
    """An arch lying on the table (as the Stage 2 rope of the knot example): the left leg straight down to its end, the
    right leg bending in toward the middle over its last 12 cm."""
    h = (L - np.pi * w / 2 - 0.13) / 2
    u = np.linspace(np.pi, 0, 40)
    arc = np.c_[w / 2 * np.cos(u), y0 + 0.04 + h + w / 2 * np.sin(u)]
    return np.vstack([[[-w / 2, y0 + 0.04 + h - (L - np.pi * w / 2 - 0.13) / 2]], arc, [[w / 2, y0 + 0.04]], [[w / 2 - 0.07, y0 - 0.06]]])


def bench(wd):
    from tool_library.skills.construct_rope import construct_rope
    from tool_library.skills.stage_on_bench import stage_on_bench
    pts = u_points(1.2)
    L = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    b = construct_rope(str(wd / "assets/rope"), name="rope", centreline=pts.tolist(), radius=R, segments=int(round(L / 0.037)) + 1,
                       grips=False, include_floor=False, friction=[0.7, 0.02, 0.001], density=1100)
    m0 = mujoco.MjModel.from_xml_path(b["xml"])
    bodies = ["rope_" + m0.body(i).name for i in range(1, m0.nbody)]
    stage_on_bench(str(ROOT / "setup" / "scene.xml"), assets=[{"name": "rope", "xml": b["xml"], "x": 0.0, "y": 0.0}], out_xml=str(wd / "scene.xml"),
                   rig={"origin": [0, 0, 0], "R": np.eye(3).tolist()})
    (wd / "scene_recon.json").write_text(json.dumps({"scene_xml": "scene.xml", "objects": [
        {"id": "rope", "type": "deformable", "skill": "construct_rope", "asset_xml": b["xml"], "root_body": bodies[0], "bodies": bodies}]}))
    return str(wd / "scene.xml"), str(wd / "scene_recon.json"), bodies


def main():
    from tool_library.skills.plan_knot_moves import plan_knot_moves
    from tool_library.skills.execute_subgoal_plan import execute_subgoal_plan
    from tool_library.verifiers.rope_knot_check import rope_knot_check
    wd = Path(tempfile.mkdtemp(prefix="dexagent_knot_lay_", dir=os.environ.get("TMPDIR")))
    scene, recon, bodies = bench(wd)
    n = len(bodies)
    k = n - 3                                                   # a link of the right leg where it bends in to its end
    tpl = {"left": str(ROOT / "tool_library/tests/fixtures/knot_pinch_left.json")}
    t0 = time.time()
    r = plan_knot_moves(scene, str(wd / "plan"), scene_recon_json=recon, pinch_poses=tpl, sides=["left"],
                        moves=[{"link": bodies[k], "target": [-0.30, 0.42], "side": "left"}])
    ent = r.get("entries") or []
    check("plan_knot_moves plans the given move as a lay entry (a prefix of a tie: partial) with its crossing order",
          r.get("partial") is True and len(ent) == 1 and ent[0]["motion"] == "lay" and ent[0].get("expect_order") is not None,
          f"{[(e['side'], e['object'], e['target_pos'], e.get('expect_order')) for e in ent]} {r.get('reason')} ({time.time() - t0:.0f} s)")
    if not ent:
        return 1
    want = ent[0]["expect_order"]
    t0 = time.time()
    s = execute_subgoal_plan(scene, {"scene_recon_json": recon, "subgoals": ent}, str(wd / "exec"))
    o = s["subgoals"][-1]
    meas = o.get("measured") or {}
    check("the lay executes in contact physics: the link at its target, let go, at rest, the planned crossing order",
          bool(s["ok"]), f"failed={o['failed']} crossings={meas.get('crossings')} object_end={meas.get('object_end')} ({time.time() - t0:.0f} s)")
    kc = rope_knot_check(scene=s["files"]["scene_physics"], trajectory=s["files"]["trajectory"], expect="unknot")
    check("rope_knot_check reads the executed rope as an unknotted loop", kc["ok"] and kc["frames"][0]["n_self_crossings"] >= 1,
          f"{kc['type']} {kc['frames'][0]['n_self_crossings']} crossings (planned '{want}')")
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}  ({wd})")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
