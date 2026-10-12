"""optimize_grasp_pose: one call finds a pose that passes the real stress tests from a deliberately bad start.

Cases (synthetic parametric objects, both hands, several sizes):
  * a box on the setup bench (stage_on_bench + hand_scene_from_recon), a hand-typed palm-down guess 6 cm off as the start,
    family 'rigid', every candidate screened for the right station's arm first;
  * a cylinder: a grasp generate_rigid_grasps found, turned 30 deg about the approach and shifted 15 mm along the pads'
    line and 10 mm out (it no longer lifts), local search only (family 'seeds');
  * a plate edge: a 140 x 100 x 8 mm plate on a pedestal, plate_edge_pinch_pose's pinch on its +x edge (the open hand
    shoves the plate on the way in), local search only;
  * a cable link: a link of a 10-link cable bridging two blocks, a handle_pinch_pose pinch tilted 35 deg about the link
    (its pads never seat), local search only, run through the launcher path (a no-op command prefix);
  * a 3 mm sheet lying flat on the table: nothing passes, and the result names the failure classes and a hint.
Every returned pose is re-checked with grasp_lift_test and grasp_hold_test outside the optimizer.

    MUJOCO_GL=egl python tool_library/tests/test_optimize_grasp_pose.py
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

from tool_library.common import hand as H  # noqa: E402
from tool_library.skills.generate_rigid_grasps import generate_rigid_grasps  # noqa: E402
from tool_library.skills.hand_scene_from_recon import hand_scene_from_recon  # noqa: E402
from tool_library.skills.handle_pinch_pose import handle_pinch_pose  # noqa: E402
from tool_library.skills.optimize_grasp_pose import DEFAULT_BOUNDS, _theta, optimize_grasp_pose, perturb  # noqa: E402
from tool_library.skills.plate_edge_pinch_pose import plate_edge_pinch_pose  # noqa: E402
from tool_library.skills.stage_on_bench import stage_on_bench  # noqa: E402
from tool_library.verifiers.grasp_hold_test import grasp_hold_test  # noqa: E402
from tool_library.verifiers.grasp_lift_test import grasp_lift_test  # noqa: E402

FAILS = []
TMP = Path(tempfile.mkdtemp(prefix="dexagent_optgrasp_", dir=os.environ.get("TMPDIR")))
HAND = {s: str(ROOT / "setup" / "robot" / f"sharpa_{s}.xml") for s in ("left", "right")}


def check(name, cond, info=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({info})" if info else ""), flush=True)
    if not cond:
        FAILS.append(name)


def write(name, body):
    p = TMP / f"{name}.xml"
    p.write_text(f'<mujoco model="{name}"><worldbody>{body}</worldbody></mujoco>')
    return str(p)


def scene(name, body, side, pos, grasp_body=None, yaw=0.0):
    s = {"hand_xml": HAND[side], "side": side, "floor": True, "objects": [{"name": name, "xml": write(name, body), "pos": list(pos), "yaw_deg": yaw}]}
    if grasp_body:
        s["grasp_body"] = grasp_body
    return s


def recheck(sc, pose):
    lift = grasp_lift_test(sc, pose)
    hold = grasp_hold_test(sc, pose)
    return lift["ok"] and hold["ok"], f"lift {[x['achieved_mm'] for x in lift['lifts']]} mm, hold slip {hold.get('slip_mm_s')} mm/s"


def report(r):
    return (f"{r['n_evals']} evaluations in {r['iterations']} iterations ({r['n_screened_out']} screened out, {r['n_arm_rejected']} "
            f"arm-rejected, {r['n_tested']} stress-tested), {r['seconds']} s, family {r['family']}, failure classes {r['failure_classes']}")


def case_bench_box():
    """A box on the bench, the right station's arm screened for every candidate."""
    t0 = time.time()
    ws = TMP / "bench"
    (ws / "assets" / "box").mkdir(parents=True)
    box_xml = ws / "assets" / "box" / "box.xml"
    box_xml.write_text('<mujoco model="box"><worldbody><body name="box"><freejoint/>'
                       '<geom name="box_body" type="box" size="0.024 0.019 0.028" pos="0 0 0.028" mass="0.1" rgba="0.8 0.45 0.2 1"/>'
                       '</body></worldbody></mujoco>')
    stage_on_bench(str(ROOT / "setup" / "scene.xml"), assets=[{"name": "box", "xml": str(box_xml), "x": 0.12, "y": 0.38, "yaw_deg": 20.0}],
                   out_xml=str(ws / "scene.xml"), rig={"origin": [0, 0, 0], "R": np.eye(3).tolist()})
    m = mujoco.MjModel.from_xml_path(str(ws / "scene.xml"))
    root = next(m.body(b).name for b in range(m.nbody) if m.body(b).name.startswith("box_") and m.body_parentid[b] == 0)
    (ws / "scene_recon.json").write_text(json.dumps({"scene_xml": "scene.xml", "objects": [
        {"id": "box", "type": "rigid", "asset_xml": "assets/box/box.xml", "root_body": root, "bodies": [root],
         "pose": {"x": 0.12, "y": 0.38, "z": 0.0, "yaw_deg": 20.0}}]}))
    hs = hand_scene_from_recon(str(ws / "scene_recon.json"), "right", "box", out_json=str(ws / "hs_box.json"))
    sc = H.load_scene_spec(str(ws / "hs_box.json"))
    # a hand-typed guess: palm down, fingers forward, 6 cm above and behind the box (nothing touches)
    R = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
    bad = {"side": "right", "pos": [0.12, 0.30, 0.12], "euler": [float(v) for v in H.mat_to_euler(R)],
           "finger_targets": H.HandRig.build(mujoco.MjModel.from_xml_path(HAND["right"]), "", "right").finger_targets(0.3),
           "grip_fingers": ["thumb", "index"], "approach": [0.0, 0.0, 1.0]}
    b0 = grasp_lift_test(sc, bad)
    r = optimize_grasp_pose(str(ws / "hs_box.json"), family="rigid", starts=bad, bench_scene=str(ws / "scene.xml"),
                            arm_waypoints=[[0.0, 0.0, 0.10]], out_json=str(ws / "best_box.json"))
    ok, info = recheck(sc, r["pose"]) if r["ok"] else (False, "")
    arm = (r["pose"] or {}).get("optimize", {}).get("arm") or {}
    check("bench box (right): from a guess that touches nothing, a pose the right arm holds (grasp, pre-grasp, 10 cm lift) and that "
          "lifts and holds", not b0["ok"] and r["ok"] and ok and arm.get("ok") is True and Path(r["out_json"]).exists()
          and r["pose"]["stress_tests"]["grasp_lift_test"]["ok"] and r.get("passing_json"),
          f"start fails at {b0['first_fail']}; {report(r)}; {info}; arm residual {arm.get('residual_mm')} mm ({time.time() - t0:.0f} s)")


def case_cylinder():
    t0 = time.time()
    sc = scene("cyl", '<body name="cyl"><freejoint/><geom name="cyl_body" type="cylinder" size="0.026 0.045" pos="0 0 0.045" mass="0.12"/></body>',
               "left", (-0.12, 0.42, 0.0))
    g = generate_rigid_grasps(sc, yaw_step_deg=90, include_top=False, finger_dirs=(), thumb_oppose_fracs=(0.4,), pinch_ik_iters=160)
    ref = next((c for c in g["grasps"][:4] if grasp_lift_test(sc, c, lift_heights_m=(0.06, 0.10))["ok"]), None)
    if ref is None:
        check("cylinder (left): a reference grasp", False, "generate_rigid_grasps found none that lifts")
        return
    u = np.zeros(len(DEFAULT_BOUNDS))
    th = _theta(u, ref, DEFAULT_BOUNDS)
    th.update(roll_deg=30.0, off_pinch_m=0.015, off_approach_m=0.010)
    bad = perturb(sc, ref, th)
    b0 = grasp_lift_test(sc, bad)
    r = optimize_grasp_pose(sc, family="seeds", starts=[bad], out_json=str(TMP / "best_cyl.json"))
    ok, info = recheck(sc, r["pose"]) if r["ok"] else (False, "")
    check("cylinder (left): a generated grasp turned 30 deg and shifted 15 mm is verified or repaired by local search to one that "
          "lifts and holds", r["ok"] and ok, f"start passes={b0['ok']}; {report(r)}; {info} ({time.time() - t0:.0f} s)")


def case_plate():
    t0 = time.time()
    sc = scene("plate", '<body name="ped" pos="0 0 0.02"><geom name="ped_g" type="box" size="0.025 0.02 0.02" rgba="0.5 0.5 0.5 1"/></body>'
                        '<body name="plate" pos="0 0 0.044"><freejoint/><geom name="plate_g" type="box" size="0.07 0.05 0.004" mass="0.05"/></body>',
               "right", (0.05, 0.40, 0.0), grasp_body="plate_plate")
    bad = plate_edge_pinch_pose(sc, "plate_g", "+x", min_out_m=0.01, inset_m=0.018)
    b0 = grasp_lift_test(sc, bad)
    r = optimize_grasp_pose(sc, family="seeds", starts=bad, max_evals=192, iterations=12, out_json=str(TMP / "best_plate.json"))
    ok, info = recheck(sc, r["pose"]) if r["ok"] else (False, "")
    check("plate edge (right): a plate-edge pinch whose open hand shoves the plate is searched into one that lifts and holds",
          not b0["ok"] and r["ok"] and ok, f"start fails at {b0['first_fail']}; {report(r)}; {info} ({time.time() - t0:.0f} s)")


def cable_body(n=10, L=0.03, r=0.005):
    body = ""
    for i in range(n):
        j = '<freejoint/>' if i == 0 else (f'<joint name="cable_jy{i}" type="hinge" axis="0 1 0" stiffness="0.03" damping="0.002"/>'
                                           f'<joint name="cable_jz{i}" type="hinge" axis="0 0 1" stiffness="0.03" damping="0.002"/>')
        pos = f'{-0.5 * n * L} 0 0.046' if i == 0 else f'{L} 0 0'
        body += (f'<body name="cable_l{i}" pos="{pos}">{j}<geom name="cable_g{i}" type="capsule" fromto="0 0 0 {L} 0 0" size="{r}" '
                 f'mass="0.004" rgba="0.9 0.2 0.2 1"/>')
    body += "</body>" * n
    blocks = ''.join(f'<body name="block{k}" pos="{x} 0 0.02"><geom name="block{k}_g" type="box" size="0.03 0.03 0.02"/></body>'
                     for k, x in ((0, -0.11), (1, 0.11)))
    return blocks + body


def case_cable():
    t0 = time.time()
    sc = scene("cable", cable_body(), "right", (0.0, 0.40, 0.0), grasp_body="cable_cable_l5")
    bad = handle_pinch_pose(sc, "cable_cable_g5", s=0.5, fingers=("thumb", "index"), tilt_deg=35, yaw_step_deg=45, pitches_deg=(-45, 0, 45), iters=200)
    b0 = grasp_lift_test(sc, bad)
    r = optimize_grasp_pose(sc, family="seeds", starts=bad, launcher=["env"], launcher_batches=2, workers=4, out_json=str(TMP / "best_cable.json"))
    ok, info = recheck(sc, r["pose"]) if r["ok"] else (False, "")
    check("cable link (right): a pinch tilted 35 deg about the link never seats; the search (through the launcher path) finds one "
          "that lifts the link and the cable hanging from it, and holds", not b0["ok"] and r["ok"] and ok,
          f"start fails at {b0['first_fail']}; {report(r)}; {info} ({time.time() - t0:.0f} s)")
    # auto picks the bar family for a cable link (its own capsule, not the chain hanging from it)
    from tool_library.skills.optimize_grasp_pose import auto_family
    brick = ('<body name="brick"><freejoint/><geom name="brick_g" type="box" size="0.016 0.008 0.0055" pos="0 0 0.0055" mass="0.003"/>'
             '<geom name="brick_top" type="box" size="0.016 0.008 0.0005" pos="0 0 0.0115" mass="0.0005"/></body>')
    check("auto family: a cable link is a bar, the plate a plate, the cylinder rigid, and a stud brick whose top geom is a thin "
          "plate is still rigid (the whole body decides)", auto_family(scene("brick", brick, "left", (0.0, 0.4, 0.0)))[0] == "rigid" and
          auto_family(sc)[0] == "bar" and auto_family(scene("plate2", '<body name="plate"><freejoint/><geom name="plate_g" type="box" size="0.07 0.05 0.004" pos="0 0 0.004" mass="0.05"/></body>',
                                                            "right", (0.0, 0.4, 0.0)))[0] == "plate"
          and auto_family(scene("cyl2", '<body name="cyl"><freejoint/><geom name="cyl_body" type="cylinder" size="0.026 0.045" pos="0 0 0.045" mass="0.12"/></body>',
                                "left", (0.0, 0.4, 0.0)))[0] == "rigid")


def case_impossible():
    t0 = time.time()
    sc = scene("sheet", '<body name="sheet"><freejoint/><geom name="sheet_g" type="box" size="0.05 0.035 0.0015" pos="0 0 0.0015" mass="0.02"/></body>',
               "right", (0.05, 0.40, 0.0))
    R = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
    start = {"side": "right", "pos": [0.05, 0.32, 0.03], "euler": [float(v) for v in H.mat_to_euler(R)],
             "finger_targets": H.HandRig.build(mujoco.MjModel.from_xml_path(HAND["right"]), "", "right").finger_targets(0.5),
             "grip_fingers": ["thumb", "index"], "approach": [0.0, 0.0, 1.0]}
    r = optimize_grasp_pose(sc, family="seeds", starts=start, max_evals=32, iterations=2)
    check("a 3 mm sheet flat on the table: nothing passes, the failure classes and a hint say why", not r["ok"] and r["failure_classes"]
          and r["hint"] and r["best_tries"], f"{report(r)}; hint: {r['hint']} ({time.time() - t0:.0f} s)")


def main():
    t0 = time.time()
    case_cable()
    case_plate()
    case_cylinder()
    case_bench_box()
    case_impossible()
    bad = 0
    for kw in ({"family": "seeds"}, {"family": "nope", "starts": {"pos": [0, 0, 0], "euler": [0, 0, 0]}}, {"tests": ["no_such_test"]}):
        try:
            optimize_grasp_pose(scene("b", '<body name="b"><freejoint/><geom type="box" size="0.02 0.02 0.02" pos="0 0 0.02"/></body>', "right", (0, 0.4, 0)), **kw)
        except ValueError:
            bad += 1
    check("misuse raises ValueError (seeds without starts, unknown family, unknown test)", bad == 3, f"{bad}/3")
    print(f"{len(FAILS)} failed, {time.time() - t0:.0f} s")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
