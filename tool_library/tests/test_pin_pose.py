"""pin and pin_pose: one hand steadies a part, or a cable end, while the other hand acts. The failed runs had nothing for it:
the braid's free junction was dragged by every strand pull, and no hold kept the left hand out of the way of the hand
turning a bottle cap. Cases on the setup bench, each run by the real executor in contact physics:

1. a cabinet (a free box) whose drawer the right hand pushes shut, two sizes and placements: without a pin the whole
   cabinet slides with the drawer (the outcome names it: base_moved); with the left hand's press pin from pin_pose (kept
   out of the zone the right hand pushes in, screened for the left arm) the cabinet stays within 2 mm on EVERY frame of
   the push (`pinned`), and the drawer shuts;
2. a free cable end pressed by the left hand (pin_pose on the end link) while the right hand pinches a link 30 cm along
   and drags it 5 cm: the end stays within 2 mm (without the pin it is dragged);
3. the planner: a pinning hand may only release, go home or wait; a later entry carries the press cap of the pinning
   hand (with_pin_caps) and the pin's reference pose; the release eases the press off first and ends the pin;
4. the outcome rejects a pinned part moved 5 mm on a few frames, and a pinning hand that lost the part;
5. packaging a physical primitive with forged stress claims and an invalid video is refused by the Stage 3 gate.
About ten minutes on an idle machine.

    MUJOCO_GL=egl python tool_library/tests/test_pin_pose.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

from tool_library.common import bimanual as B  # noqa: E402
from tool_library.common import episode as E  # noqa: E402
from tool_library.common import outcome as O  # noqa: E402
from tool_library.skills.construct_rope import construct_rope  # noqa: E402
from tool_library.skills.execute_subgoal_plan import execute_subgoal_plan  # noqa: E402
from tool_library.skills.hand_scene_from_recon import hand_scene_from_recon  # noqa: E402
from tool_library.skills.optimize_grasp_pose import optimize_grasp_pose  # noqa: E402
from tool_library.skills.optimize_pose_for_reach import optimize_pose_for_reach  # noqa: E402
from tool_library.skills.pin_pose import pin_pose  # noqa: E402
from tool_library.skills.plan_subgoal_keyframes import plan_subgoal_keyframes  # noqa: E402
from tool_library.skills.push_pose_from_point import push_pose_from_point  # noqa: E402
from tool_library.skills.stage_on_bench import stage_on_bench  # noqa: E402
from tool_library.skills.write_stage4_config import write_stage4_config  # noqa: E402
from tool_library.verifiers.stage3_gate_suite import stage3_gate_suite  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dexagent_pin_", dir=os.environ.get("TMPDIR")))
FAILS = 0
WALL = 0.006


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def cabinet_xml(d, name, hx, hy, hz, mass=0.25, drawer_open=0.05, slide_friction=8.0):
    """A cabinet (a free box of walls) with a drawer on a slide along -y, `drawer_open` out; the slide's dry friction is
    more than the cabinet's own friction on the table, so a push on the drawer drags an unpinned cabinet."""
    d = Path(d); d.mkdir(parents=True, exist_ok=True)
    ix, iy, iz = hx - WALL, hy - WALL, hz - WALL
    gx, gy, gz = ix - 0.002, iy - 0.004, iz - 0.003
    box = lambda n, s, p, c: f'<geom name="{name}_{n}" type="box" size="{s[0]} {s[1]} {s[2]}" pos="{p[0]} {p[1]} {p[2]}" rgba="{c}"/>'
    walls = "".join([box("top", (hx, hy, WALL / 2), (0, 0, hz - WALL / 2), "0.55 0.4 0.3 1"), box("bot", (hx, hy, WALL / 2), (0, 0, -hz + WALL / 2), "0.55 0.4 0.3 1"),
                     box("sl", (WALL / 2, hy, iz), (-hx + WALL / 2, 0, 0), "0.55 0.4 0.3 1"), box("sr", (WALL / 2, hy, iz), (hx - WALL / 2, 0, 0), "0.55 0.4 0.3 1"),
                     box("back", (ix, WALL / 2, iz), (0, hy - WALL / 2, 0), "0.55 0.4 0.3 1")])
    drawer = "".join([box("dfront", (hx, WALL / 2, hz - 0.002), (0, -gy - WALL / 2, 0), "0.8 0.7 0.5 1"), box("dbot", (gx, gy, WALL / 2), (0, 0, -gz + WALL / 2), "0.8 0.7 0.5 1"),
                      box("dl", (WALL / 2, gy, gz / 2), (-gx + WALL / 2, 0, -gz / 2), "0.8 0.7 0.5 1"), box("dr", (WALL / 2, gy, gz / 2), (gx - WALL / 2, 0, -gz / 2), "0.8 0.7 0.5 1")])
    I = (mass * (hy * hy + hz * hz) / 3, mass * (hx * hx + hz * hz) / 3, mass * (hx * hx + hy * hy) / 3)
    (d / f"{name}.xml").write_text(
        f'<mujoco model="{name}"><worldbody><body name="{name}" pos="0 0 {hz}"><freejoint/>'
        f'<inertial pos="0 0 0" mass="{mass}" diaginertia="{I[0]} {I[1]} {I[2]}"/>{walls}'
        f'<body name="{name}_drawer" pos="0 {-drawer_open} 0"><joint name="{name}_slide" type="slide" axis="0 -1 0" '
        f'range="{-drawer_open} {0.10 - drawer_open}" frictionloss="{slide_friction}" damping="2" armature="0.01"/>'
        f'<inertial pos="0 0 0" mass="0.06" diaginertia="1e-4 1e-4 1e-4"/>{drawer}</body></body></worldbody></mujoco>')
    return str(d / f"{name}.xml"), gy


def bench(ws, assets, objects):
    ws = Path(ws); ws.mkdir(parents=True, exist_ok=True)
    stage_on_bench(str(ROOT / "setup" / "scene.xml"), assets=assets, out_xml=str(ws / "scene.xml"), rig={"origin": [0, 0, 0], "R": np.eye(3).tolist()})
    (ws / "scene_recon.json").write_text(json.dumps({"scene_xml": "scene.xml", "objects": objects}))
    return str(ws / "scene.xml"), str(ws / "scene_recon.json")


def outcomes(d):
    return json.loads((Path(d) / "outcomes.json").read_text())


def cabinet_case(k, hx, hy, hz, cx, cy, with_no_pin=False):
    wd = TMP / f"cab{k}"
    open_ = 0.05
    xml, gy = cabinet_xml(wd / "assets", "cab", hx, hy, hz, drawer_open=open_)
    scene, recon = bench(wd / "stage2", [{"name": "cab", "xml": xml, "x": cx, "y": cy}],
                         [{"id": "cab", "type": "articulated", "asset_xml": xml, "root_body": "cab_cab", "bodies": ["cab_cab", "cab_cab_drawer"]}])
    hand_scene_from_recon(recon, "left", "cab", out_json=str(wd / "hs_left.json"))
    hand_scene_from_recon(recon, "right", "cab", out_json=str(wd / "hs_right.json"))
    # the right index pushes the drawer front shut; turned about the push normal until the right arm holds it
    yf = cy - open_ - gy - WALL
    push_pt = [cx + 0.3 * hx, yf, hz]
    pp = push_pose_from_point(str(wd / "hs_right.json"), push_pt, [0, -1, 0], finger="index")
    (wd / "push0.json").write_text(json.dumps(pp, default=float))
    optimize_pose_for_reach(scene, "right", str(wd / "push0.json"), hand_scene=str(wd / "hs_right.json"), axis="normal", range_deg=45, step_deg=15,
                            approach_m=0.04, out_json=str(wd / "push.json"))
    # the pin: a press on the cabinet's top, out of the 5 cm around the drawer front the right hand works in
    zone = [{"center": push_pt, "radius": 0.05}]
    r = pin_pose(str(wd / "hs_left.json"), body="cab_cab", keep_clear=zone, bench_scene=scene, side="left", out_json=str(wd / "pin.json"))
    best = r["pose"] or {}
    clr = (best.get("pin") or {}).get("keep_clear_mm")
    check(f"cabinet {k}: pin_pose finds an arm-reachable press on the top, out of the push zone", r["ok"] and clr is not None and clr >= 10.0
          and (best["pin"].get("reach") or {}).get("ok"), f"{best.get('pin')} rejected {r['rejected']}")
    if not r["ok"]:
        return
    push = {"index": 2, "motion": "push", "side": "right", "object": "cab_cab_drawer", "joint": "cab_cab_slide", "target": -open_ + 0.002,
            "pose": str(wd / "push.json"), "hand_scene": str(wd / "hs_right.json"), "hand_follows": "level"}
    pin = {"index": 1, "motion": "pin", "side": "left", "object": "cab_cab", "pose": str(wd / "pin.json"), "hand_scene": str(wd / "hs_left.json")}
    if with_no_pin:
        s0 = execute_subgoal_plan(scene, {"subgoals": [push]}, str(wd / "nopin"))
        o0 = outcomes(wd / "nopin")[-1]
        bm = (o0.get("measured") or {}).get("base_moved") or {}
        check(f"cabinet {k}: without a pin the drawer push drags the whole cabinet (the outcome says base_moved)", not s0["ok"]
              and bm.get("moved_m", 0) > 0.02, f"base moved {bm}, failed {o0.get('failed')}")
    s = execute_subgoal_plan(scene, {"subgoals": [pin, push]}, str(wd / "pin"))
    o = outcomes(wd / "pin")
    pm = ((o[-1].get("measured") or {}).get("pinned") or {}).get("cab_cab") or {}
    check(f"cabinet {k}: pinned by the left hand, the cabinet stays within 2 mm on every frame while the drawer shuts", s["ok"]
          and o[-1]["checks"].get("pinned:cab_cab") is True and pm.get("max_dev_mm", 9) <= 2.0 and pm.get("pinning_hand_touching_end"),
          f"pinned {pm}, joint end {(o[-1].get('measured') or {}).get('joint_end')}, {[(x['index'], x['ok'], x.get('failed')) for x in o]}")
    return scene, wd, o


def cable_case():
    wd = TMP / "cable"
    R_, L = 0.012, 0.6
    b = construct_rope(str(wd / "assets/rope"), name="rope", length=L, radius=R_, segments=24, pos=[0, 0, R_], grips=False, include_floor=False,
                       friction=[1.0, 0.02, 0.001], density=700)
    m0 = mujoco.MjModel.from_xml_path(b["xml"])
    bodies = ["rope_" + m0.body(i).name for i in range(1, m0.nbody)]
    scene, recon = bench(wd / "stage2", [{"name": "rope", "xml": b["xml"], "x": -0.24, "y": 0.42}],
                         [{"id": "rope", "type": "deformable", "skill": "construct_rope", "asset_xml": b["xml"], "root_body": bodies[0], "bodies": bodies}])
    m = mujoco.MjModel.from_xml_path(scene); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    links = [n for n in bodies if "B_" in n and not n.endswith(("first", "last"))]
    link = links[int(0.55 * len(links))]
    hand_scene_from_recon(recon, "left", "rope", out_json=str(wd / "hs_left.json"))
    hand_scene_from_recon(recon, "right", "rope", grasp_body=link, out_json=str(wd / "hs_right.json"))
    end = "rope_ropeB_first"
    pe, pl = d.xpos[m.body(end).id].copy(), d.xpos[m.body(link).id].copy()
    r = pin_pose(str(wd / "hs_left.json"), body=end, keep_clear=[{"center": pl.tolist(), "radius": 0.08}], bench_scene=scene, side="left",
                 out_json=str(wd / "pin.json"))
    check("cable: pin_pose finds a reachable press on the free end link", r["ok"], f"{(r['pose'] or {}).get('pin')} rejected {r['rejected']}")
    g = next(i for i in range(m.ngeom) if m.geom_bodyid[i] == m.body(link).id and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_CAPSULE)
    # the pinched link is swung 5 cm about the pinned end (a cable cannot stretch: a straight sideways pull would
    # tension it), lifted 1.5 cm so it slides over the table
    lift = 0.015
    rad = pl - pe; rad[2] = 0.0
    th = -0.05 / np.linalg.norm(rad)
    tgt = pe + Rot.from_rotvec([0.0, 0.0, th]).as_matrix() @ rad * (1 - 0.002 / np.linalg.norm(rad)); tgt[2] = pl[2] + lift
    # the executor turns a stress-tested pose by at most max_reach_turn_deg for its arm and refuses a larger turn (another
    # grasp): the right arm cannot hold a link pinch solved for the hand alone (handle_pinch_pose's needs a 58 deg turn), so
    # the bar family's pinches are searched with the station's arm screened at the grasp, the lift and the end of the
    # drag, and lift-tested (as an agent would)
    opt = optimize_grasp_pose(str(wd / "hs_right.json"), family="bar", bar=m.geom(g).name, bench_scene=scene,
                              arm_waypoints=[[0.0, 0.0, lift], (tgt - pl).tolist()], tests=["grasp_lift_test"], lift_heights_m=[0.04],
                              max_evals=128, iterations=8, workers=3, out_json=str(wd / "pinch.json"))
    check("cable: a right pinch on the link that the right arm holds and that lifts it", opt["ok"],
          f"{opt.get('failure_classes')} {opt.get('hint') or ''}"[:600])
    if not opt["ok"]:
        return
    grab = {"index": 2, "motion": "grasp_lift", "side": "right", "object": link, "pose": str(wd / "pinch.json"), "hand_scene": str(wd / "hs_right.json"), "lift_m": lift}
    drag = {"index": 3, "motion": "carry", "side": "right", "object": link, "target_pos": tgt.tolist(), "duration_s": 2.0}
    pin = {"index": 1, "motion": "pin", "side": "left", "object": end, "pose": str(wd / "pin.json"), "hand_scene": str(wd / "hs_left.json")}
    res = {}
    for tag, plan in (("nopin", [grab, drag]), ("pin", [pin, grab, drag])):
        s = execute_subgoal_plan(scene, {"subgoals": plan}, str(wd / tag))
        if not s["ok"]:
            check(f"cable: {tag} physical plan executes", False, json.dumps(s.get("first_failure"))[:1500])
            return
        ep = E.load_episode(wd / tag / "trajectory.npz")
        runtime = mujoco.MjModel.from_xml_path(str(wd / tag / "scene_physics.xml"))
        P0 = E.body_track(runtime, ep["qpos"], end)[0]; PL = E.body_track(runtime, ep["qpos"], link)[0]
        res[tag] = (s, outcomes(wd / tag), float(np.max(np.linalg.norm(P0 - P0[0], axis=1))), float(np.linalg.norm((PL[-1] - PL[0])[:2])))
    s, o, dev, moved = res["pin"]
    pm = ((o[-1].get("measured") or {}).get("pinned") or {}).get(end) or {}
    check("cable: the pinned free end stays within 2 mm while the right hand drags a link 5 cm", s["ok"] and dev <= 0.002 and moved >= 0.04
          and o[-1]["checks"].get(f"pinned:{end}") is True, f"end moved {1000 * dev:.2f} mm, link moved {1000 * moved:.1f} mm, pinned {pm}, "
          f"{[(x['index'], x['ok'], x.get('failed')) for x in o]}")
    check("cable: without the pin the same drag moves the free end", res["nopin"][2] > 0.004 and res["nopin"][3] >= 0.04,
          f"end moved {1000 * res['nopin'][2]:.1f} mm, link moved {1000 * res['nopin'][3]:.1f} mm")


def planner_and_outcome(scene, wd, o):
    pin = {"index": 1, "motion": "pin", "side": "left", "object": "cab_cab", "pose": str(wd / "pin.json"), "hand_scene": str(wd / "hs_left.json")}
    r1 = plan_subgoal_keyframes(scene, pin, auto_via=False)
    st = json.loads(json.dumps(r1["state"], default=float))
    rec = st["pinned"]["left"]
    check("a press pin records its force cap on the press axis (20 N on z), its travel past the contact, and holds the part",
          rec["mode"] == "press" and rec["caps"][2] == 20.0 and rec["caps"][0] == 40.0 and rec["travel_m"] > 0.003 and st["held"]["left"] == "cab_cab"
          and r1["rollout"]["wrist_force_n"]["left"]["phases"] == ["left: press cab_cab (pinned)"], json.dumps(rec))
    push = {"index": 2, "motion": "push", "side": "right", "object": "cab_cab_drawer", "joint": "cab_cab_slide", "target": -0.048,
            "pose": str(wd / "push.json"), "hand_scene": str(wd / "hs_right.json"), "hand_follows": "level"}
    r2 = plan_subgoal_keyframes(scene, push, state=json.loads(json.dumps(st)), auto_via=False)
    pins = r2["expect"].get("pinned") or []
    check("the other hand's entry expects the pinned part to stay put (and does not list it as a bystander)", len(pins) == 1 and pins[0]["body"] == "cab_cab"
          and pins[0]["tol_m"] == 0.002 and "cab_cab" not in r2["expect"].get("untouched", []), json.dumps(pins))
    caps = B.with_pin_caps([10.0, 10.0, 40.0], st)
    check("the executor keeps the pinning hand's press cap through the other hand's entries", caps == {"left": [40.0, 40.0, 20.0], "right": [10.0, 10.0, 40.0]}
          and B.with_pin_caps(None, {"pinned": {}}) is None, str(caps))
    try:
        plan_subgoal_keyframes(scene, dict(push, side="left"), state=json.loads(json.dumps(st)), auto_via=False)
        err = None
    except ValueError as e:
        err = str(e)
    check("a pinning hand is refused any motion but release, home or wait", err is not None and "pins" in err, (err or "")[:100])
    rel = plan_subgoal_keyframes(scene, {"index": 3, "motion": "release", "side": "left"}, state=json.loads(json.dumps(st)), auto_via=False)
    labs = [k["label"] for k in rel["keyframes"]]
    check("the release eases the press off before opening, and ends the pin", "left: ease off cab_cab" in labs and labs.index("left: ease off cab_cab") <
          labs.index("left: let go of cab_cab") and not rel["state"].get("pinned") and rel["state"]["held"]["left"] is None, str(labs))
    # the outcome: the delivered push re-checked, then with the cabinet moved 5 mm on a few frames, then with the pinning hand away
    model = mujoco.MjModel.from_xml_path(str(wd / "pin" / "scene_physics.xml"))
    ep = E.load_episode(wd / "pin" / "trajectory.npz")
    Q = np.array(ep["qpos"]); idx = list(range(*o[-1]["frames"])) + [o[-1]["frames"][1]]
    exp = json.loads(json.dumps(o[-1]["expect"]))
    ok0 = O.evaluate(model, Q, 30.0, idx, json.loads(json.dumps(exp)))
    j = model.joint(next(jj for jj in range(model.njnt) if model.jnt_type[jj] == mujoco.mjtJoint.mjJNT_FREE and model.body(model.jnt_bodyid[jj]).name == "cab_cab")).id
    a = int(model.jnt_qposadr[j])
    Q2 = Q.copy(); mid = idx[len(idx) // 2]
    Q2[mid:mid + 6, a] += 0.005
    ev = O.evaluate(model, Q2, 30.0, idx, json.loads(json.dumps(exp)))
    check("the outcome rejects a pinned part that moved 5 mm on a few frames (every frame is judged, not the ends)", ok0["checks"].get("pinned:cab_cab") is True
          and ev["checks"].get("pinned:cab_cab") is False and "pins" in ev["measured"].get("hint", ""), str(ev["measured"].get("pinned")))
    arm_joints = E.robot_parts(model)["left"]["arm_joints"]
    arm_qpos = [int(model.jnt_qposadr[j]) for j in arm_joints]
    Q3 = Q.copy(); Q3[idx[-1], arm_qpos] = Q[0, arm_qpos]
    ev3 = O.evaluate(model, Q3, 30.0, idx, json.loads(json.dumps(exp)))
    check("the outcome rejects a pinning hand that is off the part at the end", ev3["checks"].get("pinned:cab_cab") is False
          and ev3["measured"]["pinned"]["cab_cab"]["pinning_hand_touching_end"] is False, str(ev3["measured"].get("pinned")))


def gate_suite(scene, wd, o):
    """The pin-and-push episode as a Stage 3 deliverable, re-measured by the gate suite the finish harness runs."""
    import shutil
    ws = wd / "ws"
    (ws / "stage1").mkdir(parents=True, exist_ok=True)
    sem = {"task_prompt": "hold the cabinet and push its drawer shut", "objects": [{"id": "cab", "type": "articulated", "role": "manipulated", "properties": []}],
           "subgoals": [{"index": 1, "acting_hand": "left", "motion": "hold", "object_refs": ["cab"], "goal": "the cab is held steady by the left hand"},
                        {"index": 2, "acting_hand": "right", "motion": "push", "object_refs": ["cab"], "goal": "the drawer of the cab is pushed shut"}]}
    (ws / "stage1" / "scene_semantics.json").write_text(json.dumps(sem))
    shutil.copytree(wd / "stage2", ws / "stage2")
    cam = {"width": 640, "height": 480, "fx": 700.0, "fy": 700.0, "cx": 320.0, "cy": 240.0, "fovy_deg": 38.0,
           "pos": [0.0188, -0.1829, 0.5578], "quat_wxyz": [0.881567, 0.470897, 0.0249467, 0.021763]}
    (ws / "stage2" / "camera.json").write_text(json.dumps(cam))
    rec = json.loads((ws / "stage2" / "scene_recon.json").read_text()); rec["camera_json"] = "camera.json"
    (ws / "stage2" / "scene_recon.json").write_text(json.dumps(rec))
    s3 = ws / "stage3"
    shutil.copytree(wd / "pin", s3 / "episode")
    for f in ("trajectory.npz", "phases.json"):
        shutil.copy(s3 / "episode" / f, s3 / f)
    import imageio.v2 as iio
    w = iio.get_writer(str(ws / "human.mp4"), fps=30)
    for _ in range(10):
        w.append_data(np.full((480, 640, 3), 120, np.uint8))
    w.close()
    (s3 / "sim_video.mp4").write_bytes(b"0")
    write_stage4_config(str(ws / "stage2" / "scene_recon.json"), str(s3 / "episode"), str(s3 / "stage4_task"), human_video=str(ws / "human.mp4"), name="pin")
    subs = []
    for r in o:
        e = {"index": r["index"], "stage1_indices": [r["index"]], "motion": r["motion"], "side": r["side"], "object": r["object"], "frames": r["frames"],
             "outcome": {"ok": r["ok"], "checks": r.get("checks"), "expect": r.get("expect")}}
        if r["motion"] == "pin":
            e["pose"] = {"source": "pin_pose", "stress_tests": {"pin_pose": {"ok": True}}}
        else:                                      # the drawer's slide shut (its closed end is the low end of this model's range)
            e["goal_predicates"] = [{"class": "joint_target", "bodies": {"A": "cab_cab_drawer"}, "params": {"joint": "cab_cab_slide", "target": -0.048, "tol": 0.006}}]
        subs.append(e)
    mf = {"scene_semantic": "../stage1/scene_semantics.json", "scene_recon": "../stage2/scene_recon.json", "subgoals": subs,
          "episode": {"trajectory": "trajectory.npz", "sim_video": "sim_video.mp4"}, "stage4": {"task_dir": "stage4_task"},
          "checks": {n: {"ok": True} for n in ("episode_physics_check", "robot_self_collision_check", "arm_reach_check", "stage4_handoff_check")}}
    (s3 / "trajectory_manifest.json").write_text(json.dumps(mf))
    rep = stage3_gate_suite(str(s3), str(ws / "stage1" / "scene_semantics.json"), str(ws / "stage2" / "scene_recon.json"), draws=0)
    check("a forged one-byte video and unverified stress claims cannot pass the Stage 3 gate even after a successful primitive",
          not rep["ok"], "; ".join(rep["issues"])[:600])


def main():
    first = cabinet_case(1, 0.06, 0.07, 0.045, 0.04, 0.42, with_no_pin=True)
    cabinet_case(2, 0.05, 0.06, 0.04, 0.0, 0.47)
    if first:
        planner_and_outcome(*first)
        gate_suite(*first)
    cable_case()
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}  ({TMP})")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
