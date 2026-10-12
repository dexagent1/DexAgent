"""Stage 3 gate suite on a shipped episode: the verified cup episode passes, and every copy with one low-level error
injected (a teleport, a joint jump, the hand mounted off its arm flange, the arm through the table, fingers through the cup, the cup
moved by nothing, a float, a label gap, a stronger or wider robot, contacts switched off) is rejected by the check that
owns that error.

    MUJOCO_GL=egl python tool_library/tests/test_stage3_gate_suite.py
"""
import json
import os
import re
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.common import examples as EX  # noqa: E402
from tool_library.verifiers.stage3_gate_suite import stage3_gate_suite  # noqa: E402

FAILS = 0
EXAMPLE = ROOT / "examples/default/place_cup_in_bowl"
VIDEO = ROOT / "stage1_semantic_understanding/examples/place_cup_in_bowl/video.mp4"


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def flagged(rep):
    """Names of the checks that raised a blocking issue."""
    return sorted({i.split(":")[0].split("[")[0] for i in rep["issues"]})


def held_at_end_cases():
    """The bodies the end-state audit lets a hand end holding, followed through every entry (no physics)."""
    from tool_library.verifiers.stage3_gate_suite import _held_at_end
    four_cups = {"subgoals": [
        {"motion": "grasp_lift", "side": "right", "object": "teal"}, {"motion": "grasp_lift", "side": "left", "object": "near"},
        {"motion": "carry", "side": "right", "object": "teal"}, {"motion": "grasp_more", "side": "right", "object": "green"},
        {"motion": "grasp_more", "side": "left", "object": "far"},
        {"motion": "both", "hands": {"left": {"motion": "carry", "object": "far"}, "right": {"motion": "carry", "object": "green"}}}]}
    check("two cups in each hand at the end (grasp_more) are all held", sorted(_held_at_end(four_cups)) == ["far", "green", "near", "teal"],
          str(_held_at_end(four_cups)))
    placed = {"subgoals": [{"motion": "grasp_lift", "side": "right", "object": "cup"}, {"motion": "place", "side": "right", "object": "cup"}]}
    check("a placed object is not held", _held_at_end(placed) == [], str(_held_at_end(placed)))
    held_by_other = {"subgoals": [{"motion": "hold", "side": "left", "object": "bottle"},
                                  {"motion": "grasp_lift", "side": "right", "object": "cap"}, {"motion": "place", "side": "right", "object": "cap"}]}
    check("an object the other hand still holds at the end stays held", _held_at_end(held_by_other) == ["bottle"], str(_held_at_end(held_by_other)))
    released = {"subgoals": [{"motion": "grasp_lift", "side": "right", "object": "a"}, {"motion": "grasp_more", "side": "right", "object": "b"},
                             {"motion": "place", "side": "right", "object": "b"}, {"motion": "push", "side": "left", "object": "c", "release": True}]}
    check("putting one of two objects down keeps the other; a push holds nothing", _held_at_end(released) == ["a"], str(_held_at_end(released)))


def main():
    held_at_end_cases()
    if not (EXAMPLE / "stage3/trajectory.npz").exists():
        print("SKIP: examples/default/place_cup_in_bowl has no Stage 3"); return 0
    tmp = Path(tempfile.mkdtemp(prefix="dexagent_gate_suite_", dir=os.environ.get("TMPDIR")))
    run = tmp / "run"
    for st in (1, 2, 3):
        EX.reuse_example(str(EXAMPLE), st, str(run / f"stage{st}"), video=str(VIDEO))
    ws, sem, recon = run / "stage3", run / "stage1/scene_semantics.json", run / "stage2/scene_recon.json"
    scene = ws / "scene_executed.xml"
    good_traj, good_scene = (ws / "trajectory.npz").read_bytes(), scene.read_text()

    rep = stage3_gate_suite(str(ws), str(sem), str(recon), draws=3)
    check("the verified episode passes every check, the Stage 4 gates included", rep["ok"], "; ".join(rep["issues"])[:400])
    check("the Stage 4 randomization and physics replay gates ran",
          {"stage4_randomization_gate", "stage4_physics_replay_gate"} <= set(rep["reports"]), str(sorted(rep["skipped"])))

    m = mujoco.MjModel.from_xml_path(str(scene)); d = mujoco.MjData(m)
    Z = dict(np.load(ws / "trajectory.npz", allow_pickle=True))
    Q = np.asarray(Z["qpos"], float); T = len(Q); k = T // 2
    adr = lambda name: int(m.jnt_qposadr[m.joint(name).id])
    cup = next(m.joint(j).name for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and "cup" in m.joint(j).name)
    side = "left"                                                      # the hand that carries the cup
    fingers = [m.joint(j).name for j in range(m.njnt)
               if m.joint(j).name.startswith(f"robot_{side}_") and re.search(r"(index|middle|ring|thumb)_(MCP_FE|PIP|DIP|IP)", m.joint(j).name)]

    def cup_moving():
        P = []
        for t in range(T):
            d.qpos[:] = Q[t]; mujoco.mj_kinematics(m, d); P.append(d.xpos[m.joint(cup).bodyid].copy())
        return np.linalg.norm(np.diff(np.array(P), axis=0), axis=1) > 1e-4

    def episode(q=None, labels=None):
        z = dict(Z); z["qpos"] = Q if q is None else q
        if labels is not None:
            z["labels"] = np.array(labels)
        np.savez_compressed(ws / "trajectory.npz", **z)

    def tampered(fn):
        r = ET.fromstring(good_scene); fn(r); scene.write_text(ET.tostring(r, encoding="unicode"))

    moving = cup_moving()
    still = next(t for t in range(5, T - 50) if not moving[t:t + 45].any())     # a window where nothing touches the cup
    labels = [str(x) for x in Z["labels"]]

    def q_with(fn):
        q = Q.copy(); fn(q); return q

    def stronger(r):
        for a in list(r.iter("general")) + list(r.iter("position")):
            if (a.get("name") or "").startswith(f"robot_{side}_") and "index" in (a.get("name") or ""):
                kp = float((a.get("gainprm") or a.get("kp") or "1").split()[0]) * 10
                a.set("gainprm", f"{kp:g}"); a.set("biasprm", f"0 {-kp:g} 1") if a.tag == "general" else a.set("kp", f"{kp:g}")
                a.set("forcerange", "-50 50"); a.set("forcelimited", "true")

    def wider(r):
        for jt in r.iter("joint"):
            if re.match(r"robot_.*_arm_dof_joint[1-6]$", jt.get("name") or ""):
                jt.set("range", "-6.28 6.28")

    def no_contacts(r):
        for g in r.iter("geom"):
            if (g.get("name") or "").startswith("robot_"):
                g.set("contype", "0"); g.set("conaffinity", "0")
        for c in r.iter("contact"):
            for p in [p for p in c if p.tag == "pair" and "robot_" in p.get("geom1", "") + p.get("geom2", "")]:
                c.remove(p)

    cup_body = int(m.jnt_bodyid[m.joint(cup).id])
    cup_geoms = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == cup_body]
    hand_geoms = [g for g in range(m.ngeom) if (m.body(int(m.geom_bodyid[g])).name or "").startswith(f"robot_{side}_{side}_")]

    def cup_depth(q):
        d.qpos[:] = q; mujoco.mj_kinematics(m, d)
        return -min(mujoco.mj_geomDistance(m, d, g, c, 0.05, None) for g in hand_geoms for c in cup_geoms)

    def into_cup(q):
        # the carrying hand's finger flexions set, joint by joint, to the value in their range that takes the fingers
        # deepest into the held cup (a pinch from above: closing every joint fully curls the fingertips away from it,
        # into nothing), held for 40 frames; well past episode_physics_check's 8 mm
        x = q[k].copy()
        for _ in range(2):
            for f in fingers:
                tries = [(cup_depth(np.where(np.arange(len(x)) == adr(f), v, x)), v) for v in np.linspace(*m.jnt_range[m.joint(f).id], 9)]
                best = max(tries)
                if best[0] > cup_depth(x):
                    x[adr(f)] = best[1]
        assert cup_depth(x) > 0.012, f"the fingers cannot be driven into the cup ({1000 * cup_depth(x):.1f} mm)"
        q[k:k + 40, [adr(f) for f in fingers]] = x[[adr(f) for f in fingers]]

    def flange_offset(r):
        # the rigid robot has no wrist joints to slide: the hand is a fixed child of its arm flange, so "the hand off its
        # arm" is a hand body moved 3 cm on the flange (a changed mount)
        b = r.find(f".//body[@name='robot_{side}_{side}_base_yaw']")
        x, y, z = (float(v) for v in (b.get("pos") or "0 0 0").split())
        b.set("pos", f"{x + 0.03:g} {y:g} {z:g}")

    faults = [  # (name, trajectory, scene edit, checks of which at least one must flag it)
        ("cup teleports 5 cm", q_with(lambda q: q.__setitem__((slice(k, None), adr(cup)), q[k:, adr(cup)] + 0.05)), None,
         {"object_teleport_audit", "episode_physics_check", "dynamics_consistency_audit"}),
        ("arm joint jumps 0.6 rad for one frame", q_with(lambda q: q.__setitem__((k, adr(f"robot_{side}_arm_dof_joint1")), q[k, adr(f"robot_{side}_arm_dof_joint1")] + 0.6)), None,
         {"robot_joint_jump_audit", "joint_acceleration_audit"}),
        ("shoulder dives into the table", q_with(lambda q: q.__setitem__((slice(k, k + 40), adr(f"robot_{side}_arm_dof_joint2")), q[k:k + 40, adr(f"robot_{side}_arm_dof_joint2")] + 0.9)), None,
         {"swept_collision_audit", "episode_physics_check", "table_workspace_bounds_audit", "arm_reach_check"}),
        ("fingers close through the cup", q_with(into_cup), None,
         {"object_visual_penetration_audit", "episode_physics_check", "swept_collision_audit", "hand_object_relative_jitter_audit"}),
        ("cup slides 10 cm with nothing touching it", q_with(lambda q: q.__setitem__((slice(None), adr(cup)), q[:, adr(cup)] + 0.10 * np.clip((np.arange(T) - still) / 40.0, 0, 1))), None,
         {"contactless_motion_audit", "dynamics_consistency_audit", "end_state_rest_audit", "stage1_goal_gate"}),
        ("cup floats 6 cm at the end", q_with(lambda q: q.__setitem__((slice(T - 40, None), adr(cup) + 2), q[T - 40:, adr(cup) + 2] + 0.06)), None,
         {"end_state_rest_audit", "free_fall_plausibility_audit", "dynamics_consistency_audit", "stage1_goal_gate"}),
    ]
    for name, q, edit, owners in faults:
        episode(q)
        r = stage3_gate_suite(str(ws), str(sem), str(recon), draws=0)
        f = flagged(r)
        check(f"rejected: {name}", not r["ok"] and bool(owners & set(f)), f"flagged by {f}")
    episode()
    lab = list(labels); lab[k:k + 20] = [""] * 20
    episode(labels=lab)
    r = stage3_gate_suite(str(ws), str(sem), str(recon), draws=0)
    check("rejected: 20 frames without a label", not r["ok"] and bool({"uncovered_frames_check", "phase_label_continuity_audit"} & set(flagged(r))), str(flagged(r)))
    episode()
    for name, fn, owners in (("hand mounted 3 cm off its arm flange", flange_offset, {"rigid_arm_hand_check"}),
                             ("finger servos 10x stronger", stronger, {"actuator_limit_integrity_check", "robot_model_integrity_check"}),
                             ("arm joint ranges widened to +-2 pi", wider, {"robot_joint_range_check", "robot_model_integrity_check"}),
                             ("robot contacts switched off", no_contacts, {"contact_exclusion_audit", "robot_model_integrity_check"})):
        tampered(fn)
        r = stage3_gate_suite(str(ws), str(sem), str(recon), draws=0)
        check(f"rejected: {name}", not r["ok"] and bool(owners & set(flagged(r))), f"flagged by {flagged(r)}")
        scene.write_text(good_scene)
    (ws / "trajectory.npz").write_bytes(good_traj)
    shutil.rmtree(tmp, ignore_errors=True)
    print("\nALL PASS" if not FAILS else f"\n{FAILS} FAILED")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
