"""Tests of physics_informed_retargeting (the first Stage 3 method module), its optimizer adapter common/spider_bridge.py
and the tracking references the method modules exchange (common/tracking.py).

Unit tests (CPU, synthetic, no optimizer needed): bench scenes from setup/ with a parametric box of several sizes and
poses, a reference that reaches, closes and lifts it; the optimizer's robot and scene compile; bench <-> optimizer
conversions are exact (hand forward kinematics and object poses agree); resampling keeps unit quaternions; the mandatory
preparation grounds the object, puts fingertips on its surface one pad radius out and puts the thumb opposite the
fingers; the force-equivalent command; Stage 1 expectations; the human hand mapping recovers a known pose; the agent's
best attempt becomes a tracking reference and a track_reference plan; without the optimizer the module reports itself
unavailable; with the optimizer replaced by a stand-in that returns the reference, the whole module runs and writes its
reference and plan.

Integration test (skipped without the optimizer's checkout, its python and a GPU: set DEXAGENT_SPIDER_DIR and
DEXAGENT_SPIDER_PYTHON, or models.spider_dir / environments.spider in dexagent.yaml): the optimizer on the synthetic
lift with a small sample budget, the result re-simulated on the bench.

    python tool_library/tests/test_physics_informed_retargeting.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.common import episode as E  # noqa: E402
from tool_library.common import hand as H  # noqa: E402
from tool_library.common import spider_bridge as SB  # noqa: E402
from tool_library.skills.stage_on_bench import stage_on_bench  # noqa: E402
from tool_library.common import tracking as TR  # noqa: E402
from tool_library.skills import physics_informed_retargeting as PIR  # noqa: E402
from tool_library.skills.physics_informed_retargeting import portable_scene  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dexagent_spider_", dir=os.environ.get("TMPDIR")))
SETUP = ROOT / "setup"
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


BOX_XML = """<mujoco model="{name}">
  <worldbody>
    <body name="{name}">
      <freejoint name="{name}_free"/>
      <geom name="{name}_g" type="box" size="{hx} {hy} {hz}" pos="0 0 {hz}" mass="{mass}" rgba="0.2 0.5 0.8 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def make_bench(tag, half, x, y, yaw_deg, second=True):
    """A box (half extents `half`) on the bench at (x, y, yaw) and, optionally, a second box as a bystander."""
    d = TMP / tag
    d.mkdir(parents=True, exist_ok=True)
    (d / "box.xml").write_text(BOX_XML.format(name="box", hx=half[0], hy=half[1], hz=half[2], mass=0.08))
    assets = [{"name": "box", "xml": str(d / "box.xml"), "x": x, "y": y, "yaw_deg": yaw_deg}]
    if second:
        (d / "other.xml").write_text(BOX_XML.format(name="other", hx=0.03, hy=0.03, hz=0.02, mass=0.1))
        assets.append({"name": "other", "xml": str(d / "other.xml"), "x": x + 0.15, "y": y + 0.05, "yaw_deg": 0.0})
    stage_on_bench(str(SETUP / "scene.xml"), assets=assets, out_xml=str(d / "scene.xml"), rig={"origin": [0, 0, 0], "R": np.eye(3).tolist()})
    return mujoco.MjModel.from_xml_path(str(d / "scene.xml")), d / "scene.xml"


def box_body(model, name="box"):
    return next(model.body(b).name for b in range(model.nbody) if model.body(b).name.startswith(f"{name}_") and model.body_parentid[b] == 0)


def lift_reference(model, side="right", T=90, rise=0.08):
    """Keyframe-free synthetic plan: the hand comes down over the box, closes, lifts it rigidly (as a plan carries it)."""
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, 0)
    mujoco.mj_forward(model, d)
    q0 = d.qpos.copy()
    cols = SB.hand_columns(model)[side]
    free = E.object_free_joints(model)
    b = box_body(model)
    a = free[b]
    p_obj = q0[a:a + 3].copy()
    Q = np.tile(q0, (T, 1))
    w0 = q0[cols["wrist"][:3]].copy()
    grasp = p_obj + np.array([0.0, -0.02, 0.12])
    labels = []
    for t in range(T):
        u = t / (T - 1)
        if u < 0.4:
            s = u / 0.4
            Q[t, cols["wrist"][:3]] = (1 - s) * w0 + s * grasp; labels.append(f"{side}: reach for {b} #1")
        elif u < 0.6:
            Q[t, cols["wrist"][:3]] = grasp
            Q[t, cols["fingers"]] = (u - 0.4) / 0.2 * 0.6; labels.append(f"{side}: close on {b} #1")
        else:
            s = (u - 0.6) / 0.4
            Q[t, cols["wrist"][:3]] = grasp + [0, 0, s * rise]
            Q[t, cols["fingers"]] = 0.6
            Q[t, a + 2] = p_obj[2] + s * rise; labels.append(f"{side}: lift {b} #1")
    return Q, labels


# --------------------------------------------------------------------------------------------------- robot --
def test_robot_xml():
    for sides, nq in ((["right"], 28), (["left"], 28), (["right", "left"], 56)):
        x = SB.write_robot_xml(SETUP, sides, TMP / "robot" / SB.robot_xml_name(sides))
        m = mujoco.MjModel.from_xml_path(x)
        caps = sorted({round(float(m.actuator_forcerange[a][1]), 3) for a in range(m.nu)})
        grav = all(float(m.body_gravcomp[b]) == 1.0 for b in range(1, m.nbody))
        wrist_kp = {round(float(m.actuator_gainprm[a][0])) for a in range(m.nu) if "_pos_" in m.actuator(a).name or "_rot_" in m.actuator(a).name}
        check(f"robot xml {'+'.join(sides)}: nq {m.nq}, nu {m.nu}, caps {caps}, gravcomp, the bench's wrist stiffness {wrist_kp}",
              m.nq == nq and m.nu == nq and caps == sorted({H.FINGER_FORCE_NM, H.WRIST_FORCE_N, H.WRIST_TORQUE_NM}) and grav
              and wrist_kp == {round(SB.BENCH_WRIST_KP)})
        names = [m.joint(j).name for j in range(m.njnt)]
        first = sides[0]
        check(f"robot xml {'+'.join(sides)}: SPIDER's qpos order (right hand first, wrist then fingers)",
              names[:6] == [f"{first}_{k}" for k in SB.WRIST_SUFFIX] and all(f"{s}_palm" in [m.site(i).name for i in range(m.nsite)] for s in sides))


# ------------------------------------------------------------------------------------------ scene + conversions --
CASES = [("small", (0.02, 0.02, 0.03), 0.12, 0.42, 0.0), ("wide", (0.045, 0.03, 0.025), 0.18, 0.50, 35.0),
         ("tall", (0.025, 0.025, 0.06), 0.08, 0.38, -60.0)]


def test_scene_and_conversions():
    for tag, half, x, y, yaw in CASES:
        model, scene = make_bench(tag, half, x, y, yaw)
        Q, labels = lift_reference(model)
        roles = SB.infer_roles(model, Q, labels)
        b = box_body(model)
        check(f"{tag}: roles = right hand on the box, the other box an obstacle", roles["sides"] == ["right"] and roles["objects"] == {"right": b}
              and box_body(model, "other") in roles["statics"], str(roles["objects"]))
        paths = SB.dataset_paths(TMP / f"ds_{tag}", "right", tag)
        rx = SB.write_robot_xml(SETUP, roles["sides"], paths["robot_xml"])
        sc = SB.write_scene(model, Q[0], roles, rx, paths["robot_task"], paths["assets"] / "objects")
        sm = mujoco.MjModel.from_xml_path(sc["scene"])
        check(f"{tag}: SPIDER scene compiles (nq {sm.nq} = 28 + 7, {sm.npair} explicit pairs, obstacle kept)",
              sm.nq == 35 and sm.nu == 28 and sm.npair > 0 and sc["layout"]["statics"] == [box_body(model, "other")])
        mujoco.MjModel.from_xml_path(sc["scene_eq"])
        S = SB.bench_to_spider(model, Q, sm, sc["layout"])
        back = SB.spider_to_bench(model, np.tile(Q[0], (len(Q), 1)), sm, sc["layout"], S)
        cols = SB.hand_columns(model)["right"]
        a = E.object_free_joints(model)[b]
        idx = cols["wrist"] + cols["fingers"] + list(range(a, a + 7))
        check(f"{tag}: bench -> SPIDER -> bench round trip is exact", np.abs(back[:, idx] - Q[:, idx]).max() < 1e-12)
        # the same hand in the same frame: fingertip sites and the object body agree on every frame
        d, ds = mujoco.MjData(model), mujoco.MjData(sm)
        err_tip, err_obj = 0.0, 0.0
        bid, sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b), mujoco.mj_name2id(sm, mujoco.mjtObj.mjOBJ_BODY, "right_object")
        for t in range(0, len(Q), 7):
            d.qpos[:] = Q[t]; mujoco.mj_kinematics(model, d)
            ds.qpos[:] = S[t]; mujoco.mj_kinematics(sm, ds)
            for f in SB.FINGERS:
                p1 = d.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"robot_right_right_{f}_tip")]
                p2 = ds.site_xpos[mujoco.mj_name2id(sm, mujoco.mjtObj.mjOBJ_SITE, f"right_{f}_tip")]
                err_tip = max(err_tip, float(np.linalg.norm(p1 - p2)))
            err_obj = max(err_obj, float(np.linalg.norm(d.xpos[bid] - ds.xpos[sid])), float(np.abs(d.xmat[bid] - ds.xmat[sid]).max()))
        check(f"{tag}: fingertips and object pose agree between bench and SPIDER scene", err_tip < 1e-6 and err_obj < 1e-6,
              f"(tip {err_tip:.1e} m, object {err_obj:.1e})")
        # same mass and inertia
        ja_b, ja_s = int(model.jnt_dofadr[int(model.body_jntadr[bid])]), int(sm.jnt_dofadr[int(sm.body_jntadr[sid])])
        check(f"{tag}: the object keeps its mass, inertia and free-joint dynamics (armature, damping, friction loss)",
              abs(float(sm.body_mass[sid]) - float(model.body_mass[bid])) < 1e-9
              and np.allclose(np.sort(sm.body_inertia[sid]), np.sort(model.body_inertia[bid]), rtol=1e-6, atol=1e-12)
              and np.allclose(sm.dof_armature[ja_s:ja_s + 6], model.dof_armature[ja_b:ja_b + 6])
              and np.allclose(sm.dof_frictionloss[ja_s:ja_s + 6], model.dof_frictionloss[ja_b:ja_b + 6]))


def test_resample():
    t = np.linspace(0, 1, 11)
    q = np.zeros((11, 8))
    ang = np.linspace(0, np.pi / 2, 11)
    q[:, 0] = t
    q[:, 4] = np.cos(ang / 2); q[:, 7] = np.sin(ang / 2)
    q[5:, 4:] *= -1                                      # a sign flip of the same rotations
    r = SB.resample(q, t, np.linspace(0, 1, 37), quat_adr=[4])
    norms = np.linalg.norm(r[:, 4:], axis=1)
    check("resample: joints linear, quaternions unit, ends exact", np.allclose(r[:, 0], np.linspace(0, 1, 37)) and np.allclose(norms, 1.0)
          and np.allclose(r[0], q[0]) and np.allclose(np.abs(r[-1, 4:]), np.abs(q[-1, 4:])))


# ---------------------------------------------------------------------------------------------- preparation --
def test_prep():
    for tag, half, x, y, yaw in CASES:
        model, scene = make_bench(tag, half, x, y, yaw, second=False)
        Q, labels = lift_reference(model)
        roles = SB.infer_roles(model, Q, labels)
        paths = SB.dataset_paths(TMP / f"dsp_{tag}", "right", tag)
        rx = SB.write_robot_xml(SETUP, roles["sides"], paths["robot_xml"])
        sc = SB.write_scene(model, Q[0], roles, rx, paths["robot_task"], paths["assets"] / "objects")
        sm = mujoco.MjModel.from_xml_path(sc["scene"])
        S = SB.bench_to_spider(model, Q, sm, sc["layout"])
        kp = SB.keypoints_from_spider_qpos(sm, S, ["right"])
        lift = 0.02
        kp["qpos_obj_right"][:, 2] += lift                # an ungrounded export: everything floats 2 cm
        kp["qpos_wrist_right"][:, 2] += lift
        kp["qpos_finger_right"][:, :, 2] += lift
        # fingertips 1 cm off two faces of the box in the frames it moves: index/middle on +x, thumb misplaced on +x too
        win = SB.contact_window(kp["qpos_obj_right"][:, :3], 30.0)
        R = np.zeros(9)
        for t in range(win[0], win[1] + 1):
            p = kp["qpos_obj_right"][t, :3]
            mujoco.mju_quat2Mat(R, kp["qpos_obj_right"][t, 3:]); Rm = R.reshape(3, 3)
            c = p + Rm @ np.array([0, 0, half[2]])
            kp["qpos_finger_right"][t, 1, :3] = c + Rm @ np.array([half[0] + 0.01, 0.008, 0.0])
            kp["qpos_finger_right"][t, 2, :3] = c + Rm @ np.array([half[0] + 0.01, -0.008, 0.0])
            kp["qpos_finger_right"][t, 0, :3] = c + Rm @ np.array([half[0] + 0.012, 0.0, 0.005])
            for i in (3, 4):
                kp["qpos_finger_right"][t, i, :3] = c + np.array([0, 0, 0.3])
        out, rep = SB.prep_keypoints(kp, sm, ["right"], 30.0)
        check(f"{tag}: ground removes the 2 cm float", abs(rep["dz"] + lift) < 2e-3, f"(dz {rep['dz']:+.4f})")
        t = (win[0] + win[1]) // 2
        p = out["qpos_obj_right"][t, :3]; mujoco.mju_quat2Mat(R, out["qpos_obj_right"][t, 3:]); Rm = R.reshape(3, 3)
        c = p + Rm @ np.array([0, 0, half[2]])
        loc = lambda i: Rm.T @ (out["qpos_finger_right"][t, i, :3] - c)
        pad = rep["sides"]["right"]["pad_m"]
        gap = [loc(i)[0] - half[0] for i in (1, 2)]
        check(f"{tag}: index and middle on the surface, one pad radius out", all(abs(g - pad) < 1.5e-3 for g in gap), f"({gap[0]*1000:.1f}, {gap[1]*1000:.1f} mm, pad {pad*1000:.1f})")
        th = loc(0)
        check(f"{tag}: the thumb is re-derived on the opposite face", th[0] < -half[0] + 1e-3 and abs(-th[0] - half[0] - pad) < 2e-3,
              f"(thumb x {th[0]*1000:.1f} mm vs face {-half[0]*1000:.1f})")
        check(f"{tag}: contacts and opposition counted", rep["sides"]["right"]["opposition_frames"] > 0 and out["contact_right"][t, :3].all()
              and not out["contact_right"][t, 3:].any())


def test_force_equivalent():
    q = np.array([0.10, 0.50]); c = np.array([0.12, 0.40])
    cmd = SB.force_equivalent_command(q, c, [1000.0, 300.0], [10000.0, 300.0])
    check("force-equivalent command: same servo force at the bench's stiffness", np.allclose(cmd, [0.102, 0.40])
          and np.allclose(10000 * (cmd[0] - q[0]), 1000 * (c[0] - q[0])))


def test_mjwp_settings():
    r, x, h = (SB.mjwp_settings(p) for p in ("refine", "explore", "retarget"))
    o = SB.mjwp_settings("refine", {"num_samples": 64})
    check("mjwp settings: refine samples tighter than explore, retarget explores widely, overrides win, the bench's requirements kept",
          r["joint_noise_scale"] < x["joint_noise_scale"] < h["joint_noise_scale"] and r["joint_rew_scale"] > h["joint_rew_scale"]
          and o["num_samples"] == 64 and all(v["njmax_per_env"] >= 351 and v["sanity_check_seconds"] == 0.0 for v in (r, x, h)))
    try:
        SB.mjwp_settings("nope"); bad = False
    except ValueError:
        bad = True
    check("mjwp settings: an unknown profile is an error", bad)


def test_stage1_expect():
    model, scene = make_bench("expect", (0.02, 0.02, 0.03), 0.12, 0.42, 0.0)
    Q, labels = lift_reference(model, rise=0.08)
    b = box_body(model)
    e1 = SB.stage1_expect({"motion": "grasp_lift", "object_refs": ["box"]}, model, Q, [0, len(Q) - 1], "right", b)
    e2 = SB.stage1_expect({"motion": "carry", "object_refs": ["box"]}, model, Q, [0, len(Q) - 1], "right", b)
    e3 = SB.stage1_expect({"motion": "place", "object_refs": ["box"]}, model, Q, [0, len(Q) - 1], "right", b)
    e4 = SB.stage1_expect({"motion": "unscrew", "object_refs": ["box"]}, model, Q, [0, len(Q) - 1], "right", b)
    end = SB.body_track(model, Q[-1:], b)[0][0]
    check("stage1 expect: grasp_lift = half the reference rise (2-5 cm), held, no slip; bystander untouched",
          e1["rise_m"] == 0.04 and e1["in_hand"] and e1["no_slip"] and e1["untouched"] == [box_body(model, "other")])
    check("stage1 expect: carry / place targets are where the reference ends", np.allclose(e2["target_pos"], end, atol=1e-4)
          and e3["released"] and e3["at_rest"] and e3["upright"] and e3["target_axes"] == "xy")
    check("stage1 expect: a motion the backend does not produce gives None", e4 is None)


def test_human_mapping():
    """The Sharpa's own keypoints moved by a known root pose map back to that pose, at scale 1."""
    from tool_library.common import human_prior as HP
    from scipy.spatial.transform import Rotation as Rot
    for side in ("right", "left"):
        xml = SETUP / "robot" / f"sharpa_{side}.xml"
        ref = HP.sharpa_reference(xml, side)
        T = 4
        R_true = Rot.from_euler("xyz", [[0.3, -0.2, 1.0], [0.1, 0.4, -0.5], [-0.6, 0.2, 0.2], [0, 0, 0]]).as_matrix()
        p_true = np.array([[0.1, 0.4, 0.2], [0.0, 0.5, 0.1], [-0.2, 0.3, 0.3], [0, 0, 0]])
        kp = {"wrist": np.zeros((T, 3))} | {f: np.zeros((T, 4, 3)) for f in SB.FINGERS}
        for t in range(T):
            kp["wrist"][t] = p_true[t] + R_true[t] @ ref["kp"]["wrist"]
            for f in SB.FINGERS:
                kp[f][t] = p_true[t] + ref["kp"][f] @ R_true[t].T
        hk = SB.human_keypoints({side: kp}, {side: xml})[side]
        err_p = np.abs(hk["root"][:, :3, 3] - p_true).max()
        err_R = np.abs(hk["root"][:, :3, :3] - R_true).max()
        m = mujoco.MjModel.from_xml_path(str(xml)); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        tip0 = d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{side}_index_tip")]
        check(f"human mapping {side}: the root pose comes back exactly, the tips at scale 1", err_p < 1e-9 and err_R < 1e-9
              and np.linalg.norm(hk["tips"][3, 1] - tip0) < 1e-6, f"(pos {err_p:.1e}, rot {err_R:.1e})")


# ------------------------------------------------------------------------------------------------ workspace --
def _attempt_workspace(tag, model, Q, labels):
    """A Stage 3 workspace holding three executor folders like execute_subgoal_plan writes them: the one whose entries
    cover the most Stage 1 subgoals ('episode') must be chosen."""
    ws = TMP / tag / "stage3"
    jn = np.array(E.joint_names(model))
    half = len(Q) // 2
    b = box_body(model)
    for name, n_sub, ok in (("episode_try1", 1, 0), ("episode", 2, 1), ("ek3", 1, 1)):
        d = ws / name; d.mkdir(parents=True)
        sub = np.r_[np.ones(half, int), np.full(len(Q) - half, n_sub, int)]
        E.save_episode(d / "trajectory_proposal.npz", Q + 0.001, [f"{l} #1" for l in labels], 30.0, subgoal=sub, joint_names=jn, reference=Q)
        rows = [{"index": 1, "tag": "subgoal_01", "motion": "grasp_lift", "side": "right", "object": b, "ok": ok > 0,
                 "stage1_indices": [1], "frames": [0, half - 1 if n_sub == 2 else len(Q) - 1]}]
        if n_sub == 2:
            rows.append({"index": 2, "tag": "subgoal_02", "motion": "carry", "side": "right", "object": b, "ok": ok > 1,
                         "stage1_indices": [2], "frames": [half, len(Q) - 1]})
        (d / "outcomes.json").write_text(json.dumps(rows))
    return ws


def _semantics(path, model):
    b = box_body(model)
    sem = {"task_prompt": "lift the box", "hands": {"used": ["right"]},
           "subgoals": [{"index": 1, "motion": "grasp_lift", "acting_hand": "right", "object_refs": [b]},
                        {"index": 2, "motion": "carry", "acting_hand": "right", "object_refs": [b]}]}
    Path(path).write_text(json.dumps(sem))
    return sem


def test_reference_and_plan():
    model, scene = make_bench("ws", (0.02, 0.02, 0.03), 0.12, 0.42, 0.0)
    Q, labels = lift_reference(model)
    ws = _attempt_workspace("ws", model, Q, labels)
    sem = _semantics(TMP / "ws" / "sem.json", model)
    path, info = TR.reference_from_attempt(ws, model, TMP / "ws" / "ref.npz", sem=sem)
    ref = TR.load_reference(path, model)
    check("tracking reference: the attempt covering most Stage 1 subgoals, its commanded reference, one segment per entry",
          info["attempt"].endswith("/episode") and np.allclose(ref["qpos"], Q) and [s["stage1_indices"] for s in ref["segments"]] == [[1], [2]]
          and info["stage1_missing"] == [] and all("#" not in l for l in ref["labels"]), json.dumps(info)[:200])
    plan = TR.write_plan(TMP / "ws" / "plan.json", path, model, sem)
    e1, e2 = plan["subgoals"]
    check("plan: one track_reference entry per segment, its frames, side, object and the motion's own expectations",
          [e["motion"] for e in plan["subgoals"]] == ["track_reference"] * 2 and e1["frames"][0] == 0 and e2["frames"][1] == len(Q) - 1
          and e1["side"] == "right" and e1["object"] == box_body(model) and e1["expect"].get("in_hand") and "rise_m" in e1["expect"]
          and not Path(e1["reference"]).is_absolute(), json.dumps(e1)[:300])
    ex = TR.tracking_expect(model, Q, e2["frames"], obj=box_body(model))
    check("tracking expectation: the object ends where the reference ends it", np.allclose(ex["target_pos"], Q[-1, E.object_free_joints(model)[box_body(model)]:][:3], atol=1e-5))
    from tool_library.common.subgoal import SubgoalPlanner
    sg = dict(e1, reference=str(TMP / "ws" / e1["reference"]))
    sp = SubgoalPlanner(model, Q[0], fps=30.0)
    out = sp.run(sg)
    n = e1["frames"][1] - e1["frames"][0] + 1
    check("executor planner: track_reference gives one keyframe per row (plus the start), one phase per label run",
          len(out["keyframes"]) == n + 1 and len(out["phases"]) == len(dict.fromkeys(ref["labels"][:n])) and out["expect"]["object"] == box_body(model),
          f"({len(out['keyframes'])} keyframes, {len(out['phases'])} phases)")
    (TMP / "elsewhere").mkdir()
    m2 = mujoco.MjModel.from_xml_path(portable_scene(scene, TMP / "elsewhere" / "scene.xml"))
    check("portable scene loads from another folder", m2.nq == model.nq)


def test_unavailable():
    model, scene = make_bench("na", (0.02, 0.02, 0.03), 0.12, 0.42, 0.0)
    sem_p = TMP / "na" / "sem.json"; _semantics(sem_p, model)
    recon_p = TMP / "na" / "scene_recon.json"; recon_p.write_text(json.dumps({"scene_xml": "scene.xml", "objects": []}))
    r = PIR.physics_informed_retargeting(TMP / "na" / "stage3", sem_p, recon_p, spider_dir=str(TMP / "no_checkout"), spider_python=None)
    check("without the optimizer's checkout the module reports itself unavailable (the next method runs), not a crash",
          r["ok"] is False and r.get("unavailable") is True and "checkout" in r.get("error", ""), r.get("error", "")[:120])


def test_module_with_stand_in_optimizer():
    """The whole module with the GPU optimizer replaced by a stand-in that returns the kinematic trajectory it was given
    as its result: inputs written, result read back and converted, checked on the bench's runtime, reference and plan."""
    from unittest.mock import patch
    model, scene = make_bench("mod", (0.02, 0.02, 0.03), 0.12, 0.42, 0.0, second=False)
    Q, labels = lift_reference(model, T=45)
    ws = _attempt_workspace("mod", model, Q, labels)
    sem_p = TMP / "mod" / "sem.json"; _semantics(sem_p, model)
    recon_p = TMP / "mod" / "scene_recon.json"; recon_p.write_text(json.dumps({"scene_xml": "scene.xml", "objects": []}))

    def stand_in(spider_dir, spider_python, ds_dir, embodiment, task, overrides=None, profile="refine", device="cuda:0", log=None, timeout=7200):
        p = SB.dataset_paths(ds_dir, embodiment, task)
        z = np.load(p["trial"] / "trajectory_kinematic.npz")
        sm = mujoco.MjModel.from_xml_path(str(p["robot_task"] / "scene.xml"))
        S = z["qpos"]
        C = np.zeros((len(S), sm.nu))
        for a in range(sm.nu):                     # position servos commanded where the joints are: no push
            C[:, a] = S[:, int(sm.jnt_qposadr[int(sm.actuator_trnid[a][0])])]
        out = p["trial"] / "trajectory_mjwp.npz"
        np.savez(out, qpos=S, ctrl=C, time=np.arange(1, len(S) + 1) * 0.02)
        return str(out)
    with patch.object(SB, "check_spider", lambda *a: None), patch.object(SB, "run_spider_mjwp", stand_in):
        r = PIR.physics_informed_retargeting(ws, sem_p, recon_p, spider_dir="x", spider_python="x", scene=str(scene), profiles=["refine"])
    ok = r.get("ok") and Path(r["reference"]).is_file() and Path(r["plan"]).is_file()
    plan = json.loads(Path(r["plan"]).read_text()) if ok else {}
    ref = TR.load_reference(r["reference"], model) if ok else {}
    check("module: optimizer inputs written, its result converted to a tracking reference on the bench's columns, a plan of "
          "track_reference entries, the bench check run", ok and len(ref["qpos"]) == len(Q) and [e["motion"] for e in plan["subgoals"]] == ["track_reference"] * 2
          and "worst_mean_mm" in r["bench_check"] and r["coverage"]["missing"] == [], json.dumps({k: r.get(k) for k in ("ok", "error", "bench_check")}, default=str)[:400])


# ---------------------------------------------------------------------------------------------- integration --
def _spider_paths():
    d, py = os.environ.get("DEXAGENT_SPIDER_DIR"), os.environ.get("DEXAGENT_SPIDER_PYTHON")
    if not (d and py):
        try:
            import yaml
            cfg = yaml.safe_load((ROOT / "dexagent.yaml").read_text()) or {}
            d = d or (cfg.get("models") or {}).get("spider_dir")
            py = py or (cfg.get("environments") or {}).get("spider")
        except Exception:
            pass
    return d, py


def test_integration():
    d, py = _spider_paths()
    if not (d and py and Path(d, "examples", "run_mjwp.py").exists() and Path(py).exists()):
        print("[SKIP] integration: no SPIDER checkout / python (DEXAGENT_SPIDER_DIR, DEXAGENT_SPIDER_PYTHON)")
        return
    gpu = subprocess.run([py, "-c", "import torch; print(torch.cuda.is_available())"], capture_output=True, text=True)
    if "True" not in gpu.stdout:
        print("[SKIP] integration: SPIDER's python sees no GPU")
        return
    model, scene = make_bench("integ", (0.02, 0.02, 0.03), 0.12, 0.42, 0.0, second=False)
    Q, labels = lift_reference(model, T=60)
    roles = SB.infer_roles(model, Q, labels)
    paths = SB.dataset_paths(TMP / "ds_integ", "right", "integ")
    rx = SB.write_robot_xml(SETUP, roles["sides"], paths["robot_xml"])
    sc = SB.write_scene(model, Q[0], roles, rx, paths["robot_task"], paths["assets"] / "objects")
    sm = mujoco.MjModel.from_xml_path(sc["scene"])
    t = np.arange(len(Q)) / 30.0
    ts = np.arange(0, t[-1] + 1e-9, 0.02)
    S = SB.resample(SB.bench_to_spider(model, Q, sm, sc["layout"]), t, ts, SB.quat_columns(sm))
    paths["trial"].mkdir(parents=True, exist_ok=True)
    np.savez(paths["trial"] / "trajectory_kinematic.npz", qpos=S, qvel=SB.qvel_of(sm, S, 0.02), contact=np.zeros((len(S), 10)),
             contact_pos=np.zeros((len(S), 10, 3)), frequency=50.0)
    (paths["robot_task"] / "task_info.json").write_text(json.dumps({"ref_dt": 0.02}))
    mj = SB.run_spider_mjwp(d, py, TMP / "ds_integ", "right", "integ", overrides={"num_samples": 64, "max_num_iterations": 2, "save_video": False})
    res = SB.read_mjwp(mj, 0.01)
    check("integration: SPIDER's MJWP ran on the bench's scene and returned the full horizon",
          res["qpos"].shape[1] == sm.nq and res["t"][-1] >= ts[-1] - 0.21 and np.isfinite(res["qpos"]).all(), f"({len(res['qpos'])} steps)")
    Qc, Qx = SB.bench_command_from_spider(model, Q, sm, sc["layout"], res, t)
    roll = E.servo_rollout(E.build_execution_model(str(scene))[0], Qc, 30.0, qpos_start=Q[0])
    check("integration: the result re-simulates in the bench's physics", roll["unstable"] is None and np.isfinite(roll["qpos"]).all())


if __name__ == "__main__":
    for fn in (test_robot_xml, test_scene_and_conversions, test_resample, test_prep, test_force_equivalent, test_mjwp_settings, test_stage1_expect,
               test_human_mapping, test_reference_and_plan, test_unavailable, test_module_with_stand_in_optimizer, test_integration):
        try:
            fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            check(fn.__name__, False, f"crashed: {type(e).__name__}: {e}")
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n{'ALL PASS' if FAILS == 0 else f'{FAILS} FAILED'}")
    sys.exit(1 if FAILS else 0)
