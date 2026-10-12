"""Independent negative cases for release acceptance, with real synthetic geometry."""
import copy
import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

from tool_library.common import task_gates as G
from tool_library.common.evidence import file_digest, record_tool_evidence
from tool_library.tests import test_stage1_goal_gate as fixture
from tool_library.verifiers.trajectory_manifest_check import _verify_stress_evidence, _bind_execution_pose
from tool_library.verifiers.robot_self_collision_check import robot_self_collision_check
from tool_library.verifiers.swept_collision_audit import swept_collision_audit


class ReleaseGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="dexagent_gate_hardening_")
        self.ws = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_generator_cannot_skip_manipulation_or_change_contact_target(self):
        (self.ws / "scene.xml").write_text(fixture.XML)
        (self.ws / "sem.json").write_text(json.dumps(fixture.SEM))
        short = fixture.build(self.ws, 0.053)
        man = fixture.manifest(self.ws, "skip", {1: {"skipped": True, "skip_kind": "no_robot_action", "reason": "already complete"}})
        self.assertFalse(fixture.run(self.ws, man, short, only=[1])["ok"])
        # The lavender brick touches the table, but Stage 1 requires the yellow brick.
        pred = {"class": "contact", "bodies": {"A": "lavender_brick_lavender_brick", "B": "table"}}
        man = fixture.manifest(self.ws, "target", {1: {"goal_predicates": [pred]}})
        result = fixture.run(self.ws, man, short, only=[1])
        self.assertFalse(result["ok"])
        self.assertTrue(any(p["source"] == "goal_predicates" and p["ok"] for p in result["subgoals"][0]["predicates"]))

    def test_only_independent_observation_can_be_skipped(self):
        self.assertTrue(G.no_robot_action_allowed({"motion": "observe", "goal": "Look at the camera", "object_refs": []}))
        self.assertFalse(G.no_robot_action_allowed({"motion": "wait", "goal": "Hold the cup still", "object_refs": []}))
        self.assertFalse(G.no_robot_action_allowed({"motion": "wait", "goal": "Wait", "object_refs": ["cup"]}))

    def test_missing_checks_and_diagnostic_modes_do_not_pass(self):
        suite = importlib.import_module("tool_library.verifiers.stage3_gate_suite")
        (self.ws / "scene.xml").write_text("<mujoco/>")
        (self.ws / "trajectory.npz").write_bytes(b"not read by the orchestration mock")
        (self.ws / "trajectory_manifest.json").write_text(json.dumps({"scene_executed": "scene.xml", "subgoals": []}))
        (self.ws / "recon.json").write_text(json.dumps({"scene_xml": "scene.xml"}))
        def check(name, **kw):
            return suite.stage3_gate_suite(str(self.ws), "sem.json", str(self.ws / "recon.json"), **kw)
        with patch.object(suite, "_call", return_value={"ok": True}):
            self.assertTrue(check("all")["ok"])
            self.assertFalse(check("skip", skip=["arm_reach_check"])["ok"])
            self.assertFalse(check("draws", draws=0)["ok"])
            self.assertFalse(check("quality", strict_quality=False)["ok"])
        def missing(name, kwargs):
            if name == suite.PHYSICS_GATE:
                raise ImportError("simulator missing")
            return {"ok": True}
        with patch.object(suite, "_call", side_effect=missing):
            self.assertFalse(check("missing")["ok"])
        with patch.object(suite, "_call", return_value={"ok": True, "skipped": "no input"}):
            self.assertFalse(check("unavailable")["ok"])

    def test_augmentation_waits_for_complete_original_episode_acceptance(self):
        suite = importlib.import_module("tool_library.verifiers.stage3_gate_suite")
        (self.ws / "trajectory.npz").write_bytes(b"orchestration fixture")
        (self.ws / "trajectory_manifest.json").write_text(json.dumps({"scene_executed": "scene.xml",
            "subgoals": [{"index": 1, "frames": [0, 1], "expect": {"released": True}}]}))
        (self.ws / "scene.xml").write_text("<mujoco/>")
        (self.ws / "recon.json").write_text(json.dumps({"scene_xml": "scene.xml"}))
        for failure in ("swept_collision_audit", "subgoal_outcome_check", None):
            seen = []
            def call(name, kwargs):
                seen.append(name)
                return {"ok": name != failure, "issues": ["negative fixture"] if name == failure else []}
            with self.subTest(failure=failure), patch.object(suite, "_call", side_effect=call):
                result = suite.stage3_gate_suite(str(self.ws), "sem.json", str(self.ws / "recon.json"))
            self.assertEqual(result["ok"], failure is None)
            if failure:
                self.assertNotIn(suite.PHYSICS_GATE, seen)
                self.assertEqual(set(result["deferred"]), {suite.PHYSICS_GATE, suite.RANDOMIZATION_GATE})
            else:
                self.assertLess(seen.index("subgoal_outcome_check"), seen.index(suite.RANDOMIZATION_GATE))
                self.assertEqual(seen[-1], suite.PHYSICS_GATE)
                self.assertEqual(result["deferred"], {})

    def test_forged_stress_success_is_independently_replayed(self):
        mod = importlib.import_module("tool_library.verifiers.grasp_hold_test")
        scene = {"side": "left", "grasp_body": "cup"}
        pose = {"pos": [0, 0, 0], "euler": [0, 0, 0], "finger_targets": {}}
        (self.ws / "pose.json").write_text(json.dumps(pose))
        (self.ws / "hand.json").write_text(json.dumps(scene))
        args = {"scene": scene, "pose": pose, "slip_max_mm_s": 1e9, "lift_m": 0}
        lib = ROOT / "tool_library"
        report = record_tool_evidence(lib, mod.TOOL, args, {"ok": True}, self.ws)
        metadata = {"file": "pose.json", "hand_scene": "hand.json"}
        entry = {"side": "left", "object": "cup"}
        seen = []
        def independent(scene, pose, hold_s=0.8, slip_max_mm_s=10.0, lift_m=0.08, maintain_N=0.0):
            seen.append((slip_max_mm_s, lift_m))
            return {"ok": False, "lifted": False}
        trusted = {"grasp_hold_test": file_digest(lib / "verifiers/grasp_hold_test.py")}
        with patch.object(mod, "grasp_hold_test", side_effect=independent) as fn:
            # Preserve the real callable signature while spying on the invocation.
            fn.__signature__ = __import__("inspect").signature(independent)
            with self.assertRaisesRegex(ValueError, "independent.*failed"):
                _verify_stress_evidence("grasp_hold_test", report, metadata, entry, None, self.ws, lib, trusted_verifiers=trusted)
        self.assertEqual(seen, [(10.0, 0.08)])
        (self.ws / "pose.json").write_text(json.dumps({**pose, "pos": [1, 0, 0]}))
        with self.assertRaisesRegex(ValueError, "different pose"):
            _verify_stress_evidence("grasp_hold_test", report, metadata, entry, None, self.ws, lib, trusted_verifiers=trusted)

    def test_same_hand_interpenetration_and_between_frame_robot_collision(self):
        xml = '''<mujoco><worldbody>
        <body name="robot_left_thumb_tip" pos="-.02 0 .2"><joint name="thumb_slide" type="slide" axis="1 0 0"/><geom name="thumb" type="sphere" size=".01" mass=".01"/></body>
        <body name="robot_left_index_tip" pos=".02 0 .2"><geom name="index" type="sphere" size=".01" mass=".01"/></body>
        </worldbody></mujoco>'''
        scene = self.ws / "collision.xml"
        scene.write_text(xml)
        traj = self.ws / "collision.npz"
        np.savez(traj, qpos=np.array([[0.], [.04], [0.]]), fps=30., labels=np.array(["left: move"] * 3))
        result = robot_self_collision_check(str(scene), str(traj))
        self.assertFalse(result["ok"], result)
        self.assertIn("hand-hand", result["failed"])
        # Both saved endpoints are clear: the moving thumb crosses the fixed index.
        np.savez(traj, qpos=np.array([[0.], [.08]]), fps=30., labels=np.array(["left: move"] * 2))
        result = swept_collision_audit(str(scene), str(traj), max_sub=40)
        self.assertFalse(result["ok"], result)
        self.assertGreater(result["n_events"], 0, result)
        with self.assertRaisesRegex(ValueError, "baseline-relative"):
            robot_self_collision_check(str(scene), str(traj), baseline=str(traj))

    def test_changed_executor_pose_requires_new_stress_input(self):
        candidate = {"pos": [0, 0, .1], "euler": [0, 0, 0], "finger_targets": {}}
        selected = {**candidate, "pos": [.005, 0, .1]}
        for name, value in (("candidate.json", candidate), ("used.json", selected),
                            ("hand.json", {"side": "left", "grasp_body": "cup"})):
            (self.ws / name).write_text(json.dumps(value))
        entry = {"motion": "grasp_lift", "side": "left", "object": "cup",
                 "pose": {"file": "candidate.json", "hand_scene": "hand.json"}}
        executed = {"selected_pose_evidence": {"path": "used.json", "sha256": file_digest(self.ws / "used.json"),
                    "hand_scene_path": "hand.json", "hand_scene_sha256": file_digest(self.ws / "hand.json")}}
        issues = []
        _bind_execution_pose(entry, executed, "left", self.ws, "entry", issues)
        self.assertTrue(any("actual selected" in i for i in issues), issues)
        entry["pose"]["file"] = "used.json"
        issues = []
        _bind_execution_pose(entry, executed, "left", self.ws, "entry", issues)
        self.assertEqual(issues, [])
        (self.ws / "used.json").write_text(json.dumps(candidate))
        _bind_execution_pose(entry, executed, "left", self.ws, "entry", issues)
        self.assertTrue(any("changed after execution" in i for i in issues), issues)

    def test_rigid_mount_accepts_fk_and_rejects_float_changed_mount_and_nan(self):
        from tool_library.common.rigid_robot import compile_rigid
        from tool_library.verifiers.rigid_arm_hand_check import rigid_arm_hand_check
        setup = ROOT / "setup/scene.xml"
        runtime = compile_rigid(str(setup))
        scene = self.ws / "rigid.xml"
        scene.write_text(runtime.xml)
        traj = self.ws / "rigid.npz"
        Q = np.repeat(runtime.model.key_qpos[0:1], 3, axis=0)
        arm = runtime.model.joint("robot_left_arm_dof_joint1")
        Q[:, int(runtime.model.jnt_qposadr[arm.id])] += [0., .02, .04]
        np.savez(traj, qpos=Q, fps=30.)
        report = rigid_arm_hand_check(str(scene), str(traj))
        self.assertTrue(report["ok"], report)
        spec = mujoco.MjSpec.from_file(str(scene))
        hand = next(b for b in spec.bodies if b.name == "robot_left_left_base_yaw")
        hand.pos = np.asarray(hand.pos) + [.005, 0., 0.]
        spec.compile()
        (self.ws / "changed.xml").write_text(spec.to_xml())
        self.assertFalse(rigid_arm_hand_check(str(self.ws / "changed.xml"), str(traj))["ok"])
        Q[1, 0] = np.nan
        np.savez(traj, qpos=Q, fps=30.)
        self.assertFalse(rigid_arm_hand_check(str(scene), str(traj))["ok"])
        model = mujoco.MjModel.from_xml_path(str(setup))
        np.savez(traj, qpos=np.repeat(model.key_qpos[0:1], 3, axis=0), fps=30.)
        self.assertFalse(rigid_arm_hand_check(str(setup), str(traj))["ok"])

    def test_runtime_integrity_uses_canonical_rigid_robot_and_rejects_tampering(self):
        from tool_library.common.rigid_robot import compile_rigid
        from tool_library.verifiers.robot_model_integrity_check import robot_model_integrity_check
        from tool_library.verifiers.actuator_limit_integrity_check import actuator_limit_integrity_check
        from tool_library.verifiers.robot_joint_range_check import robot_joint_range_check
        runtime = compile_rigid(str(ROOT / "setup/scene.xml"))
        scene = self.ws / "runtime_integrity.xml"
        scene.write_text(runtime.xml)
        for check in (robot_model_integrity_check, actuator_limit_integrity_check, robot_joint_range_check):
            with self.subTest(check=check.__name__):
                result = check(str(scene))
                self.assertTrue(result["ok"], result)
        spec = mujoco.MjSpec.from_file(str(scene))
        motor = next(a for a in spec.actuators if a.name == "robot_left_arm_dof_joint1_motor")
        motor.forcerange = [-280., 280.]
        spec.compile()
        changed = self.ws / "stronger.xml"
        changed.write_text(spec.to_xml())
        self.assertFalse(actuator_limit_integrity_check(str(changed))["ok"])
        spec = mujoco.MjSpec.from_file(str(scene))
        hand = next(b for b in spec.bodies if b.name == "robot_left_left_base_yaw")
        hand.gravcomp = 1.
        spec.compile()
        changed.write_text(spec.to_xml())
        self.assertFalse(robot_model_integrity_check(str(changed))["ok"])

    def test_recording_rejects_edited_frames_and_missing_steps(self):
        from tool_library.common.rigid_robot import compile_rigid, SCHEMA
        from tool_library.common.episode import joint_names
        from tool_library.verifiers.physics_recording_check import physics_recording_check
        runtime = compile_rigid(str(ROOT / "setup/scene.xml"))
        scene = self.ws / "recorded.xml"
        scene.write_text(runtime.xml)
        model = mujoco.MjModel.from_xml_path(str(scene))
        dt = float(model.opt.timestep)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        state_spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        def state():
            out = np.zeros(mujoco.mj_stateSize(model, state_spec))
            mujoco.mj_getState(model, data, out, state_spec)
            return out
        start = state()
        qs, vs, cs = [data.qpos.copy()], [data.qvel.copy()], [data.ctrl.copy()]
        for _ in range(2):
            mujoco.mj_step(model, data)
            qs.append(data.qpos.copy()); vs.append(data.qvel.copy()); cs.append(data.ctrl.copy())
        Q, V, C = np.array(qs), np.array(vs), np.array(cs)
        z = dict(qpos=Q, qvel=V, ctrl=C, time_s=np.arange(3) * dt,
                 initial_qpos=Q[0], initial_qvel=V[0], substep_qpos=Q[1:].copy(),
                 substep_qvel=V[1:].copy(), substep_ctrl=C[1:].copy(), substep_time_s=np.arange(1, 3) * dt,
                 initial_state=start, final_state=state(), state_spec=state_spec,
                 scene_sha256=file_digest(scene),
                 robot_schema=SCHEMA, joint_names=np.array(joint_names(model)), actuator_names=np.array([model.actuator(i).name for i in range(model.nu)]))
        traj = self.ws / "recorded.npz"
        np.savez(traj, **z)
        self.assertTrue(physics_recording_check(str(scene), str(traj))["ok"])
        for key in ("qpos", "qvel", "ctrl", "substep_time_s"):
            changed = {k: np.array(v).copy() for k, v in z.items()}
            changed[key].flat[-1] += .01
            np.savez(traj, **changed)
            with self.subTest(field=key):
                self.assertFalse(physics_recording_check(str(scene), str(traj))["ok"])
        np.savez(traj, qpos=Q, fps=30.)
        self.assertFalse(physics_recording_check(str(scene), str(traj))["ok"])

        # Consistently falsifying both render and dense arrays is still rejected
        # by independent integration of the real recorded controls.
        changed = {k: np.array(v).copy() for k, v in z.items()}
        changed["qpos"][1, 0] += .01
        changed["substep_qpos"][0, 0] += .01
        np.savez(traj, **changed)
        result = physics_recording_check(str(scene), str(traj))
        self.assertFalse(result["ok"])
        self.assertTrue(any("control replay diverges" in i for i in result["issues"]), result)

    def test_recorded_constraint_changes_are_not_physical_authority(self):
        from tool_library.common.episode import joint_names
        from tool_library.common.rigid_robot import SCHEMA
        from tool_library.verifiers.physics_recording_check import physics_recording_check
        scene = self.ws / "constraints.xml"
        scene.write_text('<mujoco><worldbody><body name="object" pos="0 0 .2"><freejoint name="object_free"/>'
                         '<geom type="sphere" size=".01" mass=".1"/></body></worldbody>'
                         '<equality><weld name="artificial_hold" body1="object" active="false"/></equality></mujoco>')
        m = mujoco.MjModel.from_xml_path(str(scene)); d = mujoco.MjData(m)
        spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        initial = np.zeros(mujoco.mj_stateSize(m, spec)); mujoco.mj_getState(m, d, initial, spec)
        q0, v0 = d.qpos.copy(), d.qvel.copy()
        # The recording really follows these dynamics, but an untrusted command
        # has glued the object in place. Exact replay alone must not certify it.
        d.eq_active[:] = 1
        mujoco.mj_step(m, d)
        final = np.zeros_like(initial); mujoco.mj_getState(m, d, final, spec)
        z = dict(qpos=np.stack([q0, d.qpos]), qvel=np.stack([v0, d.qvel]), ctrl=np.zeros((2, 0)),
                 time_s=np.array([0., m.opt.timestep]), initial_qpos=q0, initial_qvel=v0,
                 substep_qpos=d.qpos[None].copy(), substep_qvel=d.qvel[None].copy(), substep_ctrl=np.zeros((1, 0)),
                 substep_time_s=np.array([m.opt.timestep]), initial_state=initial, final_state=final, state_spec=spec,
                 joint_names=np.array(joint_names(m)), actuator_names=np.array([], dtype=str), robot_schema=SCHEMA,
                 scene_sha256=file_digest(scene), eq_names=np.array(["artificial_hold"]), eq_active=np.array([[0], [1]]),
                 eq_active_start=np.array([0]), substep_eq_active=np.array([[1]]), substep_eq_active_before=np.array([[1]]),
                 substep_eq_data=m.eq_data[None].copy())
        trajectory = self.ws / "constraints.npz"; np.savez(trajectory, **z)
        report = physics_recording_check(str(scene), str(trajectory))
        self.assertFalse(report["ok"])
        self.assertTrue(any("independently recomputed mechanism" in i for i in report["issues"]), report)

    def test_execution_cannot_change_stage2_objects_or_initial_state(self):
        from tool_library.common import episode as E, rigid_robot as RR
        from tool_library.verifiers.execution_scene_integrity_check import execution_scene_integrity_check
        proposal, _ = E.build_execution_model(str(ROOT / "setup/scene.xml"))
        spec = RR._SPECS[proposal].copy()
        body = spec.worldbody.add_body(name="test_object", pos=[0., .2, .03])
        body.add_freejoint(name="test_object_free")
        body.add_geom(name="test_object_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[.02, .02, .02], mass=.1)
        for key in spec.keys:
            key.qpos = np.r_[key.qpos, [0., .2, .03, 1., 0., 0., 0.]]
            key.qvel = np.r_[key.qvel, np.zeros(6)]
        spec.compile()
        source = self.ws / "stage2.xml"; source.write_text(spec.to_xml())
        recon = self.ws / "scene_recon.json"; recon.write_text(json.dumps({"scene_xml": str(source)}))
        expected, _ = E.build_execution_model(str(source)); runtime = RR.compile_rigid(expected)
        scene = self.ws / "executed.xml"; scene.write_text(runtime.xml)
        trajectory = self.ws / "initial.npz"
        data = mujoco.MjData(runtime.model)
        mujoco.mj_resetDataKeyframe(runtime.model, data, 0)
        state_spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        initial_state = np.zeros(mujoco.mj_stateSize(runtime.model, state_spec))
        mujoco.mj_getState(runtime.model, data, initial_state, state_spec)
        np.savez(trajectory, qpos=data.qpos[None], qvel=data.qvel[None],
                 initial_state=initial_state, state_spec=state_spec)
        report = execution_scene_integrity_check(str(scene), str(recon), str(trajectory))
        self.assertTrue(report["ok"], report)
        np.savez(trajectory, qpos=data.qpos[None], qvel=data.qvel[None])
        self.assertFalse(execution_scene_integrity_check(str(scene), str(recon), str(trajectory))["ok"])
        changed = mujoco.MjSpec.from_file(str(scene))
        obj_geom = next(g for g in changed.geoms if g.name == "test_object_geom")
        obj_geom.mass = .01
        changed.compile(); (self.ws / "lighter.xml").write_text(changed.to_xml())
        self.assertFalse(execution_scene_integrity_check(str(self.ws / "lighter.xml"), str(recon))["ok"])
        changed = mujoco.MjSpec.from_file(str(scene))
        geom = next(g for g in changed.geoms if g.name == "test_object_geom")
        geom.friction = [5., .1, .1]
        changed.compile(); (self.ws / "sticky.xml").write_text(changed.to_xml())
        self.assertFalse(execution_scene_integrity_check(str(self.ws / "sticky.xml"), str(recon))["ok"])
        q = runtime.model.key_qpos[0:1].copy(); q[0, -7] += .01
        np.savez(trajectory, qpos=q, qvel=runtime.model.key_qvel[0:1],
                 initial_state=initial_state, state_spec=state_spec)
        self.assertFalse(execution_scene_integrity_check(str(scene), str(recon), str(trajectory))["ok"])
        welded = runtime.xml.replace('</mujoco>', '<equality><weld name="extra_grasp" body1="test_object" body2="robot_left_left_base_yaw"/></equality></mujoco>')
        (self.ws / "welded.xml").write_text(welded)
        self.assertFalse(execution_scene_integrity_check(str(self.ws / "welded.xml"), str(recon))["ok"])

    def test_mechanism_changes_are_recomputed_from_trusted_stage2(self):
        from tool_library.common.episode import joint_names
        from tool_library.common.rigid_robot import MechanismReplay, SCHEMA
        from tool_library.verifiers.physics_recording_check import physics_recording_check
        from tool_library.skills.execute_subgoal_plan import thread_rules_from_recon
        scene = self.ws / "thread.xml"
        scene.write_text('<mujoco><worldbody><body name="cap"><joint name="cap_lift" type="slide" axis="0 0 1"/>'
                         '<geom type="sphere" size=".01" mass=".1"/></body></worldbody>'
                         '<equality><joint name="engaged" joint1="cap_lift" polycoef="0 0 0 0 0"/></equality>'
                         '<keyframe><key qpos=".02"/></keyframe></mujoco>')
        recon = self.ws / "recon.json"
        recon.write_text(json.dumps({"scene_xml": str(scene), "objects": [{"equalities": {"engaged": "engaged"},
                                      "joints": {"lift": "cap_lift"}, "params": {"release_travel_m": .01}}]}))
        m = mujoco.MjModel.from_xml_path(str(scene)); d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        engine = MechanismReplay(m, d, thread_rules_from_recon(recon))
        spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        def full_state():
            x = np.zeros(mujoco.mj_stateSize(m, spec)); mujoco.mj_getState(m, d, x, spec); return x
        initial, ms0, ed0 = full_state(), engine.snapshot(), m.eq_data.copy()
        qs, vs, es, before, ed = [d.qpos.copy()], [d.qvel.copy()], [d.eq_active.copy()], [], []
        for i in range(3):
            before.append(d.eq_active.copy()); ed.append(m.eq_data.copy())
            mujoco.mj_step(m, d); engine.post_step(d, i); mujoco.mj_forward(m, d)
            qs.append(d.qpos.copy()); vs.append(d.qvel.copy()); es.append(d.eq_active.copy())
        q, v, eqhist = np.array(qs), np.array(vs), np.array(es)
        z = dict(qpos=q, qvel=v, ctrl=np.zeros((4, 0)), time_s=np.arange(4)*m.opt.timestep,
                 initial_qpos=q[0], initial_qvel=v[0], substep_qpos=q[1:], substep_qvel=v[1:], substep_ctrl=np.zeros((3, 0)),
                 substep_time_s=np.arange(1, 4)*m.opt.timestep, initial_state=initial, final_state=full_state(), state_spec=spec,
                 joint_names=np.array(joint_names(m)), actuator_names=np.array([], dtype=str), robot_schema=SCHEMA,
                 scene_sha256=file_digest(scene), eq_names=np.array(["engaged"]), eq_active=eqhist, eq_active_start=eqhist[0],
                 substep_eq_active=eqhist[1:], substep_eq_active_before=np.array(before), substep_eq_data=np.array(ed),
                 eq_data_start=ed0, mechanism_initial=json.dumps(ms0), mechanism_final=json.dumps(engine.snapshot()))
        trajectory = self.ws / "thread.npz"; np.savez(trajectory, **z)
        report = physics_recording_check(str(scene), str(trajectory), str(recon))
        self.assertTrue(report["ok"], report)
        self.assertFalse(physics_recording_check(str(scene), str(trajectory))["ok"])
        self.assertEqual(eqhist[:, 0].tolist(), [1, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
