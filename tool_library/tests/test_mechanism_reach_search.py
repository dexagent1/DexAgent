"""Generic mechanism search: synchronized geometry, finite evidence, diverse contacts.

Small kinematic fixtures isolate search correctness; mocked IK/probes below
test state plumbing, not a physical task success. Real arm collision regression
coverage lives in test_arm_collision_screens.py.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root: runs as a script too
from tool_library.common import episode as E, plan as P, armhand as AH, hand as H
from tool_library.skills.optimize_pose_for_reach import (
    _joint_sweep_states, _mechanism_command_states, _anchor_to_state, optimize_pose_for_reach)
from tool_library.common.subgoal import SubgoalPlanner
from tool_library.skills.reachable_contact_search import face_points, _station_origin, _headings, reachable_contact_search
from tool_library.skills.diagnose_stage3_failure import diagnose_stage3_failure


ROOT = Path(__file__).resolve().parents[2]


def absolute_spec():
    source = ROOT / "setup/scene.xml"
    spec = mujoco.MjSpec.from_file(str(source))
    for asset in spec.meshes:
        if asset.file and not Path(asset.file).is_absolute():
            asset.file = str(source.parent / (spec.meshdir or "") / asset.file)
    for asset in spec.textures:
        if asset.file and not Path(asset.file).is_absolute():
            asset.file = str(source.parent / (spec.texturedir or "") / asset.file)
    spec.meshdir = ""; spec.texturedir = ""
    return spec


class MechanismSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="mechanism_search_")
        spec = absolute_spec()
        # Two unrelated fixtures: horizontal-axis flap and vertical-axis lever.
        for name, axis, pos in [("flap", [1, 0, 0], [1.4, 1.1, .8]),
                                ("lever", [0, 0, 1], [1.2, 1.2, .6])]:
            body = spec.worldbody.add_body(name=name, pos=pos)
            body.add_joint(name=name + "_hinge", type=mujoco.mjtJoint.mjJNT_HINGE,
                           axis=axis, limited=True, range=[-3.2, 3.2])
            body.add_geom(name=name + "_panel", type=mujoco.mjtGeom.mjGEOM_BOX,
                          size=[.004, .02, .002], mass=.05)
        for key in spec.keys:
            key.qpos = list(key.qpos) + [0., 0.]
        cls.path = Path(cls.tmp.name) / "fixtures.xml"
        cls.path.write_text(spec.to_xml())
        cls.model = spec.compile()
        cls.rigs = P.scene_rigs(cls.model)
        d = mujoco.MjData(cls.model); mujoco.mj_resetDataKeyframe(cls.model, d, 0)
        cls.pose = dict(pos=[1.5, 1.2, .7], euler=[.2, -.1, .3],
                        finger_targets=cls.rigs["right"].finger_targets(.1), grip_fingers=["index"],
                        allowed_bodies=["test_obstacle"])
        P.set_wrist(cls.model, d, cls.rigs["right"], cls.pose["pos"], Rotation.from_euler("XYZ", cls.pose["euler"]).as_matrix())
        cls.rigs["right"].set_fingers(d, cls.pose["finger_targets"])
        P.set_wrist(cls.model, d, cls.rigs["left"], [1.4, 1.1, .95], np.eye(3))
        cls.q = d.qpos.copy()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_rigid_and_level_follow_geometry_on_distinct_hinges(self):
        m, rigs = self.model, self.rigs
        for joint in ["flap_hinge", "lever_hinge"]:
            for mode in ["rigid", "level"]:
                with self.subTest(joint=joint, mode=mode):
                    q = self.q.copy()
                    d = mujoco.MjData(m); d.qpos[:] = q; mujoco.mj_forward(m, d)
                    b = int(m.jnt_bodyid[m.joint(joint).id])
                    p0, R0 = P.wrist_world(m, d, rigs["right"])
                    c0 = p0 + np.array([.02, .01, -.01])
                    local = d.xmat[b].reshape(3, 3).T @ ((c0 if mode == "level" else p0) - d.xpos[b])
                    states = _joint_sweep_states(m, rigs, "right", q,
                        dict(joint=joint, to=np.pi / 2, steps=1, hand_follows=mode, co_sides=["left"]), c0)
                    self.assertEqual(len(states), 9)
                    for _, state in states:
                        d.qpos[:] = state; mujoco.mj_forward(m, d)
                        p, R = P.wrist_world(m, d, rigs["right"])
                        point = p + c0 - p0 if mode == "level" else p
                        np.testing.assert_allclose(d.xmat[b].reshape(3, 3).T @ (point - d.xpos[b]), local, atol=1e-10)
                        if mode == "level":
                            np.testing.assert_allclose(R, R0, atol=1e-10)
                    self.assertGreater(np.linalg.norm(states[-1][1] - q), .1)
                    # A co-holder actually follows; it is not left at its old wrist pose.
                    left = [rigs["left"].wrist_qpos[k] for k in ("x", "y", "z")]
                    self.assertGreater(np.linalg.norm(states[-1][1][left] - q[left]), .01)
                    np.testing.assert_array_equal(q, self.q)

    def test_invalid_sweep_never_silently_selects_rigid(self):
        for invalid in [dict(hand_follows="typo"), dict(steps=0), dict(steps=1.5), dict(to=10.),
                        dict(to=float("nan")), dict(**{"from": .5}), dict(co_sides=["right"]), dict(joint="missing")]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _joint_sweep_states(self.model, self.rigs, "right", self.q,
                    dict(dict(joint="flap_hinge", to=.5), **invalid), self.pose["pos"])

    @staticmethod
    def fake_solver(model, side):
        adr = E.robot_parts(model)[side]["arm_qadr"]
        home = np.asarray(model.key_qpos[0] if model.nkey else model.qpos0)[adr]
        def solve(p, R, previous=None):
            return 0., 0., home.copy()
        solve.arm_qadr = adr
        return solve

    def run_search(self, path=None, probe=None, **kwargs):
        with patch("tool_library.skills.optimize_pose_for_reach._reach_solver", self.fake_solver), \
             patch.object(AH, "ArmProbe", return_value=probe or type("Probe", (), {"check": lambda *a, **kw: []})()):
            return optimize_pose_for_reach(str(path or self.path), "right", self.pose,
                axis="vertical", range_deg=0, tilt_axis=None, approach_m=0, **kwargs)

    def test_probe_receives_moving_joint_state_and_reports_sweep_collision(self):
        samples = []
        joint = self.model.joint("lever_hinge").qposadr[0]
        class Probe:
            def check(self, Q, **kw):
                samples.append(float(Q[0, joint]))
                return [{"geoms": ["arm_hull", "lever_panel"], "depth_mm": 5.}] if Q[0, joint] > .5 else []
        r = self.run_search(probe=Probe(), sweep=dict(joint="lever_hinge", to=1., steps=2))
        self.assertFalse(r["reach"]["ok"], r)
        self.assertGreater(max(samples), .5)
        self.assertTrue(r["reach"]["arm_hit"]["where"].startswith("sweep"))
        self.assertEqual(r["search_status"], "inconclusive")

    def test_clean_sweep_is_a_sampled_witness_not_physics_pass(self):
        r = self.run_search(sweep=dict(joint="lever_hinge", to=.6, steps=1, hand_follows="level"))
        self.assertTrue(r["reach"]["ok"], r)
        self.assertGreaterEqual(r["reach"]["n_path_states"], 5)
        self.assertEqual(r["checked_motion"]["sweep"]["hand_follows"], "level")
        self.assertTrue(r["pose"]["reach_optimized"]["checked_motion"]["execution_required"])

    def test_hand_collision_mid_sweep_including_allowed_body_is_rejected(self):
        states = _joint_sweep_states(self.model, self.rigs, "right", self.q,
            dict(joint="lever_hinge", to=np.pi / 2, steps=2), self.pose["pos"])
        d = mujoco.MjData(self.model); d.qpos[:] = states[3][1]; mujoco.mj_forward(self.model, d)
        pos = d.xpos[self.rigs["right"].palm_body].copy()
        spec = mujoco.MjSpec.from_file(str(self.path))
        obstacle = spec.worldbody.add_body(name="test_obstacle", pos=pos)
        obstacle.add_geom(name="test_obstacle_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[.025, 0, 0])
        path = Path(self.tmp.name) / "blocked.xml"; path.write_text(spec.to_xml())
        initial = self.run_search(path)
        self.assertTrue(initial["reach"]["ok"], initial)
        swept = self.run_search(path, sweep=dict(joint="lever_hinge", to=np.pi / 2, steps=2))
        self.assertFalse(swept["reach"]["ok"], swept)
        self.assertGreater(swept["reach"]["overlap_mm"], 3.)
        self.assertIn("test_obstacle_geom", swept["reach"]["overlap_with"])
        self.assertTrue(swept["reach"]["overlap_at"].startswith("sweep"))

    def test_thin_face_grid_never_samples_outside_and_single_point_is_center(self):
        points, normal = face_points(self.path, "flap_panel", "+x", n=4, inset_m=.03)
        self.assertEqual(len(points), 1)
        np.testing.assert_allclose(points[0], [1.404, 1.1, .8])
        points, _ = face_points(self.path, "flap_panel", "+z", n=1, inset_m=0.)
        np.testing.assert_allclose(points[0], [1.4, 1.1, .802])
        self.assertEqual(normal, [1., 0., 0.])

    def test_station_heading_uses_configured_base_and_handles_normal_alignment(self):
        spec = mujoco.MjSpec.from_file(str(self.path))
        m = spec.compile()
        j = next(j for j in range(m.njnt) if m.joint(j).name.startswith("robot_right_arm_"))
        body = spec.body(m.body(int(m.jnt_bodyid[j])).name)
        body.pos = (np.asarray(body.pos) + [.31, -.17, .06]).tolist()
        path = Path(self.tmp.name) / "station.xml"; path.write_text(spec.to_xml())
        # The setup mounts the arm in a rotated frame; a local displacement
        # rotates in world coordinates but must retain its length.
        delta = _station_origin(path, "right") - _station_origin(self.path, "right")
        self.assertAlmostEqual(float(np.linalg.norm(delta)), float(np.linalg.norm([.31, -.17, .06])), places=6)
        dirs = _headings([0, 0, 1], [0, 0, 1], np.zeros(3), 4)
        np.testing.assert_allclose(np.linalg.norm(dirs, axis=1), 1.)
        np.testing.assert_allclose(np.asarray(dirs)[:, 2], 0.)
        np.testing.assert_allclose(np.asarray(dirs)[0], -np.asarray(dirs)[2], atol=1e-10)

    def test_search_explores_headings_and_returns_selected_motion_mode(self):
        calls = []
        def flat(scene, point, normal, finger_dir, **kw):
            return dict(finger_dir=finger_dir)
        def optimize(scene, side, pose, **kw):
            calls.append((pose["finger_dir"], kw["sweep"]["hand_follows"]))
            ok = pose["finger_dir"][0] < -.9 and kw["sweep"]["hand_follows"] == "level"
            return dict(pose=pose, reach=dict(ok=ok, worst_pos_err_mm=0 if ok else 10, worst_rot_err_deg=0),
                        rotation_deg={}, budget_reached=False, n_evaluated=1)
        with patch("tool_library.skills.flat_hand_pose.flat_hand_pose", flat), \
             patch("tool_library.skills.optimize_pose_for_reach.optimize_pose_for_reach", optimize), \
             patch("tool_library.skills.reachable_contact_search._station_origin", return_value=np.array([0., 0., 0.])):
            r = reachable_contact_search(self.path, "right", {}, candidates=[dict(point=[1., 0., 0.], normal=[0., 0., 1.])],
                 sweep=dict(joint="flap_hinge", to=.5, hand_follows="auto"), time_budget_s=None)
        self.assertEqual(len(calls), 8)
        self.assertEqual(r["n_reachable"], 1)
        self.assertEqual(r["best"]["hand_follows"], "level")
        self.assertLess(r["best"]["pose"]["finger_dir"][0], -.9)

    def test_short_budget_explores_distinct_locations_before_extra_headings(self):
        now, calls = [0.], []
        def flat(scene, point, normal, finger_dir, **kw):
            return dict(point=point, finger_dir=finger_dir)
        def optimize(scene, side, pose, **kw):
            now[0] += 1.
            calls.append((pose["point"], kw["sweep"]["hand_follows"]))
            return dict(pose=pose, reach=dict(ok=False, worst_pos_err_mm=10., worst_rot_err_deg=10.),
                        rotation_deg={}, budget_reached=True, n_evaluated=1)
        with patch("tool_library.skills.flat_hand_pose.flat_hand_pose", flat), \
             patch("tool_library.skills.optimize_pose_for_reach.optimize_pose_for_reach", optimize), \
             patch("tool_library.skills.reachable_contact_search.time.monotonic", side_effect=lambda: now[0]), \
             patch("tool_library.skills.reachable_contact_search._station_origin", return_value=np.zeros(3)):
            r = reachable_contact_search(self.path, "right", {},
                candidates=[dict(point=[1., k / 10., 0.], normal=[0., 0., 1.]) for k in range(6)],
                sweep=dict(joint="flap_hinge", to=.5, hand_follows="auto"), time_budget_s=3.5)
        self.assertEqual([p[1] for p, _ in calls], [0., .1, .2, .3])
        self.assertEqual(r["n_candidates"], 4)
        self.assertEqual(r["n_requested"], 48)
        self.assertEqual(r["n_contact_locations_evaluated"], 4)
        self.assertEqual(r["n_contact_locations_requested"], 6)
        self.assertTrue(r["budget_reached"])
        self.assertEqual(r["search_status"], "inconclusive")

    def test_short_rotation_budget_reaches_tilt_before_exhausting_primary_axis(self):
        now = [0.]
        target = Rotation.from_rotvec([np.radians(15.), 0., 0.]).as_matrix() @ H.euler_to_mat(self.pose["euler"])
        def solver(model, side):
            base = self.fake_solver(model, side)
            def solve(p, R, previous=None):
                now[0] += 1.
                _, _, qa = base(p, R, previous)
                return .1 * np.linalg.norm(R - target), 0., qa
            solve.arm_qadr = base.arm_qadr
            return solve
        with patch("tool_library.skills.optimize_pose_for_reach._reach_solver", solver), \
             patch("tool_library.skills.optimize_pose_for_reach.time.monotonic", side_effect=lambda: now[0]), \
             patch.object(AH, "ArmProbe", return_value=type("Probe", (), {"check": lambda *a, **kw: []})()):
            r = optimize_pose_for_reach(str(self.path), "right", self.pose,
                axis="vertical", range_deg=90., step_deg=15., tilt_axis=[1., 0., 0.],
                tilt_range_deg=30., approach_m=0., time_budget_s=4.)
        self.assertTrue(r["reach"]["ok"], r)
        self.assertEqual(r["rotation_deg"], dict(primary=0., tilt=15.))
        self.assertEqual(r["n_evaluated"], 4)
        self.assertTrue(r["budget_reached"])
        self.assertEqual([c["angles"] for c in r["candidates"]], [[0., 0.], [15., 0.], [-15., 0.], [0., 15.]])

    def test_optimizer_screens_held_object_translation_and_orientation(self):
        spec = mujoco.MjSpec.from_file(str(self.path))
        body = spec.worldbody.add_body(name="payload", pos=[1.5, 1.2, .55])
        body.add_freejoint(name="payload_free")
        body.add_geom(name="payload_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[.005, 0, 0], mass=.02)
        for key in spec.keys:
            key.qpos = list(key.qpos) + [1.5, 1.2, .55, 1., 0., 0., 0.]
        path = Path(self.tmp.name) / "payload.xml"; path.write_text(spec.to_xml())
        model = spec.compile(); adr = int(model.joint("payload_free").qposadr[0]); seen = []
        class Probe:
            def check(self, Q, **kw):
                seen.append(Q[0, adr:adr + 7].copy())
                return []
        r = self.run_search(path, probe=Probe(), waypoints=[dict(object="payload", target_pos=[1.6, 1.3, .6],
                                                                target_euler_deg=[10., 0., 30.], steps=3)])
        self.assertTrue(r["reach"]["ok"], r)
        self.assertEqual(r["reach"]["n_path_states"], 4)
        np.testing.assert_allclose(seen[-1][:3], [1.6, 1.3, .6])
        self.assertGreater(np.linalg.norm(seen[-1][3:] - seen[0][3:]), .1)

    def test_unreachable_approach_midpoint_cannot_pass_reachable_endpoints(self):
        original = self.fake_solver
        def solver(model, side):
            base = original(model, side)
            def solve(p, R, previous=None):
                _, er, qa = base(p, R, previous)
                return (.1 if .73 < p[2] < .75 else 0.), er, qa
            solve.arm_qadr = base.arm_qadr
            return solve
        with patch("tool_library.skills.optimize_pose_for_reach._reach_solver", solver), \
             patch.object(AH, "ArmProbe", return_value=type("Probe", (), {"check": lambda *a, **kw: []})()):
            r = optimize_pose_for_reach(str(self.path), "right", dict(self.pose, approach=[0., 0., 1.]),
                axis="vertical", range_deg=0, tilt_axis=None, approach_m=.08)
        self.assertFalse(r["reach"]["ok"], r)
        self.assertGreater(r["reach"]["pregrasp_pos_err_mm"], 50.)

    def test_failed_finite_search_cannot_move_stations_or_waive_goals(self):
        r = diagnose_stage3_failure(reason="Stage 2 must be revised: no pose can reach after 20 sampled contacts",
                                    verdict={"ok": False, "issues": ["arm_unreachable", "stage1_goal_gate: missing goal"]},
                                    check_hand_scenes=False)
        self.assertEqual(r["reach_search_status"], "inconclusive")
        self.assertFalse((r["upstream"] or {}).get("verified"))
        self.assertNotIn("place_arm_stations", [x["tool"] for x in r["repairs"]])
        self.assertNotIn("report_blocked", [x["tool"] for x in r["repairs"]])
        self.assertNotIn("deviation {", json.dumps(r["repairs"]))
        self.assertIn("unverified hypothesis", r["summary"])

    def test_command_sweep_matches_planner_on_distinct_hinges(self):
        # Compare the public motion planner, including its actual press clamp,
        # with the search's commanded wrist path. These are kinematics, not a
        # claim that the far-away synthetic fixture is physically reachable.
        for body in ("flap", "lever"):
            for mode in ("rigid", "level"):
                for recorded_contacts in (False, True):
                    with self.subTest(body=body, mode=mode, contacts=recorded_contacts):
                        joint = body + "_hinge"
                        q = self.q.copy(); q[self.model.joint(joint).qposadr[0]] = .2
                        pose = dict(self.pose)
                        if recorded_contacts:
                            pose.update(points={"index": [1.45, 1.21, .68]},
                                        normals={"index": [0., 0., 1.]}, press_travel_m=.006)
                        sweep = dict(joint=joint, to=.9, steps=10, hand_follows=mode)
                        states = _mechanism_command_states(self.model, self.rigs, "right", q, pose, sweep)
                        planned = SubgoalPlanner(self.model, q).run(dict(index=1, motion="close_joint", side="right",
                            object=body, pose=pose, joint=joint, target=.9, steps=10, hand_follows=mode))
                        frames = [f for f in planned["keyframes"] if f["label"] == f"right: grip {body}"
                                  or f["label"].startswith(f"right: close {body}")]
                        self.assertEqual(len(states), 11)
                        self.assertEqual(len(frames), len(states))
                        for (_, expected), frame in zip(states, frames):
                            np.testing.assert_allclose(expected, frame["qpos"], atol=1e-10)

    def test_pose_anchor_follows_current_part_exactly_once(self):
        d = mujoco.MjData(self.model); d.qpos[:] = self.q
        d.qpos[self.model.joint("lever_hinge").qposadr[0]] = .6
        anchor = dict(body="lever", pos=[1.2, 1.2, .6], quat_wxyz=[1., 0., 0., 0.])
        pose = dict(self.pose, anchor=anchor, points={"index": [1.4, 1.2, .7]},
                    normals={"index": [0., 1., 0.]}, approach=[0., 1., 0.])
        result = _anchor_to_state(self.model, d, pose)
        R = Rotation.from_euler("z", .6).as_matrix(); origin = np.array(anchor["pos"])
        np.testing.assert_allclose(result["pos"], origin + R @ (np.array(pose["pos"]) - origin))
        np.testing.assert_allclose(result["points"]["index"], origin + R @ (np.array(pose["points"]["index"]) - origin))
        np.testing.assert_allclose(result["normals"]["index"], R @ pose["normals"]["index"])
        repeated = _anchor_to_state(self.model, d, result)
        np.testing.assert_allclose(repeated["pos"], result["pos"], atol=1e-10)
        np.testing.assert_allclose(repeated["euler"], result["euler"], atol=1e-10)

    def test_hand_scene_anchor_uses_materialized_pose_not_stale_record(self):
        current = mujoco.MjData(self.model); current.qpos[:] = self.q
        materialized = mujoco.MjData(self.model); materialized.qpos[:] = self.q
        adr = self.model.joint("flap_hinge").qposadr[0]
        current.qpos[adr] = .7; materialized.qpos[adr] = .3
        mujoco.mj_forward(self.model, materialized)
        hs = dict(grasp_body="flap", grasp_body_pose=dict(body="flap", pos=[1.4, 1.1, .8], quat_wxyz=[1., 0., 0., 0.]))
        with patch.object(H, "open_scene", return_value=(self.model, materialized, self.rigs["right"], {})):
            result = _anchor_to_state(self.model, current, self.pose, hs)
        origin = np.array([1.4, 1.1, .8])
        expected = origin + Rotation.from_euler("x", .4).apply(np.array(self.pose["pos"]) - origin)
        np.testing.assert_allclose(result["pos"], expected, atol=1e-10)

    def test_commanded_clamp_unreachable_rejects_reachable_touch_pose(self):
        rig = self.rigs["right"]
        clamp = H.executor_clamp(rig, self.pose)
        shift = H.pad_shift_world(rig, clamp["closed"], clamp["press"], self.pose["euler"])["index"]
        self.assertGreater(np.linalg.norm(shift), 1e-5)
        # This surface normal makes finger flexion leave the surface, so the
        # shared clamp commands a wrist press. Touch-only screening would pass.
        pose = dict(self.pose, normals={"index": (shift / np.linalg.norm(shift)).tolist()}, press_travel_m=.006)
        self.assertIsNotNone(H.executor_clamp(rig, pose)["wrist_press"])
        def solver(model, side):
            original = self.fake_solver(model, side)
            def solve(p, R, previous=None):
                return (.05 if np.linalg.norm(np.array(p) - pose["pos"]) > .003 else 0., 0., original(p, R)[2])
            solve.arm_qadr = original.arm_qadr
            return solve
        with patch("tool_library.skills.optimize_pose_for_reach._reach_solver", solver), \
             patch.object(AH, "ArmProbe", return_value=type("Probe", (), {"check": lambda *a, **kw: []})()):
            r = optimize_pose_for_reach(str(self.path), "right", pose, axis="vertical", range_deg=0,
                tilt_axis=None, approach_m=0, sweep=dict(joint="lever_hinge", to=0., steps=1))
        self.assertFalse(r["reach"]["ok"], r)
        self.assertEqual(r["reach"]["n_command_states"], 2)
        self.assertGreater(r["reach"]["worst_pos_err_mm"], 40.)

    def test_planar_tilt_keeps_surface_normal_and_rejects_points_leaving_plane(self):
        def solver(model, side):
            original = self.fake_solver(model, side)
            def solve(p, R, previous=None):
                return .1 * np.linalg.norm(R - target), 0., original(p, R)[2]
            solve.arm_qadr = original.arm_qadr
            return solve
        for collinear, heading in ((True, 0.), (True, 15.), (False, 0.)):
            target = (Rotation.from_rotvec([0., 0., -np.radians(heading)]).as_matrix()
                      @ Rotation.from_rotvec([np.radians(15.), 0., 0.]).as_matrix() @ H.euler_to_mat(self.pose["euler"]))
            pose = dict(self.pose, points={"index": [1.4, 1.2, .7], "middle": [1.5, 1.2, .7],
                                          "ring": [1.6, 1.2 if collinear else 1.23, .7]},
                        normals={f: [0., 0., 1.] for f in ("index", "middle", "ring")}, approach=[0., 0., 1.])
            with patch("tool_library.skills.optimize_pose_for_reach._reach_solver", solver), \
                 patch.object(AH, "ArmProbe", return_value=type("Probe", (), {"check": lambda *a, **kw: []})()):
                r = optimize_pose_for_reach(str(self.path), "right", pose, axis="normal", range_deg=heading,
                    tilt_axis=[1., 0., 0.], tilt_range_deg=15., step_deg=15., approach_m=0., time_budget_s=None)
            self.assertEqual(r["reach"]["ok"], collinear, r)
            self.assertEqual(r["checked_motion"]["planar_contact_geometry"], "plane only")
            np.testing.assert_allclose(r["pose"]["normals"]["index"], [0., 0., 1.])
            np.testing.assert_allclose(r["pose"]["approach"], [0., 0., 1.])
            if collinear:
                self.assertEqual(r["rotation_deg"]["primary"], heading)
                self.assertEqual(r["rotation_deg"]["tilt"], 15.)
                np.testing.assert_allclose(r["pivot"], [1.5, 1.2, .7])
            else:
                tilted = next(c for c in r["candidates"] if c["angles"] == [0., 15.])
                self.assertGreater(tilted["surface_point_error_mm"], 1.)
                self.assertFalse(tilted["ok"])

    def test_planar_geometry_checks_real_finger_gap_through_slider_sweep(self):
        m, d = self.model, mujoco.MjData(self.model)
        d.qpos[:] = self.q; mujoco.mj_forward(m, d)
        # Construct an independent small panel tangent to the actual index
        # collision geometry. Its position is measured from the synthetic
        # test hand, not from any task trajectory or contact proposal.
        support_points = []
        for g in self.rigs["right"].collision_gids:
            if "_index_" not in m.body(int(m.geom_bodyid[g])).name:
                continue
            c, R, size = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g]
            typ = int(m.geom_type[g])
            if typ == int(mujoco.mjtGeom.mjGEOM_MESH):
                mid = int(m.geom_dataid[g]); start, count = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
                verts = m.mesh_vert[start:start + count] @ R.T + c
                support_points.append(verts[np.argmax(verts[:, 2])])
            elif typ in (int(mujoco.mjtGeom.mjGEOM_SPHERE), int(mujoco.mjtGeom.mjGEOM_CAPSULE)):
                end = c if typ == int(mujoco.mjtGeom.mjGEOM_SPHERE) else c + np.sign(R[2, 2]) * R[:, 2] * size[1]
                support_points.append(end + [0., 0., size[0]])
        point = max(support_points, key=lambda v: v[2]) + [0., 0., .001]
        spec = mujoco.MjSpec.from_file(str(self.path))
        panel = spec.worldbody.add_body(name="contact_panel", pos=point + [0., 0., .001])
        panel.add_joint(name="contact_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0., 0., 1.])
        panel.add_geom(name="contact_panel_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[.008, .008, .001], mass=.02)
        for key in spec.keys:
            key.qpos = list(key.qpos) + [0.]
        path = Path(self.tmp.name) / "contact_panel.xml"; path.write_text(spec.to_xml())
        for offset in (0., -.02):
            pose = dict(self.pose, pos=(np.asarray(self.pose["pos"]) + [0., 0., offset]).tolist(),
                        points={"index": point.tolist()}, normals={"index": [0., 0., -1.]})
            with patch("tool_library.skills.optimize_pose_for_reach._reach_solver", self.fake_solver), \
                 patch.object(AH, "ArmProbe", return_value=type("Probe", (), {"check": lambda *a, **kw: []})()):
                r = optimize_pose_for_reach(str(path), "right", pose, axis="normal", range_deg=0.,
                    tilt_axis=None, approach_m=0., state=np.r_[self.q, 0.], sweep=dict(joint="contact_slide", to=.03, steps=3))
            self.assertEqual(r["reach"]["ok"], offset == 0., r)
            self.assertEqual(r["checked_motion"]["planar_contact_geometry"], "plane and finger-to-part gaps")
            if offset:
                self.assertGreater(r["reach"]["contact_gap_mm"], 15.)
            else:
                self.assertLess(r["reach"]["contact_gap_mm"], 2.)
                self.assertEqual(r["reach"]["n_path_states"], 4)

    def test_recorded_bench_state_used_by_optimizer_and_conflict_rejected(self):
        q = self.q.copy(); adr = self.model.joint("lever_hinge").qposadr[0]; q[adr] = .4
        hs = {"bench_state": {"joint_names": E.joint_names(self.model), "qpos": q.tolist()}}
        seen = []
        class Probe:
            def check(self, Q, **kw):
                seen.append(float(Q[0, adr])); return []
        r = self.run_search(hand_scene=hs, probe=Probe())
        self.assertTrue(r["reach"]["ok"], r)
        self.assertTrue(seen); np.testing.assert_allclose(seen, .4)
        with self.assertRaisesRegex(ValueError, "different object states"):
            self.run_search(hand_scene=hs, state=self.q)

    def test_search_samples_current_part_and_stamps_bench_anchor(self):
        q = self.q.copy(); q[self.model.joint("lever_hinge").qposadr[0]] = .4
        hs = {"grasp_body": "lever", "bench_state": {"joint_names": E.joint_names(self.model), "qpos": q.tolist()}}
        observed = []
        def flat(scene, point, normal, finger_dir, **kw):
            observed.append((point, normal)); return dict(self.pose)
        def optimize(scene, side, pose, **kw):
            np.testing.assert_allclose(kw["state"], q)
            self.assertEqual(pose["anchor"]["body"], "lever")
            actual = Rotation.from_quat(np.roll(pose["anchor"]["quat_wxyz"], -1)).as_matrix()
            np.testing.assert_allclose(actual, Rotation.from_euler("z", .4).as_matrix(), atol=1e-10)
            return dict(pose=pose, reach=dict(ok=True, worst_pos_err_mm=0, worst_rot_err_deg=0),
                        rotation_deg={}, budget_reached=False, n_evaluated=1)
        with patch("tool_library.skills.flat_hand_pose.flat_hand_pose", flat), \
             patch("tool_library.skills.optimize_pose_for_reach.optimize_pose_for_reach", optimize):
            r = reachable_contact_search(self.path, "right", hs, geom="lever_panel", face="+x", n=1,
                                         finger_dir=[0., 0., 1.], time_budget_s=None)
        self.assertEqual(r["n_reachable"], 1)
        point, normal = face_points(self.path, "lever_panel", "+x", n=1, state=q)
        np.testing.assert_allclose(observed[0][0], point[0]); np.testing.assert_allclose(observed[0][1], normal)

    def test_parked_co_holder_is_invalid_input_not_unreachable(self):
        sweep = dict(joint="lever_hinge", to=.5, co_sides=["left"])
        with self.assertRaisesRegex(ValueError, "not in contact with the moving part"):
            self.run_search(sweep=sweep)
        with patch("tool_library.skills.flat_hand_pose.flat_hand_pose") as flat:
            with self.assertRaisesRegex(ValueError, "not in contact with the moving part"):
                reachable_contact_search(self.path, "right", {}, geom="lever_panel", sweep=sweep)
            flat.assert_not_called()

    def test_physics_required_diagnosis_reexecutes_prefix_and_rebuilds_state(self):
        r = diagnose_stage3_failure(outcomes=[dict(index=3, motion="hold", side="left", object="flexible_end",
            ok=False, failed=["state_requires_physics"], error="prior contact has no predicted terminal state")],
            check_hand_scenes=False)
        self.assertIn("state_requires_physics", [x["class"] for x in r["classes"]])
        execute = next(x for x in r["repairs"] if x["tool"] == "execute_subgoal_plan")
        self.assertFalse(execute["args"]["dry_run"])
        self.assertTrue(execute["args"]["resume"])
        self.assertNotIn("from_subgoal", execute["args"])
        self.assertIn("hand_scene_from_recon", [x["tool"] for x in r["repairs"]])
        self.assertNotIn("report_blocked", [x["tool"] for x in r["repairs"]])


if __name__ == "__main__":
    unittest.main()
