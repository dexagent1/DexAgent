"""Generic physical folded-arm overshoot plus timing-policy unit negatives.

The physical fixture is a joint grid and quintic sweep, independent of sample
plans. Policy/archival negatives use injected reports and are not physics proof.
"""
import copy
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root: runs as a script too
from tool_library.common import arm_timing as AT, armhand as AH
from tool_library.common import episode as E, rigid_robot as RR
from tool_library.common import outcome as O
from tool_library.skills import execute_subgoal_plan as SP


class ArmTimingRetryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, _ = E.build_execution_model(RR.SETUP / "scene.xml")
        cls.runtime = RR.compile_rigid(cls.model)
        cls.parts = E.robot_parts(cls.model)
        cls.work = Path(tempfile.mkdtemp(prefix="dexagent_arm_timing_"))
        print(f"Arm timing regression artifacts: {cls.work}", flush=True)

    def reference(self):
        m = self.runtime.model
        cols = E.robot_parts(m)["left"]["arm_qadr"]
        q = m.key_qpos[0].copy()
        q[cols] = np.radians([0., 150., 165., -30., -15., -60.])
        u = np.linspace(0., 1., 46)
        u = u ** 3 * (10. - 15. * u + 6. * u * u)
        Q = np.tile(q, (62, 1))
        Q[:46, cols] += u[:, None] * np.radians([0., -30., -60., 30., 0., 0.])
        Q[46:] = Q[45]
        proposal = self.runtime.to_proposal_qpos(Q)
        arms = {s: proposal[:, p["arm_qadr"]].copy() for s, p in self.parts.items()}
        return {"qpos": proposal, "arm": arms, "labels": ["left: carry fixture"] * 46 + ["left: hold fixture"] * 16,
                "src_index": np.arange(len(Q)), "report": {"ok": True, "infeasible": [], "retimed_s": 0.,
                       **{s: {"branch_jumps": [], "peak_speed_deg_s": float(np.degrees(np.abs(np.diff(a, axis=0))).max() * 30.)}
                          for s, a in arms.items()}}}

    def test_subdivision_preserves_all_vertices_fixed_mount_and_phase_windows(self):
        cp = self.reference()
        original = copy.deepcopy(cp)
        slow = AT.slow_reference(self.model, cp, 30., 3)
        np.testing.assert_array_equal(self.runtime.to_runtime_qpos(slow["qpos"])[::3],
                                      self.runtime.to_runtime_qpos(cp["qpos"]))
        self.runtime.check_reference(slow["qpos"])
        np.testing.assert_array_equal(cp["qpos"], original["qpos"])
        self.assertEqual(slow["labels"][::3], cp["labels"])
        self.assertLessEqual(slow["report"]["left"]["peak_speed_deg_s"], 25.1)
        exp = {"no_slip": True, "pads_at_label": "left: hold fixture"}
        O.resolve_label_frames(exp, slow["labels"])
        self.assertEqual(exp["slip_from_rel"], 0)
        self.assertEqual(exp["pads_at_rel"], len(slow["qpos"]) - 1)
        numbered = {"hands": {"left": {"slip_from_rel": 10, "seated": {"frames_rel": [12, 18]}}}}
        self.assertEqual(AT.scale_expectation_frames(numbered, 3)["hands"]["left"],
                         {"slip_from_rel": 30, "seated": {"frames_rel": [36, 54]}})

    def test_retry_policy_rejects_other_failures_branches_and_exhausted_budget(self):
        cp = self.reference()
        ev = {"failed": ["arm_speed"], "measured": {"arm_speed": {"left": 270.}}}
        self.assertEqual(AT.retry_factor(ev, cp, 135., 0, 1), 3)
        for failure in ("hand_penetration", "no_slip", "arm_collision", "servo_tracking"):
            self.assertIsNone(AT.retry_factor(dict(ev, failed=["arm_speed", failure]), cp, 135., 0, 1))
        self.assertIsNone(AT.retry_factor(ev, cp, 135., AT.MAX_RETRIES, 1))
        self.assertIsNone(AT.retry_factor(ev, cp, 135., 0, 0))
        cp["report"]["left"]["branch_jumps"] = [20]
        self.assertIsNone(AT.retry_factor(ev, cp, 135., 0, 1))
        with self.assertRaisesRegex(ValueError, "discontinuous"):
            AT.slow_reference(self.model, cp, 30., 2)
        with self.assertRaisesRegex(ValueError, "limit"):
            AT.slow_reference(self.model, self.reference(), 30., 3, max_frames=100)

    def test_between_frame_velocity_spike_and_position_fallback(self):
        # Injected evidence: output endpoints recover, but physical samples move.
        m = self.runtime.model
        part = E.robot_parts(m)["left"]
        velocities = np.zeros((2, m.nv))
        velocities[0, part["arm_dadr"][0]] = np.radians(200.)
        diag = {"qpos": np.tile(m.key_qpos[0], (2, 1)), "initial_qvel": np.zeros(m.nv),
                "substep_qvel": velocities, "substep_time_s": np.array([.002, .004])}
        measured = AT.runtime_speed_peaks(m, diag, 30.)["left"]
        self.assertAlmostEqual(measured["peak_speed_deg_s"], 200.)
        self.assertEqual(measured["speed_sample"], 0)
        self.assertEqual(measured["speed_frame"], 1)
        del diag["substep_qvel"]
        diag["initial_qpos"] = m.key_qpos[0].copy()
        positions = np.tile(m.key_qpos[0], (2, 1))
        positions[0, part["arm_qadr"][0]] += np.radians(200.) * .002
        diag["substep_qpos"] = positions
        self.assertAlmostEqual(AT.runtime_speed_peaks(m, diag, 30.)["left"]["peak_speed_deg_s"], 200.)

    def test_failed_trial_archive_does_not_overwrite_previous_evidence(self):
        directory = self.work / "archive_test"
        directory.mkdir()
        payload = b"injected archival fixture, not a physical recording"
        (directory / "rollout_runtime.npz").write_bytes(payload)
        ev = {"failed": ["arm_speed"], "measured": {"arm_speed": {"left": 200.}}}
        first = AT.archive_failed_trial(directory, ev, self.reference(), 2, 1)
        second = AT.archive_failed_trial(directory, ev, self.reference(), 2, 1)
        self.assertNotEqual(first, second)
        self.assertEqual((Path(first) / "rollout_runtime.npz").read_bytes(), payload)
        self.assertEqual(json.loads((Path(first) / "outcome.json").read_text())["evaluation"], ev)

    def exercise_executor(self, name, fail_second=False, always_fast=False, max_attempts=6):
        """Actual wait physics; injected speed diagnostics test retry orchestration."""
        original_rollout, original_evaluate = SP.rollout_keyframes_physics, O.evaluate
        trials, evaluations = [], []
        def rollout(*args, **kwargs):
            result = original_rollout(*args, **kwargs)
            diag = result["_diag"]
            trials.append({"state_arg": kwargs.get("state_start"), "initial": diag["initial_state"].copy(),
                           "mechanism_arg": kwargs.get("mechanism_state_start"),
                           "mechanism_initial": copy.deepcopy(diag["mechanism_initial"])})
            if len(trials) == 1 or always_fast:
                dof = E.robot_parts(RR.compile_rigid(args[0]).model)["left"]["arm_dadr"][0]
                diag["substep_qvel"][0, dof] = np.radians(200.)
            return result
        def evaluate(*args, **kwargs):
            result = original_evaluate(*args, **kwargs)
            evaluations.append(copy.deepcopy(result))
            if fail_second and len(evaluations) == 2:
                result["ok"] = False
                result["checks"]["hand_penetration"] = False
                result["failed"].append("hand_penetration")
            return result
        directory = self.work / name
        with patch.object(SP, "rollout_keyframes_physics", side_effect=rollout), patch.object(O, "evaluate", side_effect=evaluate):
            result = SP.execute_subgoal_plan(str(RR.SETUP / "scene.xml"),
                     {"subgoals": [{"index": 1, "motion": "wait", "side": "left", "duration_s": .1}]},
                     str(directory), plan_arm_paths=False, max_attempts=max_attempts)
        return result, trials, evaluations, directory

    def test_executor_retries_same_complete_state_and_keeps_bad_evidence(self):
        result, trials, evaluations, directory = self.exercise_executor("executor_pass")
        self.assertTrue(result["ok"], result.get("first_failure"))
        self.assertEqual(len(trials), 2)
        self.assertEqual(len(evaluations), 2)
        np.testing.assert_array_equal(trials[1]["state_arg"], trials[0]["initial"])
        np.testing.assert_array_equal(trials[1]["initial"], trials[0]["initial"])
        self.assertEqual(trials[1]["mechanism_arg"], trials[0]["mechanism_initial"])
        saved = list((directory / "subgoal_01/timing_attempts").glob("*/rollout_runtime.npz"))
        self.assertEqual(len(saved), 1)
        self.assertTrue((saved[0].parent / "outcome.json").is_file())
        self.assertEqual(len(result["subgoals"][0]["measured"]["arm"]["timing_retries"]), 1)

    def test_executor_rechecks_other_gates_and_bounds_trials(self):
        result, trials, evaluations, _ = self.exercise_executor("executor_new_failure", fail_second=True)
        self.assertFalse(result["ok"])
        self.assertEqual(result["first_failure"]["failed"], ["hand_penetration"])
        self.assertEqual(len(trials), 2)
        result, trials, _, _ = self.exercise_executor("executor_bounded", always_fast=True)
        self.assertFalse(result["ok"])
        self.assertEqual(len(trials), AT.MAX_RETRIES + 1)
        result, trials, _, _ = self.exercise_executor("executor_budget", max_attempts=1)
        self.assertFalse(result["ok"])
        self.assertEqual(len(trials), 1)

    def test_physical_lag_overshoot_replays_slower_from_identical_state(self):
        cp = self.reference()
        m = self.runtime.model
        Q = self.runtime.to_runtime_qpos(cp["qpos"])
        parts = E.robot_parts(m)
        cols = parts["left"]["arm_qadr"]
        self.assertLess(cp["report"]["left"]["peak_speed_deg_s"], 90.)
        probe = AH.ArmProbe(RR.SETUP / "scene.xml")
        self.assertFalse(probe.check(cp["qpos"], cp["labels"], stride=1))
        baseline = RR.servo_rollout(m, Q, 30.)
        speed = AT.runtime_speed_peaks(m, baseline, 30.)["left"]["peak_speed_deg_s"]
        lag = float(np.degrees(np.abs(baseline["qpos"][:, cols] - Q[:, cols])).max())
        self.assertGreater(speed, 200.)
        self.assertGreater(lag, 15.)
        ev = {"failed": ["arm_speed"], "measured": {"arm_speed": {"left": speed}}}
        factor = AT.retry_factor(ev, cp, 135., 0, 1)
        slow = AT.slow_reference(self.model, cp, 30., factor)
        new_Q = self.runtime.to_runtime_qpos(slow["qpos"])
        retimed = RR.servo_rollout(m, new_Q, 30., state_start=baseline["initial_state"],
                                   mechanism_state_start=baseline["mechanism_initial"])
        np.testing.assert_array_equal(retimed["initial_state"], baseline["initial_state"])
        new_speed = AT.runtime_speed_peaks(m, retimed, 30.)["left"]["peak_speed_deg_s"]
        self.assertLess(new_speed, 135.)
        (self.work / "scene_physics.xml").write_text(self.runtime.xml)
        reports = {}
        for name, trial, ref, labels in (("baseline", baseline, Q, cp["labels"]),
                                          ("retimed", retimed, new_Q, slow["labels"])):
            self.assertIsNone(trial["unstable"])
            self.assertLess(float(trial["hand_pen_m"].max()), .003)
            self.assertLess(float(trial["object_pen_m"].max()), .003)
            self.assertLess(float(trial["track_pos_m"].max()), .025)
            self.assertLess(float(trial["track_rot_deg"].max()), 15.)
            proposed = self.runtime.to_proposal_qpos(trial["qpos"])
            self.assertFalse(probe.check(proposed, labels, stride=1))
            arrays = RR.episode_arrays(m, trial, ref)
            arrays.update(labels=np.asarray(labels), fps=30.)
            np.savez_compressed(self.work / f"{name}.npz", **arrays)
            reports[name] = {"peak_speed_deg_s": speed if name == "baseline" else new_speed,
                             "max_track_mm": 1000 * float(trial["track_pos_m"].max()),
                             "max_track_deg": float(trial["track_rot_deg"].max())}
        (self.work / "report.json").write_text(json.dumps(reports, indent=1))


if __name__ == "__main__":
    unittest.main()
