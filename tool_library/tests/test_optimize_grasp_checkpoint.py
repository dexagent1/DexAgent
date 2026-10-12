"""Cooperative search stops preserve work without turning cached claims into PASS."""
import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MUJOCO_GL", "egl")
M = importlib.import_module("tool_library.skills.optimize_grasp_pose")
P = {"pos": [0, 0, .1], "euler": [0, 0, 0], "source": "generic test"}


def failed(job):
    return {"index": job["index"], "start": job["start_index"], "theta": job["theta"],
            "pose": dict(job["start"]), "score": -sum(abs(x) for x in job["theta"].values()),
            "ok": False, "stage": "tests", "reason": "grasp_lift_test:lift", "tests": {}}


def mock_binding(inputs, *args):
    return json.loads(json.dumps(M._jsonable(inputs)))


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.marker = self.path / "stop"
        self.journal = self.path / "search.json"
        self.args = dict(hand_scene={"objects": []}, starts=P, family="seeds", workers=1,
                         max_evals=12, population=4, iterations=3, seed=41,
                         checkpoint_json=str(self.journal), checkpoint_batch_size=2, stop_file=str(self.marker))
        self.binding_patch = patch.object(M, "_checkpoint_binding", side_effect=mock_binding)
        self.binding_patch.start()
        self.addCleanup(self.binding_patch.stop)

    def call(self, **kw):
        return M.optimize_grasp_pose(**dict(self.args, **kw))

    def stop_after(self, count, result=failed):
        called = []
        def evaluate(job):
            called.append(job)
            if len(called) == count:
                self.marker.touch()
            return result(job)
        with patch.object(M, "evaluate", side_effect=evaluate):
            out = self.call()
        return out, called

    def test_resumed_search_matches_uninterrupted_and_reuses_failed_work(self):
        out, called = self.stop_after(3)
        self.assertEqual(out["status"], "incomplete")
        self.assertFalse(out["ok"])
        self.assertIsNone(out["pose"])
        self.assertEqual(out["passing"], [])
        self.assertEqual(len(called), 4)  # finish the in-flight two-job batch
        record = json.loads(self.journal.read_text())
        state = record["calls"][0]["optimizer_state"]
        self.assertEqual(set(("mu", "sig", "rng_state", "starts")) - set(state), set())
        self.marker.unlink()
        with patch.object(M, "evaluate", side_effect=failed) as evaluate:
            resumed = self.call(resume=True)
        self.assertEqual(evaluate.call_count, 8)
        with patch.object(M, "evaluate", side_effect=failed):
            baseline = self.call(checkpoint_json=str(self.path / "baseline.json"))
        for key in ("ok", "pose", "best_tries", "n_evals", "iterations", "failure_classes", "passing"):
            self.assertEqual(resumed[key], baseline[key], key)
        self.assertEqual(resumed["status"], "budget_exhausted")
        self.assertEqual(json.loads(self.journal.read_text())["calls"],
                         json.loads((self.path / "baseline.json").read_text())["calls"])

    def test_changed_inputs_reject_before_worker_start(self):
        self.stop_after(1)
        self.marker.unlink()
        for kw in ({"max_evals": 13}, {"seed": 42}, {"test_args": {"grasp_lift_test": {"settle_s": 9}}},
                   {"starts": dict(P, pos=[1, 0, .1])}):
            with self.subTest(kw=kw), patch.object(M, "_Pool") as pool:
                with self.assertRaisesRegex(ValueError, "checkpoint rejected"):
                    self.call(resume=True, **kw)
                pool.assert_not_called()

    def test_existing_stop_starts_no_workers_and_resume_does_not_reset_budget(self):
        self.marker.touch()
        with patch.object(M, "_Pool") as pool:
            out = self.call()
            pool.assert_not_called()
        self.assertEqual(out["n_evals"], 0)
        self.assertFalse(out["complete"])
        self.marker.unlink()
        with patch.object(M, "evaluate", side_effect=failed) as evaluate:
            out = self.call(resume=True)
        self.assertEqual(evaluate.call_count, 12)
        self.assertEqual(out["n_evals"], 12)

    def test_pass_in_stopped_batch_is_not_output_and_is_rechecked_on_resume(self):
        def pass_result(job):
            return dict(failed(job), ok=True, score=20, reason=None)
        self.args.update(max_evals=2, out_json=str(self.path / "pose.json"))
        out, _ = self.stop_after(1, result=pass_result)
        self.assertFalse(out["ok"])
        self.assertFalse((self.path / "pose.json").exists())
        self.marker.unlink()
        with patch.object(M, "evaluate", side_effect=failed) as evaluate:
            resumed = self.call(resume=True)
        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual(resumed["cached_evaluations_rechecked"], 2)
        self.assertFalse(resumed["ok"])
        self.assertFalse((self.path / "pose.json").exists())

    def test_forged_pass_is_rechecked_and_modified_job_rejected(self):
        self.args.update(max_evals=2)
        self.stop_after(1)
        record = json.loads(self.journal.read_text())
        record["calls"][0]["results"][0].update(ok=True, score=999)
        self.journal.write_text(json.dumps(record))
        self.marker.unlink()
        with patch.object(M, "evaluate", side_effect=failed) as evaluate:
            out = self.call(resume=True)
        self.assertFalse(out["ok"])
        self.assertEqual(evaluate.call_count, 1)
        record["calls"][0]["jobs"][0]["start"]["pos"][0] = 100
        self.journal.write_text(json.dumps(record))
        with patch.object(M, "_Pool") as pool:
            with self.assertRaisesRegex(ValueError, "candidate jobs differ"):
                self.call(resume=True)
            pool.assert_not_called()

    def test_seed_generation_is_checkpointed_without_regenerating_completed_jobs(self):
        self.args.update(family="bar", starts=None, max_evals=2)
        jobs = [{"kind": "start", "slot": i} for i in range(3)]
        generated = []
        def evaluate(job):
            if job["kind"] == "start":
                generated.append(job["slot"])
                if len(generated) == 1:
                    self.marker.touch()
                return {"starts": [dict(P, source=str(job["slot"]))]}
            return failed(job)
        with patch.object(M, "family_jobs", return_value=jobs), patch.object(M, "evaluate", side_effect=evaluate):
            out = self.call(checkpoint_batch_size=1)
            self.assertEqual(out["completed_seed_jobs"], 1)
            self.marker.unlink()
            out = self.call(resume=True, checkpoint_batch_size=1)
        self.assertEqual(generated, [0, 1, 2])
        self.assertEqual(out["n_evals"], 2)
        self.assertTrue(out["complete"])

    def test_actual_worker_processes_close_normally_after_stop(self):
        pool = M._CheckpointPool(self.journal, False, self.marker, 2, {}, 2, None, 2)
        real_run = pool._run
        processes = []
        def run(jobs):
            results = real_run(jobs)
            processes.extend(p for p, log in pool.pool.procs)
            self.marker.touch()
            return results
        with patch.object(pool, "_run", side_effect=run):
            with self.assertRaises(M._SearchStopped):
                pool.map([{"kind": "invalid"}, {"kind": "invalid"}])
        with patch.object(processes[0], "kill", side_effect=AssertionError("must close normally")), \
                patch.object(processes[1], "kill", side_effect=AssertionError("must close normally")):
            pool.close()
        self.assertEqual([p.returncode for p in processes], [0, 0])
        self.assertTrue(all(p.stdin.closed and p.stdout.closed for p in processes))

    def test_resume_flags_require_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "require checkpoint_json"):
            self.call(checkpoint_json=None)
        with self.assertRaisesRegex(ValueError, "positive"):
            self.call(checkpoint_batch_size=0)


class BindingAndLauncherTest(unittest.TestCase):
    def test_real_passing_pose_is_reverified_after_cooperative_stop(self):
        from tool_library.tests import grasp_robust_fixtures as F
        with tempfile.TemporaryDirectory() as tmp:
            marker, journal, pose = [Path(tmp) / name for name in ("stop", "search.json", "pose.json")]
            args = dict(hand_scene=F.scene(), starts=F.pose(F.BOX_PINCH3), family="seeds", workers=1,
                        max_evals=1, tests=["grasp_lift_test"], checkpoint_json=str(journal),
                        checkpoint_batch_size=1, stop_file=str(marker), out_json=str(pose))
            actual = M.evaluate
            def stop_after_evaluation(job):
                result = actual(job)
                marker.touch()
                return result
            with patch.object(M, "evaluate", side_effect=stop_after_evaluation):
                stopped = M.optimize_grasp_pose(**args)
            self.assertFalse(stopped["ok"])
            self.assertFalse(pose.exists())
            self.assertTrue(json.loads(journal.read_text())["calls"][0]["results"][0]["ok"])
            marker.unlink()
            with patch.object(M, "evaluate", wraps=actual) as evaluate:
                resumed = M.optimize_grasp_pose(**args, resume=True)
            self.assertEqual(evaluate.call_count, 1)
            self.assertTrue(resumed["ok"], resumed)
            self.assertTrue(pose.exists())
            self.assertTrue(resumed["pose"]["stress_tests"]["grasp_lift_test"]["ok"])

    def test_compiled_physics_and_source_code_changes_invalidate(self):
        from tool_library.tests import grasp_robust_fixtures as F
        with tempfile.TemporaryDirectory() as tmp:
            scene = F.box_variant(tmp)
            inputs = {"scene": scene}
            before = M._checkpoint_binding(inputs, scene, None, None)
            same = M._checkpoint_binding(inputs, scene, None, None)
            self.assertEqual(before, same)
            F.box_variant(tmp, mass=.3)
            changed = M._checkpoint_binding(inputs, scene, None, None)
            self.assertNotEqual(before["details"]["physics"]["hand_model"], changed["details"]["physics"]["hand_model"])
            self.assertNotEqual(before["sha256"], changed["sha256"])
            original = Path.read_bytes
            def changed_code(path):
                return original(path) + (b"\n# changed" if path.resolve() == Path(M.__file__).resolve() else b"")
            with patch.object(Path, "read_bytes", new=changed_code):
                code_change = M._checkpoint_binding(inputs, scene, None, None)
            self.assertNotEqual(changed["details"]["code"], code_change["details"]["code"])
            self.assertNotEqual(changed["sha256"], code_change["sha256"])

    def test_launcher_restores_original_order_for_eval_and_seed_jobs(self):
        class Process:
            def __init__(self, command, **kw):
                spec = json.loads(Path(command[-2]).read_text())
                Path(command[-1]).write_text(json.dumps([{"slot": job["slot"]} for job in spec["jobs"]]))
            def communicate(self):
                return "", ""
        with tempfile.TemporaryDirectory() as tmp, patch.object(M.tempfile, "mkdtemp", return_value=tmp), \
                patch.object(M.subprocess, "Popen", new=Process):
            for kind in ("eval", "start"):
                results = M._launch([{"kind": kind, "slot": i} for i in range(7)], ["wrapper"], 3, 2)
                self.assertEqual([r["slot"] for r in results], list(range(7)))


if __name__ == "__main__":
    unittest.main()
