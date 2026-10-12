"""Our own sample in Stage 3: the example's assets are fitted to the run's video, the positional difference against the
example is always applied to its trajectory (zero included), and the stage fails only when the shifted trajectory is
out of reach; the delivery is one Stage 4 accepts. A recorded twin episode is published only for undisplaced objects."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent import reuse
from agent.artifacts import accepted_delivery
from tool_library.common import examples as EX

OBJS = [{"id": "cup", "root_body": "cup_cup", "pose": {"x": 0.1, "y": 0.4, "yaw_deg": 0.0}},
        {"id": "bowl", "root_body": "bowl_bowl", "pose": {"x": -0.05, "y": 0.42, "yaw_deg": 0.0}}]


class ExampleTrajectoryReuseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.ex = root / "examples" / "place_cup_in_bowl"
        (self.ex / "stage3" / "stage4_task").mkdir(parents=True)
        (self.ex / "stage2").mkdir()
        (self.ex / "stage3" / "trajectory.npz").write_bytes(b"example trajectory")
        (self.ex / "stage3" / "stage4_task" / "stage4.json").write_text(json.dumps({"scene": "scene.xml", "trajectory": "trajectory.npz",
                                                                                   "randomize": {"objects": {"cup_cup": {}, "bowl_bowl": {}}}}))
        (self.ex / "stage2" / "scene_recon.json").write_text(json.dumps({"objects": OBJS}))
        self.recon = root / "run" / "stage2" / "scene_recon.json"
        self.recon.parent.mkdir(parents=True)
        self.recon.write_text(json.dumps({"objects": OBJS}))
        self.public = root / "run" / "stage3"
        self.ws = self.public / ".candidates" / "a"
        self.ws.mkdir(parents=True)
        self.cfg = {"run": {"name": "t"}, "examples": {"reuse": True}}

    def _copy(self, example, stage, ws, video=None):
        task = Path(ws) / "stage4_task"
        task.mkdir(exist_ok=True)
        (task / "stage4.json").write_text(json.dumps({"scene": "scene.xml", "trajectory": "trajectory.npz"}))
        (task / "trajectory.npz").write_bytes(b"example trajectory")
        (Path(ws) / "trajectory.npz").write_bytes(b"example trajectory")
        return {"files": [str(Path(ws) / "trajectory.npz")]}

    def run_with(self, layout, adapt=None):
        match = {"match": "identical", "example": str(self.ex), "task": "place_cup_in_bowl"}
        with patch.object(reuse, "lookup", return_value=match), \
             patch.object(EX, "fit_layout", return_value=layout) as fit, \
             patch.object(EX, "reuse_example", side_effect=self._copy), \
             patch.object(EX, "adapt_example", return_value=adapt or {}) as ad, \
             patch.object(reuse, "_install_shifted") as inst:
            out = reuse.stage3_example_trajectory(self.cfg, self.ws, self.public, "video.mp4", "sem.json", self.recon)
        if out.get("finished"):
            (self.public / "run.json").write_text(json.dumps(out))
        self.assertEqual(fit.call_args.kwargs["objects"], {"cup_cup", "bowl_bowl"})   # the movable objects are fitted
        return out, ad, inst, json.loads((self.ws / "reuse.json").read_text())

    def test_unmoved_objects_still_go_through_the_shift(self):
        layout = {"objects": {"cup_cup": [0.0, 0.0, 0.0], "bowl_bowl": [0.0, 0.0, 0.0]}, "unmatched": [], "fits": {}}
        out, adapt, install, decision = self.run_with(layout, adapt={"failed": [], "report": {}})
        self.assertTrue(out["finished"])
        self.assertEqual(decision["mode"], "shifted")     # never "seen it, reuse it": the zero difference is applied too
        self.assertEqual(adapt.call_args.args[1], layout)
        install.assert_called_once()
        self.assertIsNotNone(accepted_delivery(self.public))   # the Stage 4 handoff accepts it

    def test_moved_objects_shift_the_trajectory(self):
        out, adapt, install, decision = self.run_with({"objects": {"cup_cup": [0.03, 0.0, 0.0], "bowl_bowl": [0.0, 0.0, 0.0]}, "unmatched": []},
                                                      adapt={"failed": ["joint_speed", "penetration"], "report": {}})
        self.assertTrue(out["finished"])                  # only reach can fail a shifted example
        self.assertEqual(decision["mode"], "shifted")
        self.assertEqual(adapt.call_args.kwargs["checks"], reuse.ONLY_REACH)
        install.assert_called_once()

    def test_out_of_reach_fails_the_stage(self):
        out, _, install, decision = self.run_with({"objects": {"cup_cup": [0.4, 0.25, 0.0], "bowl_bowl": [0.0, 0.0, 0.0]}, "unmatched": []},
                                                  adapt={"failed": ["ik"], "report": {"ik": {"ok": False}}})
        self.assertFalse(out["finished"])
        self.assertEqual(decision["mode"], "failed")
        install.assert_not_called()
        self.assertFalse((self.public / "accepted.json").exists())

    def test_other_objects_leave_the_stage_to_the_agent(self):
        self.recon.write_text(json.dumps({"objects": OBJS[:1] + [{"id": "plate", "root_body": "plate_plate", "pose": {"x": 0, "y": 0}}]}))
        with patch.object(reuse, "lookup", return_value={"match": "same_task", "example": str(self.ex), "task": "x"}):
            self.assertIsNone(reuse.stage3_example_trajectory(self.cfg, self.ws, self.public, "v.mp4", "s.json", self.recon))


class TwinEpisodeReuseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.ex = root / "examples" / "scissor_cutting"
        (self.ex / "stage3").mkdir(parents=True)
        (self.ex / "stage2").mkdir()
        (self.ex / "stage3" / "twin_episode.json").write_text(json.dumps({"kind": "Isaac Sim twin episode", "files": {}}))
        (self.ex / "stage2" / "scene_recon.json").write_text(json.dumps({"objects": OBJS}))
        self.recon = root / "run" / "stage2" / "scene_recon.json"
        self.recon.parent.mkdir(parents=True)
        self.recon.write_text(json.dumps({"objects": OBJS}))
        self.public = root / "run" / "stage3"
        self.ws = self.public / ".candidates" / "a"
        self.ws.mkdir(parents=True)
        self.cfg = {"run": {"name": "t"}, "examples": {"reuse": True}}

    def _copy(self, example, stage, ws, video=None):
        (Path(ws) / "sim_video.mp4").write_bytes(b"twin video")
        return {"files": [str(Path(ws) / "sim_video.mp4")]}

    def run_with(self, layout):
        match = {"match": "identical", "example": str(self.ex), "task": "scissor_cutting"}
        with patch.object(reuse, "lookup", return_value=match), \
             patch.object(EX, "fit_layout", return_value=layout) as fit, \
             patch.object(EX, "reuse_example", side_effect=self._copy) as copy:
            out = reuse.stage3_twin_episode(self.cfg, self.ws, self.public, "video.mp4", "sem.json", self.recon)
        fit.assert_called_once()                          # the position fit always runs first
        if out and out.get("finished"):
            (self.public / "run.json").write_text(json.dumps(out))
        return out, copy

    def test_undisplaced_objects_publish_the_twin(self):
        out, copy = self.run_with({"objects": {"cup_cup": [0.0004, 0.0, 0.1], "bowl_bowl": [0.0, 0.0, 0.0]}, "unmatched": [], "fits": {}})
        self.assertTrue(out["finished"])
        decision = json.loads((self.ws / "reuse.json").read_text())
        self.assertEqual(decision["mode"], "twin_episode")
        self.assertIn("cup_cup", decision["object_displacement"])
        bundle = accepted_delivery(self.public)
        self.assertEqual((bundle / "sim_video.mp4").read_bytes(), b"twin video")

    def test_displaced_objects_leave_the_stage_to_the_agent(self):
        out, copy = self.run_with({"objects": {"cup_cup": [0.03, 0.0, 0.0], "bowl_bowl": [0.0, 0.0, 0.0]}, "unmatched": [], "fits": {}})
        self.assertIsNone(out)
        copy.assert_not_called()
        self.assertFalse((self.public / "accepted.json").exists())
        self.assertFalse(json.loads((self.ws / "twin_episode_check.json").read_text())["used"])


if __name__ == "__main__":
    unittest.main()
