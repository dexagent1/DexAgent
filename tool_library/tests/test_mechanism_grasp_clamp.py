"""Mechanism stress tools must physically execute the same clamp as Stage 3."""
import copy
import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MUJOCO_GL", "osmesa")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import grasp_robust_fixtures as F
from tool_library.common import hand as H


class MechanismClampTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="mechanism_clamp_")
        path = Path(cls.tmp.name) / "slider.xml"
        path.write_text('<mujoco><worldbody><body name="part" pos="0 0 .025">'
                        '<joint name="slide" type="slide" axis="1 0 0" range="-.1 .1" damping="1"/>'
                        '<geom type="box" size=".025 .02 .025" mass=".12"/>'
                        '</body></worldbody></mujoco>')
        cls.scene = F.scene(objects=[{"name": "fixture", "xml": str(path), "pos": [0., 0., 0.]}])
        cls.model, cls.data, cls.rig, cls.ids = H.open_scene(cls.scene)
        cls.pose = F.pose(F.BOX_PINCH3)
        cls.tools = [(name, getattr(importlib.import_module("tool_library.verifiers." + name), name)) for name in
                     ("grasp_turn_test", "grasp_turn_test_keep_state", "grasp_slide_pull_test", "grasp_slide_pull_test_preshaped")]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_tool(self, name, fn, pose):
        commands = []
        native = H.hold_ctrl

        def record(data, rig, pos, euler, fingers):
            commands.append((np.asarray(pos, float).copy(), dict(fingers or {})))
            return native(data, rig, pos, euler, fingers)

        distance = {"sweep_m": .001} if "turn" in name else {"travel_m": .001}
        with patch.object(H, "executor_clamp", wraps=H.executor_clamp) as shared, patch.object(H, "hold_ctrl", side_effect=record):
            report = fn(self.scene, pose, "fixture_slide", seconds=.04, **distance)
        self.assertEqual(shared.call_count, 1)
        self.assertIsInstance(report["ok"], bool)
        return commands

    def test_guarded_opposed_clamp_is_sent_to_all_four_physical_servos(self):
        expected = H.executor_clamp(self.rig, self.pose)["press"]
        raw = H.press_targets(self.rig, self.pose["finger_targets"], self.pose["grip_fingers"],
                              depth_rad=self.pose["press_depth_rad"], squeezed=self.pose["finger_targets_squeezed"])
        self.assertTrue(any(abs(expected[k] - raw[k]) > 1e-6 for k in expected))
        for name, fn in self.tools:
            with self.subTest(tool=name):
                commands = self.run_tool(name, fn, self.pose)
                self.assertTrue(any(all(abs(values[k] - expected[k]) < 1e-12 for k in expected) for _, values in commands))

    def test_one_sided_wrist_press_is_integrated_and_opening_keeps_contact_shape(self):
        pose = copy.deepcopy(self.pose)
        pose["grip_fingers"] = ["index"]
        pose.pop("finger_targets_open", None)
        pose.pop("finger_targets_squeezed", None)
        closed = pose["finger_targets"]
        curl = H.press_targets(self.rig, closed, ["index"], depth_rad=pose["press_depth_rad"], max_travel_m=H.ONE_SIDED_PRESS_M)
        shift = H.pad_shift_world(self.rig, closed, curl, pose["euler"])["index"]
        self.assertGreater(np.linalg.norm(shift), 1e-5)
        pose["normals"] = {"index": (shift / np.linalg.norm(shift)).tolist()}
        clamp = H.executor_clamp(self.rig, pose)
        self.assertTrue(clamp["grip_posed"])
        self.assertAlmostEqual(np.linalg.norm(clamp["wrist_press"]), H.ONE_SIDED_PRESS_M)
        pressed = np.asarray(pose["pos"]) + clamp["wrist_press"]
        for name, fn in self.tools:
            with self.subTest(tool=name):
                commands = self.run_tool(name, fn, pose)
                self.assertTrue(any(np.max(np.abs(pos - pressed)) < 1e-12 for pos, _ in commands))
                # open_scene first parks the hand while settling the object;
                # inspect the actual approach command, not that initializer.
                pre = np.asarray(pose["pos"]) + .06 * np.asarray(pose["approach"])
                first = next(values for pos, values in commands if np.max(np.abs(pos - pre)) < 1e-10)
                for key in closed:
                    if H.finger_of_joint(key) == "index":
                        self.assertAlmostEqual(first[key], closed[key], places=12)


if __name__ == "__main__":
    unittest.main()
