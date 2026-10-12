"""Actual Sharpa mesh regressions for grasp fixture and idle-finger validity."""
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import grasp_robust_fixtures as F
import mujoco
from tool_library.common import hand as H


class GraspSelfCollisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.data, cls.rig, cls.ids = H.open_scene(F.scene())

    def depth(self, pose, targets):
        H.apply_pose(self.model, self.data, self.rig, pose, targets)
        return H.hand_self_overlap(self.model, self.data, self.rig)

    def test_former_positive_fixture_has_crossed_visible_fingers(self):
        pose = F.pose("box_right_crossed_invalid")
        self.assertGreater(self.depth(pose, pose["finger_targets"]), .010)
        # Fingertip/object exceptions must never excuse hand/hand contact.
        self.assertGreater(H.deepest_overlap(self.model, self.data, self.rig, self.ids), .010)

    def test_generated_positive_open_close_and_clamp_are_self_clear(self):
        pose = F.pose(F.BOX_PINCH3)
        targets = {"open": pose["finger_targets_open"], "closed": pose["finger_targets"],
                   "press": H.executor_clamp(self.rig, pose)["press"]}
        for name, values in targets.items():
            with self.subTest(phase=name):
                self.assertLess(self.depth(pose, values), .0015)
        self.assertNotEqual(targets["open"], targets["closed"])

    def test_idle_curl_avoids_self_contact_without_changing_active_opening(self):
        pose = F.pose(F.BOX_PINCH3)
        closed = pose["finger_targets"]
        idle = {"ring", "pinky"}
        opened = H.open_targets(self.rig, closed, .1, curl=idle)
        self.assertLess(self.depth(pose, opened), .0015)
        for key, value in closed.items():
            if H.finger_of_joint(key) in idle:
                continue
            lo, _ = self.rig.finger_range[key]
            expected = value + (lo - value) * .1 if H.joint_role(key) in ("flex", "pinky_arch") else value
            self.assertAlmostEqual(opened[key], expected, places=14)
        unsafe = dict(opened)
        for key in unsafe:
            if H.finger_of_joint(key) in idle and H.joint_role(key) in ("flex", "pinky_arch"):
                unsafe[key] = self.rig.finger_range[key][1]
        self.assertGreater(self.depth(pose, unsafe), .005)

    def test_planning_without_contact_pairs_uses_the_same_opening_and_clamp(self):
        spec = mujoco.MjSpec.from_file(str(ROOT / "setup/robot/sharpa_right.xml"))
        H.add_hand_self_contacts(spec)
        paired = spec.compile()
        for pair in list(spec.pairs):
            if (pair.name or "").startswith("handself__"):
                spec.delete(pair)
        unpaired = spec.compile()
        self.assertGreater(paired.npair, unpaired.npair)
        rigs = [H.HandRig.build(m, "", "right") for m in (paired, unpaired)]
        pose = F.pose(F.BOX_PINCH3)
        opened = [H.approach_open_targets(r, pose["finger_targets"], .15, pose["grip_fingers"])
                  for r in rigs]
        self.assertEqual(opened[0], opened[1])
        self.assertEqual(H.executor_clamp(rigs[0], pose), H.executor_clamp(rigs[1], pose))
        invalid = F.pose("box_right_crossed_invalid")
        for model, rig in zip((paired, unpaired), rigs):
            data = mujoco.MjData(model)
            H.apply_pose(model, data, rig, invalid)
            self.assertGreater(H.hand_self_overlap(model, data, rig), .010)


if __name__ == "__main__":
    unittest.main()
