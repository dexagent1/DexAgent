"""Contact at a failed joint goal is not necessarily an obstacle to that goal."""
import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root: runs as a script too
from tool_library.common.mechanism_contact import joint_contact_directions
from tool_library.common import outcome as O
from tool_library.tests.test_mechanism_contact_drive import fixture


class JointContactDirectionTests(unittest.TestCase):
    def test_floor_support_blocks_descent_but_not_lift(self):
        m = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <geom name="floor" type="plane" size="1 1 .1"/>
          <body name="part" pos="0 0 .0095"><joint name="slide" type="slide" axis="0 0 1"/>
          <geom type="box" size=".02 .02 .01"/></body></worldbody></mujoco>''')
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        for target, opposed in ((.1, False), (-.1, True)):
            rows = joint_contact_directions(m, d, m.joint("slide").id, target, {m.body("part").id})
            self.assertTrue(rows)
            self.assertTrue(all(r["opposes_target"] == opposed for r in rows))
            self.assertTrue(all(not r["movable"] for r in rows))
            report = O.evaluate(m, np.tile(d.qpos, (5, 1)), 30., list(range(5)),
                                {"joint": "slide", "joint_target": target})
            self.assertFalse(report["ok"])
            self.assertEqual(bool(report["measured"].get("joint_blocked_by")), opposed)

    def test_rotated_hinge_contact_direction_matches_real_constraint_torque(self):
        for rotation in ((0., 0., 0.), (.7, -.4, 1.2)):
            for face in (-1, 1):
                m, d, _, _, _ = fixture("hinge", rotation, face, friction=0.)
                effort = float(d.qfrc_constraint[0])
                self.assertGreater(abs(effort), 1e-6)
                for target in (-.3, .3):
                    rows = joint_contact_directions(m, d, m.joint("joint").id, target,
                                                    {m.body("panel").id})
                    self.assertTrue(rows)
                    self.assertTrue(all(r["opposes_target"] == (effort * target < 0) for r in rows))

    def test_motion_in_an_ancestor_makes_other_contact_movable(self):
        m = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body name="part"><joint name="slide" type="slide" axis="1 0 0"/>
          <geom type="sphere" size=".02"/></body>
          <body name="carrier" pos=".039 0 0"><freejoint/>
          <body name="pad"><geom type="sphere" size=".02"/></body></body>
          </worldbody></mujoco>''')
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        rows = joint_contact_directions(m, d, m.joint("slide").id, .1, {m.body("part").id})
        self.assertTrue(rows)
        self.assertTrue(all(r["movable"] and r["opposes_target"] for r in rows))
        self.assertEqual(joint_contact_directions(m, d, m.joint("slide").id, .1,
                         {m.body("part").id}, {m.body("pad").id}), [])


if __name__ == "__main__":
    unittest.main()
