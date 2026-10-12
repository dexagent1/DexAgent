"""Final rest cannot disguise a dropped or displaced placement."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mujoco
import numpy as np
from tool_library.common import outcome as O


class ReleaseSettleOutcomeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <geom type="plane" size="1 1 .1"/>
          <body name="item" pos="0 0 .01"><freejoint/>
            <geom type="box" size=".02 .015 .01" mass=".05"/>
          </body></worldbody></mujoco>''')

    def check_motion(self, displacement):
        Q = np.tile(self.model.qpos0, (31, 1))
        Q[:7, :3] += np.linspace(np.asarray(displacement), np.zeros(3), 7)
        return O.evaluate(self.model, Q, 30, list(range(len(Q))),
                          {"object": "item", "released": True, "at_rest": True})

    def test_stable_placement_passes(self):
        report = self.check_motion([0, 0, 0])
        self.assertTrue(report["checks"]["release_settle"], report)

    def test_final_rest_does_not_excuse_large_fall_or_slide(self):
        for delta in ([0, 0, .045], [.04, 0, 0]):
            with self.subTest(delta=delta):
                report = self.check_motion(delta)
                self.assertTrue(report["checks"]["at_rest"], report)
                self.assertFalse(report["checks"]["release_settle"], report)
                self.assertIn("release_settle", report["failed"])


if __name__ == "__main__":
    unittest.main()
