"""Support belongs to the connected object, not only its free-joint body."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tool_library.verifiers.support_check import support_check


class SupportTreeTests(unittest.TestCase):
    def test_child_contact_supports_free_root(self):
        xml = '''<mujoco><worldbody><geom type="plane" size="1 1 .1"/>
        <body name="object" pos="0 0 .12"><freejoint/><geom type="sphere" size=".02" mass=".05"/>
        <body name="foot" pos="0 0 -.10"><geom type="box" size=".1 .1 .02" mass=".1"/></body>
        </body></worldbody></mujoco>'''
        r = support_check(xml, seconds=.1)
        self.assertTrue(r['ok'], r)
        self.assertTrue(any('foot' in pair for pair in r['free_object_contacts']['object']))

    def test_self_contact_does_not_count_as_support(self):
        xml = '''<mujoco><option gravity="0 0 0"/><worldbody>
        <body name="object" pos="0 0 .4"><freejoint/><geom name="g1" type="sphere" size=".02" mass=".05"/>
        <body name="part" pos="0 .03 0"><joint axis="0 0 1"/><geom name="g2" type="sphere" size=".02" mass=".05"/></body>
        </body></worldbody><contact><pair geom1="g1" geom2="g2"/></contact></mujoco>'''
        r = support_check(xml, seconds=0)
        self.assertFalse(r['ok'], r)
        self.assertEqual(r['floating'], ['object'])


if __name__ == '__main__':
    unittest.main()
