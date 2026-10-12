"""A nested input repair must invalidate a cached physical segment."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root: runs as a script too
from tool_library.common.subgoal import load_pose
from tool_library.skills.execute_subgoal_plan import _resume_signature, _pose_candidates, _load_plan


class ResumeDependencyBindingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old_cwd = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, self.old_cwd)
        self.pose = self.root / 'pose.json'
        self.pose.write_text(json.dumps({'pos': [0., 0., .1], 'euler': [0., 0., 0.]}))

    def signature(self, entry):
        return _resume_signature(entry, [], {'fps': 30.}, 'unchanged physical scene')

    def test_nested_candidate_and_grasp_wrappers_bind_actual_pose_changes(self):
        for value in (['pose.json'], {'grasps': ['pose.json']}):
            with self.subTest(value=value):
                wrapper = self.root / 'ranked.json'; wrapper.write_text(json.dumps(value))
                entry = {'index': 1, 'motion': 'grasp_lift', 'side': 'right', 'pose_candidates': str(wrapper)}
                before = self.signature(entry)
                loaded = load_pose(_pose_candidates(entry)[0])
                loaded['pos'][2] += .1
                self.pose.write_text(json.dumps(loaded))
                self.assertNotEqual(before, self.signature(entry))
                self.assertEqual(load_pose(_pose_candidates(entry)[0])['pos'], loaded['pos'])

    def test_workspace_pose_is_bound_even_when_wrapper_has_same_named_neighbor(self):
        folder = self.root / 'nested'; folder.mkdir()
        (folder / 'ranked.json').write_text('["pose.json"]')
        (folder / 'pose.json').write_text('{"pos":[0,0,9]}')
        (folder / 'plan.json').write_text(json.dumps({'subgoals': [{'index': 1, 'motion': 'grasp_lift', 'pose': 'ranked.json'}]}))
        entry = _load_plan(str(folder / 'plan.json'))[0][0]
        self.assertEqual(load_pose(_pose_candidates(entry)[0])['pos'][2], .1)
        before = self.signature(entry)
        self.pose.write_text('{"pos":[0,0,.2]}'.replace('.2', '0.2'))
        self.assertNotEqual(before, self.signature(entry))

    def test_parent_relative_files_and_cycles_are_stable_and_content_bound(self):
        folder = self.root / 'nested'; folder.mkdir()
        (folder / 'first.json').write_text('{"hand_scene":"second.json"}')
        second = folder / 'second.json'
        second.write_text('{"source":"first.json","anchor":{"pos":[0,0,1]}}')
        entry = {'hand_scene': str(folder / 'first.json')}
        before = self.signature(entry)
        self.assertEqual(before, self.signature(entry))
        second.write_text('{"source":"first.json","anchor":{"pos":[0,0,2]}}')
        self.assertNotEqual(before, self.signature(entry))

    def test_included_geometry_changes_invalidate_nested_hand_scene(self):
        xml = self.root / 'object.xml'; included = self.root / 'part.xml'
        xml.write_text('<mujoco><include file="part.xml"/></mujoco>')
        included.write_text('<mujoco><worldbody><geom size=".02"/></worldbody></mujoco>')
        scene = self.root / 'hand.json'; scene.write_text(json.dumps({'objects': [{'xml': str(xml)}]}))
        entry = {'hand_scene': str(scene)}
        before = self.signature(entry)
        included.write_text('<mujoco><worldbody><geom size=".03"/></worldbody></mujoco>')
        self.assertNotEqual(before, self.signature(entry))

    def test_unrelated_files_do_not_invalidate_and_missing_file_appearance_does(self):
        entry = {'pose': str(self.pose), 'hand_scene': 'not_created.json'}
        before = self.signature(entry)
        (self.root / 'unrelated.json').write_text('{"x":1}')
        self.assertEqual(before, self.signature(entry))
        (self.root / 'not_created.json').write_text('{"objects":[]}')
        self.assertNotEqual(before, self.signature(entry))


if __name__ == '__main__':
    unittest.main()
