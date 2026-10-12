import unittest
from pathlib import Path

from agent.devtools import DevTools


class LibrarySearchTests(unittest.TestCase):
    def setUp(self):
        self.dev = DevTools.__new__(DevTools)
        self.dev.preferred_categories = ['Trajectory & Execution']
        self.dev.index = {'tools': [
            dict(name='build_container', type='skill', category='Object Reconstruction',
                 summary='make a shape ' * 200, description='primitive solid', inputs=[]),
            dict(name='plan_contact_path', type='skill', category='Trajectory & Execution',
                 summary='Plan movement.', description='Find reachable contact waypoints for a mechanism sweep.', inputs=[]),
            dict(name='validate_contact_path', type='verifier', category='Episode Audits',
                 summary='Check contact.', description='Audit movement.', inputs=[]),
        ]}

    def names(self, query, kind='any'):
        return [t['name'] for t in self.dev.library_search(query, kind)['tools']]

    def test_description_capability_beats_repeated_generic_query_words(self):
        self.assertEqual(self.names('make a reachable mechanism sweep')[0], 'plan_contact_path')

    def test_exact_name_wins_even_outside_stage_categories(self):
        self.assertEqual(self.names('build_container')[0], 'build_container')

    def test_type_filter_and_whole_words(self):
        self.assertEqual(self.names('contact', 'verifier'), ['validate_contact_path'])
        self.assertEqual(self.names('lid'), [])  # not the substring in 'solid' or 'validate'
        self.assertEqual(self.names(''), [])

    def test_real_execution_and_contact_tools_are_discoverable(self):
        dev = DevTools(Path(__file__).resolve().parents[2] / 'tool_library', Path('/tmp'))
        dev.preferred_categories = ['Grasp & Poses', 'Trajectory & Execution']
        for query, expected in [('plan execute subgoal trajectory', 'execute_subgoal_plan'),
                                ('grasp surface contact reachable arm', 'reachable_contact_search'),
                                ('render trajectory video', 'render_trajectory_video')]:
            with self.subTest(query=query):
                self.assertIn(expected, [t['name'] for t in dev.library_search(query)['tools'][:3]])


if __name__ == '__main__':
    unittest.main()
