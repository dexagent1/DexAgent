import unittest
from pathlib import Path

from agent.library import _json_type, load_index, tool_schema


class LibrarySchemaTypesTests(unittest.TestCase):
    def test_containers_do_not_inherit_nested_scalar_types(self):
        self.assertEqual(_json_type('dict[str, bool]'), {'type': 'object'})
        self.assertEqual(_json_type('list[dict[str, float | bool]]')['type'], 'array')
        self.assertNotEqual(_json_type('callable(params: dict) -> {gates: dict[str, bool]}').get('type'), 'boolean')

    def test_top_level_union_preserves_inline_and_file_inputs(self):
        self.assertEqual({s['type'] for s in _json_type('dict | str | list')['anyOf']}, {'object', 'string', 'array'})
        self.assertEqual({s['type'] for s in _json_type('bool | None')['anyOf']}, {'boolean', 'null'})
        self.assertEqual(_json_type('int | float'), {'type': 'number'})

    def test_executor_plan_and_pose_schemas_accept_all_documented_forms(self):
        index = load_index(Path(__file__).resolve().parents[2] / 'tool_library')
        tools = {t['name']: t for t in index['tools']}
        schema = tool_schema(tools['execute_subgoal_plan'])['input_schema']['properties']['plan']
        self.assertEqual({s['type'] for s in schema['anyOf']}, {'object', 'array', 'string'})
        pose = tool_schema(tools['optimize_pose_for_reach'])['input_schema']['properties']['pose']
        self.assertEqual({s['type'] for s in pose['anyOf']}, {'object', 'string'})


if __name__ == '__main__':
    unittest.main()
