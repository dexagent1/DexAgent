"""Handoff must not reinsert segment-boundary frames removed by execution."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # the repository: runs as a script too
from tool_library.skills.write_stage4_config import _subgoals, _recorded_releases  # noqa: E402


class ExecutionWindowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, rows):
        (self.path / "outcomes.json").write_text(json.dumps(rows))

    def test_joined_windows_and_release_events_share_the_recorded_timeline(self):
        rows = [dict(index=1, n_frames=10, frames=[0, 9], stage1_indices=[1, 2]),
                dict(index=3, n_frames=6, frames=[10, 14], released=[dict(equality="latch", frame=2)]),
                dict(index=4, n_frames=4, frames=[15, 17], released=[dict(equality="join", frame=0)])]
        self.write(rows)
        goals = _subgoals(None, self.path)
        self.assertEqual([g["frames"] for g in goals], [[0, 9], [10, 14], [15, 17]])
        self.assertEqual(goals[0]["stage1_indices"], [1, 2])
        self.assertEqual(_recorded_releases(self.path), [dict(equality="latch", frame=11),
                                                       dict(equality="join", frame=14)])

    def test_legacy_unduplicated_lengths_remain_supported(self):
        self.write([dict(index=1, n_frames=4), dict(index=2, n_frames=3)])
        self.assertEqual([g["frames"] for g in _subgoals(None, self.path)], [[0, 3], [4, 6]])

    def test_invalid_joined_windows_reject(self):
        for frames in ([5, 7], [4, 4], [3, 5], [4, 8]):
            self.write([dict(index=1, n_frames=4, frames=[0, 3]), dict(index=2, n_frames=3, frames=frames)])
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                _subgoals(None, self.path)


if __name__ == "__main__":
    unittest.main()
