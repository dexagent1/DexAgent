"""A same-shaped clip or matching filename cannot substitute for the Stage 1 source."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agent.input_identity import check_video_identity


class InputIdentityTests(unittest.TestCase):
    actual = {"path": "/video.mp4", "sha256": "a" * 64, "frames": 116, "fps": 30.,
              "duration_s": 116 / 30, "width": 1920, "height": 1080}

    def check_identity(self, semantic, expected=None):
        with patch("agent.input_identity.video_identity", return_value=self.actual):
            return check_video_identity("/video.mp4", semantic, expected_identity=expected)

    def test_matching_metadata_does_not_hide_different_take(self):
        expected = dict(self.actual, sha256="b" * 64)
        result = self.check_identity({"video": expected})
        self.assertFalse(result["ok"])
        self.assertIn("SHA-256", result["issues"][0])

    def test_missing_identity_is_not_a_pass(self):
        result = self.check_identity({"video": {"frames": 116, "fps": 30.}})
        self.assertFalse(result["ok"])
        self.assertFalse(result["identity_verified"])

    def test_same_digest_with_wrong_annotation_timing_fails(self):
        result = self.check_identity({"video": dict(self.actual, frames=107)})
        self.assertFalse(result["ok"])
        self.assertTrue(result["identity_verified"])

    def test_windows_are_checked_against_actual_source(self):
        result = self.check_identity({"video": self.actual, "subgoals": [{"index": 1, "time_window_s": [2, 4]}]})
        self.assertFalse(result["ok"])
        self.assertIn("time window", result["issues"][0])

    def test_recorded_source_identity_passes(self):
        self.assertTrue(self.check_identity({}, expected=self.actual)["ok"])


if __name__ == "__main__":
    unittest.main()
