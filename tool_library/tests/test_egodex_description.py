"""Reversible EgoDex recordings must select the direction recorded in the HDF5."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tool_library.skills.read_egodex_episode import episode_description


class DescriptionTests(unittest.TestCase):
    descriptions = {"llm_description": "Insert a plushie into a drawer.",
                    "llm_description2": "Remove a plushie from a drawer."}

    def test_selected_reverse_description(self):
        self.assertEqual(episode_description({**self.descriptions, "which_llm_description": "2"}),
                         self.descriptions["llm_description2"])

    def test_bytes_and_first_description(self):
        self.assertEqual(episode_description({**self.descriptions, "which_llm_description": b"1"}),
                         self.descriptions["llm_description"])

    def test_ambiguous_fallback_fails(self):
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            episode_description(self.descriptions)

    def test_single_description_fallback(self):
        self.assertEqual(episode_description({"llm_description": "Move the cup."}), "Move the cup.")

    def test_none_placeholder_is_not_a_second_description(self):
        # one-direction episodes (e.g. assemble_jenga) store the string "None" as llm_description2
        self.assertEqual(episode_description({"llm_description": "Assemble a Jenga tower.", "llm_description2": "None"}),
                         "Assemble a Jenga tower.")
        self.assertEqual(episode_description({"llm_description": "Stack.", "llm_description2": b"none"}), "Stack.")

    def test_invalid_selector_and_missing_selected_description_fail(self):
        for data in [{**self.descriptions, "which_llm_description": "0"},
                     {"llm_description": "Insert", "which_llm_description": "2"}, {}]:
            with self.assertRaises(ValueError):
                episode_description(data)


if __name__ == "__main__":
    unittest.main()
