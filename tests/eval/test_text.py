from __future__ import annotations

import unittest

from eval.text import (
    match_rank,
    normalize_title,
    normalize_whitespace,
    quote_is_grounded,
)


class TestNormalizeTitle(unittest.TestCase):
    def test_case_underscore_punct_fold(self) -> None:
        self.assertEqual(normalize_title("Knight_Artorias"), "knight artorias")
        self.assertEqual(
            normalize_title("Manus, Father of the Abyss"),
            "manus father of the abyss",
        )


class TestMatchRank(unittest.TestCase):
    def test_exact(self) -> None:
        self.assertEqual(match_rank("Dark", "Dark"), 0)
        self.assertEqual(match_rank("Manus", "manus"), 0)

    def test_short_entity_exact_only(self) -> None:
        self.assertIsNone(match_rank("Dark", "Dark Souls"))
        self.assertIsNone(match_rank("Sif", "Great Grey Wolf Sif"))
        self.assertIsNone(match_rank("Gwyn", "Soul of Gwyn, Lord of Cinder"))

    def test_whole_word_containment(self) -> None:
        self.assertEqual(match_rank("Artorias", "Knight Artorias"), 1)
        self.assertEqual(
            match_rank("Great Grey Wolf", "Great Grey Wolf Sif"), 1
        )

    def test_substring_inside_word_does_not_match(self) -> None:
        self.assertIsNone(match_rank("Light", "Sunlight Blade"))
        self.assertIsNone(match_rank("Cycle", "Cycles"))


class TestQuotes(unittest.TestCase):
    def test_whitespace_normalized_substring(self) -> None:
        section = "Gwyn found the First Flame.\nHe linked it."
        self.assertTrue(quote_is_grounded("Gwyn found   the First Flame.", section))
        self.assertTrue(quote_is_grounded("Gwyn found the First Flame. He linked it.", section))
        self.assertFalse(quote_is_grounded("Gwyn was a dragon.", section))

    def test_normalize_whitespace(self) -> None:
        self.assertEqual(normalize_whitespace(" a\n\t b  c "), "a b c")


if __name__ == "__main__":
    unittest.main()
