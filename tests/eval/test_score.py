from __future__ import annotations

import unittest

from eval.score import unjudged_counts


class TestUnjudgedCounts(unittest.TestCase):
    def test_counts_missing_ids_per_question(self) -> None:
        run = {"q001": ["a", "b", "c"], "q002": ["d", "e"]}
        qrels = {
            "q001": {"judgments": {"a": 3, "b": 0}},
            "q002": {"judgments": {"d": 1, "e": 2}},
        }
        self.assertEqual(unjudged_counts(run, qrels),
                         {"q001": 1, "q002": 0})

    def test_question_absent_from_qrels_all_unjudged(self) -> None:
        run = {"q003": ["x", "y"]}
        self.assertEqual(unjudged_counts(run, {}), {"q003": 2})

    def test_empty_run(self) -> None:
        self.assertEqual(unjudged_counts({}, {}), {})


if __name__ == "__main__":
    unittest.main()
