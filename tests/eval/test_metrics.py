from __future__ import annotations

import unittest

from eval.metrics import (
    claim_coverage_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    score_run,
)

QRELS = {
    "q001": {
        "judgments": {"a": 3, "b": 2, "c": 0, "d": 1},
        "supports_claims": {"a": ["c1"], "b": ["c2"]},
    }
}
QUESTIONS = {
    "q001": {
        "id": "q001",
        "category": "lore",
        "claims": [
            {"id": "c1", "importance": "core"},
            {"id": "c2", "importance": "supporting"},
        ],
    }
}
ANSWER = {
    "claims": [
        {"id": "c1", "importance": "core"},
        {"id": "c2", "importance": "supporting"},
    ]
}


def _results(ids: list[str]) -> list[dict]:
    return [{"document_id": doc_id} for doc_id in ids]


class TestRankMetrics(unittest.TestCase):
    def test_ndcg_rewards_grade_order(self) -> None:
        self.assertGreater(
            ndcg_at_k(["a", "c"], QRELS["q001"]["judgments"], 5),
            ndcg_at_k(["c", "a"], QRELS["q001"]["judgments"], 5),
        )
        self.assertAlmostEqual(ndcg_at_k([], QRELS["q001"]["judgments"], 5), 0.0)

    def test_binary_metrics(self) -> None:
        judgments = QRELS["q001"]["judgments"]
        self.assertAlmostEqual(recall_at_k(["a", "c"], judgments, 5, 2), 0.5)
        self.assertAlmostEqual(mrr_at_k(["c", "a"], judgments, 5, 2), 0.5)
        self.assertAlmostEqual(precision_at_k(["a", "c", "b"], judgments, 5, 2), 0.4)
        self.assertIsNone(recall_at_k(["a"], {}, 5, 2))

    def test_claim_coverage(self) -> None:
        entry = QRELS["q001"]
        self.assertAlmostEqual(
            claim_coverage_at_k(["a"], entry, ANSWER, 5, importances=("core",)), 1.0
        )
        self.assertAlmostEqual(
            claim_coverage_at_k(["a"], entry, ANSWER, 5, importances=("core", "supporting")),
            0.5,
        )
        self.assertAlmostEqual(
            claim_coverage_at_k(["c"], entry, ANSWER, 5, importances=("core",)), 0.0
        )


class TestScoreRun(unittest.TestCase):
    def test_per_question_keys_are_question_ids(self) -> None:
        result = score_run(
            {"q001": _results(["a", "c"])},
            QRELS,
            QUESTIONS,
            {"q001": ANSWER},
            ks=[5, 10],
            min_grade=2,
        )
        self.assertEqual(list(result["per_question"]), ["q001"])
        self.assertEqual(result["per_question"]["q001"]["category"], "lore")
        self.assertIn("ndcg@5", result["per_question"]["q001"]["metrics"])
        self.assertAlmostEqual(result["coverage"], 1.0)
        self.assertFalse(result["partial"])

    def test_unjudged_ids_score_zero_and_mark_partial(self) -> None:
        result = score_run(
            {"q001": _results(["a", "zzz"])},
            QRELS,
            QUESTIONS,
            {"q001": ANSWER},
            ks=[5],
            min_grade=2,
        )
        self.assertTrue(result["partial"])
        self.assertEqual(result["unjudged"]["n_ids"], 1)
        self.assertEqual(result["unjudged"]["by_question"], {"q001": 1})
        self.assertEqual(result["per_question"]["q001"]["unjudged_ids"], ["zzz"])

    def test_by_category(self) -> None:
        result = score_run(
            {"q001": _results(["a"])},
            QRELS,
            QUESTIONS,
            {"q001": ANSWER},
            ks=[5],
            min_grade=2,
        )
        self.assertIn("lore", result["by_category"])
        self.assertEqual(result["by_category"]["lore"]["n"], 1)


if __name__ == "__main__":
    unittest.main()
