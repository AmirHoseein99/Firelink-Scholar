from __future__ import annotations

import unittest

from eval.agreement import collect, pair_metrics, probe_pairs


def _entry(pairs: list[tuple[int, int]]) -> dict:
    judgments = {f"d{i}": h for i, (h, _) in enumerate(pairs)}
    probe_llm = {f"d{i}": l for i, (_, l) in enumerate(pairs)}
    return {"judgments": judgments, "probe_llm": probe_llm}


class TestPairMetrics(unittest.TestCase):
    def test_exact_and_binary(self) -> None:
        m = pair_metrics([(3, 3), (0, 0), (2, 1), (1, 0)])
        self.assertEqual(m["n_pairs"], 4)
        self.assertAlmostEqual(m["exact"], 0.5)
        # binary cutoff grade >= 2: (2,1) disagree, (1,0) agree
        self.assertAlmostEqual(m["binary"], 0.75)

    def test_confusion_counts(self) -> None:
        m = pair_metrics([(0, 1), (0, 1), (3, 3)])
        self.assertEqual(m["confusion"], {"0->1": 2, "3->3": 1})

    def test_empty(self) -> None:
        m = pair_metrics([])
        self.assertEqual(m["n_pairs"], 0)
        self.assertIsNone(m["exact"])


class TestCollect(unittest.TestCase):
    def test_collect_pools_and_filters(self) -> None:
        judgments = {
            "q001": _entry([(3, 3), (0, 1)]),
            "q002": _entry([(2, 2)]),
            "q003": {"judgments": {"d0": 3}},  # no probe_llm -> excluded
        }
        report = collect(judgments)
        self.assertEqual(set(report["per_question"]), {"q001", "q002"})
        self.assertEqual(report["pooled"]["n_pairs"], 3)

        only = collect(judgments, {"q002"})
        self.assertEqual(set(only["per_question"]), {"q002"})

    def test_probe_pairs_skips_unjudged(self) -> None:
        entry = {"judgments": {"d0": 3}, "probe_llm": {"d0": 2, "d1": 1}}
        self.assertEqual(probe_pairs(entry), [(3, 2)])


if __name__ == "__main__":
    unittest.main()
