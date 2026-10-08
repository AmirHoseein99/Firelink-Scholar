from __future__ import annotations

import unittest

from eval.merge_qrels import (
    aggregate_agreement,
    build_final_entry,
    compute_agreement,
    merge_judgments,
    unknown_ids,
)


class TestMergeJudgments(unittest.TestCase):
    def test_human_overrides_llm(self) -> None:
        llm = {"a": 3, "b": 1, "c": 2}
        human = {"a": 1, "b": 2}
        merged, counts = merge_judgments(llm, human)
        self.assertEqual(merged, {"a": 1, "b": 2, "c": 2})
        self.assertEqual(counts, {"llm": 1, "human": 0, "human_override": 2})

    def test_manual_only_ids_survive(self) -> None:
        # q039-style manual `a`-added uuids absent from any pool
        llm = {"pool1": 2, "pool2": 0}
        human = {"manual-a": 3, "manual-b": 1}
        merged, counts = merge_judgments(llm, human)
        self.assertEqual(merged["manual-a"], 3)
        self.assertEqual(merged["manual-b"], 1)
        self.assertEqual(counts["human"], 2)
        self.assertEqual(counts["llm"], 2)
        self.assertEqual(counts["human_override"], 0)

    def test_empty_llm_side(self) -> None:
        merged, counts = merge_judgments({}, {"h1": 2})
        self.assertEqual(merged, {"h1": 2})
        self.assertEqual(counts, {"llm": 0, "human": 1, "human_override": 0})

    def test_empty_human_side(self) -> None:
        merged, counts = merge_judgments({"l1": 0}, {})
        self.assertEqual(merged, {"l1": 0})
        self.assertEqual(counts, {"llm": 1, "human": 0, "human_override": 0})


class TestUnknownIds(unittest.TestCase):
    def test_unknown_counted(self) -> None:
        known = {"a", "b"}
        judgments = {"a": 3, "b": 0, "typo-uuid": 2, "another": 1}
        self.assertEqual(unknown_ids(judgments, known), ["typo-uuid", "another"])


class TestAgreement(unittest.TestCase):
    def test_exact_and_binary(self) -> None:
        llm = {"a": 3, "b": 1, "c": 0, "d": 2}
        human = {"a": 3, "b": 2, "c": 1, "d": 0}
        rep = compute_agreement(llm, human)
        self.assertEqual(rep["n_common"], 4)
        # exact: only "a"; binary(>=2): a yes, b yes(llm 1 <2, human 2 >=2 -> no)
        # b: llm 1 (not rel), human 2 (rel) -> disagree; c: 0 vs 1 both <2 agree;
        # d: 2 vs 0 -> llm rel, human not -> disagree
        self.assertEqual(rep["exact_match"], 0.25)
        self.assertEqual(rep["binary_at_2"], 0.5)

    def test_confusion_matrix_cells(self) -> None:
        llm = {"a": 0, "b": 1, "c": 2, "d": 3}
        human = {"a": 0, "b": 1, "c": 2, "d": 3}
        rep = compute_agreement(llm, human)
        matrix = rep["confusion_matrix"]["matrix"]
        self.assertEqual(matrix[0][0], 1)
        self.assertEqual(matrix[1][1], 1)
        self.assertEqual(matrix[2][2], 1)
        self.assertEqual(matrix[3][3], 1)
        self.assertEqual(rep["confusion_matrix"]["rows"], "human")
        self.assertEqual(rep["confusion_matrix"]["cols"], "llm")

    def test_no_overlap(self) -> None:
        rep = compute_agreement({"a": 1}, {"b": 2})
        self.assertEqual(rep["n_common"], 0)
        self.assertIsNone(rep["exact_match"])
        self.assertIsNone(rep["binary_at_2"])

    def test_aggregate_across_questions(self) -> None:
        llm_qrels = {
            "q001": {"judgments": {"a": 3, "b": 1}},
            "q002": {"judgments": {"c": 0}},
        }
        human_qrels = {
            "q001": {"judgments": {"a": 3, "b": 2}},
            "q002": {"judgments": {"c": 0}},
        }
        agg = aggregate_agreement(llm_qrels, human_qrels)
        self.assertEqual(agg["overall"]["n_common"], 3)
        # exact: a + c = 2/3; binary: all three (3>=2, 1<2 vs 2>=2 -> b: llm<2,
        # human>=2 -> disagree) -> 2/3
        self.assertAlmostEqual(agg["overall"]["exact_match"], 2 / 3)
        self.assertAlmostEqual(agg["overall"]["binary_at_2"], 2 / 3)
        self.assertIn("q001", agg["per_question"])
        self.assertEqual(agg["overall"]["confusion_matrix"]["matrix"][0][0], 1)


class TestBuildFinalEntry(unittest.TestCase):
    def test_schema_and_ordering(self) -> None:
        merged, counts = merge_judgments(
            {"a": 1, "b": 3, "c": 0}, {"b": 2, "d": 3}
        )
        entry = build_final_entry(
            "q001",
            merged,
            counts,
            {"question": "Q?", "game": "Dark Souls", "category": "entity",
             "graded_at": "2026-01-01T00:00:00Z"},
            "glm-5.3-flash",
            n_unknown=1,
        )
        # sorted grade-desc, then doc_id
        self.assertEqual(
            list(entry["judgments"].items()),
            [("d", 3), ("b", 2), ("a", 1), ("c", 0)],
        )
        self.assertEqual(entry["grade_counts"],
                         {"0": 1, "1": 1, "2": 1, "3": 1})
        self.assertEqual(entry["n_judgments"], 4)
        self.assertEqual(entry["n_unknown_doc_ids"], 1)
        self.assertEqual(entry["source_counts"],
                         {"llm": 2, "human": 1, "human_override": 1})
        self.assertEqual(entry["model"], "glm-5.3-flash")
        self.assertEqual(entry["question"], "Q?")


if __name__ == "__main__":
    unittest.main()
