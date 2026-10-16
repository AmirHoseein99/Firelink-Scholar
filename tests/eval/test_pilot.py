from __future__ import annotations

import unittest

from eval.pilot import build_report, evaluate_gates, retrieve_ids


class _Doc:
    def __init__(self, doc_id: str, title: str):
        self.page_content = "text"
        self.metadata = {"document_id": doc_id, "title": title}


def _pool(*pairs: tuple[str, list[str]]) -> list[dict]:
    return [{"document_id": d, "title": d.upper(), "sources": s}
            for d, s in pairs]


def _fixtures() -> dict:
    dense_ids = [f"dn{i}" for i in range(1, 8)]
    q2_dense = [f"q2x{i}" for i in range(1, 7)]
    human = {
        "q039": {"m1": 3, "m2": 2, "m3": 3, "junk": 0},
        "q002": {"a1": 3, "a2": 0},
    }
    v3_pools = {
        "q039": _pool(
            ("m1", ["human"]), ("m2", ["human"]),
            ("m3", ["human", "gold"]), ("tpl", ["template"]),
            *[(d, ["dense"]) for d in dense_ids]),
        "q002": _pool(
            ("a1", ["human", "gold"]), ("a2", ["human"]),
            ("dn2", ["dense"]),
            *[(d, ["dense"]) for d in q2_dense]),
    }
    v3_judgments = {
        "q039": {
            "judgments": {"m1": 3, "m2": 2, "m3": 3, "junk": 0, "tpl": 1,
                          **{d: 0 for d in dense_ids}},
            "probe_llm": {"m1": 3, "m2": 2},
        },
        "q002": {
            "judgments": {"a1": 3, "a2": 0, "dn2": 1,
                          **{d: 0 for d in q2_dense}},
            "probe_llm": {"a1": 3, "a2": 1},
        },
    }
    v2_pools = {
        # m3 (human-relevant) was outside the v2 pool: the recall hole
        "q039": {"m1", "m2", *dense_ids},
        "q002": {"a1", "a2", "dn2", *q2_dense},
    }
    v2_judgments = {
        "q039": {"m1": 3, "m2": 2, **{d: 0 for d in dense_ids}},
        "q002": {"a1": 3, "a2": 0, "dn2": 1, **{d: 0 for d in q2_dense}},
    }
    run = {
        "q039": [("m1", "M1", 0.9), ("m2", "M2", 0.8), ("m3", "M3", 0.7),
                 *[(d, d.upper(), 0.5) for d in dense_ids]],
        "q002": [("a1", "A1", 0.9), ("a2", "A2", 0.8), ("dn2", "D2", 0.7),
                 *[(d, d.upper(), 0.5) for d in q2_dense]],
    }
    return {"human": human, "v3_pools": v3_pools, "v3_judgments": v3_judgments,
            "v2_pools": v2_pools, "v2_judgments": v2_judgments, "run": run}


def _questions() -> list[dict]:
    return [
        {"id": "q039", "question": "Manus legacy?", "category": "multihop"},
        {"id": "q002", "question": "Artorias?", "category": "entity"},
    ]


def _build(f: dict | None = None) -> dict:
    f = f or _fixtures()
    return build_report(
        _questions(), f["run"], f["v3_pools"], f["v3_judgments"],
        f["human"], f["v2_pools"], f["v2_judgments"])


class TestBuildReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = _build()

    def test_per_id_flags(self) -> None:
        q039 = self.report["per_question"]["q039"]
        by_id = {r["doc_id"]: r for r in q039["retrieved"]}
        self.assertFalse(by_id["m3"]["in_v2_pool"])
        self.assertTrue(by_id["m1"]["in_v2_pool"])
        self.assertEqual(by_id["m3"]["human_grade"], 3)
        self.assertEqual(by_id["m2"]["v2_llm_grade"], 2)
        self.assertIn("gold", by_id["m3"]["v3_sources"])
        self.assertTrue(by_id["m1"]["judged_anywhere"])

    def test_summaries(self) -> None:
        s = self.report["per_question"]["q039"]["summary"]
        self.assertEqual(s["retrieved_count"], 10)
        self.assertEqual(s["n_not_in_v2_pool"], 1)
        self.assertEqual(s["n_not_judged_anywhere"], 0)
        self.assertAlmostEqual(s["v2_pool_relevant_recall"], 2 / 3)
        self.assertAlmostEqual(s["v3_seed_relevant_recall"], 1.0)
        self.assertEqual(s["template_contribution"], 1)
        self.assertEqual(s["human_relevant_total"], 3)

    def test_probe_agreement_pooled(self) -> None:
        pairs = self.report["probe_agreement"]
        # q039: (3,3), (2,2); q002: (3,3), (0,1)
        self.assertEqual(pairs["pooled"]["n_pairs"], 4)
        self.assertAlmostEqual(pairs["pooled"]["exact"], 0.75)
        self.assertAlmostEqual(pairs["pooled"]["binary"], 1.0)


class TestGates(unittest.TestCase):
    def test_all_pass_on_good_fixtures(self) -> None:
        gates = _build()["gates"]
        self.assertTrue(gates["q039_recovery"]["pass"])
        self.assertTrue(gates["baseline_coverage"]["pass"])
        self.assertTrue(gates["template_pool_sanity"]["pass"])
        self.assertTrue(gates["grade_sanity"]["pass"])
        self.assertTrue(gates["no_regression"]["pass"])

    def test_q039_recovery_fails_when_relevant_missing(self) -> None:
        f = _fixtures()
        f["v3_pools"]["q039"] = [
            c for c in f["v3_pools"]["q039"] if c["document_id"] != "m3"]
        self.assertFalse(_build(f)["gates"]["q039_recovery"]["pass"])

    def test_no_regression_fails_on_overwritten_human_grade(self) -> None:
        f = _fixtures()
        f["v3_judgments"]["q039"]["judgments"]["m1"] = 1
        self.assertFalse(_build(f)["gates"]["no_regression"]["pass"])

    def test_baseline_coverage_fails_on_unjudged_v2(self) -> None:
        f = _fixtures()
        f["v2_judgments"]["q002"] = {"a1": 3}
        self.assertFalse(_build(f)["gates"]["baseline_coverage"]["pass"])

    def test_template_junk_cap_fails(self) -> None:
        f = _fixtures()
        f["v3_pools"]["q002"] = _pool(
            *[("junk-tpl", ["template"]) for _ in range(81)])
        self.assertFalse(_build(f)["gates"]["template_pool_sanity"]["pass"])

    def test_evaluate_gates_direct(self) -> None:
        gates = evaluate_gates({}, 0.9, True)
        self.assertTrue(gates["grade_sanity"]["pass"])
        self.assertTrue(gates["no_regression"]["pass"])
        gates = evaluate_gates({}, 0.5, False)
        self.assertFalse(gates["grade_sanity"]["pass"])
        self.assertFalse(gates["no_regression"]["pass"])


class TestRetrieveIds(unittest.TestCase):
    def test_tuple_pairs(self) -> None:
        items = [(_Doc("id-1", "Title 1"), 0.9), (_Doc("id-2", "Title 2"), 0.5)]
        self.assertEqual(retrieve_ids(lambda q, k: items, "q", 2),
                         [("id-1", "Title 1", 0.9), ("id-2", "Title 2", 0.5)])

    def test_skips_bare_items(self) -> None:
        self.assertEqual(retrieve_ids(lambda q, k: ["raw string"], "q", 1), [])


if __name__ == "__main__":
    unittest.main()
