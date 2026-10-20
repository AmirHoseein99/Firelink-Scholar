from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import call, patch

from src.retriever import querying


def _doc(doc_id: str):
    return SimpleNamespace(metadata={"document_id": doc_id})


def _ranked(*doc_ids: str):
    return [(_doc(doc_id), 1.0) for doc_id in doc_ids]


class TestReciprocalRankScore(unittest.TestCase):
    def test_fuses_ranks_and_skips_missing_ids(self) -> None:
        fused = querying.reciprocal_rank_score(
            [
                _ranked("a", "b"),
                [(_doc("a"), 1.0), (SimpleNamespace(metadata={}), 1.0), (_doc("b"), 1.0)],
            ]
        )
        self.assertEqual([doc_id for doc_id, _ in fused], ["a", "b"])
        self.assertGreater(fused[0][1], fused[1][1])

    def test_weights_change_order(self) -> None:
        fused = querying.reciprocal_rank_score(
            [_ranked("a"), _ranked("b")],
            weights=[3.0, 1.0],
        )
        self.assertEqual([doc_id for doc_id, _ in fused], ["a", "b"])

    def test_weight_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            querying.reciprocal_rank_score([_ranked("a")], weights=[1.0, 2.0])


class TestFuse(unittest.TestCase):
    def test_returns_top_k_documents(self) -> None:
        results = querying._fuse([_ranked("a", "b", "c")], k=2)
        self.assertEqual([doc.metadata["document_id"] for doc, _ in results], ["a", "b"])


class TestEnhancedRetrieval(unittest.TestCase):
    def test_default_skips_rewrites(self) -> None:
        with (
            patch.object(querying, "enhance_queries") as enhance,
            patch.object(
                querying, "contextfromvectors", return_value=_ranked("a", "b")
            ) as dense,
        ):
            results = querying.contextfromvectors_enhanced("original", k=2)

        enhance.assert_not_called()
        dense.assert_called_once_with(query="original", k=2)
        self.assertEqual(
            [doc.metadata["document_id"] for doc, _ in results], ["a", "b"]
        )

    def test_rewrites_keep_original_ranking(self) -> None:
        calls = []

        def dense(*, query: str, k: int):
            calls.append(query)
            return {
                "original": _ranked("original-only", "shared"),
                "combined": _ranked("shared", "combined-only"),
                "entity": _ranked("entity-only"),
            }[query]

        with (
            patch.object(
                querying,
                "enhance_queries",
                return_value=["combined", "entity"],
            ),
            patch.object(querying, "contextfromvectors", side_effect=dense),
        ):
            results = querying.contextfromvectors_enhanced(
                "original", k=3, use_rewrites=True
            )

        self.assertEqual(calls, ["original", "combined", "entity"])
        ids = [doc.metadata["document_id"] for doc, _ in results]
        self.assertIn("original-only", ids[:2])
        self.assertIn("shared", ids[:2])
        self.assertEqual(ids[-1], "entity-only")


class TestHybridEnhancedRetrieval(unittest.TestCase):
    def test_uses_rewrites_for_bm25_not_dense(self) -> None:
        with (
            patch.object(
                querying,
                "enhance_queries",
                return_value=["combined", "entity"],
            ),
            patch.object(
                querying,
                "contextfromvectors",
                return_value=_ranked("dense"),
            ) as dense,
            patch.object(
                querying,
                "bm25_retrieval",
                side_effect=[_ranked("sparse"), _ranked("e1"), _ranked("e2")],
            ) as sparse,
        ):
            results = querying.contextfromvectors_hybrid_enhanced("original", k=2)

        dense.assert_called_once_with(query="original", k=6)
        self.assertEqual(
            sparse.call_args_list,
            [
                call(query="original", k=6),
                call(query="combined", k=6),
                call(query="entity", k=6),
            ],
        )
        self.assertEqual(results[0][0].metadata["document_id"], "dense")

    def test_falls_back_to_plain_hybrid_when_rewrites_fail(self) -> None:
        with (
            patch.object(querying, "enhance_queries", return_value=["original"]),
            patch.object(
                querying,
                "contextfromvectors",
                return_value=_ranked("dense"),
            ),
            patch.object(
                querying,
                "bm25_retrieval",
                return_value=_ranked("sparse"),
            ) as sparse,
        ):
            results = querying.contextfromvectors_hybrid_enhanced("original", k=2)

        sparse.assert_called_once_with(query="original", k=6)
        self.assertEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()
