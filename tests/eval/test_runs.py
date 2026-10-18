from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from eval.runs import (
    collect_run,
    load_run,
    normalize_retrieval,
    run_filename,
    save_run,
    unjudged_counts,
)


def _doc(doc_id: str, title: str = "T"):
    return SimpleNamespace(metadata={"document_id": doc_id, "title": title})


class TestNormalizeRetrieval(unittest.TestCase):
    def test_document_pairs_and_duplicates(self) -> None:
        results = [
            (_doc("a"), 0.9),
            (_doc("a"), 0.8),
            (_doc("b", "B"), 0.7),
            ({"document_id": "c", "title": "C", "score": 0.6}, 0.0),
            ({"title": "no id"}, 0.0),
        ]
        normalized = normalize_retrieval(results)
        self.assertEqual(
            [item["document_id"] for item in normalized], ["a", "b", "c"]
        )
        self.assertEqual(normalized[1]["title"], "B")

    def test_bare_id_pairs(self) -> None:
        self.assertEqual(
            normalize_retrieval([("x", 1.0)]),
            [{"document_id": "x", "title": "", "score": 1.0}],
        )


class TestCollectRun(unittest.TestCase):
    def test_passes_k_explicitly(self) -> None:
        calls = []

        def retrieve(*, query: str, k: int):
            calls.append((query, k))
            return [(_doc("a"), 1.0)]

        run = collect_run(
            "dense",
            [
                {"id": "q001", "question": "one"},
                {"id": "q002", "question": "two"},
            ],
            7,
            retrieve_fn=retrieve,
        )
        self.assertEqual(calls, [("one", 7), ("two", 7)])
        self.assertEqual(run["q001"][0]["document_id"], "a")


class TestSaveLoad(unittest.TestCase):
    def test_round_trip_and_unjudged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp)
            run = {"q001": [{"document_id": "a", "title": "A", "score": 1.0}]}
            qrels = {"q001": {"judgments": {}}}
            path = save_run("dense", 10, run, qrels, runs_dir=runs_dir)
            self.assertEqual(path.name, run_filename("dense", 10))
            payload = load_run(path)
            self.assertEqual(payload["config"]["system"], "dense")
            self.assertEqual(payload["config"]["k"], 10)
            self.assertEqual(payload["n_retrieved"], 1)
            self.assertEqual(payload["n_unjudged"], 1)
            self.assertEqual(unjudged_counts(payload["run"], qrels), {"q001": 1})


if __name__ == "__main__":
    unittest.main()
