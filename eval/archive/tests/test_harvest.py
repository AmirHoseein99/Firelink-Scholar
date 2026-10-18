from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eval.harvest import (
    candidate_from_chunk,
    harvest_run,
    new_ids_for_question,
)


def _chunk(doc_id: str, title: str = "Some Page") -> dict:
    return {
        "document_id": doc_id,
        "title": title,
        "section": "lead",
        "game": "Dark Souls",
        "entity_type": "item",
        "page_content": "content " + doc_id,
    }


class TestNewIds(unittest.TestCase):
    def test_skips_judged_known_and_unresolved(self) -> None:
        run_ids = ["a", "b", "c", "unresolved:x"]
        judged = {"a": 3}
        pool = {"b"}
        self.assertEqual(new_ids_for_question(run_ids, judged, pool), ["c"])

    def test_empty(self) -> None:
        self.assertEqual(
            new_ids_for_question(["a"], {"a": 1}, {"a"}), [])


class TestCandidateFromChunk(unittest.TestCase):
    def test_shape_and_truncation(self) -> None:
        chunk = _chunk("x")
        chunk["page_content"] = "y" * 3000
        cand = candidate_from_chunk("x", chunk)
        self.assertEqual(cand["sources"], ["run"])
        self.assertEqual(cand["document_id"], "x")
        self.assertEqual(len(cand["snippet"]), 1000)
        self.assertIsNone(cand["dense_score"])


class TestHarvestRun(unittest.TestCase):
    def test_extends_pool_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cand_dir = Path(tmp)
            pool_path = cand_dir / "q001.json"
            pool_path.write_text(json.dumps(
                [{"document_id": "known", "title": "T", "section": "s",
                  "game": "g", "entity_type": "e", "snippet": "x",
                  "sources": ["dense"], "dense_score": 0.5,
                  "bm25_score": None}]
            ))
            chunks_by_id = {
                "new1": _chunk("new1"), "new2": _chunk("new2", "Other"),
            }
            run = {"q001": ["known", "new1", "new2", "gone"]}
            judged = {"q001": {"known": 2}}

            summary = harvest_run(run, judged, chunks_by_id, cand_dir)
            self.assertEqual(summary["q001"]["n_new"], 2)
            self.assertEqual(summary["q001"]["n_missing"], 1)
            pool = json.loads(pool_path.read_text())
            self.assertEqual([c["document_id"] for c in pool],
                             ["known", "new1", "new2"])
            self.assertEqual(pool[1]["sources"], ["run"])

            again = harvest_run(run, judged, chunks_by_id, cand_dir)
            self.assertEqual(again, {})
            pool = json.loads(pool_path.read_text())
            self.assertEqual(len(pool), 3)

    def test_creates_missing_pool_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cand_dir = Path(tmp)
            chunks_by_id = {"n1": _chunk("n1")}
            summary = harvest_run(
                {"q009": ["n1"]}, {}, chunks_by_id, cand_dir)
            self.assertEqual(summary["q009"]["n_new"], 1)
            pool = json.loads((cand_dir / "q009.json").read_text())
            self.assertEqual(pool[0]["document_id"], "n1")

    def test_nothing_new_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cand_dir = Path(tmp)
            summary = harvest_run({"q001": ["x"]}, {"q001": {"x": 0}},
                                  {"x": _chunk("x")}, cand_dir)
            self.assertEqual(summary, {})
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
