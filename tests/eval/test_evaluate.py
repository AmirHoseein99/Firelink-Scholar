from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from eval.evaluate import evaluate
from eval.llm import Backend
from eval.prompts_grade import GRADE_PROMPT_SHA256

BACKEND = Backend("test", "http://test/v1", "x", "judge-model")


def _doc(doc_id: str):
    return SimpleNamespace(metadata={"document_id": doc_id, "title": "T " + doc_id})


def _dataset() -> dict:
    return {
        "judge": {
            "backend": "freellm",
            "base_url": "http://test/v1",
            "model": "judge-model",
            "temperature": 0,
            "prompt_sha256": GRADE_PROMPT_SHA256,
        },
        "answer_builder": {"model": "test-model", "prompt_sha256": "x"},
        "claims": {"max_single_game": 8, "max_synthesis": 12},
        "metrics": {"ks": [5, 10], "min_grade": 2},
        "retrieval": {"systems": ["dense", "hybrid"], "k": 10},
    }


def _questions() -> list[dict]:
    return [
        {
            "id": "q001",
            "question": "Who was Artorias?",
            "game": "Dark Souls",
            "category": "lore",
            "expected_entities": ["Artorias of the Abyss"],
        }
    ]


def _answers() -> dict:
    return {
        "q001": {
            "status": "verified",
            "claims": [
                {"id": "c1", "text": "core", "importance": "core"},
                {"id": "c2", "text": "support", "importance": "supporting"},
            ],
        }
    }


def _retriever_for(mapping: dict[str, list[str]]):
    def factory(system: str):
        def retrieve(*, query: str, k: int):
            return [(_doc(doc_id), 1.0) for doc_id in mapping[system][:k]]

        return retrieve

    return factory


def _chunks(ids):
    return {
        doc_id: {
            "document_id": doc_id,
            "title": "T " + doc_id,
            "game": "Dark Souls",
            "section": "lead",
            "page_content": "content " + doc_id,
        }
        for doc_id in ids
    }


def _grader(calls: list[int], grades: dict[str, int] | None = None):
    grades = grades or {"a": 3, "b": 2, "c": 2}

    def call(messages, **kwargs):
        calls.append(1)
        content = messages[-1]["content"]
        ids = re.findall(r"document_id: (\S+)", content)
        return json.dumps(
            {
                "grades": [
                    {
                        "n": number,
                        "grade": grades.get(doc_id, 1),
                        "claims": ["c1"] if grades.get(doc_id, 1) >= 2 else [],
                        "other_relevant": False,
                    }
                    for number, doc_id in enumerate(ids, start=1)
                ]
            }
        )

    return call


class TestEvaluate(unittest.TestCase):
    def _run(self, tmp: Path, mapping, calls):
        qrels: dict = {}
        result = evaluate(
            systems=list(mapping),
            questions=_questions(),
            answers=_answers(),
            qrels=qrels,
            dataset=_dataset(),
            backend=BACKEND,
            k=10,
            runs_dir=tmp / "runs",
            reports_dir=tmp / "reports",
            qrels_path=tmp / "qrels.json",
            retriever=_retriever_for(mapping),
            chunk_fetcher=_chunks,
            call=_grader(calls),
        )
        return qrels, result

    def test_runs_all_systems_grades_union_and_rescores_all(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            calls: list[int] = []
            qrels, result = self._run(
                tmp, {"dense": ["a", "b"], "hybrid": ["b", "c"]}, calls
            )
            self.assertEqual(set(qrels["q001"]["judgments"]), {"a", "b", "c"})
            self.assertEqual(len(calls), 1)
            self.assertEqual(
                set(result["scores"]), {"dense_k10", "hybrid_k10"}
            )
            dense = json.loads((tmp / "runs" / "dense_k10.run.json").read_text())
            hybrid = json.loads((tmp / "runs" / "hybrid_k10.run.json").read_text())
            self.assertEqual(dense["n_unjudged"], 0)
            self.assertEqual(hybrid["n_unjudged"], 0)
            self.assertTrue((tmp / "reports" / "scores.json").exists())
            self.assertEqual(
                dense["metrics"]["qrels_sha256"],
                hybrid["metrics"]["qrels_sha256"],
            )

    def test_order_independent(self) -> None:
        mapping = {"dense": ["a", "b"], "hybrid": ["b", "c"]}
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            qrels_a, result_a = self._run(Path(first), mapping, [])
            qrels_b, result_b = self._run(
                Path(second), dict(reversed(list(mapping.items()))), []
            )
            self.assertEqual(
                qrels_a["q001"]["judgments"], qrels_b["q001"]["judgments"]
            )
            self.assertEqual(
                result_a["scores"]["dense_k10"]["overall"],
                result_b["scores"]["dense_k10"]["overall"],
            )


if __name__ == "__main__":
    unittest.main()
