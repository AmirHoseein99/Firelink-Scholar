from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from eval.grade import (
    JudgeError,
    grade_run_payloads,
    parse_grades,
)
from eval.llm import Backend
from eval.prompts_grade import GRADE_PROMPT_SHA256

BACKEND = Backend(
    name="test", base_url="http://test/v1", api_key="x", model="judge-model"
)


def _chunk(doc_id: str) -> dict:
    return {
        "document_id": doc_id,
        "title": "Title " + doc_id,
        "game": "Dark Souls",
        "section": "lead",
        "page_content": "content " + doc_id,
    }


def _payload(qid: str, ids: list[str]) -> dict:
    return {
        "config": {"system": "dense", "k": 10},
        "run": {qid: [{"document_id": doc_id, "title": "T", "score": 1.0} for doc_id in ids]},
    }


def _dataset(prompt_hash: str = GRADE_PROMPT_SHA256) -> dict:
    return {
        "judge": {
            "backend": "freellm",
            "base_url": "http://test/v1",
            "model": "judge-model",
            "temperature": 0,
            "prompt_sha256": prompt_hash,
        },
        "claims": {"max_single_game": 8, "max_synthesis": 12},
        "metrics": {"ks": [5, 10], "min_grade": 2},
    }


def _question(qid: str = "q001") -> dict:
    return {
        "id": qid,
        "question": "Who was Artorias?",
        "game": "Dark Souls",
        "category": "lore",
        "expected_entities": ["Artorias of the Abyss"],
    }


def _answer(qid: str = "q001", status: str = "verified") -> dict:
    return {
        qid: {
            "status": status,
            "claims": [
                {"id": "c1", "text": "A core claim.", "importance": "core"},
                {"id": "c2", "text": "A supporting claim.", "importance": "supporting"},
            ],
        }
    }


def _grader(grade_map: dict[str, int], calls: list[int]):
    def call(messages, **kwargs):
        calls.append(1)
        content = messages[-1]["content"]
        ids = re.findall(r"document_id: (\S+)", content)
        grades = [
            {
                "n": number,
                "grade": grade_map.get(doc_id, 1),
                "claims": ["c1"] if grade_map.get(doc_id, 1) >= 2 else [],
                "other_relevant": False,
            }
            for number, doc_id in enumerate(ids, start=1)
        ]
        return json.dumps({"grades": grades})

    return call


class TestParseGrades(unittest.TestCase):
    def test_tolerant_parsing(self) -> None:
        batch = [{"document_id": "a"}, {"document_id": "b"}, {"document_id": "c"}]
        raw = json.dumps(
            {
                "grades": [
                    {"n": 1, "grade": 3, "claims": ["c1", "unknown"]},
                    {"n": 1, "grade": 0, "claims": []},
                    {"n": 2, "grade": 9, "claims": []},
                    {"n": 3, "grade": 1, "claims": ["c1"]},
                    {"n": 99, "grade": 2, "claims": []},
                ]
            }
        )
        parsed, unmapped = parse_grades(raw, batch, {"c1"})
        self.assertEqual(parsed[1]["grade"], 3)
        self.assertEqual(parsed[1]["claims"], ["c1"])
        self.assertEqual(parsed[3]["claims"], [])
        self.assertEqual(unmapped, [2])

    def test_other_relevant_only_for_relevant_grades(self) -> None:
        batch = [{"document_id": "a"}]
        parsed, _ = parse_grades(
            json.dumps({"grades": [{"n": 1, "grade": 1, "other_relevant": True}]}),
            batch,
            set(),
        )
        self.assertFalse(parsed[1]["other_relevant"])


class TestGradeRunPayloads(unittest.TestCase):
    def _run(self, payloads, chunks, qrels, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            qrels_path = Path(tmp) / "qrels.json"
            return grade_run_payloads(
                payloads,
                questions=[_question()],
                answers=_answer(),
                qrels=qrels,
                dataset=_dataset(),
                backend=BACKEND,
                qrels_path=qrels_path,
                chunk_fetcher=lambda ids: {doc_id: chunks[doc_id] for doc_id in ids if doc_id in chunks},
                **kwargs,
            )

    def test_union_is_graded_once(self) -> None:
        chunks = {doc_id: _chunk(doc_id) for doc_id in ("a", "b", "c")}
        qrels: dict = {}
        calls: list[int] = []
        summary = self._run(
            [_payload("q001", ["a", "b"]), _payload("q001", ["b", "c"])],
            chunks,
            qrels,
            call=_grader({"a": 3, "b": 2, "c": 1}, calls),
        )
        self.assertEqual(summary["graded"], 3)
        self.assertEqual(len(calls), 1)
        self.assertEqual(set(qrels["q001"]["judgments"]), {"a", "b", "c"})
        self.assertEqual(qrels["q001"]["judgments"]["a"], 3)
        self.assertEqual(qrels["q001"]["supports_claims"]["a"], ["c1"])
        self.assertEqual(qrels["q001"]["n_judgments"], 3)

    def test_existing_judgments_are_skipped(self) -> None:
        chunks = {doc_id: _chunk(doc_id) for doc_id in ("a", "b", "c")}
        qrels = {"q001": {"judgments": {"a": 3}, "supports_claims": {}}}
        calls: list[int] = []
        summary = self._run(
            [_payload("q001", ["a", "b", "c"])],
            chunks,
            qrels,
            call=_grader({}, calls),
        )
        self.assertEqual(summary["skipped"], 1)
        self.assertEqual(summary["graded"], 2)
        self.assertEqual(qrels["q001"]["judgments"]["a"], 3)
        self.assertEqual(set(qrels["q001"]["judgments"]), {"a", "b", "c"})

    def test_missing_qdrant_ids_are_reported_not_added(self) -> None:
        chunks = {"a": _chunk("a")}
        qrels: dict = {}
        calls: list[int] = []
        summary = self._run(
            [_payload("q001", ["a", "b"])],
            chunks,
            qrels,
            call=_grader({}, calls),
        )
        self.assertEqual(summary["graded"], 1)
        self.assertEqual(
            [item["document_id"] for item in summary["missing"]], ["b"]
        )
        self.assertEqual(set(qrels["q001"]["judgments"]), {"a"})

    def test_unconfigured_judge_refused(self) -> None:
        dataset = _dataset()
        dataset["judge"].update({"configured": False, "model": "", "backend": ""})
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(JudgeError):
            grade_run_payloads(
                [_payload("q001", ["a"])],
                questions=[_question()],
                answers=_answer(),
                qrels={},
                dataset=dataset,
                backend=BACKEND,
                qrels_path=Path(tmp) / "qrels.json",
                chunk_fetcher=lambda ids: {"a": _chunk("a")},
                call=_grader({}, []),
            )

    def test_judge_model_mismatch_refused(self) -> None:
        backend = Backend("test", "http://test/v1", "x", "other-model")
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(JudgeError):
            grade_run_payloads(
                    [_payload("q001", ["a"])],
                    questions=[_question()],
                    answers=_answer(),
                    qrels={},
                    dataset=_dataset(),
                    backend=backend,
                    qrels_path=Path(tmp) / "qrels.json",
                    chunk_fetcher=lambda ids: {"a": _chunk("a")},
                    call=_grader({}, []),
                )

    def test_prompt_hash_mismatch_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(JudgeError):
            grade_run_payloads(
                    [_payload("q001", ["a"])],
                    questions=[_question()],
                    answers=_answer(),
                    qrels={},
                    dataset=_dataset("bad-hash"),
                    backend=BACKEND,
                    qrels_path=Path(tmp) / "qrels.json",
                    chunk_fetcher=lambda ids: {"a": _chunk("a")},
                    call=_grader({}, []),
                )

    def test_unverified_answer_blocks_grading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(JudgeError):
            grade_run_payloads(
                    [_payload("q001", ["a"])],
                    questions=[_question()],
                    answers=_answer(status="needs_review"),
                    qrels={},
                    dataset=_dataset(),
                    backend=BACKEND,
                    qrels_path=Path(tmp) / "qrels.json",
                    chunk_fetcher=lambda ids: {"a": _chunk("a")},
                    call=_grader({}, []),
                )


if __name__ == "__main__":
    unittest.main()
