from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import AsyncMock, patch

import httpx

from eval import llm_grade
from eval.llm_grade import (
    RUBRIC_TEXT,
    SNIPPET_LEN,
    build_user_prompt,
    grade_batch,
    grade_batch_sync,
    grading_targets,
    merge_question_entry,
    parse_grades,
)

BATCH = [
    {"document_id": f"doc-{i}", "title": f"Title {i}", "section": f"S{i}",
     "entity_type": "character", "snippet": f"snippet {i} text"}
    for i in range(1, 4)
]
BATCH_IDS = [c["document_id"] for c in BATCH]


class TestParseGrades(unittest.TestCase):
    def test_valid_strict_json(self) -> None:
        raw = ('{"grades": [{"n": 1, "grade": 3}, {"n": 2, "grade": 0}, '
               '{"n": 3, "grade": 1}]}')
        grades, discarded = parse_grades(raw, BATCH_IDS)
        self.assertEqual(grades, {"doc-1": 3, "doc-2": 0, "doc-3": 1})
        self.assertEqual(discarded, 0)

    def test_markdown_fenced_json(self) -> None:
        raw = ('```json\n{"grades": [{"n": 1, "grade": 2}, {"n": 2, "grade": 2},'
               ' {"n": 3, "grade": 2}]}\n```')
        grades, discarded = parse_grades(raw, BATCH_IDS)
        self.assertEqual(grades, {"doc-1": 2, "doc-2": 2, "doc-3": 2})
        self.assertEqual(discarded, 0)

    def test_json_with_surrounding_text(self) -> None:
        raw = ('Here are the grades:\n{"grades": [{"n": 1, "grade": 1}, '
               '{"n": 2, "grade": 0}, {"n": 3, "grade": 3}]}\nDone.')
        grades, _ = parse_grades(raw, BATCH_IDS)
        self.assertEqual(grades, {"doc-1": 1, "doc-2": 0, "doc-3": 3})

    def test_bare_list_variant(self) -> None:
        raw = '[{"n": 1, "grade": 3}, {"n": 2, "grade": 3}, {"n": 3, "grade": 0}]'
        grades, discarded = parse_grades(raw, BATCH_IDS)
        self.assertEqual(grades, {"doc-1": 3, "doc-2": 3, "doc-3": 0})
        self.assertEqual(discarded, 0)

    def test_mapping_variant(self) -> None:
        raw = '{"grades": {"1": 2, "2": 1, "3": 3}}'
        grades, _ = parse_grades(raw, BATCH_IDS)
        self.assertEqual(grades, {"doc-1": 2, "doc-2": 1, "doc-3": 3})

    def test_alternate_key_names(self) -> None:
        raw = ('{"results": [{"index": "1", "relevance": 2}, '
               '{"index": 2, "score": 1}, {"index": 3, "rating": 0}]}')
        grades, _ = parse_grades(raw, BATCH_IDS)
        self.assertEqual(grades, {"doc-1": 2, "doc-2": 1, "doc-3": 0})

    def test_out_of_batch_ids_discarded(self) -> None:
        raw = ('{"grades": [{"n": 1, "grade": 2}, {"n": 4, "grade": 3}, '
               '{"n": 0, "grade": 1}]}')
        grades, discarded = parse_grades(raw, BATCH_IDS)
        self.assertEqual(grades, {"doc-1": 2})
        self.assertEqual(discarded, 2)

    def test_out_of_range_grades_discarded(self) -> None:
        raw = ('{"grades": [{"n": 1, "grade": 4}, {"n": 2, "grade": -1}, '
               '{"n": 3, "grade": 3}]}')
        grades, discarded = parse_grades(raw, BATCH_IDS)
        self.assertEqual(grades, {"doc-3": 3})
        self.assertEqual(discarded, 2)

    def test_unmappable_entries_discarded(self) -> None:
        raw = '{"grades": [{"n": 1, "grade": "high"}, {"n": 2, "grade": 2}]}'
        grades, discarded = parse_grades(raw, BATCH_IDS)
        self.assertEqual(grades, {"doc-2": 2})
        self.assertEqual(discarded, 1)

    def test_duplicate_n_first_wins(self) -> None:
        raw = ('{"grades": [{"n": 1, "grade": 3}, {"n": 1, "grade": 0}, '
               '{"n": 2, "grade": 1}, {"n": 3, "grade": 1}]}')
        grades, discarded = parse_grades(raw, BATCH_IDS)
        self.assertEqual(grades["doc-1"], 3)
        self.assertEqual(discarded, 1)

    def test_invalid_json_returns_empty(self) -> None:
        grades, discarded = parse_grades("no json here at all", BATCH_IDS)
        self.assertEqual(grades, {})
        self.assertEqual(discarded, 0)


class TestPromptBuilder(unittest.TestCase):
    def test_rubric_embedded(self) -> None:
        q = {"question": "Who was Gwyn?", "expected_entities": ["Gwyn"]}
        prompt = build_user_prompt(q, BATCH)
        self.assertIn(RUBRIC_TEXT, prompt)
        self.assertIn("3 — the chunk directly and authoritatively", prompt)
        self.assertIn("Negative example", prompt)

    def test_question_entities_and_candidates_embedded(self) -> None:
        q = {"question": "Who was Gwyn?",
             "expected_entities": ["Gwyn", "Lord of Sunlight"]}
        prompt = build_user_prompt(q, BATCH)
        self.assertIn("Question: Who was Gwyn?", prompt)
        self.assertIn("Gwyn, Lord of Sunlight", prompt)
        for i, cand in enumerate(BATCH, start=1):
            self.assertIn(f"[{i}] title='Title {i}'", prompt)
            self.assertIn(f"section='S{i}'", prompt)
            self.assertIn(cand["snippet"], prompt)
        self.assertIn('"grades": [{"n": 1, "grade": <0..3>}', prompt)
        self.assertIn("covering all 3 candidates", prompt)

    def test_snippet_truncated_to_limit(self) -> None:
        q = {"question": "Q?", "expected_entities": []}
        long_cand = [{"document_id": "d1", "title": "T", "section": "S",
                      "entity_type": "E", "snippet": "x" * (SNIPPET_LEN * 2)}]
        prompt = build_user_prompt(q, long_cand)
        self.assertIn("x" * SNIPPET_LEN, prompt)
        self.assertNotIn("x" * (SNIPPET_LEN + 1), prompt)

    def test_empty_expected_entities(self) -> None:
        prompt = build_user_prompt({"question": "Q?", "expected_entities": []},
                                   BATCH)
        self.assertIn("(none listed)", prompt)


class TestGradeBatchAsync(unittest.IsolatedAsyncioTestCase):
    Q: ClassVar[dict] = {
        "id": "q001",
        "question": "Who was Gwyn?",
        "expected_entities": [],
    }
    URL: ClassVar[str] = "http://test/v1/chat/completions"

    @staticmethod
    def _ok(content: str) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}}]}
        )

    def _client(self, responses: list[httpx.Response]) -> httpx.AsyncClient:
        queue = list(responses)

        def handler(request: httpx.Request) -> httpx.Response:
            if not queue:
                return httpx.Response(500, json={"error": "queue empty"})
            return queue.pop(0)

        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def test_happy_path_full_coverage(self) -> None:
        content = ('{"grades": [{"n": 1, "grade": 3}, {"n": 2, "grade": 0}, '
                   '{"n": 3, "grade": 2}]}')
        async with self._client([self._ok(content)]) as client:
            grades, err = await grade_batch(
                client, self.URL, {}, self.Q, BATCH
            )
        self.assertIsNone(err)
        self.assertEqual(grades, {"doc-1": 3, "doc-2": 0, "doc-3": 2})

    async def test_retries_on_503_then_incomplete(self) -> None:
        # 503 -> retry (sleep patched out), then valid JSON covering only 1/3:
        # batch is retried once, still incomplete -> error, partial grades kept
        content = '{"grades": [{"n": 1, "grade": 3}]}'
        async with self._client([httpx.Response(503), self._ok(content),
                                 self._ok(content)]) as client:
            with patch("asyncio.sleep", new=AsyncMock()):
                grades, err = await grade_batch(
                    client, self.URL, {}, self.Q, BATCH
                )
        self.assertEqual(grades, {"doc-1": 3})
        self.assertIn("incomplete JSON grading (1/3 candidates)", err)

    async def test_non_retryable_401_short_circuits(self) -> None:
        async with self._client([httpx.Response(401, text="unauthorized")]) as client:
            grades, err = await grade_batch(
                client, self.URL, {}, self.Q, BATCH
            )
        self.assertEqual(grades, {})
        self.assertIn("HTTP 401", err)


class TestGradeBatchSync(unittest.TestCase):
    Q: ClassVar[dict] = {
        "id": "q001",
        "question": "Who was Gwyn?",
        "expected_entities": [],
    }
    URL: ClassVar[str] = "http://test/v1/chat/completions"

    @staticmethod
    def _ok(content: str) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}}]}
        )

    def _client(self, responses: list[httpx.Response]) -> httpx.Client:
        queue = list(responses)

        def handler(request: httpx.Request) -> httpx.Response:
            if not queue:
                return httpx.Response(500, json={"error": "queue empty"})
            return queue.pop(0)

        return httpx.Client(transport=httpx.MockTransport(handler))

    def test_happy_path_full_coverage(self) -> None:
        content = ('{"grades": [{"n": 1, "grade": 1}, {"n": 2, "grade": 3}, '
                   '{"n": 3, "grade": 0}]}')
        with self._client([self._ok(content)]) as client:
            grades, err = grade_batch_sync(
                client, self.URL, {}, self.Q, BATCH
            )
        self.assertIsNone(err)
        self.assertEqual(grades, {"doc-1": 1, "doc-2": 3, "doc-3": 0})

    def test_retries_on_503_then_incomplete(self) -> None:
        content = '{"grades": [{"n": 1, "grade": 3}]}'
        with (self._client([httpx.Response(503), self._ok(content),
                            self._ok(content)]) as client,
              patch("time.sleep")):
            grades, err = grade_batch_sync(
                client, self.URL, {}, self.Q, BATCH
            )
        self.assertEqual(grades, {"doc-1": 3})
        self.assertIn("incomplete JSON grading (1/3 candidates)", err)

    def test_non_retryable_401_short_circuits(self) -> None:
        with self._client([httpx.Response(401, text="unauthorized")]) as client:
            grades, err = grade_batch_sync(
                client, self.URL, {}, self.Q, BATCH
            )
        self.assertEqual(grades, {})
        self.assertIn("HTTP 401", err)


class TestAnchoredPrompt(unittest.TestCase):
    ANCHORS: ClassVar[dict] = {
        "gold": {"title": "Gold Page", "section": "lead",
                 "snippet": "GOLDTEXT"},
        "contrast": {"title": "Contrast Page", "section": "lore",
                     "snippet": "CONTRASTTEXT"},
    }

    def test_rubric_mention_is_zero_line(self) -> None:
        self.assertIn("Merely mentioning or name-dropping", RUBRIC_TEXT)
        self.assertIn("grade 0, not 1", RUBRIC_TEXT)

    def test_anchors_embedded(self) -> None:
        prompt = build_user_prompt(
            {"question": "Q?", "expected_entities": []}, BATCH, self.ANCHORS)
        self.assertIn("[ANCHOR GOLD — grade 3] title='Gold Page'", prompt)
        self.assertIn("[ANCHOR CONTRAST — grade 0] title='Contrast Page'",
                      prompt)
        self.assertIn("GOLDTEXT", prompt)
        self.assertIn("CONTRASTTEXT", prompt)

    def test_no_anchors_omits_section(self) -> None:
        prompt = build_user_prompt(
            {"question": "Q?", "expected_entities": []}, BATCH)
        self.assertNotIn("[ANCHOR", prompt)
        self.assertNotIn("[ANCHOR", build_user_prompt(
            {"question": "Q?", "expected_entities": []}, BATCH,
            {"gold": None, "contrast": None}))

    def test_anchor_snippet_truncated(self) -> None:
        anchors = {"gold": {"title": "G", "section": "s",
                            "snippet": "x" * 5000}, "contrast": None}
        prompt = build_user_prompt(
            {"question": "Q?", "expected_entities": []}, BATCH, anchors)
        self.assertIn("x" * 700, prompt)
        self.assertNotIn("x" * 701, prompt)


class TestMergeQuestionEntry(unittest.TestCase):
    Q: ClassVar[dict] = {"id": "q001", "question": "Q?", "game": "Dark Souls",
                         "category": "entity"}

    def test_human_wins_and_llm_fills_unjudged(self) -> None:
        old = {"judgments": {"a": 1}, "probe_llm": {},
               "sources": {"human": 0, "llm": 1, "run": 0}}
        entry = merge_question_entry(old, self.Q, human={"a": 2, "b": 3},
                                     new_llm={"a": 0, "c": 1}, new_probes={})
        self.assertEqual(entry["judgments"], {"a": 2, "b": 3, "c": 1})
        self.assertEqual(entry["sources"],
                         {"human": 2, "llm": 1, "run": 0})
        self.assertEqual(entry["grade_counts"], {"1": 1, "2": 1, "3": 1})
        self.assertEqual(entry["n_judgments"], 3)

    def test_probes_recorded_not_merged(self) -> None:
        human = {"a": 3}
        entry = merge_question_entry(None, self.Q, human,
                                     new_llm={"a": 1}, new_probes={"a": 1})
        self.assertEqual(entry["judgments"], {"a": 3})
        self.assertEqual(entry["probe_llm"], {"a": 1})
        self.assertEqual(entry["sources"]["llm"], 0)

    def test_carried_llm_grades_persist(self) -> None:
        old = {"judgments": {"x": 2}, "sources": {"run": 4}}
        entry = merge_question_entry(old, self.Q, human={"a": 3},
                                     new_llm={"b": 1}, new_probes={})
        self.assertEqual(entry["judgments"], {"a": 3, "x": 2, "b": 1})
        self.assertEqual(entry["sources"]["run"], 4)

    def test_idempotent_rerun_is_stable(self) -> None:
        first = merge_question_entry(None, self.Q, {"a": 3}, {"b": 1}, {})
        second = merge_question_entry(first, self.Q, {"a": 3}, {}, {})
        second.pop("updated_at")
        first.pop("updated_at")
        self.assertEqual(first, second)


class TestGradingTargets(unittest.TestCase):
    def test_unjudged_probes_and_anchors(self) -> None:
        pool = [{"document_id": d} for d in ("a", "b", "c", "gold", "ctr")]
        targets = grading_targets(
            pool,
            judged={"a": 3, "b": 0, "gold": 3, "ctr": 0},
            probe_llm={},
            probe_ids={"b"},
            anchor_ids={"gold", "ctr"},
        )
        self.assertEqual([t["document_id"] for t in targets], ["b", "c"])

    def test_probe_with_llm_grade_skipped(self) -> None:
        pool = [{"document_id": d} for d in ("a", "b")]
        targets = grading_targets(
            pool, judged={"a": 3, "b": 2}, probe_llm={"b": 2},
            probe_ids={"b"}, anchor_ids=set())
        self.assertEqual(targets, [])

    def test_fresh_pool_excludes_anchors(self) -> None:
        pool = [{"document_id": d} for d in ("gold", "ctr", "x")]
        targets = grading_targets(pool, judged={}, probe_llm={},
                                  probe_ids=set(), anchor_ids={"gold", "ctr"})
        self.assertEqual([t["document_id"] for t in targets], ["x"])


class TestIdLevelResume(unittest.TestCase):
    """Full sync pipeline against a mock endpoint, run twice."""

    POOL: ClassVar[list[dict]] = [
        {"document_id": f"doc-{i}", "title": f"T{i}", "section": "s",
         "entity_type": "e", "snippet": "text"} for i in range(1, 4)
    ]

    def _setup_dir(self, tmp: str) -> dict[str, Path]:
        base = Path(tmp)
        cand = base / "candidates"
        cand.mkdir()
        (cand / "q900.json").write_text(json.dumps(self.POOL))
        (base / "seeds.json").write_text(json.dumps(
            {"q900": {"judgments": {"doc-1": 3}}}))
        (base / "anchors.json").write_text(json.dumps({}))
        return {"base": base, "cand": cand, "seeds": base / "seeds.json",
                "anchors": base / "anchors.json",
                "judgments": base / "judgments.json"}

    @staticmethod
    def _args(paths: dict[str, Path]) -> argparse.Namespace:
        return argparse.Namespace(
            questions=_questions_file(paths),
            candidates=paths["cand"], judgments=paths["judgments"],
            anchors=paths["anchors"], human_seeds=paths["seeds"],
            start=None, limit=None, batch_size=2, concurrency=1,
        )

    def test_second_run_grades_only_missing_ids(self) -> None:
        calls: list[list[str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            content = payload["messages"][1]["content"]
            batch_lines = [line for line in content.splitlines()
                           if line.startswith("[")]
            calls.append(len(batch_lines))
            grades = [{"n": i + 1, "grade": 2}
                      for i in range(len(batch_lines))]
            return httpx.Response(200, json={"choices": [{"message": {
                "content": json.dumps({"grades": grades})}}]})

        with tempfile.TemporaryDirectory() as tmp:
            paths = self._setup_dir(tmp)
            args = self._args(paths)
            settings_stub = _settings_stub()
            client_factory = _mock_client_factory(handler)
            with (patch.object(llm_grade, "settings", settings_stub),
                  patch.object(llm_grade.httpx, "Client", client_factory)):
                llm_grade._run_grading_sync(args)
            entry = json.loads(paths["judgments"].read_text())["q900"]
            self.assertEqual(entry["judgments"],
                             {"doc-1": 3, "doc-2": 2, "doc-3": 2})
            self.assertEqual(entry["sources"],
                             {"human": 1, "llm": 2, "run": 0})
            # first run targeted only the 2 unjudged non-human ids
            self.assertEqual(calls, [2])

            n_calls_first = len(calls)
            # drop one LLM-graded id: resume must re-grade exactly that one
            entry["judgments"].pop("doc-3")
            paths["judgments"].write_text(json.dumps(
                {"q900": entry}))
            with (patch.object(llm_grade, "settings", settings_stub),
                  patch.object(llm_grade.httpx, "Client", client_factory)):
                llm_grade._run_grading_sync(args)
            self.assertEqual(len(calls) - n_calls_first, 1)
            entry = json.loads(paths["judgments"].read_text())["q900"]
            self.assertEqual(entry["judgments"],
                             {"doc-1": 3, "doc-2": 2, "doc-3": 2})

    def test_already_complete_question_skipped(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("no HTTP expected")

        with tempfile.TemporaryDirectory() as tmp:
            paths = self._setup_dir(tmp)
            entry = merge_question_entry(
                None, {"id": "q900", "question": "Q?", "game": "g",
                       "category": "c"},
                {"doc-1": 3, "doc-2": 2, "doc-3": 2}, {}, {})
            paths["judgments"].write_text(json.dumps({"q900": entry}))
            args = self._args(paths)
            with (patch.object(llm_grade, "settings", _settings_stub()),
                  patch.object(llm_grade.httpx, "Client",
                               _mock_client_factory(handler))):
                llm_grade._run_grading_sync(args)
            loaded = json.loads(paths["judgments"].read_text())
            self.assertEqual(loaded["q900"]["judgments"],
                             entry["judgments"])


def _questions_file(paths: dict[str, Path]) -> Path:
    q = [{"id": "q900", "question": "Q?", "game": "Dark Souls",
          "category": "entity", "expected_entities": []}]
    p = paths["base"] / "questions.json"
    p.write_text(json.dumps(q))
    return p


def _settings_stub():
    from types import SimpleNamespace

    return SimpleNamespace(neuralwatt_api_key="key",
                           neuralwatt_base_url="http://test",
                           neuralwatt_model="test-model")


def _mock_client_factory(handler):
    real_client = httpx.Client

    def factory(*a, **kw):
        kw.pop("timeout", None)
        return real_client(transport=httpx.MockTransport(handler), **kw)
    return factory


if __name__ == "__main__":
    unittest.main()
