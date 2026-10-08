from __future__ import annotations

import unittest
from typing import ClassVar
from unittest.mock import AsyncMock, patch

import httpx

from eval.llm_grade import (
    RUBRIC_TEXT,
    SNIPPET_LEN,
    build_user_prompt,
    grade_batch,
    grade_batch_sync,
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


if __name__ == "__main__":
    unittest.main()
