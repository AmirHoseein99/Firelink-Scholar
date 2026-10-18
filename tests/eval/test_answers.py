from __future__ import annotations

import json
import unittest

from eval.answers import (
    CandidateSelector,
    build_answer_for_question,
    parse_claims,
)
from eval.corpus import Page, Section
from eval.llm import Backend

BACKEND = Backend(
    name="test", base_url="http://test/v1", api_key="x", model="test-model"
)


def _page(
    game: str,
    title: str,
    path: str,
    sections: dict[str, str],
    entity_type: str = "boss",
) -> Page:
    return Page(
        source_path=path,
        title=title,
        game=game,
        entity_type=entity_type,
        sections=tuple(
            Section(
                page_title=title,
                game=game,
                entity_type=entity_type,
                source_path=path,
                section_key=key,
                text=text,
            )
            for key, text in sections.items()
        ),
    )


SECTION_TEXT = (
    "Artorias was one of Gwyn's Four Knights. "
    "He walked the Abyss and was corrupted by it. "
    "Sif guarded his grave in Darkroot Garden."
)

CANDIDATES = [
    {
        "source_path": "Dark Souls/boss/artorias.json",
        "page_title": "Artorias of the Abyss",
        "game": "Dark Souls",
        "section_key": "lead",
        "text": SECTION_TEXT,
    }
]


class TestParseClaims(unittest.TestCase):
    def test_accepts_grounded_claims_and_assigns_ids(self) -> None:
        payload = {
            "claims": [
                {
                    "text": "Artorias was one of Gwyn's Four Knights.",
                    "quote": "Artorias was one of Gwyn's Four Knights.",
                    "path": "Dark Souls/boss/artorias.json",
                    "section": "lead",
                    "importance": "core",
                },
                {
                    "text": "Sif guarded his grave.",
                    "quote": "Sif   guarded his grave in Darkroot Garden.",
                    "path": "Dark Souls/boss/artorias.json",
                    "section": "lead",
                    "importance": "supporting",
                },
            ]
        }
        claims, dropped = parse_claims(payload, CANDIDATES, max_claims=8)
        self.assertEqual(dropped, [])
        self.assertEqual([claim["id"] for claim in claims], ["c1", "c2"])
        self.assertEqual(claims[0]["source"]["title"], "Artorias of the Abyss")

    def test_rejects_fabricated_quote_and_unknown_citation(self) -> None:
        payload = {
            "claims": [
                {
                    "text": "Fabricated.",
                    "quote": "Artorias was a dragon.",
                    "path": "Dark Souls/boss/artorias.json",
                    "section": "lead",
                    "importance": "core",
                },
                {
                    "text": "Unknown path.",
                    "quote": SECTION_TEXT,
                    "path": "missing.json",
                    "section": "lead",
                    "importance": "core",
                },
            ]
        }
        claims, dropped = parse_claims(payload, CANDIDATES, max_claims=8)
        self.assertEqual(claims, [])
        self.assertEqual(len(dropped), 2)

    def test_respects_max_claims(self) -> None:
        entries = [
            {
                "text": f"Claim {index}",
                "quote": SECTION_TEXT,
                "path": "Dark Souls/boss/artorias.json",
                "section": "lead",
                "importance": "core",
            }
            for index in range(5)
        ]
        claims, _ = parse_claims({"claims": entries}, CANDIDATES, max_claims=3)
        self.assertEqual(len(claims), 3)


class TestCandidateSelector(unittest.TestCase):
    def test_filters_by_game(self) -> None:
        pages = [
            _page(
                "Dark Souls",
                "Artorias of the Abyss",
                "ds1.json",
                {"lead": "Dark Souls Artorias Abyss"},
            ),
            _page(
                "Dark Souls III",
                "Knight Artorias",
                "ds3.json",
                {"lead": "Dark Souls III Artorias Abyss"},
            ),
        ]
        selector = CandidateSelector(pages)
        question = {
            "question": "Who was Artorias?",
            "game": "Dark Souls",
            "expected_entities": ["Artorias of the Abyss"],
        }
        candidates = selector.select(question)
        self.assertTrue(candidates)
        self.assertTrue(all(c.game == "Dark Souls" for c in candidates))
        self.assertNotIn("ds3.json", {c.source_path for c in candidates})

    def test_short_entity_does_not_flood_candidates(self) -> None:
        pages = [
            _page("Dark Souls", "Dark Souls", "game.json", {"lead": "Dark"}),
            _page("Dark Souls", "Darkroot Garden", "darkroot.json", {"lead": "Garden"}),
        ]
        selector = CandidateSelector(pages)
        question = {
            "question": "garden",
            "game": "Dark Souls",
            "expected_entities": ["Dark"],
        }
        candidates = selector.select(question)
        self.assertNotIn("game.json", {c.source_path for c in candidates})


class TestBuildAnswer(unittest.TestCase):
    def _question(self) -> dict:
        return {
            "id": "q001",
            "question": "Who was Artorias?",
            "game": "Dark Souls",
            "expected_entities": ["Artorias of the Abyss"],
        }

    def test_verified_answer_from_three_claims(self) -> None:
        pages = [
            _page(
                "Dark Souls",
                "Artorias of the Abyss",
                "ds1.json",
                {"lead": SECTION_TEXT},
            )
        ]
        payload = {
            "claims": [
                {
                    "text": "Artorias was one of Gwyn's Four Knights.",
                    "quote": "Artorias was one of Gwyn's Four Knights.",
                    "path": "ds1.json",
                    "section": "lead",
                    "importance": "core",
                },
                {
                    "text": "He walked the Abyss and was corrupted by it.",
                    "quote": "He walked the Abyss and was corrupted by it.",
                    "path": "ds1.json",
                    "section": "lead",
                    "importance": "core",
                },
                {
                    "text": "Sif guarded his grave in Darkroot Garden.",
                    "quote": "Sif guarded his grave in Darkroot Garden.",
                    "path": "ds1.json",
                    "section": "lead",
                    "importance": "supporting",
                },
            ]
        }
        answer = build_answer_for_question(
            self._question(),
            CandidateSelector(pages),
            backend=BACKEND,
            max_claims=8,
            retries=0,
            call=lambda messages, **kwargs: json.dumps(payload),
        )
        self.assertEqual(answer["status"], "verified")
        self.assertEqual(len(answer["claims"]), 3)
        self.assertIn("Artorias", answer["answer"])
        self.assertEqual(answer["builder"]["model"], "test-model")

    def test_no_core_claim_needs_review(self) -> None:
        pages = [
            _page(
                "Dark Souls",
                "Artorias of the Abyss",
                "ds1.json",
                {"lead": SECTION_TEXT},
            )
        ]
        payload = {
            "claims": [
                {
                    "text": "Sif guarded his grave.",
                    "quote": "Sif guarded his grave in Darkroot Garden.",
                    "path": "ds1.json",
                    "section": "lead",
                    "importance": "supporting",
                }
            ]
        }
        answer = build_answer_for_question(
            self._question(),
            CandidateSelector(pages),
            backend=BACKEND,
            max_claims=8,
            retries=0,
            call=lambda messages, **kwargs: json.dumps(payload),
        )
        self.assertEqual(answer["status"], "needs_review")
        self.assertIn("core", answer["review_reason"])


if __name__ == "__main__":
    unittest.main()
