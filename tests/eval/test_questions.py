from __future__ import annotations

import unittest

from eval.questions import validate_questions

TITLES = {
    "Dark Souls": [
        "Artorias of the Abyss",
        "First Flame",
        "Great Grey Wolf Sif",
        "Gwyn, Lord of Cinder",
        "Lore",
    ],
    "Dark Souls II": ["Nashandra", "Vendrick"],
    "Dark Souls III": ["Knight Artorias", "Lords of Cinder", "Sullyvahn"],
}


def _question(**overrides):
    question = {
        "id": "q001",
        "question": "Who was Artorias?",
        "game": "Dark Souls",
        "category": "lore",
        "expected_entities": ["Artorias of the Abyss", "Great Grey Wolf Sif"],
    }
    question.update(overrides)
    return question


class TestValidateQuestions(unittest.TestCase):
    def _report(self, questions):
        return validate_questions(questions, titles=TITLES)

    def test_valid_question(self) -> None:
        report = self._report([_question()])
        self.assertTrue(report["ok"])
        self.assertEqual(report["must_fix"], 0)

    def test_duplicate_id_and_text(self) -> None:
        report = self._report([_question(), _question(id="q001")])
        flags = report["questions"]["q001"]["flags"]
        codes = {flag["code"] for flag in flags}
        self.assertIn("duplicate_id", codes)
        self.assertIn("duplicate_text", codes)

    def test_short_entity_does_not_match_long_title(self) -> None:
        report = self._report([_question(expected_entities=["Sif"])])
        self.assertIn(
            "Sif", report["questions"]["q001"]["unresolved_entities"]
        )

    def test_cross_game_entity_is_unresolved(self) -> None:
        report = self._report(
            [_question(game="Dark Souls", expected_entities=["Knight Artorias"])]
        )
        self.assertEqual(
            report["questions"]["q001"]["unresolved_entities"],
            ["Knight Artorias"],
        )

    def test_generic_entity_blocklist(self) -> None:
        report = self._report([_question(expected_entities=["Dark"])])
        codes = {flag["code"] for flag in report["questions"]["q001"]["flags"]}
        self.assertIn("generic_entities", codes)

    def test_invalid_category_and_game(self) -> None:
        report = self._report(
            [_question(category="entity", game="Elden Ring")]
        )
        codes = {flag["code"] for flag in report["questions"]["q001"]["flags"]}
        self.assertIn("invalid_category", codes)
        self.assertIn("invalid_game", codes)

    def test_too_many_entities(self) -> None:
        report = self._report(
            [_question(expected_entities=["Lore"] * 8)]
        )
        codes = {flag["code"] for flag in report["questions"]["q001"]["flags"]}
        self.assertIn("too_many_entities", codes)

    def test_compound_question_is_warning_not_must_fix(self) -> None:
        report = self._report(
            [
                _question(
                    question=(
                        "Who and what and when and where and why and how?"
                    )
                )
            ]
        )
        entry = report["questions"]["q001"]
        self.assertFalse(entry["must_fix"])
        self.assertIn(
            "compound_question", {flag["code"] for flag in entry["flags"]}
        )

    def test_all_game_accepts_entity_in_any_game(self) -> None:
        report = self._report(
            [_question(game="All", expected_entities=["Knight Artorias"])]
        )
        self.assertTrue(report["ok"])

    def test_alias_record_resolves_to_declared_title(self) -> None:
        report = self._report(
            [
                _question(
                    expected_entities=[
                        {"name": "Sif", "resolves_to": "Great Grey Wolf Sif"}
                    ]
                )
            ]
        )
        self.assertTrue(report["ok"])


if __name__ == "__main__":
    unittest.main()
