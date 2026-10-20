from __future__ import annotations

import unittest

from src.retriever.enhancement import _extract_queries


class TestExtractQueries(unittest.TestCase):
    def test_json_object(self) -> None:
        self.assertEqual(
            _extract_queries(
                '{"queries": ["Knight Artorias", "Great Grey Wolf Sif"]}'
            ),
            ["Knight Artorias", "Great Grey Wolf Sif"],
        )

    def test_json_list(self) -> None:
        self.assertEqual(
            _extract_queries('["parrying", "backstabbing", "guard breaking"]'),
            ["parrying", "backstabbing", "guard breaking"],
        )

    def test_json_single_query_key(self) -> None:
        self.assertEqual(
            _extract_queries('{"query": "Seath betrayal Everlasting Dragons"}'),
            ["Seath betrayal Everlasting Dragons"],
        )

    def test_fenced_json(self) -> None:
        self.assertEqual(
            _extract_queries(
                '```json\n{"queries": ["Frampt", "Kaathe"]}\n```'
            ),
            ["Frampt", "Kaathe"],
        )

    def test_json_embedded_in_prose(self) -> None:
        self.assertEqual(
            _extract_queries(
                'Here are the queries: {"queries": ["Siegward", "Undead Settlement"]}'
            ),
            ["Siegward", "Undead Settlement"],
        )

    def test_numbered_lines(self) -> None:
        self.assertEqual(
            _extract_queries("1. Knight Artorias\n2. Great Grey Wolf Sif"),
            ["Knight Artorias", "Great Grey Wolf Sif"],
        )

    def test_plain_single_line(self) -> None:
        self.assertEqual(
            _extract_queries("Knight Artorias Abysswalker lore history"),
            ["Knight Artorias Abysswalker lore history"],
        )

    def test_dedupes_and_caps(self) -> None:
        self.assertEqual(
            _extract_queries(
                '{"queries": ["Sif", "sif", "Artorias", "Abysswalker"]}',
                max_queries=2,
            ),
            ["Sif", "Artorias"],
        )

    def test_empty_response(self) -> None:
        self.assertEqual(_extract_queries("   "), [])

    def test_truncated_json_is_not_treated_as_query(self) -> None:
        self.assertEqual(
            _extract_queries('{"queries":["Knight Artorias","K'),
            [],
        )

    def test_json_array_without_close_is_rejected(self) -> None:
        self.assertEqual(_extract_queries('["Knight Artorias", "Great Grey Wolf Sif"'), [])

    def test_strips_query_prefix_and_quotes(self) -> None:
        self.assertEqual(
            _extract_queries('Query: "Estus Flask uses"'),
            ["Estus Flask uses"],
        )


if __name__ == "__main__":
    unittest.main()
