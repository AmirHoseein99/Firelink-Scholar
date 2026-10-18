from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from eval.corpus import (
    Page,
    Section,
    fetch_chunks,
    match_pages,
    parse_page,
    section_lookup,
    titles_by_game,
)


def _page(game: str, title: str, path: str, sections: dict[str, str]) -> Page:
    return Page(
        source_path=path,
        title=title,
        game=game,
        entity_type="boss",
        sections=tuple(
            Section(
                page_title=title,
                game=game,
                entity_type="boss",
                source_path=path,
                section_key=key,
                text=text,
            )
            for key, text in sections.items()
        ),
    )


class TestParsePage(unittest.TestCase):
    def test_metadata_keys_are_not_sections_and_lists_join(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = {
                "id": "x",
                "infobox": {"a": "b"},
                "infobox_extra": {},
                "infobox_raw": {},
                "schema_version": "1.0",
                "tables": [],
                "source": {
                    "title": "Test Boss",
                    "game": "Dark Souls",
                    "categories": ["Boss"],
                },
                "entity_type": "boss",
                "lead": "Lead text",
                "description": ["one", "two"],
                "empty": "",
            }
            path = root / "Dark Souls" / "boss" / "test.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(payload), encoding="utf-8")

            page = parse_page(path, root=root)
            self.assertIsNotNone(page)
            keys = [section.section_key for section in page.sections]
            self.assertEqual(keys, ["lead", "description"])
            self.assertEqual(page.sections[1].text, "one\ntwo")
            self.assertEqual(page.source_path, "Dark Souls/boss/test.json")

    def test_titles_by_game_sorted_unique(self) -> None:
        pages = [
            _page("Dark Souls", "B", "b.json", {"lead": "b"}),
            _page("Dark Souls", "A", "a.json", {"lead": "a"}),
            _page("Dark Souls II", "A", "c.json", {"lead": "a"}),
        ]
        self.assertEqual(
            titles_by_game(pages),
            {"Dark Souls": ["A", "B"], "Dark Souls II": ["A"]},
        )


class TestGameFiltering(unittest.TestCase):
    def test_same_title_in_two_games_is_not_collapsed(self) -> None:
        pages = [
            _page("Dark Souls", "Artorias of the Abyss", "ds1.json", {"lead": "1"}),
            _page("Dark Souls III", "Knight Artorias", "ds3.json", {"lead": "3"}),
        ]
        matched = match_pages(["Artorias"], pages, "Dark Souls")
        self.assertEqual(list(matched), [("Dark Souls", "Artorias of the Abyss")])

    def test_all_game_sees_both(self) -> None:
        pages = [
            _page("Dark Souls", "Artorias of the Abyss", "ds1.json", {"lead": "1"}),
            _page("Dark Souls III", "Knight Artorias", "ds3.json", {"lead": "3"}),
        ]
        matched = match_pages(["Artorias"], pages, "All")
        self.assertEqual(len(matched), 2)


class TestSectionLookup(unittest.TestCase):
    def test_lookup_by_path_and_section(self) -> None:
        pages = [_page("Dark Souls", "Lore", "Dark Souls/lore.json", {"lead": "x"})]
        lookup = section_lookup(pages)
        self.assertEqual(lookup[("Dark Souls/lore.json", "lead")].text, "x")


class TestFetchChunks(unittest.TestCase):
    def test_payloads_are_keyed_by_document_id(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.calls = []

            def retrieve(self, **kwargs):
                self.calls.append(kwargs)
                return [
                    SimpleNamespace(
                        payload={
                            "page_content": "body",
                            "metadata": {
                                "document_id": "id1",
                                "title": "T",
                                "game": "Dark Souls",
                                "section": "lead",
                            },
                        }
                    )
                ]

        client = FakeClient()
        chunks = fetch_chunks(["id1", "id2"], client=client)
        self.assertEqual(set(chunks), {"id1"})
        self.assertEqual(chunks["id1"]["page_content"], "body")
        self.assertEqual(client.calls[0]["ids"], ["id1", "id2"])


if __name__ == "__main__":
    unittest.main()
