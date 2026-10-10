from __future__ import annotations

import unittest

from eval.build_seeds import (
    ANCHOR_SNIPPET_LEN,
    SNIPPET_LEN,
    TEMPLATE_JUNK_CAP,
    build_anchors_for_question,
    build_pool,
    extract_lead_links,
    extract_template_links,
    match_rank,
    normalize_title,
)
from src.cleaner.registry import Registry

WIKITEXT = """\
{{Enemy
|hp         = 813
|drops      = [[Big Pilgrim's Key]]<br />[[Demon's Great Hammer]]<br />[[Humanity (Item){{!}}Humanity]]
|location   = [[Undead Asylum#Location|see location]]
|image      = [[File:Asylum Demon.png|270px]]
}}{{BossesNav}}
{{For|2=Demon (disambiguation)}}
The {{Japanese|'''Asylum Demon'''|不死院のデーモン|lead=yes}} is a [[boss]] in ''[[Dark Souls]]''.
<ref>See [[Design Works (Book)|Design Works]] p. 100.</ref>
[[Category:Dark Souls Enemies]]
"""

LEAD_MARKDOWN = """\
{{Enemy|name=Four Kings|location=[[Abyss]]|drops=[[Bequeathed Lord Soul Shard]]}}\
The  are a [boss](boss) in *[[Dark Souls]]*.

They guard [[Cat (Dark Souls)]] and [New Londo Ruins](New Londo Ruins).
[External](https://example.com/x) and [[:Category:Dark Souls Bosses]] stay out.
"""


def _chunk(doc_id: str, title: str, content: str = "body") -> dict:
    return {
        "document_id": doc_id,
        "title": title,
        "section": "lead",
        "game": "Dark Souls",
        "entity_type": "boss",
        "page_content": content,
    }


def _ids(pool: list[dict]) -> list[str]:
    return [c["document_id"] for c in pool]


def _sources(pool: list[dict], doc_id: str) -> list[str]:
    return next(c for c in pool if c["document_id"] == doc_id)["sources"]


class TestNormalize(unittest.TestCase):
    def test_case_underscore_punct_fold(self) -> None:
        self.assertEqual(normalize_title("Knight_Artorias"), "knight artorias")
        self.assertEqual(normalize_title("Manus, Father of the Abyss"),
                         "manus father of the abyss")
        self.assertEqual(normalize_title("  Dark   (Damage  Type) "),
                         "dark damage type")


class TestMatchRank(unittest.TestCase):
    def test_exact(self) -> None:
        self.assertEqual(match_rank("Dark", "Dark"), 0)
        self.assertEqual(match_rank("Manus", "manus"), 0)

    def test_short_entity_exact_only(self) -> None:
        self.assertIsNone(match_rank("Dark", "Dark Souls"))
        self.assertIsNone(match_rank("Dark", "Dark (Damage Type)"))
        self.assertIsNone(match_rank("Sif", "Great Grey Wolf Sif"))

    def test_whole_word_containment(self) -> None:
        self.assertEqual(match_rank("Manus", "Manus, Father of the Abyss"), 1)
        self.assertEqual(match_rank("Artorias", "Knight Artorias"), 1)
        self.assertEqual(match_rank("Artorias", "Artorias the Abysswalker"), 1)
        self.assertEqual(match_rank("Great Grey Wolf", "Great Grey Wolf Sif"),
                         1)
        self.assertEqual(match_rank("Undead", "Undead Asylum"), 1)

    def test_substring_inside_word_does_not_match(self) -> None:
        self.assertIsNone(match_rank("Light", "Sunlight Blade"))
        self.assertIsNone(match_rank("Light", "Lightning (Dark Souls)"))
        self.assertIsNone(match_rank("Cycle", "Cycles"))

    def test_empty(self) -> None:
        self.assertIsNone(match_rank("", "x"))
        self.assertIsNone(match_rank("x", ""))


class TestTemplateLinks(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.targets = extract_template_links(WIKITEXT, Registry.default())

    def test_infobox_params_collected(self) -> None:
        for expected in ("Big Pilgrim's Key", "Demon's Great Hammer",
                         "Humanity (Item)", "Undead Asylum"):
            self.assertIn(expected, self.targets)

    def test_section_anchor_stripped(self) -> None:
        self.assertIn("Undead Asylum", self.targets)
        self.assertNotIn("Undead Asylum#Location", self.targets)

    def test_label_pipe_target_resolution(self) -> None:
        self.assertIn("Humanity (Item)", self.targets)
        self.assertNotIn("Humanity", self.targets)

    def test_file_namespace_excluded(self) -> None:
        self.assertNotIn("File:Asylum Demon.png", self.targets)

    def test_denylist_templates_skipped(self) -> None:
        self.assertNotIn("Demon (disambiguation)", self.targets)


class TestLeadLinks(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.targets = extract_lead_links(LEAD_MARKDOWN)

    def test_raw_wikilinks_and_md_links(self) -> None:
        for expected in ("Abyss", "Bequeathed Lord Soul Shard", "Dark Souls",
                         "Cat (Dark Souls)", "New Londo Ruins"):
            self.assertIn(expected, self.targets)

    def test_md_target_with_spaces(self) -> None:
        self.assertIn("New Londo Ruins", self.targets)

    def test_external_and_category_excluded(self) -> None:
        self.assertNotIn("https://example.com/x", self.targets)
        self.assertNotIn(":Category:Dark Souls Bosses", self.targets)
        self.assertNotIn("Category:Dark Souls Bosses", self.targets)

    def test_leading_colon_target_excluded(self) -> None:
        self.assertFalse(any(t.startswith(":") for t in self.targets))


class TestBuildPool(unittest.TestCase):
    def _setup(self) -> dict:
        chunks = [
            _chunk("h1", "Manus, Father of the Abyss"),
            _chunk("h2", "Manus, Father of the Abyss"),
            _chunk("h3", "Darklurker"),
            _chunk("t1", "Nashandra"),
            _chunk("t2", "Nashandra"),
            _chunk("t3", "Pages That Link To Manus"),
            _chunk("g1", "Abyss"),
            _chunk("d0", "Dense Only"),
            _chunk("b1", "BM25 Only"),
            _chunk("dk1", "Dark (Mechanic)"),
            _chunk("flood1", "Pages Located In Dark Souls"),
        ]
        chunks_by_id = {c["document_id"]: c for c in chunks}
        title_to_ids: dict[str, list[str]] = {}
        for c in chunks:
            title_to_ids.setdefault(c["title"], []).append(c["document_id"])
        titles_by_norm = {
            normalize_title(t): [t]
            for t in {c["title"] for c in chunks}
        }
        link_index = {
            "by_target": {
                "manus": ["Nashandra", "Pages That Link To Manus"],
                "dark": ["Dark (Mechanic)"],
                "dark souls": ["Pages Located In Dark Souls"],
                "sunlight blade": ["Sunlight Blade Flooder"],
            },
            "by_page": {},
        }
        return {
            "chunks_by_id": chunks_by_id,
            "title_to_ids": title_to_ids,
            "titles_by_norm": titles_by_norm,
            "link_index": link_index,
        }

    def _question(self) -> dict:
        return {"id": "q039", "question": "Manus legacy?",
                "expected_entities": ["Manus"]}

    class _FakeClient:
        def query_points(self, **kwargs):
            limit = kwargs["limit"]
            hits = [
                type("H", (), {"score": 0.9, "payload": {"metadata": {
                    "document_id": f"d{i}"}}})()
                for i in range(limit)
            ]
            return type("R", (), {"points": hits})()

    def _bm25(self, chunks):
        from rank_bm25 import BM25Okapi

        from eval.build_seeds import _tokenize

        return BM25Okapi([_tokenize(c["page_content"]) for c in chunks])

    def test_ordering_human_template_gold_dense_bm25(self) -> None:
        s = self._setup()
        chunks = list(s["chunks_by_id"].values())
        human = {"h1": 3, "h2": 2, "g1": 0}
        pool, counts = build_pool(
            self._question(), human, s["chunks_by_id"], s["title_to_ids"],
            s["titles_by_norm"], s["link_index"], self._FakeClient(),
            self._bm25(chunks), chunks, dense_k=50, bm25_k=50, cap=200,
        )
        ids = _ids(pool)
        self.assertEqual(set(ids[:3]), {"h1", "h2", "g1"})
        for doc_id in ("h1", "h2", "g1"):
            self.assertEqual(_sources(pool, doc_id)[0], "human")
        for doc_id in ("t1", "t2", "t3"):
            self.assertIn(doc_id, ids)
            self.assertIn("template", _sources(pool, doc_id))
        self.assertIn("gold", _sources(pool, "h1"))
        self.assertEqual(_sources(pool, "g1"), ["human"])
        self.assertIn("d0", ids)
        self.assertIn("dense", _sources(pool, "d0"))
        self.assertLess(ids.index("t1"), ids.index("d0"))
        self.assertEqual(counts["human"], 3)

    def test_short_entity_does_not_flood_via_link_graph(self) -> None:
        s = self._setup()
        q = {"id": "q039", "question": "Dark?",
             "expected_entities": ["Dark"]}
        chunks = list(s["chunks_by_id"].values())
        pool, _ = build_pool(
            q, {}, s["chunks_by_id"], s["title_to_ids"], s["titles_by_norm"],
            s["link_index"], self._FakeClient(), self._bm25(chunks), chunks,
            dense_k=0, bm25_k=0, cap=200,
        )
        ids = _ids(pool)
        self.assertEqual(ids, ["dk1"])
        self.assertEqual(_sources(pool, "dk1"), ["template"])

    def test_word_substring_does_not_flood(self) -> None:
        s = self._setup()
        q = {"id": "q025", "question": "Light?",
             "expected_entities": ["Light"]}
        s["chunks_by_id"]["flood2"] = _chunk("flood2", "Sunlight Blade Flooder")
        s["title_to_ids"]["Sunlight Blade Flooder"] = ["flood2"]
        s["titles_by_norm"]["sunlight blade flooder"] = [
            "Sunlight Blade Flooder"]
        chunks = list(s["chunks_by_id"].values())
        pool, _ = build_pool(
            q, {}, s["chunks_by_id"], s["title_to_ids"], s["titles_by_norm"],
            s["link_index"], self._FakeClient(), self._bm25(chunks), chunks,
            dense_k=0, bm25_k=0, cap=200,
        )
        self.assertEqual(_ids(pool), [])

    def test_template_tier_capped(self) -> None:
        s = self._setup()
        for i in range(40):
            page = f"Z Page {i:02d}"
            for j in range(3):
                doc_id = f"tp{i:02d}-{j}"
                s["chunks_by_id"][doc_id] = _chunk(doc_id, page)
                s["title_to_ids"].setdefault(page, []).append(doc_id)
            s["titles_by_norm"][normalize_title(page)] = [page]
            s["link_index"]["by_target"][f"z page {i:02d}"] = [page]
        q = {"id": "qX", "question": "Z?",
             "expected_entities": ["Z Page"]}
        chunks = list(s["chunks_by_id"].values())
        pool, counts = build_pool(
            q, {}, s["chunks_by_id"], s["title_to_ids"], s["titles_by_norm"],
            s["link_index"], self._FakeClient(), self._bm25(chunks), chunks,
            dense_k=0, bm25_k=0, cap=500,
        )
        template_ids = [c["document_id"] for c in pool
                        if "template" in c["sources"]]
        self.assertEqual(len(template_ids), TEMPLATE_JUNK_CAP)
        self.assertEqual(counts["template"], TEMPLATE_JUNK_CAP)

    def test_cap_keeps_human_ids(self) -> None:
        s = self._setup()
        human = {"h1": 3, "h2": 2}
        human.update({f"x{i}": 1 for i in range(20)})
        for i in range(20):
            s["chunks_by_id"][f"x{i}"] = _chunk(f"x{i}", f"Filler {i}")
        chunks = list(s["chunks_by_id"].values())
        pool, _ = build_pool(
            self._question(), human, s["chunks_by_id"], s["title_to_ids"],
            s["titles_by_norm"], s["link_index"], self._FakeClient(),
            self._bm25(chunks), chunks, dense_k=50, bm25_k=50, cap=5,
        )
        ids = _ids(pool)
        self.assertEqual(len(pool), 22)
        self.assertIn("h1", ids)
        self.assertIn("h2", ids)
        self.assertNotIn("t1", ids)

    def test_idempotence(self) -> None:
        s = self._setup()
        chunks = list(s["chunks_by_id"].values())
        human = {"h1": 3, "g1": 0}
        first, _ = build_pool(
            self._question(), human, s["chunks_by_id"], s["title_to_ids"],
            s["titles_by_norm"], s["link_index"], self._FakeClient(),
            self._bm25(chunks), chunks, dense_k=50, bm25_k=50, cap=200,
        )
        second, _ = build_pool(
            self._question(), human, s["chunks_by_id"], s["title_to_ids"],
            s["titles_by_norm"], s["link_index"], self._FakeClient(),
            self._bm25(chunks), chunks, dense_k=50, bm25_k=50, cap=200,
        )
        self.assertEqual(first, second)

    def test_snippet_truncated(self) -> None:
        s = self._setup()
        s["chunks_by_id"]["h1"]["page_content"] = "y" * (SNIPPET_LEN * 2)
        chunks = list(s["chunks_by_id"].values())
        pool, _ = build_pool(
            self._question(), {"h1": 3}, s["chunks_by_id"], s["title_to_ids"],
            s["titles_by_norm"], s["link_index"], self._FakeClient(),
            self._bm25(chunks), chunks, dense_k=0, bm25_k=0, cap=200,
        )
        self.assertEqual(len(pool[0]["snippet"]), SNIPPET_LEN)


class TestAnchors(unittest.TestCase):
    def _chunks(self) -> dict[str, dict]:
        return {
            f"d{i:02d}": _chunk(f"d{i:02d}", f"Page {i}", "z" * 900)
            for i in range(10)
        }

    def test_gold_contrast_lowest_doc_id(self) -> None:
        chunks = self._chunks()
        judged = {"d05": 3, "d02": 3, "d07": 0, "d04": 0, "d01": 2}
        anchor = build_anchors_for_question(
            {"id": "q001"}, judged, chunks)
        self.assertEqual(anchor["gold"]["doc_id"], "d02")
        self.assertEqual(anchor["contrast"]["doc_id"], "d04")
        self.assertEqual(len(anchor["gold"]["snippet"]), ANCHOR_SNIPPET_LEN)

    def test_probes_every_fifth_sorted(self) -> None:
        chunks = self._chunks()
        judged = {f"d{i:02d}": 3 for i in range(10)}
        anchor = build_anchors_for_question(
            {"id": "q001"}, judged, chunks)
        self.assertEqual(anchor["probes"], ["d04", "d09"])

    def test_missing_anchor_grade_returns_none(self) -> None:
        chunks = self._chunks()
        judged = {"d01": 2, "d02": 1}
        anchor = build_anchors_for_question(
            {"id": "q001"}, judged, chunks)
        self.assertIsNone(anchor["gold"])
        self.assertIsNone(anchor["contrast"])

    def test_unknown_ids_skipped(self) -> None:
        chunks = self._chunks()
        judged = {"d01": 3, "zzz": 0}
        anchor = build_anchors_for_question(
            {"id": "q001"}, judged, chunks)
        self.assertEqual(anchor["gold"]["doc_id"], "d01")
        self.assertIsNone(anchor["contrast"])

    def test_no_judgable_ids(self) -> None:
        self.assertIsNone(
            build_anchors_for_question({"id": "q001"}, {}, self._chunks()))


if __name__ == "__main__":
    unittest.main()
