from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.cleaner.clean import clean_one
from src.cleaner.models import CleaningReport
from src.cleaner.registry import Registry


def _write_envelope(path: Path, page_id: int, title: str, wikitext: str, revisions: list[dict] | None = None, categories: list[str] | None = None) -> None:
    if revisions is None:
        revisions = [
            {
                "revid": 1,
                "timestamp": "2026-01-01T00:00:00Z",
                "slots": {"main": {"contentmodel": "wikitext", "contentformat": "text/x-wiki", "*": wikitext}},
            }
        ]
    envelope = {
        "query": {
            "pages": {
                str(page_id): {
                    "pageid": page_id,
                    "ns": 0,
                    "title": title,
                    "revisions": revisions,
                    "categories": [{"ns": 14, "title": c} for c in (categories or [])],
                }
            }
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(envelope, ensure_ascii=False), encoding="utf-8")


class RevisionSelectionTest(unittest.TestCase):
    def test_pick_greatest_timestamp(self):
        wt = "body"
        revisions = [
            {
                "revid": 1,
                "timestamp": "2026-01-01T00:00:00Z",
                "slots": {"main": {"contentmodel": "wikitext", "contentformat": "text/x-wiki", "*": "older"}},
            },
            {
                "revid": 2,
                "timestamp": "2026-02-01T00:00:00Z",
                "slots": {"main": {"contentmodel": "wikitext", "contentformat": "text/x-wiki", "*": wt}},
            },
        ]
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "raw" / "uncategorized" / "x.json"
            _write_envelope(src, 42, "X", wt, revisions=revisions, categories=[])
            report = CleaningReport()
            outcome = clean_one(source_path=src, source_root=src.parent.parent, output_root=Path(td) / "out", registry=Registry.default(), report=report)
            self.assertIn("multiple_revisions", outcome.warnings)
            self.assertEqual(report.totals.pages, 1)

    def test_tiebreak_by_revid(self):
        revisions = [
            {
                "revid": 1,
                "timestamp": "2026-01-01T00:00:00Z",
                "slots": {"main": {"contentmodel": "wikitext", "contentformat": "text/x-wiki", "*": "x"}},
            },
            {
                "revid": 99,
                "timestamp": "2026-01-01T00:00:00Z",
                "slots": {"main": {"contentmodel": "wikitext", "contentformat": "text/x-wiki", "*": "y"}},
            },
        ]
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "raw" / "uncategorized" / "x.json"
            _write_envelope(src, 7, "Y", "y", revisions=revisions, categories=[])
            report = CleaningReport()
            clean_one(source_path=src, source_root=src.parent.parent, output_root=Path(td) / "out", registry=Registry.default(), report=report)
            self.assertEqual(report.totals.pages, 1)


class DeterminismTest(unittest.TestCase):
    def test_byte_identical_clean(self):
        wt = "{{DaSIIIWeapon|name = X |price = 100 |notes = N/A}}\n\n==H1==\nbody 1 {{Note|x}}\n\n==H2==\nbody 2\n[[File:f.png]]\n"
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "raw" / "Dark Souls III" / "Weapons" / "X.json"
            _write_envelope(src, 1001, "X", wt, categories=["Category:Dark Souls III: Weapons"])
            for label, out_root in (("first", Path(td) / "out1"), ("second", Path(td) / "out2")):
                rep = CleaningReport()
                clean_one(source_path=src, source_root=src.parent.parent, output_root=out_root, registry=Registry.default(), report=rep)
            files1 = sorted((Path(td) / "out1").rglob("*.json"))
            files2 = sorted((Path(td) / "out2").rglob("*.json"))
            self.assertEqual([p.relative_to(Path(td) / "out1") for p in files1], [p.relative_to(Path(td) / "out2") for p in files2])
            for a, b in zip(files1, files2):
                self.assertEqual(a.read_bytes(), b.read_bytes(), f"files differ: {a}")


class NoOverwriteTest(unittest.TestCase):
    def test_existing_outputs_skipped(self):
        wt = "body"
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "raw" / "uncategorized" / "x.json"
            _write_envelope(src, 5, "X", wt, categories=[])
            report = CleaningReport()
            clean_one(source_path=src, source_root=src.parent.parent, output_root=Path(td) / "out", registry=Registry.default(), report=report)
            r2 = CleaningReport()
            outcome = clean_one(source_path=src, source_root=src.parent.parent, output_root=Path(td) / "out", registry=Registry.default(), report=r2)
            self.assertEqual(outcome.status, "skipped_existing")
            self.assertEqual(r2.totals.skipped_existing, 1)


class MalformedInputTest(unittest.TestCase):
    def test_bad_json(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "raw" / "uncategorized" / "x.json"
            src.parent.mkdir(parents=True, exist_ok=True)
            src.write_text("not json", encoding="utf-8")
            report = CleaningReport()
            outcome = clean_one(source_path=src, source_root=src.parent.parent, output_root=Path(td) / "out", registry=Registry.default(), report=report)
            self.assertEqual(outcome.status, "malformed")
            self.assertEqual(report.totals.malformed, 1)


class TablesRegularTest(unittest.TestCase):
    def test_regular_table_csv_json(self):
        import mwparserfromhell as mw

        from src.cleaner import tables as T

        wt = (
            "Intro.\n"
            "{| class=\"wikitable\"\n"
            "! A !! B\n"
            "| 1 || 2\n"
            "| 3 || 4\n"
            "|}\n"
        )
        parsed = mw.parse(wt)
        table_tag = T.find_tables_in_wikicode(parsed)[0]
        pt = T.parse_table(table_tag)
        ok, reason = T._table_is_regular(pt)
        self.assertTrue(ok, reason)
        self.assertEqual(pt.columns, ["A", "B"])


if __name__ == "__main__":
    unittest.main()
