from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.cleaner.clean import clean_one
from src.cleaner.models import CleaningReport
from src.cleaner.registry import Registry


def _write_envelope(
    path: Path,
    page_id: int,
    title: str,
    wikitext: str,
    categories: list[str] | None = None,
) -> None:
    envelope = {
        "query": {
            "pages": {
                str(page_id): {
                    "pageid": page_id,
                    "ns": 0,
                    "title": title,
                    "revisions": [
                        {
                            "revid": 1,
                            "timestamp": "2026-01-01T00:00:00Z",
                            "slots": {
                                "main": {
                                    "contentmodel": "wikitext",
                                    "contentformat": "text/x-wiki",
                                    "*": wikitext,
                                }
                            },
                        }
                    ],
                    "categories": [{"ns": 14, "title": c} for c in (categories or [])],
                }
            }
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(envelope, ensure_ascii=False), encoding="utf-8")


class WeaponInfoboxTest(unittest.TestCase):
    def test_dasIII_weapon_routing(self):
        wt = (
            "{{DaSIIIWeapon\n"
            "|name = Crow Talons\n"
            "|image = Crow Talons.png\n"
            "|weapon-type = [[Claws (Dark Souls III)|Claw]]\n"
            "|buy-price = N/A\n"
            "|phys-atk = 91\n"
            "|phys-def = -\n"
            "}}\n"
            "The '''Crow Talons''' is a test.\n"
        )
        with self.subTest():
            import tempfile

            with tempfile.TemporaryDirectory() as td:
                src = Path(td) / "raw" / "Dark Souls III" / "Claws" / "Crow Talons.json"
                _write_envelope(
                    src, 143021, "Crow Talons", wt, ["Category:Dark Souls III: Weapons"]
                )
                report = CleaningReport()
                clean_one(
                    source_path=src,
                    source_root=src.parent.parent,
                    output_root=Path(td) / "out",
                    registry=Registry.default(),
                    report=report,
                )
                out = list((Path(td) / "out").rglob("*.json"))
                self.assertTrue(out)
                payload = json.loads(out[0].read_text(encoding="utf-8"))
                self.assertEqual(payload["entity_type"], "weapon")
                self.assertEqual(payload["infobox"]["phys-atk"], "91")
                self.assertIsNone(payload["infobox"]["buy-price"])
                self.assertEqual(payload["infobox_raw"]["buy-price"], "N/A")
                self.assertEqual(payload["source"]["template"], "DaSIIIWeapon")


class CharacterInfoboxTest(unittest.TestCase):
    def test_character_routing(self):
        wt = (
            "{{Infobox character\n"
            "| theme = DS1\n"
            "| image = X.png\n"
            "| appears_in = [[Dark Souls]]\n"
            "| appears_in = [[Dark Souls II]]\n"
            "| mentioned_in = [[Dark Souls III]]\n"
            "| voiced_by = N/A\n"
            "}}\n"
            "Lore text.\n"
        )
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "raw" / "Dark Souls" / "Characters" / "Smough.json"
            _write_envelope(
                src, 2597, "Smough", wt, ["Category:Dark Souls: Characters"]
            )
            report = CleaningReport()
            clean_one(
                source_path=src,
                source_root=src.parent.parent,
                output_root=Path(td) / "out",
                registry=Registry.default(),
                report=report,
            )
            payload = json.loads(
                next(iter((Path(td) / "out").rglob("*.json"))).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(payload["entity_type"], "character")
            self.assertEqual(payload["infobox"]["theme"], "DS1")
            self.assertEqual(
                payload["infobox"]["appears_in"],
                ["[[Dark Souls]]", "[[Dark Souls II]]"],
            )
            self.assertEqual(payload["infobox"]["mentioned_in"], ["[[Dark Souls III]]"])
            self.assertIsNone(payload["infobox"]["voiced_by"])


class GenericDisambiguationTest(unittest.TestCase):
    def test_disambiguation_no_infobox(self):
        wt = "{{disambig}}\n*[[Attunement (Dark Souls)]]\n*[[Attunement (Dark Souls II)]]\n"
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            src = (
                Path(td) / "raw" / "uncategorized" / "Attunement (disambiguation).json"
            )
            _write_envelope(
                src, 1, "Attunement (disambiguation)", wt, ["Category:Disambiguations"]
            )
            report = CleaningReport()
            clean_one(
                source_path=src,
                source_root=src.parent.parent,
                output_root=Path(td) / "out",
                registry=Registry.default(),
                report=report,
            )
            payload = json.loads(
                next(iter((Path(td) / "out").rglob("*.json"))).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(payload["entity_type"], "generic")
            self.assertEqual(payload["infobox"], {})
            self.assertIn("Attunement (Dark Souls)", payload["sections"][0]["markdown"])


if __name__ == "__main__":
    unittest.main()
