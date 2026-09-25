from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from refiner.pipeline import transform


def _payload(extra: dict | None = None) -> dict:
    base = {
        "id": "mediawiki:1",
        "entity_type": "item",
        "description": [
            "A test item.<br />For testing.",
            "Second paragraph with [link](Somewhere).",
        ],
        "sections": [
            {"heading": None, "markdown": "Lead with {{MiscItem}}."},
            {"heading": "Availability", "markdown": "## Availability\n\n*Sold by X."},
        ],
        "infobox_extra": {"name": "[[Foo|Foo]]", "price": "100"},
    }
    if extra:
        base.update(extra)
    return base


def test_string_value_is_cleaned():
    out = transform({"name": "Foo<br/>Bar"})
    assert out["name"] == "Foo\nBar"
    assert "<br" not in out["name"]


def test_list_of_strings_joined_with_blank_line():
    out = transform({"description": ["First line.", "Second line."]})
    assert out["description"] == "First line.\n\nSecond line."


def test_list_of_strings_each_cleaned_then_joined():
    out = transform({"description": ["A [shrine](Firelink Shrine).", "Has {{MiscItem}}."]})
    assert "(" not in out["description"]
    assert "{{" not in out["description"]
    assert "A shrine." in out["description"]
    assert "Has" in out["description"]


def test_list_with_empty_items_dropped():
    out = transform({"description": ["", "Real text.", "  "]})
    assert out["description"] == "Real text."


def test_infobox_extra_dict_passes_through_unchanged():
    payload = _payload()
    out = transform(payload)
    assert out["infobox_extra"] == payload["infobox_extra"]


def test_non_string_types_passthrough():
    payload = {"id": 1234, "score": 4.5, "flag": True, "nothing": None}
    out = transform(payload)
    assert out["id"] == 1234
    assert out["score"] == 4.5
    assert out["flag"] is True
    assert out["nothing"] is None


def test_sections_key_replaced_by_exploded_keys():
    payload = _payload()
    out = transform(payload)
    assert "sections" not in out
    assert "lead" in out or "availability" in out


def test_clean_prose_keeps_mediawiki_id_intact():
    from refiner.normalise import clean_prose

    assert clean_prose("mediawiki:6696") == "mediawiki:6696"
    assert clean_prose("mediawiki:16016") == "mediawiki:16016"


def test_clean_prose_strips_two_letter_interwiki_lines():
    from refiner.normalise import clean_prose

    text = "Real prose.\npl:Nowa Gra Plus\nru:Новая игра +"
    out = clean_prose(text)
    assert "pl:" not in out
    assert "ru:" not in out


def test_sections_exploded_into_top_level_keys():
    payload = {
        "description": [],
        "sections": [
            {"heading": None, "markdown": "Lead prose with {{MiscItem}}."},
            {"heading": "Availability", "markdown": "## Availability\n\n*Sold by X."},
        ],
        "other": "x",
    }
    out = transform(payload)
    assert "sections" not in out
    assert "lead" in out
    assert "availability" in out
    assert "{{" not in out["lead"]
    assert "##" not in out["availability"]
    assert out["other"] == "x"


def test_sections_duplicate_headings_get_suffix():
    payload = {
        "sections": [
            {"heading": "Notes", "markdown": "first"},
            {"heading": "Notes", "markdown": "second"},
            {"heading": None, "markdown": "lead"},
        ],
    }
    out = transform(payload)
    assert out["notes"] == "first"
    assert out["notes_2"] == "second"
    assert out["lead"] == "lead"


def test_sections_empty_markdown_skipped():
    payload = {
        "sections": [
            {"heading": "Empty", "markdown": "<!-- table -->"},
            {"heading": "Real", "markdown": "Some real prose."},
        ],
    }
    out = transform(payload)
    assert "empty" not in out
    assert out["real"] == "Some real prose."


def test_parent_with_children_becomes_array():
    payload = {
        "sections": [
            {
                "heading": "Attacks",
                "markdown": "### Attacks\n\nThe attacks are similar to Vanguard Demon's.",
            },
            {
                "heading": "'''Butt Slam'''",
                "markdown": "#### '''Butt Slam'''\n\n<!-- table -->\n\nFlies up, pauses.",
            },
            {
                "heading": "'''Hammer Drive'''",
                "markdown": "#### '''Hammer Drive'''\n\n<!-- table -->\n\nUsed when the player is in front.",
            },
        ],
    }
    out = transform(payload)
    assert "attacks" in out
    attacks = out["attacks"]
    assert isinstance(attacks, list)
    assert len(attacks) == 2
    assert attacks[0]["name"] == "Butt Slam"
    assert attacks[0]["description"] == "Flies up, pauses."
    assert attacks[1]["name"] == "Hammer Drive"
    assert "butt_slam" not in out
    assert "hammer_drive" not in out


def test_child_name_strips_template_and_ref_noise():
    payload = {
        "sections": [
            {
                "heading": "Attacks",
                "markdown": "### Attacks\n\nintro.",
            },
            {
                "heading": "'''Double Swing'''<ref group=\"note\" name=\"damage\">x</ref>",
                "markdown": "#### '''Double Swing'''<ref group=\"note\" name=\"damage\">x</ref>\n\nA combo.",
            },
        ],
    }
    out = transform(payload)
    assert out["attacks"][0]["name"] == "Double Swing"
    assert "<ref" not in out["attacks"][0]["name"]
    assert "'''" not in out["attacks"][0]["name"]


def test_parent_without_children_stays_a_string():
    payload = {
        "sections": [
            {
                "heading": "Lore",
                "markdown": "## Lore\n\nOnce upon a time.",
            },
        ],
    }
    out = transform(payload)
    assert out["lore"] == "Once upon a time."
