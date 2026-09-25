from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from refiner.normalise import clean_prose


def test_clean_prose_strips_br():
    assert "<br" not in clean_prose("First.<br />Second.<br/>Third.")


def test_clean_prose_strips_interwiki_languages():
    text = "Real prose.\npl:Nowa Gra Plus\nru:Новая игра +"
    out = clean_prose(text)
    assert "pl:" not in out
    assert "ru:" not in out
    assert "Real prose." in out


def test_clean_prose_strips_nbps():
    assert "\xa0" not in clean_prose("Hello\xa0world")


def test_clean_prose_flattens_wikilinks():
    out = clean_prose("A [shrine](Firelink Shrine) and [[Orange Ring|Orange Ring]].")
    assert "[" not in out
    assert "shrine" in out
    assert "Orange Ring" in out


def test_clean_prose_strips_table_marker_and_refs():
    out = clean_prose("<!-- table --> some text <ref name=foo/> more.")
    assert "<!--" not in out
    assert "<ref" not in out


def test_clean_prose_strips_templates_and_nested():
    out = clean_prose("Hello {{MiscItem}} and {{Confuse|Foo {{!}}{{!}}Bar}}")
    assert "{{" not in out


def test_clean_prose_drops_leading_heading_line():
    out = clean_prose("## Availability\n\n*Sold by X.")
    assert "##" not in out
    assert "Sold by X" in out


def test_clean_prose_strips_arbitrary_html_tags():
    out = clean_prose(
        "Outside.\n<center>centered text</center>\n<small>tiny</small>\n<wiki>wikitext</wiki>"
    )
    assert "<" not in out


def test_clean_prose_converts_mw_bold_italic():
    out = clean_prose("'''bold''' and ''italic'' text")
    assert "'''" not in out
    assert "''" not in out
    assert "bold" in out
    assert "italic" in out


def test_clean_prose_strips_definition_list_markers():
    out = clean_prose("Before.\n:some def\n: another\nAfter.")
    assert "some def" in out
    assert "another" in out


def test_clean_prose_keeps_prose_unchanged_on_clean_input():
    text = "A simple sentence.\n\nAnother paragraph here."
    out = clean_prose(text)
    assert "A simple sentence." in out
    assert "Another paragraph here." in out
