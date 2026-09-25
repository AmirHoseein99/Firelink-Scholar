from __future__ import annotations

import json
import re
import unicodedata

_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_HTML_TAG_RE = re.compile(
    r"<\s*[a-zA-Z][a-zA-Z0-9:_\-]*\b[^<>]*?/\s*>"
    r"|<\s*[a-zA-Z][a-zA-Z0-9:_\-]*\b[^<>]*?>.*?</\s*[a-zA-Z][a-zA-Z0-9:_\-]*\s*>"
    r"|<\s*[a-zA-Z][a-zA-Z0-9:_\-]*(?:\s+[^<>]*)?\s*/?>",
    re.DOTALL,
)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_REF_RE = re.compile(
    r"<ref(?:erences)?\b[^>]*?(?:/>|>.*?</ref(?:erences)?>)",
    re.DOTALL | re.IGNORECASE,
)
_TABBER_RE = re.compile(r"<tabber>", re.IGNORECASE)
_TABBER_END_RE = re.compile(r"</tabber>", re.IGNORECASE)
_TAB_RE = re.compile(r"\|\|")
_BOLD_MW_RE = re.compile(r"'''(.+?)'''", re.DOTALL)
_ITALIC_MW_RE = re.compile(r"''(.+?)''", re.DOTALL)
_NOWIKI_RE = re.compile(r"<\s*?/?\s*?nowiki\s*?/?\s*>", re.IGNORECASE)
_DEFINITION_RE = re.compile(r"^[ \t]*:[ \t]*", re.MULTILINE)
_INTERWIKI_LINE_RE = re.compile(r"^[a-z]{2,3}:[^\n]*$", re.MULTILINE | re.IGNORECASE)
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^()]*(?:\([^()]*\)[^()]*)*)\)")
_WIKILINK_BRACKET_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
_NBSP_RE = re.compile("\xa0")
_ZERO_W_RE = re.compile("\u200b")
_BLANK_LINE_RE = re.compile(r"\n{3,}")
_HEADING_LINE_RE = re.compile(r"^\s{0,3}#{1,6}\s+[^\n]*$\n?", re.MULTILINE)
_BULLET_RE = re.compile(r"^(?P<indent>[ \t]*)\*[ \t]*(?=[^ \t*])", re.MULTILINE)
_TRAIL_WS_RE = re.compile(r"[ \t]+(?=\n)")
_REPEATED_SPACE_RE = re.compile(r"[ \t]{2,}")
_TEMPLATE_INNER_RE = re.compile(r"\{\{[^{}]*\}\}")


def _strip_balanced_templates(text: str) -> str:
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if i + 1 < n and text[i] == "{" and text[i + 1] == "{":
            depth = 1
            j = i + 2
            while j < n and depth > 0:
                if j + 1 < n and text[j] == "{" and text[j + 1] == "{":
                    depth += 1
                    j += 2
                    continue
                if j + 1 < n and text[j] == "}" and text[j + 1] == "}":
                    depth -= 1
                    j += 2
                    continue
                j += 1
            if depth == 0:
                out.append(" ")
                i = j
                continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _decode_entities(text: str) -> str:
    return (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
    )


def _flat_markdown_link(match: re.Match) -> str:
    label = match.group(1).strip()
    target = match.group(2).strip()
    if not target or target.startswith("#"):
        return label
    target_l = target.lower()
    if target_l.startswith(("category:", "file:", "image:", "media:")):
        return ""
    return label


def _flat_wikilink(match: re.Match) -> str:
    target = match.group(1).strip()
    label = match.group(2)
    out_label = label.strip() if label else target.replace("_", " ")
    target_l = target.lower()
    if target_l.startswith(("category:", "file:", "image:", "media:")):
        return ""
    if ":" in target and not target.startswith(":"):
        return out_label
    return out_label


def clean_prose(text: str | None) -> str:
    if not text:
        return ""
    s = unicodedata.normalize("NFC", text)

    s = _NBSP_RE.sub(" ", s)
    s = _ZERO_W_RE.sub("", s)
    s = _decode_entities(s)

    s = _BOLD_MW_RE.sub(r"**\1**", s)
    s = _ITALIC_MW_RE.sub(r"*\1*", s)
    s = _NOWIKI_RE.sub("", s)
    s = _BR_RE.sub("\n", s)
    s = _COMMENT_RE.sub("", s)
    s = _REF_RE.sub("", s)
    s = _HTML_TAG_RE.sub("", s)
    s = _TABBER_RE.sub("", s)
    s = _TABBER_END_RE.sub("\n", s)
    s = _TAB_RE.sub("; ", s)

    s = _strip_balanced_templates(s)
    while True:
        prev = s
        s = _TEMPLATE_INNER_RE.sub("", s)
        if s == prev:
            break

    s = _MARKDOWN_LINK_RE.sub(_flat_markdown_link, s)
    s = _WIKILINK_BRACKET_RE.sub(_flat_wikilink, s)

    s = _INTERWIKI_LINE_RE.sub("", s)

    s = _DEFINITION_RE.sub("", s)
    s = _HEADING_LINE_RE.sub("", s)
    s = _BULLET_RE.sub(lambda m: m.group("indent"), s)

    s = _REPEATED_SPACE_RE.sub(" ", s)
    s = _TRAIL_WS_RE.sub("", s)
    s = _BLANK_LINE_RE.sub("\n\n", s)

    return s.strip("\n ")
