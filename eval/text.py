"""Text normalization and entity/title matching shared by the benchmark.

``match_rank`` is carried over from the retired ``eval/build_seeds.py`` on
purpose. The short-entity rule (entities shorter than five characters match
exactly only) prevents "Dark" from flooding a candidate set with every page
that mentions "Dark Souls", and "Light" from matching "Sunlight Blade".
"""

from __future__ import annotations

import re

SHORT_ENTITY_LEN = 5
_TOKEN_RE = re.compile(r"[a-z0-9']+")


def normalize_title(value: str) -> str:
    """Lowercase, fold punctuation and underscores to spaces."""
    text = value.replace("_", " ").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def tokenize(value: str) -> list[str]:
    return _TOKEN_RE.findall(value.lower())


def match_rank(entity: str, target: str) -> int | None:
    """Match quality of an expected entity against a title.

    0 = exact normalized match, 1 = every entity token appears as a whole
    word in the target, ``None`` = no match. Entities shorter than
    :data:`SHORT_ENTITY_LEN` match exactly only.
    """
    entity_norm = normalize_title(entity)
    target_norm = normalize_title(target)
    if not entity_norm or not target_norm:
        return None
    if entity_norm == target_norm:
        return 0
    if len(entity_norm) < SHORT_ENTITY_LEN:
        return None
    if set(entity_norm.split()) <= set(target_norm.split()):
        return 1
    return None


def normalize_quote(value: str) -> str:
    """Whitespace-normalized text used for verbatim quote verification."""
    return normalize_whitespace(value)


def quote_is_grounded(quote: str, section_text: str) -> bool:
    """True when the quote is a whitespace-normalized substring of the section."""
    quote_norm = normalize_quote(quote)
    return bool(quote_norm) and quote_norm in normalize_quote(section_text)
