from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .models import PageOutcome, SourceMetadata, nfc


class LoaderError(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class LoadedPage:
    metadata: SourceMetadata
    wikitext: str
    warnings: list[str]
    categories: list[str]


def _select_revision(revisions: list[dict]) -> tuple[dict, list[str]]:
    if not revisions:
        raise LoaderError("no_revisions")
    if len(revisions) == 1:
        return revisions[0], []
    warnings = ["multiple_revisions"]
    chosen = max(
        revisions,
        key=lambda r: (
            r.get("timestamp", ""),
            int(r.get("revid", 0) or 0),
        ),
    )
    return chosen, warnings


_INTERWIKI_PREFIX = re.compile(r"^[a-z][a-z0-9_-]*:$", re.IGNORECASE)


def _coerce_int(d: dict, key: str) -> int:
    v = d.get(key, 0)
    if isinstance(v, int):
        return v
    if isinstance(v, str) and v.isdigit():
        return int(v)
    return 0


def load_raw(path: Path, source_root: Path) -> LoadedPage:
    """Read a MediaWiki JSON envelope from `path` and return parsed content.

    `source_root` is the directory under which `path` lives; used to compute
    the relative `path` field of the cleaned metadata.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise LoaderError(f"read_error: {e}") from e

    try:
        envelope = json.loads(text)
    except json.JSONDecodeError as e:
        raise LoaderError(f"json_decode: {e}") from e

    if not isinstance(envelope, dict) or "query" not in envelope:
        raise LoaderError("missing_query")
    query = envelope.get("query") or {}
    pages = query.get("pages") or {}
    if not isinstance(pages, dict) or len(pages) != 1:
        raise LoaderError(
            f"expected_one_page_got_{len(pages) if isinstance(pages, dict) else 'invalid'}"
        )

    (_, page_data) = next(iter(pages.items()))
    if not isinstance(page_data, dict):
        raise LoaderError("invalid_page_dict")

    page_id = _coerce_int(page_data, "pageid")
    title = page_data.get("title")
    if not isinstance(title, str) or not title.strip():
        raise LoaderError("missing_title")

    revisions = page_data.get("revisions") or []
    if not isinstance(revisions, list):
        raise LoaderError("invalid_revisions")
    chosen, warnings = _select_revision(revisions)

    slots = chosen.get("slots") or {}
    main = slots.get("main") or {}
    wikitext = main.get("*") or main.get("content")
    if not isinstance(wikitext, str):
        raise LoaderError("missing_main_slot")
    wikitext = nfc(wikitext)

    categories_raw = page_data.get("categories") or []
    cats: list[str] = []
    for c in categories_raw:
        if isinstance(c, dict) and isinstance(c.get("title"), str):
            cats.append(nfc(c["title"]))
    cats = sorted(set(cats))

    rev_id = _coerce_int(chosen, "revid")
    timestamp = str(chosen.get("timestamp") or "")

    try:
        rel = path.relative_to(source_root).as_posix()
    except ValueError:
        rel = path.as_posix()

    parts = rel.split("/", 1)
    game = parts[0] if parts else "uncategorized"

    metadata = SourceMetadata(
        path=rel,
        game=game,
        page_id=page_id,
        title=nfc(title),
        revision_id=rev_id,
        timestamp=nfc(timestamp),
        categories=cats,
        template=None,
        revisions_observed=len(revisions),
    )

    return LoadedPage(
        metadata=metadata,
        wikitext=wikitext,
        warnings=warnings,
        categories=cats,
    )


def make_outcome(
    *,
    source_path: str,
    status: str,
    output_path: str | None = None,
    entity_type: str | None = None,
    template: str | None = None,
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
) -> PageOutcome:
    return PageOutcome(
        source_path=source_path,
        status=status,
        output_path=output_path,
        entity_type=entity_type,
        template=template,
        warnings=list(warnings or []),
        errors=list(errors or []),
    )
