"""Read-only access to ``data/structured`` and to Qdrant chunk payloads.

The structured corpus is the source of truth for answer keys. This module
mirrors the section rules of ``src.retriever/indexing.py`` (metadata keys are
not sections; list sections are one section whose items are newline-joined)
without ever recomputing chunk ids. Chunk ids are always looked up in Qdrant
through the process-wide client singleton.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.paths import STRUCTURED_DIR
from eval.text import match_rank, normalize_title

logger = logging.getLogger(__name__)

# Same metadata keys ignored by src/retriever/indexing.py::sections_to_ignore.
METADATA_KEYS = frozenset(
    {
        "id",
        "infobox",
        "infobox_extra",
        "infobox_raw",
        "schema_version",
        "tables",
        "source",
        "entity_type",
    }
)


@dataclass(frozen=True)
class Section:
    page_title: str
    game: str
    entity_type: str
    source_path: str
    section_key: str
    text: str

    @property
    def location(self) -> tuple[str, str]:
        return self.source_path, self.section_key


@dataclass(frozen=True)
class Page:
    source_path: str
    title: str
    game: str
    entity_type: str
    sections: tuple[Section, ...]


def _section_text(value: Any) -> str:
    if isinstance(value, list):
        parts = [item if isinstance(item, str) else str(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, str):
        return value
    return "" if value is None else str(value)


def parse_page(path: Path, *, root: Path = STRUCTURED_DIR) -> Page | None:
    """Parse one structured page; return None when it has no usable source."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    source = data.get("source") or {}
    title = source.get("title")
    game = source.get("game")
    if not title or not game:
        return None
    try:
        source_path = Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError:
        source_path = Path(path)
    source_path_text = Path(source_path).as_posix()
    entity_type = data.get("entity_type") or ""
    sections: list[Section] = []
    for key, value in data.items():
        if key in METADATA_KEYS:
            continue
        text = _section_text(value)
        if not text:
            continue
        sections.append(
            Section(
                page_title=title,
                game=game,
                entity_type=entity_type,
                source_path=source_path_text,
                section_key=key,
                text=text,
            )
        )
    return Page(
        source_path=source_path_text,
        title=title,
        game=game,
        entity_type=entity_type,
        sections=tuple(sections),
    )


def iter_pages(root: Path = STRUCTURED_DIR) -> Iterator[Page]:
    for path in sorted(Path(root).rglob("*.json")):
        page = parse_page(path, root=root)
        if page is not None:
            yield page


def load_pages(root: Path = STRUCTURED_DIR) -> list[Page]:
    return list(iter_pages(root))


def iter_sections(pages: Iterable[Page] | None = None) -> Iterator[Section]:
    for page in pages if pages is not None else iter_pages():
        yield from page.sections


def titles_by_game(pages: Iterable[Page] | None = None) -> dict[str, list[str]]:
    """Sorted unique page titles for each game."""
    titles: dict[str, set[str]] = {}
    for page in pages if pages is not None else iter_pages():
        titles.setdefault(page.game, set()).add(page.title)
    return {game: sorted(values) for game, values in titles.items()}


def pages_for_game(pages: Iterable[Page], game: str) -> list[Page]:
    if game == "All":
        return list(pages)
    return [page for page in pages if page.game == game]


def match_pages(
    entities: Iterable[str],
    pages: Iterable[Page],
    game: str,
) -> dict[tuple[str, str], int]:
    """Map ``(game, page title) -> best match rank`` for expected entities.

    Pages are filtered by ``game`` unless the question is cross-game
    (``game == "All"``). This is the guard against DS3 pages (Gwyn, Artorias,
    Ornstein, Smough) defining DS1 answers.
    """
    candidates = pages_for_game(pages, game)
    matched: dict[tuple[str, str], int] = {}
    for page in candidates:
        for entity in entities:
            rank = match_rank(entity, page.title)
            if rank is not None:
                key = (page.game, page.title)
                matched[key] = min(matched.get(key, 9), rank)
    return matched


def resolve_entities(
    entities: Iterable[str],
    titles: Iterable[str],
) -> dict[str, list[str]]:
    """Map each entity to the page titles it matches, best rank first."""
    ranked: dict[str, list[tuple[int, str]]] = {}
    title_list = list(titles)
    for entity in entities:
        for title in title_list:
            rank = match_rank(entity, title)
            if rank is not None:
                ranked.setdefault(entity, []).append((rank, title))
    return {
        entity: [title for _, title in sorted(values, key=lambda pair: (pair[0], pair[1]))]
        for entity, values in ranked.items()
    }


def sections_by_path(pages: Iterable[Page]) -> dict[str, Page]:
    return {page.source_path: page for page in pages}


def section_lookup(pages: Iterable[Page]) -> dict[tuple[str, str], Section]:
    lookup: dict[tuple[str, str], Section] = {}
    for page in pages:
        for section in page.sections:
            lookup[(page.source_path, section.section_key)] = section
    return lookup


def title_lookup(pages: Iterable[Page]) -> dict[tuple[str, str], Page]:
    return {(page.game, page.title): page for page in pages}


def entity_candidates(
    entities: Iterable[str],
    pages: list[Page],
    game: str,
) -> list[Page]:
    """Pages whose title matches an expected entity within the question's game."""
    matched = match_pages(entities, pages, game)
    by_title = title_lookup(pages_for_game(pages, game))
    return [
        by_title[key]
        for key in sorted(matched, key=lambda key: (matched[key], key))
        if key in by_title
    ]


# --------------------------------------------------------------------------- #
# Qdrant lookup (ids are never derived, only fetched)
# --------------------------------------------------------------------------- #

def _chunk_from_payload(point: Any) -> dict[str, Any] | None:
    payload = getattr(point, "payload", None) or {}
    metadata = payload.get("metadata") or {}
    document_id = metadata.get("document_id")
    if not document_id:
        return None
    return {
        "document_id": document_id,
        "title": metadata.get("title", ""),
        "game": metadata.get("game", ""),
        "category": metadata.get("category", []),
        "entity_type": metadata.get("entity_type", ""),
        "section": metadata.get("section", ""),
        "page_content": payload.get("page_content", ""),
    }


def fetch_chunks(
    ids: Iterable[str],
    *,
    client: Any | None = None,
    batch_size: int = 256,
) -> dict[str, dict[str, Any]]:
    """Fetch chunk payloads by document id.

    Uses the process-wide Qdrant client singleton by default. Missing ids are
    simply absent from the returned mapping.
    """
    wanted = [doc_id for doc_id in dict.fromkeys(ids) if doc_id]
    if not wanted:
        return {}
    if client is None:
        from src.retriever.vector_store import get_qdrant_client

        client = get_qdrant_client()
    from src.config import settings

    chunks: dict[str, dict[str, Any]] = {}
    for start in range(0, len(wanted), batch_size):
        batch = wanted[start : start + batch_size]
        points = client.retrieve(
            collection_name=settings.COLLECTION_NAME,
            ids=batch,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            chunk = _chunk_from_payload(point)
            if chunk is not None:
                chunks[chunk["document_id"]] = chunk
    logger.debug("Fetched %d/%d chunk(s) from Qdrant", len(chunks), len(wanted))
    return chunks


def scroll_chunks(*, client: Any | None = None, limit: int = 512) -> dict[str, dict[str, Any]]:
    """Return every chunk payload keyed by document id."""
    if client is None:
        from src.retriever.vector_store import get_qdrant_client

        client = get_qdrant_client()
    from src.config import settings

    chunks: dict[str, dict[str, Any]] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=settings.COLLECTION_NAME,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            chunk = _chunk_from_payload(point)
            if chunk is not None:
                chunks[chunk["document_id"]] = chunk
        if offset is None:
            break
    return chunks


def titles_match(entities: Iterable[str], titles: Iterable[str]) -> dict[str, list[str]]:
    """Convenience wrapper used by question validation."""
    return resolve_entities(entities, titles)


def normalized_title_set(titles: Iterable[str]) -> set[str]:
    return {normalize_title(title) for title in titles}
