from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_SAFE_RE = re.compile(r"[^A-Za-z0-9._/]+")


def sanitize_title(title: str) -> str:
    """Return a filesystem-safe path segment for a MediaWiki title.

    Replaces reserved characters with ``_`` while preserving ``.``, ``_``,
    and ``/``. A leading ``File:`` or other namespace prefix is preserved by
    replacing the colon with a path separator so namespaces stay grouped on
    disk. ``.`` and ``..`` segments are dropped to prevent path traversal.
    """

    text = title.strip().replace(":", "/").replace("\\", "/")
    text = _SAFE_RE.sub("_", text)
    parts: list[str] = []
    for raw in text.split("/"):
        cleaned = raw.strip("_")
        if not cleaned or cleaned in {".", ".."}:
            continue
        parts.append(cleaned)
    if not parts:
        return "_unnamed"
    return "/".join(parts)


def _extract_page(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    pages = payload.get("query", {}).get("pages")
    if not isinstance(pages, Mapping) or not pages:
        return None
    for page in pages.values():
        if isinstance(page, Mapping):
            return dict(page)
    return None


async def dump_raw_json(payload: Mapping[str, Any], root: Path) -> Path:
    """Write a MediaWiki page envelope to ``root`` as JSON.

    The output path is ``<root>/<safe_title>/<safe_title>.json`` so every
    page is contained in its own deterministic directory. Existing files
    are overwritten in place; callers wanting skip behaviour should check
    first.
    """

    page = _extract_page(payload)
    title = str(page.get("title", "_unnamed")) if page else "_unnamed"
    safe = sanitize_title(title)
    page_dir = root / safe
    page_dir.mkdir(parents=True, exist_ok=True)
    out_path = page_dir / f"{safe.split('/')[-1]}.json"
    tmp_path = page_dir / f".{out_path.name}.tmp"
    data = json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True)
    tmp_path.write_text(data, encoding="utf-8")
    tmp_path.replace(out_path)
    return out_path
