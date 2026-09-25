from __future__ import annotations

import json
import re
from pathlib import Path

from .normalise import clean_prose


_LIST_SEP = "\n\n"
_SECTION_HEADING_RE = re.compile(r"[^a-z0-9]+")
_HEADING_LINE_RE = re.compile(r"^(\s{0,3})(#{2,6})\s+(.+?)\s*#*\s*$")
_BOLD_RE = re.compile(r"^'''(.+?)'''$")


def _section_key(heading: str | None) -> str:
    if heading is None or not heading.strip():
        return "lead"
    s = heading.strip().lower()
    s = _SECTION_HEADING_RE.sub("_", s).strip("_")
    return s or "section"


def _heading_level(md: str) -> int | None:
    if not md:
        return None
    for line in md.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(#{2,6})\s", line)
        return len(m.group(1)) if m else None
    return None


def _strip_first_heading_lines(md: str) -> str:
    """Drop leading markdown heading lines (one or more consecutive)."""
    if not md:
        return ""
    lines = md.split("\n")
    i = 0
    while i < len(lines):
        if re.match(r"^\s{0,3}#{2,6}\s", lines[i]):
            i += 1
            continue
        if lines[i].strip() == "":
            i += 1
            continue
        break
    return "\n".join(lines[i:]).lstrip("\n").rstrip()


def _child_display_name(section: dict) -> str:
    """Best-effort name extraction for a sub-section."""
    raw: str | None = None
    md = section.get("markdown") or ""
    if md:
        for line in md.split("\n"):
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^#{2,6}\s+(.+?)\s*#*\s*$", line)
            if not m:
                break
            raw = m.group(1).strip()
            bold = _BOLD_RE.match(raw)
            if bold:
                raw = bold.group(1).strip()
            break
    if raw is None:
        heading = section.get("heading")
        if isinstance(heading, str) and heading.strip():
            raw = heading.strip()
            bold = _BOLD_RE.match(raw)
            if bold:
                raw = bold.group(1).strip()
    if not raw:
        return "Unknown"
    name = clean_prose(raw).strip().strip("*").strip()
    return name or "Unknown"


def _group_parent_children(value: list) -> tuple[dict[int, list[int]], set[int]]:
    """Identify h2/h3 parents with h4 children. Returns (children_of, parent_indices)."""
    children_of: dict[int, list[int]] = {}
    parent_indices: set[int] = set()
    parent_stack: list[tuple[int, int]] = []

    for i, sec in enumerate(value):
        if not isinstance(sec, dict):
            continue
        md = sec.get("markdown") or ""
        level = _heading_level(md)
        if level is None:
            continue
        if level <= 3:
            while parent_stack and parent_stack[-1][1] >= level:
                parent_stack.pop()
            parent_stack.append((i, level))
        elif level == 4:
            if parent_stack:
                p_idx, _ = parent_stack[-1]
                children_of.setdefault(p_idx, []).append(i)
                parent_indices.add(p_idx)

    return children_of, parent_indices


def transform(cleaned: dict) -> dict:
    """Top-level: clean strings, join list-of-strings, explode ``sections``.

    For ``sections``, h2/h3 sections that contain h4 child sections are
    emitted as a single array of ``{name, description}`` under the parent's
    slugified key. Sections without children become a single cleaned-string
    top-level entry as before.
    """
    out: dict = {}
    for key, value in cleaned.items():
        if isinstance(value, str):
            out[key] = clean_prose(value)
            continue
        if (
            isinstance(value, list)
            and value
            and all(isinstance(x, str) for x in value)
        ):
            parts = [clean_prose(x) for x in value]
            parts = [p for p in parts if p]
            out[key] = _LIST_SEP.join(parts) if parts else ""
            continue
        if key == "sections" and isinstance(value, list):
            children_of, parent_indices = _group_parent_children(value)
            seen: dict[str, int] = {}
            consumed: set[int] = set()

            for i, sec in enumerate(value):
                if i in consumed or not isinstance(sec, dict):
                    continue
                heading = sec.get("heading")
                markdown = sec.get("markdown") or ""

                if i in parent_indices:
                    items = []
                    for ci in children_of[i]:
                        if ci in consumed:
                            continue
                        consumed.add(ci)
                        csec = value[ci]
                        cmd = csec.get("markdown") or ""
                        body = _strip_first_heading_lines(cmd)
                        desc = clean_prose(body) if body else ""
                        if not desc:
                            continue
                        items.append({
                            "name": _child_display_name(csec),
                            "description": desc,
                        })
                    consumed.add(i)
                    if not items:
                        continue
                    base = _section_key(heading if isinstance(heading, str) else None)
                    count = seen.get(base, 0)
                    seen[base] = count + 1
                    new_key = base if count == 0 else f"{base}_{count + 1}"
                    out[new_key] = items
                    continue

                cleaned_md = clean_prose(markdown)
                if not cleaned_md:
                    continue
                consumed.add(i)
                base = _section_key(heading if isinstance(heading, str) else None)
                count = seen.get(base, 0)
                seen[base] = count + 1
                new_key = base if count == 0 else f"{base}_{count + 1}"
                out[new_key] = cleaned_md
            continue
        out[key] = value
    return out


def refine_one(
    *,
    source_path: Path,
    cleaned_root: Path,
    structured_root: Path,
    force: bool = False,
) -> dict:
    rel = source_path.relative_to(cleaned_root).as_posix()
    parts = Path(rel).parts
    if len(parts) < 3:
        raise ValueError(f"unexpected path layout: {rel}")

    with source_path.open("r", encoding="utf-8") as f:
        cleaned = json.load(f)

    if not isinstance(cleaned, dict):
        raise ValueError(f"top-level JSON not an object: {rel}")

    out = transform(cleaned)

    game, etype = parts[0], parts[1]
    output_path = structured_root / game / etype / parts[-1]

    if output_path.exists() and not force:
        return {"status": "skipped_existing", "output_path": str(output_path)}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")

    return {"status": "ok", "output_path": str(output_path)}
