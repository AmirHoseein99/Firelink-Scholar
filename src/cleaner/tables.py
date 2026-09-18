from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import mwparserfromhell as mw
from mwparserfromhell.nodes import Tag

from .models import IrregularTable, TableReference

_ROW_SPLIT_RE = re.compile(r"(?m)^\s*\|[-+]")


@dataclass
class ParsedCell:
    text: str
    raw: str


@dataclass
class ParsedRow:
    cells: list[ParsedCell]
    is_header: bool


@dataclass
class ParsedTable:
    raw: str
    caption: str | None
    rows: list[ParsedRow]
    columns: list[str]

    @property
    def header_row(self) -> ParsedRow | None:
        for r in self.rows:
            if r.is_header:
                return r
        return None


def _split_table_lines(raw: str) -> list[str]:
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    return text.split("\n")


def find_tables_in_wikicode(parsed: mw.Wikicode) -> list[Tag]:
    return [
        n
        for n in parsed.nodes
        if isinstance(n, Tag) and str(n).lstrip().startswith("{|")
    ]


def assign_tables_to_sections(
    parsed: mw.Wikicode,
) -> list[tuple[Tag, str | None]]:
    """Yield (table_tag, section_heading) tuples in source order.

    Section headings are tracked by walking top-level nodes; the last
    `==Heading==` seen before a table is its section.
    """
    current: str | None = None
    out: list[tuple[Tag, str | None]] = []
    for n in parsed.nodes:
        from mwparserfromhell.nodes import Heading as _H

        if isinstance(n, _H):
            current = n.title.strip()
            continue
        if isinstance(n, Tag) and str(n).lstrip().startswith("{|"):
            out.append((n, current))
    return out


_CELL_TERMINATOR = re.compile(r"\|\|")


def _split_cells(line: str) -> list[str]:
    raw = line.strip()
    raw = raw.removeprefix("|")
    raw = raw.removeprefix("+")
    parts: list[str] = []
    buf: list[str] = []
    i = 0
    depth = 0
    while i < len(raw):
        if raw[i] == "[" and i + 1 < len(raw) and raw[i + 1] == "[":
            depth += 1
            buf.append("[[")
            i += 2
            continue
        if raw[i] == "]" and i + 1 < len(raw) and raw[i + 1] == "]":
            depth = max(0, depth - 1)
            buf.append("]]")
            i += 2
            continue
        if raw[i : i + 2] == "||" and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
            i += 2
            continue
        buf.append(raw[i])
        i += 1
    parts.append("".join(buf).strip())
    return parts


def _cell_text(cell_str: str) -> str:
    text = cell_str.strip()
    text = re.sub(r"\{\{[^}]*\}\}", "", text)
    text = re.sub(
        r"\[\[[^]|]*(?:\|([^]]*))?\]\]", lambda m: m.group(1) or m.group(0)[2:-2], text
    )
    text = re.sub(r"'''([^']+)'''", r"\1", text)
    text = re.sub(r"''([^']+)''", r"\1", text)
    text = re.sub(r"<br\s*/?>", " ", text)
    text = (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&nbsp;", " ")
    )
    return text.strip()


def parse_table(tag: Tag) -> ParsedTable:
    raw = str(tag)
    lines = _split_table_lines(raw)
    rows: list[ParsedRow] = []
    caption: str | None = None
    pending_header: list[ParsedCell] = []
    pending_body: list[ParsedCell] = []
    in_table = False
    column_count: int | None = None

    def flush_header() -> None:
        nonlocal pending_header
        nonlocal column_count
        if pending_header:
            rows.append(ParsedRow(cells=pending_header, is_header=True))
            if column_count is None:
                column_count = len(pending_header)
            pending_header = []

    def flush_body() -> None:
        nonlocal pending_body
        if pending_body:
            rows.append(ParsedRow(cells=pending_body, is_header=False))
            pending_body = []

    def cell_count_for(n: int) -> int:
        return column_count if column_count else n

    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith("{|"):
            in_table = True
            if "|+" in s:
                cap_match = re.search(r"\|\+([^|]*)", s)
                if cap_match:
                    caption = _cell_text(cap_match.group(1).strip())
            continue
        if not in_table:
            continue
        if s.startswith("|}"):
            flush_header()
            flush_body()
            break
        if s.startswith("|-"):
            flush_header()
            flush_body()
            continue
        if s.startswith("|+"):
            flush_header()
            flush_body()
            caption = _cell_text(s[2:])
            continue
        if s.startswith("!"):
            flush_body()
            pending_header.extend(_parse_header_line(s))
            continue
        if s.startswith("|"):
            flush_header()
            cells_raw = _parse_body_line(s)
            if column_count is None:
                column_count = len(cells_raw) if cells_raw else 0
            if not cells_raw or column_count <= 0:
                continue
            if len(cells_raw) == column_count:
                rows.append(ParsedRow(cells=cells_raw, is_header=False))
            elif len(cells_raw) == 1 and column_count > 1:
                pending_body.append(cells_raw[0])
                while len(pending_body) >= column_count:
                    rows.append(
                        ParsedRow(cells=pending_body[:column_count], is_header=False)
                    )
                    pending_body = pending_body[column_count:]
            else:
                pending_body.extend(cells_raw)
                while len(pending_body) >= column_count:
                    rows.append(
                        ParsedRow(cells=pending_body[:column_count], is_header=False)
                    )
                    pending_body = pending_body[column_count:]
            continue

    flush_header()
    flush_body()

    columns: list[str] = []
    if rows and rows[0].is_header:
        columns = [c.text for c in rows[0].cells]
    return ParsedTable(raw=raw, caption=caption, rows=rows, columns=columns)


def _parse_body_line(s: str) -> list[ParsedCell]:
    inner = s[1:]
    if "||" in inner:
        cells_str = _split_cells(s)
    else:
        cells_str = [inner.strip()]
    result: list[ParsedCell] = []
    for c in cells_str:
        if not c.strip():
            continue
        c_strip = c
        if "|" in c_strip:
            attrs, _, body = c_strip.partition("|")
            attr_lower = attrs.strip().lower().rstrip()
            if any(
                attr_lower.startswith(tok)
                for tok in ("colspan=", "rowspan=", "style=", "class=")
            ) and not attrs.strip().startswith("[["):
                c_strip = body.strip()
        result.append(ParsedCell(text=_cell_text(c_strip), raw=c_strip))
    return result


def _parse_header_line(s: str) -> list[ParsedCell]:
    """Parse one (or more) header-style cells starting with ``!``.

    Cells are separated by ``!!``. Each cell may have HTML attributes such as
    ``scope="col"`` or ``colspan=2``; we strip those to keep the visible label.
    """
    inner = s.lstrip("!")
    parts: list[str] = []
    buf: list[str] = []
    i = 0
    depth = 0
    while i < len(inner):
        if inner[i : i + 2] == "!!" and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
            i += 2
            continue
        if inner[i] == "[" and i + 1 < len(inner) and inner[i + 1] == "[":
            depth += 1
            buf.append("[[")
            i += 2
            continue
        if inner[i] == "]" and i + 1 < len(inner) and inner[i + 1] == "]":
            depth = max(0, depth - 1)
            buf.append("]]")
            i += 2
            continue
        buf.append(inner[i])
        i += 1
    parts.append("".join(buf).strip())
    result: list[ParsedCell] = []
    for p in parts:
        p_strip = p
        if "|" in p_strip:
            attrs, _, body = p_strip.partition("|")
            attr_lower = attrs.strip().lower()
            if any(
                attr_lower.startswith(tok)
                for tok in ("scope=", "colspan=", "rowspan=", "style=", "class=")
            ):
                p_strip = body.strip()
            else:
                p_strip = p_strip.lstrip("|").strip()
        else:
            p_strip = p_strip.strip()
        if p_strip:
            result.append(ParsedCell(text=_cell_text(p_strip), raw=p_strip))
    return result


def _table_is_regular(table: ParsedTable) -> tuple[bool, str | None]:
    if not table.rows:
        return False, "empty"
    header = table.header_row
    if header is None:
        return False, "no_header"
    n_cols = len(header.cells)
    if n_cols == 0:
        return False, "no_header"
    for idx, row in enumerate(table.rows[1:], start=1):
        if row.is_header:
            return False, "header_in_body"
        if len(row.cells) != n_cols:
            # mwparserfromhell flattens, allow len mismatch as irregular
            return False, f"row_count_mismatch:{len(row.cells)}!={n_cols}"
    if "rowspan" in table.raw or "colspan" in table.raw:
        return False, "rowspan" if "rowspan" in table.raw else "colspan"
    if re.search(r"\n\s*\{\|", table.raw):
        return False, "nested"
    for row in table.rows:
        for cell in row.cells:
            if "\n" in cell.raw:
                return False, "multiline_cell"
    return True, None


def extract_tables(
    parsed: mw.Wikicode,
    *,
    page_id: int,
    title: str,
    source_path: str,
    game: str,
    entity_type: str,
    safe_slug: str,
    output_root: Path,
    force: bool,
) -> tuple[
    list[TableReference],
    list[IrregularTable],
    list[tuple[int, str]],
]:
    """Extract regular tables; emit sidecars; return references and report items.

    Returns ``(table_refs, irregular, (table_idx, csv_path) `` pairs used to
    replace the table block in the markdown.
    """
    sections = assign_tables_to_sections(parsed)
    refs: list[TableReference] = []
    irregular: list[IrregularTable] = []
    replacements: list[tuple[int, str]] = []

    sections_index: dict[int, tuple[Tag, str | None]] = {
        i: item for i, item in enumerate(sections, start=1)
    }

    for table_index, (tag, section_heading) in sections_index.items():
        parsed_table = parse_table(tag)
        is_regular, reason = _table_is_regular(parsed_table)

        re.sub(r"[^A-Za-z0-9_-]+", "-", (section_heading or "lead")).strip(
            "-"
        ).lower() or "lead"
        base_filename = f"{safe_slug}-{page_id}-t{table_index}"

        if not is_regular:
            irregular.append(
                IrregularTable(
                    path=source_path,
                    section=section_heading,
                    table_index=table_index,
                    reason=reason or "irregular",
                )
            )
            replacements.append(
                (
                    table_index,
                    f"<!-- table: {table_index} (irregular: {reason}); see prose -->",
                )
            )
            continue

        header = parsed_table.header_row
        body_rows = [r for r in parsed_table.rows[1:] if not r.is_header]
        columns = [c.text for c in header.cells]

        records = []
        for row in body_rows:
            rec: dict[str, str] = {}
            for col, cell in zip(columns, row.cells):
                rec[col] = cell.text
            records.append(rec)

        table_dir = output_root / "tables" / game / entity_type
        table_dir.mkdir(parents=True, exist_ok=True)
        csv_path = table_dir / f"{base_filename}.csv"
        json_path = table_dir / f"{base_filename}.json"
        marker = f"<!-- table: {table_index} see {csv_path.relative_to(output_root).as_posix()} -->"

        if csv_path.exists() or json_path.exists():
            marker = f"<!-- table: {table_index} (existing output preserved; see {csv_path.relative_to(output_root).as_posix()}) -->"

        refs.append(
            TableReference(
                csv_path=str(csv_path.relative_to(output_root)),
                json_path=str(json_path.relative_to(output_root)),
                source_section=section_heading,
                table_index=table_index,
                row_count=len(body_rows),
                columns=columns,
            )
        )

        if not (csv_path.exists() and json_path.exists()) or force:
            _write_table_csv(
                csv_path,
                source_path,
                game,
                page_id,
                title,
                section_heading,
                table_index,
                columns,
                body_rows,
            )
            _write_table_json(json_path, parsed_table.caption, columns, records)

        replacements.append((table_index, marker))

    return refs, irregular, replacements


def _write_table_csv(
    path: Path,
    source_path: str,
    game: str,
    page_id: int,
    title: str,
    section_heading: str | None,
    table_index: int,
    columns: list[str],
    rows: list[ParsedRow],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(columns)
        for i, row in enumerate(rows, start=1):
            writer.writerow(
                [
                    source_path,
                    game,
                    page_id,
                    title,
                    section_heading or "",
                    table_index,
                    i,
                ]
                + [cell.text for cell in row.cells]
            )


def _write_table_json(
    path: Path,
    caption: str | None,
    columns: list[str],
    records: list[dict[str, str]],
) -> None:
    payload = {
        "schema_version": "1.0",
        "caption": caption,
        "columns": columns,
        "records": records,
    }
    path.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
