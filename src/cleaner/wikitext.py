from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import mwparserfromhell as mw
from mwparserfromhell.nodes import Heading, Tag, Template, Text, Wikilink

from .registry import FieldSpec, Registry, TemplateSpec

HTML_ENTITIES = {
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
    "&quot;": '"',
    "&nbsp;": "\u00a0",
}

_INTERWIKI_PREFIX_RE = re.compile(r"^[a-z][a-z0-9_-]*:[^/]", re.IGNORECASE)
_FILE_LINK_RE = re.compile(r"^(?P<ns>file|image|media):", re.IGNORECASE)
_CATEGORY_LINK_RE = re.compile(r"^category:", re.IGNORECASE)

_BLANK_LINE_RUN_RE = re.compile(r"\n{3,}")
_TRAILING_WS_RE = re.compile(r"[ \t]+(?=\n)")


def _decode_entities(text: str) -> str:
    def repl(m: re.Match) -> str:
        s = m.group(0)
        return HTML_ENTITIES.get(s, s)
    return re.sub(r"&(?:amp|lt|gt|quot|nbsp);", repl, text)


@dataclass
class InfoboxExtract:
    template_name: str
    raw_items: list[tuple[str, str]]


@dataclass
class ParsedDescription:
    paragraphs: list[str]
    consumed_templates: list[Template]


def _is_missing_value(raw: str) -> bool:
    s = raw.strip()
    return s in {"", "N/A", "NA", "n/a", "None", "none", "-", "?"}


def _param_raw(param) -> str:
    """Serialise a mwparserfromhell parameter to its raw string form."""
    return str(param.value).strip()


def _param_named_keys(template: Template) -> list[str]:
    keys: list[str] = []
    for p in template.params:
        n = str(p.name).strip()
        if n:
            keys.append(n)
    return keys


def find_first_infobox_template(wikicode: mw.Wikicode, prefixes: tuple[str, ...], registry: Registry | None = None) -> Template | None:
    for t in wikicode.ifilter_templates(recursive=True):
        name = t.name.strip()
        if registry is not None and registry.is_denylist_template(name):
            continue
        if name.lower().startswith("infobox"):
            return t
        for prefix in prefixes:
            pl = prefix.lower()
            if name.lower().startswith(pl) and not name.lower().startswith(pl + "typebox") and not name.lower().startswith(pl + "weaponbox"):
                return t
    return None


def _iter_top_level_nodes(node):
    if hasattr(node, "nodes"):
        yield from node.nodes


def extract_description(wikicode: mw.Wikicode, name: str = "Description") -> ParsedDescription:
    paragraphs: list[str] = []
    consumed: list[Template] = []

    parent_map = _build_parent_map(wikicode)
    for t in wikicode.ifilter_templates(recursive=True):
        if t.name.strip().lower() != name.lower():
            continue
        parent = parent_map.get(id(t))
        # Skip if nested inside another template or inside a Tag (table cell)
        if parent is None or isinstance(parent, (Template, Tag)):
            continue

        first = second = third = ""
        extras: list[tuple[str, str]] = []
        positional: list[str] = []
        for p in t.params:
            k = str(p.name).strip()
            v = _param_raw(p)
            lk = k.lower()
            if lk in {"first paragraph", "first_paragraph", "first"}:
                first = v
            elif lk in {"second paragraph", "second_paragraph", "second"}:
                second = v
            elif lk in {"third paragraph", "third_paragraph", "third"}:
                third = v
            elif k:
                extras.append((k, v))
            else:
                positional.append(v)

        ordered = []
        if first:
            ordered.append(first)
        if second:
            ordered.append(second)
        if third:
            ordered.append(third)
        for _, v in extras:
            if v.strip():
                ordered.append(v)
        for v in positional:
            if v.strip():
                ordered.append(v)

        for raw in ordered:
            if raw.strip():
                paragraphs.append(_render_to_markdown_for_description(raw))

        consumed.append(t)

    return ParsedDescription(paragraphs=paragraphs, consumed_templates=consumed)


def _build_parent_map(root: mw.Wikicode) -> dict[int, object]:
    """Map id(child) -> parent for every descendant of ``root``.

    The map is built in a single O(N) walk and lets lookups run in O(1).
    Building a fresh map per call site keeps each site independent of any
    in-flight mutation of the tree.
    """
    parent_map: dict[int, object] = {}
    children = getattr(root, "nodes", None)
    if children:
        for c in children:
            parent_map[id(c)] = root
            _populate_parent_map(c, root, parent_map)
    return parent_map


def _populate_parent_map(node: object, parent: object, parent_map: dict[int, object]) -> None:
    children = getattr(node, "nodes", None)
    if not children:
        return
    for c in children:
        parent_map[id(c)] = node
        _populate_parent_map(c, node, parent_map)


def _find_parent(root: mw.Wikicode, target) -> object | None:
    """Return the parent of ``target`` in ``root``, or None if not found."""
    return _build_parent_map(root).get(id(target))


def _remove_template(root: mw.Wikicode, t: Template, parent_map: dict[int, object] | None = None) -> None:
    parent = (parent_map or _build_parent_map(root)).get(id(t))
    if parent is None:
        return
    container = getattr(parent, "nodes", None)
    if container is None:
        return
    try:
        container.remove(t)
    except ValueError:
        pass


def _convert_link(text: str) -> str | None:
    """Return Markdown rendering for a Wikilink, or None for content we drop."""
    raw = str(text)
    if not raw.startswith("[[") or not raw.endswith("]]"):
        return None
    inner = raw[2:-2]
    if "|" in inner:
        target, _, label = inner.partition("|")
        target = target.strip()
        label = label.strip()
    else:
        target = inner.strip()
        target = target.removeprefix(":")
        label = target.replace("_", " ")
    if _CATEGORY_LINK_RE.match(target):
        return ""
    if _FILE_LINK_RE.match(target):
        return ""
    if ":" in target and _INTERWIKI_PREFIX_RE.match(target):
        return label
    return f"[{label}]({target})"


def _convert_wikilink(node: Wikilink) -> str | None:
    raw = str(node)
    return _convert_link(raw)


def _convert_template_to_markdown(t: Template) -> str:
    name = t.name.strip() or "Template"
    inner_parts = []
    for p in t.params:
        key = str(p.name).strip()
        val = _param_raw(p)
        if key and not key.isdigit():
            inner_parts.append(f"{key}={val}")
        else:
            inner_parts.append(val)
    body = "|".join(inner_parts)
    return f"{{{{{name}|{body}}}}}" if body else f"{{{{{name}}}}}"


def _tag_contents_inline(tag: Tag, ctx: dict, marker: str) -> str:
    contents = getattr(tag, "contents", None)
    if contents is None or not hasattr(contents, "nodes"):
        return ""
    inner = "".join(_convert_node(c, ctx) for c in contents.nodes)
    return f"{marker}{inner}{marker}"


def _convert_tag_to_markdown(tag: Tag) -> str:
    raw = str(tag)
    if raw.startswith("{|"):
        return raw
    return raw


HEADING_LEVEL_MAP = {
    2: "##",
    3: "###",
    4: "####",
    5: "#####",
    6: "######",
}


_BOLD_RE = re.compile(r"'''(.*?)'''", re.DOTALL)
_ITALIC_RE = re.compile(r"''(.*?)''", re.DOTALL)
_INTERWIKI_BARE_RE = re.compile(r"(?m)^\s*(?P<pref>[a-z]{2,}:)\S.*$")


def _convert_bold_italic(text: str) -> str:
    text = _BOLD_RE.sub(r"**\1**", text)
    text = _ITALIC_RE.sub(r"*\1*", text)
    return text


def _convert_interwiki_text(text: str) -> str:
    """Convert bare-line interwiki prefixes (e.g. ``ru:Foo``) to plain text.

    These are language links without square brackets that mwparserfromhell
    parses as plain text. We strip them so they don't leak into Markdown.
    """
    return _INTERWIKI_BARE_RE.sub("", text)


def _convert_node(n, ctx: dict) -> str:
    if isinstance(n, Text):
        return _convert_bold_italic(_convert_interwiki_text(_decode_entities(str(n))))
    if isinstance(n, Wikilink):
        link = _convert_wikilink(n)
        return link if link is not None else ""
    if isinstance(n, Template):
        name = n.name.strip()
        if ctx["registry"].is_denylist_template(name):
            return ""
        if name.lower() == ctx.get("description_name", "description").lower():
            return ""
        routed_id = id(ctx.get("routed_template")) if ctx.get("routed_template") else None
        if routed_id is not None and id(n) == routed_id:
            return ""
        return _convert_template_to_markdown(n)
    if isinstance(n, Tag):
        raw = str(n)
        if raw.startswith("{|"):
            return f"\n{ctx.get('table_placeholder', '<!-- table -->')}\n"
        if raw.startswith("'''") and raw.endswith("'''") and len(raw) >= 6:
            inner = raw[3:-3]
            return f"**{_convert_bold_italic(_decode_entities(inner))}**"
        if raw.startswith("''") and raw.endswith("''") and len(raw) >= 4:
            inner = raw[2:-2]
            return f"*{_convert_bold_italic(_decode_entities(inner))}*"
        tag_name = raw.lstrip("<").split(">", 1)[0].split(" ", 1)[0].rstrip("/").lower()
        if tag_name == "br":
            return "  \n"
        if tag_name in {"onlyinclude", "noinclude", "includeonly", "ref", "gallery"}:
            contents = getattr(n, "contents", None)
            if tag_name == "ref":
                return ""
            if tag_name == "gallery":
                return ""
            if contents is not None and hasattr(contents, "nodes"):
                inner_parts = [_convert_node(c, ctx) for c in contents.nodes]
                return "".join(inner_parts)
        if tag_name in {"i", "em"}:
            return _tag_contents_inline(n, ctx, "*")
        if tag_name in {"b", "strong"}:
            return _tag_contents_inline(n, ctx, "**")
        return raw
    if isinstance(n, Heading):
        HEADING_LEVEL_MAP.get(n.level, "######")
        return ""
    return _convert_bold_italic(_decode_entities(str(n)))


def _node_iter_at_top(wikicode: mw.Wikicode):
    yield from wikicode.nodes


def _walk_sections(wikicode: mw.Wikicode, description_name: str = "Description") -> list[tuple[str | None, str | None, list]]:
    """Walk top-level nodes and group them into sections.

    Each yielded tuple is ``(heading_title, heading_markdown, body_nodes)``.
    ``heading_title`` is ``None`` for the lead. ``heading_markdown`` is the
    full heading line (including ``##``) suitable for prepending to the body,
    or ``None`` for the lead.
    """
    sections: list[tuple[str | None, str | None, list]] = [(None, None, [])]
    for n in wikicode.nodes:
        if isinstance(n, Heading):
            h_title = _decode_entities(str(n.title)).strip()
            marker = HEADING_LEVEL_MAP.get(n.level, "######")
            sections.append((h_title, f"{marker} {h_title}", []))
            continue
        sections[-1][2].append(n)
    return sections


def _render_nodes(nodes: list, registry: Registry, description_name: str, table_marker: str | None, routed_template: Template | None) -> str:
    ctx = {
        "registry": registry,
        "description_name": description_name,
        "table_placeholder": table_marker or "<!-- table -->",
        "routed_template": routed_template,
    }
    parts: list[str] = []
    for n in nodes:
        s = _convert_node(n, ctx)
        if s:
            parts.append(s)
    text = "".join(parts)
    return text


def _normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip(" \t") for line in s.split("\n")]
    s = "\n".join(lines)
    s = _BLANK_LINE_RUN_RE.sub("\n\n", s)
    return s.strip("\n")


def _render_to_markdown_for_description(text: str) -> str:
    return _normalize_text(text)


def extract_infobox(template: Template) -> InfoboxExtract:
    raw_items: list[tuple[str, str]] = []
    for p in template.params:
        k = str(p.name).strip()
        v = _param_raw(p)
        if k:
            raw_items.append((k, v))
    return InfoboxExtract(template_name=template.name.strip(), raw_items=raw_items)


def split_infobox_params(
    raw_items: list[tuple[str, str]],
    spec: TemplateSpec | None,
) -> tuple[dict[str, str | list[str] | None], dict[str, str], dict[str, str]]:
    """Split raw infobox params into recognised / extra / raw buckets.

    - For recognised fields, missing values become None; multi-values become a list.
    - Extra keys land in infobox_extra verbatim.
    - The original parameter text is kept (last value wins for raw_map).
    """
    recognised: dict[str, str | list[str] | None] = {}
    extra: dict[str, str] = {}
    raw_map: dict[str, str] = {}

    field_index: dict[str, FieldSpec] = {}
    if spec is not None:
        for f in spec.fields:
            field_index[f.canonical.lower()] = f

    seen_extra: set[str] = set()

    for key, val in raw_items:
        key_norm = key.strip()
        raw_map[key_norm] = val
        fspec = field_index.get(key_norm.lower())
        if _is_missing_value(val) and fspec is not None:
            recognised.setdefault(fspec.canonical, None)
        elif fspec is not None:
            if fspec.multi:
                bucket = recognised.setdefault(fspec.canonical, [])
                if isinstance(bucket, list):
                    bucket.append(val)
            else:
                if fspec.canonical in recognised and recognised[fspec.canonical] != val:
                    extra.setdefault(key_norm, val)
                else:
                    recognised[fspec.canonical] = val
        else:
            if key_norm not in seen_extra:
                extra[key_norm] = val
                seen_extra.add(key_norm)

    return recognised, extra, raw_map


def markdown_for_page(
    wikitext: str,
    registry: Registry,
    *,
    parsed: mw.Wikicode | None = None,
    infobox_template: Template | None = None,
    consumed_descriptions: list[Template] | None = None,
    table_markers: dict[int, str] | None = None,
) -> list[tuple[str | None, str]]:
    """Return normalised (heading, markdown) section list.

    When ``parsed`` is provided, this function mutates it in place to drop
    denylist templates and matched Description templates before rendering. Do
    not re-use ``parsed`` for other purposes after this call returns. When
    ``parsed`` is ``None`` a fresh tree is parsed from ``wikitext``.
    """
    if parsed is None:
        parsed = mw.parse(wikitext)

    routed = infobox_template
    if routed is not None:
        for t in parsed.ifilter_templates(recursive=False):
            if t.name.strip() == routed.name.strip():
                routed = t
                break

    parent_map = _build_parent_map(parsed)
    seen_denied: set[int] = set()
    for t in parsed.ifilter_templates(recursive=True):
        if id(t) in seen_denied:
            continue
        if registry.is_denylist_template(t.name):
            _remove_template(parsed, t, parent_map)
            seen_denied.add(id(t))

    if consumed_descriptions:
        for t in consumed_descriptions:
            _remove_template(parsed, t)

    sections = _walk_sections(parsed)

    rendered: list[tuple[str | None, str]] = []
    for heading, heading_md, nodes in sections:
        body = _render_nodes(nodes, registry, "Description", "<!-- table -->", routed)
        if heading_md and body:
            md = f"{heading_md}\n{body}"
        elif heading_md:
            md = heading_md
        else:
            md = body
        rendered.append((heading, _normalize_text(md)))
    return rendered
