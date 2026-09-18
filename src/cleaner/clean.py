from __future__ import annotations

from pathlib import Path

import mwparserfromhell as mw

from . import tables as T
from . import wikitext as W
from .io import write_cleaned_page
from .loader import LoaderError, load_raw
from .models import (
    CleanedSection,
    CleaningReport,
    PageOutcome,
    build_cleaned_page,
    nfc,
)
from .registry import Registry
from .router import entity_type_enum, route

def clean_one(
    *,
    source_path: Path,
    source_root: Path,
    output_root: Path,
    registry: Registry,
    report: CleaningReport,
    force: bool = False,
) -> PageOutcome:
    """Clean a single raw file; mutate `report` and write outputs.

    Returns the ``PageOutcome`` describing what happened.
    """
    relative = _rel(source_path, source_root)
    try:
        loaded = load_raw(source_path, source_root)
    except LoaderError as e:
        outcome = PageOutcome(
            source_path=relative,
            status="malformed",
            errors=[e.reason],
        )
        # report.malformed_inputs.append(outcome)
        report.totals.malformed += 1
        return outcome

    parsed = mw.parse(loaded.wikitext)

    infobox = W.find_first_infobox_template(parsed, registry.infobox_prefixes, registry=registry)
    infobox_template_name = infobox.name.strip() if infobox is not None else None

    desc = W.extract_description(parsed)

    router = route(
        source_path=relative,
        categories=loaded.metadata.categories,
        infobox_template_name=infobox_template_name,
        registry=registry,
    )

    entity_type_value = router.entity_type
    template_to_record = router.template

    infobox_extract = W.extract_infobox(infobox) if infobox is not None else None
    if infobox_extract is not None:
        recognised, extra, raw_map = W.split_infobox_params(
            infobox_extract.raw_items, router.spec
        )
        loaded.metadata.template = infobox_extract.template_name
        template_to_record = infobox_extract.template_name
        for k in raw_map:
            report.by_infobox_field[nfc(k)] += 1
        for k in extra:
            report.unknown_infobox_fields[nfc(k)] += 1
    else:
        recognised, extra, raw_map = {}, {}, {}

    unknown_templates = _collect_unknown_templates(parsed, registry)
    for name in unknown_templates:
        report.unknown_templates[name] += 1

    safe_slug = _safe_slug_for(loaded.metadata.title)
    output_path = (
        output_root
        / loaded.metadata.game
        / entity_type_value
        / f"{safe_slug}-{loaded.metadata.page_id}.json"
    )

    # if output_path.exists() and not force:
    #     outcome = PageOutcome(
    #         source_path=relative,
    #         status="skipped_existing",
    #         output_path=str(output_path.relative_to(output_root)),
    #         entity_type=entity_type_value,
    #         template=template_to_record,
    #         warnings=["exists"],
    #     )
    #     report.skipped_existing.append(outcome)
    #     report.totals.skipped_existing += 1
    #     return outcome

    sections = W.markdown_for_page(
        loaded.wikitext,
        registry,
        parsed=parsed,
        infobox_template=infobox,
        consumed_descriptions=desc.consumed_templates,
    )

    sections_clean: list[CleanedSection] = [
        CleanedSection(heading=h, markdown=md) for h, md in sections
    ]

    table_refs, irregular, _replacements = T.extract_tables(
        parsed,
        page_id=loaded.metadata.page_id,
        title=loaded.metadata.title,
        source_path=relative,
        game=loaded.metadata.game,
        entity_type=entity_type_value,
        safe_slug=safe_slug,
        output_root=output_root,
        force=force,
    )

    for ir in irregular:
        report.irregular_tables.append(ir)
    report.totals.irregular_tables += len(irregular)
    if table_refs:
        report.totals.pages_with_tables += 1

    report.by_entity_type[entity_type_value] += 1
    if template_to_record:
        report.by_template[template_to_record] += 1
    if entity_type_value == "generic":
        report.totals.generic += 1
    else:
        report.totals.recognized += 1

    payload = build_cleaned_page(
        page_id=loaded.metadata.page_id,
        source=loaded.metadata.model_copy(update={"template": template_to_record}),
        entity_type=entity_type_enum(entity_type_value),
        infobox=recognised,
        infobox_extra=extra,
        infobox_raw=raw_map,
        description=desc.paragraphs,
        sections=sections_clean,
        tables=table_refs,
    )

    payload = _normalise_payload(payload)

    write_cleaned_page(output_path, payload)
    report.totals.pages += 1

    outcome = PageOutcome(
        source_path=relative,
        status="ok",
        output_path=str(output_path.relative_to(output_root)),
        entity_type=entity_type_value,
        template=template_to_record,
        warnings=list(loaded.warnings),
    )
    return outcome


def _collect_unknown_templates(parsed: mw.Wikicode, registry: Registry) -> list[str]:
    names: list[str] = []
    for t in parsed.ifilter_templates(recursive=True):
        n = t.name.strip().lower()
        if registry.is_denylist_template(n):
            continue
        if registry.find_infobox_spec(t.name) is not None:
            continue
        names.append(t.name.strip())
    return names


def _safe_slug_for(title: str) -> str:
    from .registry import safe_slug
    return safe_slug(title)


def _rel(p: Path, root: Path) -> str:
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return p.as_posix()


def _normalise_payload(payload: dict) -> dict:
    """Final deterministic scrub: NFC strings, sorted keys, etc.

    The writer does the actual JSON serialisation; this is a safety pass.
    """
    def walk(v):
        if isinstance(v, dict):
            return {k: walk(val) for k, val in sorted(v.items())}
        if isinstance(v, list):
            return [walk(x) for x in v]
        if isinstance(v, str):
            return nfc(v)
        return v
    return walk(payload)
