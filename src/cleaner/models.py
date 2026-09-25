from __future__ import annotations

import unicodedata
from collections import Counter
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


class EntityType(str, Enum):
    weapon = "weapon"
    armor = "armor"
    ring = "ring"
    shield = "shield"
    item = "item"
    character = "character"
    location = "location"
    enemy = "enemy"
    boss = "boss"
    spell = "spell"
    mechanic = "mechanic"
    storyline = "storyline"
    generic = "generic"


class SourceMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    game: str
    page_id: int
    title: str
    revision_id: int
    timestamp: str
    categories: list[str] = Field(default_factory=list)
    template: str | None = None
    revisions_observed: int = 1


class CleanedSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    heading: str | None = None
    markdown: str


class TableReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    csv_path: str
    json_path: str
    source_section: str | None
    table_index: int
    row_count: int
    columns: list[str]


class PageOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_path: str
    status: str
    output_path: str | None = None
    entity_type: str | None = None
    template: str | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class IrregularTable(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    section: str | None
    table_index: int
    reason: str


class Totals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pages: int = 0
    recognized: int = 0
    generic: int = 0
    malformed: int = 0
    skipped_existing: int = 0
    pages_with_tables: int = 0
    irregular_tables: int = 0


class CleaningReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    totals: Totals = Field(default_factory=Totals)
    by_entity_type: Counter[str] = Field(default_factory=Counter)
    by_template: Counter[str] = Field(default_factory=Counter)
    by_infobox_field: Counter[str] = Field(default_factory=Counter)
    unknown_templates: Counter[str] = Field(default_factory=Counter)
    unknown_infobox_fields: Counter[str] = Field(default_factory=Counter)
    irregular_tables: list[IrregularTable] = Field(default_factory=list)
    malformed_inputs: list[PageOutcome] = Field(default_factory=list)
    skipped_existing: list[PageOutcome] = Field(default_factory=list)


def normalize_counter(c: Counter[str]) -> dict[str, int]:
    return dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))


class GenericPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    id: str
    source: SourceMetadata
    entity_type: EntityType = EntityType.generic
    infobox: dict[str, str | list[str] | None] = Field(default_factory=dict)
    infobox_extra: dict[str, str] = Field(default_factory=dict)
    infobox_raw: dict[str, str] = Field(default_factory=dict)
    description: list[str] = Field(default_factory=list)
    sections: list[CleanedSection] = Field(default_factory=list)
    tables: list[TableReference] = Field(default_factory=list)


def build_cleaned_page(
    *,
    page_id: int,
    source: SourceMetadata,
    entity_type: EntityType,
    infobox: dict[str, str | list[str] | None],
    infobox_extra: dict[str, str],
    infobox_raw: dict[str, str],
    description: list[str],
    sections: list[CleanedSection],
    tables: list[TableReference],
) -> dict[str, Any]:
    page = GenericPage(
        id=f"mediawiki:{page_id}",
        source=source,
        entity_type=entity_type,
        infobox=infobox,
        infobox_extra=infobox_extra,
        infobox_raw=infobox_raw,
        description=description,
        sections=sections,
        tables=tables,
    )
    return page.model_dump(mode="json")
