from __future__ import annotations

from dataclasses import dataclass

from .models import EntityType
from .registry import Registry


@dataclass
class RouterResult:
    entity_type: str
    template: str | None
    spec: object | None  # TemplateSpec


def _first_path_segment(rel: str) -> str:
    return rel.split("/", 1)[0] if rel else "uncategorized"


def _directory_under_game(rel: str) -> str:
    parts = rel.split("/")
    if len(parts) >= 2:
        return parts[1]
    return ""


def route(
    *,
    source_path: str,
    categories: list[str],
    infobox_template_name: str | None,
    registry: Registry,
) -> RouterResult:
    """Pick an entity_type and routed template spec."""
    if infobox_template_name:
        spec = registry.find_infobox_spec(infobox_template_name)
        if spec is not None:
            return RouterResult(
                entity_type=spec.entity_type,
                template=spec.template_name,
                spec=spec,
            )
    for cat in categories:
        et = registry.entity_from_category(cat)
        if et is not None:
            return RouterResult(entity_type=et, template=None, spec=None)

    game = _first_path_segment(source_path)
    if game == "uncategorized":
        directory = _directory_under_game(source_path)
        if directory:
            et = registry.entity_from_directory(directory)
            if et is not None:
                return RouterResult(entity_type=et, template=None, spec=None)

    return RouterResult(entity_type="generic", template=None, spec=None)


def entity_type_enum(name: str) -> EntityType:
    return EntityType(name)
