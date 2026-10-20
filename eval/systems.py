"""Single registry of retrieval systems under evaluation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

SYSTEMS: dict[str, str] = {
    "dense": "contextfromvectors",
    "enhanced": "contextfromvectors_enhanced",
    "bm25": "contextfromvectors_bm25",
    "hybrid": "contextfromvectors_hybrid",
    "hybrid_enhanced": "contextfromvectors_hybrid_enhanced",
}


def system_names() -> list[str]:
    return list(SYSTEMS)


def resolve_systems(spec: str | None) -> list[str]:
    """Resolve ``all`` or a comma-separated list to registry names."""
    if not spec or spec == "all":
        return system_names()
    requested = [name.strip() for name in spec.split(",") if name.strip()]
    unknown = [name for name in requested if name not in SYSTEMS]
    if unknown:
        raise ValueError(
            f"unknown system(s): {', '.join(unknown)}; "
            f"choose from {', '.join(SYSTEMS)}"
        )
    return requested


def get_retriever(name: str) -> Callable[..., list[Any]]:
    """Import a retriever lazily so tests never open Qdrant.

    Note: ``contextfromvectors_hybrid`` fuses BM25 with plain dense retrieval,
    while ``contextfromvectors_hybrid_enhanced`` fuses BM25 with
    ``contextfromvectors_enhanced`` (LLM-rewritten dense retrieval).
    """
    if name not in SYSTEMS:
        raise KeyError(f"unknown system: {name!r}")
    from src.retriever import querying

    return getattr(querying, SYSTEMS[name])


def retrieve(name: str, query: str, k: int) -> list[Any]:
    return get_retriever(name)(query=query, k=k)
