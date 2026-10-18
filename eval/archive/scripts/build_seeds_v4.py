"""Build the v4 evaluation seed pools: all-system, quota'd, LLM-judged.

Run:
    python -m eval.build_seeds_v4 seeds --only q002,q005,q013,q023,q025,q039
    python -m eval.build_seeds_v4 seeds                  # all questions

``seeds`` writes one pool file per question to ``eval/v4/candidates/qNNN.json``
plus ``_build_summary.json`` (per-source composition, for quota tuning).
Unlike the v3 pools (human -> template -> gold -> dense -> bm25, human ids
always winning), v4 pools are composed from every compared system plus the
human-judged ids, with a per-source quota so no single system crowds out the
others. There is NO gold-pages tier: the LLM grades every pool id fresh, and
human grades live only in ``eval/v4/seeds/anchors.json`` (never in qrels).

Pool order (deduped, first source wins, total cap applied last with human
ids always kept above the cap):

    1. human    — ids already judged in eval/v3/seeds/human-judgments.json.
                  They are LLM-graded like every other pool id; their human
                  grades provide the anchors and the probe holdout only.
                  (The question set now has 100 questions; q051–q100 have no
                  human seeds, so they get no anchors/probes and are graded
                  by the rubric alone.)
    2. template — expected-entity pages via titles + link graph (same
                  junk cap of 80 ids as v3; reuses the v3 link index).
    3. dense    — contextfromvectors top-50
    4. enhanced — contextfromvectors_enhanced top-50
    5. bm25     — sparse_retrieval.bm25_retrieval top-50
    6. hybrid   — contextfromvectors_hybrid top-50

Each system source contributes at most ``--quota`` NEW ids (dedup
first-come), so heavy overlap between dense/enhanced/hybrid cannot push
BM25-only or template ids out of the pool. Ids reached by several systems
carry every system in their ``sources`` list (provenance is free; only NEW
ids consume quota). Total cap ``--cap`` (300).

Requires the embedding proxy on localhost:3001 (dense/enhanced/hybrid legs)
and the NeuralWatt endpoint (query enhancement). Uses the shared
``get_qdrant_client`` singleton, so it can never fight the retriever's
vector store over the local Qdrant file lock.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval.build_seeds import (  # noqa: E402
    SNIPPET_LEN,
    _load_human_seeds,
    _load_questions,
    _parse_only,
    load_or_build_link_index,
    matched_pages_for_entities,
    scroll_all_chunks,
    title_indexes,
)
from src.cleaner.registry import Registry  # noqa: E402
from src.config import settings  # noqa: E402

V4_DIR = REPO_ROOT / "eval" / "v4"
DEFAULT_INDEX = (
    REPO_ROOT / "eval" / "v3" / "seeds" / "link-index.json"
    if (REPO_ROOT / "eval" / "v3" / "seeds" / "link-index.json").exists()
    else V4_DIR / "seeds" / "link-index.json"
)

SYSTEM_SOURCES = ("dense", "enhanced", "bm25", "hybrid")


def _doc_id_of(item) -> str | None:
    doc = None
    if isinstance(item, tuple) and len(item) == 2:
        doc = item[0]
    elif hasattr(item, "metadata"):
        doc = item
    if doc is None:
        return None
    return (getattr(doc, "metadata", None) or {}).get("document_id")


def _score_of(item) -> float | None:
    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1],
                                                                 (int, float)):
        return float(item[1])
    return None


def system_hits(source: str, query: str, k: int) -> list[tuple[str, float | None]]:
    """Call one production retrieval system; returns [(doc_id, score)]."""
    if source == "bm25":
        from src.retriever.sparse_retrieval import bm25_retrieval

        items = bm25_retrieval(query=query, k=k)
    else:
        from src.retriever import querying

        fn = {
            "dense": querying.contextfromvectors,
            "enhanced": querying.contextfromvectors_enhanced,
            "hybrid": querying.contextfromvectors_hybrid,
        }[source]
        items = fn(query, k)
    out: list[tuple[str, float | None]] = []
    for item in items:
        doc_id = _doc_id_of(item)
        if doc_id:
            out.append((doc_id, _score_of(item)))
    return out


def build_pool_v4(
    q: dict,
    human_judgments: dict[str, int],
    chunks_by_id: dict[str, dict],
    title_to_ids: dict[str, list[str]],
    titles_by_norm: dict[str, list[str]],
    link_index: dict,
    system_k: int,
    quota: int,
    cap: int,
    template_cap: int,
) -> tuple[list[dict], dict[str, int], dict[str, int]]:
    """Build one pool; returns (pool, tagged_counts_per_source, fresh_counts).

    ``counts`` is the auditable composition (how many pool ids each source
    tagged); ``fresh`` is how many NEW ids each system source contributed
    (what the quota constrains).
    """
    candidates: dict[str, dict] = {}
    order: list[str] = []
    counts: dict[str, int] = {"human": 0, "template": 0,
                              **{s: 0 for s in SYSTEM_SOURCES}}
    fresh: dict[str, int] = {s: 0 for s in SYSTEM_SOURCES}

    def _add(doc_id: str, source: str, score: float | None = None,
             score_key: str | None = None) -> bool:
        meta = chunks_by_id.get(doc_id)
        if meta is None:
            return False
        existing = candidates.get(doc_id)
        if existing is None:
            order.append(doc_id)
            existing = candidates[doc_id] = {
                "document_id": doc_id,
                "title": meta["title"],
                "section": meta["section"],
                "game": meta["game"],
                "entity_type": meta["entity_type"],
                "snippet": meta["page_content"][:SNIPPET_LEN],
                "sources": [source],
                "dense_score": None,
                "bm25_score": None,
            }
        elif source not in existing["sources"]:
            existing["sources"].append(source)
        if score is not None and score_key:
            key = {"dense": "dense_score", "bm25": "bm25_score"}[score_key]
            if existing[key] is None:
                existing[key] = score
        return True

    human_ids = sorted(d for d in human_judgments if d in chunks_by_id)
    missing_human = len(human_judgments) - len(human_ids)
    if missing_human:
        print(f"    note: {missing_human} human id(s) not in Qdrant, "
              f"excluded from pool", file=sys.stderr)
    for doc_id in human_ids:
        if _add(doc_id, "human"):
            counts["human"] += 1

    template_pages = matched_pages_for_entities(
        q.get("expected_entities") or [], link_index, titles_by_norm
    )
    template_added = template_truncated = 0
    for page in sorted(template_pages, key=lambda p: (template_pages[p], p)):
        for doc_id in sorted(title_to_ids.get(page, [])):
            if template_added >= template_cap:
                template_truncated += 1
                continue
            if _add(doc_id, "template"):
                template_added += 1
                counts["template"] += 1
    if template_truncated:
        print(f"    note: template tier truncated at the {template_cap}-id "
              f"junk cap ({template_truncated} id(s) skipped)", file=sys.stderr)

    for source in SYSTEM_SOURCES:
        try:
            hits = system_hits(source, q["question"], system_k)
        except Exception as e:  # noqa: BLE001 - system failure is non-fatal
            print(f"    {source} failed for {q['id']}: {e}", file=sys.stderr)
            fresh[source] = 0
            continue
        score_key = "dense" if source == "dense" else (
            "bm25" if source == "bm25" else None)
        added = 0
        for doc_id, score in hits:
            if doc_id in candidates:
                # Already pooled via an earlier source: tag provenance only,
                # free of quota, so sources stays fully auditable.
                _add(doc_id, source, score, score_key)
                continue
            if added >= quota:
                continue  # quota spent; only tag ids that are already pooled
            if _add(doc_id, source, score, score_key):
                added += 1
        fresh[source] = added

    for source in SYSTEM_SOURCES:
        counts[source] = sum(
            1 for c in candidates.values() if source in c["sources"])

    kept = [d for d in order if "human" in candidates[d]["sources"]]
    for d in order:
        if len(kept) >= cap:
            break
        if d not in kept:
            kept.append(d)
    return [candidates[d] for d in kept], counts, fresh


def cmd_seeds(args: argparse.Namespace) -> None:
    if not Path(settings.QDRANT_PATH).exists():
        sys.exit(f"Qdrant storage not found at {settings.QDRANT_PATH}; "
                 f"run the indexing pipeline first")
    only = _parse_only(args.only)
    questions = _load_questions(args.questions, only)
    human = _load_human_seeds(args.human_seeds)
    registry = Registry.default()

    print("Loading link graph index...")
    link_index = load_or_build_link_index(args.index, registry,
                                          rebuild=args.rebuild_index)

    print("Opening shared Qdrant client (retriever singleton)...")
    from src.retriever.vector_store import get_qdrant_client

    client = get_qdrant_client()
    chunks = scroll_all_chunks(client)
    chunks_by_id = {c["document_id"]: c for c in chunks}
    title_to_ids, titles_by_norm = title_indexes(chunks)
    print(f"  {len(chunks)} chunks, {len(title_to_ids)} titles")

    args.out.mkdir(parents=True, exist_ok=True)
    built_at = datetime.now(UTC).isoformat(timespec="seconds").replace(
        "+00:00", "Z")
    summary_rows = []
    for q in questions:
        pool, counts, fresh = build_pool_v4(
            q, human.get(q["id"], {}), chunks_by_id, title_to_ids,
            titles_by_norm, link_index, args.system_k, args.quota,
            args.cap, args.template_cap,
        )
        out_path = args.out / f"{q['id']}.json"
        tmp = out_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(pool, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        os.replace(tmp, out_path)
        summary_rows.append({"question": q["id"], "pool_size": len(pool),
                             "counts": counts, "fresh": fresh})
        src = " ".join(f"{name}={counts[name]}"
                       for name in ("human", "template", *SYSTEM_SOURCES))
        fresh_str = " ".join(f"{name}+{fresh[name]}" for name in SYSTEM_SOURCES)
        print(f"  {q['id']}: pool={len(pool)} ({src}; fresh {fresh_str}) "
              f"-> {out_path.name}")

    summary_path = args.out / "_build_summary.json"
    summary = {
        "built_at": built_at,
        "config": {
            "system_k": args.system_k,
            "quota": args.quota,
            "cap": args.cap,
            "template_cap": args.template_cap,
        },
        "questions": summary_rows,
    }
    tmp = summary_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    os.replace(tmp, summary_path)
    print(f"\nWrote {len(questions)} pool file(s) to {args.out}")
    print(f"Wrote per-source composition summary to {summary_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p_seeds = sub.add_parser("seeds", help="build per-question v4 seed pools")
    p_seeds.add_argument("--questions", type=Path,
                         default=REPO_ROOT / "eval" / "questions.json")
    p_seeds.add_argument("--human-seeds", type=Path,
                         default=REPO_ROOT / "eval" / "v3" / "seeds"
                         / "human-judgments.json")
    p_seeds.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    p_seeds.add_argument("--out", type=Path,
                         default=V4_DIR / "candidates")
    p_seeds.add_argument("--only", type=str, default=None,
                         help="comma-separated question ids (e.g. q002,q005)")
    p_seeds.add_argument("--system-k", type=int, default=50,
                         help="top-k retrieved from each system source")
    p_seeds.add_argument("--quota", type=int, default=60,
                         help="max NEW ids per system source")
    p_seeds.add_argument("--cap", type=int, default=300,
                         help="total pool cap (human ids always kept)")
    p_seeds.add_argument("--template-cap", type=int, default=80)
    p_seeds.add_argument("--rebuild-index", action="store_true")
    p_seeds.set_defaults(func=cmd_seeds)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
