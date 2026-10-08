"""Build a per-question candidate pool by unioning dense + BM25 + entity filters.

Run:
    python -m eval.build_candidates --questions eval/questions.json --out eval/v1/candidates --k 50

v2 deeper pools:
    python -m eval.build_candidates --out eval/v2/candidates --dense-k 100 --bm25-k 100 --cap 250

``--k`` sets the pool depth for both dense and BM25; ``--dense-k``/``--bm25-k``
override it per source. Entity/title/slug matches are always added uncapped;
``--cap`` applies to the final ranked list only. Re-running is idempotent;
per-question candidate files are overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# These constants are hardcoded in src/retriever/querying.py and duplicated
# here to avoid importing that module at load time (it instantiates a
# langchain QdrantVectorStore that probes the embedding model on init and
# would race with our own QdrantClient over the SQLite lock).
COLLECTION_NAME = "darksoulswiki"
QDRANT_PATH = str(REPO_ROOT / "data" / "qdrant")


class _EMBED_CACHE:
    """Lazy holder for the OpenAI embeddings client."""

SNIPPET_LEN = 1000
MAX_CANDIDATES_PER_QUESTION = 80
BM25_TOKEN_RE = re.compile(r"[a-z0-9]+")
WIKI_MARKUP_RE = re.compile(
    r"\[\[[^\]]*\]\]|\{\{[^{}]*\}\}|'''?|''|`+|^=+\s*=*$|={2,}",
    flags=re.MULTILINE,
)


def _strip_wiki(s: str) -> str:
    return WIKI_MARKUP_RE.sub(" ", s)


def _tokenize(text: str) -> list[str]:
    return BM25_TOKEN_RE.findall(_strip_wiki(text).lower())


def _scroll_all_chunks(client: QdrantClient) -> list[dict]:
    """Pull every point from the collection with its payload."""
    out: list[dict] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=512,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            payload = p.payload or {}
            meta = payload.get("metadata") or {}
            out.append(
                {
                    "document_id": meta.get("document_id", str(p.id)),
                    "title": meta.get("title", ""),
                    "section": meta.get("section", ""),
                    "game": meta.get("game", ""),
                    "entity_type": meta.get("entity_type", ""),
                    "page_content": payload.get("page_content", ""),
                }
            )
        if offset is None:
            break
    return out


def _build_title_index(chunks: list[dict]) -> dict[str, list[str]]:
    """Map title -> [document_id, ...]. All chunks of a page share a title,
    so a title match already pulls the full page."""
    idx: dict[str, list[str]] = {}
    for c in chunks:
        idx.setdefault(c["title"], []).append(c["document_id"])
    return idx


def _build_slug_index() -> dict[str, list[str]]:
    """Map ``slugified entity token`` -> [title, ...].

    Used as a fallback when the expected entity does not appear verbatim in
    the page title but does appear in the source-file slug. Looked up once
    at startup from ``data/structured/<Game>/**/*.json``.
    """
    idx: dict[str, list[str]] = {}
    structured = REPO_ROOT / "data" / "structured"
    for jf in structured.rglob("*.json"):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        title = (data.get("source") or {}).get("title", "")
        if not title:
            continue
        stem_l = jf.stem.lower()
        for frag in stem_l.split("-"):
            if len(frag) >= 4:
                idx.setdefault(frag, []).append(title)
    return idx


def _rank_candidates(candidates: dict[str, dict]) -> list[dict]:
    return sorted(
        candidates.values(),
        key=lambda c: (
            -len(c["sources"]),
            -(c["dense_score"] if c["dense_score"] is not None else -1.0),
            -(c["bm25_score"] if c["bm25_score"] is not None else -1.0),
            c["title"],
            c["document_id"],
        ),
    )


def build_for_question(
    q: dict,
    client: QdrantClient,
    chunks_by_id: dict[str, dict],
    title_to_chunk_ids: dict[str, list[str]],
    slug_index: dict[str, list[str]],
    bm25: BM25Okapi,
    chunks: list[dict],
    dense_k: int,
    bm25_k: int | None = None,
    cap: int = MAX_CANDIDATES_PER_QUESTION,
) -> list[dict]:
    if bm25_k is None:
        bm25_k = dense_k
    candidates: dict[str, dict] = {}

    def _add(doc_id: str, source: str, *, dense=None, bm25=None) -> None:
        if doc_id not in chunks_by_id:
            return
        meta = chunks_by_id[doc_id]
        existing = candidates.get(doc_id)
        if existing is None:
            candidates[doc_id] = {
                "document_id": doc_id,
                "title": meta["title"],
                "section": meta["section"],
                "game": meta["game"],
                "entity_type": meta["entity_type"],
                "snippet": meta["page_content"][:SNIPPET_LEN],
                "sources": [source],
                "dense_score": dense,
                "bm25_score": bm25,
            }
        else:
            if source not in existing["sources"]:
                existing["sources"].append(source)
            if dense is not None and existing["dense_score"] is None:
                existing["dense_score"] = dense
            if bm25 is not None and existing["bm25_score"] is None:
                existing["bm25_score"] = bm25

    # 1) Dense pool (uses raw Qdrant client + a local OpenAI embeddings
    #    instance so we don't import src.retriever.querying, which would
    #    instantiate a competing QdrantVectorStore on the same SQLite).
    try:
        if not hasattr(_EMBED_CACHE, "_embed"):
            from src.config import settings
            _EMBED_CACHE._embed = OpenAIEmbeddings(  # type: ignore[attr-defined]
                model="qwen3-embedding-0.6b",
                base_url="http://localhost:3001/v1",
                api_key=settings.freellm_api_key,
                check_embedding_ctx_length=False,
            )
        embed = _EMBED_CACHE._embed  # type: ignore[attr-defined]
        query_vec = embed.embed_query(q["question"])
        raw = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=dense_k,
            with_payload=True,
        ).points
        for hit in raw:
            doc_id = ((hit.payload or {}).get("metadata") or {}).get("document_id")
            if doc_id:
                _add(doc_id, "dense", dense=float(hit.score))
    except Exception as e:  # noqa: BLE001 - dense failure (HTTP/server/etc) is non-fatal, BM25+entity still run
        print(f"  dense failed for {q['id']}: {e}", file=sys.stderr)

    # 2) BM25 pool
    q_tokens = _tokenize(q["question"])
    if q_tokens:
        bm25_scores = bm25.get_scores(q_tokens)
        top_idx = sorted(
            range(len(bm25_scores)),
            key=lambda i: -bm25_scores[i],
        )[:bm25_k]
        for idx in top_idx:
            score = bm25_scores[idx]
            if score <= 0:
                continue
            _add(chunks[idx]["document_id"], "bm25", bm25=float(score))

    # 3) Entity/title filter
    matched_titles: set[str] = set()
    for entity in q.get("expected_entities", []) or []:
        ent_l = entity.lower()
        # (a) substring match against Qdrant payload titles
        for title, chunk_ids in title_to_chunk_ids.items():
            if ent_l in title.lower():
                for cid in chunk_ids:
                    _add(cid, "entity")
                    matched_titles.add(title)
        # (b) slug fallback: entity token -> title via on-disk filename index
        for frag in ent_l.split():
            if len(frag) >= 4:
                for title in slug_index.get(frag, []):
                    if title in matched_titles:
                        continue
                    for cid in title_to_chunk_ids.get(title, []):
                        _add(cid, "entity")
                    matched_titles.add(title)

    ranked = _rank_candidates(candidates)
    return ranked[:cap]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--questions",
        type=Path,
        default=REPO_ROOT / "eval" / "questions.json",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "eval" / "v1" / "candidates",
    )
    ap.add_argument("--k", type=int, default=50,
                    help="default pool depth for both dense and BM25")
    ap.add_argument("--dense-k", type=int, default=None,
                    help="override dense pool depth (default: --k)")
    ap.add_argument("--bm25-k", type=int, default=None,
                    help="override BM25 pool depth (default: --k)")
    ap.add_argument("--cap", type=int, default=MAX_CANDIDATES_PER_QUESTION,
                    help="final per-question candidate cap")
    args = ap.parse_args()

    dense_k = args.dense_k if args.dense_k is not None else args.k
    bm25_k = args.bm25_k if args.bm25_k is not None else args.k

    if not Path(QDRANT_PATH).exists():
        sys.exit(f"Qdrant storage not found at {QDRANT_PATH}; "
                 f"run the indexing pipeline first")

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {len(questions)} questions; scrolling Qdrant collection...")
    client = QdrantClient(path=QDRANT_PATH)
    chunks = _scroll_all_chunks(client)
    print(f"  loaded {len(chunks)} chunks")

    chunks_by_id = {c["document_id"]: c for c in chunks}
    title_to_chunk_ids = _build_title_index(chunks)
    print(f"  unique titles in payloads: {len(title_to_chunk_ids)}")

    print("Building slug fallback index from on-disk JSON...")
    slug_index = _build_slug_index()
    print(f"  indexed {len(slug_index)} slug tokens")
    if not slug_index:
        print(
            "  warning: slug fallback index is empty (data/structured missing?); "
            "entity matching relies on title matching only",
            file=sys.stderr,
        )

    print("Tokenizing corpus for BM25...")
    tokenized_corpus = [_tokenize(c["page_content"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    print("  done")

    for q in questions:
        out_path = args.out / f"{q['id']}.json"
        ranked = build_for_question(
            q,
            client,
            chunks_by_id,
            title_to_chunk_ids,
            slug_index,
            bm25,
            chunks,
            dense_k,
            bm25_k,
            cap=args.cap,
        )
        tmp = out_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(ranked, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, out_path)
        sources_count = sum(1 for c in ranked if len(c["sources"]) >= 2)
        print(
            f"  {q['id']}: {len(ranked)} candidates "
            f"({sources_count} from >=2 sources)"
        )

    print(f"\nWrote {len(questions)} files to {args.out}")


if __name__ == "__main__":
    main()
