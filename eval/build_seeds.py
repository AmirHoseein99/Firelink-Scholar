"""Build the v3 evaluation seeds: link graph, per-question seed pools, anchors.

Run:
    python -m eval.build_seeds seeds --only q002,q005,q013,q023,q025,q039
    python -m eval.build_seeds seeds                      # all 50 questions
    python -m eval.build_seeds build-anchors --only q002,...
    python -m eval.build_seeds seeds --rebuild-index      # force link rebuild

``seeds`` writes one pool file per question to ``eval/v3/candidates/qNNN.json``.
Pool order (deduped, first source wins, cap applied last with human ids always
kept):

    1. human   — ids already judged in eval/v3/seeds/human-judgments.json
                 (imported verbatim, never re-graded)
    2. template— pages matched via expected entities against page titles and
                 the link graph (infobox/template-parameter wikilinks from
                 data/processed wikitext + lead-section links from
                 data/cleaned markdown; denylist/navbox templates excluded)
    3. gold    — the full page of every human-judged chunk with grade >= 2
    4. dense   — dense retrieval hits
    5. bm25    — BM25 hits

Entities shorter than 5 characters (e.g. "Dark") match exactly only; longer
entities match exactly or when all their tokens appear as whole words in the
target — so a location=[[Dark Souls]] infobox line cannot flood a pool. The
template/link-graph tier is itself capped at 80 ids per question (the junk
cap checked by the pilot gate), ordered best-match pages first.

``build-anchors`` writes ``eval/v3/seeds/anchors.json``: per question the gold
anchor (human grade 3, lowest doc_id) and contrast anchor (human grade 0,
lowest doc_id) with 700-char Qdrant snippets, plus the probe holdout (every
5th human-judged id, sorted by doc_id). Anchors are excluded from LLM
grading; probes are graded by both sides to measure agreement.

The single Qdrant scroll in ``seeds``/``build-anchors`` also validates that
imported human ids still exist in the collection.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import mwparserfromhell
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.cleaner.registry import Registry
from src.config import settings

COLLECTION_NAME = "darksoulswiki"
QDRANT_PATH = str(REPO_ROOT / "data" / "qdrant")
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
CLEANED_DIR = REPO_ROOT / "data" / "cleaned"

SNIPPET_LEN = 1000
ANCHOR_SNIPPET_LEN = 700
SHORT_ENTITY_LEN = 5  # entities shorter than this match exactly only
GRADE_GOLD_PAGE = 2
TEMPLATE_JUNK_CAP = 80  # max template/link-graph ids added to a pool
BM25_TOKEN_RE = re.compile(r"[a-z0-9]+")
WIKI_MARKUP_RE = re.compile(
    r"\[\[[^\]]*\]\]|\{\{[^{}]*\}\}|'''?|''|`+|^=+\s*=*$|={2,}",
    flags=re.MULTILINE,
)
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
REF_RE = re.compile(r"<ref\b[^>]*/>|<ref\b[^>]*>.*?</ref>", flags=re.DOTALL)
EXCLUDED_TARGET_RE = re.compile(
    r"^(:?\s*(category|file|image|wikipedia|help|template|mediawiki|portal|"
    r"special|talk|user)?:|https?://|#[^|]*)",
    flags=re.IGNORECASE,
)
NON_GAME_DIRS = {"reports", "tables", "uncategorized"}


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def normalize_title(s: str) -> str:
    s = s.replace("_", " ").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _strip_wiki(s: str) -> str:
    return WIKI_MARKUP_RE.sub(" ", s)


def _tokenize(text: str) -> list[str]:
    return BM25_TOKEN_RE.findall(_strip_wiki(text).lower())


def match_rank(entity: str, target: str) -> int | None:
    """Match quality of an expected entity against a title/link target.

    0 = exact (normalized) match, 1 = every entity token appears as a whole
    word in the target. Entities shorter than 5 chars (e.g. "Dark") only ever
    match exactly, so a location=[[Dark Souls]] infobox line cannot flood a
    pool via the "dark" token. None = no match.
    """
    e, t = normalize_title(entity), normalize_title(target)
    if not e or not t:
        return None
    if e == t:
        return 0
    if len(e) < SHORT_ENTITY_LEN:
        return None
    if set(e.split()) <= set(t.split()):
        return 1
    return None


# --------------------------------------------------------------------------- #
# Link graph
# --------------------------------------------------------------------------- #

def extract_template_links(wikitext: str, registry: Registry) -> list[str]:
    """Wikilink targets inside non-denylisted template parameter values."""
    text = wikitext.replace("{{!}}", "|")
    text = REF_RE.sub(" ", text)
    targets: list[str] = []
    for tpl in mwparserfromhell.parse(text).filter_templates():
        if registry.is_denylist_template(str(tpl.name).strip()):
            continue
        for param in tpl.params:
            for link in mwparserfromhell.parse(str(param.value)).filter_wikilinks():
                target = str(link.title).strip()
                if target and not EXCLUDED_TARGET_RE.match(target):
                    targets.append(target.split("#")[0].strip() or target)
    return targets


def extract_lead_links(markdown: str) -> list[str]:
    """Wikilink targets from a cleaned lead section (md links + raw wikilinks)."""
    targets: list[str] = []
    for _, raw in MD_LINK_RE.findall(markdown):
        raw = raw.strip()
        if raw and not EXCLUDED_TARGET_RE.match(raw):
            targets.append(raw.split("#")[0].strip() or raw)
    for link in mwparserfromhell.parse(markdown).filter_wikilinks():
        target = str(link.title).strip()
        if target and not EXCLUDED_TARGET_RE.match(target):
            targets.append(target.split("#")[0].strip() or target)
    return targets


def build_link_index(registry: Registry) -> dict:
    """Scan data/processed + data/cleaned; entity -> pages linking to it."""
    by_target: dict[str, set[str]] = {}
    by_page: dict[str, set[str]] = {}

    def _add(page_title: str, targets: list[str]) -> None:
        if not page_title:
            return
        page = by_page.setdefault(page_title, set())
        for t in targets:
            nt = normalize_title(t)
            if not nt:
                continue
            page.add(nt)
            by_target.setdefault(nt, set()).add(page_title)

    for game_dir in sorted(p for p in PROCESSED_DIR.iterdir() if p.is_dir()):
        if game_dir.name in NON_GAME_DIRS:
            continue
        for jf in sorted(game_dir.rglob("*.json")):
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            for page in (data.get("query") or {}).get("pages", {}).values():
                try:
                    wikitext = page["revisions"][0]["slots"]["main"]["*"]
                except (KeyError, TypeError, IndexError):
                    continue
                _add(page.get("title", ""),
                     extract_template_links(wikitext, registry))

    if CLEANED_DIR.exists():
        for jf in sorted(CLEANED_DIR.rglob("*.json")):
            if any(part in NON_GAME_DIRS for part in jf.parts):
                continue
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            title = (data.get("source") or {}).get("title", "")
            sections = data.get("sections") or []
            lead = sections[0].get("markdown", "") if sections else ""
            _add(title, extract_lead_links(lead))

    return {
        "version": 1,
        "built_at": now_iso(),
        "n_pages": len(by_page),
        "by_target": {k: sorted(v) for k, v in sorted(by_target.items())},
        "by_page": {k: sorted(v) for k, v in sorted(by_page.items())},
    }


def load_or_build_link_index(path: Path, registry: Registry,
                             rebuild: bool = False) -> dict:
    if path.exists() and not rebuild:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"  warning: unparsable {path}, rebuilding", file=sys.stderr)
    index = build_link_index(registry)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
    print(f"  built link index: {index['n_pages']} pages, "
          f"{len(index['by_target'])} distinct targets -> {path}")
    return index


def matched_pages_for_entities(
    entities: list[str],
    link_index: dict,
    titles_by_norm: dict[str, list[str]],
) -> dict[str, int]:
    """Pages matched to expected entities via titles and the link graph,
    with the best (lowest) match rank per page."""
    pages: dict[str, int] = {}
    for entity in entities:
        for title_norm, titles in titles_by_norm.items():
            rank = match_rank(entity, title_norm)
            if rank is not None:
                for title in titles:
                    pages[title] = min(pages.get(title, 9), rank)
        for target, linked_pages in link_index.get("by_target", {}).items():
            rank = match_rank(entity, target)
            if rank is not None:
                for title in linked_pages:
                    pages[title] = min(pages.get(title, 9), rank)
    return pages


# --------------------------------------------------------------------------- #
# Qdrant scroll + retrieval sources
# --------------------------------------------------------------------------- #

def scroll_all_chunks(client: QdrantClient) -> list[dict]:
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


def title_indexes(chunks: list[dict]) -> tuple[dict[str, list[str]],
                                               dict[str, list[str]]]:
    title_to_ids: dict[str, list[str]] = {}
    norm_to_titles: dict[str, list[str]] = {}
    for c in chunks:
        title_to_ids.setdefault(c["title"], []).append(c["document_id"])
        norm_to_titles.setdefault(normalize_title(c["title"]), []).append(
            c["title"]
        )
    return title_to_ids, {k: sorted(set(v)) for k, v in norm_to_titles.items()}


def dense_hits(q: dict, client: QdrantClient, k: int) -> list[tuple[str, float]]:
    from langchain_openai import OpenAIEmbeddings

    embed = OpenAIEmbeddings(
        model="qwen3-embedding-0.6b",
        base_url="http://localhost:3001/v1",
        api_key=settings.freellm_api_key,
        check_embedding_ctx_length=False,
    )
    query_vec = embed.embed_query(q["question"])
    raw = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=k,
        with_payload=True,
    ).points
    hits = []
    for hit in raw:
        doc_id = ((hit.payload or {}).get("metadata") or {}).get("document_id")
        if doc_id:
            hits.append((doc_id, float(hit.score)))
    return hits


def bm25_hits(q: dict, bm25: BM25Okapi, chunks: list[dict],
              k: int) -> list[tuple[str, float]]:
    tokens = _tokenize(q["question"])
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)
    top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return [(chunks[i]["document_id"], scores[i])
            for i in top if scores[i] > 0]


# --------------------------------------------------------------------------- #
# Seed pool
# --------------------------------------------------------------------------- #

def build_pool(
    q: dict,
    human_judgments: dict[str, int],
    chunks_by_id: dict[str, dict],
    title_to_ids: dict[str, list[str]],
    titles_by_norm: dict[str, list[str]],
    link_index: dict,
    client: QdrantClient,
    bm25: BM25Okapi,
    chunks: list[dict],
    dense_k: int,
    bm25_k: int,
    cap: int,
) -> tuple[list[dict], dict[str, int]]:
    candidates: dict[str, dict] = {}
    order: list[str] = []

    def _add(doc_id: str, source: str, *, dense=None, bm25=None) -> bool:
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
                "dense_score": dense,
                "bm25_score": bm25,
            }
        elif source not in existing["sources"]:
            existing["sources"].append(source)
        if dense is not None and existing["dense_score"] is None:
            existing["dense_score"] = dense
        if bm25 is not None and existing["bm25_score"] is None:
            existing["bm25_score"] = bm25
        return True

    counts = {"human": 0, "template": 0, "gold": 0, "dense": 0, "bm25": 0}

    human_ids = sorted(d for d, g in human_judgments.items() if d in chunks_by_id)
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
    template_added = 0
    template_truncated = 0
    for page in sorted(template_pages, key=lambda p: (template_pages[p], p)):
        for doc_id in sorted(title_to_ids.get(page, [])):
            if template_added >= TEMPLATE_JUNK_CAP:
                template_truncated += 1
                continue
            if _add(doc_id, "template"):
                template_added += 1
                counts["template"] += 1
    if template_truncated:
        print(f"    note: template tier truncated at the {TEMPLATE_JUNK_CAP}-id "
              f"junk cap ({template_truncated} id(s) skipped)", file=sys.stderr)

    gold_pages = sorted(
        {
            chunks_by_id[d]["title"]
            for d, g in human_judgments.items()
            if g >= GRADE_GOLD_PAGE and d in chunks_by_id
        }
    )
    for page in gold_pages:
        for doc_id in sorted(title_to_ids.get(page, [])):
            if _add(doc_id, "gold"):
                counts["gold"] += 1

    try:
        dense = dense_hits(q, client, dense_k)
    except Exception as e:  # noqa: BLE001 - dense failure is non-fatal
        print(f"    dense failed for {q['id']}: {e}", file=sys.stderr)
        dense = []
    dense_ids = {doc_id for doc_id, _ in dense}
    for doc_id, score in dense:
        if _add(doc_id, "dense", dense=score):
            counts["dense"] += 1

    for doc_id, score in bm25_hits(q, bm25, chunks, bm25_k):
        if doc_id in dense_ids:
            continue
        if _add(doc_id, "bm25", bm25=score):
            counts["bm25"] += 1

    kept = [d for d in order if "human" in candidates[d]["sources"]]
    for d in order:
        if len(kept) >= cap:
            break
        if d not in kept:
            kept.append(d)
    return [candidates[d] for d in kept], counts


# --------------------------------------------------------------------------- #
# Anchors + probes
# --------------------------------------------------------------------------- #

def build_anchors_for_question(
    q: dict,
    human_judgments: dict[str, int],
    chunks_by_id: dict[str, dict],
) -> dict | None:
    judged = {d: g for d, g in human_judgments.items() if d in chunks_by_id}
    if not judged:
        return None
    sorted_ids = sorted(judged)

    def _anchor(grade: int) -> dict | None:
        for doc_id in sorted_ids:
            if judged[doc_id] == grade:
                meta = chunks_by_id[doc_id]
                return {
                    "doc_id": doc_id,
                    "title": meta["title"],
                    "section": meta["section"],
                    "snippet": meta["page_content"][:ANCHOR_SNIPPET_LEN],
                }
        return None

    return {
        "gold": _anchor(3),
        "contrast": _anchor(0),
        "probes": sorted_ids[4::5],
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _parse_only(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    return {part.strip() for part in raw.split(",") if part.strip()}


def _load_questions(path: Path, only: set[str] | None) -> list[dict]:
    questions = json.loads(path.read_text(encoding="utf-8"))
    if only is not None:
        unknown = only - {q["id"] for q in questions}
        if unknown:
            sys.exit(f"--only references unknown question ids: {sorted(unknown)}")
        questions = [q for q in questions if q["id"] in only]
    return questions


def _load_human_seeds(path: Path) -> dict[str, dict[str, int]]:
    seeds = json.loads(path.read_text(encoding="utf-8"))
    return {qid: entry["judgments"] for qid, entry in seeds.items()}


def cmd_seeds(args: argparse.Namespace) -> None:
    if not Path(QDRANT_PATH).exists():
        sys.exit(f"Qdrant storage not found at {QDRANT_PATH}; "
                 f"run the indexing pipeline first")
    only = _parse_only(args.only)
    questions = _load_questions(args.questions, only)
    human = _load_human_seeds(args.human_seeds)
    registry = Registry.default()

    print("Loading link graph index...")
    link_index = load_or_build_link_index(args.index, registry,
                                          rebuild=args.rebuild_index)

    print("Scrolling Qdrant collection...")
    client = QdrantClient(path=QDRANT_PATH)
    try:
        chunks = scroll_all_chunks(client)
        chunks_by_id = {c["document_id"]: c for c in chunks}
        title_to_ids, titles_by_norm = title_indexes(chunks)
        print(f"  {len(chunks)} chunks, {len(title_to_ids)} titles")

        print("Tokenizing corpus for BM25...")
        bm25 = BM25Okapi([_tokenize(c["page_content"]) for c in chunks])
        print("  done")

        args.out.mkdir(parents=True, exist_ok=True)
        for q in questions:
            human_judgments = human.get(q["id"], {})
            pool, counts = build_pool(
                q, human_judgments, chunks_by_id, title_to_ids, titles_by_norm,
                link_index, client, bm25, chunks, args.dense_k, args.bm25_k,
                args.cap,
            )
            out_path = args.out / f"{q['id']}.json"
            tmp = out_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(pool, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            os.replace(tmp, out_path)
            print(
                f"  {q['id']}: pool={len(pool)} "
                f"(human={counts['human']} template={counts['template']} "
                f"gold={counts['gold']} dense={counts['dense']} "
                f"bm25={counts['bm25']}) -> {out_path.name}"
            )
        print(f"\nWrote {len(questions)} pool file(s) to {args.out}")
    finally:
        client.close()


def cmd_build_anchors(args: argparse.Namespace) -> None:
    if not Path(QDRANT_PATH).exists():
        sys.exit(f"Qdrant storage not found at {QDRANT_PATH}; "
                 f"run the indexing pipeline first")
    only = _parse_only(args.only)
    questions = _load_questions(args.questions, only)
    human = _load_human_seeds(args.human_seeds)

    print("Scrolling Qdrant collection...")
    client = QdrantClient(path=QDRANT_PATH)
    try:
        chunks = scroll_all_chunks(client)
    finally:
        client.close()
    chunks_by_id = {c["document_id"]: c for c in chunks}
    print(f"  {len(chunks)} chunks")

    anchors: dict[str, dict] = {}
    for q in questions:
        if q["id"] not in human:
            print(f"  {q['id']}: no human seeds, skipped")
            continue
        anchor = build_anchors_for_question(q, human[q["id"]], chunks_by_id)
        if anchor is None:
            print(f"  {q['id']}: no judgable human ids, skipped")
            continue
        anchors[q["id"]] = anchor
        gold = anchor["gold"]["doc_id"] if anchor["gold"] else None
        contrast = anchor["contrast"]["doc_id"] if anchor["contrast"] else None
        if gold is None or contrast is None:
            print(f"  {q['id']}: WARNING missing "
                  f"{'gold' if gold is None else 'contrast'} anchor")
        print(f"  {q['id']}: gold={gold} contrast={contrast} "
              f"probes={len(anchor['probes'])}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(anchors, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    os.replace(tmp, args.out)
    print(f"\nWrote anchors for {len(anchors)} question(s) to {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p_seeds = sub.add_parser("seeds", help="build per-question seed pools")
    p_seeds.add_argument("--questions", type=Path,
                         default=REPO_ROOT / "eval" / "questions.json")
    p_seeds.add_argument("--human-seeds", type=Path,
                         default=REPO_ROOT / "eval" / "v3" / "seeds"
                         / "human-judgments.json")
    p_seeds.add_argument("--index", type=Path,
                         default=REPO_ROOT / "eval" / "v3" / "seeds"
                         / "link-index.json")
    p_seeds.add_argument("--out", type=Path,
                         default=REPO_ROOT / "eval" / "v3" / "candidates")
    p_seeds.add_argument("--only", type=str, default=None,
                         help="comma-separated question ids (e.g. q002,q005)")
    p_seeds.add_argument("--dense-k", type=int, default=50)
    p_seeds.add_argument("--bm25-k", type=int, default=50)
    p_seeds.add_argument("--cap", type=int, default=200)
    p_seeds.add_argument("--rebuild-index", action="store_true",
                         help="rebuild link-index.json even if it exists")
    p_seeds.set_defaults(func=cmd_seeds)

    p_anchors = sub.add_parser("build-anchors",
                               help="build gold/contrast anchors + probes")
    p_anchors.add_argument("--questions", type=Path,
                           default=REPO_ROOT / "eval" / "questions.json")
    p_anchors.add_argument("--human-seeds", type=Path,
                           default=REPO_ROOT / "eval" / "v3" / "seeds"
                           / "human-judgments.json")
    p_anchors.add_argument("--out", type=Path,
                           default=REPO_ROOT / "eval" / "v3" / "seeds"
                           / "anchors.json")
    p_anchors.add_argument("--only", type=str, default=None,
                           help="comma-separated question ids (e.g. q002,q005)")
    p_anchors.set_defaults(func=cmd_build_anchors)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
