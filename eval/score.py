"""Score retrieval functions from src/retriever/querying.py against qrels v1.

Usage:
    # Baseline: the production function at its current k
    python -m eval.score run --fn contextfromvectors --k 5

    # Any evolution = a new function in querying.py, scored directly:
    python -m eval.score run --fn contextfromvectors --k 10
    python -m eval.score run --fn contextfromvectors_hybrid --k 10

    # Compare saved runs
    python -m eval.score score \
        --run eval/v1/runs/contextfromvectors_k5.run.json \
        --run eval/v1/runs/contextfromvectors_hybrid_k10.run.json

Retriever contract: any callable ``(query: str, k: int) -> list`` where each
item is a Document, a (Document, score) pair, or a plain chunk-content
string. Documents carry the qrels id in ``metadata["document_id"]``; bare
strings are resolved via a content->id map (ambiguous content = flagged,
counted, and scored 0). Requires the LLM proxy on localhost:3001.

Deliberately does NOT import src.retriever.indexing (it re-runs indexing at
import time) or generation.py (input() at import time).
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from langchain_core.documents import Document

QRELS_PATH = REPO_ROOT / "eval" / "v1" / "qrels.final.json"
QUESTIONS_PATH = REPO_ROOT / "eval" / "questions.json"
RUNS_DIR = REPO_ROOT / "eval" / "v1" / "runs"

KS = (5, 10, 20)  # score cutoffs; clipped to the run's own k


# --------------------------------------------------------------------------- #
# Calling the production retriever
# --------------------------------------------------------------------------- #

def _load_prod_fn(name: str):
    """Import querying.py (opens the vector store) and grab the function."""
    from src.retriever import querying

    fn = getattr(querying, name, None)
    if not callable(fn):
        sys.exit(f"{name!r} not found in src/retriever/querying.py")
    return fn


def _call_with_k(fn, query: str, k: int):
    """Call fn(query, k) when its signature accepts k, else fn(query)."""
    try:
        params = inspect.signature(fn).parameters
        has_k = "k" in params
    except (TypeError, ValueError):
        has_k = True
    if not has_k:
        return fn(query)
    try:
        return fn(query, k)
    except TypeError:
        return fn(query, k=k)


def _as_doc_and_content(item) -> tuple[Document | None, str]:
    """Accept Document, (Document, score), or a plain content string."""
    if isinstance(item, Document):
        return item, item.page_content
    if (
        isinstance(item, (tuple, list))
        and len(item) == 2
        and hasattr(item[0], "page_content")
    ):
        return item[0], item[0].page_content
    if isinstance(item, str):
        return None, item
    raise TypeError(f"unrecognized retrieval item type: {type(item)!r}")


def _doc_id_of(doc: Document | None) -> str | None:
    if doc is None:
        return None
    return (getattr(doc, "metadata", None) or {}).get("document_id")


def _content_id_map() -> dict[str, list[str]]:
    """page_content -> [document_id, ...] from the same Qdrant collection."""
    from src.config import settings
    from src.retriever import querying

    client = getattr(querying.qdrant, "client", None)
    if client is None:  # fallback; normally the store exposes .client
        from qdrant_client import QdrantClient

        client = QdrantClient(path=settings.QDRANT_PATH)

    collection = getattr(querying, "COLLECTION_NAME", settings.COLLECTION_NAME)
    mapping: dict[str, list[str]] = defaultdict(list)
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=512,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            payload = p.payload or {}
            content = payload.get("page_content", "")
            doc_id = ((payload.get("metadata") or {}).get("document_id")) or str(p.id)
            mapping[content].append(doc_id)
        if offset is None:
            break
    return mapping


def retrieve_prod(fn, questions: list[dict], k: int, strict: bool) -> tuple[dict, dict]:
    run: dict[str, list[str]] = {}
    n_unresolved = 0
    ambiguous: set[str] = set()
    id_map: dict[str, list[str]] | None = None

    for q in questions:
        items = _call_with_k(fn, q["question"], k)
        if len(items) < k:
            print(
                f"  note: {q['id']}: requested k={k} but fn returned "
                f"{len(items)} (hardcoded k?)",
                file=sys.stderr,
            )
        ids: list[str] = []
        for item in items:
            doc, content = _as_doc_and_content(item)
            doc_id = _doc_id_of(doc)
            if doc_id is None:  # bare string: resolve content -> id
                if id_map is None:
                    print(
                        "  building content->id map (fn returns bare strings; "
                        "prefer returning Documents)...",
                        file=sys.stderr,
                    )
                    id_map = _content_id_map()
                cands = id_map.get(content, [])
                doc_id = cands[0] if len(cands) == 1 else f"unresolved:{hash(content)}"
                if len(cands) > 1:
                    ambiguous.add(content[:60])
            if doc_id.startswith("unresolved:"):
                n_unresolved += 1
            ids.append(doc_id)
        run[q["id"]] = ids
        print(f"  {q['id']}: {len(ids)} hits")

    if ambiguous:
        print(
            f"\n  WARNING: {len(ambiguous)} ambiguous contents (same text, "
            f"multiple chunk ids); scored as grade 0:",
            file=sys.stderr,
        )
        for c in sorted(ambiguous)[:5]:
            print(f"    {c!r}", file=sys.stderr)
    if strict and n_unresolved:
        sys.exit(
            f"--strict: {n_unresolved} unresolved retrievals; fix the "
            f"retriever to return Documents with metadata['document_id']"
        )
    return run, {
        "n_unresolved": n_unresolved,
        "ambiguous_contents": sorted(ambiguous)[:10],
    }


# --------------------------------------------------------------------------- #
# Metrics (graded 0..3; unjudged/unresolved docs count as grade 0)
# --------------------------------------------------------------------------- #

def _dcg(gains: list[float]) -> float:
    return sum((2.0**g - 1.0) / math.log2(i + 2) for i, g in enumerate(gains))


def _relevant(judgments: dict[str, int], min_grade: int) -> set[str]:
    return {d for d, g in judgments.items() if g >= min_grade}


def ndcg_at_k(run_ids: list[str], judgments: dict[str, int], k: int) -> float:
    gains = [judgments.get(doc_id, 0) for doc_id in run_ids[:k]]
    ideal = sorted(judgments.values(), reverse=True)[:k]
    denom = _dcg([float(g) for g in ideal])
    return _dcg(gains) / denom if denom > 0 else 0.0


def recall_at_k(run_ids, judgments, k, min_grade) -> float | None:
    rel = _relevant(judgments, min_grade)
    if not rel:
        return None
    return sum(1 for d in run_ids[:k] if d in rel) / len(rel)


def mrr_at_k(run_ids, judgments, k, min_grade) -> float | None:
    rel = _relevant(judgments, min_grade)
    if not rel:
        return None
    for i, d in enumerate(run_ids[:k]):
        if d in rel:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(run_ids, judgments, k, min_grade) -> float | None:
    rel = _relevant(judgments, min_grade)
    if not rel:
        return None
    return sum(1 for d in run_ids[:k] if d in rel) / k


def _metric_keys(ks: list[int]) -> list[str]:
    keys = [f"ndcg@{k}" for k in ks]
    keys += [f"recall@{k}" for k in ks]
    keys += [f"mrr@{k}" for k in ks]
    keys += [f"prec@{k}" for k in ks]
    return keys


def score_run(run: dict, qrels: dict, ks: list[int], min_grade: int) -> dict:
    per_q_by_cat: dict[str, list[dict]] = defaultdict(list)
    for qid in sorted(qrels):
        entry = qrels[qid]
        if qid not in run:
            print(f"  warning: {qid} missing from run (scored as empty)",
                  file=sys.stderr)
        run_ids = run.get(qid, [])
        qm: dict[str, float] = {}
        for k in ks:
            qm[f"ndcg@{k}"] = ndcg_at_k(run_ids, entry["judgments"], k)
        for k in ks:
            for name, fn in (("recall", recall_at_k), ("mrr", mrr_at_k),
                             ("prec", precision_at_k)):
                val = fn(run_ids, entry["judgments"], k, min_grade)
                if val is not None:
                    qm[f"{name}@{k}"] = val
        per_q_by_cat[entry["category"]].append(qm)

    def _aggregate(rows: list[dict]) -> dict:
        out: dict[str, float] = {}
        for key in _metric_keys(ks):
            vals = [r[key] for r in rows if key in r]
            if vals:
                out[key] = sum(vals) / len(vals)
        out["n"] = len(rows)
        return out

    all_rows = [r for rows in per_q_by_cat.values() for r in rows]
    return {"min_grade": min_grade,
            "overall": _aggregate(all_rows),
            "by_category": {c: _aggregate(rs)
                            for c, rs in sorted(per_q_by_cat.items())}}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def _fmt(v, width: int = 8) -> str:
    return f"{v:.4f}".rjust(width) if isinstance(v, float) else "      -  ".rjust(width)


def print_metrics_block(title: str, metrics: dict, ks: list[int],
                        min_grade: int) -> None:
    print(f"\n=== {title} (n={metrics['n']}, binary cutoff: grade >= "
          f"{min_grade}) ===")
    for key in _metric_keys(ks):
        print(f"  {key:12s} {_fmt(metrics.get(key))}")


def print_category_table(metrics: dict, ks: list[int]) -> None:
    print("\n=== Per category ===")
    keys = _metric_keys(ks)
    print(f"  {'category':12s} {'n':>3s} " + " ".join(k.rjust(8) for k in keys))
    for cat, m in metrics["by_category"].items():
        print(f"  {cat:12s} {m['n']:>3d} " + " ".join(_fmt(m.get(k)) for k in keys))


def print_comparison(runs: list[tuple[str, dict]], ks_by_name: list[list[int]]) -> None:
    print("\n=== Comparison (overall) ===")
    rows = _metric_keys(ks_by_name[0])
    name_w = max(max(len(n) for n, _ in runs), 8)
    print(f"  {'metric':12s} " + " ".join(n.rjust(name_w) for n, _ in runs))
    for key in rows:
        cells = [_fmt(m.get(key), name_w) for _, m in runs]
        print(f"  {key:12s} " + " ".join(cells))
    print(f"  {'n':12s} " + " ".join(f"{m['n']}".rjust(name_w) for _, m in runs))


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

def _clip_ks(k: int) -> list[int]:
    return [x for x in dict.fromkeys([*KS, k]) if x <= k]


def cmd_run(args: argparse.Namespace) -> None:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    qrels = json.loads(args.qrels.read_text(encoding="utf-8"))

    print("Loading retriever from src/retriever/querying.py...")
    fn = _load_prod_fn(args.fn)
    print("Warming up embeddings (LLM proxy on :3001)...")
    try:
        _call_with_k(fn, "warmup", args.k)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Retriever warmup failed (is the LLM proxy on "
                 f"localhost:3001 running?): {e}")

    print(f"\nRetrieving for {len(questions)} questions (fn={args.fn}, k={args.k})...")
    run, diag = retrieve_prod(fn, questions, args.k, args.strict)

    ks = _clip_ks(args.k)
    metrics = score_run(run, qrels, ks, args.min_grade)
    print_metrics_block(f"Overall ({args.fn}, k={args.k})", metrics["overall"],
                        ks, args.min_grade)
    print_category_table(metrics, ks)
    if diag["n_unresolved"]:
        print(f"\n  unresolved retrievals scored as grade 0: {diag['n_unresolved']}")

    out = args.out or RUNS_DIR / f"{args.fn}_k{args.k}.run.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "fn": args.fn,
            "k": args.k,
            "min_grade": args.min_grade,
            "qrels": str(args.qrels.relative_to(REPO_ROOT)),
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
        "diagnostics": diag,
        "run": run,
        "metrics": metrics,
    }
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    os.replace(tmp, out)
    print(f"\nWrote run to {out}")


def cmd_score(args: argparse.Namespace) -> None:
    qrels = json.loads(args.qrels.read_text(encoding="utf-8"))
    loaded: list[tuple[str, dict]] = []
    ks_by_name: list[list[int]] = []
    for raw in args.run:
        p = Path(raw)
        data = json.loads(p.read_text(encoding="utf-8"))
        cfg = data.get("config", {})
        k = cfg.get("k") or max(len(v) for v in data["run"].values())
        min_grade = (args.min_grade if args.min_grade is not None
                     else cfg.get("min_grade", 2))
        ks = _clip_ks(k)
        metrics = score_run(data["run"], qrels, ks, min_grade)
        name = p.name.removesuffix(".run.json")
        loaded.append((name, metrics[args.category] if args.category
                       else metrics["overall"]))
        ks_by_name.append(ks)
    if len({tuple(ks) for ks in ks_by_name}) > 1:
        print("  note: runs retrieved at different k; compare only shared "
              "cutoffs", file=sys.stderr)
    print_comparison(loaded, ks_by_name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run a querying.py function + score it")
    p_run.add_argument("--fn", default="contextfromvectors",
                       help="function name in src/retriever/querying.py")
    p_run.add_argument("--k", type=int, default=10)
    p_run.add_argument("--min-grade", type=int, default=2,
                       help="grade threshold for binary metrics (default 2)")
    p_run.add_argument("--strict", action="store_true",
                       help="abort if any retrieval cannot be resolved to a chunk id")
    p_run.add_argument("--qrels", type=Path, default=QRELS_PATH)
    p_run.add_argument("--out", type=Path, default=None)
    p_run.set_defaults(func=cmd_run)

    p_score = sub.add_parser("score", help="score / compare saved run files")
    p_score.add_argument("--run", type=Path, action="append", required=True)
    p_score.add_argument("--category", default=None)
    p_score.add_argument("--min-grade", type=int, default=None)
    p_score.add_argument("--qrels", type=Path, default=QRELS_PATH)
    p_score.set_defaults(func=cmd_score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
