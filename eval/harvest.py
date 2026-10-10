"""Harvest a retrieval run file: extend seed pools with unjudged ids.

Run:
    python -m eval.harvest --run eval/v3/runs/contextfromvectors_k10.run.json

This is the growth step of the run-based judgment cycle. When a retriever
change surfaces chunks the judgment set has never seen, ``eval.score`` trips
its unjudged counter; ``harvest`` appends exactly those ids (pulled from the
Qdrant payload, tagged ``sources: ["run"]``) to the per-question pool files.
The next ``python -m eval.llm_grade`` then delta-grades only the new ids
(id-level resume) and merges them into ``eval/v3/judgments.json``. Human
grades are never overwritten.

Idempotent: ids already in a pool or already judged are skipped, so
harvesting the same run twice is a no-op.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from qdrant_client import QdrantClient

from eval.build_seeds import SNIPPET_LEN, scroll_all_chunks

COLLECTION_NAME = "darksoulswiki"
QDRANT_PATH = str(REPO_ROOT / "data" / "qdrant")


def new_ids_for_question(
    run_ids: list[str],
    judged: dict[str, int],
    existing_pool_ids: set[str],
) -> list[str]:
    """Run ids that are neither judged nor already in the pool."""
    return [
        doc_id for doc_id in run_ids
        if not doc_id.startswith("unresolved:")
        and doc_id not in judged
        and doc_id not in existing_pool_ids
    ]


def candidate_from_chunk(doc_id: str, chunk: dict) -> dict:
    return {
        "document_id": doc_id,
        "title": chunk["title"],
        "section": chunk["section"],
        "game": chunk["game"],
        "entity_type": chunk["entity_type"],
        "snippet": chunk["page_content"][:SNIPPET_LEN],
        "sources": ["run"],
        "dense_score": None,
        "bm25_score": None,
    }


def harvest_run(
    run: dict[str, list[str]],
    judged_by_q: dict[str, dict[str, int]],
    chunks_by_id: dict[str, dict],
    candidates_dir: Path,
) -> dict[str, dict]:
    """Append unjudged run ids to the per-question pools.

    Returns {qid: {"n_new": int, "n_missing": int}} for questions with
    anything new.
    """
    summary: dict[str, dict] = {}
    for qid, run_ids in sorted(run.items()):
        pool_path = candidates_dir / f"{qid}.json"
        pool: list[dict] = []
        if pool_path.exists():
            pool = json.loads(pool_path.read_text(encoding="utf-8"))
        existing_ids = {c["document_id"] for c in pool}
        judged = judged_by_q.get(qid, {})
        fresh = new_ids_for_question(run_ids, judged, existing_ids)
        if not fresh:
            continue
        added = missing = 0
        for doc_id in fresh:
            chunk = chunks_by_id.get(doc_id)
            if chunk is None:
                missing += 1
                print(f"  warning: {qid}: run id {doc_id} not in Qdrant, "
                      f"skipped", file=sys.stderr)
                continue
            pool.append(candidate_from_chunk(doc_id, chunk))
            added += 1
        if added:
            tmp = pool_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(pool, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            os.replace(tmp, pool_path)
            summary[qid] = {"n_new": added, "n_missing": missing}
            print(f"  {qid}: +{added} new id(s) appended to "
                  f"{pool_path.name} ({len(pool)} total)"
                  + (f", {missing} missing from Qdrant" if missing else ""))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True,
                    help="run file written by eval.score (or eval.pilot)")
    ap.add_argument("--judgments", type=Path,
                    default=REPO_ROOT / "eval" / "v3" / "judgments.json")
    ap.add_argument("--candidates", type=Path,
                    default=REPO_ROOT / "eval" / "v3" / "candidates")
    args = ap.parse_args()

    if not Path(QDRANT_PATH).exists():
        sys.exit(f"Qdrant storage not found at {QDRANT_PATH}; "
                 f"run the indexing pipeline first")

    data = json.loads(args.run.read_text(encoding="utf-8"))
    run = data.get("run", data)
    judged_by_q = {
        qid: entry.get("judgments", {})
        for qid, entry in json.loads(
            args.judgments.read_text(encoding="utf-8")).items()
    } if args.judgments.exists() else {}

    print("Scrolling Qdrant collection...")
    client = QdrantClient(path=QDRANT_PATH)
    chunks_by_id = {
        c["document_id"]: c for c in scroll_all_chunks(client)
    }
    print(f"  {len(chunks_by_id)} chunks")

    args.candidates.mkdir(parents=True, exist_ok=True)
    summary = harvest_run(run, judged_by_q, chunks_by_id, args.candidates)
    total = sum(s["n_new"] for s in summary.values())
    if not summary:
        print("Nothing to harvest: every retrieved id is already judged "
              "or already in a pool.")
    else:
        print(f"\nHarvested {total} new id(s) across {len(summary)} "
              f"question(s). Run `python -m eval.llm_grade` to delta-grade "
              f"them, then re-score.")


if __name__ == "__main__":
    main()
