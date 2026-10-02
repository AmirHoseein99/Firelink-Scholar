"""Validate and finalize the graded qrels.

Run:
    python -m eval.aggregate --qrels eval/v1/qrels.json \
        --out eval/v1/qrels.final.json

- Verifies every ``document_id`` exists in Qdrant (likely typos otherwise).
- Prints per-question grade histograms and coverage warnings.
- Writes a sorted copy with judgments reordered (grade desc, then doc_id).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from qdrant_client import QdrantClient

COLLECTION_NAME = "darksoulswiki"
QDRANT_PATH = str(REPO_ROOT / "data" / "qdrant")


def _all_doc_ids(client: QdrantClient) -> set[str]:
    ids: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1024,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        ids.update(str(p.id) for p in points)
        if offset is None:
            break
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--qrels",
        type=Path,
        default=REPO_ROOT / "eval" / "v1" / "qrels.json",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "eval" / "v1" / "qrels.final.json",
    )
    args = ap.parse_args()

    qrels = json.loads(args.qrels.read_text(encoding="utf-8"))
    print(f"Loaded {len(qrels)} graded questions")

    print(f"Indexing Qdrant doc ids at {QDRANT_PATH}...")
    client = QdrantClient(path=QDRANT_PATH)
    known_ids = _all_doc_ids(client)
    print(f"  {len(known_ids)} chunks in collection")

    final: dict = {}
    needs_regrade: list[str] = []
    empty: list[str] = []

    for qid in sorted(qrels):
        entry = qrels[qid]
        judgments = entry.get("judgments", {})
        hist = Counter(judgments.values())
        unknown = [d for d in judgments if d not in known_ids]
        if unknown:
            print(f"  {qid}: {len(unknown)} unknown doc_ids: {unknown[:5]}")
        if not judgments:
            empty.append(qid)
        elif hist.get(3, 0) == 0:
            needs_regrade.append(qid)

        ordered = dict(
            sorted(
                judgments.items(),
                key=lambda kv: (-kv[1], kv[0]),
            )
        )
        final[qid] = {
            **entry,
            "judgments": ordered,
            "grade_counts": {
                str(g): hist.get(g, 0) for g in (0, 1, 2, 3)
            },
            "n_judgments": len(judgments),
            "n_unknown_doc_ids": len(unknown),
        }

    print("\n=== Coverage ===")
    print(f"  questions with no judgments:    {len(empty)}  {empty}")
    print(f"  questions without a grade-3 hit: {len(needs_regrade)}  {needs_regrade}")

    print("\n=== Per-question summary ===")
    for qid in sorted(final):
        e = final[qid]
        gc = e["grade_counts"]
        print(
            f"  {qid}  {e['category']:11s}  "
            f"3={gc['3']} 2={gc['2']} 1={gc['1']} 0={gc['0']}  "
            f"n={e['n_judgments']}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(final, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote final qrels to {args.out}")


if __name__ == "__main__":
    main()
