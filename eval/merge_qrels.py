"""Merge LLM-graded v2 qrels with human v1 ground truth and validate.

Run:
    python -m eval.merge_qrels

- Starts from the LLM judgments (``eval/v2/llm-qrels.json``) and overlays the
  human grades from ``eval/v1/qrels.final.json`` (human wins, including
  manual `a`-added chunk ids that are outside any v2 pool).
- Questions present only in one input are kept, so all 50 questions survive.
- Validates every judged id against the Qdrant collection and reports
  unknowns, per-question coverage vs the v2 pool size, and grade histograms.
- Computes an LLM-vs-human agreement report on overlapping pairs (exact
  match, binary at grade >= 2, 4x4 confusion matrix) and writes it to
  ``eval/v2/agreement.json``. Exact agreement < 0.5 prints a prominent
  warning (soft gate — human grades override anyway).
- Writes ``eval/v2/qrels.json`` in the exact ``qrels.final.json`` schema plus
  provenance (``model``, ``source_counts``), so ``eval/score.py --qrels``
  works unchanged.
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


def _all_doc_ids(client: QdrantClient, collection: str) -> set[str]:
    ids: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=1024,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        ids.update(str(p.id) for p in points)
        if offset is None:
            break
    return ids


def unknown_ids(judgments: dict[str, int], known_ids: set[str]) -> list[str]:
    """Doc ids judged but absent from the Qdrant collection."""
    return [d for d in judgments if d not in known_ids]


def merge_judgments(
    llm_j: dict[str, int],
    human_j: dict[str, int],
) -> tuple[dict[str, int], dict[str, int]]:
    """Overlay human judgments on LLM ones (human wins).

    Returns (merged judgments, source counts {llm, human, human_override}).
    """
    merged = dict(llm_j)
    counts = {"llm": 0, "human": 0, "human_override": 0}
    for doc_id in set(llm_j) | set(human_j):
        if doc_id in human_j:
            merged[doc_id] = human_j[doc_id]
            counts["human_override" if doc_id in llm_j else "human"] += 1
        else:
            counts["llm"] += 1
    return merged, counts


def compute_agreement(
    llm_j: dict[str, int],
    human_j: dict[str, int],
) -> dict:
    """Agreement of LLM vs human grades on the overlapping doc ids."""
    common = sorted(set(llm_j) & set(human_j))
    n = len(common)
    exact = sum(1 for d in common if llm_j[d] == human_j[d])
    binary = sum(
        1 for d in common if (llm_j[d] >= 2) == (human_j[d] >= 2)
    )
    matrix = [[0] * 4 for _ in range(4)]  # rows: human 0..3, cols: llm 0..3
    for d in common:
        matrix[human_j[d]][llm_j[d]] += 1
    return {
        "n_common": n,
        "exact_match": exact / n if n else None,
        "binary_at_2": binary / n if n else None,
        "confusion_matrix": {
            "rows": "human",
            "cols": "llm",
            "matrix": matrix,
        },
    }


def aggregate_agreement(
    llm_qrels: dict,
    human_qrels: dict,
) -> dict:
    """Overall + per-question agreement across all questions."""
    per_question: dict[str, dict] = {}
    total = {"n_common": 0, "exact": 0, "binary": 0}
    matrix_sum = [[0] * 4 for _ in range(4)]
    for qid in sorted(set(llm_qrels) & set(human_qrels)):
        rep = compute_agreement(
            llm_qrels[qid].get("judgments", {}),
            human_qrels[qid].get("judgments", {}),
        )
        if rep["n_common"]:
            per_question[qid] = rep
            total["n_common"] += rep["n_common"]
            rows = rep["confusion_matrix"]["matrix"]
            for h in range(4):
                for l in range(4):
                    matrix_sum[h][l] += rows[h][l]
            total["exact"] = sum(matrix_sum[h][h] for h in range(4))
            total["binary"] = sum(
                matrix_sum[h][l]
                for h in range(4)
                for l in range(4)
                if (h >= 2) == (l >= 2)
            )
    overall = {
        "n_common": total["n_common"],
        "exact_match": (
            total["exact"] / total["n_common"] if total["n_common"] else None
        ),
        "binary_at_2": (
            total["binary"] / total["n_common"] if total["n_common"] else None
        ),
        "confusion_matrix": {
            "rows": "human",
            "cols": "llm",
            "matrix": matrix_sum,
        },
    }
    return {"overall": overall, "per_question": per_question}


def build_final_entry(
    qid: str,
    merged: dict[str, int],
    counts: dict[str, int],
    meta: dict,
    model: str | None,
    n_unknown: int,
) -> dict:
    hist = Counter(merged.values())
    ordered = dict(sorted(merged.items(), key=lambda kv: (-kv[1], kv[0])))
    return {
        "question": meta.get("question", ""),
        "game": meta.get("game", "Dark Souls"),
        "category": meta.get("category", ""),
        "graded_at": meta.get("graded_at", ""),
        "model": model,
        "judgments": ordered,
        "grade_counts": {str(g): hist.get(g, 0) for g in (0, 1, 2, 3)},
        "n_judgments": len(merged),
        "n_unknown_doc_ids": n_unknown,
        "source_counts": counts,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--llm", type=Path,
                    default=REPO_ROOT / "eval" / "v2" / "llm-qrels.json")
    ap.add_argument("--human", type=Path,
                    default=REPO_ROOT / "eval" / "v1" / "qrels.final.json")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "eval" / "v2" / "qrels.json")
    ap.add_argument("--agreement-out", type=Path,
                    default=REPO_ROOT / "eval" / "v2" / "agreement.json")
    ap.add_argument("--candidates", type=Path,
                    default=REPO_ROOT / "eval" / "v2" / "candidates",
                    help="v2 pool dir used for the coverage check")
    ap.add_argument("--collection", type=str, default=COLLECTION_NAME)
    args = ap.parse_args()

    llm_qrels = json.loads(args.llm.read_text(encoding="utf-8"))
    human_qrels = json.loads(args.human.read_text(encoding="utf-8"))
    print(f"Loaded {len(llm_qrels)} LLM-graded, "
          f"{len(human_qrels)} human-graded questions")

    print(f"Indexing Qdrant doc ids at {QDRANT_PATH}...")
    client = QdrantClient(path=QDRANT_PATH)
    known_ids = _all_doc_ids(client, args.collection)
    print(f"  {len(known_ids)} chunks in collection {args.collection!r}")

    all_qids = sorted(set(llm_qrels) | set(human_qrels))
    final: dict = {}
    missing_meta: list[str] = []
    coverage_warnings: list[str] = []
    n_unknown_total = 0

    for qid in all_qids:
        llm_entry = llm_qrels.get(qid, {})
        human_entry = human_qrels.get(qid, {})
        llm_j = llm_entry.get("judgments", {})
        human_j = human_entry.get("judgments", {})
        merged, counts = merge_judgments(llm_j, human_j)

        # human entry is the ground truth for metadata; fall back to llm
        meta = dict(human_entry or llm_entry)
        if not meta.get("question"):
            missing_meta.append(qid)
        model = llm_entry.get("model") if llm_entry else None

        unknown = unknown_ids(merged, known_ids)
        n_unknown_total += len(unknown)
        if unknown:
            print(f"  {qid}: {len(unknown)} unknown doc_ids: {unknown[:5]}")

        # coverage check: merged judgments should be pool size + human-only
        pool_path = args.candidates / f"{qid}.json"
        if args.candidates.exists() and pool_path.exists():
            pool = json.loads(pool_path.read_text(encoding="utf-8"))
            pool_ids = {c["document_id"] for c in pool}
            expected = len(pool_ids) + sum(
                1 for d in human_j if d not in pool_ids
            )
            if len(merged) != expected:
                coverage_warnings.append(
                    f"{qid}: {len(merged)} judgments, expected {expected} "
                    f"(pool {len(pool_ids)} + human-only); failed batches?"
                )
        elif not args.candidates.exists():
            if not coverage_warnings:
                coverage_warnings.append(
                    f"candidates dir {args.candidates} missing; "
                    f"coverage check skipped"
                )

        final[qid] = build_final_entry(
            qid, merged, counts, meta, model, len(unknown)
        )

    agreement = aggregate_agreement(llm_qrels, human_qrels)
    args.agreement_out.parent.mkdir(parents=True, exist_ok=True)
    args.agreement_out.write_text(
        json.dumps(agreement, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== Agreement (LLM vs human, overlapping pairs) ===")
    ov = agreement["overall"]
    print(f"  overlapping judgments: {ov['n_common']}")
    if ov["exact_match"] is None:
        print("  no overlap — nothing to compare")
    else:
        print(f"  exact match rate:    {ov['exact_match']:.3f}")
        print(f"  binary agree (>=2):  {ov['binary_at_2']:.3f}")
        print("  confusion (rows=human 0..3, cols=llm 0..3):")
        for h, row in enumerate(ov["confusion_matrix"]["matrix"]):
            print(f"    human={h}: {row}")
        if ov["exact_match"] < 0.5:
            print(
                "\n  *** WARNING: exact agreement < 0.5. Human grades still "
                "override, but review the rubric/prompt before trusting the "
                "LLM-only grades. ***"
            )

    print("\n=== Coverage ===")
    print(f"  questions in output: {len(final)} (expect 50)")
    print(f"  unknown doc ids total: {n_unknown_total}")
    if missing_meta:
        print(f"  questions without metadata: {missing_meta}")
    for w in coverage_warnings:
        print(f"  coverage: {w}")

    print("\n=== Per-question summary ===")
    for qid in sorted(final):
        e = final[qid]
        gc = e["grade_counts"]
        sc = e["source_counts"]
        print(
            f"  {qid}  {e['category']:11s}  "
            f"3={gc['3']} 2={gc['2']} 1={gc['1']} 0={gc['0']}  "
            f"n={e['n_judgments']}  "
            f"(llm={sc['llm']} human={sc['human']} "
            f"override={sc['human_override']})"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(final, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote merged qrels to {args.out}")
    print(f"Wrote agreement report to {args.agreement_out}")


if __name__ == "__main__":
    main()
