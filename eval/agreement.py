"""LLM-vs-human agreement report on probe pairs.

Run:
    python -m eval.agreement                       # report for all questions
    python -m eval.agreement --only q002,q005      # subset

Probes are the human-judged holdout ids (every 5th human id, sorted by
doc_id) that the LLM also graded; their human grade stays authoritative in
``judgments`` while the LLM grade is stored under ``probe_llm``. This module
compares the two sides per question and pooled: exact agreement and binary
agreement at the grade >= 2 cutoff (the same cutoff score.py defaults to).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

BINARY_CUTOFF = 2


def probe_pairs(entry: dict) -> list[tuple[int, int]]:
    """(human_grade, llm_grade) pairs for one question's probe holdout."""
    judgments = entry.get("judgments", {})
    pairs = []
    for doc_id, llm in entry.get("probe_llm", {}).items():
        human = judgments.get(doc_id)
        if human is not None:
            pairs.append((human, llm))
    return pairs


def pair_metrics(pairs: list[tuple[int, int]]) -> dict:
    if not pairs:
        return {"n_pairs": 0, "exact": None, "binary": None,
                "confusion": {}}
    exact = sum(1 for h, l in pairs if h == l)
    binary = sum(1 for h, l in pairs
                 if (h >= BINARY_CUTOFF) == (l >= BINARY_CUTOFF))
    confusion = Counter(f"{h}->{l}" for h, l in pairs)
    return {
        "n_pairs": len(pairs),
        "exact": exact / len(pairs),
        "binary": binary / len(pairs),
        "confusion": dict(sorted(confusion.items())),
    }


def collect(judgments: dict, only: set[str] | None = None) -> dict:
    per_q: dict[str, dict] = {}
    all_pairs: list[tuple[int, int]] = []
    for qid in sorted(judgments):
        if only is not None and qid not in only:
            continue
        metrics = pair_metrics(probe_pairs(judgments[qid]))
        if metrics["n_pairs"]:
            per_q[qid] = metrics
            all_pairs.extend(probe_pairs(judgments[qid]))
    return {"per_question": per_q, "pooled": pair_metrics(all_pairs)}


def print_report(report: dict) -> None:
    pooled = report["pooled"]
    print("=== LLM-vs-human probe agreement ===")
    print(f"  {'question':10s} {'pairs':>5s} {'exact':>7s} {'binary':>7s}")
    for qid, m in report["per_question"].items():
        exact = f"{m['exact']:.3f}" if m["exact"] is not None else "-"
        binary = f"{m['binary']:.3f}" if m["binary"] is not None else "-"
        print(f"  {qid:10s} {m['n_pairs']:>5d} {exact:>7s} {binary:>7s}")
    if pooled["n_pairs"]:
        print(f"  {'pooled':10s} {pooled['n_pairs']:>5d} "
              f"{pooled['exact']:.3f} {pooled['binary']:.3f}")
    if pooled.get("confusion"):
        print(f"  confusion (human->llm): {pooled['confusion']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judgments", type=Path,
                    default=REPO_ROOT / "eval" / "v3" / "judgments.json")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "eval" / "v3" / "reports"
                    / "agreement.json")
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated question ids (e.g. q002,q005)")
    args = ap.parse_args()

    judgments = json.loads(args.judgments.read_text(encoding="utf-8"))
    only = ({p.strip() for p in args.only.split(",") if p.strip()}
            if args.only else None)
    report = collect(judgments, only)
    report["generated_at"] = datetime.now(UTC).isoformat(
        timespec="seconds").replace("+00:00", "Z")
    print_report(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
