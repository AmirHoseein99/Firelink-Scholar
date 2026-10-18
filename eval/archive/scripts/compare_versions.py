"""Build the v4 version-comparison report from run files + judgments.

Run:
    python -m eval.compare_versions --label v4 \
        --runs eval/v4/runs/contextfromvectors_k10.run.json \
               eval/v4/runs/contextfromvectors_enhanced_k10.run.json \
               eval/v4/runs/contextfromvectors_bm25_k10.run.json \
               eval/v4/runs/contextfromvectors_hybrid_k10.run.json

Writes ``eval/v4/reports/version_comparison.json`` (schema follows the v3
report of the same name): overall + per-question metrics per run,
judgment-set stats for the v4 qrels (and, if available, the v3 qrels for
before/after context), LLM-vs-human agreement measured on the top-10 ids
each run actually retrieved, and the run-level unjudged tripwire totals.
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

from eval.score import score_run

PILOT_QUESTIONS = ("q002", "q005", "q013", "q023", "q025", "q039")
BINARY_CUTOFF = 2
COVERAGE_DEPTH = 10


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"missing required input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def judgment_set_stats(
    judgments: dict,
    runs: list[tuple[str, dict]],
    label: str,
) -> dict:
    """Grade distribution + how much of each run's top-10 is judged."""
    n_total = 0
    grade_counts: Counter = Counter()
    for entry in judgments.values():
        for g in entry.get("judgments", {}).values():
            grade_counts[g] += 1
            n_total += 1
    stats = {
        "n_judged_total": n_total,
        "grade_counts": {str(g): n for g, n in sorted(grade_counts.items())},
        "pct_relevant(>=2)": round(
            100.0 * sum(n for g, n in grade_counts.items() if g >= 2) / n_total, 1)
        if n_total else None,
    }
    coverage = []
    for _name, data in runs:
        for qid, ids in data.get("run", {}).items():
            judged = judgments.get(qid, {}).get("judgments", {})
            depth = ids[:COVERAGE_DEPTH]
            if depth:
                coverage.append(
                    sum(1 for d in depth if d in judged) / len(depth))
    stats[f"judged_coverage_of_retrieved_{COVERAGE_DEPTH}"] = (
        round(100.0 * sum(coverage) / len(coverage)) if coverage else None)
    stats["label"] = label
    return stats


def agreement_on_retrieved(
    judgments: dict,
    human_seeds: dict,
    runs: list[tuple[str, dict]],
) -> dict:
    """LLM(v4 qrels)-vs-human agreement on each run's retrieved top-10."""
    out: dict[str, dict] = {}
    for name, data in runs:
        pairs: list[tuple[int, int]] = []
        per_id: dict[str, list[dict]] = {}
        for qid, ids in data.get("run", {}).items():
            judged = judgments.get(qid, {}).get("judgments", {})
            human = human_seeds.get(qid, {})
            rows = []
            for doc_id in ids[:COVERAGE_DEPTH]:
                if doc_id in human and doc_id in judged:
                    pairs.append((human[doc_id], judged[doc_id]))
                    rows.append({"doc_id": doc_id, "human": human[doc_id],
                                 "llm_v4": judged[doc_id]})
            if rows:
                per_id[qid] = rows
        n = len(pairs)
        exact = sum(1 for h, l in pairs if h == l) / n if n else None
        binary = (sum(1 for h, l in pairs
                      if (h >= BINARY_CUTOFF) == (l >= BINARY_CUTOFF)) / n
                  if n else None)
        out[name] = {
            "n": n,
            "exact": round(exact, 3) if exact is not None else None,
            "binary@2": round(binary, 3) if binary is not None else None,
        }
        if per_id and any(q in PILOT_QUESTIONS for q in per_id):
            out[name]["per_id"] = {
                q: rows for q, rows in sorted(per_id.items())
                if q in PILOT_QUESTIONS
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default="v4")
    ap.add_argument("--runs", type=Path, action="append", required=True)
    ap.add_argument("--judgments", type=Path,
                    default=REPO_ROOT / "eval" / "v4" / "judgments.json")
    ap.add_argument("--baseline-judgments", type=Path,
                    default=REPO_ROOT / "eval" / "v3" / "judgments.json")
    ap.add_argument("--human-seeds", type=Path,
                    default=REPO_ROOT / "eval" / "v3" / "seeds"
                    / "human-judgments.json")
    ap.add_argument("--dense-bias", type=Path,
                    default=REPO_ROOT / "eval" / "v4" / "reports"
                    / "dense_bias.json")
    ap.add_argument("--min-grade", type=int, default=2)
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "eval" / "v4" / "reports"
                    / "version_comparison.json")
    args = ap.parse_args()

    judgments = _load(args.judgments)
    human_seeds = {
        qid: entry["judgments"]
        for qid, entry in _load(args.human_seeds).items()
    }

    runs: list[tuple[str, dict]] = []
    metrics_ks: set[int] = set()
    for raw in args.runs:
        data = _load(raw)
        name = raw.name.removesuffix(".run.json")
        k = data.get("config", {}).get("k") or max(
            len(v) for v in data["run"].values())
        ks = [x for x in (5, 10, 20) if x <= k]
        metrics_ks.update(ks)
        metrics = score_run(data["run"], judgments, ks, args.min_grade)
        runs.append((name, data))
        data["_metrics"] = metrics
        data["_ks"] = ks
        data["_unjudged_total"] = data.get("n_unjudged_total")

    overall = {
        name: {**{k2: v2 for k2, v2 in data["_metrics"]["overall"].items()},
               "n_unjudged_total": data["_unjudged_total"]}
        for name, data in runs
    }
    per_question_ndcg10 = {
        name: {
            qid: round(m.get("ndcg@10", 0.0), 4)
            for qid, m in data["_metrics"]["by_category"].items()
        }
        for name, data in runs
    }

    judgment_stats = {args.label: judgment_set_stats(judgments, runs, args.label)}
    if args.baseline_judgments.exists():
        baseline = _load(args.baseline_judgments)
        judgment_stats["v3"] = judgment_set_stats(baseline, runs, "v3")

    report = {
        "generated_at": datetime.now(UTC).isoformat(
            timespec="seconds").replace("+00:00", "Z"),
        "label": args.label,
        "runs": [_rel(raw) for raw in args.runs],
        "metrics_ks": sorted(metrics_ks),
        "min_grade": args.min_grade,
        "overall": overall,
        "per_question_ndcg10": per_question_ndcg10,
        "judgment_set_stats": judgment_stats,
        "agreement_with_human_on_retrieved_10": agreement_on_retrieved(
            judgments, human_seeds, runs),
        "dense_bias_context": _load(args.dense_bias).get("pooled")
        if args.dense_bias.exists() else None,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print(f"=== {args.label} comparison -> {args.out} ===")
    keys = [k2 for k2 in overall[next(iter(overall))] if k2 != "n"]
    header = f"  {'run':40s} " + " ".join(k2.rjust(8) for k2 in keys)
    print(header)
    for name, _ in runs:
        m = overall[name]
        print(f"  {name:40s} " + " ".join(
            (f"{m[k2]:.4f}" if isinstance(m.get(k2), float)
             else str(m.get(k2, "-"))).rjust(8) for k2 in keys))
    for label, stats in judgment_stats.items():
        print(f"  judgment set {label}: {stats['n_judged_total']} judged, "
              f"pct_relevant(>=2)={stats['pct_relevant(>=2)']}%, "
              f"coverage={stats.get(f'judged_coverage_of_retrieved_{COVERAGE_DEPTH}')}%")


if __name__ == "__main__":
    main()
