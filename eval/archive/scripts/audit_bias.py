"""Quantify the dense-retriever circularity baked into the v3 human gold.

Run:
    python -m eval.audit_bias                    # write eval/v4/reports/dense_bias.json
    python -m eval.audit_bias --out /tmp/d.json  # custom output path

The v1 human qrels (``eval/v1/qrels.final.json``) were judged on candidate
pools whose head is the dense retriever's own top-k, and
``eval/v3/seeds/human-judgments.json`` imports those grades verbatim. This
script measures, per question and pooled, how much of the v3 human gold
(grade >= 2) sits inside the v1 dense run's top-10 — the "before" baseline
that v4's all-system pools are meant to fix. It also records the v3 metrics
anomaly (dense ndcg@10 vs hybrid ndcg@10) for the README narrative.

Output schema (``eval/v4/reports/dense_bias.json``):
    {generated_at, inputs, provenance_check, pooled, per_question}
where ``provenance_check`` verifies the v3-gold == v1-gold identity claim and
``per_question`` lists, for each question, the overlap ids as evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MIN_GRADE = 2
DENSE_K = 10


def _load(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"missing required input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def audit_question(
    qid: str,
    human_gold: set[str],
    dense_run_ids: list[str],
) -> dict:
    top = dense_run_ids[:DENSE_K]
    in_gold = [d for d in top if d in human_gold]
    return {
        "n_human_gold": len(human_gold),
        "n_dense_top10_in_gold": len(in_gold),
        "dense_top10_precision": len(in_gold) / len(top) if top else None,
        "n_gold_in_dense_top10": len(in_gold),
        "frac_gold_in_dense_top10": (
            len(in_gold) / len(human_gold) if human_gold else None
        ),
        "gold_ids_in_dense_top10": in_gold,
    }


def _metric_of(run_file: dict, key: str) -> float | None:
    return ((run_file.get("metrics") or {}).get("overall") or {}).get(key)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--human-seeds", type=Path,
                    default=REPO_ROOT / "eval" / "v3" / "seeds"
                    / "human-judgments.json")
    ap.add_argument("--v1-qrels", type=Path,
                    default=REPO_ROOT / "eval" / "v1" / "qrels.final.json")
    ap.add_argument("--v1-dense-run", type=Path,
                    default=REPO_ROOT / "eval" / "v1" / "runs"
                    / "contextfromvectors_k10.run.json")
    ap.add_argument("--v3-dense-run", type=Path,
                    default=REPO_ROOT / "eval" / "v3" / "runs"
                    / "contextfromvectors_k10.run.json")
    ap.add_argument("--v3-hybrid-run", type=Path,
                    default=REPO_ROOT / "eval" / "v3" / "runs"
                    / "contextfromvectors_hybrid_k10.run.json")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "eval" / "v4" / "reports"
                    / "dense_bias.json")
    args = ap.parse_args()

    human = _load(args.human_seeds)
    v1_qrels = _load(args.v1_qrels)
    v1_run = _load(args.v1_dense_run)

    per_question: dict[str, dict] = {}
    fracs: list[float] = []
    overlaps: list[int] = []
    gold_totals: list[int] = []
    for qid in sorted(human):
        judgments = human[qid].get("judgments", {})
        human_gold = {d for d, g in judgments.items() if g >= MIN_GRADE}
        v1_run_ids = (v1_run.get("run") or {}).get(qid) or []
        row = audit_question(qid, human_gold, v1_run_ids)
        v1_gold = {
            d for d, g in ((v1_qrels.get(qid) or {}).get("judgments") or {}).items()
            if g >= MIN_GRADE
        }
        row["v1_gold_identical_to_v3"] = v1_gold == human_gold
        per_question[qid] = row
        gold_totals.append(len(human_gold))
        overlaps.append(row["n_gold_in_dense_top10"])
        if row["frac_gold_in_dense_top10"] is not None:
            fracs.append(row["frac_gold_in_dense_top10"])

    n_identical = sum(1 for r in per_question.values()
                      if r["v1_gold_identical_to_v3"])
    provenance = {
        "claim": "v3 human gold is imported verbatim from v1 qrels",
        "questions_total": len(per_question),
        "questions_with_identical_gold": n_identical,
        "all_identical": n_identical == len(per_question),
    }

    pooled = {
        "n_questions": len(per_question),
        "human_gold_total": sum(gold_totals),
        "gold_ids_in_v1_dense_top10_total": sum(overlaps),
        "mean_frac_gold_in_dense_top10": (
            statistics.fmean(fracs) if fracs else None),
        "median_frac_gold_in_dense_top10": (
            statistics.median(fracs) if fracs else None),
        "n_questions_majority_gold_in_dense_top10": sum(
            1 for f in fracs if f > 0.5),
    }

    v3_dense = _load(args.v3_dense_run) if args.v3_dense_run.exists() else None
    v3_hybrid = (_load(args.v3_hybrid_run)
                 if args.v3_hybrid_run.exists() else None)
    v3_metrics = {
        "dense_ndcg@10": _metric_of(v3_dense, "ndcg@10") if v3_dense else None,
        "hybrid_ndcg@10": _metric_of(v3_hybrid, "ndcg@10") if v3_hybrid else None,
        "note": "v3 anomaly this audit explains: dense structurally wins "
                "because human gold is dense-top-k-shaped",
    }

    report = {
        "generated_at": datetime.now(UTC).isoformat(
            timespec="seconds").replace("+00:00", "Z"),
        "inputs": {
            "human_seeds": _rel(args.human_seeds),
            "v1_qrels": _rel(args.v1_qrels),
            "v1_dense_run": _rel(args.v1_dense_run),
            "v3_dense_run": _rel(args.v3_dense_run) if v3_dense else None,
            "v3_hybrid_run": _rel(args.v3_hybrid_run) if v3_hybrid else None,
            "min_grade": MIN_GRADE,
            "dense_k": DENSE_K,
        },
        "provenance_check": provenance,
        "v3_metrics_anomaly": v3_metrics,
        "pooled": pooled,
        "per_question": per_question,
    }

    print(f"provenance: {n_identical}/{len(per_question)} questions have "
          f"v3 gold identical to v1 qrels gold")
    print(f"pooled: {pooled['gold_ids_in_v1_dense_top10_total']}/"
          f"{pooled['human_gold_total']} human-gold ids "
          f"(grade >= {MIN_GRADE}) sit in the v1 dense top-10 "
          f"(mean frac {pooled['mean_frac_gold_in_dense_top10']:.3f})")
    if v3_metrics["dense_ndcg@10"] is not None:
        print(f"v3 metrics: dense ndcg@10={v3_metrics['dense_ndcg@10']:.3f} "
              f"vs hybrid ndcg@10={v3_metrics['hybrid_ndcg@10']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    os.replace(tmp, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
