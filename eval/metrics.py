"""Retrieval metrics over the accumulated retrieved-and-judged qrels."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from eval.storage import sha256_json


def dcg(gains: Iterable[float]) -> float:
    return sum((2.0**gain - 1.0) / math.log2(index + 2) for index, gain in enumerate(gains))


def ndcg_at_k(run_ids: list[str], judgments: dict[str, int], k: int) -> float:
    gains = [float(judgments.get(doc_id, 0)) for doc_id in run_ids[:k]]
    ideal = sorted((float(grade) for grade in judgments.values()), reverse=True)[:k]
    denominator = dcg(ideal)
    return dcg(gains) / denominator if denominator > 0 else 0.0


def _relevant(judgments: dict[str, int], min_grade: int) -> set[str]:
    return {doc_id for doc_id, grade in judgments.items() if grade >= min_grade}


def recall_at_k(
    run_ids: list[str], judgments: dict[str, int], k: int, min_grade: int
) -> float | None:
    relevant = _relevant(judgments, min_grade)
    if not relevant:
        return None
    return sum(1 for doc_id in run_ids[:k] if doc_id in relevant) / len(relevant)


def mrr_at_k(
    run_ids: list[str], judgments: dict[str, int], k: int, min_grade: int
) -> float | None:
    relevant = _relevant(judgments, min_grade)
    if not relevant:
        return None
    for index, doc_id in enumerate(run_ids[:k], start=1):
        if doc_id in relevant:
            return 1.0 / index
    return 0.0


def precision_at_k(
    run_ids: list[str], judgments: dict[str, int], k: int, min_grade: int
) -> float | None:
    relevant = _relevant(judgments, min_grade)
    if not relevant:
        return None
    return sum(1 for doc_id in run_ids[:k] if doc_id in relevant) / k


def claim_coverage_at_k(
    run_ids: list[str],
    qrels_entry: dict[str, Any],
    answer_entry: dict[str, Any] | None,
    k: int,
    *,
    importances: tuple[str, ...] = ("core",),
) -> float | None:
    """Fraction of claims of the given importances supported in the top-k.

    The denominator is the answer key, not the growing qrels, so this metric
    is stable as more systems add judgments.
    """
    if not answer_entry:
        return None
    wanted = {
        claim["id"]
        for claim in answer_entry.get("claims", [])
        if claim.get("importance") in importances
    }
    if not wanted:
        return None
    supports = qrels_entry.get("supports_claims", {}) if qrels_entry else {}
    supported: set[str] = set()
    for doc_id in run_ids[:k]:
        supported.update(supports.get(doc_id, []))
    return len(wanted & supported) / len(wanted)


def _metric_keys(ks: Iterable[int]) -> list[str]:
    keys: list[str] = []
    keys += [f"ndcg@{k}" for k in ks]
    keys += [f"recall@{k}" for k in ks]
    keys += [f"mrr@{k}" for k in ks]
    keys += [f"precision@{k}" for k in ks]
    keys += [f"claim_coverage@{k}" for k in ks]
    keys += [f"claim_coverage_all@{k}" for k in ks]
    return keys


def _aggregate(rows: list[dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for key in _metric_keys(ks):
        values = [row[key] for row in rows if row.get(key) is not None]
        if values:
            aggregate[key] = sum(values) / len(values)
    aggregate["n"] = len(rows)
    return aggregate


def score_run(
    run: dict[str, list[dict[str, Any]]],
    qrels: dict[str, Any],
    questions_by_id: dict[str, dict[str, Any]],
    answers_by_id: dict[str, Any] | None = None,
    *,
    ks: Iterable[int] = (5, 10, 20),
    min_grade: int = 2,
    qrels_sha256: str | None = None,
) -> dict[str, Any]:
    """Score one run and return per-question metrics keyed by ``qNNN``.

    Unjudged retrieved ids score 0 (the standard pessimistic assumption) and
    are counted per question. ``partial`` is true when any retrieved id is
    ungraded.
    """
    ks = list(ks)
    qrels_hash = qrels_sha256 if qrels_sha256 is not None else (
        sha256_json(qrels) if qrels else ""
    )
    qids = sorted(set(qrels) | set(run))
    per_question: dict[str, Any] = {}
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_retrieved = 0
    total_unjudged = 0

    for qid in qids:
        results = run.get(qid, [])
        run_ids = [item.get("document_id") for item in results if item.get("document_id")]
        entry = qrels.get(qid) or {}
        judgments = entry.get("judgments", {}) or {}
        question = questions_by_id.get(qid) or {}
        answer = (answers_by_id or {}).get(qid)
        category = question.get("category", "uncategorized")
        unjudged_ids = [doc_id for doc_id in run_ids if doc_id not in judgments]

        metrics: dict[str, Any] = {}
        for k in ks:
            metrics[f"ndcg@{k}"] = ndcg_at_k(run_ids, judgments, k)
        for k in ks:
            metrics[f"recall@{k}"] = recall_at_k(run_ids, judgments, k, min_grade)
            metrics[f"mrr@{k}"] = mrr_at_k(run_ids, judgments, k, min_grade)
            metrics[f"precision@{k}"] = precision_at_k(run_ids, judgments, k, min_grade)
            metrics[f"claim_coverage@{k}"] = claim_coverage_at_k(
                run_ids, entry, answer, k, importances=("core",)
            )
            metrics[f"claim_coverage_all@{k}"] = claim_coverage_at_k(
                run_ids, entry, answer, k, importances=("core", "supporting")
            )

        per_question[qid] = {
            "category": category,
            "metrics": metrics,
            "n_retrieved": len(run_ids),
            "unjudged": len(unjudged_ids),
            "unjudged_ids": unjudged_ids,
        }
        by_category[category].append(metrics)
        total_retrieved += len(run_ids)
        total_unjudged += len(unjudged_ids)

    all_rows = [row for rows in by_category.values() for row in rows]
    coverage = (
        (total_retrieved - total_unjudged) / total_retrieved
        if total_retrieved
        else 1.0
    )
    return {
        "per_question": per_question,
        "by_category": {
            category: _aggregate(rows, ks)
            for category, rows in sorted(by_category.items())
        },
        "overall": _aggregate(all_rows, ks),
        "n_scored": sum(1 for row in per_question.values() if row["n_retrieved"]),
        "n_questions": len(per_question),
        "min_grade": min_grade,
        "ks": ks,
        "qrels_sha256": qrels_hash,
        "unjudged": {
            "n_ids": total_unjudged,
            "n_retrieved": total_retrieved,
            "by_question": {
                qid: row["unjudged"]
                for qid, row in per_question.items()
                if row["unjudged"]
            },
        },
        "coverage": coverage,
        "partial": total_unjudged > 0,
    }


def metric_value(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    return float(value) if isinstance(value, (int, float)) else None
