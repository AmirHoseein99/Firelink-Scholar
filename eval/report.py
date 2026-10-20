"""Human-readable status and comparison reports."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from eval import paths
from eval.storage import now_iso, save_json

STATUS_METRICS = ("ndcg@10", "recall@10", "mrr@10", "precision@10", "claim_coverage@10")


def _fmt(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "-"


def status_text(
    *,
    dataset: dict[str, Any],
    questions: list[dict[str, Any]],
    answers: dict[str, Any],
    qrels: dict[str, Any],
    runs: dict[str, dict[str, Any]],
    run_payloads: dict[str, dict[str, Any]],
    warnings: Iterable[str] = (),
) -> str:
    n_questions = len(questions)
    verified = sum(
        1 for qid in (question["id"] for question in questions)
        if (answers.get(qid) or {}).get("status") == "verified"
    )
    needs_review = [
        qid for qid in (question["id"] for question in questions)
        if (answers.get(qid) or {}).get("status") != "verified"
    ]
    n_graded = sum(len(entry.get("judgments", {})) for entry in qrels.values())
    retrieved = sum(int(payload.get("n_retrieved", 0)) for payload in run_payloads.values())
    unjudged = sum(int(payload.get("n_unjudged", 0)) for payload in run_payloads.values())
    coverage = (retrieved - unjudged) / retrieved if retrieved else 1.0
    coverage_text = (
        f"Coverage: {coverage:.0%} of retrieved ids graded"
        if retrieved
        else "Coverage: n/a (no runs)"
    )
    judge = dataset.get("judge", {})
    if not judge.get("configured", judge.get("model")):
        judge_line = "Judge: unconfigured (grading disabled)"
    else:
        judge_line = (
            f"Judge: {judge.get('model', '?')} via {judge.get('base_url', '?')} "
            f"(prompt {str(judge.get('prompt_sha256', ''))[:12]})"
        )
    lines = [
        f"Dataset: {paths.relative(paths.EVAL_DIR)}/ ({n_questions} questions)",
        f"Answer keys: {verified}/{n_questions} verified"
        + (f" (needs review: {', '.join(needs_review)})" if needs_review else ""),
        judge_line,
        f"Qrels: {len(qrels)}/{n_questions} questions; {n_graded} retrieved ids graded",
        "Runs: " + (", ".join(sorted(runs)) if runs else "(none)"),
        coverage_text + (f" ({unjudged} ungraded)" if unjudged else ""),
    ]
    warning_list = list(warnings)
    if unjudged:
        warning_list.append(f"{unjudged} retrieved ids are ungraded; run `evaluate` or `grade`")
    if needs_review:
        warning_list.append("unverified answer keys block grading")
    lines.append(
        "Warnings: " + ("; ".join(warning_list) if warning_list else "none")
    )
    return "\n".join(lines)


def comparison_text(scores: dict[str, dict[str, Any]]) -> str:
    if not scores:
        return "No scored runs."
    names = sorted(scores)
    lines = ["metric".ljust(22) + "".join(name.rjust(16) for name in names)]
    for metric in STATUS_METRICS:
        values = [
            _fmt((scores[name].get("overall") or {}).get(metric)) for name in names
        ]
        lines.append(metric.ljust(22) + "".join(value.rjust(16) for value in values))
    coverage_values = [_fmt(scores[name].get("coverage")) for name in names]
    lines.append("coverage".ljust(22) + "".join(value.rjust(16) for value in coverage_values))
    lines.append("partial".ljust(22) + "".join(
        str(scores[name].get("partial", False)).rjust(16) for name in names
    ))
    return "\n".join(lines)


def build_scores_report(
    *,
    dataset: dict[str, Any],
    scores: dict[str, dict[str, Any]],
    run_payloads: dict[str, dict[str, Any]],
    warnings: Iterable[str] = (),
) -> dict[str, Any]:
    return {
        "generated_at": now_iso(),
        "judge": dataset.get("judge", {}),
        "qrels_sha256": next(
            (score.get("qrels_sha256") for score in scores.values() if score), ""
        ),
        "runs": scores,
        "run_configs": {
            name: payload.get("config", {}) for name, payload in run_payloads.items()
        },
        "warnings": list(warnings),
    }


def write_scores_report(
    report: dict[str, Any],
    path: Path = paths.REPORTS / "scores.json",
) -> Path:
    save_json(path, report)
    return path


def _metric_sort_key(metric: str) -> tuple[str, int, str]:
    name, _, suffix = metric.partition("@")
    return (name, int(suffix) if suffix.isdigit() else -1, metric)


def scores_csv_rows(scores: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten per-run scores into one row per run."""
    names = sorted(scores)
    metrics = sorted(
        {metric for score in scores.values() for metric in (score.get("overall") or {})},
        key=_metric_sort_key,
    )
    rows = []
    for name in names:
        score = scores[name]
        overall = score.get("overall") or {}
        row: dict[str, Any] = {"run": name}
        row.update({metric: overall.get(metric) for metric in metrics})
        row["coverage"] = score.get("coverage")
        row["partial"] = score.get("partial", False)
        row["n_unjudged"] = (score.get("unjudged") or {}).get("n_ids")
        row["qrels_sha256"] = score.get("qrels_sha256", "")
        rows.append(row)
    return rows


def write_scores_csv(
    scores: dict[str, dict[str, Any]],
    path: Path = paths.REPORTS / "scores.csv",
) -> Path:
    rows = scores_csv_rows(scores)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["run"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def snapshot_warnings(run_payloads: dict[str, dict[str, Any]]) -> list[str]:
    hashes = {
        (payload.get("config") or {}).get("qrels_sha256_at_creation", "")
        for payload in run_payloads.values()
    }
    if len(hashes) > 1:
        return [
            (
                "run files were created against different qrels snapshots; "
                "all runs were re-scored against the current qrels"
            )
        ]
    return []
