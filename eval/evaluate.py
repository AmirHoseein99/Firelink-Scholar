"""Evaluate orchestration: run systems, grade the union once, score all runs."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from eval import paths
from eval.dataset import claim_limits, default_k, metric_settings
from eval.grade import grade_run_payloads
from eval.llm import Backend
from eval.metrics import score_run
from eval.report import (
    build_scores_report,
    comparison_text,
    snapshot_warnings,
    write_scores_report,
)
from eval.runs import (
    collect_run,
    iter_run_files,
    load_run,
    resolve_run_paths,
    save_run,
)
from eval.storage import load_json, now_iso, save_json, sha256_json

logger = logging.getLogger(__name__)


def score_payloads(
    payloads: dict[str, dict[str, Any]],
    *,
    qrels: dict[str, Any],
    questions_by_id: dict[str, dict[str, Any]],
    answers_by_id: dict[str, Any] | None = None,
    ks: Iterable[int] = (5, 10, 20),
    min_grade: int = 2,
    write_back: bool = True,
    runs_dir: Path = paths.RUNS,
) -> dict[str, dict[str, Any]]:
    """Score run payloads against one qrels snapshot."""
    qrels_hash = sha256_json(qrels) if qrels else ""
    scores: dict[str, dict[str, Any]] = {}
    for name, payload in payloads.items():
        result = score_run(
            payload.get("run", {}),
            qrels,
            questions_by_id,
            answers_by_id,
            ks=ks,
            min_grade=min_grade,
            qrels_sha256=qrels_hash,
        )
        scores[name] = result
        payload["metrics"] = result
        payload["n_unjudged"] = result["unjudged"]["n_ids"]
        payload["n_unjudged_total"] = result["unjudged"]["n_ids"]
        payload["scored_at"] = now_iso()
        logger.debug(
            "%s: scored ndcg@10=%s, coverage=%s, partial=%s",
            name,
            (result.get("overall") or {}).get("ndcg@10"),
            result.get("coverage"),
            result.get("partial"),
        )
        if write_back:
            config = payload.setdefault("config", {})
            system = config.get("system") or name
            k = int(config.get("k") or 0)
            if system and k:
                save_json(runs_dir / f"{system}_k{k}.run.json", payload)
    return scores


def score_all_runs(
    *,
    qrels: dict[str, Any],
    questions_by_id: dict[str, dict[str, Any]],
    answers_by_id: dict[str, Any] | None = None,
    ks: Iterable[int] = (5, 10, 20),
    min_grade: int = 2,
    run_paths: Iterable[Path] | None = None,
    runs_dir: Path = paths.RUNS,
    write_back: bool = True,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    paths_to_score = list(run_paths) if run_paths is not None else iter_run_files(runs_dir)
    logger.info(
        "Scoring %d run file(s) against %d qrels question(s)",
        len(paths_to_score),
        len(qrels),
    )
    payloads = {
        path.name[: -len(".run.json")] if path.name.endswith(".run.json") else path.stem: load_run(path)
        for path in paths_to_score
    }
    scores = score_payloads(
        payloads,
        qrels=qrels,
        questions_by_id=questions_by_id,
        answers_by_id=answers_by_id,
        ks=ks,
        min_grade=min_grade,
        write_back=write_back,
        runs_dir=runs_dir,
    )
    return scores, payloads


def execute_runs(
    *,
    systems: Iterable[str],
    questions: list[dict[str, Any]],
    k: int,
    qrels: dict[str, Any],
    runs_dir: Path = paths.RUNS,
    retriever: Callable[[str], Callable[..., list[Any]]] | None = None,
    merge: bool = False,
    questions_path: Path = paths.QUESTIONS,
) -> dict[str, dict[str, Any]]:
    systems = list(systems)
    logger.info(
        "Running %s at k=%d on %d question(s)",
        ", ".join(systems),
        k,
        len(questions),
    )
    payloads: dict[str, dict[str, Any]] = {}
    for system in systems:
        run = collect_run(system, questions, k, retrieve_fn=(
            retriever(system) if retriever is not None else None
        ))
        path = save_run(
            system,
            k,
            run,
            qrels,
            runs_dir=runs_dir,
            questions_path=questions_path,
            merge=merge,
        )
        payloads[system] = load_run(path)
    return payloads


def evaluate(
    *,
    systems: Iterable[str],
    questions: list[dict[str, Any]],
    answers: dict[str, Any],
    qrels: dict[str, Any],
    dataset: dict[str, Any],
    backend: Backend,
    k: int | None = None,
    only: set[str] | None = None,
    limit: int | None = None,
    runs_dir: Path = paths.RUNS,
    reports_dir: Path = paths.REPORTS,
    qrels_path: Path = paths.QRELS,
    questions_path: Path = paths.QUESTIONS,
    retriever: Callable[[str], Callable[..., list[Any]]] | None = None,
    chunk_fetcher: Callable[[Iterable[str]], dict[str, dict[str, Any]]] | None = None,
    call: Callable[..., str] | None = None,
    force_judge: bool = False,
    allow_unverified: bool = False,
    batch_size: int = 5,
    concurrency: int = 1,
) -> dict[str, Any]:
    """Run selected systems, grade the union of new ids, then score every run."""
    systems = list(systems)
    k = int(k if k is not None else default_k(dataset))
    selected_questions = questions
    if only:
        selected_questions = [
            question for question in selected_questions if str(question["id"]) in only
        ]
    if limit is not None:
        selected_questions = selected_questions[:limit]
    merge = only is not None or limit is not None
    logger.info(
        "Evaluate: systems=%s, k=%d, questions=%d, merge=%s",
        ", ".join(systems),
        k,
        len(selected_questions),
        merge,
    )

    new_payloads = execute_runs(
        systems=systems,
        questions=selected_questions,
        k=k,
        qrels=qrels,
        runs_dir=runs_dir,
        retriever=retriever,
        merge=merge,
        questions_path=questions_path,
    )

    logger.info("Evaluate: grading union of new retrieved ids")
    grade_kwargs: dict[str, Any] = {}
    if chunk_fetcher is not None:
        grade_kwargs["chunk_fetcher"] = chunk_fetcher
    grade_summary = grade_run_payloads(
        new_payloads.values(),
        questions=questions,
        answers=answers,
        qrels=qrels,
        dataset=dataset,
        backend=backend,
        qrels_path=qrels_path,
        only=only,
        limit=limit,
        force_judge=force_judge,
        allow_unverified=allow_unverified,
        call=call,
        batch_size=batch_size,
        concurrency=concurrency,
        **grade_kwargs,
    )

    ks, min_grade = metric_settings(dataset)
    logger.info("Evaluate: re-scoring all saved runs")
    scores, all_payloads = score_all_runs(
        qrels=qrels,
        questions_by_id={str(question["id"]): question for question in questions},
        answers_by_id=answers,
        ks=ks,
        min_grade=min_grade,
        runs_dir=runs_dir,
    )
    warnings = snapshot_warnings(all_payloads)
    report = build_scores_report(
        dataset=dataset,
        scores=scores,
        run_payloads=all_payloads,
        warnings=warnings,
    )
    report_path = write_scores_report(report, reports_dir / "scores.json")
    logger.info("Evaluate: report written to %s", paths.relative(report_path))
    return {
        "grade_summary": grade_summary,
        "scores": scores,
        "report_path": report_path,
        "table": comparison_text(scores),
    }


def run_only(
    *,
    systems: Iterable[str],
    questions: list[dict[str, Any]],
    k: int,
    qrels: dict[str, Any],
    runs_dir: Path = paths.RUNS,
    retriever: Callable[[str], Callable[..., list[Any]]] | None = None,
    only: set[str] | None = None,
    limit: int | None = None,
    questions_path: Path = paths.QUESTIONS,
) -> dict[str, dict[str, Any]]:
    selected = questions
    if only:
        selected = [question for question in selected if str(question["id"]) in only]
    if limit is not None:
        selected = selected[:limit]
    return execute_runs(
        systems=systems,
        questions=selected,
        k=k,
        qrels=qrels,
        runs_dir=runs_dir,
        retriever=retriever,
        merge=only is not None or limit is not None,
        questions_path=questions_path,
    )


def load_benchmark_questions(path: Path = paths.QUESTIONS) -> list[dict[str, Any]]:
    return list(load_json(path, default=[]) or [])


def load_answers(path: Path = paths.ANSWERS) -> dict[str, Any]:
    return load_json(path, default={}) or {}


def load_qrels(path: Path = paths.QRELS) -> dict[str, Any]:
    return load_json(path, default={}) or {}


def claim_cap(dataset: dict[str, Any], game: str) -> int:
    return claim_limits(dataset, game)[1]


def resolve_runs(
    requested: Iterable[str] | None,
    runs_dir: Path = paths.RUNS,
) -> list[Path]:
    return resolve_run_paths(requested, runs_dir=runs_dir)
