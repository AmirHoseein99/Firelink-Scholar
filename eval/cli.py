"""``python -m eval`` command-line interface."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from eval import paths
from eval.answers import build_answers, load_questions
from eval.corpus import load_pages, section_lookup
from eval.dataset import (
    DatasetError,
    answer_backend_name,
    answer_model,
    configure_judge,
    default_k,
    judge_backend_name,
    judge_configured,
    judge_model,
    load_dataset,
    metric_settings,
    save_dataset_with_qrels,
)
from eval.evaluate import evaluate, load_answers, load_qrels, run_only, score_all_runs
from eval.grade import JudgeError, grade_run_payloads
from eval.grade import load_qrels as load_qrels_file
from eval.llm import LLMError, require_backend
from eval.questions import format_validation_report, validate_questions
from eval.report import (
    build_scores_report,
    comparison_text,
    status_text,
    write_scores_report,
)
from eval.runs import load_run, resolve_run_paths
from eval.storage import save_json, sha256_file_or_empty, sha256_json
from eval.systems import resolve_systems
from eval.text import quote_is_grounded

DEFAULT_REPORT = paths.REPORTS / "question_review.json"


def _parse_only(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    return {value.strip() for value in raw.split(",") if value.strip()}


def _normalize_source_path(value: str) -> str:
    text = str(value).replace("\\", "/").lstrip("./")
    prefix = "data/structured/"
    text = text.removeprefix(prefix)
    return text


def _check_dataset_questions_hash(dataset: dict[str, Any], questions_path: Path) -> list[str]:
    current = sha256_file_or_empty(questions_path)
    expected = dataset.get("questions_sha256")
    if expected and expected != current:
        return ["questions.json changed since dataset.json was written; rebuild the dataset"]
    return []


def _check_qrels_hash(dataset: dict[str, Any], qrels: dict[str, Any]) -> list[str]:
    expected = (dataset.get("qrels") or {}).get("sha256")
    current = sha256_json(qrels) if qrels else ""
    if expected and expected != current:
        return ["qrels.json changed since dataset.json was written; run `score` or refresh the dataset"]
    return []


def cmd_status(args: argparse.Namespace) -> int:
    dataset = load_dataset()
    questions = load_questions()
    answers = load_answers()
    qrels = load_qrels()
    run_paths = resolve_run_paths(None)
    run_payloads = {path.name[: -len(".run.json")]: load_run(path) for path in run_paths}
    warnings = _check_dataset_questions_hash(dataset, paths.QUESTIONS)
    warnings += _check_qrels_hash(dataset, qrels)
    print(
        status_text(
            dataset=dataset,
            questions=questions,
            answers=answers,
            qrels=qrels,
            runs={name: payload.get("metrics", {}) for name, payload in run_payloads.items()},
            run_payloads=run_payloads,
            warnings=warnings,
        )
    )
    return 0


def _validate_answers(
    answers: dict[str, Any],
    questions: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    pages = load_pages()
    lookup = section_lookup(pages)
    question_ids = [str(question["id"]) for question in questions]
    for qid in question_ids:
        answer = answers.get(qid)
        if not answer:
            warnings.append(f"{qid}: missing answer key")
            continue
        if answer.get("status") != "verified":
            warnings.append(f"{qid}: answer key status {answer.get('status')!r}")
        for claim in answer.get("claims", []):
            source = claim.get("source") or {}
            path = _normalize_source_path(source.get("path", ""))
            section_key = str(source.get("section", ""))
            section = lookup.get((path, section_key))
            if section is None:
                errors.append(
                    f"{qid}/{claim.get('id')}: cited section not found: "
                    f"{path}:{section_key}"
                )
                continue
            if not quote_is_grounded(str(claim.get("quote", "")), section.text):
                errors.append(
                    f"{qid}/{claim.get('id')}: quote is not grounded in "
                    f"{path}:{section_key}"
                )
    return errors, warnings


def _validate_ids_in_qdrant(
    qrels: dict[str, Any],
    run_payloads: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str]]:
    from eval.corpus import fetch_chunks

    ids: list[str] = []
    for entry in qrels.values():
        ids.extend(entry.get("judgments", {}).keys())
    for payload in run_payloads.values():
        for results in (payload.get("run") or {}).values():
            ids.extend(item.get("document_id") for item in results if item.get("document_id"))
    unique_ids = list(dict.fromkeys(ids))
    try:
        found = fetch_chunks(unique_ids)
    except Exception as exc:  # noqa: BLE001 - Qdrant may be unavailable
        return [f"could not read Qdrant ids: {exc}"], []
    missing = [doc_id for doc_id in unique_ids if doc_id not in found]
    errors = []
    if missing:
        errors.append(
            f"{len(missing)} qrels/run ids are missing from Qdrant: "
            + ", ".join(missing[:10])
        )
    return errors, []


def _validate_judge(dataset: dict[str, Any], qrels: dict[str, Any]) -> list[str]:
    if not judge_configured(dataset):
        return []
    pinned = judge_model(dataset)
    prompt_hash = str(dataset["judge"].get("prompt_sha256", ""))
    errors: list[str] = []
    for qid, entry in sorted(qrels.items()):
        if entry.get("graded_by") != pinned:
            errors.append(
                f"{qid}: graded_by {entry.get('graded_by')!r} != pinned "
                f"{pinned!r}"
            )
        if entry.get("prompt_sha256") != prompt_hash:
            errors.append(f"{qid}: judge prompt hash differs from dataset.json")
    return errors


def cmd_validate(args: argparse.Namespace) -> int:
    questions = load_questions()
    report = validate_questions(questions)
    save_json(DEFAULT_REPORT, report)
    print(format_validation_report(report))
    print(f"Question review written to {paths.relative(DEFAULT_REPORT)}")
    if args.questions_only:
        return 0 if report["ok"] else 1

    dataset = load_dataset()
    qrels = load_qrels()
    answers = load_answers()
    run_payloads = {
        path.name[: -len(".run.json")]: load_run(path)
        for path in resolve_run_paths(None)
    }
    errors: list[str] = []
    warnings: list[str] = _check_dataset_questions_hash(dataset, paths.QUESTIONS)
    warnings += _check_qrels_hash(dataset, qrels)
    if not report["ok"]:
        errors.append("question validation found must-fix flags")
    answer_errors, answer_warnings = _validate_answers(answers, questions)
    errors.extend(answer_errors)
    warnings.extend(answer_warnings)
    judge_errors = _validate_judge(dataset, qrels)
    errors.extend(judge_errors)
    if not judge_configured(dataset):
        warnings.append(
            "judge is not configured in dataset.json; grading is disabled "
            "until `python -m eval configure-judge --model <model>` is run"
        )
    id_errors, id_warnings = _validate_ids_in_qdrant(qrels, run_payloads)
    errors.extend(id_errors)
    warnings.extend(id_warnings)
    if args.completeness:
        missing_answers = [
            str(question["id"]) for question in questions
            if (answers.get(str(question["id"])) or {}).get("status") != "verified"
        ]
        missing_qrels = [
            str(question["id"]) for question in questions
            if not (qrels.get(str(question["id"])) or {}).get("judgments")
        ]
        if missing_answers:
            errors.append(f"{len(missing_answers)} question(s) lack a verified answer key")
        if missing_qrels:
            warnings.append(f"{len(missing_qrels)} question(s) have no qrels judgments")

    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    if not errors and not warnings:
        print("Validation: OK")
    return 0 if not errors else 1


def _check_answer_config(dataset: dict[str, Any], backend: Any, force: bool) -> None:
    from eval.prompts_answer import ANSWER_PROMPT_SHA256

    expected_model = answer_model(dataset)
    expected_prompt = dataset["answer_builder"].get("prompt_sha256")
    if backend.model != expected_model and not force:
        raise LLMError(
            f"answer builder model mismatch: backend={backend.model!r}, "
            f"dataset={expected_model!r}; pass --force to override"
        )
    if expected_prompt and expected_prompt != ANSWER_PROMPT_SHA256 and not force:
        raise LLMError(
            "answer-builder prompt changed since dataset.json was written; "
            "pass --force to rebuild"
        )


def cmd_build_answers(args: argparse.Namespace) -> int:
    dataset = load_dataset()
    backend = require_backend(answer_backend_name(dataset))
    _check_answer_config(dataset, backend, args.force)
    summary = build_answers(
        dataset=dataset,
        backend=backend,
        only=_parse_only(args.only),
        limit=args.limit,
        start=args.start,
        force=args.force,
        retries=args.retries,
    )
    print(
        f"Answer keys: built {len(summary['built'])}, skipped "
        f"{len(summary['skipped'])}, needs review {len(summary['needs_review'])}"
    )
    if summary["needs_review"]:
        print("Needs review: " + ", ".join(summary["needs_review"]))
    return 0 if not summary["needs_review"] else 1


def cmd_run(args: argparse.Namespace) -> int:
    dataset = load_dataset()
    qrels = load_qrels()
    systems = resolve_systems(args.systems)
    k = int(args.k or default_k(dataset))
    payloads = run_only(
        systems=systems,
        questions=load_questions(),
        k=k,
        qrels=qrels,
        only=_parse_only(args.only),
        limit=args.limit,
    )
    for system, payload in payloads.items():
        print(
            f"{system}_k{k}: {payload.get('n_retrieved', 0)} retrieved, "
            f"{payload.get('n_unjudged', 0)} unjudged"
        )
    return 0


def _require_judge_backend(dataset: dict[str, Any]):
    if not judge_configured(dataset):
        raise LLMError(
            "no judge is configured in eval/dataset.json; choose one with "
            "`python -m eval configure-judge --model <model>` before grading"
        )
    return require_backend(judge_backend_name(dataset))


def cmd_configure_judge(args: argparse.Namespace) -> int:
    dataset = load_dataset()
    updated = configure_judge(
        dataset,
        model=args.model,
        backend=args.backend,
        base_url=args.base_url,
    )
    judge = updated["judge"]
    print(
        f"Judge configured: {judge['model']} via {judge['base_url']} "
        f"(backend {judge['backend']}). No answers were graded."
    )
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    dataset = load_dataset()
    backend = _require_judge_backend(dataset)
    questions = load_questions()
    answers = load_answers()
    qrels = load_qrels()
    systems = resolve_systems(args.systems)
    result = evaluate(
        systems=systems,
        questions=questions,
        answers=answers,
        qrels=qrels,
        dataset=dataset,
        backend=backend,
        k=args.k,
        only=_parse_only(args.only),
        limit=args.limit,
        force_judge=args.force_judge,
        allow_unverified=args.allow_unverified,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
    )
    save_dataset_with_qrels(dataset, qrels)
    grade = result["grade_summary"]
    print(
        f"Graded {grade['graded']} new id(s); skipped {grade['skipped']} "
        f"already judged; {len(grade['missing'])} missing from Qdrant; "
        f"{len(grade['unparsed'])} unparsed."
    )
    print()
    print(result["table"])
    print(f"\nReport: {paths.relative(result['report_path'])}")
    return 0


def cmd_grade(args: argparse.Namespace) -> int:
    dataset = load_dataset()
    backend = _require_judge_backend(dataset)
    qrels = load_qrels_file()
    answers = load_answers()
    questions = load_questions()
    run_paths = resolve_run_paths(args.run)
    payloads = [load_run(path) for path in run_paths]
    summary = grade_run_payloads(
        payloads,
        questions=questions,
        answers=answers,
        qrels=qrels,
        dataset=dataset,
        backend=backend,
        only=_parse_only(args.only),
        limit=args.limit,
        force_judge=args.force_judge,
        allow_unverified=args.allow_unverified,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
    )
    save_dataset_with_qrels(dataset, qrels)
    print(
        f"Graded {summary['graded']} new id(s); skipped {summary['skipped']} "
        f"already judged; {len(summary['missing'])} missing from Qdrant; "
        f"{len(summary['unparsed'])} unparsed."
    )
    for item in summary["missing"][:20]:
        print(f"  missing: {item['question']}:{item['document_id']}")
    for item in summary["unparsed"][:20]:
        print(f"  unparsed: {item['question']}:{item['document_id']}")
    return 0 if not summary["unparsed"] else 1


def cmd_score(args: argparse.Namespace) -> int:
    dataset = load_dataset()
    questions = load_questions()
    answers = load_answers()
    qrels = load_qrels()
    ks, min_grade = metric_settings(dataset)
    run_paths = resolve_run_paths(args.run)
    scores, payloads = score_all_runs(
        qrels=qrels,
        questions_by_id={str(question["id"]): question for question in questions},
        answers_by_id=answers,
        ks=ks,
        min_grade=min_grade,
        run_paths=run_paths,
    )
    save_dataset_with_qrels(dataset, qrels)
    warnings = _check_qrels_hash(dataset, qrels)
    report = build_scores_report(
        dataset=dataset,
        scores=scores,
        run_payloads=payloads,
        warnings=warnings,
    )
    report_path = write_scores_report(report)
    print(comparison_text(scores))
    print(f"\nReport: {paths.relative(report_path)}")
    partial = any(score.get("partial") for score in scores.values())
    if partial:
        unjudged = sum(score["unjudged"]["n_ids"] for score in scores.values())
        print(f"WARNING: {unjudged} retrieved ids are ungraded (scored as 0)")
    return 1 if (args.strict and partial) else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m eval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="one-screen benchmark summary")
    status.set_defaults(func=cmd_status)

    validate = sub.add_parser("validate", help="validate questions, answers, qrels, judge")
    validate.add_argument("--questions-only", action="store_true")
    validate.add_argument("--completeness", action="store_true")
    validate.set_defaults(func=cmd_validate)

    answers = sub.add_parser("build-answers", help="build/freeze answer keys")
    answers.add_argument("--only", default=None, help="comma-separated question ids")
    answers.add_argument("--limit", type=int, default=None)
    answers.add_argument("--start", default=None, help="start at this question id")
    answers.add_argument("--force", action="store_true")
    answers.add_argument("--retries", type=int, default=1)
    answers.set_defaults(func=cmd_build_answers)

    run = sub.add_parser("run", help="retrieve only; no grading")
    run.add_argument("--systems", default="all")
    run.add_argument("--k", type=int, default=None)
    run.add_argument("--only", default=None)
    run.add_argument("--limit", type=int, default=None)
    run.set_defaults(func=cmd_run)

    configure = sub.add_parser(
        "configure-judge",
        help="pin a judge model in dataset.json (does not grade)",
    )
    configure.add_argument("--model", required=True)
    configure.add_argument("--backend", default="freellm")
    configure.add_argument("--base-url", default=None)
    configure.set_defaults(func=cmd_configure_judge)

    ev = sub.add_parser("evaluate", help="run + grade new ids + re-score all")
    ev.add_argument("--systems", default="all")
    ev.add_argument("--k", type=int, default=None)
    ev.add_argument("--only", default=None)
    ev.add_argument("--limit", type=int, default=None)
    ev.add_argument("--batch-size", type=int, default=5)
    ev.add_argument("--concurrency", type=int, default=1)
    ev.add_argument("--force-judge", action="store_true")
    ev.add_argument("--allow-unverified", action="store_true")
    ev.set_defaults(func=cmd_evaluate)

    grade = sub.add_parser("grade", help="grade unjudged ids from existing runs")
    grade.add_argument("--run", action="append", default=None)
    grade.add_argument("--only", default=None)
    grade.add_argument("--limit", type=int, default=None)
    grade.add_argument("--batch-size", type=int, default=5)
    grade.add_argument("--concurrency", type=int, default=1)
    grade.add_argument("--force-judge", action="store_true")
    grade.add_argument("--allow-unverified", action="store_true")
    grade.set_defaults(func=cmd_grade)

    score = sub.add_parser("score", help="re-score saved runs against current qrels")
    score.add_argument("--run", action="append", default=None)
    score.add_argument("--strict", action="store_true")
    score.set_defaults(func=cmd_score)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.func(args) or 0)
    except (DatasetError, LLMError, JudgeError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
