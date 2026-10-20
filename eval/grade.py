"""Run-driven LLM grading: grade only newly retrieved, unjudged chunk ids."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from eval import paths
from eval.corpus import fetch_chunks
from eval.dataset import judge_configured, judge_model
from eval.llm import Backend, chat, extract_json
from eval.prompts_grade import GRADE_PROMPT_SHA256, build_grade_messages
from eval.runs import all_run_ids, load_run, resolve_run_paths
from eval.storage import load_json, now_iso, save_json, sha256_json

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 5


class JudgeError(RuntimeError):
    pass


def empty_entry(*, graded_by: str, prompt_sha256: str) -> dict[str, Any]:
    return {
        "judgments": {},
        "supports_claims": {},
        "other_relevant": {},
        "grade_counts": {},
        "n_judgments": 0,
        "graded_by": graded_by,
        "prompt_sha256": prompt_sha256,
        "updated_at": now_iso(),
    }


def _recount(entry: dict[str, Any]) -> None:
    counts = {"0": 0, "1": 0, "2": 0, "3": 0}
    for grade in entry["judgments"].values():
        counts[str(int(grade))] = counts.get(str(int(grade)), 0) + 1
    entry["grade_counts"] = counts
    entry["n_judgments"] = len(entry["judgments"])
    entry["updated_at"] = now_iso()


def merge_judgment(
    entry: dict[str, Any],
    doc_id: str,
    grade: int,
    claims: Iterable[str],
    other_relevant: bool,
) -> bool:
    """Append one judgment; existing judgments are never overwritten."""
    if doc_id in entry["judgments"]:
        return False
    grade = int(grade)
    if grade not in (0, 1, 2, 3):
        return False
    entry["judgments"][doc_id] = grade
    if claims:
        entry["supports_claims"][doc_id] = list(dict.fromkeys(claims))
    if other_relevant:
        entry["other_relevant"][doc_id] = True
    _recount(entry)
    return True


def parse_grades(
    raw: str,
    batch: list[dict[str, Any]],
    claim_ids: set[str],
) -> tuple[dict[int, dict[str, Any]], list[int]]:
    """Parse a grading response into ``{candidate_number: grade_record}``.

    Out-of-range numbers, invalid grades, unknown claim ids, and duplicate
    numbers are discarded (first occurrence wins). Returns the parsed records
    and the candidate numbers that were not mapped.
    """
    payload = extract_json(raw)
    if isinstance(payload, dict):
        raw_grades = payload.get("grades")
    elif isinstance(payload, list):
        raw_grades = payload
    else:
        raw_grades = None
    if not isinstance(raw_grades, list):
        return {}, list(range(1, len(batch) + 1))

    parsed: dict[int, dict[str, Any]] = {}
    for item in raw_grades:
        if not isinstance(item, dict):
            continue
        try:
            number = int(item.get("n"))
        except (TypeError, ValueError):
            continue
        if number < 1 or number > len(batch) or number in parsed:
            continue
        try:
            grade = int(item.get("grade"))
        except (TypeError, ValueError):
            continue
        if grade not in (0, 1, 2, 3):
            continue
        claims = item.get("claims")
        if not isinstance(claims, list):
            claims = []
        valid_claims = [
            str(claim) for claim in claims if str(claim) in claim_ids
        ]
        if grade <= 1:
            valid_claims = []
        other_relevant = bool(item.get("other_relevant", False)) and grade >= 2
        parsed[number] = {
            "grade": grade,
            "claims": list(dict.fromkeys(valid_claims)),
            "other_relevant": other_relevant,
        }

    unmapped = [number for number in range(1, len(batch) + 1) if number not in parsed]
    return parsed, unmapped


def _batches(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def grade_candidates(
    question: dict[str, Any],
    answer: dict[str, Any],
    chunks: list[dict[str, Any]],
    *,
    backend: Backend,
    call: Callable[..., str] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    retries: int = 1,
    concurrency: int = 1,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Grade candidate chunks, retrying unmapped candidates once.

    Returns ``(grades_by_document_id, unparsed_candidates)``.
    """
    invoke = call or chat
    claim_ids = {str(claim["id"]) for claim in answer.get("claims", [])}
    entities = [
        str(entity.get("name") if isinstance(entity, dict) else entity)
        for entity in question.get("expected_entities", [])
    ]
    results: dict[str, dict[str, Any]] = {}

    def run_batch(batch: list[dict[str, Any]]) -> tuple[dict[int, dict[str, Any]], list[int]]:
        messages = build_grade_messages(
            question=question["question"],
            game=question["game"],
            entities=entities,
            claims=answer.get("claims", []),
            candidates=batch,
        )
        logger.debug(
            "%s: grading %d candidate(s) with %s",
            question.get("id", "?"),
            len(batch),
            backend.model,
        )
        try:
            raw = invoke(messages, backend=backend, temperature=0.0)
            return parse_grades(raw, batch, claim_ids)
        except Exception as exc:  # noqa: BLE001 - keep ungraded for the tripwire
            logger.warning(
                "%s: grading request failed for %d candidate(s): %s",
                question.get("id", "?"),
                len(batch),
                exc,
            )
            return {}, list(range(1, len(batch) + 1))

    pending = list(chunks)
    for _attempt in range(retries + 1):
        still_pending: list[dict[str, Any]] = []
        batches = _batches(pending, batch_size)
        if concurrency > 1 and len(batches) > 1:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                batch_results = list(pool.map(run_batch, batches))
        else:
            batch_results = [run_batch(batch) for batch in batches]
        for batch, (parsed, unmapped) in zip(batches, batch_results):
            for number, record in parsed.items():
                doc_id = batch[number - 1]["document_id"]
                results[doc_id] = record
            for number in unmapped:
                still_pending.append(batch[number - 1])
        pending = still_pending
        if not pending:
            break
        logger.debug(
            "%s: %d candidate(s) unparsed after attempt %d; retrying",
            question.get("id", "?"),
            len(pending),
            _attempt + 1,
        )
    return results, pending


def _check_judge(
    dataset: dict[str, Any],
    backend: Backend,
    qrels: dict[str, Any],
    *,
    force_judge: bool,
) -> str:
    if not judge_configured(dataset):
        raise JudgeError(
            "no judge is configured in eval/dataset.json; choose one with "
            "`python -m eval configure-judge --model <model>` before grading"
        )
    pinned = judge_model(dataset)
    if backend.model != pinned and not force_judge:
        raise JudgeError(
            f"judge model mismatch: backend={backend.model!r}, "
            f"dataset={pinned!r}; pass --force-judge to override"
        )
    if backend.model != pinned:
        pinned = backend.model
    for qid, entry in qrels.items():
        existing = entry.get("graded_by")
        if existing and existing != pinned and not force_judge:
            raise JudgeError(
                f"qrels contain a different judge for {qid}: {existing!r} != "
                f"{pinned!r}; pass --force-judge to override"
            )
    return pinned


def grade_run_payloads(
    run_payloads: Iterable[dict[str, Any]],
    *,
    questions: Iterable[dict[str, Any]],
    answers: dict[str, Any],
    qrels: dict[str, Any],
    dataset: dict[str, Any],
    backend: Backend,
    qrels_path: Path = paths.QRELS,
    chunk_fetcher: Callable[[Iterable[str]], dict[str, dict[str, Any]]] = fetch_chunks,
    call: Callable[..., str] | None = None,
    only: set[str] | None = None,
    limit: int | None = None,
    force_judge: bool = False,
    allow_unverified: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    retries: int = 1,
    concurrency: int = 1,
) -> dict[str, Any]:
    """Grade the union of retrieved ids that are not already in qrels."""
    pinned = _check_judge(dataset, backend, qrels, force_judge=force_judge)
    prompt_hash = str(dataset["judge"]["prompt_sha256"])
    if prompt_hash != GRADE_PROMPT_SHA256 and not force_judge:
        raise JudgeError(
            "judge prompt hash mismatch: dataset has "
            f"{prompt_hash!r}, code has {GRADE_PROMPT_SHA256!r}; "
            "pass --force-judge to override"
        )

    question_list = [
        question
        for question in questions
        if (not only or str(question["id"]) in only)
    ]
    if limit is not None:
        question_list = question_list[:limit]
    union = all_run_ids(run_payloads)
    logger.info(
        "Grading with %s via %s: %d question(s), %d retrieved id(s), %d already in qrels",
        pinned,
        backend.name,
        len(question_list),
        sum(len(ids) for ids in union.values()),
        sum(len(entry.get("judgments", {})) for entry in qrels.values()),
    )

    summary: dict[str, Any] = {
        "graded": 0,
        "skipped": 0,
        "missing": [],
        "unparsed": [],
        "questions": {},
    }
    for question in question_list:
        qid = str(question["id"])
        retrieved = union.get(qid, [])
        entry = qrels.get(qid)
        if entry is None:
            entry = empty_entry(graded_by=pinned, prompt_sha256=prompt_hash)
            qrels[qid] = entry
        elif force_judge:
            entry["graded_by"] = pinned
            entry["prompt_sha256"] = prompt_hash

        judged = entry.setdefault("judgments", {})
        fresh = [doc_id for doc_id in retrieved if doc_id not in judged]
        summary["skipped"] += len(retrieved) - len(fresh)
        if not fresh:
            logger.debug(
                "%s: no new ids to grade (%d retrieved, %d already judged)",
                qid,
                len(retrieved),
                len(retrieved) - len(fresh),
            )
            if not entry.get("judgments"):
                qrels.pop(qid, None)
            continue

        logger.info(
            "%s: grading %d new id(s) (%d already judged)",
            qid,
            len(fresh),
            len(retrieved) - len(fresh),
        )
        answer = answers.get(qid)
        if not answer:
            raise JudgeError(f"{qid}: missing answer key")
        if answer.get("status") != "verified" and not allow_unverified:
            raise JudgeError(
                f"{qid}: answer key status is {answer.get('status')!r}; "
                "pass --allow-unverified to grade anyway"
            )

        chunks_by_id = chunk_fetcher(fresh)
        missing = [doc_id for doc_id in fresh if doc_id not in chunks_by_id]
        chunks = [chunks_by_id[doc_id] for doc_id in fresh if doc_id in chunks_by_id]
        summary["missing"].extend(
            {"question": qid, "document_id": doc_id} for doc_id in missing
        )
        if missing:
            logger.warning(
                "%s: %d retrieved id(s) missing from Qdrant",
                qid,
                len(missing),
            )
        if not chunks:
            if not entry.get("judgments"):
                qrels.pop(qid, None)
            continue

        grades, unresolved = grade_candidates(
            question,
            answer,
            chunks,
            backend=backend,
            call=call,
            batch_size=batch_size,
            retries=retries,
            concurrency=concurrency,
        )
        added = 0
        for chunk in chunks:
            record = grades.get(chunk["document_id"])
            if record is None:
                continue
            if merge_judgment(
                entry,
                chunk["document_id"],
                record["grade"],
                record["claims"],
                record["other_relevant"],
            ):
                added += 1
        summary["graded"] += added
        summary["unparsed"].extend(
            {
                "question": qid,
                "document_id": chunk["document_id"],
            }
            for chunk in unresolved
        )
        summary["questions"][qid] = {
            "retrieved": len(retrieved),
            "fresh": len(fresh),
            "graded": added,
            "missing": len(missing),
            "unparsed": len(unresolved),
        }
        logger.debug(
            "%s: graded %d, missing %d, unparsed %d",
            qid,
            added,
            len(missing),
            len(unresolved),
        )
        if unresolved:
            logger.warning(
                "%s: %d candidate(s) left unparsed after retries",
                qid,
                len(unresolved),
            )
        if not entry.get("judgments"):
            qrels.pop(qid, None)
        if added:
            save_json(qrels_path, qrels)

    if qrels:
        save_json(qrels_path, qrels)
    logger.info(
        "Grading done: %d graded, %d skipped, %d missing, %d unparsed; qrels now cover %d question(s)",
        summary["graded"],
        summary["skipped"],
        len(summary["missing"]),
        len(summary["unparsed"]),
        len(qrels),
    )
    return summary


def grade_run_files(
    run_files: Iterable[str | Path],
    *,
    questions: Iterable[dict[str, Any]],
    answers: dict[str, Any],
    qrels: dict[str, Any],
    dataset: dict[str, Any],
    backend: Backend,
    runs_dir: Path = paths.RUNS,
    **kwargs: Any,
) -> dict[str, Any]:
    resolved = resolve_run_paths([str(path) for path in run_files], runs_dir=runs_dir)
    payloads = [load_run(path) for path in resolved]
    return grade_run_payloads(
        payloads,
        questions=questions,
        answers=answers,
        qrels=qrels,
        dataset=dataset,
        backend=backend,
        **kwargs,
    )


def load_answers(path: Path = paths.ANSWERS) -> dict[str, Any]:
    return load_json(path, default={}) or {}


def load_qrels(path: Path = paths.QRELS) -> dict[str, Any]:
    return load_json(path, default={}) or {}


def qrels_hash(qrels: dict[str, Any]) -> str:
    return sha256_json(qrels) if qrels else ""
