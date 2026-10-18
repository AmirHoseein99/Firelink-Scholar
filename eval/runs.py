"""Retrieval runs: call systems, save run files, count unjudged ids."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from eval import paths
from eval.questions import select_question_subset
from eval.storage import (
    load_json,
    now_iso,
    save_json,
    sha256_file_or_empty,
    sha256_json,
)

SYSTEM_FNS = {
    "dense": "contextfromvectors",
    "enhanced": "contextfromvectors_enhanced",
    "bm25": "contextfromvectors_bm25",
    "hybrid": "contextfromvectors_hybrid",
}


def run_filename(system: str, k: int) -> str:
    return f"{system}_k{int(k)}.run.json"


def run_path(system: str, k: int, runs_dir: Path = paths.RUNS) -> Path:
    return Path(runs_dir) / run_filename(system, k)


def run_name(path: Path) -> str:
    name = Path(path).name
    return name[: -len(".run.json")] if name.endswith(".run.json") else Path(path).stem


def _document_id(item: Any) -> tuple[str | None, str, float]:
    if isinstance(item, dict):
        doc_id = item.get("document_id")
        return doc_id, str(item.get("title", "")), float(item.get("score", 0.0))
    document, score = item[0], item[1]
    if isinstance(document, dict):
        return (
            document.get("document_id"),
            str(document.get("title", "")),
            float(document.get("score", score)),
        )
    if isinstance(document, str):
        return document, "", float(score)
    metadata = getattr(document, "metadata", None) or {}
    doc_id = metadata.get("document_id")
    title = metadata.get("title", "")
    return doc_id, str(title), float(score)


def normalize_retrieval(results: Iterable[Any]) -> list[dict[str, Any]]:
    """Normalize retriever output to ``{document_id, title, score}`` records.

    Accepts LangChain ``(Document, score)`` pairs, ``(document_id, score)``
    pairs, and already-normalized dicts. Ids without a ``document_id`` are
    dropped rather than fabricated.
    """
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in results:
        doc_id, title, score = _document_id(item)
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        normalized.append({"document_id": doc_id, "title": title, "score": score})
    return normalized


def collect_run(
    system: str,
    questions: Iterable[dict[str, Any]],
    k: int,
    *,
    retrieve_fn: Callable[..., list[Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    if retrieve_fn is None:
        from eval.systems import get_retriever

        retrieve_fn = get_retriever(system)
    run: dict[str, list[dict[str, Any]]] = {}
    for question in questions:
        results = retrieve_fn(query=question["question"], k=k)
        run[str(question["id"])] = normalize_retrieval(results)
    return run


def unjudged_counts(
    run: dict[str, list[dict[str, Any]]],
    qrels: dict[str, Any],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for qid, results in run.items():
        judged = (qrels.get(qid) or {}).get("judgments", {})
        counts[qid] = sum(
            1 for item in results if item.get("document_id") not in judged
        )
    return counts


def total_unjudged(
    run: dict[str, list[dict[str, Any]]],
    qrels: dict[str, Any],
) -> int:
    return sum(unjudged_counts(run, qrels).values())


def build_run_payload(
    system: str,
    k: int,
    run: dict[str, list[dict[str, Any]]],
    qrels: dict[str, Any],
    *,
    questions_path: Path = paths.QUESTIONS,
) -> dict[str, Any]:
    unjudged = total_unjudged(run, qrels)
    return {
        "config": {
            "system": system,
            "function": SYSTEM_FNS.get(system, system),
            "k": int(k),
            "min_grade": 2,
            "qrels": "eval/qrels.json",
            "created_at": now_iso(),
            "qrels_sha256_at_creation": sha256_json(qrels) if qrels else "",
            "questions_sha256": sha256_file_or_empty(questions_path),
        },
        "diagnostics": {"n_unresolved": 0, "ambiguous_contents": []},
        "run": run,
        "n_retrieved": sum(len(results) for results in run.values()),
        "n_unjudged": unjudged,
        "n_unjudged_total": unjudged,
        "metrics": {},
    }


def save_run(
    system: str,
    k: int,
    run: dict[str, list[dict[str, Any]]],
    qrels: dict[str, Any],
    *,
    runs_dir: Path = paths.RUNS,
    questions_path: Path = paths.QUESTIONS,
    merge: bool = False,
) -> Path:
    path = run_path(system, k, runs_dir)
    if merge and path.exists():
        previous = load_json(path, default={}) or {}
        merged = dict(previous.get("run") or {})
        merged.update(run)
        run = merged
    payload = build_run_payload(
        system, k, run, qrels, questions_path=questions_path
    )
    save_json(path, payload)
    return path


def load_run(path: Path) -> dict[str, Any]:
    data = load_json(Path(path))
    if data is None:
        raise FileNotFoundError(path)
    return data


def iter_run_files(runs_dir: Path = paths.RUNS) -> list[Path]:
    return sorted(Path(runs_dir).glob("*.run.json"))


def resolve_run_paths(
    requested: Iterable[str] | None,
    *,
    runs_dir: Path = paths.RUNS,
) -> list[Path]:
    if not requested:
        return iter_run_files(runs_dir)
    resolved: list[Path] = []
    for value in requested:
        path = Path(value)
        if path.exists():
            resolved.append(path)
            continue
        candidate = Path(runs_dir) / value
        if candidate.exists():
            resolved.append(candidate)
            continue
        candidate = Path(runs_dir) / f"{value}.run.json"
        if candidate.exists():
            resolved.append(candidate)
            continue
        raise FileNotFoundError(f"run file not found: {value}")
    return resolved


def run_questions_for_systems(
    systems: Iterable[str],
    questions: Iterable[dict[str, Any]],
    k: int,
    *,
    retriever: Callable[[str], Callable[..., list[Any]]] | None = None,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    if retriever is None:
        from eval.systems import get_retriever

        retriever = get_retriever
    return {
        system: collect_run(system, questions, k, retrieve_fn=retriever(system))
        for system in systems
    }


def questions_for_run(
    run_payload: dict[str, Any],
    questions: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Questions present in a run, preserving question-file order."""
    present = set((run_payload.get("run") or {}).keys())
    return [question for question in questions if str(question["id"]) in present]


def all_run_ids(run_payloads: Iterable[dict[str, Any]]) -> dict[str, list[str]]:
    """Union of retrieved ids per question across run payloads, order-stable."""
    union: dict[str, list[str]] = {}
    for payload in run_payloads:
        for qid, results in (payload.get("run") or {}).items():
            bucket = union.setdefault(qid, [])
            known = set(bucket)
            for item in results:
                doc_id = item.get("document_id")
                if doc_id and doc_id not in known:
                    bucket.append(doc_id)
                    known.add(doc_id)
    return union


def subset_questions(
    questions: Iterable[dict[str, Any]],
    *,
    only: set[str] | None = None,
    limit: int | None = None,
    start: str | None = None,
) -> list[dict[str, Any]]:
    return select_question_subset(
        list(questions), only=only, limit=limit, start=start
    )
