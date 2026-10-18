"""Load and validate the pinned benchmark definition (``eval/dataset.json``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eval import paths
from eval.llm import resolve_backend
from eval.prompts_answer import ANSWER_PROMPT_SHA256
from eval.prompts_grade import GRADE_PROMPT_SHA256
from eval.storage import (
    load_json,
    now_iso,
    save_json,
    sha256_file_or_empty,
    sha256_json,
)

SCHEMA_VERSION = 1
REQUIRED_KEYS = (
    "schema_version",
    "questions",
    "claims",
    "answer_builder",
    "judge",
    "retrieval",
    "metrics",
    "qrels",
    "chunking",
)


class DatasetError(RuntimeError):
    pass


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else paths.REPO_ROOT / path


def questions_path(dataset: dict[str, Any]) -> Path:
    return _resolve(dataset["questions"])


def qrels_path(dataset: dict[str, Any]) -> Path:
    return _resolve(dataset["qrels"]["path"])


def build_dataset(
    *,
    questions_path_: Path = paths.QUESTIONS,
    qrels: dict[str, Any] | None = None,
    qrels_path_: Path = paths.QRELS,
) -> dict[str, Any]:
    """Build the canonical dataset definition from current settings/prompts.

    The judge is deliberately left unconfigured: no answer may be graded until
    a judge model is chosen explicitly with ``configure_judge``.
    """
    answer_backend = resolve_backend("freellm")
    qrels_data = qrels if qrels is not None else load_json(qrels_path_, default={}) or {}
    n_graded = sum(len(entry.get("judgments", {})) for entry in qrels_data.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": now_iso(),
        "questions": paths.relative(questions_path_),
        "questions_sha256": sha256_file_or_empty(questions_path_),
        "claims": {"max_single_game": 8, "max_synthesis": 12},
        "answer_builder": {
            "backend": answer_backend.name,
            "base_url": answer_backend.base_url,
            "model": answer_backend.model or "unset",
            "prompt_sha256": ANSWER_PROMPT_SHA256,
        },
        "judge": {
            "configured": False,
            "backend": "",
            "base_url": "",
            "model": "",
            "temperature": 0,
            "prompt_sha256": GRADE_PROMPT_SHA256,
        },
        "retrieval": {"systems": ["dense", "enhanced", "bm25", "hybrid"], "k": 10},
        "metrics": {"ks": [5, 10, 20], "min_grade": 2},
        "qrels": {
            "path": paths.relative(qrels_path_),
            "sha256": sha256_json(qrels_data) if qrels_data else "",
            "n_questions": len(qrels_data),
            "n_graded": n_graded,
        },
        "chunking": {"chunk_size": 1000, "chunk_overlap": 200},
    }


def validate_dataset(dataset: dict[str, Any]) -> None:
    for key in REQUIRED_KEYS:
        if key not in dataset:
            raise DatasetError(f"dataset.json is missing required key: {key}")
    if dataset["schema_version"] != SCHEMA_VERSION:
        raise DatasetError(
            f"unsupported dataset schema_version: {dataset['schema_version']}"
        )
    judge = dataset["judge"]
    if not judge.get("prompt_sha256"):
        raise DatasetError("dataset.json judge.prompt_sha256 is empty")
    if judge.get("configured") and not judge.get("model"):
        raise DatasetError("dataset.json judge is configured but judge.model is empty")
    if not dataset["answer_builder"].get("prompt_sha256"):
        raise DatasetError("dataset.json answer_builder.prompt_sha256 is empty")


def load_dataset(path: Path = paths.DATASET) -> dict[str, Any]:
    dataset = load_json(Path(path))
    if dataset is None:
        raise DatasetError(
            f"{path} does not exist; restore or regenerate the benchmark "
            "definition before running eval commands."
        )
    validate_dataset(dataset)
    return dataset


def ensure_dataset(path: Path = paths.DATASET) -> dict[str, Any]:
    if Path(path).exists():
        return load_dataset(path)
    dataset = build_dataset()
    save_json(path, dataset)
    return dataset


def refresh_qrels_metadata(dataset: dict[str, Any], qrels: dict[str, Any]) -> dict[str, Any]:
    refreshed = dict(dataset)
    refreshed["qrels"] = dict(dataset["qrels"])
    refreshed["qrels"]["sha256"] = sha256_json(qrels) if qrels else ""
    refreshed["qrels"]["n_questions"] = len(qrels)
    refreshed["qrels"]["n_graded"] = sum(
        len(entry.get("judgments", {})) for entry in qrels.values()
    )
    return refreshed


def save_dataset_with_qrels(
    dataset: dict[str, Any],
    qrels: dict[str, Any],
    path: Path = paths.DATASET,
) -> dict[str, Any]:
    """Persist refreshed qrels metadata into dataset.json."""
    refreshed = refresh_qrels_metadata(dataset, qrels)
    save_json(path, refreshed)
    return refreshed


def judge_configured(dataset: dict[str, Any]) -> bool:
    judge = dataset.get("judge") or {}
    return bool(judge.get("configured", judge.get("model")))


def judge_model(dataset: dict[str, Any]) -> str:
    return str(dataset["judge"].get("model") or "")


def judge_backend_name(dataset: dict[str, Any]) -> str:
    return str(dataset["judge"].get("backend") or "")


def configure_judge(
    dataset: dict[str, Any],
    *,
    model: str,
    backend: str = "freellm",
    base_url: str | None = None,
    path: Path = paths.DATASET,
) -> dict[str, Any]:
    """Pin a judge model in dataset.json. Does not grade anything."""
    if not model.strip():
        raise DatasetError("judge model must not be empty")
    resolved = resolve_backend(backend)
    judge = dict(dataset["judge"])
    judge.update(
        {
            "configured": True,
            "backend": resolved.name,
            "base_url": base_url or resolved.base_url,
            "model": model.strip(),
            "temperature": 0,
            "prompt_sha256": GRADE_PROMPT_SHA256,
        }
    )
    updated = dict(dataset)
    updated["judge"] = judge
    save_json(path, updated)
    return updated


def answer_model(dataset: dict[str, Any]) -> str:
    return str(dataset["answer_builder"]["model"])


def answer_backend_name(dataset: dict[str, Any]) -> str:
    return str(dataset["answer_builder"].get("backend", "freellm"))


def metric_settings(dataset: dict[str, Any]) -> tuple[list[int], int]:
    metrics = dataset["metrics"]
    return list(metrics["ks"]), int(metrics["min_grade"])


def claim_limits(dataset: dict[str, Any], game: str) -> tuple[int, int]:
    claims = dataset["claims"]
    if game == "All":
        return 3, int(claims["max_synthesis"])
    return 3, int(claims["max_single_game"])


def retrieval_systems(dataset: dict[str, Any]) -> list[str]:
    return list(dataset["retrieval"]["systems"])


def default_k(dataset: dict[str, Any]) -> int:
    return int(dataset["retrieval"]["k"])
