"""Canonical filesystem paths for the retrieval benchmark."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent

QUESTIONS = EVAL_DIR / "questions.json"
ANSWERS = EVAL_DIR / "answers.json"
QRELS = EVAL_DIR / "qrels.json"
RUNS = EVAL_DIR / "runs"
REPORTS = EVAL_DIR / "reports"
DATASET = EVAL_DIR / "dataset.json"
STRUCTURED_DIR = REPO_ROOT / "data" / "structured"
ARCHIVE_DIR = EVAL_DIR / "archive"


def relative(path: Path) -> str:
    """Return a repo-relative POSIX path when possible."""
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()
