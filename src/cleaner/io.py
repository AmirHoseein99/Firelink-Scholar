from __future__ import annotations

import json
from pathlib import Path

from .models import CleaningReport, nfc


def write_cleaned_page(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        + "\n"
    )
    path.write_text(text, encoding="utf-8")


def write_report(path: Path, report: CleaningReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.model_dump(mode="json")
    text = (
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        + "\n"
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def normalize_path(path: str) -> str:
    return nfc(path).rstrip("\n")
