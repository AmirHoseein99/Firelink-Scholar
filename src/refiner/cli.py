from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import refine_one

DEFAULT_CLEANED = Path("data/cleaned")
DEFAULT_STRUCTURED = Path("data/structured")
DEFAULT_REPORT = Path("data/reports/structured/refine-report.json")


def iter_cleaned_files(cleaned_root: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(cleaned_root.rglob("*.json")):
        rel = p.relative_to(cleaned_root).as_posix()
        if rel.startswith("reports/"):
            continue
        if rel.startswith("tables/"):
            continue
        if rel.startswith("uncategorized/"):
            continue
        out.append(p)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean wiki noise from each top-level string / list-of-strings field.")
    parser.add_argument("--cleaned-root", type=Path, default=DEFAULT_CLEANED)
    parser.add_argument("--structured-root", type=Path, default=DEFAULT_STRUCTURED)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    cleaned_root = args.cleaned_root.resolve()
    structured_root = args.structured_root.resolve()

    if not cleaned_root.exists():
        print(f"cleaned root not found: {cleaned_root}")
        return 2

    files = iter_cleaned_files(cleaned_root)
    if args.only:
        only = set(args.only)
        files = [p for p in files if p.relative_to(cleaned_root).as_posix() in only]

    structured_root.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    totals = {"ok": 0, "skipped_existing": 0, "malformed": 0}
    failures: list[dict] = []

    for src in files:
        try:
            result = refine_one(
                source_path=src,
                cleaned_root=cleaned_root,
                structured_root=structured_root,
                force=args.force,
            )
        except (OSError, json.JSONDecodeError, ValueError) as e:
            totals["malformed"] += 1
            failures.append({"source_path": str(src), "error": str(e)})
            continue
        totals[result["status"]] = totals.get(result["status"], 0) + 1

    with args.report.open("w", encoding="utf-8") as f:
        json.dump({"totals": totals, "failures": failures}, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(
        f"refined ok={totals.get('ok', 0)} "
        f"malformed={totals['malformed']} "
        f"skipped={totals.get('skipped_existing', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
