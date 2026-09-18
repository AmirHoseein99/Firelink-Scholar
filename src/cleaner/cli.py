from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .clean import clean_one
from .io import write_report
from .models import CleaningReport
from .registry import Registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.cleaner")
    parser.add_argument("--input", default="data/raw")
    parser.add_argument("--output", default="data/cleaned")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--report",
        default=None,
        help="Path to report.json; defaults to <output>/report.json",
    )
    args = parser.parse_args(argv)

    in_root = Path(args.input).resolve()
    out_root = Path(args.output).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    registry = Registry.default()
    report = CleaningReport()

    paths = sorted(p for p in in_root.rglob("*.json") if p.is_file())

    # for i, chunk in enumerate(chunks):
    if args.limit is not None:
        paths = paths[: args.limit]

    for path in paths:
        clean_one(
            source_path=path,
            source_root=in_root,
            output_root=out_root,
            registry=registry,
            report=report,
            force=args.force,
        )

    report_path = (
        Path(args.report) if args.report else out_root / "reports" / "report.json"
    )
    write_report(report_path, report)
    print(
        f"cleaned {report.totals.pages} pages; report at {report_path}", file=sys.stderr
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
