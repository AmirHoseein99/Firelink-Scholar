"""Interactive CLI for grading retrieval candidates on the 0..3 scale.

Run:
    python -m eval.annotate --questions eval/questions.json \
        --candidates eval/v3/candidates --out eval/v3/judgments.json

Manual-audit tool: writes plain per-question qrels entries; for the merged
v3 schema (sources counters, probe holdout) prefer ``eval.llm_grade`` and
``eval.build_seeds``. Resumable: already-graded questions are skipped on
restart. State is flushed to disk after every question (atomic write) so
Ctrl-C never loses work.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

VALID_GRADES = {"0", "1", "2", "3", "s", "b", "a", "q"}


def _load_qrels(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"  warning: could not parse {path}: {e}, starting empty", file=sys.stderr)
        return {}


def _save_qrels(path: Path, qrels: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(qrels, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _format_candidate(idx: int, total: int, cand: dict) -> str:
    sources = "+".join(cand["sources"])
    scores = []
    if cand.get("dense_score") is not None:
        scores.append(f"dense={cand['dense_score']:.3f}")
    if cand.get("bm25_score") is not None:
        scores.append(f"bm25={cand['bm25_score']:.2f}")
    score_str = " ".join(scores) if scores else ""
    header = (
        f"  {idx + 1:>3}/{total}  [{sources}]  {cand['title']!r}  "
        f"· {cand['entity_type']}/{cand['section']}"
    )
    if score_str:
        header += f"  ({score_str})"
    snippet = cand.get("snippet", "").replace("\n", " ")
    return f"{header}\n        snippet: {snippet}"


def _prompt(cand: dict, idx: int, total: int) -> str:
    print()
    print(_format_candidate(idx, total, cand))
    print(
        "        relevance [3/2/1/0/s(kip)/b(ack)/a(dd uuid)/q(uit)] > ",
        end="",
        flush=True,
    )
    return input().strip().lower()


def grade_question(q: dict, candidates: list[dict], qrels: dict) -> dict[str, int]:
    judgments: dict[str, int] = {}
    i = 0
    history: list[tuple[int, str]] = []
    total = len(candidates)
    print()
    print(f"[{q['id']}] {q['category'].title()} — {q['question']}")
    print("─" * 70)

    while i < total:
        cand = candidates[i]
        raw = _prompt(cand, i, total)
        if raw in VALID_GRADES:
            choice = raw
        elif raw == "":
            choice = "s"
        else:
            print(f"        invalid input {raw!r}, treating as 's'")
            choice = "s"

        if choice == "q":
            print("        quit requested, flushing state")
            break
        if choice == "s":
            i += 1
            continue
        if choice == "b":
            if not history:
                print("        at first candidate, cannot go back")
                continue
            prev_i, prev_doc = history.pop()
            judgments.pop(prev_doc, None)
            i = prev_i
            continue
        if choice == "a":
            print("        paste chunk UUID > ", end="", flush=True)
            new_uuid = input().strip()
            try:
                grade = int(input("        grade (0..3) > ").strip())
            except ValueError:
                print("        invalid grade, skipping")
                continue
            if not new_uuid or grade not in (0, 1, 2, 3):
                print("        invalid uuid/grade, skipping")
                continue
            judgments[new_uuid] = grade
            history.append((i, new_uuid))
            i += 1
            continue

        grade = int(choice)
        judgments[cand["document_id"]] = grade
        history.append((i, cand["document_id"]))
        i += 1

    return judgments


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--questions",
        type=Path,
        default=REPO_ROOT / "eval" / "questions.json",
    )
    ap.add_argument(
        "--candidates",
        type=Path,
        default=REPO_ROOT / "eval" / "v3" / "candidates",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "eval" / "v3" / "judgments.json",
    )
    ap.add_argument("--start", type=str, default=None,
                    help="question id to resume from (e.g. q021)")
    args = ap.parse_args()

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    qrels = _load_qrels(args.out)
    print(f"Loaded {len(qrels)} already-graded questions from {args.out}")

    started = args.start is None
    for q in questions:
        if not started:
            if q["id"] == args.start:
                started = True
            else:
                continue
        if q["id"] in qrels:
            print(f"  skipping {q['id']} (already graded)")
            continue
        cand_path = args.candidates / f"{q['id']}.json"
        if not cand_path.exists():
            print(f"  warning: no candidates for {q['id']} at {cand_path}, skipping")
            continue
        candidates = json.loads(cand_path.read_text(encoding="utf-8"))
        if not candidates:
            print(f"  warning: {q['id']} has empty candidate pool, skipping")
            continue
        judgments = grade_question(q, candidates, qrels)
        qrels[q["id"]] = {
            "question": q["question"],
            "game": q.get("game", "Dark Souls"),
            "category": q["category"],
            "graded_at": datetime.now(UTC).isoformat(
                timespec="seconds"
            ).replace("+00:00", "Z"),
            "judgments": judgments,
        }
        _save_qrels(args.out, qrels)
        n_relevant = sum(1 for g in judgments.values() if g > 0)
        print(
            f"  saved {q['id']}: {len(judgments)} judgments "
            f"({n_relevant} > 0)"
        )

    print(f"\nDone. Qrels at {args.out}")


if __name__ == "__main__":
    main()
