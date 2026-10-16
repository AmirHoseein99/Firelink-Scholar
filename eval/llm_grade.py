"""Anchored LLM grader for the v3 run-based judgment cycle.

Run:
    python -m eval.llm_grade --start q002 --limit 1      # smoke test one question
    python -m eval.llm_grade                              # full run, resumable

Requires NEURALWATT_API_KEY in .env (see .env.example). Grades candidate ids
from the per-question seed pools (``eval/v3/candidates`` by default) on the
0..3 rubric via the OpenAI-compatible chat endpoint configured by
``NEURALWATT_BASE_URL`` / ``NEURALWATT_MODEL``.

v3 semantics (id-level, anchored):
  - only ids WITHOUT a judgment are graded; re-runs grade just the delta, so
    failed batches are retried cheaply by re-running the same command;
  - every batch prompt embeds the question's gold anchor (human grade 3) and
    contrast anchor (human grade 0) from ``eval/v3/seeds/anchors.json`` plus
    the rubric line "merely mentions an expected entity = grade 0, not 1";
  - anchor ids are never graded; probe ids (human-judged holdout) ARE graded
    by the LLM, but their human grade stays authoritative in ``judgments`` —
    LLM probe grades are stored under ``probe_llm`` for agreement reports;
  - results merge into ``eval/v3/judgments.json``: human grades always win,
    LLM grades only fill unjudged ids; ``sources`` counters track provenance.

State is flushed atomically (tmp file + os.replace) after every question. By
default, batches within a question run concurrently on a single asyncio event
loop (httpx.AsyncClient + semaphore). ``--sync`` runs the same pipeline fully
synchronously (blocking httpx.Client + thread pool). Deliberately does NOT
import src.llm.client (noisy prints, SQLite-lock side effects).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config import settings

MAX_ATTEMPTS = 5  # per HTTP request (429/5xx/timeouts), exponential backoff
REQUEST_TIMEOUT = 120.0
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
SNIPPET_LEN = 1000
ANCHOR_SNIPPET_LEN = 700

GRADE_KEYS = ("grade", "relevance", "score", "g", "rating")
INDEX_KEYS = ("n", "index", "candidate", "num", "id", "doc")
GRADES_LIST_KEYS = ("grades", "results", "judgments", "relevance")

SYSTEM_PROMPT = (
    "You are a precise relevance annotator for a Dark Souls wiki retrieval "
    "evaluation. Grade each candidate chunk strictly by the given rubric and "
    "respond with only the requested JSON object."
)

RUBRIC_TEXT = """\
Relevance rubric (integer grades 0..3):
  3 — the chunk directly and authoritatively answers the question (contains
      the core fact(s) needed).
  2 — the chunk contains substantial supporting information about the asked
      entities/relationship, but is not the direct answer.
  1 — background that substantively involves an expected entity beyond a bare
      name-drop (discusses its role, properties, or events relevant to the
      question).
  0 — unrelated to the question. Merely mentioning or name-dropping an
      expected entity is grade 0, not 1.
Negative example: an upgrade/repair page that only says a weapon "cannot be
reinforced with twinkle titanite" tangentially mentions the item or mechanic
without answering anything — that is a 0, never a 3. Match on substance,
not on keyword overlap."""


def _render_anchor(label: str, grade: int, anchor: dict | None) -> list[str]:
    if not anchor:
        return []
    snippet = (anchor.get("snippet") or "").replace("\n", " ")[:ANCHOR_SNIPPET_LEN]
    return [
        (
            f"[ANCHOR {label} — grade {grade}] "
            f"title={anchor.get('title', '')!r} "
            f"section={anchor.get('section', '')!r}"
        ),
        f"    {snippet}",
        "",
    ]


def build_user_prompt(
    q: dict,
    batch: list[dict],
    anchors: dict | None = None,
) -> str:
    """Render one batch prompt: anchors + rubric + question + candidates."""
    anchors = anchors or {}
    entities = q.get("expected_entities") or []
    lines = [
        f"Question: {q['question']}",
        f"Expected entities: {', '.join(entities) if entities else '(none listed)'}",
        "",
    ]
    lines += _render_anchor("GOLD", 3, anchors.get("gold"))
    lines += _render_anchor("CONTRAST", 0, anchors.get("contrast"))
    lines += [
        RUBRIC_TEXT,
        "",
        "Candidates:",
    ]
    for i, cand in enumerate(batch, start=1):
        snippet = (cand.get("snippet") or "").replace("\n", " ")[:SNIPPET_LEN]
        lines.append(
            f"[{i}] title={cand.get('title', '')!r} | "
            f"section={cand.get('section', '')!r} | "
            f"entity_type={cand.get('entity_type', '')!r}"
        )
        lines.append(f"    {snippet}")
    lines += [
        "",
        (
            f"Grade every candidate [1]..[{len(batch)}] against the question "
            "(each independently, not relative to the other candidates)."
        ),
        (
            "Return ONLY the following JSON object — no markdown fences, no "
            "commentary, no extra keys:"
        ),
        (
            '{"grades": [{"n": 1, "grade": <0..3>}, {"n": 2, "grade": <0..3>}, '
            f"...]}} covering all {len(batch)} candidates"
        ),
    ]
    return "\n".join(lines)


def _as_int(value) -> int | None:
    """Best-effort int coercion for model-produced numbers (1, 1.0, "1")."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if re.fullmatch(r"-?\d+", s):
            return int(s)
        try:
            f = float(s)
        except ValueError:
            return None
        if f.is_integer():
            return int(f)
    return None


def _extract_json_payload(raw: str):
    """Pull the outermost JSON object/array out of a model reply."""
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s.strip())
    for attempt in (s,):
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            pass
    lo, hi = s.find("{"), s.rfind("}")
    if lo != -1 and hi > lo:
        try:
            return json.loads(s[lo : hi + 1])
        except json.JSONDecodeError:
            pass
    lo, hi = s.find("["), s.rfind("]")
    if lo != -1 and hi > lo:
        try:
            return json.loads(s[lo : hi + 1])
        except json.JSONDecodeError:
            return None
    return None


def _grades_list_from_payload(payload) -> list | None:
    """Accept the documented nesting plus tolerant variants."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in GRADES_LIST_KEYS:
            if key in payload and isinstance(payload[key], list):
                return payload[key]
        # {"grades": {"1": 2, "2": 0}} / {"1": 2, ...} mapping form
        for container in (
            payload.get("grades") if isinstance(payload.get("grades"), dict) else None,
            payload,
        ):
            if container and all(
                _as_int(k) is not None and _as_int(v) is not None
                for k, v in container.items()
            ):
                return [
                    {"n": k, "grade": v} for k, v in sorted(
                        container.items(), key=lambda kv: _as_int(kv[0])
                    )
                ]
    return None


def parse_grades(raw: str, batch_ids: list[str]) -> tuple[dict[str, int], int]:
    """Parse a batch reply into {doc_id: grade}.

    Returns (grades, n_discarded). Grades for candidate numbers outside the
    batch, unmappable entries, duplicates, and out-of-range values are
    discarded and counted. First occurrence of a duplicate wins.
    """
    payload = _extract_json_payload(raw)
    items = _grades_list_from_payload(payload)
    if items is None:
        return {}, 0
    grades: dict[str, int] = {}
    discarded = 0
    for item in items:
        n = grade = None
        if isinstance(item, dict):
            for key in INDEX_KEYS:
                if key in item:
                    n = _as_int(item[key])
                    if n is not None:
                        break
            for key in GRADE_KEYS:
                if key in item:
                    grade = _as_int(item[key])
                    if grade is not None:
                        break
        elif (
            isinstance(item, (list, tuple))
            and len(item) == 2
        ):
            n, grade = _as_int(item[0]), _as_int(item[1])
        if n is None or grade is None or not (0 <= grade <= 3):
            discarded += 1
            continue
        if not (1 <= n <= len(batch_ids)):
            discarded += 1
            continue
        doc_id = batch_ids[n - 1]
        if doc_id in grades:
            discarded += 1
            continue
        grades[doc_id] = grade
    return grades, discarded


class GradeError(Exception):
    """Non-retryable grading failure for a batch."""


RETRY_REMINDER = (
    "Your reply was not valid complete JSON. Return ONLY "
    'the JSON object {"grades": [{"n": <candidate number>, '
    '"grade": <0..3>}, ...]} with one entry per candidate, '
    "and nothing else."
)


def _base_messages(q: dict, batch: list[dict],
                   anchors: dict | None = None) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(q, batch, anchors)},
    ]


def _reminder_messages(messages: list[dict], content: str) -> list[dict]:
    return messages + [
        {"role": "assistant", "content": content},
        {"role": "user", "content": RETRY_REMINDER},
    ]


def _absorb_attempt(
    content: str,
    batch_ids: list[str],
    best: dict[str, int],
    qid: str,
) -> bool:
    """Parse one model reply into ``best`` (kept if larger).

    Returns True when the reply covered every batch candidate.
    """
    grades, discarded = parse_grades(content, batch_ids)
    if discarded:
        print(
            f"    note: {qid}: discarded {discarded} unparsable/"
            f"out-of-batch grade(s)",
            file=sys.stderr,
        )
    if len(grades) > len(best):
        best.clear()
        best.update(grades)
    return len(grades) == len(batch_ids)


async def _post_with_retry(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    payload: dict,
) -> dict:
    last_err = "never attempted"
    for attempt in range(MAX_ATTEMPTS):
        if attempt:
            delay = min(60.0, 2.0**attempt) + random.uniform(0.0, 1.0)
            await asyncio.sleep(delay)
        try:
            resp = await client.post(url, headers=headers, json=payload)
        except httpx.RequestError as e:
            last_err = f"request error: {e}"
            continue
        if resp.status_code in RETRYABLE_STATUS:
            last_err = f"HTTP {resp.status_code}"
            continue
        if resp.status_code >= 400:
            raise GradeError(
                f"HTTP {resp.status_code}: {resp.text[:200]}"
            )
        try:
            return resp.json()
        except ValueError:
            last_err = "non-JSON response body"
            continue
    raise GradeError(f"giving up after {MAX_ATTEMPTS} attempts: {last_err}")


def _content_of(data: dict) -> str:
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise GradeError(f"unexpected response shape: {json.dumps(data)[:200]}")
    if not isinstance(content, str):
        raise GradeError("response content is not a string")
    return content


async def grade_batch(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    q: dict,
    batch: list[dict],
    anchors: dict | None = None,
) -> tuple[dict[str, int], str | None]:
    """Grade one batch. Returns ({doc_id: grade}, error_or_None).

    On a JSON failure the batch is retried once with a "return only JSON"
    reminder; valid grades from either attempt are kept.
    """
    batch_ids = [c["document_id"] for c in batch]
    messages = _base_messages(q, batch, anchors)
    payload = {
        "model": settings.neuralwatt_model,
        "messages": messages,
        "temperature": 0,
    }
    best: dict[str, int] = {}
    error: str | None = None
    for attempt in range(2):
        try:
            data = await _post_with_retry(client, url, headers, payload)
            content = _content_of(data)
        except GradeError as e:
            return best, str(e)
        if _absorb_attempt(content, batch_ids, best, q["id"]):
            return best, None
        error = (
            f"incomplete JSON grading ({len(best)}/{len(batch_ids)} candidates)"
        )
        if attempt == 0:
            messages = _reminder_messages(messages, content)
    return best, error


def _split_batches(candidates: list[dict], batch_size: int) -> list[list[dict]]:
    return [
        candidates[i : i + batch_size]
        for i in range(0, len(candidates), batch_size)
    ]


async def grade_question(
    q: dict,
    candidates: list[dict],
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    batch_size: int,
    concurrency: int,
    anchors: dict | None = None,
) -> tuple[dict[str, int], list[int]]:
    """Grade one question's pending ids; returns (judgments, failed_batches).

    Batches run concurrently, bounded by ``concurrency`` via a semaphore.
    """
    batches = _split_batches(candidates, batch_size)
    judgments: dict[str, int] = {}
    failed: list[int] = []
    sem = asyncio.Semaphore(concurrency)

    async def _run(i: int) -> tuple[int, dict[str, int], str | None]:
        async with sem:
            grades, err = await grade_batch(client, url, headers, q,
                                            batches[i], anchors)
        return i, grades, err

    results = await asyncio.gather(*(_run(i) for i in range(len(batches))))
    for i, grades, err in results:
        if err:
            failed.append(i)
            print(
                f"    {q['id']} batch {i + 1}/{len(batches)} FAILED: {err}",
                file=sys.stderr,
            )
        for doc_id, g in grades.items():
            if doc_id not in judgments:
                judgments[doc_id] = g
    return judgments, failed


def _post_with_retry_sync(
    client: httpx.Client,
    url: str,
    headers: dict,
    payload: dict,
) -> dict:
    """Sync twin of _post_with_retry (blocking HTTP, time.sleep backoff)."""
    last_err = "never attempted"
    for attempt in range(MAX_ATTEMPTS):
        if attempt:
            delay = min(60.0, 2.0**attempt) + random.uniform(0.0, 1.0)
            time.sleep(delay)
        try:
            resp = client.post(url, headers=headers, json=payload)
        except httpx.RequestError as e:
            last_err = f"request error: {e}"
            continue
        if resp.status_code in RETRYABLE_STATUS:
            last_err = f"HTTP {resp.status_code}"
            continue
        if resp.status_code >= 400:
            raise GradeError(
                f"HTTP {resp.status_code}: {resp.text[:200]}"
            )
        try:
            return resp.json()
        except ValueError:
            last_err = "non-JSON response body"
            continue
    raise GradeError(f"giving up after {MAX_ATTEMPTS} attempts: {last_err}")


def grade_batch_sync(
    client: httpx.Client,
    url: str,
    headers: dict,
    q: dict,
    batch: list[dict],
    anchors: dict | None = None,
) -> tuple[dict[str, int], str | None]:
    """Sync twin of grade_batch (same retry/reminder semantics)."""
    batch_ids = [c["document_id"] for c in batch]
    messages = _base_messages(q, batch, anchors)
    payload = {
        "model": settings.neuralwatt_model,
        "messages": messages,
        "temperature": 0,
    }
    best: dict[str, int] = {}
    error: str | None = None
    for attempt in range(2):
        try:
            data = _post_with_retry_sync(client, url, headers, payload)
            content = _content_of(data)
        except GradeError as e:
            return best, str(e)
        if _absorb_attempt(content, batch_ids, best, q["id"]):
            return best, None
        error = (
            f"incomplete JSON grading ({len(best)}/{len(batch_ids)} candidates)"
        )
        if attempt == 0:
            messages = _reminder_messages(messages, content)
    return best, error


def grade_question_sync(
    q: dict,
    candidates: list[dict],
    client: httpx.Client,
    url: str,
    headers: dict,
    batch_size: int,
    concurrency: int,
    anchors: dict | None = None,
) -> tuple[dict[str, int], list[int]]:
    """Sync twin of grade_question: blocking HTTP via a thread pool."""
    batches = _split_batches(candidates, batch_size)
    judgments: dict[str, int] = {}
    failed: list[int] = []

    def _run(i: int) -> tuple[int, dict[str, int], str | None]:
        grades, err = grade_batch_sync(client, url, headers, q, batches[i],
                                       anchors)
        return i, grades, err

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for i, grades, err in pool.map(_run, range(len(batches))):
            if err:
                failed.append(i)
                print(
                    f"    {q['id']} batch {i + 1}/{len(batches)} FAILED: {err}",
                    file=sys.stderr,
                )
            for doc_id, g in grades.items():
                if doc_id not in judgments:
                    judgments[doc_id] = g
    return judgments, failed


# --------------------------------------------------------------------------- #
# judgments.json merge (human wins; probes recorded separately)
# --------------------------------------------------------------------------- #

def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"  warning: could not parse {path}: {e}, starting empty",
              file=sys.stderr)
        return {}


def _save_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def merge_question_entry(
    old_entry: dict | None,
    q: dict,
    human: dict[str, int],
    new_llm: dict[str, int],
    new_probes: dict[str, int],
) -> dict:
    """Build the v3 per-question entry. Human grades always win; new LLM
    grades only fill ids without a human judgment. Recomputed from scratch,
    so a re-run self-heals earlier partial merges."""
    old_entry = old_entry or {}
    carried_llm = {
        d: g for d, g in old_entry.get("judgments", {}).items()
        if d not in human
    }
    judgments: dict[str, int] = dict(carried_llm)
    judgments.update(new_llm)
    judgments.update(human)

    probe_llm: dict[str, int] = dict(old_entry.get("probe_llm", {}))
    probe_llm.update(new_probes)

    n_llm = sum(1 for d in judgments if d not in human)
    grade_counts = Counter(judgments.values())
    return {
        "question": q["question"],
        "game": q.get("game", "Dark Souls"),
        "category": q["category"],
        "judgments": judgments,
        "probe_llm": probe_llm,
        "sources": {
            "human": len(human),
            "llm": n_llm,
            "run": old_entry.get("sources", {}).get("run", 0),
        },
        "grade_counts": {str(g): n for g, n in sorted(grade_counts.items())},
        "n_judgments": len(judgments),
        "updated_at": _now_iso(),
    }


def grading_targets(
    pool: list[dict],
    judged: dict[str, int],
    probe_llm: dict[str, int],
    probe_ids: set[str],
    anchor_ids: set[str],
) -> list[dict]:
    """Pool entries still to grade: unjudged non-anchor ids, plus probes that
    have no LLM grade yet."""
    targets = []
    for cand in pool:
        doc_id = cand["document_id"]
        if doc_id in anchor_ids:
            continue
        if doc_id not in judged or (
            doc_id in probe_ids and doc_id not in probe_llm
        ):
            targets.append(cand)
    return targets


# --------------------------------------------------------------------------- #
# CLI run
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", type=Path,
                    default=REPO_ROOT / "eval" / "questions.json")
    ap.add_argument("--candidates", type=Path,
                    default=REPO_ROOT / "eval" / "v3" / "candidates")
    ap.add_argument("--judgments", type=Path,
                    default=REPO_ROOT / "eval" / "v3" / "judgments.json")
    ap.add_argument("--anchors", type=Path,
                    default=REPO_ROOT / "eval" / "v3" / "seeds" / "anchors.json")
    ap.add_argument("--human-seeds", type=Path,
                    default=REPO_ROOT / "eval" / "v3" / "seeds"
                    / "human-judgments.json")
    ap.add_argument("--start", type=str, default=None,
                    help="question id to start from (e.g. q001)")
    ap.add_argument("--limit", type=int, default=None,
                    help="grade at most N questions this run")
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--sync", action="store_true",
                    help="run fully synchronously (blocking httpx.Client + "
                         "thread pool) instead of asyncio; easier to debug")
    args = ap.parse_args()

    if not settings.neuralwatt_api_key:
        sys.exit(
            "NEURALWATT_API_KEY is empty. Add it to .env (see .env.example) "
            "before running the grader."
        )
    if args.batch_size < 1 or args.concurrency < 1:
        sys.exit("--batch-size and --concurrency must be >= 1")

    if args.sync:
        _run_grading_sync(args)
    else:
        asyncio.run(_run_grading(args))


def _load_anchors(path: Path) -> dict:
    data = _load_json(path)
    return data.get("anchors", data) if isinstance(data, dict) else {}


def _pending_questions(
    args: argparse.Namespace,
    judgments: dict,
    anchors: dict,
    human_seeds: dict,
) -> Iterator[tuple[dict, list[dict], int, dict | None]]:
    """Yield (question, targets, n_batches, anchors) per pending question.

    A question is pending when a pool file exists; its grading targets are
    the pool ids without a judgment (id-level resume) plus probes lacking an
    LLM grade. Anchor ids are excluded. Questions with zero targets are still
    yielded once (to merge human seeds) unless an identical entry exists.
    """
    started = args.start is None
    for q in json.loads(args.questions.read_text(encoding="utf-8")):
        if not started:
            if q["id"] == args.start:
                started = True
            else:
                continue
        cand_path = args.candidates / f"{q['id']}.json"
        if not cand_path.exists():
            continue
        pool = json.loads(cand_path.read_text(encoding="utf-8"))
        if not pool:
            print(f"  warning: {q['id']} has empty pool, skipping")
            continue
        q_anchors = anchors.get(q["id"]) or {}
        anchor_ids = {
            a["doc_id"]
            for a in (q_anchors.get("gold"), q_anchors.get("contrast"))
            if a and a.get("doc_id")
        }
        entry = judgments.get(q["id"]) or {}
        probe_ids = set(q_anchors.get("probes") or [])
        judged = {**human_seeds.get(q["id"], {}), **entry.get("judgments", {})}
        targets = grading_targets(
            pool, judged, entry.get("probe_llm", {}),
            probe_ids, anchor_ids,
        )
        entry_ok = bool(entry) and not targets
        if entry_ok:
            recomputed = merge_question_entry(entry, q,
                                              human_seeds.get(q["id"], {}),
                                              {}, {})
            recomputed.pop("updated_at")
            if recomputed == {k: v for k, v in entry.items()
                              if k != "updated_at"}:
                print(f"  skipping {q['id']} (nothing new to grade)")
                continue
        yield q, targets, len(_split_batches(targets, args.batch_size)), \
            (q_anchors or None)


def _record_result(
    args: argparse.Namespace,
    judgments: dict,
    q: dict,
    human: dict[str, int],
    new_grades: dict[str, int],
    q_anchors: dict | None,
    n_batches: int,
    failed: list[int],
) -> bool:
    """Merge one question's grades, flush atomically, print a summary line.

    Returns True if any batch failed for this question.
    """
    probe_ids = set((q_anchors or {}).get("probes") or [])
    new_probes = {d: g for d, g in new_grades.items() if d in probe_ids}
    new_llm = {d: g for d, g in new_grades.items() if d not in probe_ids}
    entry = merge_question_entry(judgments.get(q["id"]), q, human, new_llm,
                                 new_probes)
    judgments[q["id"]] = entry
    _save_json(args.judgments, judgments)
    note = f", {len(failed)}/{n_batches} batches failed" if failed else ""
    print(f"  saved {q['id']}: {entry['n_judgments']} judgments "
          f"(human={entry['sources']['human']} llm={entry['sources']['llm']}"
          f" probes={len(entry['probe_llm'])}){note}")
    return bool(failed)


def _print_run_report(
    args: argparse.Namespace,
    questions_with_failures: dict[str, int],
    graded: int,
) -> None:
    if questions_with_failures:
        print("\n=== Questions with failed batches (just re-run; id-level "
              "resume retries only the missing ids) ===")
        for qid, n in questions_with_failures.items():
            print(f"  {qid}: {n} failed batch(es)")
    print(f"\nDone. Graded {graded} question(s) this run. "
          f"Judgments at {args.judgments}")


def _run_grading_sync(args: argparse.Namespace) -> None:
    """Fully synchronous run (--sync): blocking HTTP, thread-pool batches."""
    judgments = _load_json(args.judgments)
    anchors = _load_anchors(args.anchors)
    human_seeds = {
        qid: entry["judgments"]
        for qid, entry in _load_json(args.human_seeds).items()
    }
    print(f"Loaded {len(judgments)} existing question entries from "
          f"{args.judgments}")

    url = settings.neuralwatt_base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {settings.neuralwatt_api_key}"}

    graded = 0
    questions_with_failures: dict[str, int] = {}

    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        for q, targets, n_batches, q_anchors in _pending_questions(
            args, judgments, anchors, human_seeds
        ):
            if args.limit is not None and graded >= args.limit:
                break
            if not targets:
                _record_result(args, judgments, q,
                               human_seeds.get(q["id"], {}), {}, q_anchors,
                               0, [])
                continue
            print(f"  grading {q['id']}: {len(targets)} pending id(s) in "
                  f"{n_batches} batch(es)")
            new_grades, failed = grade_question_sync(
                q, targets, client, url, headers,
                args.batch_size, args.concurrency, q_anchors,
            )
            if _record_result(args, judgments, q, human_seeds.get(q["id"], {}),
                              new_grades, q_anchors, n_batches, failed):
                questions_with_failures[q["id"]] = len(failed)
            graded += 1

    _print_run_report(args, questions_with_failures, graded)


async def _run_grading(args: argparse.Namespace) -> None:
    judgments = _load_json(args.judgments)
    anchors = _load_anchors(args.anchors)
    human_seeds = {
        qid: entry["judgments"]
        for qid, entry in _load_json(args.human_seeds).items()
    }
    print(f"Loaded {len(judgments)} existing question entries from "
          f"{args.judgments}")

    url = settings.neuralwatt_base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {settings.neuralwatt_api_key}"}

    graded = 0
    questions_with_failures: dict[str, int] = {}

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        for q, targets, n_batches, q_anchors in _pending_questions(
            args, judgments, anchors, human_seeds
        ):
            if args.limit is not None and graded >= args.limit:
                break
            if not targets:
                _record_result(args, judgments, q,
                               human_seeds.get(q["id"], {}), {}, q_anchors,
                               0, [])
                continue
            print(f"  grading {q['id']}: {len(targets)} pending id(s) in "
                  f"{n_batches} batch(es)")
            new_grades, failed = await grade_question(
                q, targets, client, url, headers,
                args.batch_size, args.concurrency, q_anchors,
            )
            if _record_result(args, judgments, q, human_seeds.get(q["id"], {}),
                              new_grades, q_anchors, n_batches, failed):
                questions_with_failures[q["id"]] = len(failed)
            graded += 1

    _print_run_report(args, questions_with_failures, graded)


if __name__ == "__main__":
    main()
