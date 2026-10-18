"""Question schema validation for the frozen benchmark question set."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from eval.corpus import resolve_entities
from eval.storage import now_iso
from eval.text import normalize_title, normalize_whitespace

GAMES = ("Dark Souls", "Dark Souls II", "Dark Souls III", "All")

# Six report groups; the historical fine-grained label is kept as
# ``subcategory`` so old reports remain interpretable.
REPORT_CATEGORIES = (
    "lore",
    "boss",
    "item_location",
    "mechanics_build",
    "quests_exploration",
    "cross_game",
)

REQUIRED_FIELDS = ("id", "question", "game", "category", "expected_entities")
MAX_ENTITIES = 7
COMPOUND_WARNING_THRESHOLD = 4

# Verified non-page strings plus generic words that never help the answer-key
# candidate selector. Matching is on normalized text.
GENERIC_ENTITY_BLOCKLIST = frozenset(
    normalize_title(value)
    for value in (
        "Quality Build",
        "Weapon Scaling",
        "Attack Rating",
        "Cycle",
        "Link",
        "Stats",
        "Memory",
        "Fire Keepers",
        "Dark",
        "Queen",
        "King",
        "Crown",
        "Serpent",
        "Wolf",
        "Giants",
    )
)

IMPERATIVE_STARTS = frozenset(
    {
        "compare",
        "contrast",
        "describe",
        "explain",
        "give",
        "identify",
        "list",
        "name",
        "outline",
        "summarize",
        "trace",
    }
)

_COORDINATION_RE = re.compile(
    r"\band\b|\bversus\b|\bvs\.?\b|compared with|;\s*|\?",
    flags=re.IGNORECASE,
)


def entity_name(entity: Any) -> str:
    """Return the display/lookup name for a string or alias record."""
    if isinstance(entity, str):
        return entity
    if isinstance(entity, dict):
        return str(entity.get("name") or entity.get("entity") or "")
    return str(entity)


def entity_target(entity: Any) -> str:
    """Return the page title an alias record resolves to, if declared."""
    if isinstance(entity, dict):
        target = entity.get("resolves_to") or entity.get("alias_of")
        if target:
            return str(target)
    return entity_name(entity)


def entity_names(question: dict[str, Any]) -> list[str]:
    return [entity_name(entity) for entity in question.get("expected_entities", [])]


def _compound_clauses(text: str) -> int:
    return 1 + len(_COORDINATION_RE.findall(text))


def _question_is_well_formed(text: str) -> bool:
    stripped = text.strip()
    if stripped.endswith("?"):
        return True
    if stripped.endswith((".", "!")):
        first = stripped.split(maxsplit=1)[0].lower().strip(":")
        return first in IMPERATIVE_STARTS
    return False


def _normalized_question(text: str) -> str:
    return normalize_whitespace(text).lower().rstrip("?.")


def validate_questions(
    questions: Iterable[dict[str, Any]],
    *,
    titles: dict[str, list[str]] | None = None,
    check_entities: bool = True,
) -> dict[str, Any]:
    """Validate the question list and return a machine-readable report.

    ``titles`` maps game -> page titles. When omitted (and ``check_entities``
    is true), ``data/structured`` is scanned.
    """
    question_list = list(questions)
    if check_entities and titles is None:
        from eval.corpus import titles_by_game

        titles = titles_by_game()

    seen_ids: dict[str, int] = {}
    seen_text: dict[str, str] = {}
    report: dict[str, Any] = {
        "generated_at": now_iso(),
        "n_questions": len(question_list),
        "counts": {},
        "questions": {},
    }
    flag_counts: dict[str, int] = {}
    must_fix_total = 0

    for index, question in enumerate(question_list):
        qid = str(question.get("id") or f"index:{index}")
        flags: list[dict[str, str]] = []

        def flag(
            code: str,
            detail: str,
            *,
            must_fix: bool = True,
            _flags: list[dict[str, str]] = flags,
        ) -> None:
            _flags.append({"code": code, "detail": detail, "must_fix": must_fix})
            flag_counts[code] = flag_counts.get(code, 0) + 1

        for field in REQUIRED_FIELDS:
            if field not in question:
                flag("missing_field", field)
        if not re.fullmatch(r"q\d{3}", qid):
            flag("invalid_id", "id must look like qNNN")

        seen_ids[qid] = seen_ids.get(qid, 0) + 1
        if seen_ids[qid] == 2:
            flag("duplicate_id", qid)

        text = str(question.get("question") or "")
        if not text.strip():
            flag("missing_question", qid)
        elif not _question_is_well_formed(text):
            flag("bad_question_form", "question must end with '?' or be an imperative")

        normalized = _normalized_question(text)
        if normalized in seen_text:
            flag("duplicate_text", f"same as {seen_text[normalized]}")
        else:
            seen_text[normalized] = qid

        game = question.get("game")
        if game not in GAMES:
            flag("invalid_game", str(game))

        category = question.get("category")
        if category not in REPORT_CATEGORIES:
            flag("invalid_category", str(category))
        subcategory = question.get("subcategory")
        if subcategory is not None and not isinstance(subcategory, str):
            flag("invalid_subcategory", str(subcategory))

        compounds = _compound_clauses(text)
        if compounds >= COMPOUND_WARNING_THRESHOLD:
            flag(
                "compound_question",
                f"{compounds} coordinated clauses",
                must_fix=False,
            )

        entities = question.get("expected_entities")
        unresolved: list[str] = []
        resolved: dict[str, list[str]] = {}
        generic: list[str] = []
        if not isinstance(entities, list) or not entities:
            flag("missing_entities", qid)
        else:
            if len(entities) > MAX_ENTITIES:
                flag("too_many_entities", f"{len(entities)} > {MAX_ENTITIES}")
            for entity in entities:
                name = entity_name(entity)
                if not name:
                    flag("empty_entity", qid)
                    continue
                if normalize_title(name) in GENERIC_ENTITY_BLOCKLIST:
                    generic.append(name)
                    continue
                target = entity_target(entity)
                if not check_entities:
                    continue
                allowed_games = GAMES[:3] if game == "All" else (game,)
                matches: list[str] = []
                for allowed_game in allowed_games:
                    game_titles = (titles or {}).get(allowed_game, [])
                    for matched_titles in resolve_entities([target], game_titles).values():
                        matches.extend(matched_titles)
                if matches:
                    resolved[name] = list(dict.fromkeys(matches))[:5]
                else:
                    unresolved.append(name)
        if generic:
            flag("generic_entities", ", ".join(generic))
        if unresolved:
            flag("unresolved_entities", ", ".join(unresolved))

        must_fix = any(item["must_fix"] for item in flags)
        if must_fix:
            must_fix_total += 1
        report["questions"][qid] = {
            "flags": flags,
            "must_fix": must_fix,
            "compound_clauses": compounds,
            "resolved_entities": resolved,
            "unresolved_entities": unresolved,
        }

    report["counts"] = dict(sorted(flag_counts.items()))
    report["must_fix"] = must_fix_total
    report["ok"] = must_fix_total == 0
    return report


def select_question_subset(
    questions: Iterable[dict[str, Any]],
    *,
    only: set[str] | None = None,
    limit: int | None = None,
    start: str | None = None,
) -> list[dict[str, Any]]:
    """Filter a question list by id set, starting id, and count."""
    selected = list(questions)
    if start is not None:
        ids = [str(question["id"]) for question in selected]
        if start not in ids:
            raise ValueError(f"unknown start question: {start}")
        selected = selected[ids.index(start) :]
    if only:
        selected = [
            question for question in selected if str(question["id"]) in only
        ]
    if limit is not None:
        selected = selected[:limit]
    return selected


def report_has_errors(report: dict[str, Any]) -> bool:
    return not report.get("ok", False)


def format_validation_report(report: dict[str, Any]) -> str:
    lines = [
        f"Questions: {report['n_questions']}",
        f"Must-fix questions: {report.get('must_fix', 0)}",
    ]
    if report.get("counts"):
        counts = ", ".join(f"{key}={value}" for key, value in report["counts"].items())
        lines.append(f"Flags: {counts}")
    for qid, entry in report.get("questions", {}).items():
        if not entry.get("must_fix"):
            continue
        details = "; ".join(
            f"{flag['code']}: {flag['detail']}"
            for flag in entry.get("flags", [])
            if flag.get("must_fix")
        )
        lines.append(f"  {qid}: {details}")
    return "\n".join(lines)
