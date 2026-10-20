"""Build frozen answer keys (claims + verbatim quotes) from data/structured."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from eval import paths
from eval.corpus import Page, Section, load_pages, pages_for_game
from eval.dataset import claim_limits
from eval.llm import Backend, chat, extract_json
from eval.prompts_answer import ANSWER_PROMPT_SHA256, build_answer_messages
from eval.questions import entity_target, select_question_subset
from eval.storage import load_json, now_iso, save_json
from eval.text import match_rank, normalize_whitespace, quote_is_grounded, tokenize

logger = logging.getLogger(__name__)

PREFERRED_SECTIONS = ("lead", "lore", "plot", "strategy", "description", "notes")


@dataclass
class Candidate:
    source_path: str
    page_title: str
    game: str
    entity_type: str
    section_key: str
    text: str

    def as_dict(self) -> dict[str, str]:
        return {
            "source_path": self.source_path,
            "page_title": self.page_title,
            "game": self.game,
            "entity_type": self.entity_type,
            "section_key": self.section_key,
            "text": self.text,
        }


class CandidateSelector:
    """Game-filtered candidate sections, with an in-process BM25 index."""

    def __init__(self, pages: Iterable[Page] | None = None) -> None:
        self.pages = list(pages) if pages is not None else load_pages()
        self._game_pages: dict[str, list[Page]] = {}
        self._bm25: dict[str, tuple[BM25Okapi, list[Section]]] = {}

    def game_pages(self, game: str) -> list[Page]:
        if game not in self._game_pages:
            self._game_pages[game] = pages_for_game(self.pages, game)
        return self._game_pages[game]

    def _index(self, game: str) -> tuple[BM25Okapi, list[Section]]:
        if game not in self._bm25:
            sections = [
                section
                for page in self.game_pages(game)
                for section in page.sections
                if section.text.strip()
            ]
            corpus = [tokenize(section.text) for section in sections]
            self._bm25[game] = (BM25Okapi(corpus), sections)
        return self._bm25[game]

    def select(
        self,
        question: dict[str, Any],
        *,
        max_candidates: int | None = None,
    ) -> list[Candidate]:
        game = question["game"]
        entities = [
            entity_target(entity) for entity in question.get("expected_entities", [])
        ]
        if max_candidates is None:
            max_candidates = 40 if game == "All" else 25
        bm25_limit = 40 if game == "All" else 20

        pages = self.game_pages(game)
        matched: dict[tuple[str, str], int] = {}
        for page in pages:
            for entity in entities:
                rank = match_rank(entity, page.title)
                if rank is not None:
                    key = (page.game, page.title)
                    matched[key] = min(matched.get(key, 9), rank)

        selected: list[Candidate] = []
        seen: set[tuple[str, str]] = set()

        def add(section: Section) -> None:
            key = (section.source_path, section.section_key)
            if key in seen:
                return
            seen.add(key)
            selected.append(
                Candidate(
                    source_path=section.source_path,
                    page_title=section.page_title,
                    game=section.game,
                    entity_type=section.entity_type,
                    section_key=section.section_key,
                    text=section.text,
                )
            )

        # 1. Preferred sections on entity-matched pages, best match first.
        matched_pages = sorted(
            (page for page in pages if (page.game, page.title) in matched),
            key=lambda page: (matched[(page.game, page.title)], page.title),
        )
        for page in matched_pages:
            for section in page.sections:
                if section.section_key.lower() in PREFERRED_SECTIONS:
                    add(section)

        # 2. BM25 over the question's game only.
        bm25, sections = self._index(game)
        query_tokens = tokenize(question["question"])
        if query_tokens:
            scores = bm25.get_scores(query_tokens)
            ranked = sorted(range(len(sections)), key=lambda i: scores[i], reverse=True)
            for index in ranked[:bm25_limit]:
                if scores[index] <= 0:
                    continue
                add(sections[index])

        # 3. Remaining sections of entity-matched pages.
        for page in matched_pages:
            for section in page.sections:
                if section.section_key.lower() not in PREFERRED_SECTIONS:
                    add(section)

        return selected[:max_candidates]


def _normalize_source_path(value: str) -> str:
    text = str(value).replace("\\", "/").lstrip("./")
    return text.removeprefix("data/structured/")


def _candidate_lookup(candidates: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in candidates:
        path = _normalize_source_path(
            str(candidate.get("source_path") or candidate.get("path") or "")
        )
        section = str(candidate.get("section_key") or candidate.get("section") or "")
        lookup[(path.lower(), section.lower())] = candidate
    return lookup


def parse_claims(
    payload: Any,
    candidates: list[dict[str, Any]],
    *,
    max_claims: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Validate model claims against supplied candidate sections.

    Returns ``(valid_claims, dropped)``. Quotes are whitespace-normalized
    substrings of the cited section; fabricated quotes are always dropped.
    """
    if isinstance(payload, dict):
        raw_claims = payload.get("claims")
    elif isinstance(payload, list):
        raw_claims = payload
    else:
        raw_claims = None
    if not isinstance(raw_claims, list):
        return [], [{"reason": "no claims list in response"}]

    lookup = _candidate_lookup(candidates)
    valid: list[dict[str, Any]] = []
    dropped: list[dict[str, str]] = []
    used_claims: set[str] = set()
    for raw in raw_claims:
        if not isinstance(raw, dict):
            dropped.append({"reason": "claim is not an object"})
            continue
        text = normalize_whitespace(str(raw.get("text") or ""))
        quote = normalize_whitespace(str(raw.get("quote") or ""))
        path = str(raw.get("path") or "")
        section = str(raw.get("section") or "")
        importance = str(raw.get("importance") or "supporting").lower()
        if importance not in ("core", "supporting"):
            importance = "supporting"
        if not text or not quote:
            dropped.append({"reason": "empty claim text or quote"})
            continue
        candidate = lookup.get(
            (_normalize_source_path(path).lower(), section.lower())
        )
        if candidate is None:
            dropped.append({"reason": f"unknown citation {path}:{section}", "text": text})
            continue
        section_text = str(candidate.get("text", ""))
        if not quote_is_grounded(quote, section_text):
            dropped.append({"reason": "quote is not a substring of the section", "text": text})
            continue
        key = text.lower()
        if key in used_claims:
            dropped.append({"reason": "duplicate claim text", "text": text})
            continue
        used_claims.add(key)
        valid.append(
            {
                "text": text,
                "importance": importance,
                "source": {
                    "path": candidate.get("source_path") or candidate.get("path"),
                    "title": candidate.get("page_title") or candidate.get("title"),
                    "section": candidate.get("section_key") or candidate.get("section"),
                },
                "quote": quote,
            }
        )
        if len(valid) >= max_claims:
            break

    for position, claim in enumerate(valid, start=1):
        claim["id"] = f"c{position}"
    return valid, dropped


def _compose_answer(claims: list[dict[str, Any]]) -> str:
    core = [claim["text"] for claim in claims if claim["importance"] == "core"]
    supporting = [claim["text"] for claim in claims if claim["importance"] != "core"]
    ordered = core + supporting
    if not ordered:
        return ""
    return " ".join(text.rstrip(" .") + "." for text in ordered)


def build_answer_for_question(
    question: dict[str, Any],
    selector: CandidateSelector,
    *,
    backend: Backend,
    max_claims: int,
    min_claims: int = 3,
    retries: int = 1,
    call: Callable[..., str] | None = None,
) -> dict[str, Any]:
    """Ask the builder for one answer key and verify every quote."""
    candidates = [candidate.as_dict() for candidate in selector.select(question)]
    logger.debug(
        "%s: %d candidate section(s) for answer key",
        question.get("id", "?"),
        len(candidates),
    )
    if not candidates:
        logger.warning("%s: no candidate sections for answer key", question.get("id", "?"))
        return {
            "question": question["question"],
            "answer": "",
            "claims": [],
            "builder": {
                "model": backend.model,
                "prompt_sha256": ANSWER_PROMPT_SHA256,
                "created_at": now_iso(),
            },
            "status": "needs_review",
            "review_reason": "no candidate sections",
        }

    messages = build_answer_messages(
        question=question["question"],
        game=question["game"],
        entities=[
            str(entity.get("name") if isinstance(entity, dict) else entity)
            for entity in question.get("expected_entities", [])
        ],
        candidates=candidates,
        min_claims=min_claims,
        max_claims=max_claims,
    )

    best: list[dict[str, Any]] = []
    best_dropped: list[dict[str, str]] = []
    last_error: str | None = None
    invoke = call or chat
    for _ in range(retries + 1):
        try:
            raw = invoke(messages, backend=backend, temperature=0.0)
            payload = extract_json(raw)
            claims, dropped = parse_claims(payload, candidates, max_claims=max_claims)
        except Exception as exc:  # noqa: BLE001 - surfaced in review_reason
            last_error = str(exc)
            logger.debug(
                "%s: answer attempt failed: %s", question.get("id", "?"), exc
            )
            continue
        if len(claims) > len(best):
            best, best_dropped = claims, dropped
        if len(claims) >= min_claims and any(
            claim["importance"] == "core" for claim in claims
        ):
            break

    has_core = any(claim["importance"] == "core" for claim in best)
    status = "verified" if has_core and len(best) >= min_claims else "needs_review"
    review_reason = None
    if status != "verified":
        if not best:
            review_reason = last_error or "no verifiable claims"
        elif not has_core:
            review_reason = "no verifiable core claim"
        else:
            review_reason = f"only {len(best)} verified claim(s)"
    return {
        "question": question["question"],
        "answer": _compose_answer(best),
        "claims": best,
        "builder": {
            "model": backend.model,
            "prompt_sha256": ANSWER_PROMPT_SHA256,
            "created_at": now_iso(),
        },
        "status": status,
        "review_reason": review_reason,
        "dropped_claims": best_dropped,
    }


def load_questions(path: Path = paths.QUESTIONS) -> list[dict[str, Any]]:
    questions = load_json(Path(path), default=[])
    return list(questions or [])


def questions_by_id(questions: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(question["id"]): question for question in questions}


def build_answers(
    *,
    dataset: dict[str, Any],
    backend: Backend,
    only: set[str] | None = None,
    limit: int | None = None,
    start: str | None = None,
    force: bool = False,
    retries: int = 1,
    questions_path: Path = paths.QUESTIONS,
    answers_path: Path = paths.ANSWERS,
    pages: list[Page] | None = None,
    call: Callable[..., str] | None = None,
) -> dict[str, Any]:
    questions = select_question_subset(
        load_questions(questions_path), only=only, limit=limit, start=start
    )
    existing: dict[str, Any] = load_json(answers_path, default={}) or {}
    selector = CandidateSelector(pages)
    summary: dict[str, Any] = {"built": [], "skipped": [], "needs_review": []}
    logger.info(
        "Building answer keys for %d question(s) with %s",
        len(questions),
        backend.model,
    )
    for position, question in enumerate(questions, start=1):
        qid = str(question["id"])
        if qid in existing and not force:
            summary["skipped"].append(qid)
            logger.debug("%s: answer key already exists; skipped", qid)
            continue
        logger.info(
            "%s: building answer key (%d/%d)", qid, position, len(questions)
        )
        _, max_claims = claim_limits(dataset, question["game"])
        answer = build_answer_for_question(
            question,
            selector,
            backend=backend,
            max_claims=max_claims,
            retries=retries,
            call=call,
        )
        existing[qid] = answer
        save_json(answers_path, existing)
        summary["built"].append(qid)
        if answer["status"] != "verified":
            summary["needs_review"].append(qid)
            logger.warning(
                "%s: answer key needs review: %s", qid, answer.get("review_reason")
            )
        else:
            logger.info("%s: verified %d claim(s)", qid, len(answer.get("claims", [])))
        if answer.get("dropped_claims"):
            logger.debug(
                "%s: dropped %d claim(s)",
                qid,
                len(answer["dropped_claims"]),
            )
    logger.info(
        "Answer keys done: %d built, %d skipped, %d need review",
        len(summary["built"]),
        len(summary["skipped"]),
        len(summary["needs_review"]),
    )
    return summary
