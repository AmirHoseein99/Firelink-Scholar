"""Prompt text for the run-driven LLM judge.

The rubric is deliberately concrete. Changing this file changes the pinned
prompt hash in ``eval/dataset.json`` and requires a new dataset version.
"""

from __future__ import annotations

import hashlib

GRADE_SYSTEM_PROMPT = """\
You are a strict retrieval judge for a Dark Souls wiki benchmark. Grade each \
candidate chunk against the supplied answer key. Judge only whether the chunk \
contains evidence for the listed claims, not whether it is a good answer and \
not whether it mentions an expected entity.
"""

GRADE_USER_TEMPLATE = """\
Question: {question}
Game: {game}
Expected entities: {entities}

Answer-key claims:
{claims}

Rubric:
- 3: directly states or supports a core claim.
- 2: substantially supports a claim, or partially supports a core claim.
- 1: relevant background that mentions the expected entities but supports no \
claim.
- 0: unrelated, or contradicts the answer key.

A chunk that is correct and question-relevant but absent from the answer key \
is not forced to 0. Grade it on the rubric and set "other_relevant": true.

Candidate chunks:
{candidates}

Return only a JSON object:
{{"grades": [{{"n": 1, "grade": 2, "claims": ["c1"], "other_relevant": false}}]}}

Rules:
- "n" is the candidate number shown above.
- "claims" lists only claim ids actually supported by that chunk (empty for \
grades 0 and 1).
- "grade" must be an integer from 0 to 3.
- Return one entry for every candidate, in any order.
"""


def prompt_sha256() -> str:
    payload = GRADE_SYSTEM_PROMPT + "\n" + GRADE_USER_TEMPLATE
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


GRADE_PROMPT_SHA256 = prompt_sha256()


def claim_block(claims: list[dict]) -> str:
    lines = []
    for claim in claims:
        lines.append(
            f"[{claim['id']}] ({claim['importance']}) {claim['text']}"
        )
    return "\n".join(lines)


def candidate_block(candidates: list[dict], max_chars: int = 1600) -> str:
    blocks = []
    for number, candidate in enumerate(candidates, start=1):
        text = str(candidate.get("page_content", ""))
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + " ..."
        blocks.append(
            "\n".join(
                [
                    f"--- CANDIDATE {number} ---",
                    f"document_id: {candidate.get('document_id', '')}",
                    f"title: {candidate.get('title', '')}",
                    f"section: {candidate.get('section', '')}",
                    f"game: {candidate.get('game', '')}",
                    text,
                ]
            )
        )
    return "\n\n".join(blocks)


def build_grade_messages(
    *,
    question: str,
    game: str,
    entities: list[str],
    claims: list[dict],
    candidates: list[dict],
) -> list[dict[str, str]]:
    user = GRADE_USER_TEMPLATE.format(
        question=question,
        game=game,
        entities=", ".join(entities) if entities else "(none)",
        claims=claim_block(claims),
        candidates=candidate_block(candidates),
    )
    return [
        {"role": "system", "content": GRADE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
