"""Prompt text for the answer-key builder.

The prompt hash is part of the benchmark definition (``eval/dataset.json``).
Changing this file changes the hash and therefore requires an explicit
answer-key rebuild.
"""

from __future__ import annotations

import hashlib

ANSWER_SYSTEM_PROMPT = """\
You build compact, verifiable answer keys for a retrieval benchmark over a \
Dark Souls wiki. Use only the candidate sections supplied by the user. Never \
add facts from memory, and never cite a section that was not supplied.
"""

ANSWER_USER_TEMPLATE = """\
Question: {question}
Game scope: {game}
Expected entities: {entities}
Allowed claim count: {min_claims}-{max_claims} claims (at least one core).

Candidate sections:
{candidates}

Return only a JSON object in this shape:
{{
  "claims": [
    {{
      "text": "one self-contained factual claim",
      "quote": "verbatim text copied from the cited section",
      "path": "candidate path exactly as shown",
      "section": "candidate section key exactly as shown",
      "importance": "core" or "supporting"
    }}
  ]
}}

Rules:
- Every claim must be directly supported by its quoted text.
- The quote must be copied verbatim from the cited candidate section, with \
only whitespace changes allowed.
- Use "core" for facts that answer the question; use "supporting" for useful \
context.
- Prefer distinct facts. Do not repeat the same fact in several claims.
- Do not cite the same quote for more than one claim.
- If the candidates do not contain enough evidence, return fewer claims \
rather than inventing any.
"""


def prompt_sha256() -> str:
    payload = ANSWER_SYSTEM_PROMPT + "\n" + ANSWER_USER_TEMPLATE
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


ANSWER_PROMPT_SHA256 = prompt_sha256()


MAX_CANDIDATE_CHARS = 1800


def candidate_block(candidates: list[dict]) -> str:
    blocks: list[str] = []
    for number, candidate in enumerate(candidates, start=1):
        text = str(candidate.get("text", ""))
        if len(text) > MAX_CANDIDATE_CHARS:
            text = text[:MAX_CANDIDATE_CHARS].rstrip() + " ..."
        blocks.append(
            "\n".join(
                [
                    f"--- CANDIDATE {number} ---",
                    f"path: {candidate.get('source_path', '')}",
                    f"game: {candidate.get('game', '')}",
                    f"title: {candidate.get('page_title', '')}",
                    f"section: {candidate.get('section_key', '')}",
                    text,
                ]
            )
        )
    return "\n\n".join(blocks)


def build_answer_messages(
    *,
    question: str,
    game: str,
    entities: list[str],
    candidates: list[dict],
    min_claims: int,
    max_claims: int,
) -> list[dict[str, str]]:
    user = ANSWER_USER_TEMPLATE.format(
        question=question,
        game=game,
        entities=", ".join(entities) if entities else "(none)",
        min_claims=min_claims,
        max_claims=max_claims,
        candidates=candidate_block(candidates),
    )
    return [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
