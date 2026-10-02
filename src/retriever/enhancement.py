import json
import logging

from src.config import settings
from src.llm.client import OpenAICompatibleAPI

logger = logging.getLogger("QueryEnhancement")

QUERY_ENHANCEMENT_PROMPT = """You are a query enhancement module for a retrieval system over a Dark Souls wiki knowledge base.

Rewrite the user's question into a single optimized search query that will retrieve the most relevant wiki passages.

Rules:
- Resolve pronouns and vague references into explicit entity names (e.g. "it" -> the thing the user means).
- Add well-known synonyms or alternative phrasings for key entities when they exist in Dark Souls lore (e.g. "Estus" -> "Estus Flask").
- Keep domain terminology such as boss names, item names, locations, and mechanics exactly as they appear in the game.
- Drop question words and conversational filler (who, what, was, is, did, "tell me about"). Keep only content words and entities.
- Preserve the original intent. Do not answer the question or add speculation.
- Output ONLY the rewritten query as plain text on a single line. No quotes, no explanations, no numbering.
Good example: "who was knight artorias?" -> "Knight Artorias Abysswalker lore history"

User question:
{query}
"""


def enhance_query(user_query: str) -> str:
    """Rewrite a user query with an LLM to improve vector retrieval.

    Falls back to the original query if enhancement fails.
    """
    model = OpenAICompatibleAPI(
        base_url="http://localhost:3001/v1",
        api_key=settings.freellm_api_key,
        model=settings.freellm_model,
        provider="FreeLlmApi",
    )

    messages = [
        {"role": "system", "content": "rewrite user questions into optimized search queries."},
        {"role": "user", "content": QUERY_ENHANCEMENT_PROMPT.format(query=user_query)},
    ]

    try:
        response = model.call(messages)
        content = response["choices"][0]["message"]["content"]
    except Exception as exc:
        logger.warning(f"Query enhancement failed, using original query: {exc}")
        return user_query

    enhanced = _extract_query(content)
    
    if not enhanced:
        logger.warning("Query enhancement returned empty output, using original query")
        return user_query

    return enhanced


def _extract_query(content: str) -> str:
    """Pull a single-line query out of a model response, tolerating quoting/json wrapping."""
    text = content.strip()

    if text.startswith("{"):
        try:
            data = json.loads(text)
            for value in data.values():
                if isinstance(value, str) and value.strip():
                    return " ".join(value.strip().splitlines()).strip()
        except json.JSONDecodeError:
            pass

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[-1].strip().strip('"').strip("'")
