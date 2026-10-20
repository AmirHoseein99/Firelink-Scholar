import json
import logging
import re
from functools import lru_cache

from src.config import settings
from src.llm.client import OpenAICompatibleAPI

logger = logging.getLogger("QueryEnhancement")

MAX_QUERIES = 3

QUERY_ENHANCEMENT_PROMPT = """You rewrite user questions into search queries for a Dark Souls wiki.

Return ONLY a JSON object of the form {{"queries": ["...", "..."]}}.

Rules:
- Produce 1 to {max_queries} queries.
- The first query must cover the whole question. Later queries may focus on one
  entity, comparison side, or step at a time.
- Preserve every named entity, boss, NPC, location, item, and game title exactly
  as the user wrote it.
- For relationship, comparison, or multi-entity questions, include one query per
  entity plus one combined query.
- For how/why/sequence questions, keep the causal or sequence terms instead of
  collapsing the question into a single topic.
- Each query must be self-contained: replace pronouns with the entity they refer
  to.
- Do not answer the question, add facts, or explain your reasoning.
- Remove only conversational filler such as "tell me about" or "what is".
- Keep the user's vocabulary; add at most one well-known alias when unambiguous.

Question:
{query}
"""


class EnhancementError(RuntimeError):
    """Raised when a query rewrite cannot be produced."""


def _backend() -> OpenAICompatibleAPI:
    """Return the configured rewrite backend, preferring OpenRouter."""
    if settings.OPENROUTER_API_KEY and settings.OPENROUTER_MODEL:
        return OpenAICompatibleAPI(
            base_url=settings.OPENROUTER_BASE_URL,
            api_key=settings.OPENROUTER_API_KEY,
            model=settings.OPENROUTER_MODEL,
            provider="OpenRouter",
        )
    if settings.NEURALWATT_API_KEY and settings.NEURALWATT_MODEL:
        return OpenAICompatibleAPI(
            base_url=settings.NEURALWATT_BASE_URL,
            api_key=settings.NEURALWATT_API_KEY,
            model=settings.NEURALWATT_MODEL,
            provider="NeuralWatt",
        )
    raise EnhancementError(
        "no query-enhancement backend configured; set OPENROUTER_* or NEURALWATT_*"
    )


@lru_cache(maxsize=2048)
def _enhance_queries_cached(user_query: str, max_queries: int) -> tuple[str, ...]:
    model = _backend()
    messages = [
        {
            "role": "system",
            "content": "rewrite user questions into search queries",
        },
        {
            "role": "user",
            "content": QUERY_ENHANCEMENT_PROMPT.format(
                query=user_query,
                max_queries=max_queries,
            ),
        },
    ]
    response = model.call(messages, temperature=0.0, max_tokens=512)
    content = _content_of(response)
    queries = _extract_queries(content, max_queries=max_queries)
    if not queries:
        # Retry once: providers occasionally return truncated or empty content.
        response = model.call(messages, temperature=0.0, max_tokens=512)
        content = _content_of(response)
        queries = _extract_queries(content, max_queries=max_queries)
    if not queries:
        raise EnhancementError(f"enhancement returned no usable query: {content[:200]!r}")
    logger.debug("enhanced %r -> %r", user_query, queries)
    return tuple(queries)


def _content_of(response: dict) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    return content if isinstance(content, str) else ""


def enhance_queries(
    user_query: str,
    *,
    max_queries: int = MAX_QUERIES,
) -> list[str]:
    """Rewrite a question into one or more self-contained retrieval queries.

    The rewrite is deterministic (temperature 0) and cached by question text,
    so repeated runs and different retrieval depths see the same queries.
    Falls back to the original query when enhancement is unavailable.
    """
    max_queries = max(1, int(max_queries))
    try:
        return list(_enhance_queries_cached(user_query, max_queries))
    except Exception as exc:  # noqa: BLE001 - retrieval must degrade, not fail
        logger.warning(
            "Query enhancement failed, using original query: %s", exc
        )
        return [user_query]


def enhance_query(user_query: str) -> str:
    """Return the combined rewritten query (backward-compatible wrapper)."""
    return enhance_queries(user_query)[0]


def _extract_queries(content: str, *, max_queries: int = MAX_QUERIES) -> list[str]:
    """Pull queries out of a model response, tolerating JSON and plain text."""
    text = content.strip()
    if not text:
        return []

    candidates = _json_queries(text)
    if not candidates:
        if _json_like(text):
            # A JSON response that failed to parse is likely truncated; do not
            # treat its raw text as a search query.
            return []
        candidates = _line_queries(text)

    queries: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        query = _clean_query(candidate)
        if not query:
            continue
        key = query.casefold()
        if key in seen:
            continue
        seen.add(key)
        queries.append(query)
        if len(queries) >= max_queries:
            break
    return queries


def _json_queries(text: str) -> list[str]:
    data = _parse_json(text)
    if data is None:
        return []

    values = data
    if isinstance(data, dict):
        values = []
        for key in ("queries", "rewritten_queries", "query"):
            value = data.get(key)
            if isinstance(value, str):
                return [value]
            if isinstance(value, list):
                values = value
                break
        else:
            values = [value for value in data.values() if isinstance(value, str)]

    if isinstance(values, str):
        return [values]
    if not isinstance(values, list):
        return []
    return [value for value in values if isinstance(value, str)]


def _parse_json(text: str) -> object | None:
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    for candidate in (fenced, text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    start = fenced.find("{")
    end = fenced.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(fenced[start : end + 1])
        except json.JSONDecodeError:
            pass
    start = fenced.find("[")
    end = fenced.rfind("]")
    if start != -1 and end > start:
        try:
            return json.loads(fenced[start : end + 1])
        except json.JSONDecodeError:
            pass
    return None


_BULLET_RE = re.compile(r"^(?:[-*+]|\d+[.)])\s+")
_JSON_START_RE = re.compile(r"^\s*[\[{]")
_JSON_HINT_RE = re.compile(r'"queries"\s*:')


def _json_like(text: str) -> bool:
    return bool(_JSON_START_RE.match(text) or _JSON_HINT_RE.search(text))


def _line_queries(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    lines = [
        line
        for line in lines
        if not line.startswith(("{", "[", "]", "}")) and '"queries"' not in line
    ]
    if not lines:
        return []
    if len(lines) == 1:
        return lines
    return [
        line for line in lines if line.startswith(("-", "*", "+")) or _BULLET_RE.match(line)
    ] or [lines[0]]


def _clean_query(value: str) -> str:
    query = " ".join(value.strip().splitlines()).strip()
    query = _BULLET_RE.sub("", query)
    query = re.sub(r"^(?:query|rewritten query)\s*:\s*", "", query, flags=re.IGNORECASE)
    query = query.strip().strip('"').strip("'").strip()
    if not query or len(query) > 300:
        return ""
    if query.startswith(("{", "[")) or '"queries"' in query:
        return ""
    if not any(character.isalnum() for character in query):
        return ""
    return query
