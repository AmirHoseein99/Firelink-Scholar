"""Minimal OpenAI-compatible chat client used by the answer builder and judge.

No retriever or Qdrant singletons are touched here. Backends resolve to the
``FREELLM_*`` / ``NEURALWATT_*`` / ``OPENROUTER_*`` settings, with Free-LLM
being the benchmark default.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from src.config import settings


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class Backend:
    name: str
    base_url: str
    api_key: str
    model: str

    @property
    def url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"


def resolve_backend(name: str) -> Backend:
    if name == "freellm":
        return Backend(
            name=name,
            base_url=settings.freellm_base_url,
            api_key=settings.freellm_api_key,
            model=settings.freellm_model,
        )
    if name == "neuralwatt":
        return Backend(
            name=name,
            base_url=settings.NEURALWATT_BASE_URL,
            api_key=settings.NEURALWATT_API_KEY,
            model=settings.NEURALWATT_MODEL,
        )
    if name == "openrouter":
        return Backend(
            name=name,
            base_url=settings.OPENROUTER_BASE_URL,
            api_key=settings.OPENROUTER_API_KEY,
            model=settings.OPENROUTER_MODEL,
        )
    raise LLMError(f"unknown LLM backend: {name!r}")


def require_backend(name: str) -> Backend:
    backend = resolve_backend(name)
    if not backend.api_key:
        raise LLMError(
            f"API key for backend {name!r} is empty. Set "
            f"{name.upper()}_API_KEY in .env."
        )
    if not backend.model:
        raise LLMError(
            f"Model for backend {name!r} is empty. Set {name.upper()}_MODEL "
            f"in .env."
        )
    return backend


def _headers(backend: Backend) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {backend.api_key}",
    }


def _payload(
    backend: Backend,
    messages: list[dict[str, str]],
    *,
    temperature: float,
) -> dict[str, Any]:
    return {
        "model": backend.model,
        "messages": messages,
        "temperature": temperature,
    }


def _content_of(data: dict[str, Any]) -> str:
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"malformed completion response: {data!r}") from exc
    if not isinstance(content, str):
        raise LLMError(f"completion content is not text: {content!r}")
    return content


def chat(
    messages: list[dict[str, str]],
    *,
    backend: Backend,
    temperature: float = 0.0,
    timeout: float = 180.0,
    retries: int = 3,
) -> str:
    """Synchronous chat completion with retry on rate limits and 5xx."""
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        response: httpx.Response | None = None
        try:
            response = httpx.post(
                backend.url,
                headers=_headers(backend),
                json=_payload(backend, messages, temperature=temperature),
                timeout=timeout,
            )
        except httpx.HTTPError as exc:
            last_error = exc
        else:
            if response.status_code == 200:
                return _content_of(response.json())
            if response.status_code == 429 or response.status_code >= 500:
                last_error = LLMError(
                    f"{backend.name} returned {response.status_code}: "
                    f"{response.text[:500]}"
                )
            else:
                raise LLMError(
                    f"{backend.name} returned {response.status_code}: "
                    f"{response.text[:500]}"
                )
        if attempt < retries:
            delay = float(2 * (attempt + 1))
            if response is not None:
                try:
                    delay = float(response.headers.get("Retry-After", delay))
                except (TypeError, ValueError):
                    pass
            time.sleep(min(delay, 30.0))
    raise LLMError(f"{backend.name} request failed after retries: {last_error}")


async def achat(
    messages: list[dict[str, str]],
    *,
    backend: Backend,
    temperature: float = 0.0,
    timeout: float = 180.0,
    retries: int = 3,
) -> str:
    """Async chat completion with retry on rate limits and 5xx."""
    last_error: Exception | None = None
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(retries + 1):
            try:
                response = await client.post(
                    backend.url,
                    headers=_headers(backend),
                    json=_payload(backend, messages, temperature=temperature),
                )
            except httpx.HTTPError as exc:
                last_error = exc
            else:
                if response.status_code == 200:
                    return _content_of(response.json())
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = LLMError(
                        f"{backend.name} returned {response.status_code}: "
                        f"{response.text[:500]}"
                    )
                else:
                    raise LLMError(
                        f"{backend.name} returned {response.status_code}: "
                        f"{response.text[:500]}"
                    )
            if attempt < retries:
                delay = 2 * (attempt + 1)
                if "response" in locals() and response is not None:
                    try:
                        delay = float(response.headers.get("Retry-After", delay))
                    except (TypeError, ValueError):
                        pass
                await asyncio.sleep(min(delay, 30.0))
    raise LLMError(f"{backend.name} request failed after retries: {last_error}")


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", flags=re.IGNORECASE)


def extract_json(raw: str) -> Any:
    """Parse a JSON object out of a model response, tolerating fences."""
    text = raw.strip()
    if text.startswith("```"):
        text = _FENCE_RE.sub("", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise LLMError(f"response contains no JSON object: {raw[:300]!r}")
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"response JSON is malformed: {raw[:300]!r}"
            ) from exc
