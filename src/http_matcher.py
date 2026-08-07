"""OpenAI-compatible HTTP backend for candidate scoring, expansion, and explanation.

Speaks the ``/chat/completions`` protocol, so the same client drives a locally
served vLLM instance and a hosted gateway. Keeping the model behind HTTP means
this project never has to depend on ``vllm`` directly, which would drag in its own
torch pin and fight the cu128 / torch 2.11 constraint this host needs.

Unlike ``TransformersMatcher``, which generates one candidate at a time, this
backend exposes ``score_many`` for concurrent scoring — the point of moving off
in-process generation in the first place.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from src.llm_matcher import (
    LLMMatchResult,
    build_match_prompt,
    build_query_expansion_prompt,
    parse_llm_match_result,
)

DEFAULT_BASE_URL = os.environ.get("INOVELREC_LLM_BASE_URL", "http://127.0.0.1:8000/v1")
DEFAULT_API_KEY_ENV = "INOVELREC_LLM_API_KEY"
DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = 1.0
DEFAULT_MAX_WORKERS = 16
DEFAULT_MAX_NEW_TOKENS = 256
RETRY_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class TokenUsage:
    """Token counts as reported by the endpoint, for real (not estimated) costing."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


def extract_usage(data: dict[str, Any]) -> TokenUsage:
    """Read the ``usage`` block, tolerating endpoints that omit it."""

    usage = data.get("usage") or {}
    return TokenUsage(
        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
    )


class ChatTransport(Protocol):
    """Minimal chat-completion transport, so tests can stub out the network."""

    def complete(self, prompt: str, max_tokens: int) -> str:
        """Return the assistant message text for a single-turn prompt."""


@dataclass
class HTTPChatTransport:
    """POSTs to an OpenAI-compatible ``/chat/completions`` endpoint."""

    model: str
    base_url: str = DEFAULT_BASE_URL
    api_key: str | None = None
    timeout: float = DEFAULT_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS
    temperature: float = 0.0
    extra_body: dict[str, Any] = field(default_factory=dict)
    _client: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if self.api_key is None:
            self.api_key = os.environ.get(DEFAULT_API_KEY_ENV)
        if self.max_retries < 1:
            raise ValueError("max_retries must be at least 1")

    @property
    def client(self) -> Any:
        """Lazily create a pooled client so imports stay cheap."""

        if self._client is None:
            import httpx

            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            # Pool generously: score_many drives many concurrent requests.
            limits = httpx.Limits(max_connections=DEFAULT_MAX_WORKERS * 2, max_keepalive_connections=DEFAULT_MAX_WORKERS)
            self._client = httpx.Client(timeout=self.timeout, headers=headers, limits=limits)
        return self._client

    def close(self) -> None:
        """Close the underlying connection pool."""

        if self._client is not None:
            self._client.close()
            self._client = None

    def build_payload(self, prompt: str, max_tokens: int) -> dict[str, Any]:
        """Build the chat-completions request body."""

        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            **self.extra_body,
        }

    def complete(self, prompt: str, max_tokens: int) -> str:
        """Send one prompt, retrying transient failures with linear backoff."""

        return self.complete_with_usage(prompt, max_tokens)[0]

    def complete_with_usage(self, prompt: str, max_tokens: int) -> tuple[str, TokenUsage]:
        """Like ``complete``, but also returns the endpoint-reported token usage."""

        import httpx

        url = f"{self.base_url}/chat/completions"
        payload = self.build_payload(prompt, max_tokens)
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.post(url, json=payload)
                if response.status_code in RETRY_STATUS_CODES:
                    last_error = RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
                else:
                    response.raise_for_status()
                    data = response.json()
                    return extract_message_content(data), extract_usage(data)
            except httpx.HTTPError as exc:
                last_error = exc
            if attempt < self.max_retries - 1 and self.backoff_seconds:
                time.sleep(self.backoff_seconds * (attempt + 1))

        raise RuntimeError(f"Chat completion failed after {self.max_retries} attempts: {last_error}")


def extract_message_content(data: dict[str, Any]) -> str:
    """Pull the assistant text out of a chat-completions response."""

    choices = data.get("choices") or []
    if not choices:
        raise ValueError(f"No choices in chat completion response: {str(data)[:200]}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        raise ValueError("Chat completion response is missing message content")
    return str(content)


class OpenAICompatibleMatcher:
    """Drop-in replacement for ``TransformersMatcher`` backed by an HTTP endpoint.

    Satisfies the ``CandidateMatcher`` (``score``), ``LLMExpansionProvider``
    (``expand_queries``), and ``ExplanationGenerator`` (``generate``) protocols.
    """

    provider = "openai_compatible"

    def __init__(
        self,
        transport: ChatTransport,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        max_workers: int = DEFAULT_MAX_WORKERS,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self.transport = transport
        self.max_new_tokens = max_new_tokens
        self.max_workers = max_workers

    def generate_response(self, prompt: str, max_new_tokens: int | None = None) -> str:
        """Generate a single response, mirroring ``TransformersMatcher``."""

        return self.transport.complete(prompt, max_tokens=max_new_tokens or self.max_new_tokens)

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Satisfy the Stage 5 ``ExplanationGenerator`` protocol."""

        return self.generate_response(prompt, max_new_tokens=max_new_tokens)

    def score(
        self,
        query: str,
        candidate: dict[str, Any],
        profile_text: str,
        max_profile_chars: int = 1200,
    ) -> LLMMatchResult:
        """Score one candidate, degrading to a parse-failure result like the local matcher."""

        prompt = build_match_prompt(
            query=query,
            candidate=candidate,
            profile_text=profile_text,
            max_profile_chars=max_profile_chars,
        )
        response = self.generate_response(prompt)
        return parse_match_response(response)

    def score_many(
        self,
        query: str,
        items: list[tuple[dict[str, Any], str]],
        max_profile_chars: int = 1200,
        on_result: Callable[[int, LLMMatchResult | None], None] | None = None,
    ) -> list[LLMMatchResult | None]:
        """Score candidates concurrently, preserving input order.

        Returns ``None`` in a slot whose request failed outright, so the caller can
        distinguish "the model said this is a bad match" from "we never got an
        answer" and avoid caching the latter.
        """

        if not items:
            return []

        results: list[LLMMatchResult | None] = [None] * len(items)

        def run(index: int) -> None:
            candidate, profile_text = items[index]
            try:
                results[index] = self.score(query, candidate, profile_text, max_profile_chars)
            except Exception:  # noqa: BLE001 - a dead request must not kill the batch
                results[index] = None
            if on_result is not None:
                on_result(index, results[index])

        workers = min(self.max_workers, len(items))
        if workers == 1:
            for index in range(len(items)):
                run(index)
            return results

        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(run, range(len(items))))
        return results

    def expand_queries(self, raw_query: str, max_queries: int) -> str:
        """Generate retrieval-friendly expanded queries as raw JSON text."""

        prompt = build_query_expansion_prompt(raw_query=raw_query, max_queries=max_queries)
        return self.generate_response(prompt, max_new_tokens=min(self.max_new_tokens, 256))


def parse_match_response(response: str) -> LLMMatchResult:
    """Parse a scoring response, falling back exactly like ``TransformersMatcher``."""

    try:
        return parse_llm_match_result(response)
    except (ValueError, json.JSONDecodeError, TypeError):
        return LLMMatchResult(
            llm_match_score=0.0,
            confidence="low",
            risk_flags=["llm_parse_failed"],
            reason=response[:180],
        )


def create_openai_compatible_matcher(
    model: str,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    max_workers: int = DEFAULT_MAX_WORKERS,
    timeout: float = DEFAULT_TIMEOUT,
) -> OpenAICompatibleMatcher:
    """Create a matcher backed by an OpenAI-compatible endpoint."""

    transport = HTTPChatTransport(
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
    )
    return OpenAICompatibleMatcher(transport, max_new_tokens=max_new_tokens, max_workers=max_workers)
