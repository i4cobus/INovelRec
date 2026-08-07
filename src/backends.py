"""Backend selection for the LLM that scores, expands, and explains.

Two interchangeable implementations satisfy the same protocols:

``transformers``
    ``TransformersMatcher`` — loads weights in-process and generates one candidate
    at a time. No server needed, but throughput is poor.

``http``
    ``OpenAICompatibleMatcher`` — talks to any OpenAI-compatible ``/chat/completions``
    endpoint (a locally served vLLM, or a hosted gateway) and scores candidates
    concurrently via ``score_many``.

Model weights and network clients are imported lazily, so selecting one backend
never pays the import cost of the other.
"""

from __future__ import annotations

from typing import Any, Literal

Backend = Literal["transformers", "http"]
BACKENDS: tuple[Backend, ...] = ("transformers", "http")
DEFAULT_BACKEND: Backend = "transformers"


def normalize_backend(backend: str) -> Backend:
    """Validate a backend name coming from a CLI flag or UI control."""

    value = backend.strip().lower()
    if value not in BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}. Expected one of: {', '.join(BACKENDS)}")
    return value  # type: ignore[return-value]


def create_matcher(
    *,
    backend: str = DEFAULT_BACKEND,
    model_name: str,
    device: str | None = None,
    max_new_tokens: int = 256,
    base_url: str | None = None,
    api_key: str | None = None,
    max_workers: int | None = None,
    timeout: float | None = None,
) -> Any:
    """Build the matcher for the requested backend.

    The returned object satisfies ``CandidateMatcher``, ``LLMExpansionProvider``,
    and ``ExplanationGenerator``, so callers can use it for Stage 4 scoring, query
    expansion, and Stage 5 explanation alike.

    ``api_key`` defaults to the ``INOVELREC_LLM_API_KEY`` environment variable.
    Prefer that over a CLI flag: arguments are visible to every user via ``ps``.
    """

    resolved = normalize_backend(backend)

    if resolved == "transformers":
        from src.llm_matcher import create_transformers_matcher

        return create_transformers_matcher(
            model_name=model_name,
            device=device,
            max_new_tokens=max_new_tokens,
        )

    from src.http_matcher import (
        DEFAULT_BASE_URL,
        DEFAULT_MAX_WORKERS,
        DEFAULT_TIMEOUT,
        create_openai_compatible_matcher,
    )

    return create_openai_compatible_matcher(
        model=model_name,
        base_url=base_url or DEFAULT_BASE_URL,
        api_key=api_key,
        max_new_tokens=max_new_tokens,
        max_workers=DEFAULT_MAX_WORKERS if max_workers is None else max_workers,
        timeout=DEFAULT_TIMEOUT if timeout is None else timeout,
    )


def backend_uses_gpu(backend: str) -> bool:
    """Return True when the backend holds model weights in this process.

    Callers that free VRAM between stages (``clear_cuda_if_needed``) only need to
    do so for in-process backends; an HTTP backend owns no local memory.
    """

    return normalize_backend(backend) == "transformers"


def as_explanation_generator(matcher: Any) -> Any:
    """Adapt a matcher to the Stage 5 ``ExplanationGenerator`` protocol.

    The HTTP matcher already implements ``generate``; the transformers matcher
    needs the thin ``TransformersExplanationGenerator`` wrapper.
    """

    if callable(getattr(matcher, "generate", None)):
        return matcher

    from src.llm_explain import TransformersExplanationGenerator

    return TransformersExplanationGenerator(matcher)
