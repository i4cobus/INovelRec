"""Transformers-backed explanation generator adapters."""

from __future__ import annotations

from src.llm_matcher import DEFAULT_LLM_MODEL, TransformersMatcher, create_transformers_matcher


class TransformersExplanationGenerator:
    """Reuse a local transformers chat model for explanation JSON generation."""

    def __init__(self, matcher: TransformersMatcher) -> None:
        self.matcher = matcher

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        return self.matcher.generate_response(prompt, max_new_tokens=max_new_tokens)


def create_explanation_generator(
    model_name: str = DEFAULT_LLM_MODEL,
    device: str | None = None,
    max_new_tokens: int = 512,
) -> TransformersExplanationGenerator:
    """Create a standalone explanation generator."""

    matcher = create_transformers_matcher(model_name=model_name, device=device, max_new_tokens=max_new_tokens)
    return TransformersExplanationGenerator(matcher)

