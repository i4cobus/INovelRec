"""Grounded recommendation explanation generation."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class RecommendationExplanation:
    """User-facing explanation for one ranked recommendation."""

    final_rank: int
    title_guess: str
    novel_id: str
    confidence: str
    why_recommended: str
    matched_preferences: list[str] = field(default_factory=list)
    possible_risks: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    user_takeaway: str = ""
    source_scores: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExplanationProgressEvent:
    index: int
    total: int
    title: str
    elapsed_seconds: float = 0.0
    confidence: str | None = None
    phase: str = "start"


@dataclass(frozen=True)
class ExplanationSummary:
    explained_candidates: int
    successful_llm_json_outputs: int
    fallback_explanations: int
    total_explanation_time: float
    average_time_per_candidate: float


class ExplanationGenerator(Protocol):
    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Generate explanation JSON text."""


ProgressCallback = Callable[[ExplanationProgressEvent], None]


def candidate_value(candidate: dict[str, Any], key: str, default: Any = "") -> Any:
    return candidate.get(key, default)


def build_explanation_prompt(
    query: str,
    candidate: dict[str, Any],
    profile_text: str = "",
    max_profile_chars: int = 1200,
) -> str:
    """Build a grounded JSON-only explanation prompt for one candidate."""

    profile = (profile_text or str(candidate.get("profile_text_preview", "")))[:max_profile_chars]
    return (
        "You are explaining an existing Chinese web novel recommendation result.\n"
        "Do not change the ranking. Use only the provided evidence.\n"
        "Do not invent plot details, popularity, author facts, ratings, or completion status.\n"
        "If evidence is limited, state uncertainty. Keep the explanation concise.\n"
        "Output valid JSON only. Do not use markdown.\n"
        "Important: the system uses sampled profile text, not full human reading.\n\n"
        f"Original user query: {query}\n"
        f"Final rank: {candidate_value(candidate, 'final_rank', '')}\n"
        f"Title: {candidate_value(candidate, 'title_guess', '')}\n"
        f"Novel ID: {candidate_value(candidate, 'novel_id', '')}\n"
        f"Semantic score: {candidate_value(candidate, 'semantic_score', '')}\n"
        f"LLM match score: {candidate_value(candidate, 'llm_match_score', '')}\n"
        f"Final score: {candidate_value(candidate, 'final_score', '')}\n"
        f"Confidence: {candidate_value(candidate, 'confidence', '')}\n"
        f"Matched preferences: {candidate_value(candidate, 'matched_preferences', [])}\n"
        f"Violated preferences: {candidate_value(candidate, 'violated_preferences', [])}\n"
        f"Risk flags: {candidate_value(candidate, 'risk_flags', [])}\n"
        f"Stage 4 reason: {candidate_value(candidate, 'reason', '')}\n"
        f"Profile evidence: {profile}\n\n"
        "Expected JSON:\n"
        '{"why_recommended":"...","matched_preferences":["..."],"possible_risks":["..."],'
        '"evidence":["..."],"user_takeaway":"..."}'
    )


def extract_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("No JSON object found")
    return json.loads(match.group(0))


def parse_explanation_json(text: str) -> dict[str, Any]:
    """Parse direct JSON first, then first embedded JSON object."""

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return extract_json_object(text)


def source_scores(candidate: dict[str, Any]) -> dict[str, Any]:
    """Collect source scores without requiring all fields."""

    return {
        "final_score": candidate.get("final_score"),
        "semantic_score": candidate.get("semantic_score"),
        "llm_match_score": candidate.get("llm_match_score"),
        "confidence_score": candidate.get("confidence_score"),
        "risk_penalty": candidate.get("risk_penalty"),
        "faiss_rank": candidate.get("faiss_rank"),
        "best_faiss_rank": candidate.get("best_faiss_rank"),
        "matched_query_count": candidate.get("matched_query_count"),
        "retrieval_sources": candidate.get("retrieval_sources"),
    }


def fallback_explanation(candidate: dict[str, Any]) -> RecommendationExplanation:
    """Create a deterministic explanation from Stage 4 fields."""

    title = str(candidate.get("title_guess", "Unknown title"))
    matched = [str(item) for item in candidate.get("matched_preferences", [])]
    risks = [str(item) for item in [*candidate.get("violated_preferences", []), *candidate.get("risk_flags", [])] if str(item)]
    reason = str(candidate.get("reason", "") or "Stage 4 selected this novel based on retrieval and local LLM matching signals.")
    evidence = []
    if reason:
        evidence.append(reason)
    preview = str(candidate.get("profile_text_preview", "")).strip()
    if preview:
        evidence.append(preview[:180])
    return RecommendationExplanation(
        final_rank=int(candidate.get("final_rank", 0) or 0),
        title_guess=title,
        novel_id=str(candidate.get("novel_id", "")),
        confidence=str(candidate.get("confidence", "unknown") or "unknown"),
        why_recommended=reason,
        matched_preferences=matched,
        possible_risks=risks or ["Evidence is based on sampled profile text, not full human reading."],
        evidence=evidence,
        user_takeaway=f"Consider {title} if the sampled profile evidence matches your requested taste.",
        source_scores=source_scores(candidate),
    )


def explanation_from_llm_data(candidate: dict[str, Any], data: dict[str, Any]) -> RecommendationExplanation:
    """Combine parsed LLM explanation JSON with required source fields."""

    return RecommendationExplanation(
        final_rank=int(candidate.get("final_rank", 0) or 0),
        title_guess=str(candidate.get("title_guess", "")),
        novel_id=str(candidate.get("novel_id", "")),
        confidence=str(candidate.get("confidence", "unknown") or "unknown"),
        why_recommended=str(data.get("why_recommended", "")),
        matched_preferences=[str(item) for item in data.get("matched_preferences", candidate.get("matched_preferences", []))],
        possible_risks=[str(item) for item in data.get("possible_risks", candidate.get("risk_flags", []))],
        evidence=[str(item) for item in data.get("evidence", [])],
        user_takeaway=str(data.get("user_takeaway", "")),
        source_scores=source_scores(candidate),
    )


def explain_recommendations(
    query: str,
    candidates: list[dict[str, Any]],
    generator: ExplanationGenerator,
    profile_lookup: dict[str, str] | None = None,
    max_profile_chars: int = 1200,
    max_new_tokens: int = 512,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[RecommendationExplanation], ExplanationSummary]:
    """Generate explanations for ranked candidates with fallback safety."""

    profile_lookup = profile_lookup or {}
    explanations: list[RecommendationExplanation] = []
    successes = 0
    fallbacks = 0
    started = time.perf_counter()

    for idx, candidate in enumerate(candidates, start=1):
        title = str(candidate.get("title_guess", ""))
        if progress_callback:
            progress_callback(ExplanationProgressEvent(index=idx, total=len(candidates), title=title, phase="start"))
        item_started = time.perf_counter()
        profile_text = profile_lookup.get(str(candidate.get("novel_id", "")), str(candidate.get("profile_text_preview", "")))
        prompt = build_explanation_prompt(query, candidate, profile_text=profile_text, max_profile_chars=max_profile_chars)
        try:
            raw = generator.generate(prompt, max_new_tokens=max_new_tokens)
            parsed = parse_explanation_json(raw)
            explanation = explanation_from_llm_data(candidate, parsed)
            successes += 1
        except Exception:
            explanation = fallback_explanation(candidate)
            fallbacks += 1
        elapsed = time.perf_counter() - item_started
        explanations.append(explanation)
        if progress_callback:
            progress_callback(
                ExplanationProgressEvent(
                    index=idx,
                    total=len(candidates),
                    title=title,
                    elapsed_seconds=elapsed,
                    confidence=explanation.confidence,
                    phase="done",
                )
            )

    total = time.perf_counter() - started
    return explanations, ExplanationSummary(
        explained_candidates=len(candidates),
        successful_llm_json_outputs=successes,
        fallback_explanations=fallbacks,
        total_explanation_time=total,
        average_time_per_candidate=total / len(candidates) if candidates else 0.0,
    )


def save_report(path: Path, content: str) -> None:
    """Save a report, creating parent directories."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

