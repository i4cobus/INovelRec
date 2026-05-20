"""Formatting for user-facing recommendation reports."""

from __future__ import annotations

import json
from typing import Literal

from src.explain import RecommendationExplanation


def format_score(value: object) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "-"


def format_text_report(query: str, explanations: list[RecommendationExplanation]) -> str:
    """Format explanations as a concise readable text/markdown report."""

    lines = ["# Recommendation Report", "", "Query:", query, ""]
    for explanation in explanations:
        scores = explanation.source_scores
        lines.extend(
            [
                f"## {explanation.final_rank}. {explanation.title_guess}",
                "",
                f"Final score: {format_score(scores.get('final_score'))}",
                f"Semantic score: {format_score(scores.get('semantic_score'))}",
                f"LLM match score: {format_score(scores.get('llm_match_score'))}",
                f"Confidence: {explanation.confidence}",
                "",
                "Why recommended:",
                explanation.why_recommended or "The explanation model did not provide a reason.",
                "",
                "Matched preferences:",
            ]
        )
        lines.extend(f"- {item}" for item in (explanation.matched_preferences or ["No explicit matched preferences were returned."]))
        lines.extend(["", "Possible risks:"])
        lines.extend(f"- {item}" for item in (explanation.possible_risks or ["No specific risks were returned."]))
        lines.extend(["", "Evidence:"])
        lines.extend(f"- {item}" for item in (explanation.evidence or ["Evidence is limited to sampled profile text and Stage 4 scoring fields."]))
        lines.extend(["", "User takeaway:", explanation.user_takeaway or "Use this as a candidate to inspect further.", ""])
    return "\n".join(lines).strip() + "\n"


def format_json_report(explanations: list[RecommendationExplanation]) -> str:
    """Format explanations as JSON."""

    return json.dumps([explanation.to_dict() for explanation in explanations], ensure_ascii=False, indent=2)


def format_report(query: str, explanations: list[RecommendationExplanation], output_format: Literal["text", "json", "markdown"]) -> str:
    """Format a report in text, markdown, or JSON."""

    if output_format == "json":
        return format_json_report(explanations)
    return format_text_report(query, explanations)

