import json

from src.explain import RecommendationExplanation
from src.report import format_json_report, format_text_report


def explanation() -> RecommendationExplanation:
    return RecommendationExplanation(
        final_rank=1,
        title_guess="《凡人修仙传》",
        novel_id="n1",
        confidence="high",
        why_recommended="Matches the requested cautious cultivation taste.",
        matched_preferences=["凡人流"],
        possible_risks=["sampled evidence only"],
        evidence=["ordinary protagonist"],
        user_takeaway="Strong candidate.",
        source_scores={"final_score": 0.84, "semantic_score": 0.66, "llm_match_score": 0.98},
    )


def test_text_report_includes_title_scores_explanation_risks_and_evidence() -> None:
    report = format_text_report("凡人流", [explanation()])
    assert "# Recommendation Report" in report
    assert "《凡人修仙传》" in report
    assert "Final score: 0.8400" in report
    assert "Matches the requested" in report
    assert "sampled evidence only" in report
    assert "ordinary protagonist" in report


def test_json_report_is_valid_json() -> None:
    report = format_json_report([explanation()])
    data = json.loads(report)
    assert data[0]["title_guess"] == "《凡人修仙传》"

