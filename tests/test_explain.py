import json
from pathlib import Path

from src.explain import (
    ExplanationProgressEvent,
    build_explanation_prompt,
    explain_recommendations,
    fallback_explanation,
    parse_explanation_json,
    save_report,
)


class FakeGenerator:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        self.prompts.append(prompt)
        return self.response


def candidate() -> dict:
    return {
        "final_rank": 1,
        "title_guess": "《凡人修仙传》",
        "novel_id": "n1",
        "semantic_score": 0.66,
        "llm_match_score": 0.98,
        "final_score": 0.84,
        "confidence": "high",
        "matched_preferences": ["凡人流", "仙侠"],
        "violated_preferences": [],
        "risk_flags": [],
        "reason": "Matches cautious grassroots cultivation.",
        "profile_text_preview": "普通山村少年，谨慎修仙。",
    }


def test_prompt_includes_grounding_fields_and_rules() -> None:
    prompt = build_explanation_prompt("凡人流", candidate(), profile_text="profile evidence")
    assert "凡人流" in prompt
    assert "《凡人修仙传》" in prompt
    assert "0.66" in prompt
    assert "profile evidence" in prompt
    assert "Do not invent plot details" in prompt
    assert "Do not change the ranking" in prompt


def test_valid_json_explanation_parsing() -> None:
    parsed = parse_explanation_json('{"why_recommended":"good","matched_preferences":["x"],"possible_risks":[],"evidence":["e"],"user_takeaway":"try"}')
    assert parsed["why_recommended"] == "good"


def test_invalid_json_fallback() -> None:
    explanations, summary = explain_recommendations("q", [candidate()], FakeGenerator("not json"))
    assert summary.fallback_explanations == 1
    assert explanations[0].matched_preferences == ["凡人流", "仙侠"]


def test_fallback_includes_preferences_and_risks() -> None:
    item = candidate()
    item["violated_preferences"] = ["系统"]
    item["risk_flags"] = ["low_confidence"]
    explanation = fallback_explanation(item)
    assert "凡人流" in explanation.matched_preferences
    assert "系统" in explanation.possible_risks
    assert "low_confidence" in explanation.possible_risks


def test_progress_callback_receives_index_and_title() -> None:
    events: list[ExplanationProgressEvent] = []
    explain_recommendations(
        "q",
        [candidate()],
        FakeGenerator('{"why_recommended":"good","matched_preferences":[],"possible_risks":[],"evidence":[],"user_takeaway":"try"}'),
        progress_callback=events.append,
    )
    assert events[0].index == 1
    assert events[0].title == "《凡人修仙传》"
    assert events[-1].phase == "done"


def test_missing_optional_fields_do_not_crash() -> None:
    explanations, _ = explain_recommendations(
        "q",
        [{"title_guess": "T"}],
        FakeGenerator('{"why_recommended":"ok","matched_preferences":[],"possible_risks":[],"evidence":[],"user_takeaway":"ok"}'),
    )
    assert explanations[0].title_guess == "T"


def test_save_report_creates_file(tmp_path: Path) -> None:
    path = tmp_path / "reports" / "example.md"
    save_report(path, "content")
    assert path.read_text(encoding="utf-8") == "content"


def test_json_output_is_valid_via_explanations() -> None:
    explanations, _ = explain_recommendations(
        "q",
        [candidate()],
        FakeGenerator('{"why_recommended":"good","matched_preferences":[],"possible_risks":[],"evidence":[],"user_takeaway":"try"}'),
    )
    text = json.dumps([item.to_dict() for item in explanations], ensure_ascii=False)
    assert json.loads(text)[0]["title_guess"] == "《凡人修仙传》"

