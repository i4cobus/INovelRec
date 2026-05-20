from pathlib import Path

from src.app_pipeline import PipelineSettings, retrieval_summary, run_discovery_pipeline, validate_runtime_files
from src.llm_matcher import LLMMatchResult
from src.query_expansion import ExpandedQuery


class FakeMatcher:
    provider = "mock"

    def score(self, query, candidate, profile_text, max_profile_chars=1200):
        return LLMMatchResult(
            llm_match_score=0.9,
            confidence="high",
            matched_preferences=["凡人流"],
            reason="Matches the sampled profile evidence.",
        )

    def generate_response(self, prompt: str, max_new_tokens: int = 512) -> str:
        return (
            '{"why_recommended":"Grounded reason","matched_preferences":["凡人流"],'
            '"possible_risks":["sampled evidence only"],"evidence":["profile evidence"],'
            '"user_takeaway":"Worth checking."}'
        )


def touch_runtime_files(tmp_path: Path) -> PipelineSettings:
    index = tmp_path / "faiss.index"
    id_map = tmp_path / "novel_id_map.json"
    profiles = tmp_path / "novel_profiles.parquet"
    index.write_bytes(b"index")
    id_map.write_text("{}", encoding="utf-8")
    profiles.write_text("", encoding="utf-8")
    return PipelineSettings(
        candidate_k=2,
        top_k_per_query=2,
        llm_candidate_k=1,
        top_k=1,
        use_query_expansion=False,
        use_domain_hints=False,
        index_path=index,
        id_map_path=id_map,
        profiles_path=profiles,
    )


def test_validate_runtime_files_reports_missing(tmp_path: Path) -> None:
    settings = PipelineSettings(index_path=tmp_path / "missing.index", id_map_path=tmp_path / "missing.json", profiles_path=tmp_path / "missing.parquet")
    messages = validate_runtime_files(settings)
    assert len(messages) == 3
    assert "FAISS index not found" in messages[0]


def test_retrieval_summary_counts_sources() -> None:
    summary = retrieval_summary(
        [ExpandedQuery("raw", "raw", 1.0)],
        [{"retrieval_sources": ["raw", "domain_hints"]}, {"retrieval_sources": ["raw"]}],
        PipelineSettings(candidate_k=2, top_k_per_query=5),
    )
    assert summary["expanded_query_count"] == 1
    assert summary["candidate_sources"]["raw"] == 2
    assert summary["candidate_sources"]["domain_hints"] == 1


def test_run_discovery_pipeline_accepts_mock_output(monkeypatch, tmp_path: Path) -> None:
    settings = touch_runtime_files(tmp_path)

    def fake_search(**kwargs):
        return [
            {
                "rank": 1,
                "score": 0.8,
                "novel_id": "n1",
                "title_guess": "凡人修仙传",
                "profile_text_preview": "普通少年修仙，谨慎成长。",
                "retrieval_sources": ["raw"],
                "matched_query_count": 1,
            }
        ]

    monkeypatch.setattr("src.app_pipeline.multi_query_semantic_search", fake_search)
    progress_events = []
    result = run_discovery_pipeline(
        query="凡人流 仙侠",
        settings=settings,
        embedder=object(),
        matcher=FakeMatcher(),
        index=object(),
        id_map={},
        profile_lookup={"n1": "普通少年修仙，谨慎成长。"},
        progress_callback=lambda stage, message, value: progress_events.append((stage, message, value)),
    )
    assert result.explanations[0].title_guess == "凡人修仙传"
    assert "Grounded reason" in result.report_markdown
    assert result.raw_dict()["ranked_candidates"][0]["selected_for_llm"] is True
    assert any(event[0] == "explain" for event in progress_events)
