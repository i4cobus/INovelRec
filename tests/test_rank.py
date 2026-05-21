from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.llm_matcher import LLMMatchResult, build_match_prompt, parse_llm_match_result
from src.rank import (
    LLMProgressEvent,
    RankingWeights,
    compute_risk_penalty,
    llm_final_score,
    rerank_candidates_with_llm,
    resolve_llm_candidate_k,
    select_llm_candidates,
    semantic_fallback_score,
    truncate_profile,
)


@dataclass
class CountingMatcher:
    result: LLMMatchResult
    calls: int = 0
    provider: str = "mock"

    def score(self, query: str, candidate: dict[str, Any], profile_text: str, max_profile_chars: int = 1200) -> LLMMatchResult:
        self.calls += 1
        assert len(profile_text) <= max_profile_chars
        return self.result


def candidates(count: int) -> list[dict[str, Any]]:
    return [
        {
            "rank": idx + 1,
            "score": 0.9 - (idx * 0.01),
            "novel_id": f"n{idx}",
            "title_guess": f"Title {idx}",
            "profile_text_preview": "profile text " * 20,
        }
        for idx in range(count)
    ]


def test_llm_candidate_k_limits_llm_calls(tmp_path: Path) -> None:
    matcher = CountingMatcher(LLMMatchResult(llm_match_score=0.8, confidence="high"))
    ranked, timing = rerank_candidates_with_llm(
        query="query",
        candidates=candidates(5),
        matcher=matcher,
        llm_candidate_k=2,
        use_cache=False,
        cache_path=tmp_path / "cache.jsonl",
    )
    assert matcher.calls == 2
    assert timing.llm_candidate_k == 2
    assert sum(1 for row in ranked if row["selected_for_llm"]) == 2


def test_llm_candidate_k_clamps_when_greater_than_candidate_k() -> None:
    resolved, warning = resolve_llm_candidate_k(candidate_k=3, llm_candidate_k=10)
    assert resolved == 3
    assert warning is not None


def test_non_selected_candidates_receive_semantic_fallback(tmp_path: Path) -> None:
    matcher = CountingMatcher(LLMMatchResult(llm_match_score=0.9, confidence="high"))
    ranked, _ = rerank_candidates_with_llm(
        query="query",
        candidates=candidates(3),
        matcher=matcher,
        llm_candidate_k=1,
        use_cache=False,
        cache_path=tmp_path / "cache.jsonl",
    )
    fallback = [row for row in ranked if not row["selected_for_llm"]]
    assert fallback
    assert {row["analysis_provider"] for row in fallback} == {"semantic_fallback"}
    assert all(row["llm_match_score"] is None for row in fallback)


def test_llm_profile_max_chars_truncates_prompt_input() -> None:
    profile = "x" * 200
    assert truncate_profile(profile, 50) == "x" * 50
    prompt = build_match_prompt("query", {"title_guess": "T", "score": 0.1}, profile, max_profile_chars=50)
    assert ("x" * 60) not in prompt


def test_cache_hit_avoids_llm_call(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.jsonl"
    matcher = CountingMatcher(LLMMatchResult(llm_match_score=0.7, confidence="medium"))
    first, first_timing = rerank_candidates_with_llm(
        query="query",
        candidates=candidates(1),
        matcher=matcher,
        llm_candidate_k=1,
        cache_path=cache_path,
    )
    second, second_timing = rerank_candidates_with_llm(
        query="query",
        candidates=candidates(1),
        matcher=matcher,
        llm_candidate_k=1,
        cache_path=cache_path,
    )
    assert matcher.calls == 1
    assert first_timing.cache_misses == 1
    assert second_timing.cache_hits == 1
    assert second[0]["analysis_provider"] == "cache"
    assert first[0]["analysis_provider"] == "mock"


def test_cache_miss_calls_llm(tmp_path: Path) -> None:
    matcher = CountingMatcher(LLMMatchResult(llm_match_score=0.7, confidence="medium"))
    _, timing = rerank_candidates_with_llm(
        query="query",
        candidates=candidates(1),
        matcher=matcher,
        llm_candidate_k=1,
        cache_path=tmp_path / "cache.jsonl",
    )
    assert matcher.calls == 1
    assert timing.cache_misses == 1


def test_candidate_output_includes_stage4_fields(tmp_path: Path) -> None:
    matcher = CountingMatcher(LLMMatchResult(llm_match_score=0.7, confidence="medium"))
    ranked, _ = rerank_candidates_with_llm(
        query="query",
        candidates=candidates(1),
        matcher=matcher,
        llm_candidate_k=1,
        use_cache=False,
        cache_path=tmp_path / "cache.jsonl",
    )
    row = ranked[0]
    assert {"selected_for_llm", "cache_hit", "analysis_provider"}.issubset(row)


def test_timing_summary_contains_expected_fields(tmp_path: Path) -> None:
    matcher = CountingMatcher(LLMMatchResult(llm_match_score=0.7, confidence="medium"))
    _, timing = rerank_candidates_with_llm(
        query="query",
        candidates=candidates(2),
        matcher=matcher,
        llm_candidate_k=1,
        use_cache=False,
        cache_path=tmp_path / "cache.jsonl",
    )
    assert hasattr(timing, "llm_scoring")
    assert hasattr(timing, "average_llm_scoring_time")
    assert hasattr(timing, "cache_hits")
    assert hasattr(timing, "final_reranking")


def test_progress_callback_receives_candidate_details(tmp_path: Path) -> None:
    events: list[LLMProgressEvent] = []
    matcher = CountingMatcher(LLMMatchResult(llm_match_score=0.7, confidence="medium"))
    rerank_candidates_with_llm(
        query="query",
        candidates=candidates(1),
        matcher=matcher,
        llm_candidate_k=1,
        use_cache=False,
        cache_path=tmp_path / "cache.jsonl",
        progress_callback=events.append,
    )
    assert events[0].index == 1
    assert events[0].title == "Title 0"
    assert events[0].cache_status == "cache miss"
    assert events[-1].llm_match_score == 0.7


def test_final_score_calculation() -> None:
    match = LLMMatchResult(llm_match_score=0.8, confidence="high")
    score = llm_final_score(0.5, match, risk_penalty=0.1, weights=RankingWeights())
    assert round(score, 4) == 0.6


def test_invalid_llm_json_falls_back_safely() -> None:
    try:
        parse_llm_match_result("not json")
    except ValueError as exc:
        assert "No JSON object" in str(exc)


def test_risk_penalty_rules() -> None:
    match = LLMMatchResult(
        llm_match_score=0.5,
        confidence="low",
        violated_preferences=["系统"],
        risk_flags=["source_site_boilerplate"],
    )
    assert compute_risk_penalty(match, "profile") == 0.25


def test_high_semantic_candidate_selected_for_llm() -> None:
    items = [
        {"novel_id": "retrieval", "title_guess": "Retrieval", "retrieval_score": 0.95, "score": 0.4, "best_faiss_rank": 20},
        {"novel_id": "semantic", "title_guess": "Semantic", "retrieval_score": 0.5, "score": 0.99, "best_faiss_rank": 15},
        {"novel_id": "other", "title_guess": "Other", "retrieval_score": 0.49, "score": 0.3, "best_faiss_rank": 10},
    ]
    selected = select_llm_candidates(items, llm_candidate_k=2)
    hit = next(item for item in selected if item["novel_id"] == "semantic")
    assert "semantic_score_top" in hit["llm_selection_reasons"]


def test_best_faiss_rank_one_selected_for_llm() -> None:
    items = [
        {"novel_id": "a", "title_guess": "A", "retrieval_score": 0.95, "score": 0.9, "best_faiss_rank": 20},
        {"novel_id": "b", "title_guess": "B", "retrieval_score": 0.9, "score": 0.8, "best_faiss_rank": 30},
        {"novel_id": "faiss1", "title_guess": "Best", "retrieval_score": 0.1, "score": 0.7, "best_faiss_rank": 1},
    ]
    selected = select_llm_candidates(items, llm_candidate_k=3)
    hit = next(item for item in selected if item["novel_id"] == "faiss1")
    assert "best_faiss_rank_top" in hit["llm_selection_reasons"]


def test_debug_target_forced_include_respects_k() -> None:
    items = [
        {"novel_id": "a", "title_guess": "A", "retrieval_score": 0.95, "score": 0.9, "best_faiss_rank": 1},
        {"novel_id": "b", "title_guess": "B", "retrieval_score": 0.9, "score": 0.8, "best_faiss_rank": 2},
        {"novel_id": "target", "title_guess": "凡人修仙传", "retrieval_score": 0.1, "score": 0.2, "best_faiss_rank": 99},
    ]
    selected = select_llm_candidates(items, llm_candidate_k=2, debug_target_title="凡人修仙传")
    assert len(selected) == 2
    hit = next(item for item in selected if item["novel_id"] == "target")
    assert "debug_target_forced" in hit["llm_selection_reasons"]


def test_rerank_output_includes_selection_reasons(tmp_path: Path) -> None:
    matcher = CountingMatcher(LLMMatchResult(llm_match_score=0.7, confidence="medium"))
    ranked, _ = rerank_candidates_with_llm(
        query="query",
        candidates=candidates(2),
        matcher=matcher,
        llm_candidate_k=1,
        use_cache=False,
        cache_path=tmp_path / "cache.jsonl",
    )
    selected = next(row for row in ranked if row["selected_for_llm"])
    assert selected["llm_selection_reasons"]


def test_semantic_fallback_score_not_extremely_low() -> None:
    assert semantic_fallback_score(0.95, matched_query_count=2) >= 0.35
