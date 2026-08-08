from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.llm_matcher import LLMMatchResult, build_match_prompt, parse_llm_match_result
from src.rank import (
    LLM_CANDIDATE_K_ALL,
    LLMProgressEvent,
    normalize_semantic_scores,
    RankingWeights,
    compute_risk_penalty,
    llm_final_score,
    rerank_candidates_with_llm,
    resolve_llm_candidate_k,
    score_component_contributions,
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


def biased_pair() -> list[dict[str, Any]]:
    """A best-possible unscored candidate against a poorly-rated scored one.

    'unscored' has the strictly higher semantic score but loses the single LLM slot,
    which is awarded on retrieval_score. 'filler' only exists to give min-max
    normalization a floor: with two candidates the pool collapses to {1.0, 0.0}.
    """

    return [
        {"novel_id": "unscored", "title_guess": "Unscored", "rank": 1, "score": 1.00, "retrieval_score": 0.10},
        {"novel_id": "scored", "title_guess": "Scored", "rank": 2, "score": 0.95, "retrieval_score": 0.99},
        {"novel_id": "filler", "title_guess": "Filler", "rank": 3, "score": 0.00, "retrieval_score": 0.01},
    ]


def rank_pair(tmp_path: Path, policy: str) -> dict[str, float]:
    matcher = CountingMatcher(LLMMatchResult(llm_match_score=0.1, confidence="low"))
    ranked, _ = rerank_candidates_with_llm(
        query="query",
        candidates=biased_pair(),
        matcher=matcher,
        llm_candidate_k=1,
        use_cache=False,
        cache_path=tmp_path / f"{policy}.jsonl",
        fallback_policy=policy,
    )
    return {row["novel_id"]: row["final_score"] for row in ranked}


def test_legacy_policy_lets_selection_beat_better_semantic_evidence(tmp_path: Path) -> None:
    """Documents the 4080-era bias: the two formulas have different ceilings."""

    scores = rank_pair(tmp_path, "legacy_semantic")
    # 'scored' wins despite the LLM rating it 0.1 and its semantic score being lower.
    assert scores["scored"] > scores["unscored"]


def test_impute_policy_keeps_candidates_on_one_scale(tmp_path: Path) -> None:
    scores = rank_pair(tmp_path, "impute")
    assert scores["unscored"] > scores["scored"]


def test_impute_policy_records_imputed_features(tmp_path: Path) -> None:
    matcher = CountingMatcher(LLMMatchResult(llm_match_score=0.4, confidence="high"))
    ranked, _ = rerank_candidates_with_llm(
        query="query",
        candidates=candidates(3),
        matcher=matcher,
        llm_candidate_k=1,
        use_cache=False,
        cache_path=tmp_path / "cache.jsonl",
    )
    unscored = [row for row in ranked if not row["selected_for_llm"]]
    assert unscored
    # Imputed from the single scored candidate, so it matches that candidate's features.
    assert all(row["imputed_llm_match_score"] == 0.4 for row in unscored)
    assert all(row["imputed_confidence_score"] == 1.0 for row in unscored)
    assert all(row["llm_match_score"] is None for row in unscored)


def test_llm_candidate_k_all_scores_every_candidate(tmp_path: Path) -> None:
    matcher = CountingMatcher(LLMMatchResult(llm_match_score=0.6, confidence="medium"))
    ranked, timing = rerank_candidates_with_llm(
        query="query",
        candidates=candidates(7),
        matcher=matcher,
        llm_candidate_k=LLM_CANDIDATE_K_ALL,
        use_cache=False,
        cache_path=tmp_path / "cache.jsonl",
    )
    assert matcher.calls == 7
    assert timing.llm_candidate_k == 7
    assert all(row["selected_for_llm"] for row in ranked)
    assert not [row for row in ranked if row["analysis_provider"] == "semantic_fallback"]


def test_resolve_llm_candidate_k_all_sentinel() -> None:
    assert resolve_llm_candidate_k(candidate_k=200, llm_candidate_k=LLM_CANDIDATE_K_ALL) == (200, None)


@dataclass
class BatchingMatcher:
    """A matcher that offers score_many, like the HTTP backend does."""

    result: LLMMatchResult
    provider: str = "batch_mock"
    batch_calls: int = 0
    single_calls: int = 0
    fail_indices: frozenset = frozenset()

    def score(self, query: str, candidate: dict[str, Any], profile_text: str, max_profile_chars: int = 1200) -> LLMMatchResult:
        self.single_calls += 1
        return self.result

    def score_many(
        self,
        query: str,
        items: list[tuple[dict[str, Any], str]],
        max_profile_chars: int = 1200,
        on_result: Any = None,
    ) -> list[LLMMatchResult | None]:
        self.batch_calls += 1
        return [None if idx in self.fail_indices else self.result for idx in range(len(items))]


def test_batching_matcher_scores_in_one_call(tmp_path: Path) -> None:
    matcher = BatchingMatcher(LLMMatchResult(llm_match_score=0.7, confidence="high"))
    ranked, _ = rerank_candidates_with_llm(
        query="query",
        candidates=candidates(6),
        matcher=matcher,
        llm_candidate_k=LLM_CANDIDATE_K_ALL,
        use_cache=False,
        cache_path=tmp_path / "cache.jsonl",
    )
    assert matcher.batch_calls == 1
    assert matcher.single_calls == 0
    assert sum(1 for row in ranked if row["selected_for_llm"]) == 6


def test_failed_batch_slot_becomes_rule_fallback_and_is_not_cached(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.jsonl"
    matcher = BatchingMatcher(
        LLMMatchResult(llm_match_score=0.7, confidence="high"),
        fail_indices=frozenset({1}),
    )
    ranked, _ = rerank_candidates_with_llm(
        query="query",
        candidates=candidates(3),
        matcher=matcher,
        llm_candidate_k=LLM_CANDIDATE_K_ALL,
        cache_path=cache_path,
    )
    failed = [row for row in ranked if row["analysis_provider"] == "rule_fallback"]
    assert len(failed) == 1
    assert "llm_request_failed" in failed[0]["risk_flags"]
    # Only the two successful results may be persisted.
    assert len(cache_path.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_sequential_matcher_still_used_when_no_batch_api(tmp_path: Path) -> None:
    matcher = CountingMatcher(LLMMatchResult(llm_match_score=0.5, confidence="medium"))
    rerank_candidates_with_llm(
        query="query",
        candidates=candidates(4),
        matcher=matcher,
        llm_candidate_k=LLM_CANDIDATE_K_ALL,
        use_cache=False,
        cache_path=tmp_path / "cache.jsonl",
    )
    assert matcher.calls == 4


def test_normalization_is_unconditional_min_max() -> None:
    """The old version passed [0,1] scores through unchanged; that hid the spread."""

    narrow = [{"novel_id": f"n{i}", "score": s} for i, s in enumerate([0.72, 0.68, 0.64])]
    normalized = normalize_semantic_scores(narrow)
    assert normalized["n0"] == 1.0
    assert normalized["n2"] == 0.0
    assert 0.4 < normalized["n1"] < 0.6


def test_normalization_does_not_change_shape_with_negative_scores() -> None:
    """Behaviour must not depend on whether a negative cosine happens to appear."""

    without = normalize_semantic_scores([{"novel_id": "a", "score": 0.9}, {"novel_id": "b", "score": 0.1}])
    with_negative = normalize_semantic_scores([{"novel_id": "a", "score": 0.9}, {"novel_id": "b", "score": -0.1}])
    assert without["a"] == with_negative["a"] == 1.0
    assert without["b"] == with_negative["b"] == 0.0


def test_normalization_handles_a_single_distinct_score() -> None:
    flat = normalize_semantic_scores([{"novel_id": "a", "score": 0.5}, {"novel_id": "b", "score": 0.5}])
    assert set(flat.values()) == {1.0}


def test_component_contributions_expose_a_decorative_reranker(tmp_path: Path) -> None:
    """A matcher returning a constant score contributes no ranking signal."""

    matcher = CountingMatcher(LLMMatchResult(llm_match_score=0.8, confidence="high"))
    ranked, _ = rerank_candidates_with_llm(
        query="query",
        candidates=candidates(6),
        matcher=matcher,
        llm_candidate_k=LLM_CANDIDATE_K_ALL,
        use_cache=False,
        cache_path=tmp_path / "cache.jsonl",
    )
    contributions = score_component_contributions(ranked)
    assert contributions["llm_match"] == 0.0
    assert contributions["semantic_share"] == 1.0


def test_component_contributions_empty_rows() -> None:
    assert score_component_contributions([]) == {}


def test_json_survives_a_reasoning_trace_containing_braces() -> None:
    """Qwen3 thinks by default; a greedy {.*} spans from the trace to the answer."""

    from src.llm_matcher import extract_json_object

    text = (
        "<think>\n先看 {关键点}：题材匹配，但 {慢热} 无法确认。\n</think>\n\n"
        '{"llm_match_score":0.62,"confidence":"medium","matched_preferences":["仙侠"]}'
    )
    assert extract_json_object(text)["llm_match_score"] == 0.62


def test_json_survives_a_truncated_reasoning_trace() -> None:
    from src.llm_matcher import extract_json_object

    text = '思考被截断 {残留\n</think>\n{"llm_match_score":0.4,"confidence":"low"}'
    assert extract_json_object(text)["llm_match_score"] == 0.4


def test_nested_objects_are_matched_by_depth() -> None:
    from src.llm_matcher import extract_json_object

    assert extract_json_object('{"a":{"b":1},"c":2}')["c"] == 2
