from src.query_expansion import ExpandedQuery
from src.search import debug_target_status, multi_query_semantic_search


def test_multi_query_merge_deduplicates_by_novel_id(monkeypatch) -> None:
    def fake_search(query, model, index, id_map, top_k):
        if query == "raw":
            return [
                {"rank": 1, "score": 0.8, "novel_id": "a", "title_guess": "A", "profile_text_preview": ""},
                {"rank": 2, "score": 0.7, "novel_id": "b", "title_guess": "B", "profile_text_preview": ""},
            ]
        return [
            {"rank": 1, "score": 0.9, "novel_id": "a", "title_guess": "A", "profile_text_preview": ""},
        ]

    monkeypatch.setattr("src.search.semantic_search", fake_search)
    results = multi_query_semantic_search(
        expanded_queries=[
            ExpandedQuery("raw", "raw", 1.0),
            ExpandedQuery("expanded", "domain_hints", 0.8),
        ],
        model=None,
        index=None,
        id_map={},
        top_k_per_query=10,
        final_candidate_k=10,
    )
    assert [result["novel_id"] for result in results].count("a") == 1
    hit = next(result for result in results if result["novel_id"] == "a")
    assert hit["best_semantic_score"] == 0.9


def test_matched_query_count_and_sources_are_preserved(monkeypatch) -> None:
    def fake_search(query, model, index, id_map, top_k):
        return [{"rank": 1, "score": 0.8, "novel_id": "a", "title_guess": "A", "profile_text_preview": ""}]

    monkeypatch.setattr("src.search.semantic_search", fake_search)
    results = multi_query_semantic_search(
        expanded_queries=[
            ExpandedQuery("q1", "raw", 1.0),
            ExpandedQuery("q2", "llm", 0.9),
            ExpandedQuery("q3", "domain_hints", 0.8),
        ],
        model=None,
        index=None,
        id_map={},
        top_k_per_query=10,
        final_candidate_k=10,
    )
    assert results[0]["matched_query_count"] == 3
    assert set(results[0]["retrieval_sources"]) == {"raw", "llm", "domain_hints"}


def test_debug_target_title_detection() -> None:
    candidates = [
        {"title_guess": "《凡人修仙传》", "novel_id": "a", "retrieval_rank": 3, "best_faiss_rank": 20},
    ]
    reranked = [
        {"title_guess": "《凡人修仙传》", "selected_for_llm": True, "final_rank": 1, "final_score": 0.9},
    ]
    status = debug_target_status("凡人修仙传", candidates, reranked, top_k=5)
    assert status["in_merged_pool"] is True
    assert status["in_llm_selected_pool"] is True
    assert status["in_final_top_k"] is True


def test_final_candidate_k_is_respected(monkeypatch) -> None:
    def fake_search(query, model, index, id_map, top_k):
        return [
            {"rank": idx + 1, "score": 1.0 - (idx * 0.01), "novel_id": str(idx), "title_guess": str(idx), "profile_text_preview": ""}
            for idx in range(10)
        ]

    monkeypatch.setattr("src.search.semantic_search", fake_search)
    results = multi_query_semantic_search(
        expanded_queries=[ExpandedQuery("q", "raw", 1.0)],
        model=None,
        index=None,
        id_map={},
        top_k_per_query=10,
        final_candidate_k=3,
    )
    assert len(results) == 3

