"""Semantic search over the FAISS novel index."""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from src.embed import SupportsEncode, encode_texts
from src.query_expansion import ExpandedQuery
from src.vector_index import load_faiss_index, load_id_map


def semantic_search(
    query: str,
    model: SupportsEncode,
    index: faiss.Index,
    id_map: dict[int, dict[str, str]],
    top_k: int = 10,
) -> list[dict[str, object]]:
    """Embed a query and return ranked FAISS search results."""

    query = query.strip()
    if not query:
        return []
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if index.ntotal == 0:
        return []

    query_embedding = encode_texts(model, [query], batch_size=1, normalize_embeddings=True)
    if query_embedding.shape[1] != index.d:
        raise ValueError(f"Query dim {query_embedding.shape[1]} does not match index dim {index.d}")

    scores, ids = index.search(np.ascontiguousarray(query_embedding, dtype=np.float32), min(top_k, index.ntotal))
    results: list[dict[str, object]] = []
    for score, faiss_id in zip(scores[0], ids[0]):
        row_id = int(faiss_id)
        if row_id < 0 or row_id not in id_map:
            continue
        metadata = id_map[row_id]
        results.append(
            {
                "rank": len(results) + 1,
                "score": float(score),
                "novel_id": metadata.get("novel_id", ""),
                "title_guess": metadata.get("title_guess", ""),
                "profile_text_preview": metadata.get("profile_text_preview", ""),
            }
        )
    return results


def multi_query_semantic_search(
    expanded_queries: list[ExpandedQuery],
    model: SupportsEncode,
    index: faiss.Index,
    id_map: dict[int, dict[str, str]],
    top_k_per_query: int,
    final_candidate_k: int,
) -> list[dict[str, object]]:
    """Run multiple semantic searches and merge candidates by novel_id."""

    if top_k_per_query <= 0:
        raise ValueError("top_k_per_query must be positive")
    if final_candidate_k <= 0:
        raise ValueError("final_candidate_k must be positive")
    merged: dict[str, dict[str, object]] = {}
    total_queries = max(len(expanded_queries), 1)

    for query in expanded_queries:
        results = semantic_search(query.text, model, index, id_map, top_k=top_k_per_query)
        for result in results:
            novel_id = str(result.get("novel_id", ""))
            if not novel_id:
                continue
            score = float(result.get("score", 0.0))
            rank = int(result.get("rank", 0))
            item = merged.setdefault(
                novel_id,
                {
                    **result,
                    "score": score,
                    "best_semantic_score": score,
                    "best_faiss_rank": rank,
                    "matched_query_count": 0,
                    "retrieval_sources": [],
                    "per_query_scores": [],
                    "source_weight_bonus": 0.0,
                },
            )
            if score > float(item["best_semantic_score"]):
                item["score"] = score
                item["best_semantic_score"] = score
                item["best_faiss_rank"] = rank
            if query.source not in item["retrieval_sources"]:
                item["retrieval_sources"].append(query.source)
            item["matched_query_count"] = int(item["matched_query_count"]) + 1
            item["source_weight_bonus"] = max(float(item["source_weight_bonus"]), query.weight)
            item["per_query_scores"].append(
                {
                    "query": query.text,
                    "source": query.source,
                    "weight": query.weight,
                    "score": score,
                    "rank": rank,
                }
            )

    for item in merged.values():
        matched_norm = min(int(item["matched_query_count"]) / total_queries, 1.0)
        item["retrieval_score"] = (
            0.70 * float(item["best_semantic_score"])
            + 0.20 * matched_norm
            + 0.10 * float(item["source_weight_bonus"])
        )

    ranked = sorted(merged.values(), key=lambda item: float(item["retrieval_score"]), reverse=True)
    for idx, item in enumerate(ranked, start=1):
        item["rank"] = idx
        item["retrieval_rank"] = idx
    return ranked[:final_candidate_k]


def debug_target_status(title: str, candidates: list[dict[str, object]], reranked: list[dict[str, object]] | None = None, top_k: int = 10) -> dict[str, object]:
    """Find a debug title in merged, LLM-selected, and final top-k pools."""

    def matches(item: dict[str, object]) -> bool:
        return title in str(item.get("title_guess", ""))

    merged_hit = next((item for item in candidates if matches(item)), None)
    llm_pool = [item for item in (reranked or []) if item.get("selected_for_llm")]
    llm_hit = next((item for item in llm_pool if matches(item)), None)
    final_hit = next((item for item in (reranked or [])[:top_k] if matches(item)), None)
    return {
        "title": title,
        "in_merged_pool": merged_hit is not None,
        "in_llm_selected_pool": llm_hit is not None,
        "in_final_top_k": final_hit is not None,
        "merged": merged_hit,
        "llm_selected": llm_hit,
        "final": final_hit,
    }


__all__ = ["load_faiss_index", "load_id_map", "semantic_search", "multi_query_semantic_search", "debug_target_status"]
