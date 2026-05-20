"""Semantic search over the FAISS novel index."""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from src.embed import SupportsEncode, encode_texts
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


__all__ = ["load_faiss_index", "load_id_map", "semantic_search"]

