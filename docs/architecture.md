# Architecture

The project builds a local semantic retrieval and recommendation pipeline over Chinese web novels. Each stage produces an artifact that the next stage consumes.

## Stage Flow

```text
raw TXT files
-> novels.parquet
-> cleaned profile generation
-> novel_profiles.parquet
-> SentenceTransformer embeddings
-> normalized vectors
-> FAISS IndexFlatIP
-> LLM/domain query expansion
-> multi-query semantic candidate retrieval
-> candidate merge
-> transformers local LLM candidate scoring
-> cache reuse
-> final hybrid reranking
-> explainable top-k recommendations
```

## Stage 3

Stage 3 embeds compact novel profiles instead of full novels. Full novels are too large for efficient embedding, and embedding them directly would mix many plot arcs and styles into a single oversized input.

The FAISS index uses normalized embedding vectors with `IndexFlatIP`. For unit-length vectors, inner product is equivalent to cosine similarity.

Stage 3 outputs:

- `data/index/faiss.index`
- `data/index/novel_id_map.json`
- `data/index/index_metadata.json`

## Stage 4

Stage 4 does not rebuild embeddings, regenerate profiles, or reclean text. It expands the raw query for retrieval recall, uses the current Stage 3 index for multi-query candidate retrieval, then reranks candidates with local transformers LLM analysis.

```text
query
-> structured preference parsing
-> LLM/domain query expansion
-> Qwen3 embedding query
-> multi-query FAISS retrieval
-> candidate merge
-> transformers local LLM scoring of llm-candidate-k candidates
-> cache reuse from data/cache/llm_rerank_cache.jsonl
-> final hybrid reranking
-> explainable top-k recommendations
```

Domain hints such as `凡人流 -> 普通资质, 草根修仙, 谨慎, 炼气, 筑基, 宗门` are retrieval-only recall hints. They are not final scoring tags. The final rank still depends on FAISS retrieval strength, local LLM candidate analysis, confidence, and risk penalties.

The local LLM returns compact JSON:

```json
{
  "llm_match_score": 0.0,
  "confidence": "high|medium|low",
  "matched_preferences": ["..."],
  "violated_preferences": ["..."],
  "risk_flags": ["..."],
  "reason": "one concise sentence"
}
```

Final score:

```text
final_score =
0.35 * normalized_semantic_score
+ 0.55 * llm_match_score
+ 0.10 * confidence_score
- risk_penalty
```

Risk penalties:

- `0.15` if `violated_preferences` is non-empty
- `0.05` if boilerplate or source-site risk is detected
- `0.05` if confidence is low

Candidates outside `llm-candidate-k` receive `analysis_provider = semantic_fallback` and keep a lower-priority semantic fallback score.
