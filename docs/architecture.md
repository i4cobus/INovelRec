# Architecture

The project builds a local semantic retrieval and recommendation pipeline over Chinese web novels. Each stage produces an artifact or reusable module that the next stage consumes.

## End-to-End Flow

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
-> Transformers local LLM candidate scoring
-> cache reuse
-> final hybrid reranking
-> local Qwen explanation generation
-> Streamlit recommendation UI
-> markdown/json export
```

## Stage 3: Vector Retrieval

Stage 3 embeds compact novel profiles instead of full novels. Full novels are too large for efficient embedding, and embedding them directly would mix many plot arcs and styles into a single oversized input.

The FAISS index uses normalized embedding vectors with `IndexFlatIP`. For unit-length vectors, inner product is equivalent to cosine similarity.

Stage 3 outputs:

- `data/index/faiss.index`
- `data/index/novel_id_map.json`
- `data/index/index_metadata.json`

## Stage 4: Recommendation Ranking

Stage 4 does not rebuild embeddings, regenerate profiles, or reclean text. It expands the raw query for retrieval recall, uses the current Stage 3 index for multi-query candidate retrieval, then reranks candidates with local Transformers LLM analysis.

```text
query
-> structured preference parsing
-> LLM/domain query expansion
-> Qwen3 embedding query
-> multi-query FAISS retrieval
-> candidate merge
-> Transformers local LLM scoring of llm-candidate-k candidates
-> cache reuse from data/cache/llm_rerank_cache.jsonl
-> final hybrid reranking
```

Domain hints such as `凡人流 -> 普通资质, 草根修仙, 谨慎, 炼气, 筑基, 宗门` are retrieval-only recall hints. They are not final scoring tags. The final rank still depends on FAISS retrieval strength, local LLM candidate analysis, confidence, and risk penalties.

Final score:

```text
final_score =
0.40 * normalized_semantic_score
+ 0.50 * llm_match_score
+ 0.10 * confidence_score
- risk_penalty
```

Candidates outside `llm-candidate-k` receive `analysis_provider = semantic_fallback` and keep a lower-priority semantic fallback score.

## Stage 5: Explanation and Reporting

Stage 5 is explanation and reporting. It does not retrieve, rerank, rebuild embeddings, or change recommendation order.

```text
query
-> Stage 4 LLM-assisted recommendation reranking
-> final ranked candidates
-> local Qwen Transformers explanation generation
-> grounded recommendation report
```

Stage 4 is ranking. Stage 5 is explanation/reporting.

The explanation prompt includes the original query, final rank, title, scores, matched preferences, violated preferences, risk flags, Stage 4 reason, and compact profile evidence. The prompt explicitly instructs the model to use only provided evidence and not invent plot details, popularity, author facts, ratings, or completion status.

If the local model returns invalid JSON, Stage 5 falls back to a deterministic explanation built from Stage 4 fields. This keeps the report pipeline stable.

## Stage 6: Streamlit Demo Application

Stage 6 is the user-facing demo/application layer.

```text
User
-> Streamlit UI
-> Stage 4 recommendation pipeline
-> Stage 5 explanation/report generator
-> interactive recommendation cards
-> markdown/json export
```

Streamlit does not contain model logic. It imports reusable pipeline functions from `src/app_pipeline.py`, Stage 4 modules, and Stage 5 modules. This keeps the core recommendation logic reusable by both CLI scripts and the app.

The app caches expensive resources:

- Embedding model with `st.cache_resource`
- Local Qwen LLM with `st.cache_resource`
- FAISS index with `st.cache_resource`
- ID map and profile lookup with `st.cache_data`

The app does not add a cheap prefilter, rebuild FAISS, regenerate embeddings, or reclean profiles.
