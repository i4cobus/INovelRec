# AI-Powered Chinese Web Novel Discovery System

This project is a local AI application that turns a personal corpus of Chinese web novels into searchable and explainable recommendation data.

It is not mainly a traditional collaborative-filtering recommender. The core engineering work is unstructured text ingestion, Chinese text cleaning, chapter-aware preprocessing, semantic profile generation, embeddings, vector retrieval, and explainable recommendation.

## Stage 1: Dataset Inventory

```bash
uv run python scripts/01_inventory.py --overwrite
```

Stage 1 scans `data/raw/` and writes `data/processed/novels.parquet`.

## Stage 2: Cleaned Novel Profiles

```bash
uv run python scripts/02_build_profiles.py --overwrite
```

Stage 2 reads the inventory, reopens successful TXT files, cleans text, estimates chapters, and writes `data/processed/novel_profiles.parquet`.

## Stage 3: Embeddings and FAISS Index

```bash
uv run python scripts/03_build_index.py --limit 100 --overwrite --device cuda
uv run python scripts/04_search_demo.py "凡人流 仙侠 慢热 理性主角 不系统" --device cuda
```

Stage 3 embeds compact profiles with `Qwen/Qwen3-Embedding-4B` and builds a FAISS `IndexFlatIP` index.

Expected outputs:

- `data/index/faiss.index`
- `data/index/novel_id_map.json`
- `data/index/index_metadata.json`

## Stage 4: Query Expansion + Transformers Local-LLM Reranking

Stage 4 does not rebuild embeddings or the FAISS index. It improves retrieval recall with query expansion, then reranks candidates with a local Hugging Face transformers model and explicit progress reporting.

```text
Layer 1: LLM/domain query expansion + multi-query FAISS retrieval.
Layer 2: transformers local LLM scores only llm-candidate-k candidates in detail.
Layer 3: final LLM-based scoring combines semantic score, LLM match score, confidence, and risk penalties.
```

LLM reranking alone cannot recover novels that never enter the FAISS candidate pool. Query expansion improves recall by searching multiple retrieval-friendly versions of the same user intent before candidate merging.

Domain hints are used only for retrieval expansion. They are not used as final ranking tags. This is different from alias-based final scoring: aliases as final scoring are too brittle, but small transparent domain hints are useful for getting likely candidates into the pool where the local LLM can judge them.

Final scoring:

```text
final_score =
0.35 * normalized_semantic_score
+ 0.55 * llm_match_score
+ 0.10 * confidence_score
- risk_penalty
```

Key options:

- `--candidate-k`: FAISS retrieval pool size. Default: `100`.
- `--top-k-per-query`: FAISS candidates fetched for each expanded query. Default: `100`.
- `--llm-candidate-k`: number of candidates sent to the local LLM. Default: `min(10, candidate-k)`.
- `--top-k`: number of final recommendations shown. Default: `10`.
- `--llm-profile-max-chars`: max profile characters sent to the LLM. Default: `1200`.
- `--llm-model`: transformers local LLM model. Default: `Qwen/Qwen3-4B-Instruct-2507`.
- `--use-query-expansion / --no-query-expansion`: use local LLM retrieval query expansion. Default: enabled.
- `--use-domain-hints / --no-domain-hints`: add small domain-hint retrieval queries. Default: enabled.
- `--max-expanded-queries`: cap retrieval query variants. Default: `5`.
- `--debug-target-title`: print whether an expected title appears in the merged pool, LLM-selected pool, and final top-k.
- `--use-cache / --no-cache`: reuse cached LLM analysis. Default: `--use-cache`.

Fast development run:

```bash
uv run python scripts/05_recommend_demo.py "凡人流 仙侠 慢热 理性主角 不系统" --candidate-k 20 --top-k-per-query 20 --llm-candidate-k 3 --top-k 5 --debug-target-title "凡人修仙传" --device cuda
```

Better quality run:

```bash
uv run python scripts/05_recommend_demo.py "凡人流 仙侠 慢热 理性主角 不系统" --use-query-expansion --use-domain-hints --candidate-k 200 --top-k-per-query 100 --llm-candidate-k 10 --top-k 10 --llm-model Qwen/Qwen3-4B-Instruct-2507 --device cuda
```

Local LLM scoring is time-consuming because each selected candidate requires a separate profile-reading and JSON-scoring pass. The JSONL cache at `data/cache/llm_rerank_cache.jsonl` avoids repeated calls for the same query, model, prompt version, profile text, and truncation size.

The CLI prints expanded retrieval queries before FAISS search, a retrieval summary after candidate merge, optional debug-title status, LLM scoring progress, and a timing summary.

The current index is a development index and should be rebuilt after final cleaning/profile rules are settled.

## Setup

```bash
uv sync
```

## Next Stages

- Stage 5: explanation layer and recommendation report generation
- Stage 6: Streamlit or FastAPI UI
