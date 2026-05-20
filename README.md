# AI-Powered Chinese Web Novel Discovery System

This project is a local AI application that turns a personal corpus of Chinese web novels into searchable, explainable recommendation results.

It is not mainly a traditional collaborative-filtering recommender. The core engineering work is unstructured text ingestion, Chinese text cleaning, profile generation, embeddings, vector retrieval, local LLM reranking, grounded explanation generation, and an interactive demo app.

## Demo

Streamlit demo interface:

![Streamlit recommendation demo](demo/demo1.png)

Explainable recommendation cards:

![Explainable recommendation results](demo/demo2.png)

## Setup

```bash
uv sync
```

## Stage 1: Dataset Inventory

```bash
uv run python scripts/01_inventory.py --overwrite
```

Stage 1 scans `data/raw/` and writes `data/processed/novels.parquet`.

## Stage 2: Cleaned Novel Profiles

```bash
uv run python scripts/02_build_profiles.py --overwrite
```

Stage 2 writes compact sampled profile rows to `data/processed/novel_profiles.parquet`.

## Stage 3: Embeddings and FAISS Index

```bash
uv run python scripts/03_build_index.py --limit 100 --overwrite --device cuda
uv run python scripts/04_search_demo.py "凡人流 仙侠 慢热 理性主角 不系统" --device cuda
```

Stage 3 embeds compact profiles with `Qwen/Qwen3-Embedding-4B` and builds a FAISS `IndexFlatIP` index.

Expected files:

- `data/index/faiss.index`
- `data/index/novel_id_map.json`
- `data/index/index_metadata.json`

## Stage 4: Query Expansion and Local LLM Reranking

Stage 4 improves retrieval recall with query expansion, then reranks candidates with a local Hugging Face Transformers Qwen model.

```bash
uv run python scripts/05_recommend_demo.py "凡人流 仙侠 慢热 理性主角 不系统" --use-query-expansion --use-domain-hints --candidate-k 200 --top-k-per-query 100 --llm-candidate-k 10 --top-k 10 --device cuda
```

Domain hints are used only for retrieval expansion, not final scoring. Final ranking is based on semantic retrieval signals, local LLM match score, confidence, and risk penalties.

Current Stage 4 scoring:

```text
final_score =
0.40 * normalized_semantic_score
+ 0.50 * llm_match_score
+ 0.10 * confidence_score
- risk_penalty
```

## Stage 5: Explanation and Recommendation Report

Stage 4 decides ranking. Stage 5 explains that ranking.

```bash
uv run python scripts/06_explain_demo.py "凡人流 仙侠 慢热 理性主角 不系统" --candidate-k 200 --llm-candidate-k 10 --top-k 5 --device cuda
```

Save a report:

```bash
uv run python scripts/06_explain_demo.py "凡人流 仙侠 慢热 理性主角 不系统" --candidate-k 200 --llm-candidate-k 10 --top-k 5 --save-report reports/fanren_xianxia.md --device cuda
```

The explanation LLM uses only Stage 4 outputs and sampled profile evidence. It does not change ranking. If JSON parsing fails, the system uses a deterministic fallback explanation.

## Stage 6: Streamlit Demo App

Run the interactive app:

```bash
uv run streamlit run src/streamlit_app.py
```

The app uses the existing FAISS index, id map, and novel profiles. It does not rebuild the index, regenerate embeddings, or reclean text.

The app demonstrates:

- Natural-language Chinese preference input
- Query expansion and multi-query FAISS retrieval
- Local Qwen LLM-assisted reranking
- Grounded explanation generation
- Recommendation cards with scores, evidence, risks, and takeaway
- Markdown and JSON report export

First run can be slow because local Hugging Face models are loaded into GPU memory. Later button clicks should be faster because Streamlit caches the embedding model, local Qwen LLM, FAISS index, id map, and profile lookup.

Fast development settings:

- `candidate-k = 50`
- `llm-candidate-k = 3`
- `top-k = 3`

Better quality settings:

- `candidate-k = 200`
- `llm-candidate-k = 10`
- `top-k = 5`

## Current Architecture

```text
raw TXT corpus
-> inventory parquet
-> cleaned compact profiles
-> Qwen3 embeddings
-> FAISS IndexFlatIP
-> query expansion / semantic retrieval
-> local Qwen reranking
-> local Qwen explanation generation
-> Streamlit recommendation UI
```

## Notes

- The app uses sampled compact profiles, not full human reading.
- The current index is a development index and should be rebuilt after final cleaning/profile decisions.
- All LLM usage is local through Hugging Face Transformers.
