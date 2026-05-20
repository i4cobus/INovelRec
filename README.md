# AI-Powered Chinese Web Novel Discovery System

This project is a local AI application that turns a personal corpus of Chinese web novels into searchable and explainable recommendation data.

It is not mainly a traditional collaborative-filtering recommender. The core engineering work is unstructured text ingestion, Chinese text cleaning, chapter-aware preprocessing, semantic profile generation, embeddings, vector retrieval, LLM-assisted reranking, and grounded recommendation reporting.

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

## Stage 4: Query Expansion and Local LLM Reranking

Stage 4 improves retrieval recall with query expansion, then reranks candidates with a local Hugging Face transformers Qwen model.

```bash
uv run python scripts/05_recommend_demo.py "凡人流 仙侠 慢热 理性主角 不系统" --use-query-expansion --use-domain-hints --candidate-k 200 --top-k-per-query 100 --llm-candidate-k 10 --top-k 10 --device cuda
```

Domain hints are used only for retrieval expansion, not final scoring. Final ranking is based on Stage 4 semantic retrieval signals, local LLM match score, confidence, and risk penalties.

## Stage 5: Explanation and Recommendation Report

Stage 4 decides ranking. Stage 5 explains that ranking.

Stage 5 uses the local Qwen3 model through Hugging Face Transformers to generate user-facing explanations grounded in:

- Stage 4 final rank and scores
- matched preferences and violated preferences
- risk flags and Stage 4 reason
- sampled profile evidence

The explanation LLM does not change the ranking. If the model returns invalid JSON, the system falls back to a deterministic explanation from Stage 4 fields. Reports state uncertainty because the system uses sampled profiles, not full human reading.

Example:

```bash
uv run python scripts/06_explain_demo.py "凡人流 仙侠 慢热 理性主角 不系统" --candidate-k 200 --llm-candidate-k 10 --top-k 5 --device cuda
```

Save a report:

```bash
uv run python scripts/06_explain_demo.py "凡人流 仙侠 慢热 理性主角 不系统" --candidate-k 200 --llm-candidate-k 10 --top-k 5 --save-report reports/fanren_xianxia.md --device cuda
```

Output formats:

- `--output-format text`
- `--output-format markdown`
- `--output-format json`

## Setup

```bash
uv sync
```

## Current Architecture

```text
raw query
-> query expansion
-> multi-query FAISS retrieval
-> local Qwen3 LLM reranking
-> final ranked candidates
-> local Qwen3 explanation generation
-> grounded recommendation report
```

## Next Stages

- Stage 6: Streamlit or FastAPI UI

