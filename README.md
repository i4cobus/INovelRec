# AI-Powered Chinese Web Novel Recommendation System

This project is an internship-ready local AI application focused on turning a large corpus of Chinese web novels into a searchable, explainable recommendation system.

It is intentionally positioned as an AI application engineering project rather than a classic collaborative-filtering recommender. The core work is on unstructured text ingestion, Chinese NLP preprocessing, semantic retrieval, embeddings, vector search, and explainable recommendation.

## Why This Is an AI Application Project

The dataset is a large collection of raw `.txt` novels. Building value from it requires:

- robust file ingestion across mixed encodings
- metadata extraction from noisy filenames and text
- text normalization for Chinese content
- semantic indexing and vector retrieval
- explainable recommendation from text evidence rather than user-user ratings

## Stage 1: Dataset Inventory

Stage 1 builds a production-style inventory pipeline that:

- recursively scans `data/raw/` for `.txt` files
- detects encoding with common Chinese encodings first, then fallback detection
- extracts file and text metadata
- estimates chapter counts with regex heuristics
- stores compact text samples for downstream inspection
- writes the dataset inventory to `data/processed/novels.parquet`

## Project Structure

```text
INovelRec/
├── README.md
├── pyproject.toml
├── data/
│   ├── raw/
│   ├── processed/
│   └── index/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── ingest.py
│   ├── schema.py
│   └── text_utils.py
├── scripts/
│   └── 01_inventory.py
├── tests/
│   └── test_ingest.py
└── docs/
```

## Setup

```bash
uv sync
```

If you want to use the environment manually:

```bash
uv venv
uv pip install -e ".[dev]"
```

## Run Stage 1 Inventory

Default run:

```bash
uv run python scripts/01_inventory.py
```

Useful options:

```bash
uv run python scripts/01_inventory.py --raw-dir data/raw --out data/processed/novels.parquet --limit 100 --overwrite
```

Behavior:

- creates the output directory automatically
- refuses to overwrite an existing parquet unless `--overwrite` is passed
- continues when individual files fail to decode
- prints progress and summary statistics

## Output Schema

The parquet file contains one row per novel with these fields:

- `novel_id`
- `file_name`
- `file_stem`
- `relative_path`
- `absolute_path`
- `file_size_bytes`
- `file_size_mb`
- `detected_encoding`
- `read_status`
- `error_message`
- `title_guess`
- `author_guess`
- `char_count`
- `line_count`
- `estimated_chapter_count`
- `first_2000_chars`
- `sample_text`
- `created_at`

## Next Stages

- Stage 2: chapter splitting
- Stage 3: novel profile generation
- Stage 4: embedding generation
- Stage 5: FAISS semantic search
- Stage 6: explainable recommendation UI with FastAPI and Streamlit

Stage 1 intentionally stops at inventory generation. It does not build chapters, embeddings, search indexes, or UI yet.

