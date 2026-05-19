# AI-Powered Chinese Web Novel Recommendation System

This project is a local AI application that turns a personal corpus of Chinese web novels into searchable and explainable recommendation data.

It is not mainly a traditional collaborative-filtering recommender. The core engineering work is unstructured text ingestion, Chinese text cleaning, chapter-aware preprocessing, semantic profile generation, embeddings, vector retrieval, and explainable recommendation.

## Stage 1: Dataset Inventory

Stage 1 scans `data/raw/` and writes `data/processed/novels.parquet`.

It includes:

- recursive `.txt` discovery
- mixed encoding detection
- stable `novel_id` generation
- basic metadata extraction
- approximate chapter count estimation
- compact raw text samples

Run:

```bash
uv run python scripts/01_inventory.py --overwrite
```

## Stage 2: Cleaned Novel Profiles

Stage 2 reads `data/processed/novels.parquet`, reopens successful TXT files with their detected encodings, cleans the text, splits likely chapters, and writes compact profile rows to `data/processed/novel_profiles.parquet`.

Profiles are designed for later semantic embedding. Stage 2 does not create embeddings, FAISS indexes, APIs, or UI.

Run:

```bash
uv run python scripts/02_build_profiles.py --overwrite
```

Useful options:

```bash
uv run python scripts/02_build_profiles.py --inventory data/processed/novels.parquet --out data/processed/novel_profiles.parquet --limit 100 --overwrite
```

## Stage 2 Output Schema

`data/processed/novel_profiles.parquet` contains:

- `novel_id`
- `title_guess`
- `author_guess`
- `char_count`
- `estimated_chapter_count`
- `profile_text`
- `opening_sample`
- `middle_sample`
- `ending_sample`

`profile_text` includes title, optional author, length, chapter count, opening sample, middle sample, and ending sample. It is capped around 1,000-3,000 Chinese characters by default.

## Project Structure

```text
INovelRec/
|-- README.md
|-- pyproject.toml
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- index/
|-- scripts/
|   |-- 01_inventory.py
|   `-- 02_build_profiles.py
|-- src/
|   |-- __init__.py
|   |-- clean.py
|   |-- config.py
|   |-- ingest.py
|   |-- profile.py
|   |-- schema.py
|   |-- split_chapters.py
|   `-- text_utils.py
|-- tests/
|   |-- test_clean.py
|   |-- test_ingest.py
|   `-- test_profile.py
`-- docs/
```

## Setup

```bash
uv sync
```

## Next Stages

- Stage 3: embedding generation
- Stage 4: FAISS semantic search
- Stage 5: explainable recommendation backend
- Stage 6: Streamlit or FastAPI UI

