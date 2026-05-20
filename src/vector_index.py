"""FAISS vector index utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd

from src.config import INDEX_DIR, PROCESSED_DATA_DIR

DEFAULT_PROFILES_PATH = PROCESSED_DATA_DIR / "novel_profiles.parquet"
DEFAULT_INDEX_PATH = INDEX_DIR / "faiss.index"
DEFAULT_ID_MAP_PATH = INDEX_DIR / "novel_id_map.json"
DEFAULT_INDEX_METADATA_PATH = INDEX_DIR / "index_metadata.json"
MIN_PROFILE_TEXT_CHARS = 50


@dataclass(frozen=True)
class LoadedProfiles:
    """Filtered profile data used for embedding and index creation."""

    dataframe: pd.DataFrame
    skipped_rows: int


def load_profiles_for_index(
    profiles_path: Path = DEFAULT_PROFILES_PATH,
    min_profile_chars: int = MIN_PROFILE_TEXT_CHARS,
    limit: int | None = None,
) -> LoadedProfiles:
    """Load and filter profile rows that are valid for embedding."""

    profiles = pd.read_parquet(profiles_path)
    required_columns = {"novel_id", "title_guess", "profile_text"}
    missing = required_columns.difference(profiles.columns)
    if missing:
        raise ValueError(f"Missing required profile columns: {sorted(missing)}")

    if limit is not None:
        profiles = profiles.head(limit)

    valid_mask = (
        profiles["novel_id"].notna()
        & profiles["profile_text"].notna()
        & (profiles["profile_text"].astype(str).str.len() >= min_profile_chars)
    )
    filtered = profiles.loc[valid_mask].copy()
    filtered["novel_id"] = filtered["novel_id"].astype(str)
    filtered["title_guess"] = filtered["title_guess"].fillna("").astype(str)
    filtered["profile_text"] = filtered["profile_text"].astype(str)
    return LoadedProfiles(dataframe=filtered.reset_index(drop=True), skipped_rows=int((~valid_mask).sum()))


def validate_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Validate embeddings and return a contiguous float32 2D array."""

    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got shape {array.shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"Embeddings must be non-empty, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Embeddings contain NaN or infinite values")
    return np.ascontiguousarray(array)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build an inner-product FAISS index for normalized vectors."""

    vectors = validate_embeddings(embeddings)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def make_id_map(profiles: pd.DataFrame, preview_chars: int = 300) -> dict[str, dict[str, str]]:
    """Create a FAISS row-index to profile metadata map."""

    id_map: dict[str, dict[str, str]] = {}
    for idx, row in profiles.reset_index(drop=True).iterrows():
        profile_text = str(row["profile_text"])
        id_map[str(idx)] = {
            "novel_id": str(row["novel_id"]),
            "title_guess": str(row.get("title_guess") or ""),
            "profile_text_preview": profile_text[:preview_chars],
        }
    return id_map


def save_faiss_index(index: faiss.Index, index_path: Path) -> None:
    """Save a FAISS index to disk."""

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))


def load_faiss_index(index_path: Path) -> faiss.Index:
    """Load a FAISS index from disk."""

    return faiss.read_index(str(index_path))


def save_id_map(id_map: dict[str, dict[str, str]], id_map_path: Path) -> None:
    """Save row-index metadata as UTF-8 JSON."""

    id_map_path.parent.mkdir(parents=True, exist_ok=True)
    id_map_path.write_text(json.dumps(id_map, ensure_ascii=False, indent=2), encoding="utf-8")


def load_id_map(id_map_path: Path) -> dict[int, dict[str, str]]:
    """Load row-index metadata and convert keys to integers."""

    raw = json.loads(id_map_path.read_text(encoding="utf-8"))
    return {int(key): value for key, value in raw.items()}


def make_index_metadata(
    *,
    model_name: str,
    embedding_dim: int,
    num_vectors: int,
    normalize_embeddings: bool,
    source_profiles: Path,
) -> dict[str, Any]:
    """Build serializable metadata for an index artifact."""

    return {
        "model_name": model_name,
        "embedding_dim": embedding_dim,
        "num_vectors": num_vectors,
        "normalize_embeddings": normalize_embeddings,
        "index_type": "IndexFlatIP",
        "source_profiles": source_profiles.as_posix(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def save_index_metadata(metadata: dict[str, Any], metadata_path: Path) -> None:
    """Save index metadata as UTF-8 JSON."""

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_can_write(paths: list[Path], overwrite: bool) -> None:
    """Raise if any output path exists and overwrite is disabled."""

    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output already exists: {joined}. Use --overwrite to replace it.")

