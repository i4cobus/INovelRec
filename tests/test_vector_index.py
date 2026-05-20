from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.vector_index import (
    build_faiss_index,
    ensure_can_write,
    load_id_map,
    load_profiles_for_index,
    make_id_map,
    save_id_map,
    validate_embeddings,
)


def test_build_index_and_search_fake_vectors() -> None:
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]], dtype=np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = build_faiss_index(embeddings)

    query = np.array([[1.0, 0.0]], dtype=np.float32)
    scores, ids = index.search(query, 2)

    assert ids[0].tolist() == [0, 2]
    assert scores[0][0] == pytest.approx(1.0)


def test_id_map_save_load(tmp_path: Path) -> None:
    profiles = pd.DataFrame(
        {
            "novel_id": ["n1"],
            "title_guess": ["Title"],
            "profile_text": ["A long enough profile text for preview testing."],
        }
    )
    id_map = make_id_map(profiles)
    path = tmp_path / "novel_id_map.json"
    save_id_map(id_map, path)

    loaded = load_id_map(path)
    assert loaded[0]["novel_id"] == "n1"
    assert loaded[0]["title_guess"] == "Title"


def test_overwrite_protection(tmp_path: Path) -> None:
    output = tmp_path / "faiss.index"
    output.write_text("exists", encoding="utf-8")

    with pytest.raises(FileExistsError):
        ensure_can_write([output], overwrite=False)
    ensure_can_write([output], overwrite=True)


def test_validate_embeddings_rejects_empty() -> None:
    with pytest.raises(ValueError):
        validate_embeddings(np.empty((0, 3), dtype=np.float32))


def test_load_profiles_skips_invalid_rows(tmp_path: Path) -> None:
    path = tmp_path / "profiles.parquet"
    pd.DataFrame(
        {
            "novel_id": ["ok", None, "short"],
            "title_guess": ["Good", "Missing", "Short"],
            "profile_text": ["x" * 60, "x" * 60, "too short"],
        }
    ).to_parquet(path, index=False)

    loaded = load_profiles_for_index(path)
    assert loaded.dataframe["novel_id"].tolist() == ["ok"]
    assert loaded.skipped_rows == 2

