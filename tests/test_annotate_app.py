import json
from pathlib import Path

import pandas as pd
import pytest

from src.annotate_app import (
    HIDDEN_COLUMNS,
    append_label,
    load_labels,
    load_sheet,
    next_unlabeled,
    write_filled_sheet,
)


def sheet_frame(count: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "query_id": "q001",
                "query": "凡人流 仙侠",
                "wanted": "凡人流|仙侠",
                "unwanted": "系统",
                "novel_id": f"n{index}",
                "title_guess": f"《书{index}》",
                "rank": index + 1,
                "system_variant": "baseline_faiss",
                "evidence": f"正文摘录{index}",
            }
            for index in range(count)
        ]
    )


def write_sheet(tmp_path: Path, count: int = 4) -> Path:
    path = tmp_path / "sheet.csv"
    sheet_frame(count).to_csv(path, index=False)
    return path


def test_sheet_requires_its_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"query_id": ["q"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_sheet(path)


def test_rank_and_variant_are_never_shown() -> None:
    """Knowing the system ranked a result first biases the human toward agreeing."""

    assert set(HIDDEN_COLUMNS) == {"rank", "system_variant"}
    source = Path("src/annotate_app.py").read_text(encoding="utf-8")
    body = source.split('def main(')[1]
    for column in HIDDEN_COLUMNS:
        assert f'row["{column}"]' not in body
        assert f"row['{column}']" not in body


def test_labels_replay_with_last_write_winning(tmp_path: Path) -> None:
    log = tmp_path / "annotations.jsonl"
    append_label(log, {"query_id": "q001", "novel_id": "n0", "relevance_label": 1, "constraint_violation": False})
    append_label(log, {"query_id": "q001", "novel_id": "n0", "relevance_label": 2, "constraint_violation": True})

    labels = load_labels(log)

    assert len(labels) == 1
    assert labels[("q001", "n0")]["relevance_label"] == 2


def test_corrupt_log_lines_are_skipped(tmp_path: Path) -> None:
    log = tmp_path / "annotations.jsonl"
    append_label(log, {"query_id": "q001", "novel_id": "n0", "relevance_label": 1})
    with log.open("a", encoding="utf-8") as handle:
        handle.write("{ truncated by a crash\n")

    assert len(load_labels(log)) == 1


def test_next_unlabeled_wraps_and_stops(tmp_path: Path) -> None:
    sheet = load_sheet(write_sheet(tmp_path, count=3))
    labels = {("q001", "n1"): {}}

    assert next_unlabeled(sheet, labels, 0) == 2
    assert next_unlabeled(sheet, labels, 2) == 0

    everything = {("q001", f"n{index}"): {} for index in range(3)}
    assert next_unlabeled(sheet, everything, 1) == 1


def test_filled_sheet_is_readable_by_the_agreement_script(tmp_path: Path) -> None:
    path = write_sheet(tmp_path, count=3)
    sheet = load_sheet(path)
    labels = {
        ("q001", "n0"): {"relevance_label": 2, "constraint_violation": True, "notes": "明确违反"},
        ("q001", "n2"): {"relevance_label": 0, "constraint_violation": False, "notes": ""},
    }

    count = write_filled_sheet(sheet, labels, path)
    reloaded = pd.read_csv(path)

    assert count == 2
    assert set(["relevance_label", "constraint_violation", "notes"]).issubset(reloaded.columns)
    filled = reloaded[reloaded["relevance_label"].notna()]
    assert len(filled) == 2
    assert bool(filled.iloc[0]["constraint_violation"]) is True


def test_next_unlabeled_stays_in_range_when_everything_is_labelled(tmp_path: Path) -> None:
    """The caller seeds this with -1 on first load; a finished sheet must not
    hand back -1 as a row number."""

    sheet = load_sheet(write_sheet(tmp_path, count=3))
    everything = {("q001", f"n{index}"): {} for index in range(3)}

    for start in (-1, 0, 2):
        assert 0 <= next_unlabeled(sheet, everything, start) < 3


def test_next_unlabeled_handles_an_empty_sheet(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    sheet_frame(1).iloc[0:0].to_csv(path, index=False)  # 保留表头，只是没有行
    assert next_unlabeled(load_sheet(path), {}, -1) == 0
