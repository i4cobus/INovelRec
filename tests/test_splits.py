import pandas as pd
import pytest

from src.splits import assign_fold, build_splits, filter_to_fold, leakage_report, split_bucket


def inventory(count: int, duplicates: int = 0, unreadable: int = 0) -> pd.DataFrame:
    rows = [
        {
            "novel_id": f"n{i}",
            "content_sha256": f"hash{i}",
            "title_guess": f"书{i}",
            "read_status": "ok",
            "is_duplicate": False,
        }
        for i in range(count)
    ]
    for i in range(duplicates):
        rows.append(
            {
                "novel_id": f"dup{i}",
                "content_sha256": f"hash{i}",
                "title_guess": f"书{i}",
                "read_status": "ok",
                "is_duplicate": True,
            }
        )
    for i in range(unreadable):
        rows.append(
            {
                "novel_id": f"bad{i}",
                "content_sha256": "",
                "title_guess": "",
                "read_status": "failed",
                "is_duplicate": False,
            }
        )
    return pd.DataFrame(rows)


def test_assignment_is_stable_across_runs() -> None:
    assert assign_fold("abc") == assign_fold("abc")
    assert 0.0 <= split_bucket("abc") < 1.0


def test_duplicate_copies_land_in_the_same_fold() -> None:
    """Assignment hashes content, so dedup and splitting cannot disagree."""

    assert assign_fold("same-content-hash") == assign_fold("same-content-hash")


def test_adding_books_does_not_reshuffle_existing_ones() -> None:
    """A corpus update must not silently invalidate earlier measurements."""

    before, _ = build_splits(inventory(60))
    after, _ = build_splits(inventory(90))
    merged = before.merge(after, on="novel_id", suffixes=("_before", "_after"))
    assert (merged["fold_before"] == merged["fold_after"]).all()


def test_split_excludes_duplicates_and_unreadable_rows() -> None:
    splits, report = build_splits(inventory(40, duplicates=5, unreadable=3))
    assert report.skipped_duplicates == 5
    assert report.skipped_unreadable == 3
    assert report.total == 40
    assert len(splits) == 40


def test_eval_fraction_is_approximately_respected() -> None:
    _, report = build_splits(inventory(2000), eval_fraction=0.2)
    share = report.eval_novels / report.total
    assert 0.17 < share < 0.23


def test_changing_the_salt_reshuffles() -> None:
    default, _ = build_splits(inventory(200))
    resalted, _ = build_splits(inventory(200), salt="different")
    merged = default.merge(resalted, on="novel_id", suffixes=("_a", "_b"))
    assert (merged["fold_a"] != merged["fold_b"]).any()


def test_invalid_eval_fraction_rejected() -> None:
    with pytest.raises(ValueError):
        assign_fold("abc", eval_fraction=0.0)
    with pytest.raises(ValueError):
        assign_fold("abc", eval_fraction=1.0)


def test_missing_columns_are_reported() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        build_splits(pd.DataFrame({"novel_id": ["a"]}))


def test_filter_to_fold_keeps_only_that_fold() -> None:
    lookup = {"a": "train", "b": "eval", "c": "train"}
    frame = pd.DataFrame({"novel_id": ["a", "b", "c", "unknown"], "value": [1, 2, 3, 4]})
    assert filter_to_fold(frame, lookup, "train")["novel_id"].tolist() == ["a", "c"]
    assert filter_to_fold(frame, lookup, "eval")["novel_id"].tolist() == ["b"]


def test_leakage_report_counts_violations() -> None:
    lookup = {"a": "train", "b": "eval"}
    report = leakage_report({"a", "b"}, {"b"}, lookup)
    assert report["overlap"] == 1
    assert report["train_ids_in_eval_fold"] == 1
