"""Deterministic train/eval book splits.

Every downstream number depends on this. The profiler and reranker are trained on
teacher labels derived from novels; if those same novels are retrievable at
evaluation time, an apparent gain cannot be separated from "the model memorised
this batch of books". A held-out set makes the gain attributable.

Two properties matter more than the split ratio:

* **Assignment is by content, not path.** Duplicate copies of one novel must land
  in the same fold, or dedup was pointless — the eval fold would still contain a
  book the model trained on. Assignment therefore hashes ``content_sha256``.
* **Assignment is stable.** Adding books to the corpus must not reshuffle the
  existing ones, otherwise every corpus update silently invalidates past results.
  Hashing gives that for free; random shuffling does not.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

Fold = Literal["train", "eval"]
DEFAULT_EVAL_FRACTION = 0.2
SPLIT_SALT = "inovelrec-book-split-v1"


@dataclass(frozen=True)
class SplitReport:
    """Split accounting for CLI summaries."""

    train_novels: int
    eval_novels: int
    skipped_duplicates: int
    skipped_unreadable: int

    @property
    def total(self) -> int:
        return self.train_novels + self.eval_novels


def split_bucket(content_sha256: str, salt: str = SPLIT_SALT) -> float:
    """Map a content hash to a stable value in [0, 1)."""

    digest = hashlib.sha256(f"{salt}:{content_sha256}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def assign_fold(content_sha256: str, eval_fraction: float = DEFAULT_EVAL_FRACTION, salt: str = SPLIT_SALT) -> Fold:
    """Assign one novel to a fold from its content hash alone."""

    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be between 0 and 1")
    return "eval" if split_bucket(content_sha256, salt) < eval_fraction else "train"


def build_splits(
    inventory: pd.DataFrame,
    eval_fraction: float = DEFAULT_EVAL_FRACTION,
    salt: str = SPLIT_SALT,
) -> tuple[pd.DataFrame, SplitReport]:
    """Attach a ``fold`` column to readable, non-duplicate inventory rows."""

    required = {"novel_id", "content_sha256", "read_status"}
    missing = required.difference(inventory.columns)
    if missing:
        raise ValueError(f"Inventory is missing columns needed for splitting: {sorted(missing)}")

    readable = inventory[inventory["read_status"] == "ok"]
    skipped_unreadable = int(len(inventory) - len(readable))

    if "is_duplicate" in readable.columns:
        unique = readable[~readable["is_duplicate"].fillna(False).astype(bool)]
    else:
        unique = readable
    skipped_duplicates = int(len(readable) - len(unique))

    assigned = unique.copy()
    assigned["fold"] = [
        assign_fold(str(value), eval_fraction=eval_fraction, salt=salt) for value in assigned["content_sha256"]
    ]
    report = SplitReport(
        train_novels=int((assigned["fold"] == "train").sum()),
        eval_novels=int((assigned["fold"] == "eval").sum()),
        skipped_duplicates=skipped_duplicates,
        skipped_unreadable=skipped_unreadable,
    )
    return assigned[["novel_id", "content_sha256", "title_guess", "fold"]] if "title_guess" in assigned.columns else assigned[["novel_id", "content_sha256", "fold"]], report


def load_fold_lookup(splits_path: Path) -> dict[str, str]:
    """Load ``novel_id -> fold`` for filtering at train or eval time."""

    frame = pd.read_parquet(splits_path, columns=["novel_id", "fold"])
    return dict(zip(frame["novel_id"].astype(str), frame["fold"].astype(str), strict=False))


def filter_to_fold(frame: pd.DataFrame, fold_lookup: dict[str, str], fold: Fold) -> pd.DataFrame:
    """Keep only rows whose ``novel_id`` belongs to the requested fold."""

    if "novel_id" not in frame.columns:
        raise ValueError("Frame must carry a novel_id column to be filtered by fold")
    keep = frame["novel_id"].astype(str).map(lambda value: fold_lookup.get(value) == fold)
    return frame[keep.fillna(False)]


def leakage_report(train_ids: set[str], eval_ids: set[str], fold_lookup: dict[str, str]) -> dict[str, int]:
    """Count ids that violate the split, for an assertion at experiment time."""

    return {
        "train_ids": len(train_ids),
        "eval_ids": len(eval_ids),
        "overlap": len(train_ids & eval_ids),
        "train_ids_in_eval_fold": sum(1 for value in train_ids if fold_lookup.get(value) == "eval"),
        "eval_ids_in_train_fold": sum(1 for value in eval_ids if fold_lookup.get(value) == "train"),
    }
