"""Build compact novel profiles from the Stage 1 inventory."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from tqdm import tqdm

from src.clean import CleaningStats, clean_novel_text_with_stats, contains_zxcs_boilerplate
from src.config import DEFAULT_OUTPUT_PATH, PROCESSED_DATA_DIR, resolve_worker_count
from src.split_chapters import split_chapters

DEFAULT_PROFILE_OUTPUT_PATH = PROCESSED_DATA_DIR / "novel_profiles.parquet"


@dataclass(frozen=True)
class ProfileBuildResult:
    """Profile build result plus skip accounting for CLI summaries."""

    dataframe: pd.DataFrame
    processed: int
    skipped_failed: int
    skipped_missing: int
    skipped_read_error: int
    skipped_duplicate: int = 0
    zxcs_boilerplate_detected: int = 0
    zxcs_boilerplate_lines_removed: int = 0
    profiles_with_remaining_boilerplate: int = 0


def read_text_with_encoding(path: Path, encoding: str | None, allow_lossy: bool = False) -> str:
    """Read a raw text file with the encoding detected during inventory.

    ``allow_lossy`` must mirror what Stage 1 decided for this file. A handful of
    novels only decode with ``errors="replace"`` (a single corrupt byte each);
    re-reading them strictly here would silently drop books Stage 1 recovered.
    """

    if not encoding:
        raise ValueError("Missing detected encoding")
    return path.read_text(encoding=encoding, errors="replace" if allow_lossy else "strict")


def compact_sample(text: str, start: int, length: int) -> str:
    """Return a compact sample window without cutting past text length."""

    if not text:
        return ""
    start = min(max(start, 0), max(len(text) - 1, 0))
    return text[start:start + length].strip()


def extract_profile_samples(cleaned_text: str, sample_chars: int = 650) -> tuple[str, str, str]:
    """Extract opening, middle, and ending samples from cleaned text."""

    opening = compact_sample(cleaned_text, 0, sample_chars)
    middle_start = max((len(cleaned_text) // 2) - (sample_chars // 2), 0)
    middle = compact_sample(cleaned_text, middle_start, sample_chars)
    ending = cleaned_text[-sample_chars:].strip() if cleaned_text else ""
    return opening, middle, ending


def make_profile_text(
    *,
    title_guess: str,
    author_guess: str | None,
    char_count: int,
    chapter_count: int,
    opening_sample: str,
    middle_sample: str,
    ending_sample: str,
    max_chars: int = 3000,
) -> str:
    """Compose a compact profile text suitable for later embedding."""

    author_line = f"作者：{author_guess}\n" if author_guess else ""
    header = (
        f"标题：{title_guess}\n"
        f"{author_line}"
        f"长度：约{char_count}字\n"
        f"章节数：约{chapter_count}章\n\n"
    )
    sections = [
        ("开篇样本", opening_sample),
        ("中段样本", middle_sample),
        ("结尾样本", ending_sample),
    ]
    body = "\n\n".join(f"{label}：\n{sample}" for label, sample in sections if sample)
    profile_text = f"{header}{body}".strip()
    if len(profile_text) <= max_chars:
        return profile_text

    available = max(max_chars - len(header) - 24, 300)
    per_sample = max(available // 3, 100)
    shortened = "\n\n".join(
        f"{label}：\n{sample[:per_sample].strip()}" for label, sample in sections if sample
    )
    return f"{header}{shortened}".strip()[:max_chars]


def build_profile_from_inventory_row(row: dict[str, Any], max_profile_chars: int = 3000) -> dict[str, Any] | None:
    """Build one profile row, returning None for failed or missing files."""

    profile, _ = build_profile_from_inventory_row_with_stats(row, max_profile_chars=max_profile_chars)
    return profile


def build_profile_from_inventory_row_with_stats(
    row: dict[str, Any],
    max_profile_chars: int = 3000,
) -> tuple[dict[str, Any] | None, CleaningStats]:
    """Build one profile row and return cleaning stats for reporting."""

    if row.get("read_status") != "ok":
        return None, CleaningStats()

    path = Path(str(row.get("absolute_path", "")))
    if not path.exists():
        return None, CleaningStats()

    raw_text = read_text_with_encoding(
        path,
        row.get("detected_encoding"),
        allow_lossy=int(row.get("decode_replacement_chars", 0) or 0) > 0,
    )
    cleaned_text, cleaning_stats = clean_novel_text_with_stats(raw_text)
    chapters = split_chapters(cleaned_text)
    chapter_count = len(chapters)
    opening, middle, ending = extract_profile_samples(cleaned_text)
    char_count = len(cleaned_text)
    title_guess = str(row.get("title_guess") or row.get("file_stem") or path.stem)
    author_value = row.get("author_guess")
    author_guess = None if pd.isna(author_value) else str(author_value)

    profile_text = make_profile_text(
        title_guess=title_guess,
        author_guess=author_guess,
        char_count=char_count,
        chapter_count=chapter_count,
        opening_sample=opening,
        middle_sample=middle,
        ending_sample=ending,
        max_chars=max_profile_chars,
    )

    return {
        "novel_id": row["novel_id"],
        "title_guess": title_guess,
        "author_guess": author_guess,
        "char_count": char_count,
        "estimated_chapter_count": chapter_count,
        "profile_text": profile_text,
        "opening_sample": opening,
        "middle_sample": middle,
        "ending_sample": ending,
    }, cleaning_stats


@dataclass(frozen=True)
class ProfileWorkerResult:
    """One worker outcome: an accepted profile or the reason it was skipped."""

    status: str
    profile: dict[str, Any] | None
    stats: CleaningStats


def build_profile_worker(row: dict[str, Any], max_profile_chars: int = 3000) -> ProfileWorkerResult:
    """Build one profile, classifying failures instead of raising.

    Module-level so it stays picklable for ``ProcessPoolExecutor``.
    """

    if row.get("read_status") != "ok":
        return ProfileWorkerResult("failed", None, CleaningStats())

    # Duplicate copies would inflate the corpus, surface twice in one result list,
    # and — worst — let the same book land on both sides of a train/eval split.
    if bool(row.get("is_duplicate", False)):
        return ProfileWorkerResult("duplicate", None, CleaningStats())

    path = Path(str(row.get("absolute_path", "")))
    if not path.exists():
        return ProfileWorkerResult("missing", None, CleaningStats())

    try:
        profile, cleaning_stats = build_profile_from_inventory_row_with_stats(row, max_profile_chars=max_profile_chars)
    except (OSError, UnicodeError, LookupError, ValueError):
        return ProfileWorkerResult("read_error", None, CleaningStats())

    if profile is None:
        return ProfileWorkerResult("read_error", None, cleaning_stats)
    return ProfileWorkerResult("ok", profile, cleaning_stats)


def build_profiles(
    inventory_path: Path = DEFAULT_OUTPUT_PATH,
    limit: int | None = None,
    max_profile_chars: int = 3000,
    max_workers: int | None = None,
) -> ProfileBuildResult:
    """Build profile records from a Stage 1 inventory parquet.

    Output keeps inventory row order regardless of worker count.
    """

    inventory = pd.read_parquet(inventory_path)
    if limit is not None:
        inventory = inventory.head(limit)

    records: list[dict[str, Any]] = []
    skipped_failed = 0
    skipped_missing = 0
    skipped_read_error = 0
    skipped_duplicate = 0
    zxcs_boilerplate_detected = 0
    zxcs_boilerplate_lines_removed = 0
    profiles_with_remaining_boilerplate = 0

    rows = inventory.to_dict(orient="records")
    worker = partial(build_profile_worker, max_profile_chars=max_profile_chars)
    workers = min(resolve_worker_count(max_workers), len(rows)) if rows else 1
    progress = partial(tqdm, total=len(rows), desc="Building profiles", unit="novel")

    if workers <= 1:
        outcomes: Iterator[ProfileWorkerResult] = progress(worker(row) for row in rows)
        results = list(outcomes)
    else:
        # spawn, not fork: see the matching note in src/ingest.py.
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
            results = list(progress(executor.map(worker, rows, chunksize=1)))

    for result in results:
        if result.status == "failed":
            skipped_failed += 1
            continue
        if result.status == "missing":
            skipped_missing += 1
            continue
        if result.status == "duplicate":
            skipped_duplicate += 1
            continue
        if result.status == "read_error" or result.profile is None:
            skipped_read_error += 1
            continue

        profile = result.profile
        if result.stats.zxcs_detected:
            zxcs_boilerplate_detected += 1
            zxcs_boilerplate_lines_removed += result.stats.zxcs_lines_removed
        if any(
            contains_zxcs_boilerplate(str(profile.get(column, "")))
            for column in ("profile_text", "opening_sample", "middle_sample", "ending_sample")
        ):
            profiles_with_remaining_boilerplate += 1
        records.append(profile)

    return ProfileBuildResult(
        dataframe=pd.DataFrame(records),
        processed=len(records),
        skipped_failed=skipped_failed,
        skipped_missing=skipped_missing,
        skipped_read_error=skipped_read_error,
        skipped_duplicate=skipped_duplicate,
        zxcs_boilerplate_detected=zxcs_boilerplate_detected,
        zxcs_boilerplate_lines_removed=zxcs_boilerplate_lines_removed,
        profiles_with_remaining_boilerplate=profiles_with_remaining_boilerplate,
    )


def write_profiles(result: ProfileBuildResult, output_path: Path) -> pd.DataFrame:
    """Write generated profiles to parquet and return the DataFrame."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.dataframe.to_parquet(output_path, index=False)
    return result.dataframe
