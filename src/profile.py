"""Build compact novel profiles from the Stage 1 inventory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.clean import CleaningStats, clean_novel_text_with_stats, contains_zxcs_boilerplate
from src.config import DEFAULT_OUTPUT_PATH, PROCESSED_DATA_DIR
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
    zxcs_boilerplate_detected: int = 0
    zxcs_boilerplate_lines_removed: int = 0
    profiles_with_remaining_boilerplate: int = 0


def read_text_with_encoding(path: Path, encoding: str | None) -> str:
    """Read a raw text file with the encoding detected during inventory."""

    if not encoding:
        raise ValueError("Missing detected encoding")
    return path.read_text(encoding=encoding)


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

    raw_text = read_text_with_encoding(path, row.get("detected_encoding"))
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


def build_profiles(
    inventory_path: Path = DEFAULT_OUTPUT_PATH,
    limit: int | None = None,
    max_profile_chars: int = 3000,
) -> ProfileBuildResult:
    """Build profile records from a Stage 1 inventory parquet."""

    inventory = pd.read_parquet(inventory_path)
    if limit is not None:
        inventory = inventory.head(limit)

    records: list[dict[str, Any]] = []
    skipped_failed = 0
    skipped_missing = 0
    skipped_read_error = 0
    zxcs_boilerplate_detected = 0
    zxcs_boilerplate_lines_removed = 0
    profiles_with_remaining_boilerplate = 0

    for row in tqdm(inventory.to_dict(orient="records"), desc="Building profiles", unit="novel"):
        if row.get("read_status") != "ok":
            skipped_failed += 1
            continue

        path = Path(str(row.get("absolute_path", "")))
        if not path.exists():
            skipped_missing += 1
            continue

        try:
            profile, cleaning_stats = build_profile_from_inventory_row_with_stats(row, max_profile_chars=max_profile_chars)
        except (OSError, UnicodeError, LookupError, ValueError):
            skipped_read_error += 1
            continue

        if profile is not None:
            if cleaning_stats.zxcs_detected:
                zxcs_boilerplate_detected += 1
                zxcs_boilerplate_lines_removed += cleaning_stats.zxcs_lines_removed
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
        zxcs_boilerplate_detected=zxcs_boilerplate_detected,
        zxcs_boilerplate_lines_removed=zxcs_boilerplate_lines_removed,
        profiles_with_remaining_boilerplate=profiles_with_remaining_boilerplate,
    )


def write_profiles(result: ProfileBuildResult, output_path: Path) -> pd.DataFrame:
    """Write generated profiles to parquet and return the DataFrame."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.dataframe.to_parquet(output_path, index=False)
    return result.dataframe
