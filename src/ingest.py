"""Inventory pipeline for raw Chinese web novel text files."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import chardet
import pandas as pd
from tqdm import tqdm

from src.schema import NovelInventoryRecord
from src.text_utils import (
    build_sample_text,
    clean_title_from_stem,
    compact_whitespace,
    estimate_chapter_count,
    guess_author,
    normalize_relative_path,
)

COMMON_ENCODINGS = ["utf-8", "utf-8-sig", "gb18030", "gbk", "big5"]


def discover_txt_files(raw_dir: Path) -> list[Path]:
    """Recursively discover `.txt` files under the raw data directory."""

    if not raw_dir.exists():
        return []
    return sorted(path for path in raw_dir.rglob("*.txt") if path.is_file())


def generate_novel_id(relative_path: str) -> str:
    """Generate a stable SHA1-based novel identifier from the relative path."""

    normalized = relative_path.replace("\\", "/")
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def detect_encoding(raw_bytes: bytes) -> tuple[str | None, str | None]:
    """Detect file encoding using fast common encodings before fallback."""

    for encoding in COMMON_ENCODINGS:
        try:
            raw_bytes.decode(encoding)
            return encoding, None
        except UnicodeDecodeError:
            continue

    detected = chardet.detect(raw_bytes)
    encoding = detected.get("encoding")
    if not encoding:
        return None, "Unable to detect encoding"

    try:
        raw_bytes.decode(encoding)
    except (UnicodeDecodeError, LookupError) as exc:
        return None, f"Fallback decode failed: {exc}"
    return encoding, None


def read_text_with_detection(path: Path) -> tuple[str | None, str | None, str | None]:
    """Read text safely and return content, encoding, and error message."""

    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        return None, None, str(exc)

    encoding, detection_error = detect_encoding(raw_bytes)
    if encoding is None:
        return None, None, detection_error

    try:
        return raw_bytes.decode(encoding), encoding, None
    except (UnicodeDecodeError, LookupError) as exc:
        return None, encoding, str(exc)


def inventory_single_file(path: Path, raw_dir: Path) -> NovelInventoryRecord:
    """Build one inventory record from a file path."""

    relative_path = normalize_relative_path(path.relative_to(raw_dir))
    stat = path.stat()
    title_guess = clean_title_from_stem(path.stem)
    author_guess = guess_author(path.stem)

    text, encoding, error = read_text_with_detection(path)
    if text is None:
        return NovelInventoryRecord(
            novel_id=generate_novel_id(relative_path),
            file_name=path.name,
            file_stem=path.stem,
            relative_path=relative_path,
            absolute_path=str(path.resolve()),
            file_size_bytes=stat.st_size,
            file_size_mb=round(stat.st_size / (1024 * 1024), 4),
            detected_encoding=encoding,
            read_status="failed",
            error_message=error,
            title_guess=title_guess,
            author_guess=author_guess,
            char_count=0,
            line_count=0,
            estimated_chapter_count=0,
            first_2000_chars="",
            sample_text="",
        )

    normalized_text = compact_whitespace(text)
    return NovelInventoryRecord(
        novel_id=generate_novel_id(relative_path),
        file_name=path.name,
        file_stem=path.stem,
        relative_path=relative_path,
        absolute_path=str(path.resolve()),
        file_size_bytes=stat.st_size,
        file_size_mb=round(stat.st_size / (1024 * 1024), 4),
        detected_encoding=encoding,
        read_status="ok",
        error_message=None,
        title_guess=title_guess,
        author_guess=author_guess,
        char_count=len(text),
        line_count=text.count("\n") + (1 if text else 0),
        estimated_chapter_count=estimate_chapter_count(text),
        first_2000_chars=normalized_text[:2000],
        sample_text=build_sample_text(text),
    )


def inventory_novels(raw_dir: Path, limit: int | None = None) -> list[NovelInventoryRecord]:
    """Inventory text files under the raw directory."""

    files = discover_txt_files(raw_dir)
    if limit is not None:
        files = files[:limit]

    records: list[NovelInventoryRecord] = []
    for path in tqdm(files, desc="Inventorying novels", unit="file"):
        try:
            records.append(inventory_single_file(path, raw_dir=raw_dir))
        except Exception as exc:  # pragma: no cover - defensive fallback
            relative_path = normalize_relative_path(path.relative_to(raw_dir))
            stat = path.stat()
            records.append(
                NovelInventoryRecord(
                    novel_id=generate_novel_id(relative_path),
                    file_name=path.name,
                    file_stem=path.stem,
                    relative_path=relative_path,
                    absolute_path=str(path.resolve()),
                    file_size_bytes=stat.st_size,
                    file_size_mb=round(stat.st_size / (1024 * 1024), 4),
                    detected_encoding=None,
                    read_status="failed",
                    error_message=str(exc),
                    title_guess=clean_title_from_stem(path.stem),
                    author_guess=guess_author(path.stem),
                    char_count=0,
                    line_count=0,
                    estimated_chapter_count=0,
                    first_2000_chars="",
                    sample_text="",
                )
            )
    return records


def records_to_dataframe(records: Iterable[NovelInventoryRecord]) -> pd.DataFrame:
    """Convert inventory records to a pandas DataFrame."""

    return pd.DataFrame(record.model_dump(mode="json") for record in records)


def write_inventory(records: Iterable[NovelInventoryRecord], output_path: Path) -> pd.DataFrame:
    """Write records to parquet and return the DataFrame."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = records_to_dataframe(records)
    df.to_parquet(output_path, index=False)
    return df

