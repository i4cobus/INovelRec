"""Inventory pipeline for raw Chinese web novel text files."""

from __future__ import annotations

import hashlib
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from multiprocessing import get_context
from itertools import repeat
from pathlib import Path
from typing import Iterable

import chardet
import pandas as pd
from tqdm import tqdm

from src.config import resolve_worker_count
from src.schema import NovelInventoryRecord
from src.text_utils import (
    build_sample_text,
    clean_title_from_stem,
    estimate_chapter_count,
    guess_author,
    normalize_relative_path,
)

COMMON_ENCODINGS = ["utf-8", "utf-8-sig", "gb18030", "gbk", "big5"]
WHITESPACE_RE = re.compile(r"\s+")

# A lossy decode is acceptable only for isolated corruption, never for a wrong codec.
MAX_LOSSY_REPLACEMENTS = 64
# Below this, chardet's guess is noise; see decode_with_detection.
MIN_CHARDET_CONFIDENCE = 0.5
# Same title + same author + length within this tolerance means the same book,
# scraped twice. Different authors sharing a title are different books.
NEAR_DUPLICATE_LENGTH_TOLERANCE = 0.05


def is_junk_file(path: Path) -> bool:
    """Reject sidecar files that merely happen to end in .txt.

    The corpus was archived on macOS, which writes an AppleDouble ``._<name>``
    resource fork beside every file. Those carry the ``.txt`` suffix, so a naive
    glob doubles the corpus with ~7.6k small binary blobs that decode into
    garbage. Hidden files and Finder metadata are excluded for the same reason.
    """

    name = path.name
    return name.startswith("._") or name.startswith(".") or name == "Thumbs.db"


def discover_txt_files(raw_dir: Path) -> list[Path]:
    """Recursively discover `.txt` files under the raw data directory."""

    if not raw_dir.exists():
        return []
    return sorted(path for path in raw_dir.rglob("*.txt") if path.is_file() and not is_junk_file(path))


def generate_novel_id(relative_path: str) -> str:
    """Generate a stable SHA1-based novel identifier from the relative path."""

    normalized = relative_path.replace("\\", "/")
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def decode_with_detection(raw_bytes: bytes) -> tuple[str | None, str | None, str | None]:
    """Decode bytes with the first encoding that works, keeping the decoded text.

    Detection *is* a successful decode, so the text is returned rather than thrown
    away and re-decoded. Over a 36 GB corpus that halves all decoding work.
    """

    for encoding in COMMON_ENCODINGS:
        try:
            return raw_bytes.decode(encoding), encoding, None
        except UnicodeDecodeError:
            continue

    detected = chardet.detect(raw_bytes)
    encoding = detected.get("encoding")
    confidence = float(detected.get("confidence") or 0.0)
    if not encoding:
        return None, None, "Unable to detect encoding"

    # A low-confidence guess is worse than no guess. chardet answers binary garbage
    # with single-byte codecs like Windows-1251 (measured confidence: 0.0017), and
    # those decode *any* byte sequence — so the file would enter the corpus as a
    # perfectly "readable" novel full of mojibake.
    if confidence < MIN_CHARDET_CONFIDENCE:
        return None, encoding, f"Encoding guess {encoding!r} rejected at confidence {confidence:.3f}"

    try:
        return raw_bytes.decode(encoding), encoding, None
    except (UnicodeDecodeError, LookupError) as exc:
        return None, encoding, f"Fallback decode failed: {exc}"


def decode_leniently(raw_bytes: bytes, encodings: list[str]) -> tuple[str, str, int] | None:
    """Last-resort decode that tolerates a few corrupt bytes.

    Measured on this corpus, the files that fail every strict codec do so at a
    single byte: a 3.1M-character novel decodes with exactly one replacement
    character. Discarding a whole book over one bad byte loses far more than it
    protects, so fall back to a lossy decode and record how much was lost, using
    the encoding that needs the fewest replacements.
    """

    best: tuple[str, str, int] | None = None
    for encoding in encodings:
        try:
            text = raw_bytes.decode(encoding, errors="replace")
        except LookupError:
            continue
        replacements = text.count("�")
        if best is None or replacements < best[2]:
            best = (text, encoding, replacements)
    return best


def detect_encoding(raw_bytes: bytes) -> tuple[str | None, str | None]:
    """Return only the detected encoding. Prefer ``decode_with_detection``."""

    _, encoding, error = decode_with_detection(raw_bytes)
    return encoding, error


def read_text_with_detection(path: Path) -> tuple[str | None, str | None, str | None]:
    """Read text safely and return content, encoding, and error message."""

    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        return None, None, str(exc)
    return decode_with_detection(raw_bytes)


def content_fingerprint(text: str) -> str:
    """Whitespace-insensitive content hash, so reformatted copies still collide."""

    return hashlib.sha256(WHITESPACE_RE.sub("", text).encode("utf-8")).hexdigest()


def inventory_single_file(path: Path, raw_dir: Path, store_sample_text: bool = False) -> NovelInventoryRecord:
    """Build one inventory record from a file path."""

    relative_path = normalize_relative_path(path.relative_to(raw_dir))
    stat = path.stat()
    title_guess = clean_title_from_stem(path.stem)
    author_guess = guess_author(path.stem)

    text, encoding, error = read_text_with_detection(path)
    replacements = 0
    if text is None:
        try:
            raw_bytes = path.read_bytes()
        except OSError:
            raw_bytes = b""
        recovered = decode_leniently(raw_bytes, COMMON_ENCODINGS)
        # Only accept a lossy decode that loses a trivial fraction of the text;
        # a garbage result would otherwise masquerade as a readable novel.
        if recovered is not None and recovered[0] and recovered[2] <= max(MAX_LOSSY_REPLACEMENTS, len(recovered[0]) // 100_000):
            text, encoding, replacements = recovered
            error = None

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
            estimated_chapter_count=0,
        )

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
        estimated_chapter_count=estimate_chapter_count(text),
        content_sha256=content_fingerprint(text),
        decode_replacement_chars=replacements,
        sample_text=build_sample_text(text) if store_sample_text else "",
    )


def safe_inventory_single_file(path: Path, raw_dir: Path, store_sample_text: bool = False) -> NovelInventoryRecord:
    """Inventory one file, degrading to a failed record instead of raising.

    Module-level so it stays picklable for ``ProcessPoolExecutor``.
    """

    try:
        return inventory_single_file(path, raw_dir=raw_dir, store_sample_text=store_sample_text)
    except Exception as exc:  # pragma: no cover - defensive fallback
        relative_path = normalize_relative_path(path.relative_to(raw_dir))
        stat = path.stat()
        return NovelInventoryRecord(
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
            estimated_chapter_count=0,
        )


@dataclass(frozen=True)
class DuplicateReport:
    """Duplicate accounting for the Stage 1 CLI summary."""

    exact_duplicates: int = 0
    near_duplicates: int = 0
    duplicate_groups: int = 0
    title_collisions: int = 0

    @property
    def total_duplicates(self) -> int:
        return self.exact_duplicates + self.near_duplicates


def find_near_duplicates(records: list[NovelInventoryRecord]) -> dict[str, str]:
    """Map novel_id -> canonical novel_id for re-scrapes of the same book.

    Exact content hashing misses these: two scrapes of one novel differ by a
    handful of characters (measured here: 《尘缘》's two copies differ by 978 of
    1.28M) yet hash differently, so they would land in opposite folds and leak
    training text into evaluation.

    The rule requires title *and* author to match and lengths to agree within a
    tolerance. Title alone is not enough — 《武魂》 appears twice at similar
    length under two different authors, and those are genuinely different books.
    """

    by_key: dict[tuple[str, str], list[NovelInventoryRecord]] = {}
    for record in records:
        author = (record.author_guess or "").strip()
        if record.read_status != "ok" or record.is_duplicate or not author or not record.title_guess:
            continue
        by_key.setdefault((record.title_guess.strip(), author), []).append(record)

    canonical_of: dict[str, str] = {}
    for group in by_key.values():
        if len(group) < 2:
            continue
        ordered = sorted(group, key=lambda item: (-item.char_count, item.relative_path))
        canonical = ordered[0]
        for other in ordered[1:]:
            longest = max(canonical.char_count, other.char_count)
            if longest and abs(canonical.char_count - other.char_count) / longest <= NEAR_DUPLICATE_LENGTH_TOLERANCE:
                canonical_of[other.novel_id] = canonical.novel_id
    return canonical_of


def mark_content_duplicates(records: list[NovelInventoryRecord]) -> tuple[list[NovelInventoryRecord], DuplicateReport]:
    """Flag records whose text is byte-for-byte identical after whitespace removal.

    The canonical copy is the first in discovery (sorted path) order; the rest get
    ``is_duplicate=True`` and point at it via ``duplicate_of``. Nothing is deleted —
    Stage 2 filters, so the inventory stays a faithful record of the directory.

    Title collisions are only *counted*, never auto-dropped: sequels legitimately
    share a title stem (《441女生寝室5》 vs 《441女生寝室》), so dropping on title
    would silently delete real books.
    """

    canonical_by_hash: dict[str, str] = {}
    marked: list[NovelInventoryRecord] = []
    duplicates = 0

    for record in records:
        fingerprint = record.content_sha256
        if record.read_status != "ok" or not fingerprint:
            marked.append(record)
            continue
        canonical = canonical_by_hash.get(fingerprint)
        if canonical is None:
            canonical_by_hash[fingerprint] = record.novel_id
            marked.append(record)
            continue
        duplicates += 1
        marked.append(record.model_copy(update={"is_duplicate": True, "duplicate_of": canonical}))

    near_canonical = find_near_duplicates(marked)
    if near_canonical:
        marked = [
            record.model_copy(update={"is_duplicate": True, "duplicate_of": near_canonical[record.novel_id]})
            if record.novel_id in near_canonical
            else record
            for record in marked
        ]

    duplicate_groups = len({record.duplicate_of for record in marked if record.is_duplicate})
    titles = [record.title_guess for record in marked if record.read_status == "ok" and not record.is_duplicate]
    title_collisions = len(titles) - len(set(titles))
    return marked, DuplicateReport(
        exact_duplicates=duplicates,
        near_duplicates=len(near_canonical),
        duplicate_groups=duplicate_groups,
        title_collisions=title_collisions,
    )


def inventory_novels(
    raw_dir: Path,
    limit: int | None = None,
    max_workers: int | None = None,
    store_sample_text: bool = False,
    mark_duplicates: bool = True,
) -> tuple[list[NovelInventoryRecord], DuplicateReport]:
    """Inventory text files under the raw directory.

    Results keep discovery (sorted path) order regardless of worker count, so
    ``novel_id`` assignment stays stable across runs.
    """

    files = discover_txt_files(raw_dir)
    if limit is not None:
        files = files[:limit]
    if not files:
        return [], DuplicateReport()

    workers = min(resolve_worker_count(max_workers), len(files))
    progress = partial(tqdm, total=len(files), desc="Inventorying novels", unit="file")
    worker = partial(safe_inventory_single_file, store_sample_text=store_sample_text)

    if workers == 1:
        records = list(progress(worker(path, raw_dir) for path in files))
    else:
        # spawn, not fork: the parent may already hold BLAS/tokenizer threads, and forking a
        # multi-threaded process risks deadlocking partway through a multi-hour corpus scan.
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
            records = list(progress(executor.map(worker, files, repeat(raw_dir), chunksize=1)))

    if not mark_duplicates:
        return records, DuplicateReport()
    return mark_content_duplicates(records)


def records_to_dataframe(records: Iterable[NovelInventoryRecord]) -> pd.DataFrame:
    """Convert inventory records to a pandas DataFrame."""

    return pd.DataFrame(record.model_dump(mode="json") for record in records)


def write_inventory(records: Iterable[NovelInventoryRecord], output_path: Path) -> pd.DataFrame:
    """Write records to parquet and return the DataFrame."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = records_to_dataframe(records)
    df.to_parquet(output_path, index=False)
    return df

