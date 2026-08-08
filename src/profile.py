"""Build compact novel profiles from the Stage 1 inventory."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from multiprocessing import get_context
from pathlib import Path
import re
from typing import Any, Iterator, Sequence

import pandas as pd
from tqdm import tqdm

from src.clean import CleaningStats, clean_novel_text_with_stats, contains_zxcs_boilerplate
from src.config import DEFAULT_OUTPUT_PATH, PROCESSED_DATA_DIR, resolve_worker_count
from src.split_chapters import CHAPTER_TITLE_RE, split_chapters

DEFAULT_PROFILE_OUTPUT_PATH = PROCESSED_DATA_DIR / "novel_profiles.parquet"

# Profile budget. The old 3000-char cap was a 4080-era compromise; Qwen3-Embedding
# takes far more, and the old profile covered only 0.095% of a novel.
DEFAULT_PROFILE_MAX_CHARS = 8000
DEFAULT_CHAPTER_SAMPLES = 10
DEFAULT_EXCERPT_CHARS = 600
DEFAULT_BLURB_CHARS = 800
# Skip the tail: the last chapters are epilogue, afterword and 番外, not the book.
FINALE_SKIP_FRACTION = 0.05
# A heading with almost nothing under it is a table-of-contents entry, not a chapter.
MIN_CHAPTER_CHARS = 200
# Below this many real chapters, chapter structure is not worth trusting.
MIN_CHAPTERS_FOR_SAMPLING = 3

SENTENCE_ENDINGS = "。！？…”』」)）"
# Anchored to line start: a bare 简介 appearing mid-sentence in the prose would
# otherwise be mistaken for the synopsis header.
BLURB_MARKERS = re.compile(r"(?m)^[\s\u3000]*(内容简介|作品简介|文案|简介)[\s\u3000]*[:：]?[\s\u3000]*")


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


def trim_to_sentence(text: str, max_chars: int) -> str:
    """Truncate at the last sentence boundary inside the budget.

    Character-offset slicing cut mid-sentence in 94% of the old corpus profiles,
    which is most of what made them read as disjointed fragments.
    """

    text = text.strip()
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    floor = max_chars // 2
    for index in range(len(window) - 1, floor, -1):
        if window[index] in SENTENCE_ENDINGS:
            return window[: index + 1].strip()
    return window.strip()


def extract_blurb(text: str, max_chars: int = DEFAULT_BLURB_CHARS, search_window: int = 6000) -> str:
    """Pull the author-written synopsis that opens most files.

    89% of this corpus ships a 内容简介 / 文案 block before chapter one. It is a
    human-written statement of genre and premise — exactly the signal retrieval
    needs — and the old three-window profile only captured it by accident, when
    it happened to fall inside the first 650 characters.
    """

    head = text[:search_window]
    match = BLURB_MARKERS.search(head)
    if not match:
        return ""
    body = head[match.end():]
    chapter = CHAPTER_TITLE_RE.search(body)
    if chapter:
        body = body[: chapter.start()]
    return trim_to_sentence(body, max_chars)


def profile_chapter_indices(
    chapter_count: int,
    samples: int = DEFAULT_CHAPTER_SAMPLES,
    finale_skip: float = FINALE_SKIP_FRACTION,
) -> list[int]:
    """Choose evenly spread chapter indices, deliberately skipping the finale.

    The old profile's third window was literally the last 650 characters of the
    file. For 69% of this corpus that is 全书完 / 番外 / 作者感言 rather than
    narrative — a slice-of-life cooking novel ended up profiled on its epilogue's
    talk of NPUs. Sampling stops short of the tail.

    ``src/evidence.py`` reads this same function to sample judge evidence from
    chapters the profile did *not* use, which is what keeps evaluation
    independent of the representation being evaluated.
    """

    if chapter_count <= 0 or samples <= 0:
        return []
    usable = max(1, int(chapter_count * (1.0 - finale_skip)))
    count = min(samples, usable)
    if count == 1:
        return [0]
    step = (usable - 1) / (count - 1)
    return sorted({int(round(index * step)) for index in range(count)})


def substantive_chapter_indices(chapters: Sequence[Any], min_chars: int = MIN_CHAPTER_CHARS) -> list[int]:
    """Indices of chapters that actually carry body text.

    Chapter headings are not always chapters. 章回体 novels often open with a table
    of contents, so ``split_chapters`` returns twenty headings with empty bodies and
    the whole novel hiding under the last one. Sampling those wastes every slot:
    《醉神香》 (580k characters) produced a profile containing nothing but its header.
    """

    return [
        index
        for index, chapter in enumerate(chapters)
        if len((getattr(chapter, "text", "") or "").strip()) >= min_chars
    ]


def window_fractions(count: int, finale_skip: float = FINALE_SKIP_FRACTION) -> list[float]:
    """Evenly spread start fractions that stop short of the tail."""

    if count <= 0:
        return []
    if count == 1:
        return [0.0]
    span = max(1.0 - finale_skip, 0.0)
    return [round(index * span / (count - 1), 6) for index in range(count)]


def character_window_excerpts(
    text: str,
    samples: int = DEFAULT_CHAPTER_SAMPLES,
    excerpt_chars: int = DEFAULT_EXCERPT_CHARS,
    fractions: Sequence[float] | None = None,
) -> list[str]:
    """Fallback sampling for novels whose chapter headings cannot be detected.

    Without it, a book that yields no headings collapses to a single "chapter"
    holding the whole text, and the profile becomes its first few hundred
    characters — worse than the three-window sampling this replaced. Measured on
    this corpus, 140 novels hit exactly that path.
    """

    if not text:
        return []
    excerpts: list[str] = []
    for fraction in (fractions if fractions is not None else window_fractions(samples)):
        start = min(int(len(text) * fraction), max(len(text) - excerpt_chars, 0))
        excerpt = trim_to_sentence(text[start : start + excerpt_chars + 200], excerpt_chars)
        if excerpt:
            excerpts.append(excerpt)
    return excerpts


def extract_chapter_excerpts(
    chapters: Sequence[Any],
    cleaned_text: str = "",
    samples: int = DEFAULT_CHAPTER_SAMPLES,
    excerpt_chars: int = DEFAULT_EXCERPT_CHARS,
) -> list[str]:
    """Take one contiguous, sentence-aligned excerpt per sampled chapter.

    Whole-chapter units preserve narrative voice and pacing — the style signal a
    scattering of 650-character slices destroys. Falls back to character windows
    when chapter detection produced too little to sample from.
    """

    usable = substantive_chapter_indices(chapters)
    if len(usable) < MIN_CHAPTERS_FOR_SAMPLING:
        return character_window_excerpts(cleaned_text, samples=samples, excerpt_chars=excerpt_chars)

    excerpts: list[str] = []
    for position in profile_chapter_indices(len(usable), samples=samples):
        chapter = chapters[usable[position]]
        excerpt = trim_to_sentence(getattr(chapter, "text", "") or "", excerpt_chars)
        if not excerpt:
            continue
        title = str(getattr(chapter, "title", "") or "").strip()
        excerpts.append(f"{title}\n{excerpt}" if title else excerpt)
    return excerpts or character_window_excerpts(cleaned_text, samples=samples, excerpt_chars=excerpt_chars)


def make_profile_text(
    *,
    title_guess: str,
    author_guess: str | None,
    char_count: int,
    chapter_count: int,
    blurb: str = "",
    chapter_excerpts: Sequence[str] | None = None,
    max_chars: int = DEFAULT_PROFILE_MAX_CHARS,
) -> str:
    """Compose the text that gets embedded and shown to the reranker.

    Order matters under truncation: the author's synopsis states genre and
    premise directly, so it survives ahead of narrative excerpts.
    """

    author_line = f"作者：{author_guess}\n" if author_guess else ""
    header = (
        f"标题：{title_guess}\n"
        f"{author_line}"
        f"长度：约{char_count}字\n"
        f"章节数：约{chapter_count}章\n"
    )
    parts = [header]
    if blurb:
        parts.append(f"\n内容简介：\n{blurb}\n")

    excerpts = list(chapter_excerpts or [])
    if excerpts:
        remaining = max_chars - len("".join(parts)) - 24
        if remaining > 0:
            per_excerpt = max(remaining // len(excerpts), 120)
            body = "\n\n".join(
                f"节选{index}：\n{trim_to_sentence(excerpt, per_excerpt)}"
                for index, excerpt in enumerate(excerpts, start=1)
            )
            parts.append(f"\n正文节选：\n{body}")
    return "".join(parts).strip()[:max_chars]


def build_profile_from_inventory_row(row: dict[str, Any], max_profile_chars: int = DEFAULT_PROFILE_MAX_CHARS) -> dict[str, Any] | None:
    """Build one profile row, returning None for failed or missing files."""

    profile, _ = build_profile_from_inventory_row_with_stats(row, max_profile_chars=max_profile_chars)
    return profile


def build_profile_from_inventory_row_with_stats(
    row: dict[str, Any],
    max_profile_chars: int = DEFAULT_PROFILE_MAX_CHARS,
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
    char_count = len(cleaned_text)

    blurb = extract_blurb(cleaned_text)
    excerpts = extract_chapter_excerpts(chapters, cleaned_text=cleaned_text)

    title_guess = str(row.get("title_guess") or row.get("file_stem") or path.stem)
    author_value = row.get("author_guess")
    author_guess = None if pd.isna(author_value) else str(author_value)

    profile_text = make_profile_text(
        title_guess=title_guess,
        author_guess=author_guess,
        char_count=char_count,
        chapter_count=chapter_count,
        blurb=blurb,
        chapter_excerpts=excerpts,
        max_chars=max_profile_chars,
    )

    return {
        "novel_id": row["novel_id"],
        "title_guess": title_guess,
        "author_guess": author_guess,
        "char_count": char_count,
        "estimated_chapter_count": chapter_count,
        "profile_text": profile_text,
        "blurb": blurb,
        "chapter_excerpt_count": len(excerpts),
        "sampled_chapter_indices": profile_chapter_indices(chapter_count),
    }, cleaning_stats


@dataclass(frozen=True)
class ProfileWorkerResult:
    """One worker outcome: an accepted profile or the reason it was skipped."""

    status: str
    profile: dict[str, Any] | None
    stats: CleaningStats


def build_profile_worker(row: dict[str, Any], max_profile_chars: int = DEFAULT_PROFILE_MAX_CHARS) -> ProfileWorkerResult:
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
    max_profile_chars: int = DEFAULT_PROFILE_MAX_CHARS,
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
