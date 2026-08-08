"""Independent evidence sampling for evaluation.

The recommender profiles a novel from a fixed set of sampled chapters. If a judge
scored relevance from those same chapters it would only be answering "does this
profile match the query", not "does this *book* match the query", and would
inherit every sampling error the system made. That is circular, and it would hide
exactly the failures evaluation exists to find.

So judge evidence is drawn from chapters the profile did **not** use. Both sides
call ``profile_chapter_indices``: the profile to pick its chapters, this module to
avoid them. Which of the remaining chapters get picked is derived from the novel
id, so a book always yields the same evidence across runs — verdict caching keys
on the evidence, and re-sampling would silently invalidate it.

The author's synopsis is the one deliberate exception. It is shared with the
profile, so agreement between judge and system is inflated slightly by it. That
is the lesser evil: the synopsis is the only place a novel states its genre
outright, and withholding it produced evidence a human annotator could not judge
against the query at all. A noisy judge poisons everything downstream; a bounded,
documented overlap does not. Independence is preserved where it actually
matters — which *narrative* the two sides read.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

from src.profile import (
    DEFAULT_CHAPTER_SAMPLES,
    MIN_CHAPTERS_FOR_SAMPLING,
    character_window_excerpts,
    profile_chapter_indices,
    substantive_chapter_indices,
    window_fractions,
)

# Fractions the pre-chapter-sampling profile used, kept so evidence stays clear of
# them for any profile built before the chapter-aligned rewrite.
PROFILE_FRACTIONS = (0.0, 0.5, 1.0)
MIN_FRACTION_DISTANCE = 0.08

DEFAULT_WINDOWS = 6
DEFAULT_WINDOW_CHARS = 700
WINDOW_SEPARATOR = "\n\n[……]\n\n"


def stable_unit_float(seed: str) -> float:
    """Map a string to a stable float in [0, 1) without touching global RNG state."""

    digest = hashlib.sha1(seed.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def judge_chapter_indices(
    novel_id: str,
    chapters: Sequence[Any],
    windows: int = DEFAULT_WINDOWS,
    profile_samples: int = DEFAULT_CHAPTER_SAMPLES,
) -> list[int]:
    """Pick chapters for the judge, excluding the ones the profile sampled.

    Both sides filter to *substantive* chapters first, so a table-of-contents
    heading can never be handed to the judge as evidence, and the disjointness
    guarantee holds against the same view the profile used.
    """

    usable = substantive_chapter_indices(chapters)
    if len(usable) < MIN_CHAPTERS_FOR_SAMPLING or windows <= 0:
        return []
    used = {usable[position] for position in profile_chapter_indices(len(usable), samples=profile_samples)}
    available = [index for index in usable if index not in used]
    if not available:
        # Tiny books: every real chapter feeds the profile. Independence is
        # impossible, so fall back to the full set rather than returning nothing.
        available = usable

    count = min(windows, len(available))
    picked: list[int] = []
    for slot in range(count):
        band_start = int(slot * len(available) / count)
        band_end = max(int((slot + 1) * len(available) / count), band_start + 1)
        offset = int(stable_unit_float(f"{novel_id}:{slot}") * (band_end - band_start))
        picked.append(available[min(band_start + offset, band_end - 1)])
    return sorted(set(picked))


def judge_window_fractions(windows: int = DEFAULT_WINDOWS, profile_samples: int = DEFAULT_CHAPTER_SAMPLES) -> list[float]:
    """Offsets midway between the profile's, for books with no chapter structure.

    Interleaving guarantees the judge reads text the profile did not, even when
    both fall back to character windows.
    """

    profile_points = window_fractions(profile_samples)
    if len(profile_points) < 2:
        return [0.5] * min(windows, 1)
    midpoints = [(profile_points[i] + profile_points[i + 1]) / 2 for i in range(len(profile_points) - 1)]
    if windows >= len(midpoints):
        return midpoints
    step = len(midpoints) / windows
    return [midpoints[min(int(index * step), len(midpoints) - 1)] for index in range(windows)]


def judge_fractions(novel_id: str, windows: int = DEFAULT_WINDOWS) -> list[float]:
    """Fractional offsets, used only when a novel has no usable chapter structure."""

    if windows < 1:
        raise ValueError("windows must be positive")

    fractions: list[float] = []
    for index in range(windows):
        band_start = (index + 0.15) / (windows + 1)
        band_width = 1.0 / (windows + 1)
        jitter = stable_unit_float(f"{novel_id}:{index}") * band_width * 0.7
        fraction = min(max(band_start + jitter, 0.0), 1.0)
        for avoided in PROFILE_FRACTIONS:
            if abs(fraction - avoided) < MIN_FRACTION_DISTANCE:
                fraction = min(avoided + MIN_FRACTION_DISTANCE * 1.5, 0.97)
        fractions.append(round(fraction, 4))
    return fractions


def window_at(text: str, fraction: float, window_chars: int) -> str:
    """Return a window of text starting at a fractional position."""

    if not text or window_chars <= 0:
        return ""
    start = int(len(text) * min(max(fraction, 0.0), 1.0))
    start = min(start, max(len(text) - window_chars, 0))
    return text[start:start + window_chars].strip()


def sample_judge_evidence(
    text: str,
    novel_id: str,
    windows: int = DEFAULT_WINDOWS,
    window_chars: int = DEFAULT_WINDOW_CHARS,
) -> str:
    """Build judge-facing evidence: the synopsis plus unused chapters."""

    if not text.strip():
        return ""

    from src.profile import extract_blurb, trim_to_sentence
    from src.split_chapters import split_chapters

    sections: list[str] = []
    blurb = extract_blurb(text)
    if blurb:
        sections.append(f"【作品简介】\n{blurb}")

    chapters = split_chapters(text)
    indices = judge_chapter_indices(novel_id, chapters, windows=windows)
    if indices:
        for index in indices:
            chapter = chapters[index]
            body = trim_to_sentence(chapter.text, window_chars)
            if not body:
                continue
            title = str(chapter.title or "").strip()
            sections.append(f"{title}\n{body}" if title else body)
    else:
        # No usable chapter structure: read the gaps between the profile's windows.
        sections.extend(
            character_window_excerpts(text, excerpt_chars=window_chars, fractions=judge_window_fractions(windows))
        )
    return WINDOW_SEPARATOR.join(section for section in sections if section.strip())


def load_raw_text_lookup(inventory_path: Path, novel_ids: set[str]) -> dict[str, str]:
    """Read raw novel text for the requested ids, skipping unreadable files.

    Judge and annotator must read the *same* evidence for their labels to be
    comparable, so both paths load text through here.
    """

    import pandas as pd

    from src.profile import read_text_with_encoding

    inventory = pd.read_parquet(
        inventory_path,
        columns=["novel_id", "absolute_path", "detected_encoding", "read_status", "decode_replacement_chars"],
    )
    inventory = inventory[inventory["novel_id"].astype(str).isin(novel_ids)]
    texts: dict[str, str] = {}
    for row in inventory.to_dict(orient="records"):
        if row.get("read_status") != "ok":
            continue
        try:
            texts[str(row["novel_id"])] = read_text_with_encoding(
                Path(str(row["absolute_path"])),
                row.get("detected_encoding"),
                allow_lossy=int(row.get("decode_replacement_chars", 0) or 0) > 0,
            )
        except (OSError, UnicodeError, LookupError, ValueError):
            continue
    return texts
