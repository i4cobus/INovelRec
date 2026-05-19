"""Text helper utilities for inventory extraction."""

from __future__ import annotations

import re
from pathlib import Path

import regex

CHAPTER_PATTERNS = [
    regex.compile(r"(?m)^\s*第[一二三四五六七八九十百千万零〇两\d]+[章回节集卷]\b"),
    regex.compile(r"(?m)^\s*正文\s*第[一二三四五六七八九十百千万零〇两\d]+章\b"),
    regex.compile(r"(?mi)^\s*chapter\s+\d+\b"),
]

TITLE_NOISE_PATTERNS = [
    re.compile(r"\s*作者[:：]\s*[^_()\[\]【】《》]+", re.IGNORECASE),
    re.compile(r"\s*by\s+[^_()\[\]【】《》]+", re.IGNORECASE),
    re.compile(r"\s*(校对版|精校版|实体版)?\s*(完本|全本|全文)(\+番外)?", re.IGNORECASE),
    re.compile(r"\s*txt$", re.IGNORECASE),
    re.compile(r"\(\s*\)"),
    re.compile(r"（\s*）"),
]

AUTHOR_PATTERNS = [
    re.compile(r"作者[:：]\s*([^_()\[\]【】《》]+)"),
    re.compile(r"\bby\s+([^_()\[\]【】《》]+)", re.IGNORECASE),
]


def normalize_relative_path(path: Path) -> str:
    """Return a stable POSIX-style relative path string."""

    return path.as_posix()


def clean_title_from_stem(file_stem: str) -> str:
    """Clean common filename noise and return a lightweight title guess."""

    title = file_stem.strip()
    for pattern in TITLE_NOISE_PATTERNS:
        title = pattern.sub("", title)
    title = re.sub(r"[【\[]?(校对版|精校版|实体版)[】\]]?", "", title, flags=re.IGNORECASE)
    title = re.sub(r"[（(]\s*[)）]", "", title)
    title = re.sub(r"\s+", " ", title).strip(" -_")
    return title or file_stem.strip()


def guess_author(text: str) -> str | None:
    """Guess author from filename-derived text when possible."""

    for pattern in AUTHOR_PATTERNS:
        match = pattern.search(text)
        if match:
            author = match.group(1).strip(" -_（）()[]【】《》")
            return author or None
    return None


def estimate_chapter_count(text: str) -> int:
    """Estimate chapter count using regex chapter header heuristics."""

    matches: set[str] = set()
    for pattern in CHAPTER_PATTERNS:
        matches.update(match.group(0).strip() for match in pattern.finditer(text))
    return len(matches)


def compact_whitespace(text: str) -> str:
    """Normalize repeated blank lines and spaces for stored samples."""

    text = text.replace("\x00", "")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_sample_text(text: str, target_total_chars: int = 4500) -> str:
    """Create a compact sample using start, middle, and end snippets."""

    cleaned = compact_whitespace(text)
    if len(cleaned) <= target_total_chars:
        return cleaned

    segment = max(target_total_chars // 3, 800)
    start = cleaned[:segment]
    middle_start = max((len(cleaned) // 2) - (segment // 2), 0)
    middle = cleaned[middle_start:middle_start + segment]
    end = cleaned[-segment:]
    return "\n\n[...]\n\n".join(part.strip() for part in (start, middle, end) if part.strip())
