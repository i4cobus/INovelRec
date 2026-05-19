"""Chapter splitting for Chinese web novels."""

from __future__ import annotations

from dataclasses import dataclass

import regex

CHAPTER_TITLE_RE = regex.compile(
    r"(?m)^\s*"
    r"(?:正文\s*)?"
    r"(?:"
    r"第[一二三四五六七八九十百千万零〇两\d]+[章节回集卷部篇]"
    r"|Chapter\s+\d+"
    r")"
    r"(?:[ \t　、:：.-][^\n]{0,80})?"
    r"\s*$",
    regex.IGNORECASE,
)


@dataclass(frozen=True)
class Chapter:
    """A chapter title and its cleaned text body."""

    index: int
    title: str
    text: str


def find_chapter_titles(text: str) -> list[regex.Match[str]]:
    """Return regex matches for likely chapter title lines."""

    return list(CHAPTER_TITLE_RE.finditer(text))


def split_chapters(text: str) -> list[Chapter]:
    """Split text into chapters using robust Chinese title-line heuristics."""

    matches = find_chapter_titles(text)
    if not matches:
        return [Chapter(index=1, title="全文", text=text.strip())] if text.strip() else []

    chapters: list[Chapter] = []
    for idx, match in enumerate(matches):
        next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        title = match.group(0).strip()
        body = text[match.end():next_start].strip()
        chapters.append(Chapter(index=idx + 1, title=title, text=body))
    return chapters
