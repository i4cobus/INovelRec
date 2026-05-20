"""Text cleaning utilities for novel profile generation."""

from __future__ import annotations

from dataclasses import dataclass
import re

AD_LINE_PATTERNS = [
    re.compile(r"https?://\S+", re.IGNORECASE),
    re.compile(r"www\.\S+", re.IGNORECASE),
    re.compile(r"^\s*.*(最新网址|最新章节|无弹窗|全文阅读|txt下载|TXT下载).*\s*$", re.IGNORECASE),
    re.compile(r"^\s*.*(请收藏本站|加入书签|手机用户请浏览|本书来自).*\s*$", re.IGNORECASE),
    re.compile(r"^\s*.*(起点中文网|纵横中文网|17K小说网|笔趣阁).*\s*$", re.IGNORECASE),
]

ZXCS_BOILERPLATE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\u77e5\u8f69\u85cf\u4e66",
        r"\u77e5\u8f69\u85cf\u4e66\u4e0b\u8f7d",
        r"\u66f4\u591a\u7cbe\u6821\u5c0f\u8bf4",
        r"\u66f4\u591a\u7cbe\u6821\u5c0f\u8bf4\u5c3d\u5728\u77e5\u8f69\u85cf\u4e66\u4e0b\u8f7d",
        r"zxcs8\.com",
        r"www\.zxcs8\.com",
        r"\bzxcs\b",
        r"https?://www\.zxcs8\.com/?",
        r"\u7cbe\u6821\u5c0f\u8bf4\u4e0b\u8f7d",
    ]
]

GLOBAL_ZXCS_BOILERPLATE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\u77e5\u8f69\u85cf\u4e66",
        r"zxcs8\.com",
        r"www\.zxcs",
        r"\u66f4\u591a\u7cbe\u6821\u5c0f\u8bf4",
        r"\u7cbe\u6821\u5c0f\u8bf4\u4e0b\u8f7d",
    ]
]

SEPARATOR_LINE_RE = re.compile(r"^\s*[=\-*_＿—～＊]{8,}\s*$")


@dataclass(frozen=True)
class CleaningStats:
    """Cleaning counters for profile build reporting."""

    zxcs_detected: bool = False
    zxcs_lines_removed: int = 0


def normalize_whitespace(text: str) -> str:
    """Normalize line endings, tabs, repeated spaces, and blank lines."""

    text = text.replace("\ufeff", "")
    text = text.replace("\x00", "")
    text = re.sub(r"\r\n?", "\n", text)
    text = text.replace("\t", " ")
    text = re.sub(r"[ \u3000]{2,}", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_safe_ad_lines(text: str) -> str:
    """Remove obvious one-line ads and source watermarks."""

    kept_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and any(pattern.search(stripped) for pattern in AD_LINE_PATTERNS):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines)


def is_zxcs_boilerplate_line(line: str, *, global_only: bool = False) -> bool:
    """Return True when a line clearly contains ZXCS/source-site boilerplate."""

    stripped = line.strip()
    if not stripped:
        return False
    patterns = GLOBAL_ZXCS_BOILERPLATE_PATTERNS if global_only else ZXCS_BOILERPLATE_PATTERNS
    return any(pattern.search(stripped) for pattern in patterns)


def is_separator_line(line: str) -> bool:
    """Return True for repeated separator lines often surrounding ad blocks."""

    return bool(SEPARATOR_LINE_RE.match(line.strip()))


def remove_zxcs_boilerplate_with_stats(text: str, edge_lines: int = 120) -> tuple[str, CleaningStats]:
    """Remove known ZXCS boilerplate lines and adjacent edge separators."""

    lines = text.splitlines()
    if not lines:
        return text, CleaningStats()

    remove_indices: set[int] = set()
    boilerplate_indices: set[int] = set()
    last_edge_start = max(len(lines) - edge_lines, 0)

    for idx, line in enumerate(lines):
        in_edge = idx < edge_lines or idx >= last_edge_start
        if is_zxcs_boilerplate_line(line, global_only=not in_edge):
            remove_indices.add(idx)
            boilerplate_indices.add(idx)

    for idx in boilerplate_indices:
        if idx < edge_lines or idx >= last_edge_start:
            for neighbor in (idx - 1, idx + 1):
                if 0 <= neighbor < len(lines) and is_separator_line(lines[neighbor]):
                    remove_indices.add(neighbor)

    if not remove_indices:
        return text, CleaningStats()

    kept = [line for idx, line in enumerate(lines) if idx not in remove_indices]
    return "\n".join(kept), CleaningStats(zxcs_detected=True, zxcs_lines_removed=len(remove_indices))


def remove_zxcs_boilerplate(text: str) -> str:
    """Remove known ZXCS source-site boilerplate blocks from novel text."""

    cleaned, _ = remove_zxcs_boilerplate_with_stats(text)
    return cleaned


def contains_zxcs_boilerplate(text: str) -> bool:
    """Return True if text still contains known ZXCS boilerplate markers."""

    return any(is_zxcs_boilerplate_line(line) for line in text.splitlines())


def clean_novel_text(text: str) -> str:
    """Clean raw novel text while preserving Chinese punctuation."""

    cleaned, _ = clean_novel_text_with_stats(text)
    return cleaned


def clean_novel_text_with_stats(text: str) -> tuple[str, CleaningStats]:
    """Clean raw novel text and return cleaning counters."""

    normalized = normalize_whitespace(text)
    without_zxcs, stats = remove_zxcs_boilerplate_with_stats(normalized)
    without_ads = remove_safe_ad_lines(without_zxcs)
    return normalize_whitespace(without_ads), stats
