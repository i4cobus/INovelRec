"""Text cleaning utilities for novel profile generation."""

from __future__ import annotations

import re

AD_LINE_PATTERNS = [
    re.compile(r"https?://\S+", re.IGNORECASE),
    re.compile(r"www\.\S+", re.IGNORECASE),
    re.compile(r"^\s*.*(最新网址|最新章节|无弹窗|全文阅读|txt下载|TXT下载).*\s*$", re.IGNORECASE),
    re.compile(r"^\s*.*(请收藏本站|加入书签|手机用户请浏览|本书来自).*\s*$", re.IGNORECASE),
    re.compile(r"^\s*.*(起点中文网|纵横中文网|17K小说网|笔趣阁).*\s*$", re.IGNORECASE),
]


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


def clean_novel_text(text: str) -> str:
    """Clean raw novel text while preserving Chinese punctuation."""

    normalized = normalize_whitespace(text)
    without_ads = remove_safe_ad_lines(normalized)
    return normalize_whitespace(without_ads)

