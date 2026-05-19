"""Data models for inventory records."""

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class NovelInventoryRecord(BaseModel):
    """Single inventory row for one novel text file."""

    novel_id: str
    file_name: str
    file_stem: str
    relative_path: str
    absolute_path: str
    file_size_bytes: int
    file_size_mb: float
    detected_encoding: str | None
    read_status: Literal["ok", "failed"]
    error_message: str | None
    title_guess: str
    author_guess: str | None
    char_count: int
    line_count: int
    estimated_chapter_count: int
    first_2000_chars: str
    sample_text: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

