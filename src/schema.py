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
    estimated_chapter_count: int

    # Whitespace-insensitive hash of the decoded text. The same book saved under two
    # filenames yields two novel_ids (which hash the path), so path identity cannot
    # detect duplicates — and undetected duplicates leak between train and eval splits.
    content_sha256: str = ""

    # >0 means the text was recovered with a lossy decode; the count is how many
    # characters could not be decoded. Downstream can filter on it if needed.
    decode_replacement_chars: int = 0
    is_duplicate: bool = False
    duplicate_of: str | None = None

    # Only populated with --store-sample-text. Nothing downstream reads it; building it
    # costs a full extra pass over every novel.
    sample_text: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

