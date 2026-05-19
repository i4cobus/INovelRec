from pathlib import Path

from src.ingest import (
    generate_novel_id,
    inventory_single_file,
    read_text_with_detection,
)
from src.text_utils import clean_title_from_stem, estimate_chapter_count


def test_generate_novel_id_is_stable() -> None:
    path = "\u5206\u7c7b/\u300a\u6d4b\u8bd5\u5c0f\u8bf4\u300b\u4f5c\u8005\uff1a\u5f20\u4e09.txt"
    assert generate_novel_id(path) == generate_novel_id(path)


def test_estimate_chapter_count() -> None:
    text = "\n".join(["\u7b2c\u4e00\u7ae0 \u5f00\u59cb", "\u6b63\u6587 \u7b2c\u4e8c\u7ae0 \u7ee7\u7eed", "Chapter 3", "\u7b2c\u4e09\u7ae0 \u518d\u4f1a"])
    assert estimate_chapter_count(text) == 4


def test_clean_title_from_stem() -> None:
    stem = "\u300a\u6d4b\u8bd5\u5c0f\u8bf4\u300b\uff08\u6821\u5bf9\u7248\u5168\u672c\uff09\u4f5c\u8005\uff1a\u5f20\u4e09"
    assert clean_title_from_stem(stem) == "\u300a\u6d4b\u8bd5\u5c0f\u8bf4\u300b"


def test_read_text_with_detection_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"
    text, encoding, error = read_text_with_detection(missing)
    assert text is None
    assert encoding is None
    assert error is not None


def test_inventory_single_file(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    novel_path = raw_dir / "\u300a\u6d4b\u8bd5\u5c0f\u8bf4\u300b\u4f5c\u8005\uff1a\u5f20\u4e09.txt"
    novel_path.write_text(
        "\u7b2c\u4e00\u7ae0 \u5f00\u59cb\n\u8fd9\u662f\u4e00\u4e2a\u6545\u4e8b\u3002\n\u7b2c\u4e8c\u7ae0 \u53d1\u5c55\n\u7ed3\u5c3e\u3002",
        encoding="utf-8",
    )

    record = inventory_single_file(novel_path, raw_dir=raw_dir)

    assert record.read_status == "ok"
    assert record.detected_encoding == "utf-8"
    assert record.title_guess
    assert record.author_guess == "\u5f20\u4e09"
    assert record.estimated_chapter_count == 2
    assert record.first_2000_chars.startswith("\u7b2c\u4e00\u7ae0")

