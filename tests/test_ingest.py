from pathlib import Path

from src.ingest import (
    generate_novel_id,
    inventory_single_file,
    read_text_with_detection,
)
from src.text_utils import clean_title_from_stem, estimate_chapter_count


def test_generate_novel_id_is_stable() -> None:
    path = "分类/《测试小说》作者：张三.txt"
    assert generate_novel_id(path) == generate_novel_id(path)


def test_estimate_chapter_count() -> None:
    text = "\n".join(["第一章 开始", "正文 第二章 继续", "Chapter 3", "第三章 再会"])
    assert estimate_chapter_count(text) == 4


def test_clean_title_from_stem() -> None:
    stem = "《测试小说》（校对版全本）作者：张三"
    assert clean_title_from_stem(stem) == "《测试小说》"


def test_read_text_with_detection_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"
    text, encoding, error = read_text_with_detection(missing)
    assert text is None
    assert encoding is None
    assert error is not None


def test_inventory_single_file(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    novel_path = raw_dir / "《测试小说》作者：张三.txt"
    novel_path.write_text("第一章 开始\n这是一个故事。\n第二章 发展\n结尾。", encoding="utf-8")

    record = inventory_single_file(novel_path, raw_dir=raw_dir)

    assert record.read_status == "ok"
    assert record.detected_encoding == "utf-8"
    assert record.title_guess
    assert record.author_guess == "张三"
    assert record.estimated_chapter_count == 2
    assert record.first_2000_chars.startswith("第一章")
