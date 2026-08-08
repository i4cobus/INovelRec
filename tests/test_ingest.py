from pathlib import Path

from src.ingest import (
    COMMON_ENCODINGS,
    content_fingerprint,
    decode_leniently,
    discover_txt_files,
    generate_novel_id,
    inventory_novels,
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
    assert record.content_sha256


def _write_corpus(raw_dir: Path, count: int) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        path = raw_dir / f"《测试{index}》作者：张三.txt"
        path.write_text(
            f"第一章 开始\n正文{index}。\n第二章 发展\n结尾{index}。",
            encoding="utf-8",
        )


def test_inventory_novels_parallel_matches_sequential(tmp_path: Path) -> None:
    """Worker count must not change results, only wall clock."""

    raw_dir = tmp_path / "raw"
    _write_corpus(raw_dir, 12)

    sequential, _ = inventory_novels(raw_dir, max_workers=1)
    parallel, _ = inventory_novels(raw_dir, max_workers=4)

    def comparable(records: list) -> list[dict]:
        # created_at defaults to now(), so it legitimately differs between runs.
        return [
            {key: value for key, value in record.model_dump(mode="json").items() if key != "created_at"}
            for record in records
        ]

    assert len(sequential) == 12
    assert [record.novel_id for record in sequential] == [record.novel_id for record in parallel]
    assert comparable(sequential) == comparable(parallel)


def test_inventory_novels_handles_empty_dir(tmp_path: Path) -> None:
    records, report = inventory_novels(tmp_path / "nothing")
    assert records == []
    assert report.exact_duplicates == 0


def test_content_fingerprint_ignores_whitespace_reformatting() -> None:
    """The same book saved with different line endings must collide."""

    assert content_fingerprint("第一章\n内容。\n") == content_fingerprint("第一章\r\n\r\n内容。")
    assert content_fingerprint("甲") != content_fingerprint("乙")


def test_duplicate_copies_are_flagged_not_deleted(tmp_path: Path) -> None:
    """novel_id hashes the path, so identical text at two paths needs content dedup."""

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    body = "第一章 开始\n正文。\n第二章 发展\n结尾。"
    (raw_dir / "《测试》作者：张三.txt").write_text(body, encoding="utf-8")
    (raw_dir / "测试_副本.txt").write_text(body.replace("\n", "\r\n"), encoding="utf-8")
    (raw_dir / "《另一本》作者：李四.txt").write_text("第一章 别的\n完全不同的内容。", encoding="utf-8")

    records, report = inventory_novels(raw_dir, max_workers=1)

    assert len(records) == 3, "the inventory stays a faithful record of the directory"
    assert report.exact_duplicates == 1
    assert report.duplicate_groups == 1
    flagged = [record for record in records if record.is_duplicate]
    assert len(flagged) == 1
    canonical_ids = {record.novel_id for record in records if not record.is_duplicate}
    assert flagged[0].duplicate_of in canonical_ids


def test_sample_text_is_off_by_default(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "a.txt").write_text("第一章 开始\n" + "内容。" * 3000, encoding="utf-8")

    off, _ = inventory_novels(raw_dir, max_workers=1)
    on, _ = inventory_novels(raw_dir, max_workers=1, store_sample_text=True)

    assert off[0].sample_text == ""
    assert on[0].sample_text


def test_macos_sidecar_files_are_not_mistaken_for_novels(tmp_path: Path) -> None:
    """A macOS-made archive ships a ._<name>.txt resource fork beside every file."""

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "《真书》作者：张三.txt").write_text("第一章 开始\n内容。", encoding="utf-8")
    (raw_dir / "._《真书》作者：张三.txt").write_bytes(b"\x00\x05\x16\x07\x00\x02\x00\x00Mac OS X")
    (raw_dir / ".DS_Store").write_bytes(b"\x00\x00\x00\x01Bud1")
    (raw_dir / ".hidden.txt").write_text("不该被收录", encoding="utf-8")

    discovered = discover_txt_files(raw_dir)
    records, _ = inventory_novels(raw_dir, max_workers=1)

    assert [path.name for path in discovered] == ["《真书》作者：张三.txt"]
    assert len(records) == 1


def test_decode_leniently_picks_the_encoding_losing_least() -> None:
    body = ("第一章 开始\n" + "内容。" * 500).encode("gb18030")
    text, encoding, replacements = decode_leniently(body[:400] + b"\xff" + body[400:], COMMON_ENCODINGS)
    assert encoding in {"gb18030", "gbk"}
    assert replacements == 1
    assert "第一章" in text


def test_single_corrupt_byte_does_not_discard_a_whole_novel(tmp_path: Path) -> None:
    """Measured on the real corpus: 11 books failed every strict codec on one byte."""

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    body = ("第一章 开始\n" + "内容内容内容。" * 3000).encode("gb18030")
    (raw_dir / "《损坏》作者：张三.txt").write_bytes(body[:5000] + b"\xff" + body[5000:])

    records, _ = inventory_novels(raw_dir, max_workers=1)

    assert records[0].read_status == "ok"
    assert records[0].decode_replacement_chars == 1
    assert records[0].char_count > 10_000


def test_pervasive_corruption_is_still_rejected(tmp_path: Path) -> None:
    """Lenient decoding must recover isolated damage, not launder a broken file."""

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "broken.txt").write_bytes(b"\xff" * 5000)

    records, _ = inventory_novels(raw_dir, max_workers=1)
    assert records[0].read_status == "failed"


def test_rescrapes_of_one_book_are_merged_but_namesakes_are_not(tmp_path: Path) -> None:
    """Same title AND author AND close length -> one book; different author -> two."""

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "《尘缘》作者：烟雨江南.txt").write_text("第一章\n" + "甲" * 20000, encoding="utf-8")
    (raw_dir / "《尘缘》（校对版全本）作者：烟雨江南.txt").write_text("第一章\n" + "甲" * 19900, encoding="utf-8")
    (raw_dir / "《武魂》作者：枫落忆痕.txt").write_text("第一章\n" + "乙" * 20000, encoding="utf-8")
    (raw_dir / "《武魂》作者：辣椒江.txt").write_text("第一章\n" + "丙" * 19900, encoding="utf-8")

    records, report = inventory_novels(raw_dir, max_workers=1)

    assert report.near_duplicates == 1
    merged = [record for record in records if record.is_duplicate]
    assert len(merged) == 1
    assert "尘缘" in merged[0].title_guess
    assert not any("武魂" in record.title_guess for record in records if record.is_duplicate)
