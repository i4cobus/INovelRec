from pathlib import Path

import pandas as pd

from src.clean import clean_novel_text
from src.profile import build_profiles, make_profile_text
from src.split_chapters import split_chapters


def test_chapter_splitting() -> None:
    text = clean_novel_text(
        "\u7b2c\u4e00\u7ae0 \u5f00\u59cb\n\u5185\u5bb91\n\n\u6b63\u6587 \u7b2c\u4e8c\u7ae0 \u7ee7\u7eed\n\u5185\u5bb92\n\nChapter 3\ncontent3"
    )
    chapters = split_chapters(text)
    assert [chapter.title for chapter in chapters] == [
        "\u7b2c\u4e00\u7ae0 \u5f00\u59cb",
        "\u6b63\u6587 \u7b2c\u4e8c\u7ae0 \u7ee7\u7eed",
        "Chapter 3",
    ]
    assert chapters[1].text == "\u5185\u5bb92"


def test_profile_length_control() -> None:
    long_sample = "\u6545\u4e8b" * 1000
    profile = make_profile_text(
        title_guess="\u6d4b\u8bd5\u4e66",
        author_guess="\u4f5c\u8005",
        char_count=100000,
        chapter_count=120,
        opening_sample=long_sample,
        middle_sample=long_sample,
        ending_sample=long_sample,
        max_chars=1200,
    )
    assert len(profile) <= 1200
    assert "\u6807\u9898\uff1a\u6d4b\u8bd5\u4e66" in profile


def test_missing_or_failed_novel_handling(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.txt"
    good_path = tmp_path / "good.txt"
    good_path.write_text("\u7b2c\u4e00\u7ae0 \u5f00\u59cb\n\u5185\u5bb9", encoding="utf-8")

    inventory = pd.DataFrame(
        [
            {
                "novel_id": "failed",
                "absolute_path": str(good_path),
                "detected_encoding": "utf-8",
                "read_status": "failed",
                "title_guess": "\u5931\u8d25",
                "author_guess": None,
            },
            {
                "novel_id": "missing",
                "absolute_path": str(missing_path),
                "detected_encoding": "utf-8",
                "read_status": "ok",
                "title_guess": "\u7f3a\u5931",
                "author_guess": None,
            },
            {
                "novel_id": "good",
                "absolute_path": str(good_path),
                "detected_encoding": "utf-8",
                "read_status": "ok",
                "title_guess": "\u6b63\u5e38",
                "author_guess": None,
            },
        ]
    )
    inventory_path = tmp_path / "inventory.parquet"
    inventory.to_parquet(inventory_path, index=False)

    result = build_profiles(inventory_path=inventory_path)

    assert result.processed == 1
    assert result.skipped_failed == 1
    assert result.skipped_missing == 1
    assert result.dataframe["novel_id"].tolist() == ["good"]

