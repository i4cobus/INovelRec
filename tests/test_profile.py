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


def test_profile_generation_removes_zxcs_boilerplate(tmp_path: Path) -> None:
    novel_path = tmp_path / "zxcs.txt"
    block = (
        "==========================================================\n"
        "\u66f4\u591a\u7cbe\u6821\u5c0f\u8bf4\u5c3d\u5728\u77e5\u8f69\u85cf\u4e66\u4e0b\u8f7d\uff1ahttp://www.zxcs8.com/\n"
        "==========================================================\n"
    )
    novel_path.write_text(
        f"{block}\u7b2c\u4e00\u7ae0 \u5f00\u59cb\n\u6838\u5fc3\u6b63\u6587\u5185\u5bb9\uff0c\u4e3b\u89d2\u8c28\u614e\u4fee\u4ed9\u3002\n{block}",
        encoding="utf-8",
    )
    inventory = pd.DataFrame(
        [
            {
                "novel_id": "zxcs",
                "absolute_path": str(novel_path),
                "detected_encoding": "utf-8",
                "read_status": "ok",
                "title_guess": "\u6e05\u6d17\u6d4b\u8bd5",
                "author_guess": None,
            }
        ]
    )
    inventory_path = tmp_path / "inventory.parquet"
    inventory.to_parquet(inventory_path, index=False)

    result = build_profiles(inventory_path=inventory_path)
    row = result.dataframe.iloc[0]

    assert result.zxcs_boilerplate_detected == 1
    assert result.zxcs_boilerplate_lines_removed == 6
    assert result.profiles_with_remaining_boilerplate == 0
    assert "\u77e5\u8f69\u85cf\u4e66" not in row["profile_text"]
    assert "zxcs" not in row["profile_text"].lower()
    assert "\u6838\u5fc3\u6b63\u6587\u5185\u5bb9" in row["profile_text"]
