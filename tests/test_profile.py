from pathlib import Path

import pandas as pd

from src.clean import clean_novel_text
from src.profile import (
    build_profiles,
    extract_blurb,
    make_profile_text,
    extract_chapter_excerpts,
    profile_chapter_indices,
    trim_to_sentence,
    window_fractions,
)
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


def test_profile_respects_the_char_budget() -> None:
    text = make_profile_text(
        title_guess="《测试》",
        author_guess="张三",
        char_count=1_000_000,
        chapter_count=500,
        blurb="简介。" * 400,
        chapter_excerpts=["第一章 起\n" + "内容。" * 400] * 10,
        max_chars=8000,
    )
    assert len(text) <= 8000
    assert "内容简介" in text
    assert "正文节选" in text


def test_blurb_survives_truncation_ahead_of_excerpts() -> None:
    """Genre lives in the synopsis, so it must not be the part that gets cut."""

    text = make_profile_text(
        title_guess="《测试》",
        author_guess=None,
        char_count=100,
        chapter_count=10,
        blurb="这是一个关于修仙的慢热故事。",
        chapter_excerpts=["正文。" * 500] * 10,
        max_chars=900,
    )
    assert "这是一个关于修仙的慢热故事。" in text


def test_excerpts_end_on_sentence_boundaries() -> None:
    """94% of the old character-offset slices ended mid-sentence."""

    assert trim_to_sentence("第一句。第二句！第三句没写完", 12) == "第一句。第二句！"
    assert trim_to_sentence("短句。", 100) == "短句。"


def test_sampling_skips_the_finale() -> None:
    """The last chapters are epilogue and afterword for 69% of this corpus."""

    indices = profile_chapter_indices(100, samples=10)
    assert max(indices) < 95
    assert len(indices) == 10
    assert indices == sorted(indices)


def test_sampling_degrades_gracefully_for_short_books() -> None:
    assert profile_chapter_indices(0) == []
    assert profile_chapter_indices(1) == [0]
    assert len(profile_chapter_indices(3, samples=10)) <= 3


def test_blurb_extraction_stops_at_the_first_chapter() -> None:
    blurb = extract_blurb("书名\n作者：X\n内容简介：\n一个故事。\n第一章 开始\n正文不应进来。")
    assert blurb == "一个故事。"
    assert extract_blurb("没有简介标记的文本") == ""


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


def test_build_profiles_parallel_matches_sequential(tmp_path: Path) -> None:
    """Worker count must not change profiles, counters, or row order."""

    rows = []
    for index in range(10):
        novel_path = tmp_path / f"novel_{index}.txt"
        novel_path.write_text(
            f"第一章 开始\n内容{index}。\n第二章 发展\n更多内容{index}。\n第三章 结束\n结尾{index}。",
            encoding="utf-8",
        )
        rows.append(
            {
                "novel_id": f"n{index}",
                "absolute_path": str(novel_path),
                "detected_encoding": "utf-8",
                "read_status": "ok",
                "title_guess": f"小说{index}",
                "author_guess": None,
            }
        )
    inventory_path = tmp_path / "inventory.parquet"
    pd.DataFrame(rows).to_parquet(inventory_path, index=False)

    sequential = build_profiles(inventory_path=inventory_path, max_workers=1)
    parallel = build_profiles(inventory_path=inventory_path, max_workers=4)

    assert sequential.processed == 10
    assert parallel.processed == 10
    assert sequential.dataframe["novel_id"].tolist() == parallel.dataframe["novel_id"].tolist()
    pd.testing.assert_frame_equal(sequential.dataframe, parallel.dataframe)


def test_lossy_decoded_novels_survive_stage_2(tmp_path: Path) -> None:
    """Stage 1 recovers books with one corrupt byte; Stage 2 must not drop them."""

    novel = tmp_path / "damaged.txt"
    body = ("第一章 开始\n" + "内容内容。" * 500).encode("gb18030")
    novel.write_bytes(body[:400] + b"\xff" + body[400:])

    inventory = pd.DataFrame(
        [
            {
                "novel_id": "damaged",
                "absolute_path": str(novel),
                "detected_encoding": "gb18030",
                "read_status": "ok",
                "title_guess": "受损",
                "author_guess": None,
                "decode_replacement_chars": 1,
            }
        ]
    )
    inventory_path = tmp_path / "inventory.parquet"
    inventory.to_parquet(inventory_path, index=False)

    result = build_profiles(inventory_path=inventory_path, max_workers=1)

    assert result.processed == 1
    assert result.skipped_read_error == 0


class _Chapter:
    def __init__(self, text: str, title: str = "") -> None:
        self.text = text
        self.title = title


def test_contents_only_headings_do_not_produce_an_empty_profile() -> None:
    """《醉神香》: 20 contents entries, the whole novel under one late heading."""

    chapters = [_Chapter("", f"第{i}回 目录") for i in range(20)]
    chapters += [_Chapter("正文。" * 400, "第20回"), _Chapter("", "第21回"), _Chapter("结尾。" * 400, "第22回")]

    excerpts = extract_chapter_excerpts(chapters, cleaned_text="全文。" * 5000)

    assert len(excerpts) >= 3
    assert all(excerpt.strip() for excerpt in excerpts)


def test_books_without_detectable_chapters_still_get_spread_samples() -> None:
    """140 novels yield a single 'chapter' holding everything; the old path gave
    them only the opening few hundred characters."""

    text = "".join(f"这是第{i}段内容，足够长以便采样。" for i in range(4000))
    excerpts = extract_chapter_excerpts([_Chapter(text, "全文")], cleaned_text=text)

    assert len(excerpts) == 10
    assert len(set(excerpts)) == 10, "samples must come from different positions"


def test_sampling_stops_short_of_the_tail_in_both_modes() -> None:
    assert max(window_fractions(10)) <= 0.95
    assert window_fractions(1) == [0.0]
    assert window_fractions(0) == []


def test_opening_chapters_are_sampled_densely() -> None:
    """黄金三章: a web novel states genre, protagonist and 金手指 up front."""

    indices = profile_chapter_indices(2733, samples=10)

    assert indices[:4] == [0, 1, 2, 3], "the opening block must be contiguous"
    assert len(indices) == 10
    assert max(indices) < int(2733 * 0.95), "the finale is still skipped"
    assert len(set(indices[4:])) == 6, "the tail must still spread"


def test_opening_block_shrinks_gracefully_for_short_books() -> None:
    assert profile_chapter_indices(3, samples=10) == [0, 1]
    assert profile_chapter_indices(1, samples=10) == [0]
    assert profile_chapter_indices(12, samples=10)[:4] == [0, 1, 2, 3]


def test_tail_still_reaches_the_late_book() -> None:
    """Pacing (慢热) can only be judged by contrasting early against late."""

    indices = profile_chapter_indices(2733, samples=10)
    assert max(indices) > 2733 * 0.9
