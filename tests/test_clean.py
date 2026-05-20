from src.clean import clean_novel_text, remove_zxcs_boilerplate, remove_zxcs_boilerplate_with_stats


def test_clean_repeated_blank_lines() -> None:
    text = "\u7b2c\u4e00\u884c\r\n\r\n\r\n\u7b2c\u4e8c\u884c\n\n\n\n\u7b2c\u4e09\u884c"
    assert clean_novel_text(text) == "\u7b2c\u4e00\u884c\n\n\u7b2c\u4e8c\u884c\n\n\u7b2c\u4e09\u884c"


def test_clean_removes_obvious_ad_lines() -> None:
    text = "\u6b63\u6587\n\u8bf7\u6536\u85cf\u672c\u7ad9\uff1awww.example.com\n\u4e2d\u6587\u6807\u70b9\uff0c\u4fdd\u7559\u3002"
    assert clean_novel_text(text) == "\u6b63\u6587\n\u4e2d\u6587\u6807\u70b9\uff0c\u4fdd\u7559\u3002"


def test_remove_zxcs_beginning_block() -> None:
    text = (
        "==========================================================\n"
        "\u66f4\u591a\u7cbe\u6821\u5c0f\u8bf4\u5c3d\u5728\u77e5\u8f69\u85cf\u4e66\u4e0b\u8f7d\uff1ahttp://www.zxcs8.com/\n"
        "==========================================================\n"
        "\u7b2c\u4e00\u7ae0 \u5c11\u5e74"
    )
    cleaned, stats = remove_zxcs_boilerplate_with_stats(text)
    assert cleaned == "\u7b2c\u4e00\u7ae0 \u5c11\u5e74"
    assert stats.zxcs_detected is True
    assert stats.zxcs_lines_removed == 3


def test_remove_zxcs_ending_block() -> None:
    text = (
        "\u6b63\u6587\u5185\u5bb9\n"
        "==========================================================\n"
        "\u66f4\u591a\u7cbe\u6821\u5c0f\u8bf4\u5c3d\u5728\u77e5\u8f69\u85cf\u4e66\u4e0b\u8f7d\uff1ahttp://www.zxcs8.com/\n"
        "=========================================================="
    )
    assert remove_zxcs_boilerplate(text) == "\u6b63\u6587\u5185\u5bb9"


def test_remove_zxcs_beginning_and_ending_blocks() -> None:
    block = (
        "==========================================================\n"
        "\u66f4\u591a\u7cbe\u6821\u5c0f\u8bf4\u5c3d\u5728\u77e5\u8f69\u85cf\u4e66\u4e0b\u8f7d\uff1ahttp://www.zxcs8.com/\n"
        "==========================================================\n"
    )
    text = f"{block}\u6b63\u6587\u5185\u5bb9\n{block}"
    cleaned = clean_novel_text(text)
    assert cleaned == "\u6b63\u6587\u5185\u5bb9"
    assert "\u77e5\u8f69\u85cf\u4e66" not in cleaned
    assert "zxcs" not in cleaned.lower()


def test_remove_global_zxcs_watermark_line() -> None:
    text = "\u7b2c\u4e00\u7ae0 \u5185\u5bb9\n\u672c\u4e66\u7531\u77e5\u8f69\u85cf\u4e66\u6574\u7406\n\u7b2c\u4e8c\u7ae0 \u5185\u5bb9"
    assert clean_novel_text(text) == "\u7b2c\u4e00\u7ae0 \u5185\u5bb9\n\u7b2c\u4e8c\u7ae0 \u5185\u5bb9"


def test_separator_safety_without_boilerplate() -> None:
    text = "\u7b2c\u4e00\u7ae0 \u6b63\u5e38\u5185\u5bb9\n==========================================================\n\u7b2c\u4e8c\u7ae0 \u6b63\u5e38\u5185\u5bb9"
    assert clean_novel_text(text) == text
