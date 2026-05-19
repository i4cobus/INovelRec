from src.clean import clean_novel_text


def test_clean_repeated_blank_lines() -> None:
    text = "\u7b2c\u4e00\u884c\r\n\r\n\r\n\u7b2c\u4e8c\u884c\n\n\n\n\u7b2c\u4e09\u884c"
    assert clean_novel_text(text) == "\u7b2c\u4e00\u884c\n\n\u7b2c\u4e8c\u884c\n\n\u7b2c\u4e09\u884c"


def test_clean_removes_obvious_ad_lines() -> None:
    text = "\u6b63\u6587\n\u8bf7\u6536\u85cf\u672c\u7ad9\uff1awww.example.com\n\u4e2d\u6587\u6807\u70b9\uff0c\u4fdd\u7559\u3002"
    assert clean_novel_text(text) == "\u6b63\u6587\n\u4e2d\u6587\u6807\u70b9\uff0c\u4fdd\u7559\u3002"

