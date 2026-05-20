from src.preferences import parse_preference_query


def test_parse_positive_and_negative_terms() -> None:
    parsed = parse_preference_query("\u51e1\u4eba\u6d41 \u4ed9\u4fa0 \u6162\u70ed \u7406\u6027\u4e3b\u89d2 \u4e0d\u7cfb\u7edf")
    assert parsed.positive_terms == ["\u51e1\u4eba\u6d41", "\u4ed9\u4fa0", "\u6162\u70ed", "\u7406\u6027", "\u4e3b\u89d2"]
    assert parsed.negative_terms == ["\u7cfb\u7edf"]


def test_parse_negative_markers() -> None:
    parsed = parse_preference_query("\u907f\u514d\u7cfb\u7edf, \u522b\u540e\u5bab; \u65e0\u5957\u8def / \u975e\u723d\u6587")
    assert parsed.positive_terms == []
    assert parsed.negative_terms == ["\u7cfb\u7edf", "\u540e\u5bab", "\u5957\u8def", "\u723d\u6587"]


def test_parse_standalone_negative_marker() -> None:
    parsed = parse_preference_query("\u4ed9\u4fa0 \u4e0d \u7cfb\u7edf")
    assert parsed.positive_terms == ["\u4ed9\u4fa0"]
    assert parsed.negative_terms == ["\u7cfb\u7edf"]

