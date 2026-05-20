from src.preferences import parse_preference_query
from src.query_expansion import (
    ExpandedQuery,
    build_expanded_queries,
    parse_llm_expanded_queries,
)


class FakeExpansionProvider:
    def __init__(self, text: str) -> None:
        self.text = text

    def expand_queries(self, raw_query: str, max_queries: int) -> str:
        return self.text


def test_raw_query_is_always_included() -> None:
    queries = build_expanded_queries(
        raw_query="凡人流 仙侠",
        structured_preference=None,
        llm_provider=None,
        use_llm_expansion=False,
        use_domain_hints=False,
    )
    assert queries[0] == ExpandedQuery(text="凡人流 仙侠", source="raw", weight=1.0)


def test_llm_expansion_json_is_parsed() -> None:
    parsed = parse_llm_expanded_queries(
        '{"expanded_queries":[{"text":"普通资质 主角 修仙 宗门 炼气 筑基 谨慎","source":"llm","weight":0.9}]}'
    )
    assert parsed[0].text.startswith("普通资质")
    assert parsed[0].source == "llm"
    assert parsed[0].weight == 0.9


def test_invalid_llm_expansion_falls_back_safely() -> None:
    queries = build_expanded_queries(
        raw_query="凡人流",
        structured_preference=parse_preference_query("凡人流"),
        llm_provider=FakeExpansionProvider("not json"),
        use_llm_expansion=True,
        use_domain_hints=False,
    )
    assert [query.source for query in queries] == ["raw"]


def test_domain_hints_are_added_for_fanren_and_xianxia() -> None:
    queries = build_expanded_queries(
        raw_query="凡人流 仙侠",
        structured_preference=parse_preference_query("凡人流 仙侠"),
        llm_provider=None,
        use_llm_expansion=False,
        use_domain_hints=True,
    )
    domain = [query for query in queries if query.source == "domain_hints"][0]
    assert "普通资质" in domain.text
    assert "灵根" in domain.text
    assert "宗门" in domain.text


def test_expanded_queries_are_capped() -> None:
    queries = build_expanded_queries(
        raw_query="凡人流 仙侠",
        structured_preference=parse_preference_query("凡人流 仙侠"),
        llm_provider=FakeExpansionProvider(
            '{"expanded_queries":['
            '{"text":"q1","source":"llm","weight":0.9},'
            '{"text":"q2","source":"llm","weight":0.8},'
            '{"text":"q3","source":"llm","weight":0.7}'
            "]}"
        ),
        use_llm_expansion=True,
        use_domain_hints=True,
        max_expanded_queries=2,
    )
    assert len(queries) == 2
    assert queries[0].source == "raw"

