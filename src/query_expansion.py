"""Query expansion for improving FAISS retrieval recall."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from src.preferences import StructuredPreference, parse_preference_query


@dataclass(frozen=True)
class ExpandedQuery:
    """One retrieval query variant."""

    text: str
    source: Literal["raw", "llm", "domain_hints"]
    weight: float


class LLMExpansionProvider(Protocol):
    """Protocol for local LLM query expansion providers."""

    def expand_queries(self, raw_query: str, max_queries: int) -> str:
        """Return raw LLM JSON text containing expanded_queries."""


DOMAIN_HINTS = {
    "凡人流": ["韩立式", "凡人修仙", "普通资质", "草根修仙", "谨慎", "低调", "炼气", "筑基", "宗门", "散修", "资源争夺"],
    "仙侠": ["修仙", "灵根", "法宝", "丹药", "金丹", "元婴", "飞升", "宗门"],
    "慢热": ["慢热成长", "稳扎稳打", "苦修", "逐步升级"],
    "理性主角": ["谨慎", "冷静", "低调", "克制", "计算"],
    "理性": ["谨慎", "冷静", "低调", "克制", "计算"],
    "主角": ["主角"],
    "不系统": ["无系统", "无面板", "无任务奖励"],
    "系统": ["无系统", "无面板", "无任务奖励"],
}


def parse_llm_expanded_queries(text: str) -> list[ExpandedQuery]:
    """Parse LLM JSON expansion output safely."""

    data = json.loads(text)
    items = data.get("expanded_queries", [])
    expanded: list[ExpandedQuery] = []
    for item in items:
        query_text = str(item.get("text", "")).strip()
        if not query_text:
            continue
        source = str(item.get("source", "llm"))
        weight = float(item.get("weight", 0.9))
        expanded.append(ExpandedQuery(text=query_text, source="llm" if source != "domain_hints" else "domain_hints", weight=max(0.0, min(weight, 1.0))))
    return expanded


def build_domain_hint_query(structured_preference: StructuredPreference) -> ExpandedQuery | None:
    """Build one compact domain-hint retrieval query."""

    hint_terms: list[str] = []
    raw_terms = [*structured_preference.positive_terms, *(f"不{term}" for term in structured_preference.negative_terms)]
    for term in raw_terms:
        for hint in DOMAIN_HINTS.get(term, []):
            if hint not in hint_terms:
                hint_terms.append(hint)
    if not hint_terms:
        return None
    return ExpandedQuery(text=" ".join(hint_terms), source="domain_hints", weight=0.8)


def append_unique_query(queries: list[ExpandedQuery], query: ExpandedQuery) -> None:
    """Append a query if its text is not already present."""

    if query.text.strip() and query.text not in {item.text for item in queries}:
        queries.append(query)


def build_expanded_queries(
    raw_query: str,
    structured_preference: StructuredPreference | None,
    llm_provider: LLMExpansionProvider | None,
    use_llm_expansion: bool = True,
    use_domain_hints: bool = True,
    max_expanded_queries: int = 5,
) -> list[ExpandedQuery]:
    """Build retrieval query variants. The raw query is always included."""

    if max_expanded_queries <= 0:
        raise ValueError("max_expanded_queries must be positive")
    structured = structured_preference or parse_preference_query(raw_query)
    queries = [ExpandedQuery(text=raw_query.strip(), source="raw", weight=1.0)]

    if use_domain_hints and len(queries) < max_expanded_queries:
        domain_query = build_domain_hint_query(structured)
        if domain_query is not None:
            append_unique_query(queries, domain_query)

    if use_llm_expansion and llm_provider is not None:
        try:
            llm_text = llm_provider.expand_queries(raw_query=raw_query, max_queries=max_expanded_queries - 1)
            for query in parse_llm_expanded_queries(llm_text):
                append_unique_query(queries, query)
                if len(queries) >= max_expanded_queries:
                    return queries[:max_expanded_queries]
        except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
            pass

    return queries[:max_expanded_queries]


def expansion_summary_by_source(expanded_queries: list[ExpandedQuery]) -> dict[str, int]:
    """Count expanded queries by source."""

    summary: dict[str, int] = {}
    for query in expanded_queries:
        summary[query.source] = summary.get(query.source, 0) + 1
    return summary
