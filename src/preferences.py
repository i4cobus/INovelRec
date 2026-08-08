"""Deterministic preference parsing for recommendation queries."""

from __future__ import annotations

from dataclasses import dataclass

import regex

SEPARATOR_RE = regex.compile(r"[\s,，;；/、|]+")
NEGATIVE_MARKERS = ("不要", "避免", "不", "别", "无", "非")

# Negative constraints whose violation can be decided by looking for the word (or a
# close variant) in the candidate's text. Only these can drive GRPO's verifiable
# reward term; everything else (小白 / 无脑 / 爽文 / 狗血 …) is a judgement call and
# has to go through the judge instead. Keeping the two sets apart is what lets the
# project report rule-verifiable and purely semantic constraints as separate arms.
RULE_CHECKABLE_NEGATIVES = frozenset(
    {
        "系统", "后宫", "穿越", "重生", "修仙", "宫斗", "种马", "争霸", "玄幻", "灵异",
        "超能力", "言情", "恋爱", "恋爱脑", "宠文", "圣母", "玛丽苏", "金手指", "魔改",
        "克苏鲁", "打脸", "虐主",
    }
)


def is_rule_checkable(negative_term: str) -> bool:
    """Return True when a negative constraint can be scored without a model."""

    return negative_term.strip() in RULE_CHECKABLE_NEGATIVES


@dataclass(frozen=True)
class PreferenceQuery:
    """Parsed positive and negative preference terms."""

    raw_query: str
    positive_terms: list[str]
    negative_terms: list[str]


StructuredPreference = PreferenceQuery


def split_query_terms(query: str) -> list[str]:
    """Split a query on common Chinese and English separators."""

    return [part.strip() for part in SEPARATOR_RE.split(query.strip()) if part.strip()]


def expand_positive_term(term: str) -> list[str]:
    """Expand simple compound terms while keeping parsing deterministic."""

    if term.endswith("主角") and len(term) > len("主角"):
        prefix = term[: -len("主角")]
        return [prefix, "主角"]
    return [term]


def strip_negative_marker(term: str) -> str | None:
    """Return the negative term if a marker is present."""

    for marker in NEGATIVE_MARKERS:
        if term == marker:
            return ""
        if term.startswith(marker) and len(term) > len(marker):
            return term[len(marker):].strip()
    return None


def append_unique(items: list[str], values: list[str] | str) -> None:
    """Append terms while preserving order and removing duplicates."""

    incoming = [values] if isinstance(values, str) else values
    for value in incoming:
        value = value.strip()
        if value and value not in items:
            items.append(value)


def parse_preference_query(query: str) -> PreferenceQuery:
    """Parse positive and negative terms from a natural-language query."""

    positive_terms: list[str] = []
    negative_terms: list[str] = []
    tokens = split_query_terms(query)
    next_is_negative = False

    for token in tokens:
        negative = strip_negative_marker(token)
        if negative is not None:
            if negative:
                append_unique(negative_terms, negative)
                next_is_negative = False
            else:
                next_is_negative = True
            continue

        if next_is_negative:
            append_unique(negative_terms, token)
            next_is_negative = False
        else:
            append_unique(positive_terms, expand_positive_term(token))

    return PreferenceQuery(raw_query=query, positive_terms=positive_terms, negative_terms=negative_terms)
