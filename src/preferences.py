"""Deterministic preference parsing for recommendation queries."""

from __future__ import annotations

from dataclasses import dataclass

import regex

SEPARATOR_RE = regex.compile(r"[\s,，;；/、|]+")
NEGATIVE_MARKERS = ("不要", "避免", "不", "别", "无", "非")


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
