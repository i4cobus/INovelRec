"""Deterministic preference parsing for recommendation queries."""

from __future__ import annotations

from dataclasses import dataclass

import regex

SEPARATOR_RE = regex.compile(r"[\s,，;；/、|]+")
NEGATIVE_MARKERS = ("不要", "避免", "不", "别", "无", "非")

# Negative constraints a keyword rule can actually decide. A term qualifies only if
# it clears two independent bars:
#
# 1. **It is narration, not commentary.** The word has to appear inside the novel.
#    Measured over 150 sampled books (see ``docs/post_training_plan.md``), the
#    original hand-written set failed this badly: 17 of its 22 terms are labels
#    readers attach to a book from outside it. 《娇女》 *is* a 宠文 and 《末世之三宫六院》
#    *is* 种马, but neither word occurs in either text, so a keyword rule scores them
#    clean. Its recall on those terms is not low, it is zero by construction.
# 2. **Readers exclude it.** 灵气 and 丹药 clear bar 1 with room to spare (26% / 25%
#    of books above the violation density) but nobody writes 「不灵气」; they are
#    genre-internal nouns, not preferences. This half is a judgement call, not a
#    measurement, and is marked as such.
#
# Only these drive GRPO's verifiable reward. Everything in META_LABEL_NEGATIVES is a
# judgement call and has to go through the judge. Keeping the two apart is what lets
# the project report rule-verifiable and purely semantic constraints as separate arms
# — and the gap between them is what the reward-hacking divergence plot measures.
IN_TEXT_NEGATIVES = frozenset(
    {
        "系统", "穿越", "重生", "修仙", "修真", "恋爱",
        "机甲", "直播", "校园", "异能", "僵尸", "变身", "兽人",
    }
)

# Reader-facing labels. A keyword rule cannot see these, so they are the *semantic*
# arm by definition rather than by oversight. Listed explicitly instead of being an
# open complement so that a new term has to be classified deliberately.
META_LABEL_NEGATIVES = frozenset(
    {
        "后宫", "宫斗", "种马", "争霸", "玄幻", "灵异", "超能力", "言情", "恋爱脑",
        "宠文", "圣母", "玛丽苏", "金手指", "魔改", "克苏鲁", "打脸", "虐主",
        "爽文", "无脑", "小白", "狗血", "开挂", "开局无敌", "速通", "独狼", "压抑", "搞笑",
    }
)

# Occurrences per 100k characters. Calibrated against the 81 human-labelled rows
# whose query carries an in-text exclusion: the violating group's median density is
# 3.45 and the clean group's is 0.03, two orders of magnitude apart, but the tails
# overlap. Two thresholds instead of one so the overlap can be declined rather than
# guessed at — a 3M-character novel that says 系统 twice ("消化系统") is neither a
# violation nor evidence of cleanliness.
VIOLATION_DENSITY = 3.0
CLEAN_DENSITY = 1.0


def is_rule_checkable(negative_term: str) -> bool:
    """Return True when a negative constraint can be scored without a model."""

    return negative_term.strip() in IN_TEXT_NEGATIVES


def term_density(text: str, term: str) -> float:
    """Occurrences of ``term`` per 100k characters of ``text``.

    Density rather than a raw count because novels here span three orders of
    magnitude in length; ten mentions means something different in a 60k-character
    novella than in a 3M-character serial.
    """

    if not text or not term:
        return 0.0
    return text.count(term) / len(text) * 1e5


def constraint_violation_by_rule(text: str, terms: list[str] | tuple[str, ...]) -> bool | None:
    """Decide whether ``text`` violates an exclusion, or decline to decide.

    This is the single definition of the verifiable reward. GRPO scores rollouts
    with it and query synthesis validates the teacher's claims with it; two copies
    would drift, and the divergence plot compares *this* rule against human labels,
    so it has to be one rule.

    Returns True (violates), False (clean), or None when the constraint is not
    rule-checkable or the evidence falls in the band between the two thresholds.
    None means "no reward signal here", never "no violation" — collapsing the two
    would teach the model that ambiguity is safe.
    """

    checkable = [term for term in terms if is_rule_checkable(term)]
    if not checkable:
        return None
    density = max(term_density(text, term) for term in checkable)
    if density >= VIOLATION_DENSITY:
        return True
    if density <= CLEAN_DENSITY:
        return False
    return None


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
