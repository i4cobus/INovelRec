"""Deterministic preference parsing for recommendation queries."""

from __future__ import annotations

from collections.abc import Mapping
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
# Each exclusion is a *set* of surface forms, not one word. The rule missed a third
# of real violations because the excluded word is often not the word the novel uses:
# a book the judge called saturated with 异能 mentions that term at density 0.25 and
# writes 觉醒者 / 能力者 throughout. Density is summed over the set — occurrences
# divided by length is additive, so the merge is exact rather than an approximation.
#
# Members passed two label-free screens. First, firing rate over all 7,656 novels,
# where the bar is *genre specificity* rather than rarity: 金丹 fires on 10.3% of the
# corpus and belongs because 修真 is a large genre, while 观众 fires on 15.1% and does
# not, because sport, theatre and courtroom scenes all have an audience.
#
# Second, rank correlation with the base term across the same corpus. A real synonym
# appears in the same books. This dropped 宿主, 打赏, 上辈子, 同桌, 操纵杆 and others
# that looked plausible and were not: 上辈子 correlates 0.05 with 重生.
#
# Neither screen touches the judge labels. Selecting the vocabulary to agree with them
# would make the reward and the evaluation share a source, and a later gain on that
# metric would be self-fulfilling — the same anti-circularity rule the project holds
# elsewhere.
TERM_SYNONYMS: dict[str, frozenset[str]] = {
    "系统": frozenset({"系统", "任务奖励"}),
    "穿越": frozenset({"穿越", "异世界"}),
    "重生": frozenset({"重生"}),
    "修仙": frozenset({"修仙", "筑基", "金丹", "元婴", "渡劫", "灵根", "仙门"}),
    "修真": frozenset({"修真", "筑基", "金丹", "元婴", "渡劫", "灵根"}),
    "恋爱": frozenset({"恋爱"}),
    "机甲": frozenset({"机甲", "驾驶舱"}),
    "直播": frozenset({"直播", "弹幕", "主播"}),
    "校园": frozenset({"校园", "班主任", "高考", "学长"}),
    "异能": frozenset({"异能", "能力者", "异能者", "超能力"}),
    "僵尸": frozenset({"僵尸", "尸毒", "养尸", "尸王"}),
    "变身": frozenset({"变身"}),
    "兽人": frozenset({"兽人", "半兽", "兽族"}),
}

# Every surface form the density table has to measure.
COUNTED_WORDS: tuple[str, ...] = tuple(sorted({word for words in TERM_SYNONYMS.values() for word in words}))

IN_TEXT_NEGATIVES = frozenset(TERM_SYNONYMS)

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


def merged_density(text: str, term: str) -> float:
    """Density of an exclusion, counting every surface form in its set.

    Occurrences over length is additive, so summing the members is exact: the merged
    density is what a single word whose spelling varied would have measured.
    """

    return sum(term_density(text, word) for word in TERM_SYNONYMS.get(term, frozenset({term})))


def merged_density_from_table(densities: Mapping[str, float], term: str) -> float:
    """Same sum, from the precomputed per-word table.

    The table stores raw surface forms rather than merged totals, so revising a
    synonym set never requires another 36 GB pass over the corpus — only this lookup
    changes.
    """

    return sum(float(densities.get(word, 0.0)) for word in TERM_SYNONYMS.get(term, frozenset({term})))


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
    return violation_from_density(max(merged_density(text, term) for term in checkable))


def violation_from_density(density: float) -> bool | None:
    """Apply the two thresholds to an already-measured density.

    Split out so the precomputed density table and the live text path share one
    definition of where the thresholds sit. Everything above this line is how the
    density was obtained; everything below it is the rule.
    """

    if density >= VIOLATION_DENSITY:
        return True
    if density <= CLEAN_DENSITY:
        return False
    return None


def constraint_violation_from_densities(
    densities: Mapping[str, float],
    terms: list[str] | tuple[str, ...],
) -> bool | None:
    """Same verdict as :func:`constraint_violation_by_rule`, from a density table.

    GRPO computes this reward inside the rollout loop, where re-reading a 3M-character
    novel per candidate is not affordable. Densities for every (novel, term) pair are
    precomputed once by ``16_build_density_table.py``; this turns the reward into a
    dictionary lookup. A term missing from the table means the corpus pass never saw
    it, which is a zero count, not an abstention.
    """

    checkable = [term for term in terms if is_rule_checkable(term)]
    if not checkable:
        return None
    return violation_from_density(max(merged_density_from_table(densities, term) for term in checkable))


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
