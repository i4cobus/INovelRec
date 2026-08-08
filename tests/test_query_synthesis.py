import json

import pytest

from src.preferences import (
    IN_TEXT_NEGATIVES,
    META_LABEL_NEGATIVES,
    constraint_violation_by_rule,
    is_rule_checkable,
)
from src.query_synthesis import (
    MAX_QUERY_CHARS,
    SHAPES,
    SynthesisTask,
    SynthesizedQuery,
    build_synthesis_prompt,
    deduplicate,
    is_duplicate,
    label_constraint_by_rule,
    looks_truncated,
    parse_synthesis_response,
    token_overlap,
    verify_constraint_claim,
)


def task() -> SynthesisTask:
    return SynthesisTask(novel_id="n0", title="《测试》", profile="标题：《测试》\n内容简介：\n一个修仙故事。")


def response(**overrides) -> str:
    item = {
        "query": "玄幻 升级流 热血 不系统",
        "shape": "kw",
        "wanted": ["玄幻", "升级流"],
        "unwanted": ["系统"],
        "seed_satisfies_constraint": True,
    }
    item.update(overrides)
    return json.dumps({"queries": [item]}, ensure_ascii=False)


def test_prompt_demands_a_violating_example() -> None:
    """A corpus-natural sample leaves the violating half far too sparse."""

    prompt = build_synthesis_prompt(task())
    assert "violates" in prompt
    assert "必须生成至少一条 violates" in prompt


def test_prompt_forbids_naming_books() -> None:
    assert "不要出现任何书名" in build_synthesis_prompt(task())


def test_parses_queries_and_flags_rule_checkable_constraints() -> None:
    parsed = parse_synthesis_response(response(), task())
    assert len(parsed) == 1
    assert parsed[0].constraint_checkable is True
    assert parsed[0].seed_novel_id == "n0"


def test_a_query_asking_for_what_it_excludes_is_dropped() -> None:
    """「类似都市异能那种…但不要异能」 is not a preference, it is noise."""

    assert parse_synthesis_response(
        response(query="类似都市异能那种主角实力强大的，但不要异能", wanted=["都市异能"], unwanted=["异能"]),
        task(),
    ) == []
    # The plain exclusion must survive — only the positive re-mention is the problem.
    assert len(parse_synthesis_response(response(query="都市 热血 战斗 不异能", unwanted=["异能"]), task())) == 1


def test_semantic_constraints_are_not_marked_checkable() -> None:
    parsed = parse_synthesis_response(response(unwanted=["无脑"]), task())
    assert parsed[0].constraint_checkable is False


def test_queries_naming_a_book_are_dropped() -> None:
    """A title in the query makes retrieval a string match on the profile header."""

    assert parse_synthesis_response(response(query="类似《凡人修仙传》那种慢热仙侠"), task()) == []


def test_absurdly_long_or_short_queries_are_dropped() -> None:
    assert parse_synthesis_response(response(query="短"), task()) == []
    assert parse_synthesis_response(response(query="长" * (MAX_QUERY_CHARS + 1)), task()) == []


def test_unknown_shape_falls_back_rather_than_failing() -> None:
    parsed = parse_synthesis_response(response(shape="freeform"), task())
    assert parsed[0].shape in SHAPES


def test_malformed_output_yields_nothing_instead_of_raising() -> None:
    assert parse_synthesis_response("完全不是 JSON", task()) == []


def test_reasoning_traces_do_not_break_parsing() -> None:
    text = "<think>\n先想想 {要点}\n</think>\n" + response()
    assert len(parse_synthesis_response(text, task())) == 1


def claim(*, satisfies: bool, unwanted: list[str], checkable: bool = True) -> SynthesizedQuery:
    return SynthesizedQuery(
        query="q", shape="kw", unwanted=unwanted,
        seed_satisfies_constraint=satisfies, constraint_checkable=checkable,
    )


def test_constraint_claim_is_verified_against_the_text_not_the_teacher() -> None:
    """Where ground truth is free, the text decides — not the teacher's opinion."""

    saturated = "他打开了系统面板。" * 200

    assert verify_constraint_claim(claim(satisfies=True, unwanted=["系统"]), "一个没有金手指的故事") is True
    assert verify_constraint_claim(claim(satisfies=True, unwanted=["系统"]), saturated) is False
    assert verify_constraint_claim(claim(satisfies=False, unwanted=["系统"]), saturated) is True


def test_one_incidental_mention_in_a_long_novel_is_not_a_violation() -> None:
    """The bug this threshold exists to fix.

    Mere presence marked 86% of a 150-book sample as containing 系统, because a
    3M-character novel says 消化系统 or 系统地学习 once. Under that rule almost every
    book violates 不系统 and the reward term is a constant.
    """

    incidental = "正" * 200_000 + "他系统地学习了剑法"

    assert constraint_violation_by_rule(incidental, ["系统"]) is False
    assert verify_constraint_claim(claim(satisfies=True, unwanted=["系统"]), incidental) is True


def test_the_rule_relabels_the_seed_instead_of_dropping_the_query() -> None:
    """The teacher's claim is measured, then overwritten — never used as a filter.

    Filtering on it collapsed the violating half of a smoke run from ~35% to under
    3%: the teacher sees 系统 once in an 8000-character profile and calls the book a
    violation, the rule reads all 3M characters and disagrees, and the query that
    exists to supply a labelled negative gets deleted. The label is free to compute,
    so it is computed.
    """

    saturated = "他打开了系统面板。" * 200
    wrong = claim(satisfies=True, unwanted=["系统"])  # teacher says clean; text says otherwise

    relabelled = label_constraint_by_rule(wrong, saturated)
    assert relabelled is not None
    assert relabelled.seed_satisfies_constraint is False
    assert verify_constraint_claim(wrong, saturated) is False  # disagreement still observable


def test_semantic_queries_pass_through_relabelling_untouched() -> None:
    semantic = claim(satisfies=True, unwanted=["爽文"], checkable=False)
    assert label_constraint_by_rule(semantic, "任何文本") == semantic


def test_the_undecidable_band_abstains_rather_than_guessing() -> None:
    """Between the two thresholds there is no reward signal, not a clean verdict."""

    borderline = ("正" * 50_000 + "系统") * 2  # ~2 per 100k, inside [CLEAN, VIOLATION)

    assert constraint_violation_by_rule(borderline, ["系统"]) is None
    # And an abstention must not be recorded as the teacher being wrong.
    assert verify_constraint_claim(claim(satisfies=True, unwanted=["系统"]), borderline) is None
    assert verify_constraint_claim(claim(satisfies=False, unwanted=["系统"]), borderline) is None


def test_meta_labels_are_not_rule_checkable() -> None:
    """《娇女》 is a 宠文; the words 宠文 never appear in it.

    These terms are labels readers attach from outside the book, so a keyword rule
    has zero recall on them by construction — not low recall. They belong to the
    semantic arm by definition, which is why the two sets are named separately.
    """

    for label in ("宠文", "种马", "爽文", "金手指", "圣母", "玛丽苏"):
        assert not is_rule_checkable(label), label
        assert constraint_violation_by_rule("任何文本" * 1000, [label]) is None


def test_in_text_and_meta_label_sets_are_disjoint() -> None:
    """A term classified into both would make the arm split ambiguous."""

    assert not (IN_TEXT_NEGATIVES & META_LABEL_NEGATIVES)


def test_semantic_constraints_cannot_be_verified() -> None:
    semantic = claim(satisfies=True, unwanted=["无脑"], checkable=False)
    assert verify_constraint_claim(semantic, "任何文本") is None


def test_duplicate_detection_uses_token_overlap() -> None:
    assert token_overlap("玄幻 升级流 热血", "玄幻 升级流 热血 不圣母") > 0.9
    assert token_overlap("玄幻 升级流", "宅斗 宫斗 权谋") == 0.0
    assert is_duplicate("玄幻 升级流 热血", ["玄幻 升级流 热血 爽文"])


def test_deduplicate_protects_the_evaluation_set() -> None:
    """A training query colliding with an evaluation query is leakage."""

    generated = [
        SynthesizedQuery(query="凡人流 仙侠 慢热 理性主角 不系统", shape="kw"),
        SynthesizedQuery(query="宅斗 宫斗 权谋 女主聪慧", shape="kw"),
    ]
    kept, dropped, leaked = deduplicate(generated, reserved=["凡人流 仙侠 慢热 理性主角 不系统"])

    assert (dropped, leaked) == (1, 0)
    assert [item.query for item in kept] == ["宅斗 宫斗 权谋 女主聪慧"]


def test_deduplicate_also_removes_self_repeats() -> None:
    generated = [SynthesizedQuery(query="玄幻 升级流 热血", shape="kw")] * 3
    kept, dropped, _ = deduplicate(generated, reserved=[])
    assert len(kept) == 1 and dropped == 2


def test_reusing_an_eval_query_positive_half_is_leakage() -> None:
    """Swapping only the exclusion slips under the whole-query overlap threshold.

    「美食 日常 温情 不重生」 vs evaluation's 「美食 日常 温情 不系统」 overlaps 0.75 as
    whole strings, but the positive half is identical, so training on it still
    reveals which books match that phrasing.
    """

    evaluation = [SynthesizedQuery(query="美食 日常 温情 不系统", shape="kw", unwanted=["系统"])]
    generated = [
        SynthesizedQuery(query="美食 日常 温情 不重生", shape="kw", unwanted=["重生"]),
        SynthesizedQuery(query="宅斗 权谋 女主聪慧 不重生", shape="kw", unwanted=["重生"]),
    ]
    kept, _, leaked = deduplicate(generated, reserved=[], reserved_queries=evaluation)

    assert leaked == 1
    assert [item.query for item in kept] == ["宅斗 权谋 女主聪慧 不重生"]


def test_sentence_shaped_queries_are_not_flagged_as_leakage() -> None:
    """A one-phrase positive side would collide with everything under this check.

    Splitting 「类似某类历史战争题材小说，但不要恋爱元素」 on whitespace yields a single
    token, so any overlap ratio is 1.0 by construction. That is tokenisation, not
    leakage, and filtering on it would delete every sentence- and comparison-shaped
    query — half the generated set.
    """

    evaluation = [SynthesizedQuery(query="历史 权谋 朝堂 不玄幻", shape="kw", unwanted=["玄幻"])]
    generated = [SynthesizedQuery(query="类似某类历史战争题材小说，但不要恋爱元素", shape="cmp", unwanted=["恋爱"])]
    kept, _, leaked = deduplicate(generated, reserved=[], reserved_queries=evaluation)

    assert leaked == 0 and len(kept) == 1


def test_a_truncated_response_is_distinguishable_from_an_empty_one() -> None:
    """Both parse to nothing; only one is fixed by a larger token budget."""

    assert looks_truncated('{"queries":[{"query":"玄幻 热血 不系统","shape":"kw"')
    assert not looks_truncated('{"queries":[]}')
    assert not looks_truncated("<think>想想 {要点}</think>" + '{"queries":[]}')
