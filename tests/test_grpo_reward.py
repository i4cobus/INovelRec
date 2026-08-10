"""Tests for the GRPO verifiable reward. CPU only, no model, no corpus."""

from __future__ import annotations

import json

from src.grpo_reward import (
    RewardWeights,
    claims_violation,
    compute_reward,
    group_advantages,
    parse_verdict,
)

VERDICT = {
    "llm_match_score": 0.8,
    "confidence": "high",
    "matched_preferences": ["仙侠"],
    "violated_preferences": [],
    "risk_flags": [],
    "reason": "题材吻合",
}


def render(**overrides: object) -> str:
    return json.dumps({**VERDICT, **overrides}, ensure_ascii=False)


def test_a_rollout_that_answers_then_rambles_still_parses() -> None:
    """The measured failure mode at temperature 1.0, and why parsing is depth-matched.

    9% of rollouts emit a complete verdict and then keep going — sometimes repeating
    the verdict, sometimes emitting unrelated text. A greedy ``{.*}`` spans from the
    first brace to the last and fails on all of them. Over 1,024 rollouts greedy
    scored 83.6% valid against depth-matched 94.8%, and every one of the differences
    was an answer that was already correct.
    """

    rambling = render() + 'wingConstants.UTF-8]];];].\n[PyCharm]\n' + render(llm_match_score=0.1)
    parsed = parse_verdict(rambling)

    assert parsed is not None
    assert parsed["llm_match_score"] == 0.8  # the FIRST object, not a merge of both


def test_termination_is_scored_outside_the_format_gate() -> None:
    """Answering correctly and then failing to stop must not cost the answer.

    94.8% of rollouts produce a valid verdict but only 91% stop cleanly, so the two
    are different events. Gating the constraint terms on termination would zero the
    reward of a model that got the constraint exactly right.
    """

    weights = RewardWeights(constraint=0.5, score=0.4, terminate=0.1)
    stopped = compute_reward(render(violated_preferences=["系统"]), terms=["系统"], rule_verdict=True, finish_reason="stop", weights=weights)
    rambled = compute_reward(render(violated_preferences=["系统"]), terms=["系统"], rule_verdict=True, finish_reason="length", weights=weights)

    assert stopped.constraint == 1.0 and rambled.constraint == 1.0
    assert abs((stopped.total - rambled.total) - weights.terminate) < 1e-9
    assert rambled.total > 0.0  # the correct answer still earns


def test_unparseable_output_earns_nothing_but_the_termination_term() -> None:
    breakdown = compute_reward("这不是 JSON", terms=["系统"], rule_verdict=True, finish_reason="length")

    assert breakdown.format_ok is False
    assert breakdown.constraint is None and breakdown.score is None
    assert breakdown.total == 0.0


def test_the_score_term_is_asymmetric() -> None:
    """Low scores are paid for on violations; high scores are never paid for.

    Rewarding a high score on a clean candidate would invent supervision — relevance
    has no verifiable ground truth here — and would let the score drift upward for
    free, which is the recall-collapse hack in reverse.
    """

    violating_low = compute_reward(render(llm_match_score=0.05, violated_preferences=["系统"]), terms=["系统"], rule_verdict=True)
    violating_high = compute_reward(render(llm_match_score=0.95, violated_preferences=["系统"]), terms=["系统"], rule_verdict=True)
    assert violating_low.score is not None and violating_high.score is not None
    assert violating_low.score > violating_high.score

    clean_low = compute_reward(render(llm_match_score=0.05), terms=["系统"], rule_verdict=False)
    clean_high = compute_reward(render(llm_match_score=0.95), terms=["系统"], rule_verdict=False)
    assert clean_low.score is None and clean_high.score is None
    assert clean_low.total == clean_high.total


def test_flagging_everything_is_not_a_winning_strategy() -> None:
    """The predicted hack, and the balanced batch is what defeats it.

    A policy that always claims a violation scores 1 on the violating half and 0 on
    the clean half. With the pool balanced 50/50 that averages to the same as always
    claiming nothing — so the only way up is actually discriminating.
    """

    always_flags = render(violated_preferences=["系统"], llm_match_score=0.0)
    never_flags = render(violated_preferences=[], llm_match_score=0.0)

    for text in (always_flags, never_flags):
        on_violating = compute_reward(text, terms=["系统"], rule_verdict=True).constraint
        on_clean = compute_reward(text, terms=["系统"], rule_verdict=False).constraint
        assert (on_violating + on_clean) == 1.0


def test_an_abstained_pair_carries_no_constraint_signal() -> None:
    """``None`` means "no reward signal", never "no violation"."""

    breakdown = compute_reward(render(), terms=["系统"], rule_verdict=None)

    assert breakdown.format_ok is True
    assert breakdown.constraint is None and breakdown.score is None
    assert breakdown.total == RewardWeights().terminate


def test_a_claim_is_recognised_through_wording() -> None:
    assert claims_violation({"violated_preferences": ["不系统"]}, ["系统"])
    assert claims_violation({"violated_preferences": ["男主 不圣母"]}, ["圣母"])
    assert not claims_violation({"violated_preferences": ["青春"]}, ["系统"])


def test_malformed_fields_fail_the_format_gate() -> None:
    assert parse_verdict(render(llm_match_score=1.7)) is None
    assert parse_verdict(render(confidence="很高")) is None
    assert parse_verdict(render(risk_flags="不是列表")) is None
    assert parse_verdict('{"llm_match_score": 0.5}') is None


def test_a_group_with_identical_rewards_contributes_no_gradient() -> None:
    """std=0 must yield zero advantage, not floating-point dust amplified by 1/eps."""

    assert group_advantages([0.6, 0.6, 0.6, 0.6]) == [0.0, 0.0, 0.0, 0.0]

    advantages = group_advantages([1.0, 0.0, 1.0, 0.0])
    assert abs(sum(advantages)) < 1e-9
    assert advantages[0] > 0 > advantages[1]
