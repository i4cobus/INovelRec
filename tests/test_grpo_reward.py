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


def test_a_constant_score_is_merely_average_under_the_pairwise_term() -> None:
    """The degenerate solution the absolute reward had, and why this one blocks it.

    ``r_score = 1 - llm_match_score`` on violations with nothing on clean candidates
    made ``score = 0`` everywhere optimal, and the policy found it: validation
    llm_match_score went 0.058 -> 0.000 in 25 steps and stayed. Since that field
    carries weight 0.50 in the final ranking, the reranker's relevance contribution
    vanished and judge-scored relevance fell against the SFT student (p=0.012).

    Scoring separation instead, any constant lands exactly on the midpoint of both
    halves — no longer optimal, merely average — while real separation beats it.
    """

    anchor = 0.5
    for constant in (0.0, 0.5, 1.0):
        text = render(llm_match_score=constant)
        on_violating = compute_reward(text, terms=["系统"], rule_verdict=True, partner_anchor=constant)
        on_clean = compute_reward(text, terms=["系统"], rule_verdict=False, partner_anchor=constant)
        assert on_violating.score == 0.5 and on_clean.score == 0.5

    # Separating in the right direction beats the constant on both halves.
    low_on_violating = compute_reward(render(llm_match_score=0.1), terms=["系统"], rule_verdict=True, partner_anchor=anchor)
    high_on_clean = compute_reward(render(llm_match_score=0.9), terms=["系统"], rule_verdict=False, partner_anchor=anchor)
    assert low_on_violating.score > 0.5 and high_on_clean.score > 0.5

    # And separating the wrong way is punished, which the absolute form never did.
    assert compute_reward(render(llm_match_score=0.9), terms=["系统"], rule_verdict=True, partner_anchor=anchor).score < 0.5


def test_no_partner_means_no_score_signal() -> None:
    """A query with no contrasting candidate contributes nothing, rather than a guess."""

    breakdown = compute_reward(render(llm_match_score=0.0), terms=["系统"], rule_verdict=True, partner_anchor=None)
    assert breakdown.score is None
    assert breakdown.constraint is not None  # the constraint term still applies


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


def test_termination_is_read_off_the_text_when_no_finish_reason_is_given() -> None:
    """verl's reward interface passes only the decoded string.

    ``compute_score`` receives ``(data_source, solution_str, ground_truth, extra_info)``
    — no ``finish_reason`` — so termination has to come from the text. Trailing
    non-whitespace after the first complete object is the defect, whether or not the
    rollout also hit the token cap.
    """

    clean = render(violated_preferences=["系统"])
    rambled = clean + "wingConstants.UTF-8]];];].\n[PyCharm]"

    assert compute_reward(clean, terms=["系统"], rule_verdict=True).terminate == 1.0
    assert compute_reward(rambled, terms=["系统"], rule_verdict=True).terminate == 0.0
    # A trailing newline is not rambling.
    assert compute_reward(clean + "\n  ", terms=["系统"], rule_verdict=True).terminate == 1.0
    # An explicit finish_reason still wins when a caller has one.
    assert compute_reward(rambled, terms=["系统"], rule_verdict=True, finish_reason="stop").terminate == 1.0
